"""NUAA source manifests and folder-provenance audits, without fold assignment."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path, PurePosixPath

from .nuaa import _FILE_SPECS, _parse_line, _subject_and_name
from .sample_manifest import REQUIRED_COLUMNS, SampleRecord, load_sample_manifest


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _index_sources(root: Path) -> tuple[list[SampleRecord], dict]:
    records = []
    origins: dict[str, str] = {}
    source_lists = {}
    for split, specs in _FILE_SPECS.items():
        for list_name, folder, label in specs:
            data = (root / list_name).read_bytes()
            counts: Counter[str] = Counter()
            for line_number, raw in enumerate(data.decode("utf-8-sig").splitlines(), 1):
                if not raw.strip():
                    continue
                origin = f"{list_name}:{line_number}"
                listed, _ = _parse_line(raw)
                normalized = listed.replace("\\", "/")
                parts = normalized.split("/")
                if (
                    len(parts) != 2
                    or any(part in {"", ".", ".."} for part in parts)
                    or ":" in normalized
                    or "\x00" in normalized
                ):
                    raise ValueError(f"{origin}: expected a relative subject/filename path")
                subject, filename = _subject_and_name(listed)
                if subject != subject.strip():
                    raise ValueError(f"{origin}: subject has surrounding whitespace")
                filepath = f"{folder}/{subject}/{filename}"
                if filepath in origins:
                    raise ValueError(
                        f"Duplicate source file {filepath}: {origins[filepath]} and {origin}"
                    )
                image_path = root.joinpath(*PurePosixPath(filepath).parts)
                if not image_path.is_file():
                    raise FileNotFoundError(f"{origin}: missing referenced file {filepath}")
                if not image_path.resolve().is_relative_to(root.resolve()):
                    raise ValueError(f"{origin}: referenced file escapes dataset root")
                origins[filepath] = origin
                counts[subject] += 1
                records.append(SampleRecord(
                    sample_id=f"NUAA:{filepath}",
                    filepath=filepath,
                    subject_id=subject,
                    class_label=label,
                    attack_type="",
                    split=split,
                    preprocessing_version="",
                ))
            source_lists[list_name] = {
                "sha256": _sha256(data),
                "split": split,
                "class_label": label,
                "sample_count": sum(counts.values()),
                "folder_id_counts": dict(sorted(counts.items())),
            }
    if not records:
        raise ValueError("NUAA source lists contain no samples")
    records.sort(key=lambda record: record.filepath)
    return records, source_lists


def validate_nuaa_source_manifest(records: list[SampleRecord], root: str | Path) -> None:
    """Require a bijection with source entries and exact source metadata agreement."""
    expected, _ = _index_sources(Path(root))
    ids = [record.sample_id for record in records]
    paths = [record.filepath for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate sample_id in NUAA manifest")
    if len(paths) != len(set(paths)):
        raise ValueError("Duplicate filepath in NUAA manifest")
    observed = {record.filepath: record for record in records}
    source = {record.filepath: record for record in expected}
    if observed.keys() != source.keys():
        raise ValueError(
            "Manifest/source coverage mismatch: "
            f"{len(source.keys() - observed.keys())} missing, "
            f"{len(observed.keys() - source.keys())} extra"
        )
    for filepath, record in source.items():
        if observed[filepath] != record:
            raise ValueError(f"Manifest/source metadata mismatch for {filepath}")


def build_nuaa_source_manifest(root: str | Path) -> tuple[list[SampleRecord], dict]:
    """Return sorted records and a deterministic audit of source lists and folders.

    subject_id is only the raw folder token, not a verified cross-class human ID.
    No attack category, preprocessing version, or human mapping is inferred.
    """
    root = Path(root)
    records, source_lists = _index_sources(root)
    folders = {}
    for folder in sorted({spec[1] for specs in _FILE_SPECS.values() for spec in specs}):
        directory = root / folder
        disk_ids = sorted(path.name for path in directory.iterdir() if path.is_dir())
        listed = [record for record in records if record.filepath.startswith(folder + "/")]
        listed_ids = sorted({record.subject_id for record in listed})
        disk_files = sorted(
            path.relative_to(root).as_posix()
            for path in directory.rglob("*")
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
        )
        listed_paths = {record.filepath for record in listed}
        by_split = {
            split: sorted({record.subject_id for record in listed if record.split == split})
            for split in _FILE_SPECS
        }
        folders[folder] = {
            "disk_folder_ids": disk_ids,
            "disk_folder_id_count": len(disk_ids),
            "listed_folder_ids": listed_ids,
            "listed_folder_id_count": len(listed_ids),
            "unlisted_folder_ids": sorted(set(disk_ids) - set(listed_ids)),
            "sample_count": len(listed),
            "disk_image_count": len(disk_files),
            "disk_image_paths_sha256": _sha256("\n".join(disk_files).encode("utf-8")),
            "unlisted_image_paths": sorted(set(disk_files) - listed_paths),
            "source_split_folder_ids": by_split,
            "source_train_test_folder_overlap": sorted(
                set(by_split["train"]) & set(by_split["test"])
            ),
        }
    client_ids = set(folders["ClientFace"]["disk_folder_ids"])
    imposter_ids = set(folders["ImposterFace"]["disk_folder_ids"])
    readme = root / "readme.txt"
    audit = {
        "audit_schema_version": 1,
        "sample_count": len(records),
        "class_counts": dict(sorted(Counter(str(r.class_label) for r in records).items())),
        "source_split_counts": dict(sorted(Counter(r.split for r in records).items())),
        "source_lists": source_lists,
        "folders": folders,
        "cross_class_folder_ids": {
            "intersection": sorted(client_ids & imposter_ids),
            "client_only": sorted(client_ids - imposter_ids),
            "imposter_only": sorted(imposter_ids - client_ids),
            "union_count": len(client_ids | imposter_ids),
        },
        "local_readme_sha256": _sha256(readme.read_bytes()) if readme.is_file() else None,
        "identity_provenance_status": "unresolved",
        "subject_id_semantics": "Raw folder token; not a verified cross-class human identity.",
        "unresolved_questions": [
            "Do equal folder IDs in ClientFace and ImposterFace denote the same human?",
            "What explains folder IDs present in only one class?",
            "Which documented mapping should canonical human-subject splitting use?",
            "Attack subtype and preprocessing version are not supplied by the source lists.",
        ],
    }
    return records, audit


def generate_nuaa_source_manifest(
    root: str | Path, manifest_path: str | Path, audit_path: str | Path
) -> dict:
    """Validate, repeat generation, and write UTF-8 CSV/JSON outside the raw root."""
    root, manifest_path, audit_path = Path(root), Path(manifest_path), Path(audit_path)
    outputs = [manifest_path.resolve(), audit_path.resolve()]
    if outputs[0] == outputs[1] or any(path.is_relative_to(root.resolve()) for path in outputs):
        raise ValueError("Manifest and audit must be distinct files outside the raw dataset root")
    records, audit = build_nuaa_source_manifest(root)
    validate_nuaa_source_manifest(records, root)
    repeated_records, repeated_audit = build_nuaa_source_manifest(root)
    if records != repeated_records or audit != repeated_audit:
        raise ValueError("NUAA sources changed during repeated generation")
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=REQUIRED_COLUMNS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(asdict(record) for record in records)
    payload = buffer.getvalue().encode("utf-8")
    audit["manifest_sha256"] = _sha256(payload)
    audit["validation"] = {
        "source_record_bijection": True,
        "unique_sample_ids": True,
        "all_referenced_files_exist": True,
        "labels_and_source_splits_match": True,
        "repeated_generation_matches": True,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_bytes(payload)
    if load_sample_manifest(manifest_path) != records:
        raise ValueError("Written manifest did not round-trip through the canonical parser")
    audit_path.write_bytes((json.dumps(audit, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--audit", required=True)
    args = parser.parse_args()
    audit = generate_nuaa_source_manifest(args.root, args.manifest, args.audit)
    print(f"Generated {audit['sample_count']} samples; human-identity provenance remains unresolved.")


if __name__ == "__main__":
    main()
