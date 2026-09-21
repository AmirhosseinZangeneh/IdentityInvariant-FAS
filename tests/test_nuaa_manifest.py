"""NUAA source/provenance checks with tiny synthetic files, never raw images."""

import json
import shutil
from dataclasses import replace

import pytest

from identity_invariant_fas.constants import ATTACK_LABEL, BONA_FIDE_LABEL
from identity_invariant_fas.data.nuaa_manifest import (
    build_nuaa_source_manifest,
    generate_nuaa_source_manifest,
    validate_nuaa_source_manifest,
)
from identity_invariant_fas.data.sample_manifest import load_sample_manifest


@pytest.fixture
def nuaa_root(tmp_path):
    root = tmp_path / "raw"
    root.mkdir()
    specs = [
        ("client_train_face.txt", "ClientFace", "0001", "train.jpg"),
        ("client_test_face.txt", "ClientFace", "0013", "test.jpg"),
        ("imposter_train_face.txt", "ImposterFace", "0001", "train.jpg"),
        ("imposter_test_face.txt", "ImposterFace", "0016", "test.jpg"),
    ]
    for name, folder, subject, filename in specs:
        image = root / folder / subject / filename
        image.parent.mkdir(parents=True, exist_ok=True)
        image.write_bytes(b"synthetic file; image decoding is not part of indexing")
        (root / name).write_text(f"{subject}\\{filename} 1 2 3 4 5 6\n", encoding="utf-8")
    return root


def test_source_records_and_provenance(nuaa_root):
    records, audit = build_nuaa_source_manifest(nuaa_root)
    assert len(records) == 4
    assert [record.filepath for record in records] == sorted(r.filepath for r in records)
    for record in records:
        folder, subject, filename = record.filepath.split("/")
        assert record.sample_id == f"NUAA:{record.filepath}"
        assert record.subject_id == subject
        assert record.class_label == (
            BONA_FIDE_LABEL if folder == "ClientFace" else ATTACK_LABEL
        )
        assert record.split == filename.removesuffix(".jpg")
        assert record.attack_type == record.preprocessing_version == ""
    assert audit["class_counts"] == {str(BONA_FIDE_LABEL): 2, str(ATTACK_LABEL): 2}
    assert audit["source_split_counts"] == {"test": 2, "train": 2}
    assert audit["cross_class_folder_ids"] == {
        "intersection": ["0001"], "client_only": ["0013"],
        "imposter_only": ["0016"], "union_count": 3,
    }
    assert audit["identity_provenance_status"] == "unresolved"
    assert audit["local_readme_sha256"] is None
    validate_nuaa_source_manifest(records, nuaa_root)


def test_repeated_generation_and_relocation_are_byte_identical(nuaa_root, tmp_path):
    manifest, audit_path = tmp_path / "samples.csv", tmp_path / "audit.json"
    generate_nuaa_source_manifest(nuaa_root, manifest, audit_path)
    first = manifest.read_bytes(), audit_path.read_bytes()
    generate_nuaa_source_manifest(nuaa_root, manifest, audit_path)
    assert (manifest.read_bytes(), audit_path.read_bytes()) == first
    moved = tmp_path / "relocated"
    shutil.copytree(nuaa_root, moved)
    generate_nuaa_source_manifest(moved, manifest, audit_path)
    assert (manifest.read_bytes(), audit_path.read_bytes()) == first
    assert len(load_sample_manifest(manifest)) == 4
    assert all(json.loads(audit_path.read_text())["validation"].values())
    assert b"\r\n" not in manifest.read_bytes()


def test_mixed_separators_and_list_order(nuaa_root):
    original, _ = build_nuaa_source_manifest(nuaa_root)
    source = nuaa_root / "client_train_face.txt"
    source.write_text("\n0001/train.jpg 1 2 3 4 5 6\n", encoding="utf-8")
    changed, _ = build_nuaa_source_manifest(nuaa_root)
    assert changed == original
    image = nuaa_root / "ClientFace/0001/another.jpg"
    image.write_bytes(b"synthetic")
    lines = ["0001/train.jpg 1 2\n", "0001/another.jpg 1 2\n"]
    source.write_text("".join(lines), encoding="utf-8")
    ordered, _ = build_nuaa_source_manifest(nuaa_root)
    source.write_text("".join(reversed(lines)), encoding="utf-8")
    reordered, _ = build_nuaa_source_manifest(nuaa_root)
    assert reordered == ordered


@pytest.mark.parametrize("destination", ["client_train_face.txt", "client_test_face.txt"])
def test_duplicate_source_entry_rejected_even_across_splits(nuaa_root, destination):
    line = (nuaa_root / "client_train_face.txt").read_text()
    with (nuaa_root / destination).open("a") as handle:
        handle.write(line)
    with pytest.raises(ValueError, match="Duplicate source file"):
        build_nuaa_source_manifest(nuaa_root)


@pytest.mark.parametrize("missing", ["ClientFace/0001/train.jpg", "client_train_face.txt"])
def test_missing_source_files(nuaa_root, missing):
    (nuaa_root / missing).unlink()
    with pytest.raises(FileNotFoundError):
        build_nuaa_source_manifest(nuaa_root)


@pytest.mark.parametrize("listed", ["../train.jpg", "/0001/train.jpg", "C:/train.jpg", "x/0001/train.jpg"])
def test_ambiguous_or_nonrelative_source_paths_rejected(nuaa_root, listed):
    (nuaa_root / "client_train_face.txt").write_text(f"{listed} 1 2\n")
    with pytest.raises(ValueError, match="relative subject/filename"):
        build_nuaa_source_manifest(nuaa_root)


@pytest.mark.parametrize("field,value", [
    ("sample_id", "changed"), ("class_label", ATTACK_LABEL), ("split", "test"),
    ("subject_id", "other"), ("attack_type", "guessed"), ("preprocessing_version", "guessed"),
])
def test_source_metadata_tampering_rejected(nuaa_root, field, value):
    records, _ = build_nuaa_source_manifest(nuaa_root)
    assert records[0].filepath == "ClientFace/0001/train.jpg"
    records[0] = replace(records[0], **{field: value})
    with pytest.raises(ValueError, match="metadata mismatch"):
        validate_nuaa_source_manifest(records, nuaa_root)


@pytest.mark.parametrize("change", ["missing", "extra", "duplicate_id", "duplicate_path"])
def test_record_bijection(nuaa_root, change):
    records, _ = build_nuaa_source_manifest(nuaa_root)
    if change == "missing":
        records.pop()
    elif change == "extra":
        records.append(replace(records[0], sample_id="extra", filepath="ClientFace/0001/extra.jpg"))
    elif change == "duplicate_id":
        records[1] = replace(records[1], sample_id=records[0].sample_id)
    else:
        records.append(replace(records[0], sample_id="different-id"))
    with pytest.raises(ValueError, match="coverage mismatch|Duplicate"):
        validate_nuaa_source_manifest(records, nuaa_root)


def test_audit_includes_unlisted_folders_and_images(nuaa_root):
    directory = nuaa_root / "ClientFace/0099"
    directory.mkdir()
    (directory / "unlisted.jpg").write_bytes(b"synthetic")
    records, audit = build_nuaa_source_manifest(nuaa_root)
    assert len(records) == 4
    client = audit["folders"]["ClientFace"]
    assert client["unlisted_folder_ids"] == ["0099"]
    assert client["unlisted_image_paths"] == ["ClientFace/0099/unlisted.jpg"]
    assert client["disk_image_count"] == 3


def test_outputs_cannot_overwrite_raw_sources(nuaa_root, tmp_path):
    with pytest.raises(ValueError, match="outside the raw dataset root"):
        generate_nuaa_source_manifest(nuaa_root, nuaa_root / "client_train_face.txt", tmp_path / "a.json")
    with pytest.raises(ValueError, match="distinct files"):
        generate_nuaa_source_manifest(nuaa_root, tmp_path / "same", tmp_path / "same")


def test_generation_failure_does_not_write_outputs(nuaa_root, tmp_path):
    (nuaa_root / "ClientFace/0001/train.jpg").unlink()
    manifest, audit = tmp_path / "samples.csv", tmp_path / "audit.json"
    with pytest.raises(FileNotFoundError):
        generate_nuaa_source_manifest(nuaa_root, manifest, audit)
    assert not manifest.exists()
    assert not audit.exists()
