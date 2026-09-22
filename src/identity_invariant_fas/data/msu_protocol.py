"""Strict MSU-MFSD public-release inventory and protocol lock; never decodes media."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

from ..constants import ATTACK_LABEL, BONA_FIDE_LABEL
from ..evaluation.identity_probe import ProbeObservation


EXPECTED_CLIENT_COUNTS = {"train": 15, "test": 20}
CAMERAS = ("android", "laptop")
ATTACK_TYPES = ("ipad_video", "iphone_video", "printed_photo")
CAPTURE_MODELS = {"android": "Google Nexus 5 front-facing camera", "laptop": "MacBook Air built-in camera"}
PRESENTATION_DEVICES = {"ipad_video": "iPad Air", "iphone_video": "iPhone 5S", "printed_photo": "printed paper"}
# Anchored grammar from the bundled README, not the legacy loader's substring match.
_NAME = re.compile(
    r"(?P<kind>real|attack)_client(?P<client>[0-9]{3})_"
    r"(?P<camera>android|laptop)_(?P<resolution>SD|HD)"
    r"(?:_(?P<attack>ipad_video|iphone_video|printed_photo))?_scene01\.(?P<extension>mp4|mov)"
)
_MEDIA_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm"}
_EVIDENCE_FILES = ("train_sub_list.txt", "test_sub_list.txt", "README.txt", "DecFrames.m")


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_msu_subjects(path: str | Path) -> tuple[str, ...]:
    """Map numeric official list tokens to three digits, as DecFrames.m does.

    Unlike the legacy set parser, duplicates (including padding aliases) fail.
    """
    tokens = Path(path).read_text(encoding="utf-8-sig").split()
    ids = []
    for token in tokens:
        if not re.fullmatch(r"[0-9]{1,3}", token) or not 1 <= int(token) <= 55:
            raise ValueError(f"Invalid documented MSU client token: {token!r}")
        ids.append(f"{int(token):03d}")
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate subject mapping in {Path(path).name}")
    return tuple(sorted(ids))


@dataclass(frozen=True)
class MSURecording:
    filepath: str
    client_id: str
    pad_partition: str
    class_label: int
    capture_device: str
    capture_model: str
    attack_type: str | None
    presentation_device: str | None
    resolution_token: str
    scenario: str
    annotation_filepath: str
    annotation_sha256: str
    byte_size: int
    session_id: None = None

    @property
    def video_id(self) -> str:
        return "MSU-MFSD:" + self.filepath

    def probe_observation(self, frame_index: int | None = None) -> ProbeObservation:
        """Preserve source grouping for a future video feature or an actual frame.

        Use records from a validated protocol. This neither samples nor reads frames.
        """
        if frame_index is not None and (type(frame_index) is not int or frame_index < 0):
            raise ValueError("frame_index must be a nonnegative integer or None")
        suffix = ":video" if frame_index is None else f":frame:{frame_index}"
        return ProbeObservation(self.video_id + suffix, self.client_id, self.video_id, None)


@dataclass(frozen=True)
class MSUProtocol:
    train_clients: tuple[str, ...]
    test_clients: tuple[str, ...]
    recordings: tuple[MSURecording, ...]
    source_fingerprints: dict[str, str]


def _parse_path(filepath: str) -> dict:
    parts = filepath.split("/")
    if len(parts) != 3 or parts[0] != "scene01" or parts[1] not in {"real", "attack"}:
        raise ValueError(f"Unlisted/unsupported recording location: {filepath}")
    match = _NAME.fullmatch(parts[2])
    if match is None:
        raise ValueError(f"Unsupported official filename grammar: {filepath}")
    info = match.groupdict()
    if info["kind"] != parts[1] or (info["kind"] == "attack") != (info["attack"] is not None):
        raise ValueError(f"Class/directory/attack metadata conflict: {filepath}")
    if info["extension"] != {"android": "mp4", "laptop": "mov"}[info["camera"]]:
        raise ValueError(f"Capture-device/format mismatch: {filepath}")
    return info


def validate_msu_protocol(protocol: MSUProtocol) -> None:
    """Validate complete public 15/20 partition and eight documented slots per client."""
    mapping = {}
    for split, clients in (("train", protocol.train_clients), ("test", protocol.test_clients)):
        if len(clients) != len(set(clients)):
            raise ValueError(f"Duplicate client in {split}")
        for client in clients:
            if not re.fullmatch(r"[0-9]{3}", client) or not 1 <= int(client) <= 55:
                raise ValueError(f"Invalid canonical client ID: {client}")
            if client in mapping:
                raise ValueError(f"Official train/test identity overlap: {client}")
            mapping[client] = split
        if len(clients) != EXPECTED_CLIENT_COUNTS[split]:
            raise ValueError(f"Expected {EXPECTED_CLIENT_COUNTS[split]} {split} subjects; got {len(clients)}")
    ids, paths, slots = set(), set(), set()
    for row in protocol.recordings:
        if row.video_id in ids or row.filepath in paths:
            raise ValueError(f"Duplicate recording ID/path: {row.filepath}")
        ids.add(row.video_id)
        paths.add(row.filepath)
        info = _parse_path(row.filepath)
        if row.client_id not in mapping:
            raise ValueError(f"Unexpected client without subject mapping: {row.client_id}")
        if row.pad_partition != mapping[row.client_id]:
            raise ValueError(f"Official partition mismatch: {row.filepath}")
        label = BONA_FIDE_LABEL if info["kind"] == "real" else ATTACK_LABEL
        expected = (info["client"], label, info["camera"], CAPTURE_MODELS[info["camera"]],
                    info["attack"], PRESENTATION_DEVICES.get(info["attack"]), info["resolution"], "scene01")
        actual = (row.client_id, row.class_label, row.capture_device, row.capture_model,
                  row.attack_type, row.presentation_device, row.resolution_token, row.scenario)
        if expected != actual or type(row.class_label) is not int or row.session_id is not None:
            raise ValueError(f"Recording metadata conflict: {row.filepath}")
        if row.annotation_filepath != str(Path(row.filepath).with_suffix(".face")).replace("\\", "/"):
            raise ValueError(f"Annotation provenance mismatch: {row.filepath}")
        if not re.fullmatch(r"[0-9a-f]{64}", row.annotation_sha256) or row.byte_size <= 0:
            raise ValueError(f"Invalid annotation fingerprint or empty recording: {row.filepath}")
        slot = (row.client_id, row.capture_device, row.attack_type)
        if slot in slots:
            raise ValueError(f"Duplicate acquisition slot: {slot}")
        slots.add(slot)
    expected_slots = {(c, camera, attack) for c in mapping for camera in CAMERAS for attack in (None, *ATTACK_TYPES)}
    if slots != expected_slots:
        missing = sorted(expected_slots - slots, key=str)
        extra = sorted(slots - expected_slots, key=str)
        raise ValueError(f"Incomplete/unlisted recording inventory; missing slots={missing}, extra slots={extra}")
    if set(protocol.source_fingerprints) != set(_EVIDENCE_FILES) or any(
        not re.fullmatch(r"[0-9a-f]{64}", value) for value in protocol.source_fingerprints.values()
    ):
        raise ValueError("Missing source metadata fingerprints")


def load_msu_protocol(root: str | Path) -> MSUProtocol:
    """Read filenames, lists, annotations and file sizes only; no media decoding."""
    root = Path(root).resolve()
    train = read_msu_subjects(root / "train_sub_list.txt")
    test = read_msu_subjects(root / "test_sub_list.txt")
    if set(train) & set(test):
        raise ValueError(f"Official train/test identity overlap: {sorted(set(train) & set(test))}")
    mapping = {client: split for split, ids in (("train", train), ("test", test)) for client in ids}
    fingerprints = {name: _digest((root / name).read_bytes()) for name in _EVIDENCE_FILES}
    files = sorted((p for p in root.rglob("*") if p.is_file()), key=lambda p: p.relative_to(root).as_posix())
    recordings, physical = [], set()
    annotations = {p.relative_to(root).as_posix() for p in files if p.suffix.lower() == ".face"}
    for path in files:
        if path.suffix.lower() not in _MEDIA_EXTENSIONS:
            continue
        relative = path.relative_to(root).as_posix()
        info = _parse_path(relative)
        if info["client"] not in mapping:
            raise ValueError(f"Unlisted recording/unexpected client: {relative}")
        if not path.resolve().is_relative_to(root):
            raise ValueError(f"Recording escapes dataset root: {relative}")
        stat = path.stat()
        key = (stat.st_dev, stat.st_ino)
        if key in physical:
            raise ValueError(f"Duplicate physical recording: {relative}")
        physical.add(key)
        face = path.with_suffix(".face")
        if not face.is_file():
            raise FileNotFoundError(f"Missing referenced face annotation: {relative}")
        if not face.resolve().is_relative_to(root):
            raise ValueError(f"Annotation escapes dataset root: {relative}")
        label = BONA_FIDE_LABEL if info["kind"] == "real" else ATTACK_LABEL
        recordings.append(MSURecording(
            filepath=relative, client_id=info["client"], pad_partition=mapping[info["client"]],
            class_label=label, capture_device=info["camera"], capture_model=CAPTURE_MODELS[info["camera"]],
            attack_type=info["attack"], presentation_device=PRESENTATION_DEVICES.get(info["attack"]),
            resolution_token=info["resolution"], scenario="scene01",
            annotation_filepath=face.relative_to(root).as_posix(),
            annotation_sha256=_digest(face.read_bytes()), byte_size=stat.st_size,
        ))
    referenced = {row.annotation_filepath for row in recordings}
    if annotations != referenced:
        raise ValueError(f"Orphan/unlisted face annotations: {sorted(annotations - referenced)}")
    protocol = MSUProtocol(train, test, tuple(recordings), fingerprints)
    validate_msu_protocol(protocol)
    return protocol


def _canonical_metadata(protocol: MSUProtocol) -> dict:
    return {
        "schema_version": 1, "dataset": "MSU-MFSD", "protocol": "official_public_train_test",
        "train_clients": sorted(protocol.train_clients), "test_clients": sorted(protocol.test_clients),
        "source_fingerprints": protocol.source_fingerprints,
        "recordings": [{**asdict(r), "video_id": r.video_id} for r in sorted(protocol.recordings, key=lambda r: r.filepath)],
    }


def _json_bytes(value) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def audit_msu_protocol(protocol: MSUProtocol) -> dict:
    """Summarize a validated inventory; readiness is metadata-only, not an experiment."""
    validate_msu_protocol(protocol)
    rows = protocol.recordings
    partitions, readiness, per_client = {}, {}, {}
    for split, clients in (("train", protocol.train_clients), ("test", protocol.test_clients)):
        subset = [r for r in rows if r.pad_partition == split]
        partitions[split] = {
            "client_ids": sorted(clients), "client_count": len(clients), "videos": len(subset),
            "bona_fide": sum(r.class_label == BONA_FIDE_LABEL for r in subset),
            "attack": sum(r.class_label == ATTACK_LABEL for r in subset),
            "attack_type_counts": dict(sorted(Counter(r.attack_type for r in subset if r.attack_type).items())),
            "capture_device_counts": dict(sorted(Counter(r.capture_device for r in subset).items())),
        }
        for client in clients:
            client_rows = [r for r in subset if r.client_id == client]
            real = [r for r in client_rows if r.class_label == BONA_FIDE_LABEL]
            per_client[client] = {
                "partition": split, "videos": len(client_rows), "bona_fide": len(real),
                "attack": len(client_rows) - len(real),
                "bona_fide_groups": sorted(r.video_id for r in real),
                "bona_fide_capture_devices": sorted(r.capture_device for r in real),
            }
        readiness[split] = {
            "bona_fide_videos_per_identity": {c: per_client[c]["bona_fide"] for c in sorted(clients)},
            "at_least_two_video_groups_per_identity": all(per_client[c]["bona_fide"] >= 2 for c in clients),
            "android_laptop_separation_possible": all(per_client[c]["bona_fide_capture_devices"] == list(CAMERAS) for c in clients),
            "session_disjoint_supported": False,
            "maximum_bona_fide_group_cv_folds": min(per_client[c]["bona_fide"] for c in clients),
        }
    return {
        "audit_schema_version": 1, "dataset": "MSU-MFSD", "protocol": "official_public_train_test",
        "source_fingerprints": protocol.source_fingerprints,
        "discovered_client_ids": sorted({r.client_id for r in rows}),
        "discovered_human_count": len({r.client_id for r in rows}),
        "human_identity_evidence": "README naming protocol plus numeric subject matching in DecFrames.m",
        "train_test_overlap": [], "official_development_partition": None,
        "recording_count": len(rows), "annotation_count": len(rows),
        "class_counts": dict(sorted(Counter(str(r.class_label) for r in rows).items())),
        "attack_type_counts": dict(sorted(Counter(r.attack_type for r in rows if r.attack_type).items())),
        "capture_device_counts": dict(sorted(Counter(r.capture_device for r in rows).items())),
        "presentation_device_counts": dict(sorted(Counter(r.presentation_device for r in rows if r.presentation_device).items())),
        "resolution_token_counts": dict(sorted(Counter(r.resolution_token for r in rows).items())),
        "media_byte_size_total": sum(r.byte_size for r in rows),
        "partitions": partitions, "per_client": dict(sorted(per_client.items())),
        "missing_expected_acquisition_slots": [], "unlisted_recordings": [],
        "missing_annotations": [], "orphan_annotations": [], "unexpected_client_ids": [],
        "duplicate_paths_or_recording_ids": [],
        "normalized_metadata_sha256": _digest(_json_bytes(_canonical_metadata(protocol))),
        "rq1_readiness": readiness,
        "rq3_official_unseen_subject_partition_validated": True,
        "validation": {
            "official_15_20_counts": True, "client_sets_disjoint": True,
            "complete_eight_video_slots_per_client": True, "source_files_exist": True,
            "nonempty_media_files": True, "unique_recording_groups": True,
            "media_decoded": False, "media_contents_hashed": False,
            "annotation_contents_fingerprinted": True,
        },
        "unresolved_ambiguities": [
            "No verified recording-session IDs; scene01 and camera tokens are not sessions.",
            "Video validity, frame indexing, rotation and crop fidelity remain untested without decoding.",
            "Metadata establishes client provenance; face content and byte-identical media copies were not checked.",
            "Subject lists enumerate humans, not files; completeness uses the documented eight acquisition slots.",
            "Attack-source camera cannot be assigned per recording from the filename; not inferred.",
        ],
    }


def write_msu_protocol_lock(root: str | Path, metadata_path: str | Path, audit_path: str | Path) -> dict:
    """Write deterministic outputs outside the raw root after repeated inventory checks."""
    root = Path(root).resolve()
    outputs = [Path(metadata_path).resolve(), Path(audit_path).resolve()]
    if outputs[0] == outputs[1] or any(path.is_relative_to(root) for path in outputs):
        raise ValueError("Lock outputs must be distinct and outside the raw dataset root")
    protocol = load_msu_protocol(root)
    report = audit_msu_protocol(protocol)
    if protocol != load_msu_protocol(root):
        raise ValueError("MSU inventory changed during repeated generation")
    for path, payload in zip(outputs, (_canonical_metadata(protocol), report)):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(_json_bytes(payload))
    return report


def verify_msu_protocol_lock(root: str | Path, audit_path: str | Path) -> None:
    """Reject drift in metadata, source lists, annotation bytes, names, or video sizes."""
    expected = json.loads(Path(audit_path).read_text(encoding="utf-8"))
    actual = audit_msu_protocol(load_msu_protocol(root))
    if actual != expected:
        raise ValueError("MSU protocol lock mismatch; review source/protocol changes before experiments")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--verify-lock", help="Check an existing audit without writing outputs")
    parser.add_argument("--metadata", help="Canonical recording JSON output")
    parser.add_argument("--audit", help="Deterministic audit JSON output")
    args = parser.parse_args()
    if args.verify_lock:
        if args.metadata or args.audit:
            parser.error("--verify-lock cannot be combined with output arguments")
        verify_msu_protocol_lock(args.root, args.verify_lock)
        print("MSU protocol lock verified; no media decoded.")
    elif args.metadata and args.audit:
        result = write_msu_protocol_lock(args.root, args.metadata, args.audit)
        print(f"Locked {result['recording_count']} source videos; no media decoded.")
    else:
        parser.error("Provide --verify-lock, or both --metadata and --audit")


if __name__ == "__main__":
    main()
