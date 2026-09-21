"""Explicit Replay-Attack source metadata; no filename-derived identity mapping.

See docs/REPLAY_PROTOCOL.md for the normalized input contract and its evidence.
The legacy replay_attack.load_replay_attack_manifest API is unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Mapping, Sequence
from urllib.parse import quote

from ..constants import ATTACK_LABEL, BONA_FIDE_LABEL
from ..evaluation.identity_probe import (
    ProbeObservation,
    ProbeSplit,
    validate_closed_set_probe_split,
)


# Original Replay-Attack cohorts, not generic dataset/probe constraints.
# Source: Idiap Replay-Attack documentation, Protocols for Biometric Recognition.
EXPECTED_CLIENT_COUNTS = {"train": 15, "devel": 15, "test": 20}
PAD_PARTITIONS = tuple(EXPECTED_CLIENT_COUNTS)


def _identifier(value: str, name: str) -> None:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{name} must be an explicit non-empty string without outer whitespace")


def _relative_path(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("filepath must be a non-empty string")
    value = value.replace("\\", "/")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or not path.parts or ":" in value or "\0" in value:
        raise ValueError("filepath must be portable and dataset-relative")
    return path.as_posix()


@dataclass(frozen=True)
class ReplayClient:
    """An explicit official client-to-cohort mapping, independent of recording purpose."""

    client_id: str
    pad_cohort: str


@dataclass(frozen=True)
class ReplayRecording:
    """One original recording, not a frame; None denotes undocumented metadata."""

    source_recording_id: str
    filepath: str
    client_id: str
    class_label: int
    purpose: str  # 'pad' or 'enroll'; enrollment is never a PAD partition
    pad_partition: str | None
    light: str | None = None
    attack_device: str | None = None
    attack_support: str | None = None
    sample_type: str | None = None
    sample_device: str | None = None
    take: int | None = None
    session_id: str | None = None
    session_source: str | None = None

    @property
    def video_id(self) -> str:
        return "Replay-Attack:recording:" + quote(self.source_recording_id, safe="")


@dataclass(frozen=True)
class ReplayProtocol:
    """Normalized export with a source-catalog reference and SHA-256 fingerprint.

    The fingerprint records producer-asserted provenance, not authentication.
    Full client/recording mappings must be exported from official metadata.
    """

    source: str
    source_sha256: str
    clients: tuple[ReplayClient, ...]
    recordings: tuple[ReplayRecording, ...]


def validate_replay_protocol(protocol: ReplayProtocol, *, require_complete: bool = True) -> None:
    """Fail on missing provenance, overlaps, duplicates, or role/metadata conflicts.

    Complete mode enforces the official roster and PAD class coverage per client.
    It does not prove inventory completeness or authenticity of supplied mappings.
    """
    _identifier(protocol.source, "source metadata reference")
    if not isinstance(protocol.source_sha256, str) or not re.fullmatch(
        r"[0-9a-f]{64}", protocol.source_sha256
    ):
        raise ValueError("source_sha256 must fingerprint the official mapping artifact")
    if not protocol.clients or not protocol.recordings:
        raise ValueError("Non-empty explicit client roster and recording metadata are required")
    clients = {}
    for client in protocol.clients:
        _identifier(client.client_id, "client_id")
        if client.pad_cohort not in PAD_PARTITIONS:
            raise ValueError(f"Unknown client cohort: {client.pad_cohort!r}")
        if client.client_id in clients:
            raise ValueError(f"Duplicate/overlapping client in roster: {client.client_id}")
        clients[client.client_id] = client.pad_cohort
    counts = Counter(clients.values())
    if require_complete and dict(counts) != EXPECTED_CLIENT_COUNTS:
        raise ValueError(f"Expected official client cardinalities {EXPECTED_CLIENT_COUNTS}; got {dict(counts)}")

    ids, paths = set(), set()
    observed = {partition: set() for partition in PAD_PARTITIONS}
    classes: dict[str, set[int]] = {}
    choices = {
        "light": {"controlled", "adverse"},
        "attack_device": {"print", "mobile", "highdef"},
        "attack_support": {"fixed", "hand"},
        "sample_type": {"photo", "video"},
        "sample_device": {"mobile", "highdef"},
    }
    for row in protocol.recordings:
        _identifier(row.source_recording_id, "source_recording_id")
        _identifier(row.client_id, "client_id")
        if row.client_id not in clients:
            raise ValueError(f"No official client mapping for {row.client_id!r}")
        path = _relative_path(row.filepath)
        if path != row.filepath:
            raise ValueError("Recording filepath must be normalized to forward slashes")
        if row.source_recording_id in ids or path in paths:
            raise ValueError(f"Duplicate source recording: {row.source_recording_id!r}, {path!r}")
        ids.add(row.source_recording_id)
        paths.add(path)
        if type(row.class_label) is not int or row.class_label not in {BONA_FIDE_LABEL, ATTACK_LABEL}:
            raise ValueError("Invalid Replay PAD class_label")
        if row.purpose == "enroll":
            if row.pad_partition is not None or row.class_label != BONA_FIDE_LABEL:
                raise ValueError("Enrollment must be bona fide and have no PAD partition")
        elif row.purpose == "pad":
            if row.pad_partition not in PAD_PARTITIONS:
                raise ValueError("PAD recordings require train/devel/test pad_partition")
            if row.pad_partition != clients[row.client_id]:
                raise ValueError(f"PAD client partition conflict/overlap for {row.client_id}")
            observed[row.pad_partition].add(row.client_id)
            classes.setdefault(row.client_id, set()).add(row.class_label)
        else:
            raise ValueError("purpose must be 'pad' or 'enroll'")
        for name, allowed in choices.items():
            value = getattr(row, name)
            if value is not None and (not isinstance(value, str) or value not in allowed):
                raise ValueError(f"Unsupported {name}: {value!r}")
        if row.class_label == BONA_FIDE_LABEL and any(
            getattr(row, name) is not None
            for name in ("attack_device", "attack_support", "sample_type", "sample_device")
        ):
            raise ValueError("Bona-fide recordings cannot carry attack-specific metadata")
        if row.take is not None and (type(row.take) is not int or row.take < 0):
            raise ValueError("take must be a nonnegative source integer or null")
        if (row.session_id is None) != (row.session_source is None):
            raise ValueError("session_id requires an explicit session_source and vice versa")
        if row.session_id is not None:
            _identifier(row.session_id, "session_id")
            _identifier(row.session_source, "session_source")
    for left, partition in enumerate(PAD_PARTITIONS):
        for other in PAD_PARTITIONS[left + 1:]:
            if observed[partition] & observed[other]:
                raise ValueError("PAD client sets overlap")
    if require_complete:
        for client in clients:
            if classes.get(client) != {BONA_FIDE_LABEL, ATTACK_LABEL}:
                raise ValueError(f"Incomplete PAD bona-fide/attack coverage for client {client}")


def _unique_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON field: {key}")
        result[key] = value
    return result


def load_replay_metadata(path: str | Path, *, require_complete: bool = True) -> ReplayProtocol:
    """Parse our explicit normalized JSON format; never infer metadata from filenames."""
    data = json.loads(Path(path).read_text(encoding="utf-8-sig"), object_pairs_hook=_unique_keys)
    if not isinstance(data, dict) or set(data) != {
        "schema_version", "dataset", "source", "source_sha256", "clients", "recordings"
    }:
        raise ValueError("Invalid Replay metadata document fields")
    if type(data["schema_version"]) is not int or data["schema_version"] != 1 or data["dataset"] != "Replay-Attack":
        raise ValueError("Expected Replay-Attack metadata schema_version 1")
    try:
        clients = tuple(ReplayClient(**row) for row in data["clients"])
        recordings = tuple(
            ReplayRecording(**{**row, "filepath": _relative_path(row["filepath"])})
            for row in data["recordings"]
        )
    except (TypeError, KeyError) as error:
        raise ValueError(f"Invalid client/recording metadata: {error}") from error
    protocol = ReplayProtocol(data["source"], data["source_sha256"], clients, recordings)
    validate_replay_protocol(protocol, require_complete=require_complete)
    return ReplayProtocol(
        protocol.source, protocol.source_sha256,
        tuple(sorted(clients, key=lambda row: row.client_id)),
        tuple(sorted(recordings, key=lambda row: row.video_id)),
    )


def replay_pad_partitions(
    protocol: ReplayProtocol, *, require_complete: bool = True
) -> dict[str, tuple[ReplayRecording, ...]]:
    """Return only PAD recordings after validating official client disjointness."""
    validate_replay_protocol(protocol, require_complete=require_complete)
    return {
        partition: tuple(sorted(
            (row for row in protocol.recordings if row.purpose == "pad" and row.pad_partition == partition),
            key=lambda row: row.video_id,
        ))
        for partition in PAD_PARTITIONS
    }


@dataclass(frozen=True)
class ReplayIdentityPreparation:
    """Aligned video/frame observation plan, not extracted features or probe results."""

    recordings: tuple[ReplayRecording, ...]
    observations: tuple[ProbeObservation, ...]
    split: ProbeSplit


def prepare_replay_identity_probe(
    protocol: ReplayProtocol,
    cohort: str,
    *,
    frame_indices: Mapping[str, Sequence[int]] | None = None,
    session_disjoint: bool = False,
    require_complete: bool = True,
) -> ReplayIdentityPreparation:
    """Enrollment -> separate real-access observations for exactly one full cohort.

    frame_indices, if supplied, maps selected video IDs to actual sampled frame
    indices. No frames, identities, sessions, or feature vectors are manufactured.
    """
    validate_replay_protocol(protocol, require_complete=require_complete)
    if cohort not in PAD_PARTITIONS:
        raise ValueError("Select one official client cohort: train, devel, or test")
    clients = {row.client_id for row in protocol.clients if row.pad_cohort == cohort}
    enrollment = sorted(
        (r for r in protocol.recordings if r.purpose == "enroll" and r.client_id in clients),
        key=lambda row: row.video_id,
    )
    access = sorted(
        (r for r in protocol.recordings if r.purpose == "pad" and r.pad_partition == cohort
         and r.class_label == BONA_FIDE_LABEL), key=lambda row: row.video_id,
    )
    if not clients or {r.client_id for r in enrollment} != clients or {r.client_id for r in access} != clients:
        raise ValueError("Enrollment/access identity classes must equal the entire selected client cohort")
    selected = tuple(enrollment + access)
    if frame_indices is not None and set(frame_indices) != {r.video_id for r in selected}:
        raise ValueError("Frame mapping must cover exactly the selected enrollment/access recordings")
    rows, aligned = [], []
    train, evaluation = [], []
    for recording in selected:
        if frame_indices is None:
            suffixes = [":video"]
        else:
            frames = tuple(frame_indices[recording.video_id])
            if not frames or any(type(i) is not int or i < 0 for i in frames) or len(set(frames)) != len(frames):
                raise ValueError("Frame indices must be non-empty, unique, nonnegative integers")
            suffixes = [f":frame:{index}" for index in sorted(frames)]
        for suffix in suffixes:
            (train if recording.purpose == "enroll" else evaluation).append(len(rows))
            rows.append(ProbeObservation(
                sample_id=recording.video_id + suffix,
                identity_id=recording.client_id,
                group_id=recording.video_id,
                session_id=recording.session_id,
            ))
            aligned.append(recording)
    split = ProbeSplit(tuple(train), tuple(evaluation))
    validate_closed_set_probe_split(rows, split, session_disjoint=session_disjoint)
    return ReplayIdentityPreparation(tuple(aligned), tuple(rows), split)


def inspect_replay_protocol(
    protocol: ReplayProtocol, *, root: str | Path | None = None,
    source_artifact: str | Path | None = None, require_complete: bool = True
) -> dict:
    """Metadata inspection; optional read-only file and source-fingerprint checks.

    A matching hash proves correspondence to the supplied artifact, not that the
    artifact is official. Its origin still needs independent verification.
    """
    partitions = replay_pad_partitions(protocol, require_complete=require_complete)
    if source_artifact is not None:
        digest = hashlib.sha256(Path(source_artifact).read_bytes()).hexdigest()
        if digest != protocol.source_sha256:
            raise ValueError("Source artifact SHA-256 does not match metadata provenance")
    if root is not None:
        root = Path(root).resolve()
        seen_paths = set()
        for row in protocol.recordings:
            path = root.joinpath(*PurePosixPath(row.filepath).parts).resolve()
            if not path.is_relative_to(root):
                raise ValueError(f"Referenced recording escapes root: {row.filepath}")
            if not path.is_file():
                raise FileNotFoundError(f"Missing recording: {row.filepath}")
            if path in seen_paths:
                raise ValueError(f"Duplicate physical recording path: {row.filepath}")
            seen_paths.add(path)
    result = {
        "dataset": "Replay-Attack",
        "source": protocol.source,
        "source_sha256": protocol.source_sha256,
        "expected_client_cardinalities_checked": require_complete,
        "referenced_files_checked": root is not None,
        "source_artifact_hash_verified": source_artifact is not None,
        "identity_mapping_authenticity": "requires independent source verification",
        "client_count": len(protocol.clients),
        "recording_count": len(protocol.recordings),
        "pad_client_disjointness_validated": True,
        "partitions": {},
    }
    for partition, records in partitions.items():
        clients = {c.client_id for c in protocol.clients if c.pad_cohort == partition}
        enroll = [r for r in protocol.recordings if r.purpose == "enroll" and r.client_id in clients]
        result["partitions"][partition] = {
            "roster_client_ids": sorted(clients),
            "pad_client_ids": sorted({r.client_id for r in records}),
            "bona_fide_videos": sum(r.class_label == BONA_FIDE_LABEL for r in records),
            "attack_videos": sum(r.class_label == ATTACK_LABEL for r in records),
            "enrollment_videos": len(enroll),
            "enrollment_client_ids": sorted({r.client_id for r in enroll}),
            "missing_enrollment_clients": sorted(clients - {r.client_id for r in enroll}),
        }
    canonical = {
        "clients": [asdict(c) for c in sorted(protocol.clients, key=lambda c: c.client_id)],
        "recordings": [asdict(r) for r in sorted(protocol.recordings, key=lambda r: r.video_id)],
    }
    result["normalized_metadata_sha256"] = hashlib.sha256(
        json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", required=True, help="Explicit normalized Replay JSON export")
    parser.add_argument("--root", help="Optionally check every referenced recording exists")
    parser.add_argument("--source-artifact", help="Verify the fingerprint of the original mapping artifact")
    parser.add_argument("--allow-partial", action="store_true", help="Inspection only; skip full roster/coverage checks")
    args = parser.parse_args()
    protocol = load_replay_metadata(args.metadata, require_complete=not args.allow_partial)
    report = inspect_replay_protocol(
        protocol, root=args.root, source_artifact=args.source_artifact,
        require_complete=not args.allow_partial,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
