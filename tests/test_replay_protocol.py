"""Synthetic Replay metadata, not actual client IDs, media, or dataset audits."""

import json
import hashlib
from dataclasses import asdict, replace

import pytest

from identity_invariant_fas.constants import ATTACK_LABEL, BONA_FIDE_LABEL
from identity_invariant_fas.data.replay_protocol import (
    EXPECTED_CLIENT_COUNTS,
    ReplayClient,
    ReplayProtocol,
    ReplayRecording,
    inspect_replay_protocol,
    load_replay_metadata,
    prepare_replay_identity_probe,
    replay_pad_partitions,
    validate_replay_protocol,
)
from identity_invariant_fas.evaluation.identity_probe import validate_closed_set_probe_split


def fixture_protocol(complete=False):
    """Invented keys/paths are deliberately unrelated to raw Replay filenames."""
    clients, recordings = [], []
    for cohort, official_count in EXPECTED_CLIENT_COUNTS.items():
        for i in range(official_count if complete else 2):
            client = f"synthetic-{cohort}-{i}"
            clients.append(ReplayClient(client, cohort))
            for kind in ("access", "attack", "enroll"):
                recording_id = f"source:{client}:{kind}"
                recordings.append(ReplayRecording(
                    source_recording_id=recording_id,
                    filepath=f"invented/{client}/{kind}.mov",
                    client_id=client,
                    class_label=ATTACK_LABEL if kind == "attack" else BONA_FIDE_LABEL,
                    purpose="enroll" if kind == "enroll" else "pad",
                    pad_partition=None if kind == "enroll" else cohort,
                    light="controlled",
                    attack_device="mobile" if kind == "attack" else None,
                    attack_support="hand" if kind == "attack" else None,
                    sample_type="video" if kind == "attack" else None,
                    sample_device="highdef" if kind == "attack" else None,
                ))
    return ReplayProtocol("synthetic fixture, not official data", "a" * 64, tuple(clients), tuple(recordings))


def write_metadata(tmp_path, protocol):
    payload = {"schema_version": 1, "dataset": "Replay-Attack", **asdict(protocol)}
    path = tmp_path / "metadata.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_explicit_metadata_parsing_and_conditions(tmp_path):
    protocol = fixture_protocol()
    records = list(protocol.recordings)
    records[0] = replace(records[0], filepath=records[0].filepath.replace("/", "\\"))
    path = write_metadata(tmp_path, replace(protocol, recordings=tuple(records)))
    loaded = load_replay_metadata(path, require_complete=False)
    assert {r.client_id for r in loaded.recordings} == {c.client_id for c in protocol.clients}
    assert {r.filepath for r in loaded.recordings} == {r.filepath for r in protocol.recordings}
    attack = next(r for r in loaded.recordings if r.class_label == ATTACK_LABEL)
    assert (attack.attack_device, attack.attack_support, attack.sample_type) == ("mobile", "hand", "video")
    assert attack.sample_device == "highdef"
    assert attack.session_id is None
    assert all("Replay-Attack:recording:source%3A" in r.video_id for r in loaded.recordings)


def test_official_cardinalities_and_mutually_disjoint_partitions():
    protocol = fixture_protocol(complete=True)
    partitions = replay_pad_partitions(protocol)
    sets = {k: {r.client_id for r in v} for k, v in partitions.items()}
    assert {k: len(v) for k, v in sets.items()} == {"train": 15, "devel": 15, "test": 20}
    assert sets["train"].isdisjoint(sets["devel"] | sets["test"])
    assert sets["devel"].isdisjoint(sets["test"])
    assert all(r.purpose == "pad" for rows in partitions.values() for r in rows)


def test_partial_metadata_never_passes_default_complete_validation():
    protocol = fixture_protocol()
    with pytest.raises(ValueError, match="cardinalities"):
        validate_replay_protocol(protocol)
    result = inspect_replay_protocol(protocol, require_complete=False)
    assert result["expected_client_cardinalities_checked"] is False
    assert result["referenced_files_checked"] is False


def test_roster_partition_overlap_rejected():
    protocol = fixture_protocol()
    clients = protocol.clients + (replace(protocol.clients[0], pad_cohort="test"),)
    with pytest.raises(ValueError, match="overlapping client"):
        validate_replay_protocol(replace(protocol, clients=clients), require_complete=False)


def test_recording_client_partition_conflict_rejected():
    protocol = fixture_protocol()
    changed = replace(protocol.recordings[0], pad_partition="test")
    with pytest.raises(ValueError, match="partition conflict/overlap"):
        validate_replay_protocol(replace(protocol, recordings=(changed,) + protocol.recordings[1:]), require_complete=False)


@pytest.mark.parametrize("duplicate", ["id", "path", "conflicting_metadata"])
def test_duplicate_source_recordings_rejected(duplicate):
    protocol = fixture_protocol()
    row = protocol.recordings[0]
    if duplicate == "id":
        row = replace(row, filepath="other.mov")
    elif duplicate == "path":
        row = replace(row, source_recording_id="different")
    else:
        row = replace(row, client_id=protocol.clients[1].client_id)
    with pytest.raises(ValueError, match="Duplicate source recording"):
        validate_replay_protocol(replace(protocol, recordings=protocol.recordings + (row,)), require_complete=False)


@pytest.mark.parametrize("client", ["", " ", "unknown-official-client"])
def test_missing_or_unmapped_client_rejected(client):
    protocol = fixture_protocol()
    row = replace(protocol.recordings[0], client_id=client)
    with pytest.raises(ValueError, match="client"):
        validate_replay_protocol(replace(protocol, recordings=(row,) + protocol.recordings[1:]), require_complete=False)


@pytest.mark.parametrize("change", [
    {"class_label": ATTACK_LABEL}, {"pad_partition": "train"},
])
def test_enrollment_cannot_be_attack_or_pad_training(change):
    protocol = fixture_protocol()
    rows = list(protocol.recordings)
    index = next(i for i, r in enumerate(rows) if r.purpose == "enroll")
    rows[index] = replace(rows[index], **change)
    with pytest.raises(ValueError, match="Enrollment"):
        validate_replay_protocol(replace(protocol, recordings=tuple(rows)), require_complete=False)


def test_closed_set_enrollment_access_preparation():
    protocol = fixture_protocol(complete=True)
    prepared = prepare_replay_identity_probe(protocol, "test")
    train, evaluation = prepared.split.train_indices, prepared.split.eval_indices
    assert len(train) == len(evaluation) == 20
    assert {prepared.observations[i].identity_id for i in train} == {
        prepared.observations[i].identity_id for i in evaluation
    }
    assert all(prepared.recordings[i].purpose == "enroll" for i in train)
    assert all(prepared.recordings[i].purpose == "pad" for i in evaluation)
    assert all(r.class_label == BONA_FIDE_LABEL for r in prepared.recordings)
    assert all(o.session_id is None for o in prepared.observations)
    validate_closed_set_probe_split(prepared.observations, prepared.split)


@pytest.mark.parametrize("mode", ["wrong_name", "missing_enroll", "missing_access", "wrong_cohort"])
def test_invalid_enrollment_cohort_rejected(mode):
    protocol = fixture_protocol()
    if mode == "wrong_name":
        cohort = "train+test"
    else:
        cohort = "test"
        client = next(c.client_id for c in protocol.clients if c.pad_cohort == cohort)
        rows = list(protocol.recordings)
        kind = "pad" if mode == "missing_access" else "enroll"
        index = next(i for i, r in enumerate(rows) if r.client_id == client and r.purpose == kind
                     and r.class_label == BONA_FIDE_LABEL)
        if mode == "wrong_cohort":
            rows[index] = replace(rows[index], client_id=protocol.clients[0].client_id)
        else:
            rows.pop(index)
        protocol = replace(protocol, recordings=tuple(rows))
    with pytest.raises(ValueError, match="cohort"):
        prepare_replay_identity_probe(protocol, cohort, require_complete=False)


def test_frame_observations_retain_source_video_groups():
    protocol = fixture_protocol()
    videos = prepare_replay_identity_probe(protocol, "devel", require_complete=False)
    frames = {r.video_id: [20, 5, 9] for r in videos.recordings}
    prepared = prepare_replay_identity_probe(protocol, "devel", frame_indices=frames, require_complete=False)
    assert len(prepared.observations) == 12
    assert len({r.group_id for r in prepared.observations}) == 4
    assert len({r.sample_id for r in prepared.observations}) == 12
    assert prepared.observations[0].sample_id.endswith(":frame:5")
    validate_closed_set_probe_split(prepared.observations, prepared.split)


@pytest.mark.parametrize("mode", ["missing", "attack", "duplicate_frame", "empty"])
def test_invalid_frame_mapping_rejected(mode):
    protocol = fixture_protocol()
    videos = prepare_replay_identity_probe(protocol, "train", require_complete=False)
    frames = {r.video_id: [1, 2] for r in videos.recordings}
    key = next(iter(frames))
    if mode == "missing":
        frames.pop(key)
    elif mode == "attack":
        frames[next(r.video_id for r in protocol.recordings if r.class_label == ATTACK_LABEL)] = [1]
    elif mode == "duplicate_frame":
        frames[key] = [1, 1]
    else:
        frames[key] = []
    with pytest.raises(ValueError, match="Frame"):
        prepare_replay_identity_probe(protocol, "train", frame_indices=frames, require_complete=False)


def test_no_sessions_inferred_from_lighting_or_purpose():
    protocol = fixture_protocol()
    with pytest.raises(ValueError, match="session metadata"):
        prepare_replay_identity_probe(protocol, "train", session_disjoint=True, require_complete=False)
    row = replace(protocol.recordings[0], session_id="invented-without-evidence")
    with pytest.raises(ValueError, match="session_source"):
        validate_replay_protocol(replace(protocol, recordings=(row,) + protocol.recordings[1:]), require_complete=False)


def test_explicit_session_metadata_is_preserved_and_checked():
    protocol = fixture_protocol()
    # Synthetic verified-session fixture only; this is not a Replay session convention.
    rows = tuple(replace(r, session_id="synthetic-session-" + r.purpose,
                         session_source="synthetic explicit session ledger") for r in protocol.recordings)
    prepared = prepare_replay_identity_probe(replace(protocol, recordings=rows), "train",
                                            session_disjoint=True, require_complete=False)
    validate_closed_set_probe_split(prepared.observations, prepared.split, session_disjoint=True)
    same_session = tuple(replace(r, session_id="one-session") for r in rows)
    with pytest.raises(ValueError, match="session_id overlap"):
        prepare_replay_identity_probe(replace(protocol, recordings=same_session), "train",
                                      session_disjoint=True, require_complete=False)


def test_deterministic_ids_reports_and_root_independence(tmp_path):
    protocol = fixture_protocol()
    first = inspect_replay_protocol(protocol, require_complete=False)
    reordered = replace(protocol, clients=tuple(reversed(protocol.clients)),
                        recordings=tuple(reversed(protocol.recordings)))
    assert inspect_replay_protocol(reordered, require_complete=False) == first
    assert prepare_replay_identity_probe(protocol, "test", require_complete=False) == (
        prepare_replay_identity_probe(reordered, "test", require_complete=False)
    )
    reports = []
    for dirname in ("first-root", "other-root"):
        root = tmp_path / dirname
        for recording in protocol.recordings:
            path = root / recording.filepath
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"synthetic placeholder; no real media")
        reports.append(inspect_replay_protocol(protocol, root=root, require_complete=False))
    assert reports[0] == reports[1]
    assert reports[0]["referenced_files_checked"] is True


def test_missing_referenced_recording_fails(tmp_path):
    with pytest.raises(FileNotFoundError, match="Missing recording"):
        inspect_replay_protocol(fixture_protocol(), root=tmp_path, require_complete=False)


def test_source_artifact_fingerprint_check(tmp_path):
    artifact = tmp_path / "synthetic-source-mapping.txt"
    artifact.write_bytes(b"synthetic catalog bytes")
    protocol = replace(fixture_protocol(), source_sha256=hashlib.sha256(artifact.read_bytes()).hexdigest())
    report = inspect_replay_protocol(protocol, source_artifact=artifact, require_complete=False)
    assert report["source_artifact_hash_verified"] is True
    artifact.write_bytes(b"changed synthetic catalog")
    with pytest.raises(ValueError, match="Source artifact SHA-256"):
        inspect_replay_protocol(protocol, source_artifact=artifact, require_complete=False)


@pytest.mark.parametrize("changes", [
    {"source": ""}, {"source_sha256": ""}, {"source_sha256": "not-a-hash"},
])
def test_missing_provenance_fails(changes):
    with pytest.raises(ValueError, match="source"):
        validate_replay_protocol(replace(fixture_protocol(), **changes), require_complete=False)


@pytest.mark.parametrize("field,value", [
    ("attack_device", "invented-device"), ("pad_partition", "dev"),
    ("filepath", "../escape.mov"), ("class_label", True), ("attack_support", "hand"),
])
def test_metadata_conflicts_fail(field, value):
    protocol = fixture_protocol()
    row = replace(protocol.recordings[0], **{field: value})
    with pytest.raises(ValueError):
        validate_replay_protocol(replace(protocol, recordings=(row,) + protocol.recordings[1:]), require_complete=False)


def test_duplicate_json_keys_rejected(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text('{"dataset":"Replay-Attack","dataset":"other"}')
    with pytest.raises(ValueError, match="Duplicate JSON"):
        load_replay_metadata(path)


def test_complete_roster_does_not_hide_missing_pad_class():
    protocol = fixture_protocol(complete=True)
    rows = tuple(r for r in protocol.recordings if r != protocol.recordings[0])
    with pytest.raises(ValueError, match="Incomplete PAD"):
        validate_replay_protocol(replace(protocol, recordings=rows))
