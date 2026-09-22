"""Protocol tests with synthetic metadata/media placeholders, without decoding."""

from dataclasses import replace
from pathlib import Path

import pytest

from identity_invariant_fas.data.msu_protocol import (
    ATTACK_TYPES,
    audit_msu_protocol,
    load_msu_protocol,
    read_msu_subjects,
    validate_msu_protocol,
    verify_msu_protocol_lock,
    write_msu_protocol_lock,
)


@pytest.fixture
def msu_root(tmp_path):
    root = tmp_path / "synthetic-msu"
    root.mkdir()
    (root / "train_sub_list.txt").write_text("\n".join(f"{i:02d}" for i in range(1, 16)))
    (root / "test_sub_list.txt").write_text("\n".join(f"{i:02d}" for i in range(16, 36)))
    for name in ("README.txt", "DecFrames.m"):
        (root / name).write_text("Synthetic provenance fixture, not the official release")
    for i in range(1, 36):
        for camera, extension in (("android", "mp4"), ("laptop", "mov")):
            for attack in (None, *ATTACK_TYPES):
                kind = "real" if attack is None else "attack"
                middle = "" if attack is None else "_" + attack
                name = f"{kind}_client{i:03d}_{camera}_SD{middle}_scene01.{extension}"
                video = root / "scene01" / kind / name
                video.parent.mkdir(parents=True, exist_ok=True)
                video.write_bytes(b"synthetic nonempty video placeholder")
                video.with_suffix(".face").write_text("0, 1, 2, 3, 4, 1, 2, 3, 4\n")
    return root


def test_numeric_subject_mapping_and_official_disjointness(msu_root):
    protocol = load_msu_protocol(msu_root)
    assert protocol.train_clients == tuple(f"{i:03d}" for i in range(1, 16))
    assert protocol.test_clients == tuple(f"{i:03d}" for i in range(16, 36))
    assert set(protocol.train_clients).isdisjoint(protocol.test_clients)
    assert len(protocol.recordings) == 280
    report = audit_msu_protocol(protocol)
    assert report["partitions"]["train"]["bona_fide"] == 30
    assert report["partitions"]["test"]["attack"] == 120
    assert report["official_development_partition"] is None
    assert report["rq1_readiness"]["test"]["maximum_bona_fide_group_cv_folds"] == 2
    assert report["rq1_readiness"]["test"]["android_laptop_separation_possible"]
    assert not report["rq1_readiness"]["test"]["session_disjoint_supported"]


def test_subject_overlap_rejected(msu_root):
    path = msu_root / "test_sub_list.txt"
    path.write_text(path.read_text().replace("16", "01", 1))
    with pytest.raises(ValueError, match="identity overlap"):
        load_msu_protocol(msu_root)


def test_padding_alias_duplicate_rejected(tmp_path):
    path = tmp_path / "subjects.txt"
    path.write_text("02\n002\n")
    with pytest.raises(ValueError, match="Duplicate subject mapping"):
        read_msu_subjects(path)


@pytest.mark.parametrize("token", ["client001", "0", "56", "1.0", ""])
def test_invalid_subject_token_or_cardinality(tmp_path, token):
    path = tmp_path / "subjects.txt"
    path.write_text(token)
    if token:
        with pytest.raises(ValueError, match="Invalid documented"):
            read_msu_subjects(path)
    else:
        assert read_msu_subjects(path) == ()  # cardinality is checked by protocol validation


@pytest.mark.parametrize("change", ["duplicate", "unmapped", "partition", "camera", "class", "session", "annotation"])
def test_recording_conflicts_fail(msu_root, change):
    protocol = load_msu_protocol(msu_root)
    row = protocol.recordings[0]
    if change == "duplicate":
        rows = protocol.recordings + (row,)
    else:
        fields = {
            "unmapped": {"client_id": "055"}, "partition": {"pad_partition": "test"},
            "camera": {"capture_device": "laptop"}, "class": {"class_label": 0},
            "session": {"session_id": "scene01"}, "annotation": {"annotation_filepath": "other.face"},
        }
        rows = (replace(row, **fields[change]),) + protocol.recordings[1:]
    with pytest.raises(ValueError):
        validate_msu_protocol(replace(protocol, recordings=rows))


def test_missing_subject_mapping_fails(msu_root):
    path = msu_root / "train_sub_list.txt"
    path.write_text(path.read_text().replace("01", "55", 1))
    with pytest.raises(ValueError, match="unexpected client"):
        load_msu_protocol(msu_root)


def test_wrong_cardinality_fails(msu_root):
    protocol = load_msu_protocol(msu_root)
    with pytest.raises(ValueError, match="Expected 15 train"):
        validate_msu_protocol(replace(protocol, train_clients=protocol.train_clients[:-1]))


def test_provenance_and_frame_group_preserved(msu_root):
    protocol = load_msu_protocol(msu_root)
    row = next(r for r in protocol.recordings if r.attack_type == "ipad_video")
    assert row.presentation_device == "iPad Air"
    assert row.capture_device == "android"
    assert "Nexus" in row.capture_model
    assert len(row.annotation_sha256) == 64
    assert len(protocol.source_fingerprints["train_sub_list.txt"]) == 64
    first, second = row.probe_observation(0), row.probe_observation(10)
    assert first.group_id == second.group_id == row.video_id
    assert first.sample_id != second.sample_id
    assert first.identity_id == row.client_id
    assert first.session_id is None
    with pytest.raises(ValueError, match="frame_index"):
        row.probe_observation(-1)


def test_missing_video_and_orphan_annotation_fail(msu_root):
    video = next(msu_root.rglob("*.mp4"))
    video.unlink()
    with pytest.raises(ValueError, match="Orphan"):
        load_msu_protocol(msu_root)
    video.with_suffix(".face").unlink()
    with pytest.raises(ValueError, match="missing slots"):
        load_msu_protocol(msu_root)


def test_missing_face_annotation_fails(msu_root):
    next(msu_root.rglob("*.face")).unlink()
    with pytest.raises(FileNotFoundError, match="Missing referenced"):
        load_msu_protocol(msu_root)


def test_unlisted_video_location_fails(msu_root):
    (msu_root / "extra.mp4").write_bytes(b"synthetic")
    with pytest.raises(ValueError, match="Unlisted/unsupported"):
        load_msu_protocol(msu_root)


def test_repeated_output_and_relocation_are_deterministic(msu_root, tmp_path):
    metadata, audit = tmp_path / "metadata.json", tmp_path / "audit.json"
    first = write_msu_protocol_lock(msu_root, metadata, audit)
    before = metadata.read_bytes(), audit.read_bytes()
    assert first == write_msu_protocol_lock(msu_root, metadata, audit)
    assert before == (metadata.read_bytes(), audit.read_bytes())
    moved = tmp_path / "relocated-synthetic"
    msu_root.rename(moved)  # synthetic fixture only
    assert first == write_msu_protocol_lock(moved, metadata, audit)
    assert before == (metadata.read_bytes(), audit.read_bytes())
    verify_msu_protocol_lock(moved, audit)
    next(moved.rglob("*.face")).write_text("changed synthetic annotation")
    with pytest.raises(ValueError, match="lock mismatch"):
        verify_msu_protocol_lock(moved, audit)


def test_outputs_cannot_modify_raw_root(msu_root, tmp_path):
    with pytest.raises(ValueError, match="outside the raw"):
        write_msu_protocol_lock(msu_root, msu_root / "new.json", tmp_path / "audit.json")


def test_legacy_loader_padding_limitation_is_reproduced(msu_root):
    from identity_invariant_fas.data.msu import index_msu_videos

    with pytest.warns(FutureWarning, match="Legacy MSU loader"), pytest.raises(ValueError, match="absent from official MSU split lists"):
        index_msu_videos(msu_root)


def test_local_release_matches_lock_when_available():
    repository = Path(__file__).parents[1]
    root = repository / "datasets/MSU"
    if not (root / "README.txt").is_file() or not (root / "scene01").is_dir():
        pytest.skip("Licensed MSU release is not installed locally")
    lock = repository / "docs/audit/msu_protocol_lock.json"
    verify_msu_protocol_lock(root, lock)
