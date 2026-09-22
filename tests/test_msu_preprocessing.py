"""Minimal synthetic media/annotations; licensed data are optional and never mutated."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("cv2")

from identity_invariant_fas.data import msu_preprocessing as prep
from identity_invariant_fas.data.msu_protocol import load_msu_protocol, write_msu_protocol_lock
from test_msu_protocol import msu_root  # shared complete synthetic release fixture


@pytest.fixture
def locked(msu_root, tmp_path):
    lock = tmp_path / "lock.json"
    write_msu_protocol_lock(msu_root, tmp_path / "metadata.json", lock)
    return msu_root, lock


@pytest.fixture
def fake_decoder(monkeypatch):
    def decode(path, indices, *, auto=False):
        for i in indices:
            yield i, np.arange(8 * 8 * 3, dtype=np.uint8).reshape(8, 8, 3), 0
    monkeypatch.setattr(prep, "decode_selected", decode)


def test_annotations_preserve_index_raw_box_and_eyes():
    raw = " 7, 1.25, 2, 4, 6, 2, 3, 3, 3 "
    rows, blanks = prep.parse_face_annotations(("\n" + raw + "\n").encode())
    assert rows[7] == prep.FaceAnnotation(7, (1.25, 2, 4, 6, 2, 3, 3, 3), 2, raw)
    assert blanks == 1


@pytest.mark.parametrize("content", [b"", b"0,1,2,3,4", b"0.5,1,2,3,4,1,2,3,4",
    b"0,NaN,2,3,4,1,2,3,4", b"0,1,2,0,4,1,2,3,4", b"0,1,,3,4,1,2,3,4", b"\xff"])
def test_malformed_rows_are_explicit_errors(content):
    with pytest.raises(prep.AnnotationError) as exc:
        prep.parse_face_annotations(content)
    assert exc.value.malformed


def test_duplicate_annotation_index_rejected_even_identical():
    with pytest.raises(prep.AnnotationError) as exc:
        prep.parse_face_annotations(b"0,1,2,3,4,1,2,3,4\n0,1,2,3,4,1,2,3,4\n")
    assert exc.value.duplicates == [{"frame_index": 0, "line": 2, "first_line": 1}]


def test_deterministic_unique_frame_selection():
    indices = [101, 7, 0, 9, 20, 30]
    assert prep.select_frame_indices(indices, 4) == (0, 7, 20, 101)
    assert prep.select_frame_indices(reversed(indices), 4) == (0, 7, 20, 101)
    assert prep.select_frame_indices(indices, 99) == (0, 7, 9, 20, 30, 101)
    assert prep.select_frame_indices(indices, 1) == (0,)
    for n in range(1, 20):
        for k in range(1, 20):
            result = prep.select_frame_indices(range(n), k)
            assert len(result) == len(set(result)) == min(n, k)


@pytest.mark.parametrize("indices,count", [([], 1), ([0, 0], 1), ([-1], 1), ([0.5], 1), ([0], 0), ([0], True)])
def test_invalid_frame_selection_rejected(indices, count):
    with pytest.raises(ValueError):
        prep.select_frame_indices(indices, count)


def test_crop_geometry_expansion_clipping_and_empty_rejection():
    assert prep.crop_box((1.5, 2, 6.5, 7), 8, 8, 0.2) == (0, 1, 8, 8)
    with pytest.raises(ValueError, match="intersection"):
        prep.crop_box((10, 10, 15, 15), 8, 8, 0)


def test_config_digest_stable_and_settings_sensitive():
    original = prep.digest(prep.json_bytes(prep.PreprocessingConfig().payload()))
    assert original == prep.digest(prep.json_bytes(prep.PreprocessingConfig().payload()))
    for config in (prep.PreprocessingConfig(1), prep.PreprocessingConfig(margin=0.1), prep.PreprocessingConfig(output_size=(4, 5))):
        assert prep.digest(prep.json_bytes(config.payload())) != original
    for config in (prep.PreprocessingConfig(0), prep.PreprocessingConfig(margin=float("nan")), prep.PreprocessingConfig(output_size=(0, 5))):
        with pytest.raises(ValueError):
            config.payload()


def test_canonical_integration_normalization_relocation_and_roundtrip(locked, tmp_path, fake_decoder):
    root, lock = locked
    protocol = load_msu_protocol(root)
    sources = [r for r in protocol.recordings if r.client_id in ("002", "016") and r.capture_device == "android" and r.attack_type in (None, "ipad_video")]
    ids = [r.video_id for r in sources]
    assert "02" in (root / "train_sub_list.txt").read_text().split()
    first = tmp_path / "first"
    report = prep.run_preprocessing(root, lock, first, mode="process", video_ids=ids)
    rows = prep.read_processed_manifest(first / "manifest.json", root, lock)
    assert len(rows) == 4
    assert rows.processing_status == "complete"
    assert rows.fidelity_status == "manual_review_pending"
    assert rows.experiment_ready is False
    assert {r.client_id for r in rows} == {"002", "016"}
    assert {r.pad_partition for r in rows} == {"train", "test"}
    assert report["group_id_preservation_validated"] is True
    for row, source in zip(rows, sorted(sources, key=lambda r: r.video_id)):
        assert row.source_video_id == source.video_id
        assert row.source_filepath == source.filepath
        assert row.frame_index == 0 and row.session_id is None
        assert row.probe_observation().group_id == source.video_id
        assert row.probe_observation().identity_id == source.client_id
        assert row.probe_observation().session_id is None
        assert row.class_label == source.class_label
        assert row.attack_type == source.attack_type
        assert row.presentation_device == source.presentation_device
        assert row.crop_box == (1, 2, 3, 4)
        assert row.output_size == (2, 2)
        assert row.annotation_values == (1, 2, 3, 4, 1, 2, 3, 4)
    relocated = tmp_path / "relocated"
    root.rename(relocated)  # only the synthetic fixture
    second = tmp_path / "second"
    assert report == prep.run_preprocessing(relocated, lock, second, mode="process", video_ids=list(reversed(ids)))
    assert (first / "manifest.json").read_bytes() == (second / "manifest.json").read_bytes()
    for row in rows:
        assert (first / row.crop_filepath).read_bytes() == (second / row.crop_filepath).read_bytes()
    with pytest.raises(FileExistsError):
        prep.run_preprocessing(relocated, lock, first, mode="process", video_ids=ids)


def test_plan_does_not_decode_and_records_actual_zero_outputs(locked, tmp_path, monkeypatch):
    root, lock = locked
    monkeypatch.setattr(prep, "decode_selected", lambda *a, **k: pytest.fail("plan decoded media"))
    report = prep.run_preprocessing(root, lock, tmp_path / "plan")
    assert report["planned_frames"] == 280
    assert report["frames_written"] == 0
    assert report["processing_status"] == "not_processed"
    assert report["experiment_ready"] is False
    assert report["source_video_count_processed"] == 0
    assert report["source_videos_index_validated"] == 280
    assert report["frames_by_pad_partition"] == {}
    assert report["planned_frames_by_pad_partition"] == {"train": 120, "test": 160}
    assert not (tmp_path / "plan/manifest.json").exists()


def test_all_invalid_annotations_accounted_before_any_crop(msu_root, tmp_path, monkeypatch):
    files = sorted(msu_root.rglob("*.face"))[:2]
    files[0].write_text("bad row\n")
    files[1].write_text("0,1,2,3,4,1,2,3,4\n" * 2)
    lock = tmp_path / "lock.json"
    write_msu_protocol_lock(msu_root, tmp_path / "metadata.json", lock)
    monkeypatch.setattr(prep, "decode_selected", lambda *a, **k: pytest.fail("invalid annotations decoded"))
    with pytest.raises(ValueError, match="Annotation audit failed"):
        prep.run_preprocessing(msu_root, lock, tmp_path / "out", mode="process")
    report = json.loads((tmp_path / "out/audit.json").read_text())
    assert len(report["malformed_annotation_rows"]) == len(report["duplicate_annotation_indices"]) == 1
    assert len(report["source_videos_failed"]) == 2
    assert report["frames_written"] == 0
    assert not (tmp_path / "out/manifest.json").exists()


@pytest.mark.parametrize("failure", ["missing_annotation", "missing_decoded", "decoder_error", "write"])
def test_failures_accounted_without_success_manifest(locked, tmp_path, monkeypatch, failure):
    root, lock = locked
    source = load_msu_protocol(root).recordings[0]
    def decode(*args, **kwargs):
        if failure == "decoder_error":
            raise prep.FrameDecodeError([0])
        if failure != "missing_decoded":
            yield (5 if failure == "missing_annotation" else 0), np.zeros((8, 8, 3), np.uint8), 0
    monkeypatch.setattr(prep, "decode_selected", decode)
    if failure == "write":
        monkeypatch.setattr(prep, "_png", lambda f: (_ for _ in ()).throw(ValueError("encoding failed")))
    with pytest.raises(ValueError):
        prep.run_preprocessing(root, lock, tmp_path / "out", mode="process", video_ids=[source.video_id])
    report = json.loads((tmp_path / "out/audit.json").read_text())
    key = {"missing_annotation": "missing_selected_annotations", "missing_decoded": "missing_decoded_frames",
           "decoder_error": "missing_decoded_frames", "write": "crop_write_failures"}[failure]
    assert report[key]
    assert report["status"] == "failed"
    assert report["processing_status"] == "failed"
    assert report["experiment_ready"] is False
    assert report["source_videos_failed"] == [source.video_id]
    assert not (tmp_path / "out/manifest.json").exists()


def test_manifest_rejects_source_crop_and_annotation_conflicts(locked, tmp_path, fake_decoder):
    root, lock = locked
    protocol = load_msu_protocol(root)
    report = prep.run_preprocessing(root, lock, tmp_path / "out", mode="process", video_ids=[protocol.recordings[0].video_id])
    path = tmp_path / "out/manifest.json"
    original = json.loads(path.read_text())
    for field, value in (("client_id", "999"), ("source_video_id", "legacy-stem"), ("pad_partition", "test"),
                         ("session_id", "scene01"), ("frame_index", -1), ("crop_box", [0, 0, 1, 1]),
                         ("sample_id", "wrong"), ("annotation_values", [0] * 8),
                         ("class_label", 1.0), ("rotation_degrees", False), ("crop_box", [1.0, 2.0, 3.0, 4.0])):
        doc = copy.deepcopy(original)
        doc["frames"][0][field] = value
        with pytest.raises(ValueError):
            prep.validate_processed_manifest(doc, protocol, report["protocol_lock_digest"])
    doc = copy.deepcopy(original)
    doc["frames"] *= 2
    with pytest.raises(ValueError, match="Duplicate"):
        prep.validate_processed_manifest(doc, protocol, report["protocol_lock_digest"])
    # Forging a line number cannot pass the canonical reader's check against raw annotations.
    doc = copy.deepcopy(original)
    doc["frames"][0]["annotation_line_number"] = 2
    path.write_bytes(prep.json_bytes(doc))
    with pytest.raises(ValueError, match="locked source"):
        prep.read_processed_manifest(path, root, lock)
    path.write_bytes(prep.json_bytes(original))
    (path.parent / original["frames"][0]["crop_filepath"]).write_bytes(b"corrupt synthetic crop")
    with pytest.raises(ValueError, match="Crop content"):
        prep.read_processed_manifest(path, root, lock)


def test_orientation_comparison_and_verified_scope(locked, tmp_path, monkeypatch):
    root, lock = locked
    protocol = load_msu_protocol(root)
    source = next(r for r in protocol.recordings if r.pad_partition == "train")
    frame = np.arange(8 * 8 * 3, dtype=np.uint8).reshape(8, 8, 3)
    def decode(path, indices, *, auto=False):
        for i in indices:
            yield i, prep.rotate_frame(frame, 180) if auto else frame, 180
    monkeypatch.setattr(prep, "decode_selected", decode)
    report = prep.run_preprocessing(root, lock, tmp_path / "out", mode="verify", video_ids=[source.video_id])
    assert report["orientation_checks"][0]["auto_equals_explicit_rotation"]
    assert report["orientation_checks"][0]["metadata_rotation_degrees"] == 180
    assert report["orientation_status"] == "manual_review_pending"
    assert report["processing_status"] == "complete"
    assert report["fidelity_status"] == "manual_review_pending"
    assert report["experiment_ready"] is False
    assert len(list((tmp_path / "out/review").glob("*.png"))) == 1
    test_video = next(r for r in protocol.recordings if r.pad_partition == "test")
    with pytest.raises(ValueError, match="training videos"):
        prep.run_preprocessing(root, lock, tmp_path / "bad", mode="verify", video_ids=[test_video.video_id])
    with pytest.raises(ValueError, match="rotations"):
        prep.rotate_frame(frame, 90)


def test_raw_root_and_changed_lock_are_rejected(locked, tmp_path):
    root, lock = locked
    with pytest.raises(ValueError, match="separate from raw"):
        prep.run_preprocessing(root, lock, root / "output")
    next(root.rglob("*.face")).write_text("changed fixture")
    with pytest.raises(ValueError, match="lock mismatch"):
        prep.run_preprocessing(root, lock, tmp_path / "output")
    assert not (tmp_path / "output").exists()


def test_status_schema_enforced_and_preserved_by_both_readers(locked, tmp_path, fake_decoder):
    root, lock = locked
    protocol = load_msu_protocol(root)
    out = tmp_path / "out"
    report = prep.run_preprocessing(root, lock, out, mode="process", video_ids=[protocol.recordings[0].video_id])
    path = out / "manifest.json"
    original = json.loads(path.read_text())
    assert "status" not in original
    result = prep.validate_processed_manifest(original, protocol, report["protocol_lock_digest"])
    assert result.processing_status == "complete"
    assert result.fidelity_status == "manual_review_pending"
    assert result.experiment_ready is False
    assert prep.read_processed_manifest(path, root, lock) == result
    with pytest.raises(ValueError, match="not experiment-ready"):
        prep.validate_processed_manifest(original, protocol, report["protocol_lock_digest"], require_experiment_ready=True)
    with pytest.raises(ValueError, match="not experiment-ready"):
        prep.read_processed_manifest(path, root, lock, require_experiment_ready=True)
    for field, value in (("processing_status", "failed"), ("processing_status", "partial"),
                         ("processing_status", "index_validated_only"), ("fidelity_status", "approved"),
                         ("experiment_ready", True), ("status", "complete")):
        changed = copy.deepcopy(original)
        changed[field] = value
        path.write_bytes(prep.json_bytes(changed))
        with pytest.raises(ValueError, match="schema/status/lock"):
            prep.validate_processed_manifest(changed, protocol, report["protocol_lock_digest"])
        with pytest.raises(ValueError, match="schema/status/lock"):
            prep.read_processed_manifest(path, root, lock)
    for field in ("processing_status", "fidelity_status", "experiment_ready"):
        changed = copy.deepcopy(original)
        del changed[field]
        with pytest.raises(ValueError, match="schema/status/lock"):
            prep.validate_processed_manifest(changed, protocol, report["protocol_lock_digest"])


def test_orientation_config_records_assumption_without_fidelity_claim():
    wording = prep.PreprocessingConfig().payload()["orientation"]
    assert "assume annotation coordinates refer to rotated frames" in wording
    assert "alignment and geometry unverified" in wording
    assert "annotations already oriented" not in wording


def test_nonzero_sparse_source_indices_and_resize_are_preserved(msu_root, tmp_path, fake_decoder):
    source = load_msu_protocol(msu_root).recordings[0]
    (msu_root / source.annotation_filepath).write_text("3,1,2,3,4,1,2,3,4\n17,1,2,3,4,1,2,3,4\n")
    lock = tmp_path / "lock.json"
    write_msu_protocol_lock(msu_root, tmp_path / "metadata.json", lock)
    out = tmp_path / "out"
    prep.run_preprocessing(msu_root, lock, out, prep.PreprocessingConfig(output_size=(5, 6)), mode="process", video_ids=[source.video_id])
    rows = prep.read_processed_manifest(out / "manifest.json", msu_root, lock)
    assert [r.frame_index for r in rows] == [3, 17]
    assert len({r.sample_id for r in rows}) == 2
    assert len({r.probe_observation().group_id for r in rows}) == 1
    assert all(r.output_size == (5, 6) for r in rows)


def test_decoder_controls_sequential_indices_and_unavailable_endpoint(monkeypatch):
    class Capture:
        def __init__(self, *args):
            self.position = 0
            self.auto = 1
            self.closed = False
        def isOpened(self):
            return True
        def getBackendName(self):
            return "FFMPEG"
        def set(self, key, value):
            assert key == prep.cv2.CAP_PROP_ORIENTATION_AUTO
            self.auto = value
            return True
        def get(self, key):
            return {prep.cv2.CAP_PROP_ORIENTATION_AUTO: self.auto,
                    prep.cv2.CAP_PROP_ORIENTATION_META: 180,
                    prep.cv2.CAP_PROP_POS_FRAMES: self.position}[key]
        def read(self):
            if self.position >= 2:
                return False, None
            self.position += 1
            return True, np.zeros((8, 8, 3), np.uint8)
        def release(self):
            self.closed = True
    cap = Capture()
    monkeypatch.setattr(prep.cv2, "VideoCapture", lambda *args: cap)
    stream = prep.decode_selected(Path("synthetic"), (1, 2))
    assert next(stream)[0] == 1
    assert cap.auto == 0
    with pytest.raises(prep.FrameDecodeError) as exc:
        next(stream)
    assert exc.value.missing == [2]
    assert cap.closed


def test_boundary_audit_retains_unavailable_annotation_without_correction(locked, monkeypatch):
    root, lock = locked
    source = load_msu_protocol(root).recordings[0]
    class EmptyCapture:
        def isOpened(self):
            return True
        def getBackendName(self):
            return "FFMPEG"
        def set(self, *args):
            return True
        def get(self, *args):
            return 0
        def read(self):
            return False, None
        def release(self):
            pass
    monkeypatch.setattr(prep.cv2, "VideoCapture", lambda *args: EmptyCapture())
    report = prep.audit_decoder_boundaries(root, lock, [source.video_id])
    assert report["records"][0]["annotated_indices_not_decoded"] == [0]
    assert report["records"][0]["sequential_successful_reads"] == 0
    assert report["records"][0]["decoded_index_range"] is None
    test_video = next(r for r in load_msu_protocol(root).recordings if r.pad_partition == "test")
    with pytest.raises(ValueError, match="training videos"):
        prep.audit_decoder_boundaries(root, lock, [test_video.video_id])


def test_real_annotation_index_audit_when_available(tmp_path):
    repository = Path(__file__).parents[1]
    root = repository / "datasets/MSU"
    if not (root / "README.txt").is_file() or not (root / "scene01").is_dir():
        pytest.skip("Licensed MSU release is not installed locally")
    report = prep.run_preprocessing(root, repository / "docs/audit/msu_protocol_lock.json", tmp_path / "real-plan")
    assert report["source_videos_index_validated"] == 280
    assert report["frames_written"] == 0
