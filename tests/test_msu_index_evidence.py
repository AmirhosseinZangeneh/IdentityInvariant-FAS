"""Index evidence stays distinct from preprocessing policy and fidelity approval."""

import json
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("cv2")

from identity_invariant_fas.data import msu_index_evidence as evidence
from identity_invariant_fas.data import msu_preprocessing as prep
from identity_invariant_fas.data.msu_protocol import load_msu_protocol, write_msu_protocol_lock
from test_msu_protocol import msu_root


def test_candidate_indices_preserve_original_and_unavailable_negative_candidate():
    assert evidence.candidate_indices(0) == {"A:i": 0, "B:i-1": -1, "C:i+1": 1}
    assert evidence.candidate_indices(300) == {"A:i": 300, "B:i-1": 299, "C:i+1": 301}
    assert evidence.candidate_indices(300) == evidence.candidate_indices(300)
    with pytest.raises(ValueError):
        evidence.candidate_indices(-1)


def test_boundary_comparison_keeps_gaps_and_unavailable_annotations():
    indices = [0, 2, 4, 5]
    before = list(indices)
    result = evidence.compare_indices(indices, 4)
    assert indices == before
    assert result["unavailable_annotation_indices"] == [4, 5]
    assert result["decoded_indices_without_annotation"] == [1, 3]
    assert result["internal_annotation_gaps"] == [1, 3]
    assert result["annotation_max"] == 5


@pytest.mark.parametrize("native,external,rc,end,messages,expected", [
    (300, 300, 0, True, [], "consistent_with_clean_eof_in_independent_decode"),
    (127, 300, 0, True, [], "decoder_count_disagreement"),
    (300, 300, 1, False, [], "external_decode_failed_unclassified"),
    (300, 300, 0, True, ["decode error"], "diagnostics_unresolved"),
    (300, 300, 0, True, ["[null] Application provided invalid, non monotonically increasing dts"], "completed_with_mux_timestamp_diagnostics"),
    (127, None, 1, False, ["[prores] error decoding picture header"], "codec_failure_evidence"),
    (300, None, 0, False, [], "unresolved_termination"),
])
def test_failed_read_alone_never_means_eof(native, external, rc, end, messages, expected):
    assert evidence.assess_stop(native, external, rc, end, messages) == expected


def test_native_codec_error_overrides_otherwise_completed_timestamp_warning():
    assert evidence.assess_stop(205, 205, 0, True,
        ["[null] non monotonically increasing dts"], ["[prores] ac tex damaged"]) == "codec_failure_evidence"


@pytest.fixture
def locked(msu_root, tmp_path):
    lock = tmp_path / "lock.json"
    write_msu_protocol_lock(msu_root, tmp_path / "metadata.json", lock)
    return msu_root, lock


@pytest.fixture
def fake_scan(monkeypatch):
    def scan(path, retain=()):
        return {"successful_reads": 1, "decoded_range": [0, 0], "reported_frame_count": 1,
                "rotation_degrees": 0, "reported_fps": 30,
                "stop_reason": "first_unsuccessful_read; not proof of EOF"}, {
                    0: np.zeros((8, 8, 3), np.uint8)} if 0 in retain else {}
    monkeypatch.setattr(evidence, "scan_video", scan)


def test_full_vs_partial_scope_and_provenance(locked, fake_scan):
    root, lock = locked
    full = evidence.boundary_audit(root, lock)
    assert full["scope"] == "full_locked_release"
    assert full["videos_attempted"] == full["videos_decoded"] == 280
    assert full["summaries"]["pad_partition"]["train"]["videos"] == 120
    assert full["summaries"]["pad_partition"]["test"]["videos"] == 160
    assert not full["experiment_ready"]
    source = load_msu_protocol(root).recordings[0]
    partial = evidence.boundary_audit(root, lock, [source.video_id])
    assert partial["scope"] == "explicit_partial_scope"
    row = partial["records"][0]
    assert row["source_video_id"] == source.video_id
    assert row["client_id"] == source.client_id
    assert row["attack_type"] == source.attack_type
    assert row["session_id"] is None
    assert partial == evidence.boundary_audit(root, lock, [source.video_id])


def test_failed_decode_not_counted_as_successful_or_clean_eof(locked, monkeypatch):
    root, lock = locked
    source = load_msu_protocol(root).recordings[0]
    def fail(*args):
        raise ValueError("Synthetic decoder unavailable")
    monkeypatch.setattr(evidence, "scan_video", fail)
    report = evidence.boundary_audit(root, lock, [source.video_id])
    assert report["videos_decoded"] == 0
    assert report["records"][0]["audit_status"] == "failed"
    assert "unavailable_annotation_indices" not in report["records"][0]


def test_review_roundtrip_no_remapping_and_relocation(locked, tmp_path, fake_scan):
    root, lock = locked
    source = load_msu_protocol(root).recordings[0]
    cases = {source.video_id: [0]}
    first = evidence.write_review(root, lock, cases, tmp_path / "review1")
    assert [r["annotation_index"] for r in first["records"]] == [0, 0, 0]
    assert [r["candidate_decoder_index"] for r in first["records"]] == [0, -1, 1]
    assert [r["available"] for r in first["records"]] == [True, False, False]
    assert first["fidelity_status"] == "manual_review_pending"
    assert first["experiment_ready"] is False
    assert json.loads((tmp_path / "review1/review.json").read_text()) == first
    assert not (tmp_path / "review1/manifest.json").exists()
    moved = tmp_path / "moved"
    root.rename(moved)  # synthetic only
    second = evidence.write_review(moved, lock, cases, tmp_path / "review2")
    assert first == second
    with pytest.raises(FileExistsError):
        evidence.write_review(moved, lock, cases, tmp_path / "review2")
    with pytest.raises(ValueError, match="outside raw"):
        evidence.write_review(moved, lock, cases, moved / "review")


def test_evidence_does_not_make_missing_frame_available_or_grant_readiness(locked, tmp_path, monkeypatch):
    root, lock = locked
    source = load_msu_protocol(root).recordings[0]
    # An alternative candidate may exist, but the production decoder request is unchanged.
    assert evidence.candidate_indices(1)["B:i-1"] == 0
    def missing(path, indices, **kwargs):
        raise prep.FrameDecodeError(list(indices))
        yield  # generator-shaped decoder
    monkeypatch.setattr(prep, "decode_selected", missing)
    with pytest.raises(prep.FrameDecodeError):
        prep.run_preprocessing(root, lock, tmp_path / "failed", mode="process", video_ids=[source.video_id])
    report = json.loads((tmp_path / "failed/audit.json").read_text())
    assert report["processing_status"] == "failed"
    assert report["experiment_ready"] is False
    assert not (tmp_path / "failed/manifest.json").exists()


def test_stop_enrichment_preserves_prior_observations_and_validates_provenance(locked, fake_scan, monkeypatch):
    root, lock = locked
    source = load_msu_protocol(root).recordings[0]
    prior = evidence.boundary_audit(root, lock, [source.video_id])
    binary = root / "ffmpeg/bin/ffmpeg.exe"
    binary.parent.mkdir(parents=True)
    binary.write_bytes(b"synthetic placeholder, never executed")
    calls = []
    def run(args, **kwargs):
        calls.append(args)
        return SimpleNamespace(returncode=0, stdout=b"frame=1\nprogress=end\n", stderr=b"")
    monkeypatch.setattr(evidence.subprocess, "run", run)
    result = evidence.enrich_stop_evidence(root, lock, prior)
    assert prior["records"][0].get("stop_assessment") is None
    for key, value in prior["records"][0].items():
        assert result["records"][0][key] == value
    assert result["records"][0]["stop_assessment"] == "consistent_with_clean_eof_in_independent_decode"
    assert "-xerror" in calls[0] and "-vsync" in calls[0]
    assert result["experiment_ready"] is False
    prior["records"][0]["client_id"] = "999"
    with pytest.raises(ValueError, match="metadata conflicts"):
        evidence.enrich_stop_evidence(root, lock, prior)
