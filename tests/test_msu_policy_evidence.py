"""Synthetic guards for frame domains, training evidence and unapproved human review."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("cv2")

from identity_invariant_fas.data import msu_policy_evidence as policy
from identity_invariant_fas.data.msu_protocol import load_msu_protocol, write_msu_protocol_lock
from identity_invariant_fas.data.msu_preprocessing import json_bytes, digest
from test_msu_protocol import msu_root


def measured_calibration(source_id, lock_digest, config):
    run = {"export_count": 3, "dup_frames": 1, "drop_frames": 1,
           "returncode": 0, "progress_end": True, "diagnostics": []}
    return {"passed": True, "training_only": True, "protocol_lock_digest": lock_digest,
            "config": config, "config_digest": digest(json_bytes(config)),
            "records": [{"source_video_id": source_id, "pad_partition": "train",
                         "actual_image2": copy.deepcopy(run), "disk_free_image2": copy.deepcopy(run),
                         "actual_file_count": 3, "numbering_contiguous_from_one": True,
                         "equivalent_count_dup_drop": True}]}


@pytest.mark.parametrize("fault", ["unknown_count", "mismatched_drop", "missing_end",
                                  "nonzero_exit", "missing_file", "test_source"])
def test_calibration_pass_flag_cannot_hide_incomplete_measurements(fault):
    report = measured_calibration("source", "lock", {})
    policy.validate_export_calibration(report)
    row = report["records"][0]
    if fault == "unknown_count":
        row["actual_image2"]["export_count"] = row["disk_free_image2"]["export_count"] = None
    elif fault == "mismatched_drop":
        row["disk_free_image2"]["drop_frames"] = 0
    elif fault == "missing_end":
        row["actual_image2"]["progress_end"] = row["disk_free_image2"]["progress_end"] = False
    elif fault == "nonzero_exit":
        row["actual_image2"]["returncode"] = row["disk_free_image2"]["returncode"] = 1
    elif fault == "missing_file":
        row["actual_file_count"] = 2
    else:
        row["pad_partition"] = "test"
    with pytest.raises(ValueError, match="calibration"):
        policy.validate_export_calibration(report)


def test_domains_export_ids_preserve_annotation_and_are_deterministic():
    assert policy.DOMAINS == ("annotation", "native_decoder", "decframes_export")
    original = 300
    first = policy.export_id("MSU-MFSD:scene01/real/example.mp4", original, "config-a")
    assert first == policy.export_id("MSU-MFSD:scene01/real/example.mp4", original, "config-a")
    assert first != policy.export_id("MSU-MFSD:scene01/real/example.mp4", original, "config-b")
    assert ":domain:decframes_export:ordinal:300:" in first
    assert original == 300
    with pytest.raises(ValueError):
        policy.export_id("source", -1, "config")


def test_export_progress_missing_counts_do_not_imply_availability():
    result = policy.parse_export_run(b"", b"Error while decoding", 1)
    assert result["export_count"] is None
    assert result["progress_end"] is False
    assert result["diagnostics"] == ["Error while decoding"]
    valid = policy.parse_export_run(b"frame=100\nframe=301\ndup_frames=1\ndrop_frames=0\nprogress=end\n", b"", 0)
    assert (valid["export_count"], valid["dup_frames"], valid["drop_frames"]) == (301, 1, 0)


@pytest.mark.parametrize("native,exported,expected", [
    ({"successful_reads":127,"reported_frame_count":300,"opencv_subprocess_recheck":{"diagnostics":["codec"]}},
     {"returncode":0,"diagnostics":["[prores] error decoding"]}, "both_paths_problematic"),
    ({"successful_reads":301,"reported_frame_count":301},
     {"returncode":0,"diagnostics":["[prores] damaged"]}, "bundled_decoder_error"),
    ({"successful_reads":127,"reported_frame_count":300},
     {"returncode":0,"diagnostics":[]}, "native_decoder_shortfall_but_bundled_export_succeeds"),
    ({"successful_reads":301,"reported_frame_count":301},
     {"returncode":0,"diagnostics":[]}, "unresolved"),
])
def test_codec_paths_are_classified_without_claiming_pixel_identity(native, exported, expected):
    assert policy.codec_classification(native, exported) == expected


def test_review_defaults_cannot_grant_readiness():
    template = policy.review_template(["original:0", "candidate:0"])
    assert template["reviewer"] is None
    assert template["experiment_ready"] is False
    assert template["fidelity_status"] == "manual_review_pending"
    assert all(v is None for row in template["cases"] for k, v in row.items() if k != "case_id")
    assert policy.readiness_requirements()["experiment_ready"] is False
    assert not policy.readiness_requirements()["frame_domain_frozen"]
    assert not policy.readiness_requirements()["human_geometry_review_completed"]


@pytest.fixture
def locked(msu_root, tmp_path):
    lock = tmp_path / "lock.json"
    write_msu_protocol_lock(msu_root, tmp_path / "metadata.json", lock)
    return msu_root, lock


def test_training_only_selection_and_no_test_influence(locked):
    root, _ = locked
    protocol = load_msu_protocol(root)
    train = next(r for r in protocol.recordings if r.pad_partition == "train")
    test = next(r for r in protocol.recordings if r.pad_partition == "test")
    audit = {"records":[{"source_video_id":train.video_id}],"partition":"train","policy_selection_eligible":True}
    first = policy.unresolved_policy(protocol, audit)
    assert first["policy_state"] == "P4"
    assert first["frozen_policy"] is None  # no speculative frozen policy to migrate
    assert not first["experiment_ready"]
    assert first == policy.unresolved_policy(protocol, audit)
    with pytest.raises(ValueError, match="training-only"):
        policy.require_training(protocol, [train.video_id, test.video_id])
    bad = copy.deepcopy(audit)
    bad["records"][0]["source_video_id"] = test.video_id
    with pytest.raises(ValueError, match="training-only"):
        policy.unresolved_policy(protocol, bad)
    bad = copy.deepcopy(audit)
    bad["partition"] = "test"
    with pytest.raises(ValueError, match="Test evidence"):
        policy.unresolved_policy(protocol, bad)
    assert first == policy.unresolved_policy(protocol, audit)


def test_human_pack_never_substitutes_missing_candidates(locked, tmp_path, monkeypatch):
    root, lock = locked
    protocol = load_msu_protocol(root)
    source = next(r for r in protocol.recordings if r.pad_partition == "train")
    monkeypatch.setattr(policy, "scan_video", lambda *args: ({"rotation_degrees":0}, {0:np.zeros((8,8,3),np.uint8)}))
    calibration = measured_calibration(source.video_id, digest(json_bytes(json.loads(lock.read_text()))), {})
    calibration["records"][0]["retained_exports"] = {
        "0": {"available": False, "original_annotation_index": 0, "export_ordinal": 0,
              "export_file_number": 1, "filepath": None, "sha256": None}}
    out = tmp_path / "review"
    before = lock.read_bytes()
    report = policy.write_human_pack(root, lock, {source.video_id:[0]}, calibration, tmp_path,
                                     {"records":[]}, tmp_path, out)
    assert lock.read_bytes() == before
    assert [c["candidate_frame_index"] for c in report["candidates"]] == [0,-1,1,0]
    assert [c["available"] for c in report["candidates"]] == [True,False,False,False]
    assert all(c["original_annotation_index"] == 0 for c in report["candidates"])
    assert report["candidates"][-1]["candidate_frame_domain"] == "decframes_export"
    assert report["candidates"][-1]["overlay_path"] is None
    assert not report["experiment_ready"]
    assert json.loads((out / "pack.json").read_text()) == report
    page = (out / "review.html").read_text()
    assert "decframes_export_i" in page and "indistinguishable" in page
    assert "Download review responses" in page
    assert 'native_i: full resolution overlay' in page
    assert 'native_i_minus_1: UNAVAILABLE (no substitution)' in page
    assert all(row["comments"] is None for row in report["review_template"]["cases"])


def test_counter_requires_matching_calibration_and_preserves_metadata(locked, monkeypatch):
    root, lock = locked
    protocol = load_msu_protocol(root)
    config = {"synthetic":"exact"}
    monkeypatch.setattr(policy, "policy_config", lambda r: config)
    monkeypatch.setattr(policy, "_rate", lambda *args: ("30/1","30.00"))
    monkeypatch.setattr(policy, "_export", lambda *args: {"export_count":1,"dup_frames":0,"drop_frames":0,
                         "returncode":0,"progress_end":True,"diagnostics":[]})
    source = next(r for r in protocol.recordings if r.pad_partition == "train")
    calibration = measured_calibration(source.video_id, digest(json_bytes(json.loads(lock.read_text()))), config)
    report = policy.count_cohort(root,lock,calibration,"train")
    assert report["count"] == 120 and report["total_export_frames"] == 120
    assert report["unavailable_annotations"] == 0
    assert all(r["first_export_number"] == r["last_export_number"] == 1 for r in report["records"])
    assert report == policy.count_cohort(root,lock,calibration,"train")
    test_report = policy.count_cohort(root,lock,calibration,"test")
    assert test_report["policy_selection_eligible"] is False
    calibration["passed"] = False
    with pytest.raises(ValueError, match="calibration"):
        policy.count_cohort(root,lock,calibration,"train")


def test_reviewed_artifact_links_scope_and_training_only_policy():
    """Reuse measured evidence without requiring licensed data or rerunning exports."""
    audit = Path(__file__).resolve().parents[1] / "docs" / "audit"
    read = lambda name: json.loads((audit / (name + ".json")).read_text())
    calibration = read("msu_policy_export_calibration")
    supplement = read("msu_policy_drop_calibration")
    train, test = read("msu_policy_train_export"), read("msu_policy_test_export")
    release, decision = read("msu_policy_export_release"), read("msu_policy_decision")
    native = {r["source_video_id"]: r for r in read("msu_index_boundaries")["records"]}
    for cal in (calibration, supplement):
        policy.validate_export_calibration(cal)
        assert cal["config_digest"] == train["config_digest"]
        assert cal["protocol_lock_digest"] == train["protocol_lock_digest"]
        assert all(native[r["source_video_id"]]["pad_partition"] == "train" for r in cal["records"])
    assert any(r["actual_image2"]["drop_frames"] > 0 for r in supplement["records"])
    all_rows = train["records"] + test["records"]
    assert len(all_rows) == len({r["source_video_id"] for r in all_rows}) == 280
    assert {r["source_video_id"] for r in all_rows} == set(native)
    for cohort, count in ((train, 120), (test, 160)):
        assert len(cohort["records"]) == cohort["count"] == count
        assert cohort["calibration_digest"] == digest(json_bytes(calibration))
        assert cohort["config_digest"] == digest(json_bytes(cohort["config"]))
        assert cohort["protocol_lock_digest"] == digest(json_bytes(read("msu_protocol_lock")))
        assert cohort["total_export_frames"] == sum(r["export_count"] for r in cohort["records"])
        assert all(r["pad_partition"] == cohort["partition"] for r in cohort["records"])
    assert {r["client_id"] for r in train["records"]}.isdisjoint({r["client_id"] for r in test["records"]})
    for row in all_rows:
        observed = native[row["source_video_id"]]
        for key in ("filepath", "client_id", "pad_partition", "capture_device", "class_label", "attack_type",
                    "annotation_filepath", "annotation_sha256", "session_id"):
            assert row[key] == observed[key]
        assert row["annotation_count"] == observed["annotation_count"]
        assert row["export_count"] == observed["annotation_max"] + 1
        assert row["unavailable_annotation_indices"] == []
    assert release["training_audit_digest"] == decision["selection_evidence_digest"] == digest(json_bytes(train))
    assert release["test_audit_digest"] == digest(json_bytes(test))
    assert test["preexisting_policy_state_digest"] == digest(json_bytes(decision))
    assert release["policy_state_before_test"] == decision
    assert train["policy_selection_eligible"] and not test["policy_selection_eligible"]
    assert decision["policy_state"] == "P4" and decision["frozen_policy"] is None
    assert not decision["test_evidence_used_for_selection"]
    assert not any(d["experiment_ready"] for d in (train, test, release, decision))
    for key in ("dup_frames", "drop_frames"):
        assert release[key] == sum(r[key] for r in all_rows)
    assert len(release["codec_records"]) == 8
    assert {r["source_video_id"] for r in release["codec_records"]} == {
        r["source_video_id"] for r in all_rows if r["diagnostics"]}
    assert all(not r["process_success_certified"] for r in release["codec_records"])


def test_stored_review_keeps_missing_candidates_and_all_human_fields_unset():
    audit = Path(__file__).resolve().parents[1] / "docs" / "audit"
    pack = json.loads((audit / "msu_policy_human_pack.json").read_text())
    template = json.loads((audit / "msu_policy_review_template.json").read_text())
    cases = json.loads((audit / "msu_policy_training_cases.json").read_text())
    assert template == pack["review_template"]
    assert template == policy.review_template([p["case_id"] for p in pack["pages"]])
    assert len(template["cases"]) == 64 and len(pack["candidates"]) == 224
    offsets = {"native_i": 0, "native_i_minus_1": -1, "native_i_plus_1": 1, "decframes_export_i": 0}
    for video, indices in cases.items():
        for index in indices:
            candidates = [c for c in pack["candidates"] if c["source_video_id"] == video
                          and c["original_annotation_index"] == index]
            assert {c["candidate_label"] for c in candidates} == set(offsets)
            for c in candidates:
                assert c["pad_partition"] == "train" and c["session_id"] is None
                assert c["candidate_frame_index"] == index + offsets[c["candidate_label"]]
                assert c["available"] == (c["overlay_path"] is not None)
                if not c["available"]:
                    assert c["overlay_sha256"] is None
    assert sum(not c["available"] for c in pack["candidates"]) == 40
    assert pack["fidelity_status"] == "manual_review_pending" and not pack["experiment_ready"]
