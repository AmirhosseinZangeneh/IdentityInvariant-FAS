"""Human-observation validation uses metadata and synthetic responses, never media."""

import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest

from identity_invariant_fas.data import msu_human_review as review

AUDIT = Path(__file__).resolve().parents[1] / "docs" / "audit"


def test_review_validation_without_opencv():
    code = """
import sys
sys.modules['cv2'] = None
import json
from pathlib import Path
from identity_invariant_fas.data import msu_human_review as review
root = Path(sys.argv[1])
evidence = {k: json.loads((root / Path(v).name).read_bytes())
            for k, v in review.EVIDENCE_FILES.items()}
actual = review.validate_human_review(
    (root / 'msu_human_review_responses.json').read_bytes(),
    **{k: evidence[k] for k in ('pack', 'template', 'lock', 'decision', 'historical')})
expected = json.loads((root / 'msu_human_review_validation.json').read_bytes())
expected.pop('provenance')
assert actual == expected
assert 'identity_invariant_fas.data.msu_preprocessing' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", code, str(AUDIT)], check=True)


@pytest.fixture
def evidence():
    return {k: json.loads((AUDIT / Path(v).name).read_bytes()) for k, v in review.EVIDENCE_FILES.items()}


@pytest.fixture
def response(evidence):
    result = copy.deepcopy(evidence["template"])
    result["reviewer"] = "Synthetic reviewer"
    for case in result["cases"]:
        for field in review.GEOMETRY_FIELDS:
            case[field] = "yes"
        case[review.TEMPORAL_FIELD] = "native_i"
    result["cases"][0]["face_box_acceptable"] = "no"
    result["cases"][0]["comments"] = "Preserve negative observation exactly."
    result["cases"][1]["orientation_acceptable"] = "uncertain"
    result["cases"][1]["comments"] = "Preserve uncertainty."
    return result


def validate(response, evidence):
    return review.validate_human_review(review.json_bytes(response), **{
        k: evidence[k] for k in ("pack", "template", "lock", "decision", "historical")})


def test_complete_review_preserves_observations_without_approval(response, evidence):
    before_response, before_evidence = copy.deepcopy(response), copy.deepcopy(evidence)
    audit = validate(response, evidence)
    assert audit["case_id_completeness"] == {"exact_match": True, "unique_cases": 64, "missing": [], "unexpected": []}
    assert audit["case_kind_counts"] == {"training_domain_comparison": 56, "original_crop": 8}
    assert audit["temporal_comparison_counts"]["native_i"] == 56
    assert audit["original_crop_temporal_exclusion_count"] == 8
    originals = [c for c in audit["cases"] if c["kind"] == "original_crop"]
    assert all(c["temporal_analysis"] == {"applicability": "not_applicable", "preference": None} for c in originals)
    assert all(c["observations"][review.TEMPORAL_FIELD] == "native_i" for c in originals)
    assert audit["negative_geometry_cases"][0]["comments"] == "Preserve negative observation exactly."
    assert audit["uncertain_geometry_cases"][0]["comments"] == "Preserve uncertainty."
    assert {c["case_id"]: c["observations"] for c in audit["cases"]} == {c["case_id"]: c for c in response["cases"]}
    assert review.json_bytes(audit) == review.json_bytes(validate(response, evidence))
    assert response == before_response and evidence == before_evidence
    assert audit["policy_state"] == "P4" and audit["frozen_policy"] is None
    assert audit["fidelity_status"] == "manual_review_pending" and audit["experiment_ready"] is False


@pytest.mark.parametrize("fault,match", [
    ("duplicate", "duplicate case"), ("missing", "coverage mismatch"), ("unexpected", "coverage mismatch"),
    ("invalid_value", "Invalid/unfilled"), ("unfilled", "Invalid/unfilled"),
    ("extra_field", "Unexpected/missing"), ("missing_field", "Unexpected/missing"),
    ("schema", "schema"), ("reviewer", "reviewer"), ("approved", "approve fidelity"),
    ("ready", "remain false"), ("temporal", "temporal preference"), ("comments", "comments"),
])
def test_invalid_responses_fail_without_repair(response, evidence, fault, match):
    if fault == "duplicate":
        response["cases"][1] = copy.deepcopy(response["cases"][0])
    elif fault == "missing":
        response["cases"].pop()
    elif fault == "unexpected":
        response["cases"][0]["case_id"] = "unexpected"
    elif fault in ("invalid_value", "unfilled"):
        response["cases"][0]["face_box_acceptable"] = "approved" if fault == "invalid_value" else None
    elif fault == "extra_field":
        response["cases"][0]["source_video_id"] = "replacement"
    elif fault == "missing_field":
        del response["cases"][0]["eye_placement_acceptable"]
    elif fault == "schema":
        response["schema"] = "unknown"
    elif fault == "reviewer":
        response["reviewer"] = " \t"
    elif fault == "approved":
        response["fidelity_status"] = "approved"
    elif fault == "ready":
        response["experiment_ready"] = True
    elif fault == "temporal":
        response["cases"][0][review.TEMPORAL_FIELD] = None
    else:
        response["cases"][0]["comments"] = 1
    before = copy.deepcopy(response)
    with pytest.raises(ValueError, match=match):
        validate(response, evidence)
    assert response == before


@pytest.mark.parametrize("fault", ["test", "annotation", "video", "cohort", "kind", "lock", "policy"])
def test_provenance_conflicts_fail_closed(response, evidence, fault):
    if fault == "test":
        evidence["pack"]["pages"][0]["partition"] = "test"
    elif fault == "annotation":
        evidence["pack"]["pages"][0]["annotation_index"] += 1
    elif fault == "video":
        evidence["pack"]["candidates"][0]["source_video_id"] = "changed"
    elif fault == "cohort":
        evidence["pack"]["candidates"][0]["client_id"] = "001"
    elif fault == "kind":
        evidence["pack"]["pages"][0]["kind"] = "original_crop"
    elif fault == "lock":
        evidence["lock"]["recording_count"] = 279
    else:
        evidence["decision"]["policy_state"] = "P2"
    with pytest.raises(ValueError):
        validate(response, evidence)


def test_original_crop_temporal_values_do_not_enter_comparison_statistics(response, evidence):
    before = validate(response, evidence)
    ids = {p["case_id"] for p in evidence["pack"]["pages"] if p["kind"] == "original_crop"}
    for row in response["cases"]:
        if row["case_id"] in ids:
            row[review.TEMPORAL_FIELD] = "decframes_export_i"
    after = validate(response, evidence)
    assert after["temporal_comparison_counts"] == before["temporal_comparison_counts"]
    assert all(c["observations"][review.TEMPORAL_FIELD] == "decframes_export_i"
               for c in after["cases"] if c["kind"] == "original_crop")


def test_duplicate_json_keys_are_rejected(evidence):
    with pytest.raises(ValueError, match="Duplicate JSON key"):
        review.validate_human_review(b'{"schema":"one","schema":"two"}', **{
            k: evidence[k] for k in ("pack", "template", "lock", "decision", "historical")})


def test_ingestion_hash_commit_binding_and_source_immutability(tmp_path, monkeypatch, response):
    blobs = {}
    for path in review.EVIDENCE_FILES.values():
        blobs[path] = (AUDIT / Path(path).name).read_bytes()
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(blobs[path])

    def git_output(command, **kwargs):
        if "rev-parse" in command:
            return "a" * 40 + "\n"
        return blobs[command[-1].split(":", 1)[1]]

    monkeypatch.setattr(review.subprocess, "check_output", git_output)
    source = tmp_path / "response.json"
    raw = review.json_bytes(response)
    source.write_bytes(raw)
    kwargs = {"expected_sha256": review.digest(raw), "evidence_ref": "fixture"}
    first = review.ingest_human_review(tmp_path, source, **kwargs)
    assert review.json_bytes(first) == review.json_bytes(review.ingest_human_review(tmp_path, source, **kwargs))
    assert source.read_bytes() == raw
    assert first["source_response_sha256"] == review.digest(raw)
    assert first["provenance"]["working_evidence_matches_commit"]
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        review.ingest_human_review(tmp_path, source, expected_sha256="0" * 64, evidence_ref="fixture")
    lock = tmp_path / review.EVIDENCE_FILES["lock"]
    before = lock.read_bytes()
    lock.write_bytes(before + b" ")
    with pytest.raises(ValueError, match="Working evidence changed"):
        review.ingest_human_review(tmp_path, source, **kwargs)
    assert source.read_bytes() == raw


def test_recorded_human_counts_and_negative_case(evidence):
    # Explicit observations are expected results, never substitutes for validation.
    raw = (AUDIT / "msu_human_review_responses.json").read_bytes()
    audit = review.validate_human_review(raw, **{
        k: evidence[k] for k in ("pack", "template", "lock", "decision", "historical")})
    recorded = json.loads((AUDIT / "msu_human_review_validation.json").read_bytes())
    provenance = recorded.pop("provenance")
    assert audit == recorded
    for name, artifact in provenance["artifacts"].items():
        assert review.digest(review.json_bytes(evidence[name])) == artifact["canonical_json_sha256"]
    assert audit["geometry_counts"] == {
        "orientation_acceptable": {"yes": 57, "uncertain": 7, "no": 0},
        "eye_placement_acceptable": {"yes": 64, "uncertain": 0, "no": 0},
        "face_box_acceptable": {"yes": 50, "uncertain": 13, "no": 1},
        "half_open_endpoint_plausible": {"yes": 57, "uncertain": 7, "no": 0},
    }
    assert audit["temporal_comparison_counts"] == dict(zip(review.TEMPORAL_VALUES, (30, 6, 6, 7, 7, 0)))
    assert len(audit["negative_geometry_cases"]) == 1 and len(audit["uncertain_geometry_cases"]) == 14
    negative = audit["negative_geometry_cases"][0]
    assert negative["source_video_id"] == "MSU-MFSD:scene01/real/real_client008_android_SD_scene01.mp4"
    assert negative["original_annotation_index"] == 298 and negative["fields"] == ["face_box_acceptable"]
