"""Ingest human observations against immutable review evidence; never approve fidelity."""

from __future__ import annotations

import argparse
import copy
import json
import re
import subprocess
from collections import Counter
from pathlib import Path

from .msu_preprocessing import digest, json_bytes

GEOMETRY_FIELDS = (
    "orientation_acceptable", "eye_placement_acceptable", "face_box_acceptable",
    "half_open_endpoint_plausible",
)
GEOMETRY_VALUES = ("yes", "uncertain", "no")
TEMPORAL_VALUES = ("decframes_export_i", "native_i", "native_i_minus_1",
                   "native_i_plus_1", "indistinguishable", "none")
TEMPORAL_FIELD = "temporal_correspondence_preference"
EVIDENCE_FILES = {
    "pack": "docs/audit/msu_policy_human_pack.json",
    "template": "docs/audit/msu_policy_review_template.json",
    "lock": "docs/audit/msu_protocol_lock.json",
    "decision": "docs/audit/msu_policy_decision.json",
    "validation": "docs/audit/msu_policy_validation.json",
    "historical": "docs/audit/msu_index_existing_crop_review.json",
}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _strict_json(data):
    def unique_pairs(pairs):
        result = {}
        for key, value in pairs:
            _require(key not in result, f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    def invalid_constant(value):
        raise ValueError(f"Invalid JSON constant: {value}")

    return json.loads(data, object_pairs_hook=unique_pairs, parse_constant=invalid_constant)


def _unique_cases(rows, label):
    _require(isinstance(rows, list), f"{label}: cases must be a list")
    result = {}
    for row in rows:
        _require(isinstance(row, dict), f"{label}: case must be an object")
        case = row.get("case_id")
        _require(isinstance(case, str) and bool(case.strip()), f"{label}: missing case ID")
        _require(case not in result, f"{label}: duplicate case ID {case}")
        result[case] = row
    return result


def validate_human_review(response_bytes, *, pack, template, lock, decision, historical):
    """Pure validation/derivation. Evidence inputs must be trusted by the caller.

    Use ingest_human_review for repository ingestion: it binds these inputs to
    an explicit Git commit and checks the working evidence is unchanged.
    No input dictionaries or source bytes are modified.
    """
    response = _strict_json(response_bytes)
    fields = {"case_id", *GEOMETRY_FIELDS, TEMPORAL_FIELD, "comments"}
    _require(isinstance(response, dict) and set(response) == {
        "schema", "reviewer", "cases", "fidelity_status", "experiment_ready"}, "Invalid response fields")
    _require(response["schema"] == template["schema"] == "msu-human-fidelity-review-v1", "Invalid response schema")
    _require(response["experiment_ready"] is False, "experiment_ready must remain false")
    _require(response["fidelity_status"] == "manual_review_pending", "Response cannot approve fidelity")
    _require(isinstance(response["reviewer"], str) and bool(response["reviewer"].strip()), "Nonempty reviewer required")
    _require(decision.get("policy_state") == "P4" and decision.get("frozen_policy") is None
             and decision.get("experiment_ready") is False, "Existing policy must remain unresolved P4")
    _require(pack.get("schema") == "msu-training-domain-human-pack-v1"
             and pack.get("experiment_ready") is False, "Invalid review pack")
    _require(pack["protocol_lock_digest"] == digest(json_bytes(lock)), "Raw protocol lock digest mismatch")
    _require(pack["historical_review_digest"] == digest(json_bytes(historical)), "Historical crop provenance mismatch")
    _require(pack["review_template"] == template, "Pack/template mismatch")
    _require(template["reviewer"] is None and template["experiment_ready"] is False
             and template["fidelity_status"] == "manual_review_pending", "Invalid original review template")
    expected = _unique_cases(template["cases"], "template")
    pages = _unique_cases(pack["pages"], "pack")
    actual = _unique_cases(response["cases"], "response")
    _require(len(expected) == len(pages) == 64 and set(expected) == set(pages), "Expected exactly 64 committed cases")
    _require(set(actual) == set(expected),
             f"Case coverage mismatch: missing={sorted(set(expected)-set(actual))}; unexpected={sorted(set(actual)-set(expected))}")
    _require(all(set(row) == fields and all(v is None for k, v in row.items() if k != "case_id")
                 for row in expected.values()), "Original template fields must remain unset")
    kinds = Counter(p["kind"] for p in pages.values())
    _require(kinds == {"training_domain_comparison": 56, "original_crop": 8}, "Expected 56 comparisons and 8 original crops")
    historical_rows = {"original-crop:" + r["sample_id"]: r for r in historical["records"]}
    train_clients = set(lock["partitions"]["train"]["client_ids"])
    test_clients = set(lock["partitions"]["test"]["client_ids"])
    _require(train_clients.isdisjoint(test_clients), "Protocol client sets overlap")
    counts = {f: Counter() for f in GEOMETRY_FIELDS}
    temporal = Counter()
    records, negative, uncertain = [], [], []
    for case in sorted(expected):
        page, row = pages[case], actual[case]
        _require(set(row) == fields, f"Unexpected/missing review fields: {case}")
        _require(page["partition"] == "train", f"Official-test/non-training review case: {case}")
        video, index = page["source_video_id"], page["annotation_index"]
        _require(type(index) is int and index >= 0, f"Invalid annotation identity: {case}")
        if page["kind"] == "training_domain_comparison":
            _require(case == f"{video}:annotation:{index}", f"Canonical annotation identity changed: {case}")
            candidates = [c for c in pack["candidates"] if c["case_id"] == case]
            _require(len(candidates) == 4 and {c["candidate_label"] for c in candidates} == set(TEMPORAL_VALUES[:4]),
                     f"Missing comparison provenance: {case}")
            sources = candidates
        else:
            _require(case in historical_rows, f"Missing original-crop provenance: {case}")
            sources = [historical_rows[case]]
        for source in sources:
            original_index = source.get("original_annotation_index", source.get("frame_index"))
            _require(source["source_video_id"] == video and original_index == index
                     and source["pad_partition"] == "train" and source["client_id"] in train_clients,
                     f"Canonical video/client/annotation identity changed: {case}")
        for field in GEOMETRY_FIELDS:
            _require(isinstance(row[field], str) and row[field] in GEOMETRY_VALUES, f"Invalid/unfilled {field}: {case}")
            counts[field][row[field]] += 1
        preference = row[TEMPORAL_FIELD]
        comparison = page["kind"] == "training_domain_comparison"
        _require((not comparison and preference is None) or
                 (isinstance(preference, str) and preference in TEMPORAL_VALUES), f"Invalid/unfilled temporal preference: {case}")
        _require(row["comments"] is None or isinstance(row["comments"], str), f"Invalid comments: {case}")
        if comparison:
            temporal[preference] += 1
        record = {"case_id": case, "kind": page["kind"], "source_video_id": video,
                  "original_annotation_index": index, "partition": "train",
                  "observations": copy.deepcopy(row),
                  "temporal_analysis": {"applicability": "comparison" if comparison else "not_applicable",
                                        "preference": preference if comparison else None}}
        records.append(record)
        for value, destination in (("no", negative), ("uncertain", uncertain)):
            affected = [f for f in GEOMETRY_FIELDS if row[f] == value]
            if affected:
                destination.append({"case_id": case, "source_video_id": video,
                                    "original_annotation_index": index, "fields": affected, "comments": row["comments"]})
    return {"schema": "msu-human-review-validation-v1", "source_response_sha256": digest(response_bytes),
            "reviewer": response["reviewer"], "scope": "64 defined training cases only; not a release-wide fidelity review",
            "case_id_completeness": {"exact_match": True, "unique_cases": len(actual), "missing": [], "unexpected": []},
            "case_kind_counts": dict(sorted(kinds.items())),
            "geometry_counts": {f: {v: counts[f][v] for v in GEOMETRY_VALUES} for f in GEOMETRY_FIELDS},
            "temporal_comparison_counts": {v: temporal[v] for v in TEMPORAL_VALUES},
            "original_crop_temporal_exclusion_count": kinds["original_crop"],
            "negative_geometry_cases": negative, "uncertain_geometry_cases": uncertain, "cases": records,
            "validation_status": "passed", "human_observation_status": "recorded_and_validated",
            "fidelity_status": response["fidelity_status"], "policy_state": "P4", "frozen_policy": None,
            "experiment_ready": False,
            "remaining_scientific_blockers": [
                "Negative and uncertain geometry observations remain unresolved; only the defined review sample was inspected.",
                "Half-open endpoint responses record visual plausibility, not proof of inclusivity/exclusivity.",
                "PittPatt input stream/timebase and annotation correspondence remain unknown; preference counts establish no domain.",
                "Native and export candidates may share pixels; no preference winner or mapping correction is inferred.",
                "All eight codec-error recordings still require documented resolution; none is approved.",
                "Frame-domain policy, coverage and preprocessing fidelity approval remain blocked; training-only model selection is separate."]}


def ingest_human_review(repository, response_path, *, expected_sha256, evidence_ref):
    """Validate exact response bytes and committed evidence, returning an audit only."""
    repository = Path(repository).resolve()
    source = Path(response_path)
    if not source.is_absolute():
        source = repository / source
    raw = source.read_bytes()
    actual_hash = digest(raw)
    _require(actual_hash == expected_sha256, f"Response SHA-256 mismatch: actual {actual_hash}")
    commit = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "--verify", "--end-of-options", f"{evidence_ref}^{{commit}}"], text=True).strip()
    _require(re.fullmatch(r"[0-9a-f]{40,64}", commit) is not None, "Invalid evidence commit")
    evidence, hashes = {}, {}
    for name, path in EVIDENCE_FILES.items():
        committed = subprocess.check_output(["git", "-C", str(repository), "show", f"{commit}:{path}"])
        current = (repository / path).read_bytes()
        # Git checkout may convert LF to CRLF. Response-byte verification above never normalizes.
        _require(current.replace(b"\r\n", b"\n") == committed.replace(b"\r\n", b"\n"), f"Working evidence changed: {path}")
        evidence[name] = _strict_json(committed)
        hashes[name] = {"path": path, "git_blob_bytes_sha256": digest(committed),
                        "canonical_json_sha256": digest(json_bytes(evidence[name]))}
    validation = evidence["validation"]
    _require(validation["validation_status"] == "passed", "Prior provenance validation did not pass")
    for name in ("pack", "template", "decision", "historical"):
        key = Path(EVIDENCE_FILES[name]).stem
        _require(validation["input_digests"][key] == hashes[name]["canonical_json_sha256"], f"Prior provenance link mismatch: {name}")
    _require(validation["protocol_lock_digest"] == hashes["lock"]["canonical_json_sha256"], "Prior raw protocol lock changed")
    audit = validate_human_review(raw, **{k: evidence[k] for k in ("pack", "template", "lock", "decision", "historical")})
    audit["provenance"] = {"evidence_commit": commit, "artifacts": hashes,
                           "digest_conventions": "Source response: exact input bytes. Git blobs: committed bytes. Canonical JSON: sorted indent=2 UTF-8 with final LF.",
                           "working_evidence_matches_commit": True}
    _require(source.read_bytes() == raw, "Source response changed during validation")
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path("."))
    parser.add_argument("--response", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--evidence-ref", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit = ingest_human_review(args.repository, args.response, expected_sha256=args.expected_sha256, evidence_ref=args.evidence_ref)
    with args.output.open("xb") as target:
        target.write(json_bytes(audit))


if __name__ == "__main__":
    main()
