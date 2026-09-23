"""Training-only policy evidence and unapproved MSU human review; no training adapter."""

from __future__ import annotations

import argparse
import html
import json
import os
import re
import subprocess
import tempfile
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np

from .msu_protocol import load_msu_protocol, verify_msu_protocol_lock
from .msu_preprocessing import digest, json_bytes, parse_face_annotations, rotate_frame
from .msu_index_evidence import scan_video

EXPORT_POLICY = "bundled-decframes-rounded-rate-image2-v1"
DOMAINS = ("annotation", "native_decoder", "decframes_export")


def export_id(video_id, ordinal, policy_digest):
    if type(ordinal) is not int or ordinal < 0:
        raise ValueError("Export ordinal must be a nonnegative integer")
    return f"{video_id}:domain:decframes_export:ordinal:{ordinal}:policy:{policy_digest}"


def policy_config(root):
    root = Path(root)
    return {"version": EXPORT_POLICY, "binary_sha256": digest((root / "ffmpeg/bin/ffmpeg.exe").read_bytes()),
            "ffprobe_sha256": digest((root / "ffmpeg/bin/ffprobe.exe").read_bytes()),
            "rate_rule": "avg_frame_rate numerator/denominator rounded to two decimals, as DecFrames.m num2str(%.02f)",
            "export": "original image2 BMP with default vsync and first filename number 1; no added -xerror",
            "count_transport": "same image2/BMP; -updatefirst 1 to Windows NUL with -y; equivalence requires training calibration",
            "rotation": "original export pixels unchanged; review only assumes metadata 0/180 should be applied",
            "correspondence": "annotation i -> filename i+1 is a candidate sidecar association, not proven annotation generation"}


def _context(root, lock):
    verify_msu_protocol_lock(root, lock)
    return load_msu_protocol(root), digest(json_bytes(json.loads(Path(lock).read_text())))


def _new_output(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    if output == root or root in output.parents or output in root.parents:
        raise ValueError("Output must be separate from raw data")
    output.mkdir(parents=True, exist_ok=False)
    return output


def require_training(protocol, ids):
    sources = {r.video_id: r for r in protocol.recordings}
    if not ids or len(ids) != len(set(ids)) or any(v not in sources or sources[v].pad_partition != "train" for v in ids):
        raise ValueError("Policy selection and review require explicit training-only sources")
    return [sources[v] for v in sorted(ids)]


def _rate(root, source):
    result = subprocess.run([str(root / "ffmpeg/bin/ffprobe.exe"), "-v", "quiet", "-show_streams",
                             str(root / source.filepath)], capture_output=True, check=True)
    match = re.search(r"^avg_frame_rate=(\d+)/(\d+)\r?$", result.stdout.decode("utf-8", errors="strict"), re.M)
    if not match or int(match[2]) == 0:
        raise ValueError("Missing source-script average frame rate")
    return f"{match[1]}/{match[2]}", f"{int(match[1]) / int(match[2]):.2f}"


def parse_export_run(stdout, stderr, returncode):
    values = {}
    for line in stdout.decode("utf-8", errors="replace").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key] = value
    diagnostics = []
    for line in stderr.decode("utf-8", errors="replace").splitlines():
        if re.search(r"error|damaged|invalid|failed|corrupt|non monoton", line, re.I):
            diagnostics.append(re.sub(r" @ [0-9A-Fa-f]+", "", line.strip()))
    return {"returncode": returncode, "progress_end": values.get("progress") == "end",
            "export_count": int(values["frame"]) if "frame" in values else None,
            "dup_frames": int(values["dup_frames"]) if "dup_frames" in values else None,
            "drop_frames": int(values["drop_frames"]) if "drop_frames" in values else None,
            "diagnostics": diagnostics}


def _export(root, source, rate, directory=None):
    if os.name != "nt":
        raise RuntimeError("Exact bundled Windows binary/NUL implementation only; no cross-platform equivalence claimed")
    command = [str(root / "ffmpeg/bin/ffmpeg.exe"), "-i", str(root / source.filepath), "-r", rate]
    if directory is None:
        command += ["-y", "-vcodec", "bmp", "-f", "image2", "-updatefirst", "1", "NUL"]
    else:
        command += [str(directory / (Path(source.filepath).stem + "_%03d.bmp"))]
    command += ["-progress", "pipe:1"]
    run = subprocess.run(command, capture_output=True)
    result = parse_export_run(run.stdout, run.stderr, run.returncode)
    result["diagnostics"] = [s.replace(str(root), "<raw-root>") for s in result["diagnostics"]]
    return result


def calibrate_counter(root, lock, cases, output):
    """Actual numbered BMP exports versus disk-free image2 on training cases only.

    Fresh temporary BMP directories are checked inside the output workspace before
    cleanup; selected original BMPs are retained for the four-domain review.
    """
    root = Path(root).resolve()
    protocol, lock_digest = _context(root, lock)
    sources = require_training(protocol, list(cases))
    output = _new_output(root, output)
    config = policy_config(root)
    rows = []
    for source in sources:
        annotations, _ = parse_face_annotations((root / source.annotation_filepath).read_bytes())
        requested = cases[source.video_id]
        if not requested or len(requested) != len(set(requested)) or any(i not in annotations for i in requested):
            raise ValueError("Calibration/review requires existing original annotation indices")
        fraction, rate = _rate(root, source)
        sink = _export(root, source, rate)
        with tempfile.TemporaryDirectory(prefix="calibration-", dir=output) as temporary:
            folder = Path(temporary).resolve()
            if not folder.is_relative_to(output):
                raise ValueError("Unsafe temporary export directory")
            actual = _export(root, source, rate, folder)
            images = sorted(folder.glob("*.bmp"))
            numbers = [int(p.stem.rsplit("_", 1)[1]) for p in images]
            sequence_valid = numbers == list(range(1, len(images) + 1))
            comparisons = {k: sink[k] == actual[k] for k in ("export_count", "dup_frames", "drop_frames", "returncode", "progress_end")}
            equivalent = (sequence_valid and bool(images) and len(images) == actual["export_count"]
                          and all(comparisons.values()) and actual["progress_end"]
                          and actual["returncode"] == 0
                          and all(type(actual[k]) is int and actual[k] >= 0
                                  for k in ("export_count", "dup_frames", "drop_frames")))
            retained = {}
            for index in sorted(requested):
                name = Path(source.filepath).stem + f"_{index+1:03d}.bmp"
                image_path = folder / name
                relative = f"retained/{digest(source.video_id.encode())}/{name}"
                item = {"original_annotation_index": index, "export_ordinal": index, "export_file_number": index + 1,
                        "available": image_path.is_file(), "filepath": None, "sha256": None}
                if image_path.is_file():
                    target = output / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    data = image_path.read_bytes()
                    with target.open("xb") as file:
                        file.write(data)
                    item.update(filepath=relative, sha256=digest(data))
                retained[str(index)] = item
        rows.append({"source_video_id": source.video_id, "pad_partition": source.pad_partition,
                     "client_id": source.client_id, "capture_device": source.capture_device,
                     "attack_type": source.attack_type, "class_label": source.class_label,
                     "avg_frame_rate": fraction, "rounded_rate": rate, "actual_image2": actual, "disk_free_image2": sink,
                     "actual_file_count": len(images), "numbering_contiguous_from_one": sequence_valid,
                     "equivalent_count_dup_drop": equivalent, "retained_exports": retained})
        print(f"Calibrated {len(rows)}/{len(sources)}", flush=True)
    report = {"schema": "msu-export-counter-calibration-v1", "protocol_lock_digest": lock_digest,
              "config": config, "config_digest": digest(json_bytes(config)), "records": rows,
              "training_only": True, "passed": all(r["equivalent_count_dup_drop"] for r in rows),
              "scope_limit": "Empirical equivalence on explicit training matrix with same image2/BMP encoder/muxer; not proof across arbitrary builds",
              "experiment_ready": False}
    (output / "calibration.json").write_bytes(json_bytes(report))
    return report


def count_cohort(root, lock, calibration, partition):
    root = Path(root).resolve()
    protocol, lock_digest = _context(root, lock)
    config = policy_config(root)
    validate_export_calibration(calibration)
    if (partition not in ("train", "test") or calibration.get("passed") is not True
            or calibration["protocol_lock_digest"] != lock_digest
            or calibration["config_digest"] != digest(json_bytes(config))):
        raise ValueError("Validated matching calibration required")
    require_training(protocol, [r["source_video_id"] for r in calibration["records"]])
    sources = [r for r in protocol.recordings if r.pad_partition == partition]
    rows = []
    for source in sources:
        fraction, rate = _rate(root, source)
        measured = _export(root, source, rate)
        annotations, _ = parse_face_annotations((root / source.annotation_filepath).read_bytes())
        count = measured["export_count"]
        rows.append(dict(asdict(source), source_video_id=source.video_id, avg_frame_rate=fraction,
                         rounded_rate=rate, **measured, first_export_number=1 if count else None,
                         last_export_number=count if count else None, annotation_count=len(annotations),
                         unavailable_annotation_indices=[i for i in sorted(annotations) if i >= count] if count is not None else None))
        if len(rows) % 20 == 0:
            print(f"Export audit {partition}: {len(rows)}/{len(sources)}", flush=True)
    verify_msu_protocol_lock(root, lock)
    return {"schema": "msu-export-domain-availability-v1", "protocol_lock_digest": lock_digest,
            "config": config, "config_digest": digest(json_bytes(config)), "calibration_digest": digest(json_bytes(calibration)),
            "partition": partition, "policy_selection_eligible": partition == "train", "records": rows,
            "count": len(rows), "total_export_frames": sum(r["export_count"] or 0 for r in rows),
            "unavailable_annotations": sum(len(r["unavailable_annotation_indices"] or []) for r in rows),
            "unknown_availability_videos": sum(r["unavailable_annotation_indices"] is None for r in rows),
            "experiment_ready": False,
            "interpretation": "Validated export-number availability model only; no annotation correspondence established"}


def validate_export_calibration(report):
    """Check measured equivalence, not just the stored pass flag.

    Codec diagnostics remain separate: matching counters never certify decoded
    pixels. Nonzero-drop coverage must be stated separately when it is absent.
    """
    if (report.get("passed") is not True or report.get("training_only") is not True
            or not report.get("records")
            or report.get("config_digest") != digest(json_bytes(report.get("config")))):
        raise ValueError("Invalid export calibration provenance")
    seen = set()
    for row in report["records"]:
        actual, counter = row["actual_image2"], row["disk_free_image2"]
        valid_counts = all(type(run.get(k)) is int and run[k] >= 0
                           for run in (actual, counter)
                           for k in ("export_count", "dup_frames", "drop_frames"))
        equal = all(actual.get(k) == counter.get(k) for k in
                    ("export_count", "dup_frames", "drop_frames", "returncode", "progress_end"))
        if (row["source_video_id"] in seen or row.get("pad_partition") != "train"
                or not valid_counts or not equal or actual.get("returncode") != 0
                or actual.get("progress_end") is not True or actual["export_count"] == 0
                or row.get("actual_file_count") != actual["export_count"]
                or row.get("numbering_contiguous_from_one") is not True
                or row.get("equivalent_count_dup_drop") is not True):
            raise ValueError("Incomplete or inconsistent export calibration measurements")
        seen.add(row["source_video_id"])


def unresolved_policy(protocol, training_audit):
    require_training(protocol, [r["source_video_id"] for r in training_audit["records"]])
    if training_audit.get("partition") != "train" or training_audit.get("policy_selection_eligible") is not True:
        raise ValueError("Test evidence cannot select or modify policy")
    return {"schema": "msu-frame-domain-policy-state-v1", "policy_state": "P4", "frozen_policy": None,
            "selection_evidence_digest": digest(json_bytes(training_audit)), "selection_partition": "train",
            "test_evidence_used_for_selection": False, "fidelity_status": "manual_review_pending", "experiment_ready": False,
            "reason": "Availability and filename association do not prove PittPatt generation correspondence; human review is absent",
            "ordering": "Source and training technical evidence assessed before new test export observations; no policy frozen"}


def codec_classification(native, exported):
    native_bad = bool(native.get("opencv_subprocess_recheck", {}).get("diagnostics"))
    export_bad = exported["returncode"] != 0 or any(re.search(r"prores|decod|damaged|corrupt", s, re.I) for s in exported["diagnostics"])
    if native_bad and export_bad:
        return "both_paths_problematic"
    if export_bad:
        return "bundled_decoder_error"
    if native["successful_reads"] < native["reported_frame_count"] and not export_bad:
        return "native_decoder_shortfall_but_bundled_export_succeeds"
    # Counts cannot establish identical decoded content, even when both paths complete.
    return "unresolved"


def review_template(case_ids):
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("Duplicate human-review case IDs")
    return {"schema": "msu-human-fidelity-review-v1", "reviewer": None, "fidelity_status": "manual_review_pending",
            "experiment_ready": False, "cases": [{"case_id": c, "orientation_acceptable": None,
                "eye_placement_acceptable": None, "face_box_acceptable": None,
                "temporal_correspondence_preference": None, "half_open_endpoint_plausible": None,
                "comments": None} for c in case_ids]}


def readiness_requirements():
    return {"frame_domain_frozen": False, "selected_frame_coverage_validated": False,
            "no_silent_annotation_loss_or_remapping": True, "codec_policy_locked": False,
            "human_geometry_review_completed": False, "human_correspondence_review_completed": False,
            "preprocessing_digest_locked": False, "training_only_model_selection_defined": False,
            "experiment_ready": False}


def validate_local_evidence(root, lock, audit_dir, pack_root, calibration_root, drop_calibration_root):
    """Read-only checks of existing evidence/assets; never decode or approve fidelity.

    The returned payload has no local absolute paths or timestamps. Metadata-only
    cohort/link checks also run in test_msu_policy_evidence without licensed data.
    """
    root, audit_dir, pack_root = Path(root), Path(audit_dir), Path(pack_root)
    protocol, lock_digest = _context(root, lock)
    reports = {}

    def read(name):
        value = json.loads((audit_dir / (name + ".json")).read_text())
        reports[name] = value
        return value

    def check(condition, message):
        if not condition:
            raise ValueError(message)

    source_evidence = read("msu_index_source_evidence")
    for path, expected in source_evidence["files"].items():
        data = (root / path).read_bytes()
        check(len(data) == expected["byte_size"] and digest(data) == expected["sha256"], "Bundled source changed")
    native = {r["source_video_id"]: r for r in read("msu_index_boundaries")["records"]}
    calibration = read("msu_policy_export_calibration")
    supplement = read("msu_policy_drop_calibration")
    for cal, folder in ((calibration, Path(calibration_root)), (supplement, Path(drop_calibration_root))):
        validate_export_calibration(cal)
        require_training(protocol, [r["source_video_id"] for r in cal["records"]])
        check(cal["protocol_lock_digest"] == lock_digest, "Calibration lock mismatch")
        check(cal["config"] == policy_config(root), "Export settings/tool mismatch")
        for row in cal["records"]:
            for item in row["retained_exports"].values():
                if item["available"]:
                    check(digest((folder / item["filepath"]).read_bytes()) == item["sha256"], "Retained BMP changed")
    train, test = read("msu_policy_train_export"), read("msu_policy_test_export")
    decision, release = read("msu_policy_decision"), read("msu_policy_export_release")
    check(decision == unresolved_policy(protocol, train), "Policy is not the training-only unresolved decision")
    check(test["preexisting_policy_state_digest"] == digest(json_bytes(decision)), "Pre-test policy link changed")
    check(release["policy_state_before_test"] == decision, "Release policy snapshot changed")
    rows = train["records"] + test["records"]
    by_id = {r["source_video_id"]: r for r in rows}
    check(len(rows) == len(by_id) == len(protocol.recordings), "Incomplete/duplicate release scope")
    annotations = {}
    for source in protocol.recordings:
        row = by_id[source.video_id]
        check(all(row[k] == v for k, v in asdict(source).items()), "Source metadata conflict")
        parsed, malformed = parse_face_annotations((root / source.annotation_filepath).read_bytes())
        check(not malformed, "Malformed annotation")
        annotations[source.video_id] = parsed
        check(row["annotation_count"] == len(parsed), "Annotation count changed")
        check(row["export_count"] is not None and row["progress_end"] and row["returncode"] == 0,
              "Incomplete export observation")
        check(row["unavailable_annotation_indices"] == [i for i in sorted(parsed) if i >= row["export_count"]],
              "Annotation/export availability conflict")
    for row in release["codec_records"]:
        video = row["source_video_id"]
        check(row["classification"] == codec_classification(native[video], by_id[video]), "Codec classification conflict")
        check(row["original_style_export_diagnostics"] == by_id[video]["diagnostics"], "Codec diagnostics conflict")
        check(not row["process_success_certified"], "Codec result cannot certify success")
    pack, template = read("msu_policy_human_pack"), read("msu_policy_review_template")
    cases, historical = read("msu_policy_training_cases"), read("msu_index_existing_crop_review")
    require_training(protocol, list(cases))
    check(pack["protocol_lock_digest"] == lock_digest, "Review lock mismatch")
    check(pack["calibration_digest"] == digest(json_bytes(calibration)), "Review calibration mismatch")
    check(pack["historical_review_digest"] == digest(json_bytes(historical)), "Historical review mismatch")
    check(pack == json.loads((pack_root / "pack.json").read_text()), "Local pack differs from audit")
    check(template == pack["review_template"] == review_template([p["case_id"] for p in pack["pages"]]),
          "Human review fields must still be unset")
    check(template == json.loads((pack_root / "review-template.json").read_text()), "Local template differs")
    page = (pack_root / "review.html").read_text(encoding="utf-8")
    embedded = re.search(r'<script id="template" type="application/json">(.*?)</script>', page, re.S)
    check(embedded is not None and json.loads(embedded[1]) == template, "HTML review template changed")
    offsets = {"native_i": 0, "native_i_minus_1": -1, "native_i_plus_1": 1, "decframes_export_i": 0}
    for candidate in pack["candidates"]:
        video, index = candidate["source_video_id"], candidate["original_annotation_index"]
        annotation = annotations[video][index]
        check(index in cases[video] and candidate["pad_partition"] == "train", "Non-training review case")
        check(candidate["raw_annotation_line"] == annotation.raw_line
              and candidate["annotation_box"] + candidate["eye_coordinates"] == list(annotation.values),
              "Review annotation changed")
        check(candidate["candidate_frame_index"] == index + offsets[candidate["candidate_label"]], "Shifted review candidate")
        if candidate["available"]:
            check(digest((pack_root / candidate["overlay_path"]).read_bytes()) == candidate["overlay_sha256"], "Overlay changed")
        else:
            check(candidate["overlay_path"] is None and candidate["overlay_sha256"] is None, "Unavailable candidate substituted")
    historical_by_id = {"original-crop:" + r["sample_id"]: r for r in historical["records"]}
    for page in pack["pages"]:
        check(all((pack_root / p).is_file() for p in page["image_paths"]), "Missing review image")
        if page["kind"] == "original_crop":
            check(digest((pack_root / page["image_paths"][1]).read_bytes()) == historical_by_id[page["case_id"]]["crop_sha256"],
                  "Original crop changed")
    return {"schema": "msu-policy-local-evidence-validation-v1", "protocol_lock_digest": lock_digest,
            "input_digests": {name: digest(json_bytes(value)) for name, value in sorted(reports.items())},
            "source_files_verified": len(source_evidence["files"]), "locked_recordings_verified": len(rows),
            "annotations_verified": sum(len(a) for a in annotations.values()),
            "calibrated_training_recordings": len(calibration["records"]) + len(supplement["records"]),
            "human_review_cases_verified": len(pack["pages"]), "codec_classifications_verified": len(release["codec_records"]),
            "validation_status": "passed", "fidelity_status": "manual_review_pending", "experiment_ready": False,
            "scope": "Read-only provenance, metadata and local asset validation; no decoding, policy adoption or human approval"}


def _png(path, image):
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".png", image, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    if not ok:
        raise ValueError("Review image encoding failed")
    data = encoded.tobytes()
    with path.open("xb") as file:
        file.write(data)
    return digest(data)


def write_human_pack(root, lock, cases, calibration, calibration_root, historical_review, historical_root, output):
    """Four visibly labeled domains plus original crops, with human decisions unset."""
    root, calibration_root = Path(root).resolve(), Path(calibration_root).resolve()
    protocol, lock_digest = _context(root, lock)
    sources = require_training(protocol, list(cases))
    validate_export_calibration(calibration)
    if calibration["protocol_lock_digest"] != lock_digest or not calibration["passed"]:
        raise ValueError("Review requires verified calibration provenance")
    by_source = {r["source_video_id"]: r for r in calibration["records"]}
    output = _new_output(root, output)
    candidates, pages = [], []
    mapping = (("native_i", "native_decoder", 0), ("native_i_minus_1", "native_decoder", -1),
               ("native_i_plus_1", "native_decoder", 1), ("decframes_export_i", "decframes_export", 0))
    for source in sources:
        requested = sorted(cases[source.video_id])
        annotations, _ = parse_face_annotations((root / source.annotation_filepath).read_bytes())
        if not requested or len(requested) != len(set(requested)) or any(i not in annotations for i in requested):
            raise ValueError("Review indices must be unique original annotations")
        observation, frames = scan_video(root / source.filepath, {i + d for i in requested for d in (-1, 0, 1) if i + d >= 0})
        rotation = observation["rotation_degrees"]
        for index in requested:
            annotation = annotations[index]
            case_id = f"{source.video_id}:annotation:{index}"
            tiles, ids = [], []
            for label, domain, offset in mapping:
                target_index = index + offset
                item = None
                if domain == "native_decoder":
                    frame = frames.get(target_index)
                else:
                    item = by_source[source.video_id]["retained_exports"][str(index)]
                    frame = None
                    if item["available"]:
                        data = (calibration_root / item["filepath"]).read_bytes()
                        if digest(data) != item["sha256"]:
                            raise ValueError("Retained export hash mismatch")
                        frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
                        if frame is None:
                            raise ValueError("Retained export cannot be decoded")
                identity = export_id(source.video_id, index, calibration["config_digest"]) if domain == "decframes_export" else f"{source.video_id}:native:{target_index}"
                candidate = dict(asdict(source), source_video_id=source.video_id, case_id=case_id,
                    original_annotation_index=index, candidate_frame_domain=domain, candidate_frame_index=target_index,
                    candidate_label=label, candidate_id=identity, export_file_number=index+1 if item else None,
                    rotation_metadata=rotation, raw_annotation_line=annotation.raw_line, annotation_line_number=annotation.line_number,
                    annotation_box=list(annotation.values[:4]), eye_coordinates=list(annotation.values[4:]),
                    available=frame is not None, overlay_path=None, overlay_sha256=None,
                    export_provenance=item, export_policy_digest=calibration["config_digest"] if item else None)
                tile = np.full((295, 360, 3), 240, np.uint8)
                if frame is not None:
                    oriented = rotate_frame(frame, rotation)
                    overlay = oriented.copy()
                    left, top, right, bottom = (round(v) for v in annotation.values[:4])
                    cv2.rectangle(overlay, (left, top), (right - 1, bottom - 1), (0, 255, 0), 2)
                    for j in (4, 6):
                        cv2.circle(overlay, tuple(round(v) for v in annotation.values[j:j+2]), 3, (0, 0, 255), -1)
                    relative = f"overlays/{digest(case_id.encode())}_{label}.png"
                    candidate.update(overlay_path=relative, overlay_sha256=_png(output / relative, overlay),
                                     oriented_pixels_sha256=digest(oriented.tobytes()))
                    tile[:240] = cv2.resize(overlay, (360, 240))
                for y, text in ((255, label), (273, f"annotation {index}; {domain} {target_index}"),
                                (289, "UNREVIEWED" if frame is not None else "UNAVAILABLE; no substitution")):
                    cv2.putText(tile, text, (4, y), cv2.FONT_HERSHEY_SIMPLEX, .40, (0, 0, 0), 1)
                candidates.append(candidate)
                ids.append(identity)
                tiles.append(tile)
            relative = f"sheets/{digest(case_id.encode())}.png"
            _png(output / relative, np.hstack(tiles))
            pages.append({"case_id": case_id, "kind": "training_domain_comparison", "image_paths": [relative],
                          "candidate_ids": ids, "source_video_id": source.video_id,
                          "annotation_index": index, "partition": "train"})
    # Copy and verify the original review imagery into this self-contained ignored pack.
    for row in historical_review["records"]:
        if row["pad_partition"] != "train":
            raise ValueError("Historical human review pack must be training-only")
        case_id = "original-crop:" + row["sample_id"]
        paths = []
        for key in ("review_overlay", "review_crop"):
            src = Path(historical_root) / row[key]
            data = src.read_bytes()
            if key == "review_crop" and digest(data) != row["crop_sha256"]:
                raise ValueError("Historical crop hash mismatch")
            relative = f"original/{digest(case_id.encode())}_{key}.png"
            target = output / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as file:
                file.write(data)
            paths.append(relative)
        pages.append({"case_id": case_id, "kind": "original_crop", "image_paths": paths,
                      "source_video_id": row["source_video_id"], "annotation_index": row["frame_index"], "partition": "train"})
    template = review_template([p["case_id"] for p in pages])
    report = {"schema": "msu-training-domain-human-pack-v1", "protocol_lock_digest": lock_digest,
              "calibration_digest": digest(json_bytes(calibration)), "historical_review_digest": digest(json_bytes(historical_review)),
              "candidates": candidates, "pages": pages, "review_template": template,
              "fidelity_status": "manual_review_pending", "experiment_ready": False,
              "geometry_assumption": "round raw coordinates; half-open rectangle; explicit metadata rotation 0/180; no verified annotation alignment"}
    (output / "pack.json").write_bytes(json_bytes(report))
    (output / "review-template.json").write_bytes(json_bytes(template))
    _review_page(output, report)
    return report


def _review_page(output, report):
    parts = ['<!doctype html><html lang="en"><meta charset="utf-8"><title>MSU human fidelity gate</title>',
             '<style>body{font:16px system-ui;margin:24px;max-width:1480px}img{max-width:100%}article{border-top:2px solid #bbb;margin:24px 0;padding:16px 0}code{overflow-wrap:anywhere}label{display:inline-block;margin:8px}textarea{display:block;width:95%;height:60px}header{position:sticky;top:0;background:#fff;padding:12px;border:1px solid #888}select,input,button{font:inherit;padding:5px}</style>',
             '<header><strong>MSU training-only review — no approval or experiment readiness</strong><br>',
             '<label>Reviewer <input id="reviewer"></label><button id="save">Download review responses (JSON)</button>',
             '<p><button id="previous">Previous case</button> <select id="case-picker"></select> <button id="next">Next case</button> <span id="case-count"></span></p>',
             '<p>Every field begins unset. Use uncertain when evidence is insufficient. Download responses before closing; nothing is sent or approved automatically.</p></header>',
             '<p>Columns: native i, native i−1, native i+1, DecFrames export ordinal i. Green boxes and red eyes reflect an unverified coordinate-space/endpoint assumption. Availability does not establish correspondence. Original eight crops are below.</p>']
    choices = {"orientation_acceptable": ["yes", "no", "uncertain"], "eye_placement_acceptable": ["yes", "no", "uncertain"],
               "face_box_acceptable": ["yes", "no", "uncertain"], "half_open_endpoint_plausible": ["yes", "no", "uncertain"],
               "temporal_correspondence_preference": ["native_i", "native_i_minus_1", "native_i_plus_1", "decframes_export_i", "indistinguishable", "none"]}
    for i, page in enumerate(report["pages"]):
        parts.append(f'<article data-case="{i}"><h2>{i+1}. {html.escape(page["kind"])}</h2><code>{html.escape(page["source_video_id"])}</code><p>Original annotation index: {page["annotation_index"]}</p>')
        for path in page["image_paths"]:
            parts.append(f'<a href="{html.escape(path)}"><img loading="lazy" src="{html.escape(path)}" alt="Full resolution review image"></a>')
        if page["kind"] == "training_domain_comparison":
            links = []
            for candidate in report["candidates"]:
                if candidate["case_id"] != page["case_id"]:
                    continue
                label = html.escape(candidate["candidate_label"])
                if candidate["available"]:
                    links.append(f'<a href="{html.escape(candidate["overlay_path"])}">{label}: full resolution overlay</a>')
                else:
                    links.append(f'{label}: UNAVAILABLE (no substitution)')
            parts.append('<p>' + ' | '.join(links) + '</p>')
        for name, values in choices.items():
            options = '<option value="">Unset</option>' + ''.join(f'<option>{v}</option>' for v in values)
            parts.append(f'<label>{name.replace("_", " ")} <select data-field="{name}">{options}</select></label>')
        parts.append('<label>Comments</label><textarea data-field="comments"></textarea></article>')
    payload = json.dumps(report["review_template"]).replace("</", "<\\/")
    parts.append('<script id="template" type="application/json">'+payload+'</script>')
    parts.append('''<script>
const articles = Array.from(document.querySelectorAll('article[data-case]'));
const picker = document.getElementById('case-picker');
articles.forEach((article, i) => {
  const option = document.createElement('option'); option.value = i;
  option.textContent = article.querySelector('h2').textContent;
  picker.appendChild(option);
});
let current = 0;
function showCase(index) {
  current = Math.max(0, Math.min(index, articles.length - 1));
  articles.forEach((article, i) => {article.hidden = i !== current;});
  picker.value = current;
  document.getElementById('case-count').textContent = `Case ${current + 1} of ${articles.length}`;
  document.getElementById('previous').disabled = current === 0;
  document.getElementById('next').disabled = current === articles.length - 1;
}
picker.addEventListener('change', () => showCase(Number(picker.value)));
document.getElementById('previous').addEventListener('click', () => showCase(current - 1));
document.getElementById('next').addEventListener('click', () => showCase(current + 1));
showCase(0);
document.getElementById('save').addEventListener('click', () => {
  const result = JSON.parse(document.getElementById('template').textContent);
  result.reviewer = document.getElementById('reviewer').value.trim() || null;
  document.querySelectorAll('article[data-case]').forEach(article => {
    const row = result.cases[Number(article.dataset.case)];
    article.querySelectorAll('[data-field]').forEach(control => {row[control.dataset.field] = control.value || null;});
  });
  const url = URL.createObjectURL(new Blob([JSON.stringify(result, null, 2)+'\\n'], {type:'application/json'}));
  const link = document.createElement('a'); link.href = url; link.download = 'msu-human-review-responses.json'; link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
});
</script></html>''')
    (output / "review.html").write_text('\n'.join(parts), encoding="utf-8")
