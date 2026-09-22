"""MSU frame-index evidence only: no crops, mapping adoption, or fidelity approval."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np

from .msu_protocol import load_msu_protocol, verify_msu_protocol_lock
from .msu_preprocessing import (
    PreprocessingConfig, digest, json_bytes, parse_face_annotations, rotate_frame,
)


def scan_video(path: Path, retain=()):
    """Sequential AUTO=0 reads to first failure. EOF versus corruption is not inferred.

    fd redirection is best effort and process-global: do not call concurrently in
    threads. Some Windows OpenCV builds use a separate CRT; use the subprocess
    diagnostics in enrich_stop_evidence, never an empty fd capture as proof.
    """
    diagnostics = tempfile.TemporaryFile()
    original_stderr = os.dup(2)
    os.dup2(diagnostics.fileno(), 2)
    cap = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG)
    saved, count = {}, 0
    try:
        if (not cap.isOpened() or cap.getBackendName() != "FFMPEG"
                or not cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)
                or cap.get(cv2.CAP_PROP_ORIENTATION_AUTO) != 0):
            raise ValueError("FFmpeg/AUTO=0 unavailable")
        header = {"reported_frame_count": cap.get(cv2.CAP_PROP_FRAME_COUNT),
                  "reported_fps": cap.get(cv2.CAP_PROP_FPS),
                  "rotation_degrees": cap.get(cv2.CAP_PROP_ORIENTATION_META)}
        wanted = set(retain)
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if cap.get(cv2.CAP_PROP_POS_FRAMES) != count + 1:
                raise ValueError("Sequential decoder position mismatch")
            if count in wanted:
                saved[count] = frame.copy()
            count += 1
    finally:
        cap.release()
        os.dup2(original_stderr, 2)
        os.close(original_stderr)
        diagnostics.seek(0)
        messages = diagnostics.read().decode("utf-8", errors="replace")
        diagnostics.close()
    messages = re.sub(r" @ [0-9A-Fa-f]+", "", messages)
    return dict(header, successful_reads=count, decoded_range=[0, count - 1] if count else None,
                decoder_diagnostics=messages.splitlines(),
                diagnostic_capture="best_effort_fd_redirect; empty output does not certify clean decoding",
                stop_reason="first_unsuccessful_read; EOF versus decoder error not independently distinguished"), saved


def compare_indices(indices, count):
    indices = sorted(indices)
    if not indices or count < 0:
        raise ValueError("Nonempty annotations and nonnegative decoded count required")
    return {"annotation_min": indices[0], "annotation_max": indices[-1], "annotation_count": len(indices),
            "internal_annotation_gaps": sorted(set(range(indices[0], indices[-1] + 1)) - set(indices)),
            "unavailable_annotation_indices": [i for i in indices if i >= count],
            "decoded_indices_without_annotation": sorted(set(range(count)) - set(indices))}


def _context(root, lock):
    verify_msu_protocol_lock(root, lock)
    protocol = load_msu_protocol(root)
    return protocol, digest(json_bytes(json.loads(Path(lock).read_text(encoding="utf-8"))))


def boundary_audit(root, lock, video_ids=None, progress=None):
    """Full release by default; failures and explicit partial scopes remain visible."""
    root = Path(root)
    protocol, lock_digest = _context(root, lock)
    sources = {r.video_id: r for r in protocol.recordings}
    ids = sorted(sources) if video_ids is None else sorted(video_ids)
    if not ids or len(ids) != len(set(ids)) or any(i not in sources for i in ids):
        raise ValueError("Invalid audit source scope")
    rows = []
    for i, video_id in enumerate(ids, 1):
        source = sources[video_id]
        annotations, _ = parse_face_annotations((root / source.annotation_filepath).read_bytes())
        row = dict(asdict(source), source_video_id=video_id)
        try:
            decoded, _ = scan_video(root / source.filepath)
            row.update(decoded)
            row.update(compare_indices(annotations, decoded["successful_reads"]))
            row["audit_status"] = "decoded_to_first_failure"
        except (ValueError, cv2.error) as exc:
            row.update(audit_status="failed", error_type=type(exc).__name__)
        rows.append(row)
        if progress:
            progress(i, len(ids))
    # Check raw provenance again, never regenerate a lock after an audit.
    verify_msu_protocol_lock(root, lock)
    summaries = {}
    for field in ("pad_partition", "capture_device", "class_label", "attack_type"):
        summaries[field] = {}
        for row in rows:
            key = str(row[field])
            group = summaries[field].setdefault(key, {"videos": 0, "failed": 0, "decoded_frames": 0,
                                                     "videos_with_unavailable_annotations": 0, "unavailable_annotations": 0})
            group["videos"] += 1
            group["failed"] += row["audit_status"] == "failed"
            group["decoded_frames"] += row.get("successful_reads", 0)
            missing = row.get("unavailable_annotation_indices", [])
            group["videos_with_unavailable_annotations"] += bool(missing)
            group["unavailable_annotations"] += len(missing)
    return {"schema": "msu-index-boundary-evidence-v1", "protocol_lock_digest": lock_digest,
            "scope": "full_locked_release" if set(ids) == set(sources) else "explicit_partial_scope",
            "libraries": PreprocessingConfig().payload()["libraries"], "source_count": len(sources),
            "videos_attempted": len(rows), "videos_decoded": sum(r["audit_status"] != "failed" for r in rows),
            "annotation_max_distribution": dict(sorted(Counter(str(r["annotation_max"]) for r in rows if "annotation_max" in r).items())),
            "summaries": summaries, "records": rows, "experiment_ready": False,
            "interpretation": "Availability under identity mapping only; no mapping established or annotations changed. No PAD outcomes used."}


def candidate_indices(annotation_index):
    if type(annotation_index) is not int or annotation_index < 0:
        raise ValueError("Original annotation index must be a nonnegative integer")
    return {"A:i": annotation_index, "B:i-1": annotation_index - 1, "C:i+1": annotation_index + 1}


def assess_stop(native_count, external_count, returncode, progress_end, diagnostics, native_diagnostics=()):
    """Conservative distinction; first failed OpenCV read alone never proves EOF."""
    combined = [s.lower() for s in [*diagnostics, *native_diagnostics]]
    if any("[prores]" in s or "error while decoding" in s for s in combined):
        return "codec_failure_evidence"
    if returncode != 0:
        return "external_decode_failed_unclassified"
    if not progress_end or external_count is None:
        return "unresolved_termination"
    if native_count != external_count:
        return "decoder_count_disagreement"
    if diagnostics or native_diagnostics:
        if not native_diagnostics and all(
            ("[null]" in s and "non monotonically increasing dts" in s)
            or s.strip().startswith("last message repeated") for s in combined
        ):
            return "completed_with_mux_timestamp_diagnostics"
        return "diagnostics_unresolved"
    return "consistent_with_clean_eof_in_independent_decode"


def enrich_stop_evidence(root, lock, existing, progress=None):
    """Retain previous OpenCV observations; independently decode with bundled -xerror/-vsync 0.

    No BMPs or crops are produced. stderr is captured at the subprocess boundary,
    which avoids the unreliable cross-CRT fd redirection of OpenCV on Windows.
    """
    root = Path(root).resolve()
    protocol, lock_digest = _context(root, lock)
    if existing["protocol_lock_digest"] != lock_digest:
        raise ValueError("Evidence/raw lock mismatch")
    sources = {r.video_id: r for r in protocol.recordings}
    report = json.loads(json.dumps(existing))
    ids = [r["source_video_id"] for r in report["records"]]
    if len(ids) != len(set(ids)) or any(v not in sources for v in ids):
        raise ValueError("Invalid prior audit source IDs")
    for row in report["records"]:
        if any(row.get(k) != v for k, v in asdict(sources[row["source_video_id"]]).items()):
            raise ValueError("Prior audit source metadata conflicts with raw protocol")
    executable = root / "ffmpeg/bin/ffmpeg.exe"
    for number, row in enumerate(report["records"], 1):
        source = sources[row["source_video_id"]]
        relative_args = ["-v", "error", "-xerror", "-i", source.filepath,
                         "-map", "0:v:0", "-vsync", "0", "-f", "null", "-", "-progress", "pipe:1"]
        args = list(relative_args)
        args[4] = str(root / source.filepath)
        run = subprocess.run([str(executable), *args], capture_output=True)
        lines = run.stdout.decode("utf-8", errors="replace").splitlines()
        counts = [int(s.split("=")[1]) for s in lines if re.fullmatch(r"frame=\d+", s)]
        messages = run.stderr.decode("utf-8", errors="replace").replace(str(root), "<raw-root>")
        messages = re.sub(r" @ [0-9A-Fa-f]+", "", messages).splitlines()
        count = counts[-1] if counts else None
        evidence = {"relative_command": ["ffmpeg/bin/ffmpeg.exe", *relative_args],
                    "returncode": run.returncode, "final_progress_frame_count": count,
                    "progress_end": "progress=end" in lines, "diagnostics": messages}
        row["independent_decode"] = evidence
        row["stop_assessment"] = assess_stop(row["successful_reads"], count, run.returncode,
                                              evidence["progress_end"], messages)
        row["opencv_diagnostic_capture"] = "prior fd capture not reliable on this Windows build; not evidence of absent codec errors"
        if row["stop_assessment"] != "consistent_with_clean_eof_in_independent_decode":
            # Recheck only anomalous sources in isolated OpenCV processes for attributable stderr.
            code = "import json,sys; from pathlib import Path; from identity_invariant_fas.data.msu_index_evidence import scan_video; print(json.dumps(scan_video(Path(sys.argv[1]))[0]))"
            child = subprocess.run([sys.executable, "-c", code, str(root / source.filepath)], capture_output=True)
            diagnostic = child.stderr.decode("utf-8", errors="replace").replace(str(root), "<raw-root>")
            row["opencv_subprocess_recheck"] = {
                "returncode": child.returncode, "observation": json.loads(child.stdout) if child.returncode == 0 else None,
                "diagnostics": re.sub(r" @ [0-9A-Fa-f]+", "", diagnostic).splitlines()}
            row["stop_assessment"] = assess_stop(
                row["successful_reads"], count, run.returncode, evidence["progress_end"], messages,
                row["opencv_subprocess_recheck"]["diagnostics"])
        if progress:
            progress(number, len(report["records"]))
    report["independent_decoder_sha256"] = digest(executable.read_bytes())
    report["stop_assessment_counts"] = dict(Counter(r["stop_assessment"] for r in report["records"]))
    report["diagnostic_note"] = "Completion/count agreement is consistent with clean EOF; no claim that unavailable indices should be corrected. Codec failures are separate from annotation endpoints."
    verify_msu_protocol_lock(root, lock)
    return report


def write_review(root, lock, cases, output):
    """Create full-frame overlays for explicit annotated indices; preserve unavailable candidates.

    `cases` maps canonical recording IDs to original annotation indices. This never
    selects/adopts a candidate mapping or creates a processed training manifest.
    """
    root, output = Path(root).resolve(), Path(output).resolve()
    if output == root or root in output.parents or output in root.parents:
        raise ValueError("Review output must be outside raw root")
    protocol, lock_digest = _context(root, lock)
    sources = {r.video_id: r for r in protocol.recordings}
    if not cases or any(v not in sources for v in cases):
        raise ValueError("Invalid review sources")
    output.mkdir(parents=True, exist_ok=False)
    rows, sheets = [], []
    for video_id, requested in sorted(cases.items()):
        source = sources[video_id]
        annotations, _ = parse_face_annotations((root / source.annotation_filepath).read_bytes())
        if not requested or len(requested) != len(set(requested)) or any(i not in annotations for i in requested):
            raise ValueError("Review requires unique existing annotation indices")
        wanted = {j for i in requested for j in candidate_indices(i).values() if j >= 0}
        decoded, frames = scan_video(root / source.filepath, wanted)
        rotation = decoded["rotation_degrees"]
        # Follow existing explicit orientation assumption without approving it.
        for index in sorted(requested):
            annotation = annotations[index]
            tiles = []
            sheet_name = f"sheets/{digest(video_id.encode())}_{index:06d}.png"
            for label, candidate in candidate_indices(index).items():
                record = dict(asdict(source), source_video_id=video_id, annotation_index=index,
                              candidate_decoder_index=candidate, candidate_mapping=label,
                              annotation_line_number=annotation.line_number, annotation_raw_line=annotation.raw_line,
                              annotation_values=list(annotation.values), rotation_metadata=rotation,
                              overlay_filepath=None, overlay_sha256=None, contact_sheet=sheet_name,
                              available=candidate in frames)
                tile = np.full((285, 360, 3), 235, np.uint8)
                if candidate in frames:
                    frame = rotate_frame(frames[candidate], rotation)
                    box = tuple(round(v) for v in annotation.values[:4])
                    overlay = frame.copy()
                    cv2.rectangle(overlay, box[:2], (box[2] - 1, box[3] - 1), (0, 255, 0), 2)
                    for j in (4, 6):
                        cv2.circle(overlay, tuple(round(v) for v in annotation.values[j:j + 2]), 3, (0, 0, 255), -1)
                    name = f"overlays/{digest(video_id.encode())}_{index:06d}_{label[0]}.png"
                    _write_png(output / name, overlay)
                    record.update(overlay_filepath=name, overlay_sha256=digest((output / name).read_bytes()),
                                  decoded_pixels_sha256=digest(frame.tobytes()))
                    tile[:240] = cv2.resize(overlay, (360, 240))
                for y, text in ((257, f"annotation {index} -> decoder {candidate} ({label})"),
                                (276, "AVAILABLE; unreviewed" if record["available"] else "UNAVAILABLE; no substitution")):
                    cv2.putText(tile, text, (5, y), cv2.FONT_HERSHEY_SIMPLEX, .44, (0, 0, 0), 1)
                tiles.append(tile)
                rows.append(record)
            _write_png(output / sheet_name, np.hstack(tiles))
            sheets.append(sheet_name)
    report = {"schema": "msu-index-candidate-review-v1", "protocol_lock_digest": lock_digest,
              "libraries": PreprocessingConfig().payload()["libraries"], "records": rows, "contact_sheets": sheets,
              "case_selection": "Explicit reviewed cases/indices; no candidate adopted; no sampling change",
              "fidelity_status": "manual_review_pending", "experiment_ready": False,
              "rendering": "Full-frame overlays: rounded raw coordinates, half-open rectangle assumption; contact-sheet resize is for display only"}
    (output / "review.json").write_bytes(json_bytes(report))
    return report


def _write_png(path, image):
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".png", image, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    if not ok:
        raise ValueError("Overlay encoding failed")
    with path.open("xb") as file:
        file.write(encoded.tobytes())


def audit_bundled_export(root, lock, video_ids, output):
    """Reproduce DecFrames.m's rounded-rate BMP export outside raw data.

    Compare selected exported images with nearby native sequential images. Pixel
    MAE is a decoder comparison, not a face-alignment metric or a mapping rule.
    """
    root, output = Path(root).resolve(), Path(output).resolve()
    if output == root or root in output.parents or output in root.parents:
        raise ValueError("Export output must be outside raw root")
    protocol, lock_digest = _context(root, lock)
    sources = {r.video_id: r for r in protocol.recordings}
    if not video_ids or len(set(video_ids)) != len(video_ids) or any(v not in sources for v in video_ids):
        raise ValueError("Invalid source scope")
    output.mkdir(parents=True, exist_ok=False)
    ffmpeg, ffprobe = root / "ffmpeg/bin/ffmpeg.exe", root / "ffmpeg/bin/ffprobe.exe"
    records = []
    for video_id in sorted(video_ids):
        source = sources[video_id]
        stream = subprocess.run([str(ffprobe), "-v", "quiet", "-show_streams", str(root / source.filepath)], capture_output=True, check=True).stdout.decode("utf-8", errors="replace")
        rate_match = re.search(r"^avg_frame_rate=(\d+)/(\d+)\r?$", stream, re.M)
        if rate_match is None or int(rate_match[2]) == 0:
            raise ValueError("Missing original-script average frame rate")
        rate = f"{int(rate_match[1]) / int(rate_match[2]):.2f}"
        directory = output / digest(video_id.encode())
        directory.mkdir()
        pattern = directory / (Path(source.filepath).stem + "_%03d.bmp")
        command = [str(ffmpeg), "-i", str(root / source.filepath), "-r", rate, str(pattern)]
        result = subprocess.run(command, capture_output=True, check=True)
        # The only absolute command paths are execution arguments; canonical evidence stays relative.
        log = result.stderr.decode("utf-8", errors="replace")
        images = sorted(directory.glob("*.bmp"))
        annotations, _ = parse_face_annotations((root / source.annotation_filepath).read_bytes())
        selected = sorted(set([min(annotations), *sorted(annotations)[:3], sorted(annotations)[len(annotations)//2], max(annotations)]))
        native_meta, native = scan_video(root / source.filepath, {i + d for i in selected for d in range(-2, 3) if i + d >= 0})
        comparisons = []
        for index in selected:
            exported_path = directory / (Path(source.filepath).stem + f"_{index+1:03d}.bmp")
            entry = {"annotation_index": index, "exported_file_number": index + 1,
                     "export_filepath": exported_path.relative_to(output).as_posix(), "available": exported_path.is_file()}
            if exported_path.is_file():
                exported = cv2.imread(str(exported_path))
                errors = {}
                for j in sorted(native):
                    if abs(j-index) <= 2:
                        errors[str(j)] = float(np.abs(exported.astype(np.int16) - native[j].astype(np.int16)).mean())
                entry.update(export_sha256=digest(exported_path.read_bytes()), nearby_native_pixel_mae=errors)
            comparisons.append(entry)
        records.append({"source_video_id": video_id, "client_id": source.client_id, "pad_partition": source.pad_partition,
                        "capture_device": source.capture_device, "class_label": source.class_label, "attack_type": source.attack_type,
                        "avg_frame_rate_fraction": f"{rate_match[1]}/{rate_match[2]}", "rounded_output_rate": rate,
                        "relative_command": ["ffmpeg/bin/ffmpeg.exe", "-i", source.filepath, "-r", rate, "<ignored-output>/" + pattern.name],
                        "export_count": len(images), "first_export": images[0].name if images else None,
                        "last_export": images[-1].name if images else None,
                        "native_decoder": native_meta, "annotation_max": max(annotations),
                        "export_unavailable_annotations": [i for i in annotations if not (directory / (Path(source.filepath).stem + f"_{i+1:03d}.bmp")).is_file()],
                        "ffmpeg_final_progress": [s.strip() for s in log.splitlines() if s.strip().startswith("frame=")][-1:],
                        "comparisons": comparisons})
    report = {"schema": "msu-bundled-export-evidence-v1", "protocol_lock_digest": lock_digest,
              "tool_sha256": {p.relative_to(root).as_posix(): digest(p.read_bytes()) for p in (ffmpeg, ffprobe)},
              "records": records, "experiment_ready": False,
              "interpretation": "Reproduces image export, not annotation generation. No sidecar-to-native mapping adopted. Raw unrotated pixel comparison; no PAD metrics."}
    (output / "audit.json").write_bytes(json_bytes(report))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--lock", default="docs/audit/msu_protocol_lock.json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--cases", help="JSON object mapping canonical source IDs to annotation indices; creates review directory")
    parser.add_argument("--enrich-existing", help="Prior boundary JSON; retain observations and add independent stop diagnostics")
    args = parser.parse_args()
    root, output = Path(args.root).resolve(), Path(args.output).resolve()
    if output == root or root in output.parents or output in root.parents or output.exists():
        parser.error("Output must be new and outside raw root")
    if args.cases and args.enrich_existing:
        parser.error("Choose candidate review or stop-diagnostic enrichment")
    if args.cases:
        write_review(root, args.lock, json.loads(Path(args.cases).read_text()), output)
    else:
        progress = lambda i, n: print(f"Audited {i}/{n}", flush=True) if i % 20 == 0 else None
        report = (enrich_stop_evidence(root, args.lock, json.loads(Path(args.enrich_existing).read_text()), progress)
                  if args.enrich_existing else boundary_audit(root, args.lock, progress=progress))
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("xb") as file:
            file.write(json_bytes(report))


if __name__ == "__main__":
    main()
