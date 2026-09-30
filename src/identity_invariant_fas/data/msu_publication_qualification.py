"""Bounded Path-B orchestration. Importing this module performs no media I/O.

The only media scope is TRAIN_QUALIFICATION; Phase A uses its frozen prefix.
The command must run in the isolated MSU runtime with already verified artifacts.
"""
from dataclasses import asdict
import importlib.metadata
from pathlib import Path
import platform
import shutil
import subprocess

import numpy as np
from PIL import Image

from . import msu_publication_preprocessing as p
from . import msu_publication_runtime as r


CATALOG_SHA256 = "cfba71ee3ffc798ea1d21a3a859dfa29f0c83210316177850dd8de562454b838"
REVIEW_POSITIONS = (0, 14, 29)
IMPLEMENTATION_FILES = (
    "src/identity_invariant_fas/__init__.py",
    "src/identity_invariant_fas/data/__init__.py",
    "src/identity_invariant_fas/data/msu_publication_preprocessing.py",
    "src/identity_invariant_fas/data/msu_publication_runtime.py",
    "src/identity_invariant_fas/data/msu_yunet_worker.py",
    "src/identity_invariant_fas/data/msu_publication_qualification.py",
    "scripts/run_msu_eight_video_qualification.py",
)
POLICY_FILES = (
    "docs/MSU_PROSPECTIVE_PREPROCESSING.md",
    "docs/PUBLICATION_EXPERIMENT_MATRIX_AMENDMENT_001.md",
    "docs/PUBLICATION_EXPERIMENT_MATRIX_AMENDMENT_002.md",
)


def _repository():
    return Path(__file__).resolve().parents[3]


def _git(repo, git_executable, *args):
    p._absolute(git_executable)
    return subprocess.run([str(git_executable), "-C", str(repo), *args],
                          stdin=subprocess.DEVNULL, capture_output=True, shell=False, timeout=30)


def implementation_state(repo, expected_head, git_executable):
    p._require(len(expected_head) == 40 and all(c in "0123456789abcdef" for c in expected_head),
               "explicit committed HEAD required")
    head = _git(repo, git_executable, "rev-parse", "HEAD")
    p._require(head.returncode == 0 and head.stdout.decode().strip() == expected_head,
               "unexpected committed HEAD")
    hashes = {}
    for name in (*IMPLEMENTATION_FILES, *POLICY_FILES):
        committed = _git(repo, git_executable, "show", "HEAD:" + name)
        path = repo / name
        # Git may check LF blobs out as CRLF on Windows. No other difference allowed.
        p._require(committed.returncode == 0 and
                   committed.stdout.replace(b"\r\n", b"\n") == path.read_bytes().replace(b"\r\n", b"\n"),
                   "uncommitted/missing implementation or policy: " + name)
        hashes[name] = p.stream_sha256(path)
    return {"head": expected_head, "files": hashes}


def verify_runtime_versions():
    actual = (("Python", platform.python_version()),) + tuple(
        (name, importlib.metadata.version(name)) for name, _ in p.RUNTIME[1:])
    p._require(actual == p.RUNTIME, "wrong isolated runtime; no recording access allowed")
    installed = {d.metadata["Name"].lower().replace("_", "-")
                 for d in importlib.metadata.distributions() if d.metadata["Name"]}
    opencv = {n for n in installed if n.startswith("opencv-")}
    p._require(opencv == {"opencv-python"} and not any(n.startswith("onnxruntime") for n in installed),
               "conflicting OpenCV/ONNX Runtime package")
    return actual


def artifact_preflight(archive, ffmpeg, ffprobe, model):
    for path in (archive, ffmpeg, ffprobe, model):
        p._absolute(path)
    p.verify_file(archive, p.ARCHIVE_SHA256)
    executables = {role: r.verify_approved_executable(path, role)
                   for role, path in (("ffmpeg", ffmpeg), ("ffprobe", ffprobe))}
    p.verify_file(model, p.MODEL_SHA256, 232589)
    # Authoritative cv2/build check in a fresh child, before any recording access.
    worker = r.infer_yunet(model, np.zeros((640, 640, 3), dtype=np.uint8))
    environment = p.EnvironmentLock(p.ARCHIVE_SHA256, p.MODEL_SHA256,
        executables["ffmpeg"].sha256, executables["ffprobe"].sha256, p.RUNTIME,
        executables["ffmpeg"].version, executables["ffprobe"].version,
        worker["opencv_build"], tuple(tuple(x) for x in worker["runtime_fingerprints"]))
    p.validate_artifact_files(environment, archive, model, ffmpeg, ffprobe)
    return executables, environment


def fresh_ignored_output(repo, raw_root, output, git_executable):
    raw, out, repo = Path(raw_root).resolve(), Path(output).resolve(), Path(repo).resolve()
    p._require(out.is_relative_to(repo / "outputs") and out != repo / "outputs",
               "qualification output must be beneath repository ignored outputs/")
    p._require(not raw.is_relative_to(out) and not out.is_relative_to(raw), "raw/output roots overlap")
    p._require(not out.exists(), "qualification output already exists")
    relative = out.relative_to(repo).as_posix()
    tracked = _git(repo, git_executable, "ls-files", "--", relative)
    p._require(tracked.returncode == 0 and not tracked.stdout, "output contains tracked paths")
    for suffix in ("png", "json"):
        ignored = _git(repo, git_executable, "check-ignore", "-q", "--", relative + "/__qualification__." + suffix)
        p._require(ignored.returncode == 0, "qualification outputs are not ignored")
    return p.fresh_output(raw, out)


def _failure(root, error):
    # Never replace the primary failure, or overwrite an existing failure marker.
    if not any((root / name).exists() for name in ("failure.json", "manifest.json", "completion.json")):
        try:
            p.record_failure(root, str(error))
        except OSError:
            pass


def _review_pair(review, run_root, source, j, selected, record, video_index):
    stem = f"{video_index:02d}-j{j:02d}"
    source_name, crop_name = stem + "-source.png", stem + "-crop.png"
    with (review / source_name).open("xb") as stream:
        Image.fromarray(selected.oriented_rgb).save(stream, format="PNG")
    with Image.open(review / source_name) as image:
        p._require(image.mode == "RGB" and np.array_equal(np.asarray(image), selected.oriented_rgb),
                   "review source PNG reload mismatch")
    with (run_root / record["output_path"]).open("rb") as src, (review / crop_name).open("xb") as dst:
        shutil.copyfileobj(src, dst)
    p._strict_equal(p.inspect_png(review / crop_name), record["output"], "review crop copy mismatch")
    return {"video_id": source.video_id, "j": j, "ordinal": selected.ordinal,
            "source_frame": source_name, "crop": crop_name}


def _process_source(root, raw_root, source, snapshot, executables, model, index, review):
    decoded, probe, selected = r.decode_selected_frames(executables, raw_root, source,
        snapshot, p.digest(snapshot), p.source_codec(source))
    r.validate_selected_frames(decoded, selected)
    evidence = root / "evidence" / f"{index:02d}"
    evidence.mkdir(parents=True)
    p.write_once(evidence / "decode.json", asdict(decoded))
    p.write_once(evidence / "probe.json", probe)
    records, detections, pairs = [], [], []
    for j, frame in enumerate(selected):
        worker = r.infer_yunet(model, frame.oriented_rgb)
        # Preserve full normalized box/landmark/score output, including failed selection.
        p.write_once(evidence / f"{j:02d}-detector.json", worker)
        record = p.make_frame_record(source, decoded.source_sha256, decoded, frame.ordinal,
            frame.oriented_rgb, worker["detections"], root, f"crops/{index:02d}/{j:02d}.png")
        p.write_once(evidence / f"{j:02d}-frame.json", record)
        records.append(record)
        detections.append(worker["detections"])
        if review is not None and j in REVIEW_POSITIONS:
            pairs.append(_review_pair(review, root, source, j, frame, record, index))
    # Selected arrays die here, before the next video's decode.
    return decoded, records, detections, pairs


def _process_run(raw_root, catalog, executables, model, environment, provenance, output,
                 *, gate=False, review=None):
    root = p.fresh_output(raw_root, output)
    try:
        snapshot = (p.codec_gate_snapshot(raw_root, catalog) if gate else
                    p.source_snapshot(raw_root, catalog, p.TRAIN_QUALIFICATION))
        p.write_once(root / "snapshot.json", snapshot)
        sources = p.qualification_sources(catalog, p.TRAIN_QUALIFICATION)
        if gate:
            sources = sources[:len(p.TRAIN_CODEC_GATE)]
        decodes, records, detections, pairs = [], [], [], []
        for index, source in enumerate(sources):
            decoded, frames, found, reviews = _process_source(root, raw_root, source, snapshot,
                executables, model, index, review)
            decodes.append(decoded)
            records.extend(frames)
            detections.extend(found)
            pairs.extend(reviews)
        if gate:
            p._strict_equal(p.codec_gate_snapshot(raw_root, catalog), snapshot, "codec gate source drift")
            p.write_once(root / "codec-gate.json", {"status": "passed", "snapshot_sha256": p.digest(snapshot),
                "decodes": [asdict(d) for d in decodes], "frames": records, "detections": detections})
            return None
        document = p.make_manifest(catalog, snapshot, environment, decodes, records, provenance)
        p.validate_manifest(document, catalog, snapshot, environment, root)
        return root, snapshot, document, detections, pairs
    except BaseException as error:
        _failure(root, error)
        raise


def _difference(first, second, field=""):
    """First exact canonical difference, with no numeric tolerance."""
    if p.canonical_bytes(first) == p.canonical_bytes(second):
        return None
    if isinstance(first, dict) and isinstance(second, dict) and first.keys() == second.keys():
        for key in first:
            mismatch = _difference(first[key], second[key], field + "." + key)
            if mismatch:
                return mismatch
    if isinstance(first, list) and isinstance(second, list) and len(first) == len(second):
        for index, (a, b) in enumerate(zip(first, second)):
            mismatch = _difference(a, b, f"{field}[{index}]")
            if mismatch:
                return mismatch
    return {"field": field, "run_1": first, "run_2": second}


def compare_runs(first, second):
    """Compare all decode/frame fields and full post-NMS output, excluding host roots."""
    _, snap1, doc1, det1, _ = first
    _, snap2, doc2, det2, _ = second
    for key, a, b in (("snapshot", snap1, snap2), ("decodes", doc1["decodes"], doc2["decodes"]),
                      ("frames", doc1["frames"], doc2["frames"]), ("detections", det1, det2)):
        mismatch = _difference(a, b, key)
        if mismatch:
            if key == "snapshot":
                for left, right in zip(snap1["sources"], snap2["sources"]):
                    if p.canonical_bytes(left) != p.canonical_bytes(right):
                        mismatch.update(video_id=left["source"]["video_id"])
                        break
            # Include a directly actionable video/sample location where applicable.
            for index, (left, right) in enumerate(zip(a if isinstance(a, list) else [],
                                                     b if isinstance(b, list) else [])):
                if p.canonical_bytes(left) != p.canonical_bytes(right):
                    if key in ("frames", "detections"):
                        frame = doc1["frames"][index]
                        mismatch.update(video_id=frame["group_id"], j=index % 30, ordinal=frame["ordinal"])
                    elif key == "decodes":
                        mismatch.update(video_id=left["video_id"])
                    break
            return mismatch
    return None


def run_qualification(*, raw_root, catalog, protocol_lock, archive, ffmpeg, ffprobe, model,
                      output, expected_head, git_executable):
    """No scope argument: the only executable scope is the frozen training eight."""
    repo = _repository()
    verify_runtime_versions()
    state = implementation_state(repo, expected_head, git_executable)
    executables, environment = artifact_preflight(archive, ffmpeg, ffprobe, model)
    for path in (raw_root, catalog, protocol_lock, output):
        p._absolute(path)
    metadata = p.load_source_catalog_metadata(raw_root, catalog, protocol_lock)
    p._require(metadata.catalog_sha256 == CATALOG_SHA256, "wrong frozen source catalog digest")
    p.qualification_sources(metadata, p.TRAIN_QUALIFICATION)
    root = fresh_ignored_output(repo, raw_root, output, git_executable)
    provenance = {"policy_document_sha256": state["files"][POLICY_FILES[0]],
        "amendment_sha256": p.digest({name: state["files"][name] for name in POLICY_FILES[1:]}),
        "implementation_sha256": p.digest({name: state["files"][name] for name in IMPLEMENTATION_FILES})}
    pending = None
    try:
        p.write_once(root / "preflight.json", {"implementation": state,
            "environment": asdict(environment), "catalog_sha256": metadata.catalog_sha256,
            "projection_sha256": metadata.projection_sha256, "scope": p.TRAIN_QUALIFICATION})
        _process_run(raw_root, metadata, executables, model, environment, provenance, root / "phase-a", gate=True)
        review = root / "review"
        review.mkdir()
        first = _process_run(raw_root, metadata, executables, model, environment, provenance,
                             root / "run-1", review=review)
        pending = first[0]
        p.write_manifest(first[0], first[2], metadata, first[1], environment, raw_root)
        pending = None
        second = _process_run(raw_root, metadata, executables, model, environment, provenance, root / "run-2")
        pending = second[0]
        mismatch = compare_runs(first, second)
        if mismatch:
            p.write_once(root / "repeatability-mismatch.json", mismatch)
            raise p.ContractError("exact repeatability mismatch; see repeatability-mismatch.json")
        p._strict_equal(implementation_state(repo, expected_head, git_executable), state, "implementation drift during run")
        p.write_manifest(second[0], second[2], metadata, second[1], environment, raw_root)
        pending = None
        pairs = first[4]
        p._require([(pair["video_id"], pair["j"]) for pair in pairs] ==
                   [(v, j) for v in p.TRAIN_QUALIFICATION for j in REVIEW_POSITIONS], "fixed review coverage")
        p.write_once(review / "index.json", {"schema": "msu-path-b-fixed-review-v1",
            "human_technical_review": "pending", "pairs": pairs,
            "checklist": ["Correct source orientation and normal decoding", "Intended visible face in crop",
                          "Expected face context; no empty/corrupt/background-only crop",
                          "No gross detector geometry failure or orientation inversion"]})
        report = {"schema": "msu-eight-source-qualification-v1", "codec_risk_phase": "passed",
            "run_1_crops": len(first[2]["frames"]), "run_2_crops": len(second[2]["frames"]),
            "repeatability": "passed", "review_pairs": len(pairs), "human_technical_review": "pending",
            "technical_qualification": "pending_human_review", "full_preprocessing_authorized": False}
        p.write_once(root / "report.json", report)
        return report
    except BaseException as error:
        if pending is not None:
            _failure(pending, error)
        _failure(root, error)
        raise


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description="Frozen eight-source Path-B technical qualification only")
    for name in ("raw-root", "catalog", "protocol-lock", "archive", "ffmpeg", "ffprobe", "model", "output",
                 "expected-head", "git-executable"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args(argv)
    run_qualification(**vars(args))


if __name__ == "__main__":
    main()
