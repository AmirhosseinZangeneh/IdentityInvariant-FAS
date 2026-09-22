"""Provenance-preserving MSU preprocessing, separate from training and the raw lock."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from ..evaluation.identity_probe import ProbeObservation
from .msu_protocol import MSUProtocol, load_msu_protocol, verify_msu_protocol_lock

SCHEMA = "msu-processed-v2"


def json_bytes(value) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@dataclass(frozen=True)
class PreprocessingConfig:
    frames_per_video: int = 30
    margin: float = 0.0
    output_size: tuple[int, int] | None = None  # width, height; None preserves crop size

    def payload(self) -> dict:
        if type(self.frames_per_video) is not int or self.frames_per_video < 1:
            raise ValueError("frames_per_video must be a positive integer")
        if type(self.margin) not in (int, float) or not math.isfinite(self.margin) or self.margin < 0:
            raise ValueError("margin must be finite and nonnegative")
        if self.output_size is not None and (
            len(self.output_size) != 2 or any(type(v) is not int or v < 1 for v in self.output_size)
        ):
            raise ValueError("output_size must contain two positive integers")
        return {
            "schema": SCHEMA, "frames_per_video": self.frames_per_video,
            "selection": "sorted-annotated-indices; k=min(n,requested); floor(i*(n-1)/(k-1)); k=1:first",
            "margin": float(self.margin), "output_size": list(self.output_size) if self.output_size else None,
            "box_rule": "half-open; expand each side by margin*extent; floor left/top; ceil right/bottom; clip",
            "orientation": "FFmpeg; AUTO=0; explicit metadata rotation 0 or 180 only; assume annotation coordinates refer to rotated frames; alignment and geometry unverified",
            "decoder": "sequential zero-based reads; verify reported next frame index; no seeking",
            "resize_interpolation": "INTER_LINEAR", "encoding": "PNG", "png_compression": 3,
            "malformed_rows": "reject entire run", "duplicate_indices": "reject even identical rows",
            "blank_rows": "count and ignore", "missing_selected_annotations": "reject",
            "libraries": {"python": platform.python_version(), "numpy": np.__version__, "opencv": cv2.__version__,
                          "codec_versions": [s.strip() for s in cv2.getBuildInformation().splitlines()
                                             if s.strip().startswith(("FFMPEG:", "avcodec:", "avformat:", "avutil:", "swscale:"))]},
        }


@dataclass(frozen=True)
class FaceAnnotation:
    frame_index: int
    values: tuple[float, ...]  # box L,T,R,B; left eye x,y; right eye x,y
    line_number: int
    raw_line: str


class AnnotationError(ValueError):
    def __init__(self, malformed, duplicates):
        super().__init__(f"Invalid face annotations: {len(malformed)} malformed rows, {len(duplicates)} duplicate indices")
        self.malformed = malformed
        self.duplicates = duplicates


def parse_face_annotations(data: bytes) -> tuple[dict[int, FaceAnnotation], int]:
    """Parse all nine original fields; never coerce fractional frame indices or overwrite rows."""
    rows, malformed, duplicates, blanks = {}, [], [], 0
    try:
        lines = data.decode("utf-8-sig", errors="strict").splitlines()
    except UnicodeDecodeError as exc:
        raise AnnotationError([{"line": None, "reason": "invalid UTF-8", "byte_offset": exc.start}], []) from exc
    for number, raw in enumerate(lines, 1):
        if not raw.strip():
            blanks += 1
            continue
        parts = [p.strip() for p in raw.split(",")] if "," in raw else raw.split()
        try:
            if len(parts) != 9 or re.fullmatch(r"[0-9]+", parts[0]) is None:
                raise ValueError("expected integer frame index and eight coordinates")
            index, values = int(parts[0]), tuple(float(v) for v in parts[1:])
            if not all(math.isfinite(v) for v in values) or values[2] <= values[0] or values[3] <= values[1]:
                raise ValueError("nonfinite coordinates or nonpositive box")
        except ValueError as exc:
            malformed.append({"line": number, "raw_line": raw, "reason": str(exc)})
            continue
        if index in rows:
            duplicates.append({"frame_index": index, "line": number, "first_line": rows[index].line_number})
        else:
            rows[index] = FaceAnnotation(index, values, number, raw)
    if not rows and not malformed:
        malformed.append({"line": None, "raw_line": "", "reason": "no annotation rows"})
    if malformed or duplicates:
        raise AnnotationError(malformed, duplicates)
    return rows, blanks


def select_frame_indices(indices, requested: int) -> tuple[int, ...]:
    """Integer-arithmetic uniform positions in sorted annotated indices (not video duration)."""
    source = list(indices)
    if (type(requested) is not int or requested < 1 or not source
            or any(type(i) is not int or i < 0 for i in source) or len(source) != len(set(source))):
        raise ValueError("Invalid frame selection: nonempty unique nonnegative indices and positive count required")
    source.sort()
    k = min(requested, len(source))
    return tuple(source[i * (len(source) - 1) // (k - 1)] for i in range(k)) if k > 1 else (source[0],)


def crop_box(values, width: int, height: int, margin: float) -> tuple[int, int, int, int]:
    left, top, right, bottom = values[:4]
    if (not all(math.isfinite(v) for v in (left, top, right, bottom, margin)) or margin < 0
            or right <= left or bottom <= top or width < 1 or height < 1):
        raise ValueError("Invalid crop geometry")
    dx, dy = (right - left) * margin, (bottom - top) * margin
    box = (max(0, math.floor(left - dx)), max(0, math.floor(top - dy)),
           min(width, math.ceil(right + dx)), min(height, math.ceil(bottom + dy)))
    if box[2] <= box[0] or box[3] <= box[1]:
        raise ValueError("Crop has no intersection with decoded frame")
    return box


def rotate_frame(frame, degrees: int):
    if degrees not in (0, 180):
        raise ValueError("Only verified candidate rotations 0 and 180 are supported")
    return cv2.rotate(frame, cv2.ROTATE_180) if degrees else frame


class FrameDecodeError(ValueError):
    def __init__(self, missing):
        super().__init__(f"Missing selected decoded frames: {missing}")
        self.missing = missing


def decode_selected(path: Path, indices: tuple[int, ...], *, auto: bool = False):
    """Yield selected sequential frames; auto=True is exclusively a verification comparator."""
    if not indices or tuple(sorted(set(indices))) != indices or any(type(i) is not int or i < 0 for i in indices):
        raise ValueError("Invalid selected indices")
    capture = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG)
    try:
        if not capture.isOpened() or capture.getBackendName() != "FFMPEG":
            raise ValueError("FFmpeg decoder unavailable")
        if not capture.set(cv2.CAP_PROP_ORIENTATION_AUTO, int(auto)) or capture.get(cv2.CAP_PROP_ORIENTATION_AUTO) != int(auto):
            raise ValueError("Decoder orientation control unavailable")
        angle = capture.get(cv2.CAP_PROP_ORIENTATION_META)
        if angle not in (0, 180):
            raise ValueError(f"Unsupported orientation metadata: {angle}")
        wanted = set(indices)
        for index in range(indices[-1] + 1):
            ok, frame = capture.read()
            if not ok:
                raise FrameDecodeError([i for i in indices if i >= index])
            if capture.get(cv2.CAP_PROP_POS_FRAMES) != index + 1:
                raise ValueError("Decoder frame position mismatch")
            if index in wanted:
                yield index, frame, int(angle)
    finally:
        capture.release()


@dataclass(frozen=True)
class ProcessedMSUFrame:
    sample_id: str
    dataset: str
    client_id: str
    pad_partition: str
    class_label: int
    source_video_id: str
    source_filepath: str
    source_byte_size: int
    frame_index: int
    capture_device: str
    attack_type: str | None
    presentation_device: str | None
    session_id: None
    annotation_filepath: str
    annotation_sha256: str
    annotation_line_number: int
    annotation_raw_line: str
    annotation_values: tuple[float, ...]
    crop_box: tuple[int, int, int, int]
    raw_frame_size: tuple[int, int]
    oriented_frame_size: tuple[int, int]
    output_size: tuple[int, int]
    rotation_degrees: int
    crop_filepath: str
    crop_sha256: str
    preprocessing_config_digest: str

    def probe_observation(self) -> ProbeObservation:
        """Use validated frames; additional frames never become additional video groups."""
        return ProbeObservation(self.sample_id, self.client_id, self.source_video_id, None)


def sample_id(video_id: str, index: int, config_digest: str) -> str:
    return f"{video_id}:frame:{index}:prep:{config_digest}"


def _crop_path(video_id: str, index: int) -> str:
    return f"crops/{digest(video_id.encode('utf-8'))}/{index:06d}.png"


@dataclass(frozen=True)
class ProcessedMSUManifest(Sequence):
    """Validated records retain processing and fidelity status when consumed."""

    frames: tuple[ProcessedMSUFrame, ...]
    processing_status: str
    fidelity_status: str

    @property
    def experiment_ready(self) -> bool:
        # Approval evidence and its validation require a separate scientific task.
        # This schema deliberately has no supported approved state.
        return False

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, index):
        return self.frames[index]


def validate_processed_manifest(document: dict, protocol: MSUProtocol, lock_digest: str,
                                *, require_experiment_ready: bool = False) -> ProcessedMSUManifest:
    """Validate provenance against the raw protocol, never reconstruct a raw lock from crops."""
    if (document.get("schema") != SCHEMA or "status" in document
            or document.get("processing_status") != "complete"
            or document.get("fidelity_status") != "manual_review_pending"
            or document.get("experiment_ready") is not False
            or document.get("protocol_lock_digest") != lock_digest):
        raise ValueError("Processed manifest schema/status/lock mismatch")
    if require_experiment_ready:
        raise ValueError("Manifest is not experiment-ready: fidelity review is pending")
    config = document["config"]
    config_digest = digest(json_bytes(config))
    if document["config_digest"] != config_digest or config["schema"] != SCHEMA:
        raise ValueError("Preprocessing config digest mismatch")
    # Check algorithm semantics while allowing inspection on a different library installation.
    expected_config = PreprocessingConfig(config["frames_per_video"], config["margin"], config["output_size"]).payload()
    expected_config["libraries"] = config["libraries"]
    if config.get("selection") == "verification:first-two-annotated-indices":
        expected_config["selection"] = config["selection"]
    if expected_config != config:
        raise ValueError("Unsupported preprocessing configuration")
    sources = {r.video_id: r for r in protocol.recordings}
    records, seen, coverage = [], set(), {}
    for raw in document["frames"]:
        row = ProcessedMSUFrame(**{k: tuple(v) if isinstance(v, list) else v for k, v in raw.items()})
        source = sources.get(row.source_video_id)
        if source is None or type(row.frame_index) is not int or row.frame_index < 0:
            raise ValueError("Invalid source/frame index")
        if (any(type(v) is not int for v in (row.class_label, row.source_byte_size, row.rotation_degrees))
                or len(row.crop_box) != 4 or any(type(v) is not int for v in row.crop_box)
                or len(row.annotation_values) != 8
                or any(type(v) not in (int, float) for v in row.annotation_values)):
            raise ValueError("Invalid processed metadata field types")
        if row.dataset != "MSU-MFSD" or row.session_id is not None:
            raise ValueError("Invalid dataset/session metadata")
        if config["selection"] == "verification:first-two-annotated-indices" and row.pad_partition != "train":
            raise ValueError("Verification frames must come from the training cohort")
        for name, source_name in (("source_filepath", "filepath"), ("source_byte_size", "byte_size"),
                                  ("client_id", "client_id"), ("pad_partition", "pad_partition"),
                                  ("class_label", "class_label"), ("capture_device", "capture_device"),
                                  ("attack_type", "attack_type"), ("presentation_device", "presentation_device"),
                                  ("annotation_filepath", "annotation_filepath"), ("annotation_sha256", "annotation_sha256")):
            if getattr(row, name) != getattr(source, source_name):
                raise ValueError(f"Source provenance conflict: {name}")
        if (row.preprocessing_config_digest != config_digest
                or row.sample_id != sample_id(source.video_id, row.frame_index, config_digest)
                or row.crop_filepath != _crop_path(source.video_id, row.frame_index)):
            raise ValueError("Noncanonical sample/crop identifier")
        if row.sample_id in seen:
            raise ValueError("Duplicate processed frame")
        seen.add(row.sample_id)
        parsed, _ = parse_face_annotations(row.annotation_raw_line.encode("utf-8"))
        if (list(parsed) != [row.frame_index] or parsed[row.frame_index].values != row.annotation_values
                or type(row.annotation_line_number) is not int or row.annotation_line_number < 1):
            raise ValueError("Annotation provenance conflict")
        if (row.rotation_degrees not in (0, 180) or row.raw_frame_size != row.oriented_frame_size
                or any(len(size) != 2 or any(type(v) is not int or v < 1 for v in size)
                       for size in (row.raw_frame_size, row.oriented_frame_size, row.output_size))):
            raise ValueError("Invalid orientation/dimensions")
        box = crop_box(row.annotation_values, *row.oriented_frame_size, config["margin"])
        expected_size = tuple(config["output_size"] or (box[2] - box[0], box[3] - box[1]))
        if row.crop_box != box or row.output_size != expected_size or re.fullmatch(r"[0-9a-f]{64}", row.crop_sha256) is None:
            raise ValueError("Crop provenance conflict")
        coverage.setdefault(row.source_video_id, []).append(row.frame_index)
        records.append(row)
    if not records or coverage != document["selected_frame_indices"]:
        raise ValueError("Incomplete selected-frame coverage")
    return ProcessedMSUManifest(tuple(records), document["processing_status"], document["fidelity_status"])


def read_processed_manifest(path: str | Path, raw_root: str | Path, lock_path: str | Path,
                            *, require_experiment_ready: bool = False) -> ProcessedMSUManifest:
    """Verify the raw lock, source annotations, selection algorithm, and actual crop bytes."""
    path, raw_root, lock_path = Path(path), Path(raw_root), Path(lock_path)
    verify_msu_protocol_lock(raw_root, lock_path)
    protocol = load_msu_protocol(raw_root)
    document = json.loads(path.read_text(encoding="utf-8"))
    lock_digest = digest(json_bytes(json.loads(lock_path.read_text(encoding="utf-8"))))
    rows = validate_processed_manifest(document, protocol, lock_digest,
                                       require_experiment_ready=require_experiment_ready)
    annotation_cache = {}
    for row in rows:
        if row.source_video_id not in annotation_cache:
            annotations, _ = parse_face_annotations((raw_root / row.annotation_filepath).read_bytes())
            annotation_cache[row.source_video_id] = annotations
            config = document["config"]
            expected = (tuple(sorted(annotations)[:2]) if config["selection"] == "verification:first-two-annotated-indices"
                        else select_frame_indices(annotations, config["frames_per_video"]))
            if list(expected) != document["selected_frame_indices"][row.source_video_id]:
                raise ValueError("Selection does not match source annotations and configuration")
        original = annotation_cache[row.source_video_id].get(row.frame_index)
        if original != FaceAnnotation(row.frame_index, row.annotation_values, row.annotation_line_number, row.annotation_raw_line):
            raise ValueError("Annotation row differs from locked source")
        crop_bytes = (path.parent / row.crop_filepath).read_bytes()
        if digest(crop_bytes) != row.crop_sha256:
            raise ValueError("Crop content digest mismatch")
        image = cv2.imdecode(np.frombuffer(crop_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None or (image.shape[1], image.shape[0]) != row.output_size:
            raise ValueError("Crop dimensions mismatch")
    return rows


def _write_new(path: Path, payload: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(payload)


def _png(frame) -> bytes:
    ok, data = cv2.imencode(".png", frame, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    if not ok:
        raise ValueError("PNG encoding failed")
    return data.tobytes()


def audit_decoder_boundaries(root: str | Path, lock_path: str | Path, video_ids: list[str]) -> dict:
    """Controlled training-only sequential decode through EOF; no crop or annotation correction.

    Report observed endpoints separately from successful preprocessing. Limited to six
    explicitly named sources; this does not certify decoder/annotation frame alignment.
    """
    root, lock_path = Path(root), Path(lock_path)
    verify_msu_protocol_lock(root, lock_path)
    by_id = {r.video_id: r for r in load_msu_protocol(root).recordings}
    if (not video_ids or len(video_ids) > 6 or len(set(video_ids)) != len(video_ids)
            or any(v not in by_id or by_id[v].pad_partition != "train" for v in video_ids)):
        raise ValueError("Boundary review requires one to six explicit training videos")
    rows = []
    for video_id in sorted(video_ids):
        source = by_id[video_id]
        indices, _ = parse_face_annotations((root / source.annotation_filepath).read_bytes())
        capture = cv2.VideoCapture(str(root / source.filepath), cv2.CAP_FFMPEG)
        try:
            if (not capture.isOpened() or capture.getBackendName() != "FFMPEG"
                    or not capture.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)
                    or capture.get(cv2.CAP_PROP_ORIENTATION_AUTO) != 0):
                raise ValueError("Boundary decoder orientation control unavailable")
            reported = capture.get(cv2.CAP_PROP_FRAME_COUNT)
            angle = capture.get(cv2.CAP_PROP_ORIENTATION_META)
            count = 0
            while True:
                ok, _ = capture.read()
                if not ok:
                    break
                count += 1
                if capture.get(cv2.CAP_PROP_POS_FRAMES) != count:
                    raise ValueError("Decoder frame position mismatch")
        finally:
            capture.release()
        rows.append({"video_id": video_id, "reported_frame_count": reported,
                     "metadata_rotation_degrees": angle, "sequential_successful_reads": count,
                     "decoded_index_range": [0, count - 1] if count else None,
                     "annotation_index_range": [min(indices), max(indices)],
                     "annotated_indices_not_decoded": sorted(i for i in indices if i >= count),
                     "scope": "Sequential AUTO=0 reads through first failed read; no crops written; no visual inspection of intermediate frames."})
    return {"schema": "msu-decoder-boundary-review-v1",
            "protocol_lock_digest": digest(json_bytes(json.loads(lock_path.read_text(encoding="utf-8")))),
            "libraries": PreprocessingConfig().payload()["libraries"], "records": rows,
            "interpretation": "Decoder frame counts are observations, not grounds for deleting or shifting source annotations. Any unavailable selected index must fail preprocessing."}


def run_preprocessing(root: str | Path, lock_path: str | Path, output_root: str | Path,
                      config: PreprocessingConfig = PreprocessingConfig(), *, mode: str = "plan",
                      video_ids: list[str] | None = None) -> dict:
    """Verify the existing lock, plan all annotations, optionally decode/write; refuse overwrites.

    plan never decodes. verify processes the first two annotated frames of up to six
    explicitly named training videos, compares AUTO orientation, and writes review overlays.
    process uses the locked uniform sampler. Neither mode runs a model.
    """
    root, lock_path, output_root = Path(root).resolve(), Path(lock_path), Path(output_root).resolve()
    if output_root == root or root in output_root.parents or output_root in root.parents:
        raise ValueError("Output must be separate from raw data")
    if mode not in {"plan", "process", "verify"}:
        raise ValueError("Unknown preprocessing mode")
    verify_msu_protocol_lock(root, lock_path)
    protocol = load_msu_protocol(root)
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    lock_digest = digest(json_bytes(lock))
    settings = config.payload()
    by_id = {r.video_id: r for r in protocol.recordings}
    ids = sorted(by_id) if video_ids is None else sorted(video_ids)
    if not ids or len(ids) != len(set(ids)) or any(v not in by_id for v in ids):
        raise ValueError("Explicit source videos must be unique and present in the lock")
    if mode == "verify":
        if video_ids is None or len(ids) > 6 or any(by_id[v].pad_partition != "train" for v in ids):
            raise ValueError("Verification requires one to six explicit training videos")
        settings["selection"] = "verification:first-two-annotated-indices"
    config_digest = digest(json_bytes(settings))
    output_root.mkdir(parents=True, exist_ok=False)
    audit = {
        "schema": SCHEMA, "mode": mode, "status": "running", "protocol_lock_reference": lock_path.name,
        "processing_status": "not_processed", "fidelity_status": "manual_review_pending", "experiment_ready": False,
        "protocol_lock_digest": lock_digest, "source_metadata_digest": lock["normalized_metadata_sha256"],
        "config": settings, "config_digest": config_digest, "source_video_count": len(by_id),
        "source_videos_processed": [], "source_videos_not_selected": sorted(set(by_id) - set(ids)),
        "source_videos_failed": [], "selected_frame_indices": {}, "annotation_rows": 0,
        "blank_annotation_rows": 0, "malformed_annotation_rows": [], "duplicate_annotation_indices": [],
        "missing_selected_annotations": [], "missing_decoded_frames": [], "crop_write_failures": [],
        "orientation_checks": [], "orientation_status": "not_checked" if mode == "plan" else "manual_review_pending",
        "group_id_preservation_validated": None, "frames_written": 0,
        "unresolved_manual_checks": ["Confirm upright face/eye overlays and source-frame alignment.",
                                     "Confirm box endpoint convention and crop geometry; half-open is a versioned assumption.",
                                     "Controlled decoder equivalence does not establish full-release crop fidelity."],
    }
    rows, plans, current = [], [], None
    try:
        # Scan every requested annotation file before writing a crop; collect all annotation errors.
        for video_id in ids:
            current = video_id
            record = by_id[video_id]
            content = (root / record.annotation_filepath).read_bytes()
            if digest(content) != record.annotation_sha256:
                raise ValueError("Annotation drift after lock verification")
            try:
                annotations, blanks = parse_face_annotations(content)
            except AnnotationError as exc:
                audit["malformed_annotation_rows"].extend(dict(video_id=video_id, **v) for v in exc.malformed)
                audit["duplicate_annotation_indices"].extend(dict(video_id=video_id, **v) for v in exc.duplicates)
                audit["source_videos_failed"].append(video_id)
                continue
            selected = tuple(sorted(annotations)[:2]) if mode == "verify" else select_frame_indices(annotations, config.frames_per_video)
            audit["selected_frame_indices"][video_id] = list(selected)
            audit["annotation_rows"] += len(annotations)
            audit["blank_annotation_rows"] += blanks
            plans.append((record, annotations, selected))
        current = None
        if audit["source_videos_failed"]:
            raise ValueError("Annotation audit failed; no crops written")
        audit["source_videos_index_validated"] = len(plans)
        audit["planned_frames"] = sum(len(p[2]) for p in plans)
        audit["annotation_index_gaps"] = {
            r.video_id: {"first": min(a), "last": max(a), "annotated": len(a),
                         "unannotated_between_first_and_last": max(a) - min(a) + 1 - len(a)}
            for r, a, _ in plans
        }
        for field in ("pad_partition", "class_label", "client_id"):
            counts = Counter()
            for record, _, selected in plans:
                counts[str(getattr(record, field))] += len(selected)
            audit["planned_frames_by_" + field] = dict(sorted(counts.items()))
        if mode == "plan":
            audit["status"] = "index_validated_only"
            return audit
        for record, annotations, selected in plans:
            current = record.video_id
            if ((root / record.filepath).stat().st_size != record.byte_size
                    or digest((root / record.annotation_filepath).read_bytes()) != record.annotation_sha256):
                raise ValueError("Source drift after planning")
            comparator = dict((i, f) for i, f, _ in decode_selected(root / record.filepath, selected, auto=True)) if mode == "verify" else {}
            observed = set()
            for index, raw_frame, angle in decode_selected(root / record.filepath, selected):
                if index not in annotations:
                    audit["missing_selected_annotations"].append({"video_id": current, "frame_index": index})
                    raise ValueError("Missing selected annotation")
                if index not in selected or index in observed:
                    raise ValueError("Decoder returned unexpected/duplicate frame")
                observed.add(index)
                annotation = annotations[index]
                frame = rotate_frame(raw_frame, angle)
                box = crop_box(annotation.values, frame.shape[1], frame.shape[0], config.margin)
                left, top, right, bottom = box
                crop = frame[top:bottom, left:right]
                if config.output_size:
                    crop = cv2.resize(crop, config.output_size, interpolation=cv2.INTER_LINEAR)
                relative = _crop_path(current, index)
                try:
                    encoded = _png(crop)
                    _write_new(output_root / relative, encoded)
                except (OSError, ValueError, cv2.error):
                    audit["crop_write_failures"].append({"video_id": current, "frame_index": index})
                    raise
                rows.append(ProcessedMSUFrame(
                    sample_id(current, index, config_digest), "MSU-MFSD", record.client_id, record.pad_partition,
                    record.class_label, current, record.filepath, record.byte_size, index, record.capture_device,
                    record.attack_type, record.presentation_device, None, record.annotation_filepath,
                    record.annotation_sha256, annotation.line_number, annotation.raw_line, annotation.values,
                    box, (raw_frame.shape[1], raw_frame.shape[0]), (frame.shape[1], frame.shape[0]),
                    (crop.shape[1], crop.shape[0]), angle, relative, digest(encoded), config_digest))
                if mode == "verify":
                    equal = np.array_equal(frame, comparator[index])
                    audit["orientation_checks"].append({"video_id": current, "frame_index": index,
                        "metadata_rotation_degrees": angle, "auto_equals_explicit_rotation": bool(equal),
                        "raw_frame_size": list(rows[-1].raw_frame_size), "oriented_frame_size": list(rows[-1].oriented_frame_size),
                        "oriented_pixels_sha256": digest(frame.tobytes()),
                        "eyes_inside_crop_box": all(left <= annotation.values[j] < right and top <= annotation.values[j + 1] < bottom for j in (4, 6))})
                    if not equal:
                        raise ValueError("AUTO and explicit orientation disagree")
                    overlay = frame.copy()
                    cv2.rectangle(overlay, (left, top), (right - 1, bottom - 1), (0, 255, 0), 2)
                    for j in (4, 6):
                        cv2.circle(overlay, (round(annotation.values[j]), round(annotation.values[j + 1])), 3, (0, 0, 255), -1)
                    _write_new(output_root / "review" / f"{digest(current.encode('utf-8'))}_{index:06d}.png", _png(overlay))
            missing = sorted(set(selected) - observed)
            if missing:
                audit["missing_decoded_frames"].append({"video_id": current, "frame_indices": missing})
                raise ValueError("Selected frame coverage incomplete")
            audit["source_videos_processed"].append(current)
        document = {"schema": SCHEMA, "processing_status": "complete",
                    "fidelity_status": "manual_review_pending", "experiment_ready": False,
                    "config": settings, "config_digest": config_digest,
                    "protocol_lock_digest": lock_digest, "selected_frame_indices": audit["selected_frame_indices"],
                    "frames": [asdict(row) for row in rows]}
        validate_processed_manifest(document, protocol, lock_digest)
        _write_new(output_root / "manifest.json", json_bytes(document))
        audit["group_id_preservation_validated"] = True
        audit["status"] = "complete_manual_review_pending"
        audit["processing_status"] = "complete"
        return audit
    except Exception as exc:
        audit["status"] = "failed"
        audit["processing_status"] = "failed"
        audit["failure_type"] = type(exc).__name__  # avoid absolute paths in exception messages
        if isinstance(exc, FrameDecodeError):
            audit["missing_decoded_frames"].append({"video_id": current, "frame_indices": exc.missing})
        if current is not None and current not in audit["source_videos_failed"]:
            audit["source_videos_failed"].append(current)
        audit["partial_frames"] = [asdict(row) for row in rows]
        raise
    finally:
        audit["frames_written"] = len(rows)
        audit["source_video_count_processed"] = len(audit["source_videos_processed"])
        audit["source_videos_not_processed"] = sorted(set(ids) - set(audit["source_videos_processed"]) - set(audit["source_videos_failed"]))
        for field in ("pad_partition", "class_label", "client_id"):
            audit["frames_by_" + field] = dict(sorted(Counter(str(getattr(row, field)) for row in rows).items()))
        _write_new(output_root / "audit.json", json_bytes(audit))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--protocol-lock", default="docs/audit/msu_protocol_lock.json")
    parser.add_argument("--output-root", required=True, help="Fresh directory outside raw data; never overwritten")
    parser.add_argument("--mode", choices=("plan", "process", "verify", "boundary"), default="plan")
    parser.add_argument("--frames-per-video", type=int, default=30)
    parser.add_argument("--margin", type=float, default=0.0)
    parser.add_argument("--output-size", type=int, nargs=2, metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--video-id", action="append")
    args = parser.parse_args()
    if args.mode == "boundary":
        output, root = Path(args.output_root).resolve(), Path(args.root).resolve()
        if output == root or root in output.parents or output in root.parents:
            parser.error("Output must be separate from raw data")
        if output.exists():
            parser.error("Output directory already exists")
        result = audit_decoder_boundaries(root, args.protocol_lock, args.video_id)
        output.mkdir(parents=True, exist_ok=False)
        _write_new(output / "audit.json", json_bytes(result))
        print("Boundary observations written; this is not a successful preprocessing run.")
        return
    result = run_preprocessing(args.root, args.protocol_lock, args.output_root,
                               PreprocessingConfig(args.frames_per_video, args.margin, tuple(args.output_size) if args.output_size else None),
                               mode=args.mode, video_ids=args.video_id)
    print(f"{result['status']}: {result['frames_written']} crops written")


if __name__ == "__main__":
    main()
