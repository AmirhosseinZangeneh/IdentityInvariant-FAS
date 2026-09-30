"""Prospective Path B contracts. No decoder invocation, downloads, or annotation I/O.

Pure validation is independent of installed production runtime versions. Real detector
construction is restricted to a fresh spawned worker; importing this module imports no cv2.
Subprocess transport/ffprobe JSON adaptation remains a separate qualification concern.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path, PurePosixPath
import platform
import re
import sys
import zipfile

import numpy as np
from PIL import Image

POLICY = "msu-publication-path-b-v1"
SCHEMA = "msu-publication-frames-v1"
TRAIN = tuple("002 003 005 006 007 008 009 011 012 021 022 034 053 054 055".split())
TEST = tuple("001 013 014 023 024 026 028 029 030 032 033 035 036 037 039 042 048 049 050 051".split())
MEDIA = {"ipad_video": "iPad Air", "iphone_video": "iPhone 5S", "printed_photo": "printed paper"}
CAMERAS = {"android": "Google Nexus 5 front-facing camera", "laptop": "MacBook Air built-in camera"}
ARCHIVE_SHA256 = "db580001caa24ac104c8cb856cd113a87b0a443f7bdf47d8c12b1d740584a2ec"
MODEL_SHA256 = "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"
MODEL_NAME = "face_detection_yunet_2023mar.onnx"
RUNTIME = (("Python", "3.12.10"), ("opencv-python", "4.11.0.86"),
           ("numpy", "2.5.2"), ("Pillow", "12.3.0"))
DETECTOR = {"model_sha256": MODEL_SHA256, "input_size": [640, 640],
            "backend": 3, "target": 0, "engine": "opencv-4.11-native-dnn", "onnx_runtime": False,
            "score_threshold": 0.90, "nms_threshold": 0.30, "top_k": 5000}

SCORE_THRESHOLD_F32 = float(np.float32(DETECTOR["score_threshold"]))


class ContractError(ValueError):
    """A required condition failed; no fallback or successful output is allowed."""


def _require(condition, reason):
    if not condition:
        raise ContractError(reason)


def canonical_bytes(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def digest(value):
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        _require(key not in result, "duplicate JSON key")
        result[key] = value
    return result


def read_json(path):
    def invalid(value):
        raise ContractError("nonfinite JSON: " + value)
    return json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=_pairs,
                      parse_constant=invalid)


def _sha(value):
    _require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value), "invalid SHA-256")
    return value


def stream_sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def relative_path(value):
    _require(isinstance(value, str) and value and "\\" not in value and ":" not in value,
             "nonportable/absolute path")
    p = PurePosixPath(value)
    _require(not p.is_absolute() and p.as_posix() == value and
             all(x not in ("", ".", "..") for x in value.split("/")), "unsafe relative path")
    return value


def _inside(root, relative):
    path = Path(root).resolve() / relative_path(relative)
    _require(path.resolve().is_relative_to(Path(root).resolve()), "path escapes root")
    return path


@dataclass(frozen=True)
class Source:
    filepath: str
    client_id: str
    pad_partition: str
    class_label: int
    capture_device: str
    capture_model: str
    attack_type: str | None
    presentation_device: str | None
    resolution_token: str
    scenario: str
    byte_size: int
    video_id: str
    session_id: None = None


def _validate_source(row):
    relative_path(row.filepath)
    attack = row.attack_type
    _require(row.client_id in (*TRAIN, *TEST) and row.capture_device in CAMERAS,
             "unsupported client/camera")
    _require(attack is None or attack in MEDIA, "unsupported attack")
    kind = "real" if attack is None else "attack"
    middle = "" if attack is None else "_" + attack
    ext = "mp4" if row.capture_device == "android" else "mov"
    path = f"scene01/{kind}/{kind}_client{row.client_id}_{row.capture_device}_SD{middle}_scene01.{ext}"
    _require(row.filepath == path and row.video_id == "MSU-MFSD:" + path,
             "canonical source path/ID conflict")
    _require(row.pad_partition == ("train" if row.client_id in TRAIN else "test") and
             type(row.class_label) is int and row.class_label == int(attack is not None) and
             row.capture_model == CAMERAS[row.capture_device] and
             row.presentation_device == MEDIA.get(attack) and row.resolution_token == "SD" and
             row.scenario == "scene01" and row.session_id is None and
             type(row.byte_size) is int and row.byte_size > 0, "source metadata conflict")


@dataclass(frozen=True)
class SourceCatalog:
    sources: tuple[Source, ...]
    catalog_sha256: str
    protocol_lock_sha256: str
    evidence: tuple[tuple[str, str], ...]

    @property
    def projection_sha256(self):
        return digest([asdict(s) for s in self.sources])


def load_source_catalog(raw_root, catalog_path, lock_path):
    """Read catalog/list/README metadata and stat videos, never annotations or media bytes."""
    catalog, lock = read_json(catalog_path), read_json(lock_path)
    _require(digest(catalog) == lock["normalized_metadata_sha256"], "catalog digest mismatch")
    _require(catalog["schema_version"] == 1 and catalog["dataset"] == "MSU-MFSD" and
             catalog["protocol"] == "official_public_train_test", "catalog schema")
    _require(catalog["train_clients"] == list(TRAIN) and catalog["test_clients"] == list(TEST),
             "official client sets")
    for split, clients in (("train", TRAIN), ("test", TEST)):
        _require(lock["partitions"][split]["client_ids"] == list(clients), "lock cohort mismatch")
    fingerprints = catalog["source_fingerprints"]
    _require(fingerprints == lock["source_fingerprints"], "source evidence binding")
    evidence = []
    for name in ("README.txt", "train_sub_list.txt", "test_sub_list.txt"):
        path = _inside(raw_root, name)
        _require(stream_sha256(path) == _sha(fingerprints[name]), "source evidence drift")
        evidence.append((name, fingerprints[name]))
        if name != "README.txt":
            tokens = path.read_text(encoding="utf-8-sig").split()
            _require(all(re.fullmatch(r"[0-9]{1,3}", t) for t in tokens), "invalid client list")
            ids = sorted(f"{int(t):03d}" for t in tokens)
            _require(ids == list(TRAIN if name.startswith("train") else TEST), "client list drift")
    rows, paths, physical, slots = [], set(), set(), set()
    for item in catalog["recordings"]:
        row = Source(**{k: item[k] for k in Source.__dataclass_fields__})
        _validate_source(row)
        _require(row.filepath not in paths, "duplicate relative path")
        paths.add(row.filepath)
        path = _inside(raw_root, row.filepath)
        _require(path.is_file() and path.stat().st_size == row.byte_size, "source missing/size drift")
        stat = path.stat()
        key = (stat.st_dev, stat.st_ino) if stat.st_ino else str(path.resolve()).casefold()
        _require(key not in physical, "physical alias")
        physical.add(key)
        slot = row.client_id, row.capture_device, row.attack_type
        _require(slot not in slots, "duplicate acquisition slot")
        slots.add(slot)
        rows.append(row)
    expected = {(c, camera, a) for c in (*TRAIN, *TEST) for camera in CAMERAS for a in (None, *MEDIA)}
    _require(len(rows) == 280 and slots == expected, "incomplete 280-video inventory")
    return SourceCatalog(tuple(sorted(rows, key=lambda r: r.filepath)), digest(catalog),
                         stream_sha256(lock_path), tuple(evidence))


def source_snapshot(raw_root, catalog):
    """Explicit opt-in media hashing; callers recheck this snapshot after processing."""
    records, seen = [], set()
    for row in catalog.sources:
        path = _inside(raw_root, row.filepath)
        before = path.stat()
        sha = stream_sha256(path)
        after = path.stat()
        _require(before.st_size == after.st_size == row.byte_size and
                 before.st_mtime_ns == after.st_mtime_ns, "source changed during hashing")
        _require(sha not in seen, "duplicate source content")
        seen.add(sha)
        records.append({"source": asdict(row), "sha256": sha})
    return {"projection_sha256": catalog.projection_sha256, "sources": records}


def verify_snapshot(raw_root, catalog, frozen):
    _require(source_snapshot(raw_root, catalog) == frozen, "source snapshot drift")


def verify_file(path, sha256, size=None):
    path = Path(path)
    _require(path.is_file() and (size is None or path.stat().st_size == size), "artifact size/missing")
    _require(stream_sha256(path) == _sha(sha256), "artifact hash mismatch")


@dataclass(frozen=True)
class EnvironmentLock:
    archive_sha256: str
    model_sha256: str
    ffmpeg_sha256: str
    ffprobe_sha256: str
    versions: tuple[tuple[str, str], ...]
    ffmpeg_version_output: str
    ffprobe_version_output: str
    opencv_build: str
    runtime_fingerprints: tuple[tuple[str, str], ...]

    def __post_init__(self):
        for value in (self.versions, self.runtime_fingerprints):
            _require(type(value) is tuple and all(type(x) is tuple and len(x) == 2 and
                     all(isinstance(v, str) for v in x) for x in value), "immutable lock entries required")

    def validate(self):
        _require(self.archive_sha256 == ARCHIVE_SHA256 and self.model_sha256 == MODEL_SHA256,
                 "unapproved artifact")
        _require(self.versions == RUNTIME, "unqualified runtime versions")
        for sha in (self.ffmpeg_sha256, self.ffprobe_sha256):
            _sha(sha)
        for tool, output in (("ffmpeg", self.ffmpeg_version_output), ("ffprobe", self.ffprobe_version_output)):
            _require(re.match(tool + r" version 8\.1\.2(?:[-\s])", output) is not None and
                     "configuration:" in output and "libavcodec" in output and "libavformat" in output,
                     "incomplete/wrong executable build evidence")
        validate_opencv_build(self.opencv_build)
        _require(tuple(n for n, _ in self.runtime_fingerprints) == tuple(n for n, _ in RUNTIME),
                 "runtime binary fingerprints required")
        for _, sha in self.runtime_fingerprints:
            _sha(sha)


def validate_artifact_files(lock, archive, model, ffmpeg, ffprobe):
    """Bind executables to the approved ZIP, without running or downloading anything."""
    lock.validate()
    verify_file(archive, ARCHIVE_SHA256)
    verify_file(model, MODEL_SHA256, 232589)
    for path, sha in ((ffmpeg, lock.ffmpeg_sha256), (ffprobe, lock.ffprobe_sha256)):
        _absolute(path)
        verify_file(path, sha)
    with zipfile.ZipFile(archive) as zipped:
        for name, sha in (("ffmpeg.exe", lock.ffmpeg_sha256), ("ffprobe.exe", lock.ffprobe_sha256)):
            entries = [n for n in zipped.namelist() if n.endswith("/bin/" + name)]
            _require(len(entries) == 1, "ambiguous archive executable")
            h = hashlib.sha256()
            with zipped.open(entries[0]) as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    h.update(chunk)
            _require(h.hexdigest() == sha, "executable not from pinned archive")


def _absolute(path):
    _require(Path(path).is_absolute(), "explicit absolute executable/source path required")
    return str(path)


def build_decode_command(executable, source, codec):
    _require(codec in ("h264", "prores"), "unsupported codec")
    return (_absolute(executable), "-nostdin", "-hide_banner", "-loglevel", "warning",
            "-nostats", "-xerror", "-hwaccel", "none", "-threads", "1", "-c:v", codec,
            "-err_detect", "explode", "-noautorotate", "-i", _absolute(source),
            "-map", "0:v:0", "-an", "-sn", "-dn", "-filter_threads", "1",
            "-fps_mode", "passthrough", "-c:v", "ppm", "-threads", "1",
            "-pix_fmt", "rgb24", "-f", "image2pipe", "pipe:1")


def build_probe_command(executable, source, codec):
    _require(codec in ("h264", "prores"), "unsupported codec")
    # ffprobe has no ffmpeg -xerror/-fps_mode controls. The result validator rejects
    # every diagnostic and requires a successful complete full-frame metadata scan.
    return (_absolute(executable), "-hide_banner", "-loglevel", "warning", "-threads", "1",
            "-codec:v", codec, "-err_detect", "explode", "-show_streams", "-show_frames",
            "-of", "json", _absolute(source))


def validate_command(command, executable, source, codec, *, probe=False):
    builder = build_probe_command if probe else build_decode_command
    _require(tuple(command) == builder(executable, source, codec), "command contract mismatch")


def portable_commands(codec, source_relative):
    """Retain exact options with portable executable references, never host paths."""
    relative_path(source_relative)
    source = str(Path.cwd() / "__source__")
    result = {}
    for name, builder in (("ffmpeg", build_decode_command), ("ffprobe", build_probe_command)):
        exe = str(Path.cwd() / ("__" + name + "__"))
        result[name] = [name if x == exe else source_relative if x == source else x
                        for x in builder(exe, source, codec)]
    return result


class PPMParser:
    """Incremental parser of FFmpeg's P6\\nW H\\n255\\n payload; no comments/variants."""
    MAX_FRAME_BYTES = 64 * 1024 * 1024  # Safety ceiling, not a resize policy.
    MAX_HEADER_BYTES = 80

    def __init__(self, expected_size=None, max_frame_bytes=MAX_FRAME_BYTES):
        _require(type(max_frame_bytes) is int and 0 < max_frame_bytes <= self.MAX_FRAME_BYTES,
                 "invalid PPM payload bound")
        self.expected_size = expected_size
        self.max_frame_bytes = max_frame_bytes
        if expected_size is not None:
            _require(len(expected_size) == 2 and all(type(x) is int and x > 0 for x in expected_size),
                     "invalid expected PPM geometry")
            w, h = expected_size
            _require(w <= max_frame_bytes // 3 // h, "PPM payload exceeds bound")
            self.max_frame_bytes = w * h * 3
        self.buffer = bytearray()
        self.shape = None
        self.closed = False
        self.header_newlines = 0

    def feed(self, chunk):
        _require(not self.closed and isinstance(chunk, bytes), "invalid parser feed")
        frames, offset = [], 0
        while offset < len(chunk):
            if self.shape is None:
                # Examine only header bytes; never scan buffered RGB payload.
                byte = chunk[offset]
                offset += 1
                self.buffer.append(byte)
                self.header_newlines += byte == 10
                _require(len(self.buffer) <= self.MAX_HEADER_BYTES, "malformed PPM header")
                if self.header_newlines < 3:
                    continue
                match = re.fullmatch(rb"P6\n([1-9][0-9]*) ([1-9][0-9]*)\n255\n", self.buffer)
                _require(match is not None, "malformed PPM header")
                w, h = map(int, match.groups())
                _require(w <= self.max_frame_bytes // 3 // h, "PPM payload exceeds bound")
                _require(self.expected_size is None or (w, h) == tuple(self.expected_size),
                         "PPM geometry mismatch")
                self.shape = h, w, 3
                self.buffer.clear()
                self.header_newlines = 0
            count = math.prod(self.shape)
            take = min(count - len(self.buffer), len(chunk) - offset)
            self.buffer.extend(memoryview(chunk)[offset:offset + take])
            offset += take
            if len(self.buffer) == count:
                frames.append(np.frombuffer(bytes(self.buffer), dtype=np.uint8).reshape(self.shape).copy())
                self.buffer.clear()
                self.shape = None
        return frames

    def finish(self):
        _require(not self.closed and not self.buffer and self.shape is None, "truncated PPM stream")
        self.closed = True


def parse_ppm(chunks, *, expected_size=None, max_frame_bytes=PPMParser.MAX_FRAME_BYTES):
    parser = PPMParser(expected_size, max_frame_bytes)
    for chunk in chunks:
        yield from parser.feed(chunk)
    parser.finish()


def normalize_orientation(rotations=(), matrix=None):
    """Signed CCW degrees. Optional matrix is normalized 3x3 Cartesian CCW, not raw fixed-point."""
    turns = []
    for value in rotations:
        _require(type(value) in (int, float) and math.isfinite(value), "nonfinite rotation")
        quarter = round(value / 90)
        _require(abs(value - quarter * 90) <= 1e-6, "non-quarter-turn rotation")
        turns.append(quarter % 4)
    if matrix is not None:
        a = np.asarray(matrix, dtype=np.float64)
        _require(a.shape == (3, 3) and np.isfinite(a).all(), "invalid orientation matrix")
        candidates = (((1, 0), (0, 1)), ((0, -1), (1, 0)),
                      ((-1, 0), (0, -1)), ((0, 1), (-1, 0)))
        matches = []
        for k, block in enumerate(candidates):
            expected = np.eye(3)
            expected[:2, :2] = block
            if np.allclose(a, expected, rtol=0, atol=1e-8):
                matches.append(k)
        _require(len(matches) == 1, "reflection/skew/scaling/unsupported matrix")
        turns.extend(matches)
    _require(len(set(turns)) <= 1, "conflicting rotation sources")
    return (turns[0] if turns else 0) * 90


def _rgb(image):
    _require(isinstance(image, np.ndarray) and image.dtype == np.uint8 and image.ndim == 3 and
             image.shape[2] == 3 and min(image.shape[:2]) > 0, "expected RGB uint8 HxWx3")


def orient_rgb(image, degrees):
    _rgb(image)
    rotation = normalize_orientation((degrees,))
    return np.ascontiguousarray(np.rot90(image, rotation // 90))


def pixel_sha256(image):
    _rgb(image)
    return hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest()


def sample_ordinals(n):
    _require(type(n) is int and n >= 30, "at least 30 complete frames required")
    return tuple(j * (n - 1) // 29 for j in range(30))


@dataclass(frozen=True)
class DecodeResult:
    video_id: str
    source_sha256: str
    snapshot_sha256: str
    executable_sha256: str
    probe_executable_sha256: str
    codec: str
    source_size: tuple[int, int]
    rotations: tuple[float, ...]
    matrix: tuple | None
    frame_hashes: tuple[str, ...]
    oriented_size: tuple[int, int]
    n: int
    stderr: str
    returncode: int
    complete: bool
    probe_stderr: str
    probe_returncode: int
    probe_complete: bool
    probe_frame_sizes: tuple[tuple[int, int], ...]
    progressive: bool
    sample_aspect_ratio: tuple[int, int]
    video_stream_count: int
    attached_picture: bool
    color_metadata: tuple[tuple[str, str | None], ...]

    def validate(self):
        _require(self.video_id.startswith("MSU-MFSD:"), "video ID")
        relative_path(self.video_id.removeprefix("MSU-MFSD:"))
        _sha(self.source_sha256)
        _sha(self.snapshot_sha256)
        _sha(self.executable_sha256)
        _sha(self.probe_executable_sha256)
        _require(tuple(k for k, _ in self.color_metadata) ==
                 ("range", "matrix", "primaries", "transfer") and
                 all(v is None or isinstance(v, str) for _, v in self.color_metadata), "color metadata")
        sample_ordinals(self.n)
        _require(self.codec in ("h264", "prores") and self.complete is True and
                 self.probe_complete is True and type(self.returncode) is int and self.returncode == 0 and
                 type(self.probe_returncode) is int and self.probe_returncode == 0 and
                 not self.stderr.strip() and not self.probe_stderr.strip(), "decode/probe failure")
        w, h = self.source_size
        _require(type(w) is int and type(h) is int and min(w, h) > 0, "invalid dimensions")
        angle = normalize_orientation(self.rotations, self.matrix)
        expected = (h, w) if angle % 180 else (w, h)
        _require(self.oriented_size == expected and len(self.frame_hashes) == self.n and
                 self.probe_frame_sizes == (self.source_size,) * self.n, "decode count/dimension mismatch")
        _require(self.progressive is True and self.sample_aspect_ratio == (1, 1) and
                 type(self.video_stream_count) is int and self.video_stream_count == 1 and
                 self.attached_picture is False, "unsupported stream layout")
        for sha in self.frame_hashes:
            _sha(sha)


def validate_decoded_frames(result, frames):
    result.validate()
    hashes = []
    for frame in frames:
        _rgb(frame)
        _require((frame.shape[1], frame.shape[0]) == result.source_size, "inconsistent PPM dimensions")
        hashes.append(pixel_sha256(orient_rgb(frame, normalize_orientation(result.rotations, result.matrix))))
    _require(tuple(hashes) == result.frame_hashes, "pixel hashes/count mismatch")


@dataclass(frozen=True)
class Letterbox:
    width: int
    height: int
    resized_width: int
    resized_height: int
    left: int
    top: int
    right: int
    bottom: int


def letterbox_geometry(width, height):
    _require(type(width) is int and type(height) is int and min(width, height) > 0, "invalid image size")
    m = max(width, height)
    wr, hr = (max(1, (1280 * x + m) // (2 * m)) for x in (width, height))
    left, top = (640 - wr) // 2, (640 - hr) // 2
    return Letterbox(width, height, wr, hr, left, top, 640 - wr - left, 640 - hr - top)


def _box(box, width, height):
    a = np.asarray(box, dtype=np.float64)
    _require(a.shape == (4,) and np.isfinite(a).all(), "invalid box")
    x, y, w, h = map(float, a)
    _require(w > 0 and h > 0 and x < width and y < height and x + w > 0 and y + h > 0,
             "box does not intersect original frame")
    return x, y, w, h


def remap_detection(box, geometry):
    _require(geometry == letterbox_geometry(geometry.width, geometry.height), "invalid letterbox")
    a = np.asarray(box, dtype=np.float64)
    _require(a.shape == (4,) and np.isfinite(a).all(), "invalid detector box")
    xd, yd, wd, hd = a
    g = geometry
    return _box(((xd-g.left)*g.width/g.resized_width, (yd-g.top)*g.height/g.resized_height,
                 wd*g.width/g.resized_width, hd*g.height/g.resized_height), g.width, g.height)


def single_detection(detections):
    _require(detections is not None and len(detections) == 1, "exactly one post-NMS face required")
    row = np.asarray(detections[0], dtype=np.float64)
    _require(row.shape == (15,) and np.isfinite(row).all() and SCORE_THRESHOLD_F32 <= row[14] <= 1,
             "invalid YuNet detection")
    return tuple(map(float, row[:4])), float(row[14])


def crop_bounds(box, width, height):
    x, y, w, h = _box(box, width, height)
    continuous = (x - .1*w, y - .1*h, x + 1.1*w, y + 1.1*h)
    _require(all(math.isfinite(v) for v in continuous), "nonfinite expanded bounds")
    bounds = (max(0, math.floor(continuous[0])), max(0, math.floor(continuous[1])),
              min(width, math.ceil(continuous[2])), min(height, math.ceil(continuous[3])))
    _require(bounds[0] < bounds[2] and bounds[1] < bounds[3], "empty crop")
    return continuous, bounds


def save_crop(original_rgb, box, output):
    _rgb(original_rgb)
    _, (left, top, right, bottom) = crop_bounds(box, original_rgb.shape[1], original_rgb.shape[0])
    cropped = Image.fromarray(original_rgb[top:bottom, left:right]).resize(
        (160, 160), Image.Resampling.BILINEAR, reducing_gap=None)
    expected = pixel_sha256(np.asarray(cropped))
    with Path(output).open("xb") as stream:
        cropped.save(stream, format="PNG")
        stream.flush()
        os.fsync(stream.fileno())
    metadata = inspect_png(output)
    _require(metadata["pixel_sha256"] == expected, "PNG reload mismatch")
    return metadata


def inspect_png(path):
    with Image.open(path) as image:
        _require(image.format == "PNG" and image.mode == "RGB" and image.size == (160, 160),
                 "invalid publication PNG")
        pixels = np.asarray(image)
        return {"png_sha256": stream_sha256(path), "pixel_sha256": pixel_sha256(pixels),
                "size": [160, 160], "mode": "RGB", "dtype": "uint8"}


def detector_worker_environment(parent):
    """Pass this environment to a fresh interpreter; never patch an already imported cv2."""
    # This OpenCV-5-only switch has no applicable semantics in the pinned 4.11 runtime.
    return {k: v for k, v in parent.items() if k != "OPENCV_FORCE_DNN_ENGINE"}


def validate_opencv_build(build):
    _require("General configuration for OpenCV 4.11.0" in build and
             not re.search(r"ONNX Runtime:\s+(?!NO\b)\S+", build),
             "wrong OpenCV build or ONNX Runtime prohibited")


def validate_cv2_runtime(cv2):
    _require(cv2.__version__ == "4.11.0", "unqualified cv2 runtime version")
    validate_opencv_build(cv2.getBuildInformation())
    _require(cv2.dnn.DNN_BACKEND_OPENCV == 3 and cv2.dnn.DNN_TARGET_CPU == 0,
             "unexpected DNN backend/target")


def create_detector_in_worker(model):
    """Only a fresh interpreter may call this; no invocation occurs on module import.

    The future launcher must pass detector_worker_environment to a new process.
    Even a correctly set environment cannot rescue an already-imported cv2.
    """
    _require("cv2" not in sys.modules, "cv2 already imported; fresh worker required")
    _require("OPENCV_FORCE_DNN_ENGINE" not in os.environ, "inapplicable OpenCV-5 engine setting")
    actual = (("Python", platform.python_version()),) + tuple(
        (name, importlib.metadata.version(name)) for name, _ in RUNTIME[1:])
    _require(actual == RUNTIME, "unqualified runtime versions")
    verify_file(model, MODEL_SHA256, 232589)
    import cv2
    validate_cv2_runtime(cv2)
    _require(not any(n == "onnxruntime" or n.startswith("onnxruntime.") for n in sys.modules),
             "ONNX Runtime prohibited")
    cv2.setNumThreads(1)
    cv2.ocl.setUseOpenCL(False)
    detector = cv2.FaceDetectorYN.create(str(model), "", (640, 640), .90, .30, 5000,
                                       cv2.dnn.DNN_BACKEND_OPENCV, cv2.dnn.DNN_TARGET_CPU)
    _require(detector.getInputSize() == (640, 640) and
             detector.getScoreThreshold() == float(np.float32(.90)) and
             detector.getNMSThreshold() == float(np.float32(.30)) and
             detector.getTopK() == 5000, "detector settings mismatch")

    def detect(rgb):
        _rgb(rgb)
        g = letterbox_geometry(rgb.shape[1], rgb.shape[0])
        bgr = np.ascontiguousarray(rgb[:, :, ::-1])
        if (g.width, g.height) != (g.resized_width, g.resized_height):
            bgr = cv2.resize(bgr, (g.resized_width, g.resized_height), interpolation=cv2.INTER_LINEAR)
        padded = cv2.copyMakeBorder(bgr, g.top, g.bottom, g.left, g.right,
                                    cv2.BORDER_CONSTANT, value=(0, 0, 0))
        return detector.detect(padded)[1]
    return detect


def _video(client, camera, attack=None):
    kind = "real" if attack is None else "attack"
    middle = "" if attack is None else "_" + attack
    ext = "mp4" if camera == "android" else "mov"
    return f"MSU-MFSD:scene01/{kind}/{kind}_client{client}_{camera}_SD{middle}_scene01.{ext}"


TRAIN_QUALIFICATION = tuple(_video(*args) for args in (
    ("005", "laptop", "iphone_video"), ("007", "laptop", "ipad_video"),
    ("008", "laptop", "ipad_video"), ("053", "laptop", "ipad_video"),
    ("002", "android"), ("003", "android"), ("002", "laptop"),
    ("002", "android", "printed_photo")))
TRAIN_CODEC_GATE = TRAIN_QUALIFICATION[:4]
TEST_CODEC_CHECK = tuple(_video(c, "laptop", a) for c, a in (
    ("023", "iphone_video"), ("028", "printed_photo"),
    ("049", "printed_photo"), ("051", "printed_photo")))


def validate_qualification_scope(video_ids):
    _require(tuple(video_ids) == TRAIN_QUALIFICATION, "exact ordered eight training sources required")


def write_once(path, document):
    payload = canonical_bytes(document)
    with Path(path).open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def fresh_output(raw_root, output_root):
    raw, out = Path(raw_root).resolve(), Path(output_root).resolve()
    _require(not raw.is_relative_to(out) and not out.is_relative_to(raw), "raw/output roots overlap")
    out.mkdir(parents=True, exist_ok=False)
    write_once(out / "state.json", {"schema": SCHEMA, "state": "partial"})
    return out


def record_failure(output_root, reason):
    root = Path(output_root)
    _require(not (root / "manifest.json").exists() and not (root / "completion.json").exists(),
             "completed output cannot be modified")
    _require(read_json(root / "state.json") == {"schema": SCHEMA, "state": "partial"}, "not a Path B output")
    write_once(root / "failure.json", {"schema": SCHEMA, "state": "failed", "reason": str(reason)})


def make_frame_record(source, source_sha256, decoded, ordinal, original_rgb, detections,
                      output_root, output_relative):
    """Materialize one verified crop; this helper runs no decoder or detector."""
    _validate_source(source)
    decoded.validate()
    _sha(source_sha256)
    _require(source_sha256 == decoded.source_sha256, "decode source hash mismatch")
    _require(source.video_id == decoded.video_id and ordinal in sample_ordinals(decoded.n),
             "frame/source binding")
    _rgb(original_rgb)
    _require((original_rgb.shape[1], original_rgb.shape[0]) == decoded.oriented_size and
             pixel_sha256(original_rgb) == decoded.frame_hashes[ordinal], "oriented frame binding")
    g = letterbox_geometry(*decoded.oriented_size)
    raw_box, score = single_detection(detections)
    mapped = remap_detection(raw_box, g)
    continuous, bounds = crop_bounds(mapped, *decoded.oriented_size)
    _require(read_json(Path(output_root) / "state.json") == {"schema": SCHEMA, "state": "partial"} and
             not any((Path(output_root) / x).exists() for x in
                     ("failure.json", "manifest.json", "completion.json")), "output is not writable")
    path = _inside(output_root, output_relative)
    _require(path.suffix == ".png", "PNG output required")
    path.parent.mkdir(parents=True, exist_ok=True)
    output = save_crop(original_rgb, mapped, path)
    return json.loads(canonical_bytes({
        "schema": SCHEMA, "dataset": "MSU-MFSD", "policy": POLICY,
        "source": asdict(source), "source_sha256": source_sha256,
        "group_id": source.video_id, "session_id": None, "n": decoded.n, "ordinal": ordinal,
        "orientation": {"rotations": decoded.rotations, "matrix": decoded.matrix,
                        "applied_degrees": normalize_orientation(decoded.rotations, decoded.matrix)},
        "source_size": decoded.source_size, "oriented_size": decoded.oriented_size,
        "oriented_rgb_sha256": decoded.frame_hashes[ordinal], "detector": dict(DETECTOR),
        "letterbox": asdict(g), "detector_box": raw_box, "score": score,
        "mapped_box": mapped, "expanded_bounds": continuous, "crop_bounds": bounds,
        "output_path": output_relative, "output": output}))


def _decode_document(value):
    """Exact dataclass fields, canonical nested tuple representation."""
    fields = set(DecodeResult.__dataclass_fields__)
    _require(isinstance(value, dict) and set(value) == fields, "decode schema")
    d = dict(value)
    for key in ("source_size", "rotations", "frame_hashes", "oriented_size", "sample_aspect_ratio"):
        d[key] = tuple(d[key])
    for key in ("probe_frame_sizes", "color_metadata"):
        d[key] = tuple(tuple(x) for x in d[key])
    if d["matrix"] is not None:
        d["matrix"] = tuple(tuple(x) for x in d["matrix"])
    result = DecodeResult(**d)
    result.validate()
    return result


def _strict_equal(actual, expected, reason):
    _require(canonical_bytes(actual) == canonical_bytes(expected), reason)


def _snapshot_sources(snapshot, catalog):
    _require(set(snapshot) == {"projection_sha256", "sources"} and
             snapshot["projection_sha256"] == catalog.projection_sha256, "snapshot catalog binding")
    rows = snapshot["sources"]
    _require(len(rows) == len(catalog.sources) == 280, "snapshot coverage")
    hashes = {}
    for item, source in zip(rows, catalog.sources):
        _require(set(item) == {"source", "sha256"}, "snapshot schema")
        _strict_equal(item["source"], asdict(source), "snapshot source mismatch")
        hashes[source.video_id] = _sha(item["sha256"])
    _require(len(set(hashes.values())) == 280, "duplicate source content")
    return hashes


def _frame_validate(frame, source, sha, decoded, root):
    fields = {"schema", "dataset", "policy", "source", "source_sha256", "group_id", "session_id",
              "n", "ordinal", "orientation", "source_size", "oriented_size", "oriented_rgb_sha256",
              "detector", "letterbox", "detector_box", "score", "mapped_box", "expanded_bounds",
              "crop_bounds", "output_path", "output"}
    _require(set(frame) == fields, "frame schema (no annotation/extra fields)")
    _require(frame["schema"] == SCHEMA and frame["dataset"] == "MSU-MFSD" and
             frame["policy"] == POLICY and frame["session_id"] is None and
             frame["group_id"] == source.video_id and frame["source_sha256"] == sha,
             "frame identity binding")
    _strict_equal(frame["source"], asdict(source), "frame source metadata")
    ordinal = frame["ordinal"]
    _require(type(ordinal) is int and ordinal in sample_ordinals(decoded.n) and
             type(frame["n"]) is int and frame["n"] == decoded.n and
             frame["oriented_rgb_sha256"] == decoded.frame_hashes[ordinal], "sample/hash binding")
    for key in ("source_size", "oriented_size"):
        _strict_equal(frame[key], getattr(decoded, key), "frame dimensions")
    _strict_equal(frame["orientation"], {"rotations": decoded.rotations, "matrix": decoded.matrix,
                  "applied_degrees": normalize_orientation(decoded.rotations, decoded.matrix)}, "orientation binding")
    _strict_equal(frame["detector"], DETECTOR, "detector contract")
    _require(type(frame["score"]) in (float, int) and math.isfinite(frame["score"]) and
             SCORE_THRESHOLD_F32 <= frame["score"] <= 1, "detection score")
    geometry = letterbox_geometry(*decoded.oriented_size)
    _strict_equal(frame["letterbox"], asdict(geometry), "letterbox mismatch")
    mapped = remap_detection(frame["detector_box"], geometry)
    continuous, bounds = crop_bounds(mapped, *decoded.oriented_size)
    for key, value in (("mapped_box", mapped), ("expanded_bounds", continuous), ("crop_bounds", bounds)):
        _strict_equal(frame[key], value, "crop geometry mismatch")
    relative_path(frame["output_path"])
    _require(frame["output_path"].endswith(".png"), "output extension")
    out = frame["output"]
    _require(set(out) == {"png_sha256", "pixel_sha256", "size", "mode", "dtype"}, "output schema")
    _sha(out["png_sha256"])
    _sha(out["pixel_sha256"])
    _strict_equal({k: out[k] for k in ("size", "mode", "dtype")},
                  {"size": [160, 160], "mode": "RGB", "dtype": "uint8"}, "output layout")
    if root is not None:
        _strict_equal(inspect_png(_inside(root, frame["output_path"])), out, "output integrity")


def make_manifest(catalog, snapshot, environment, decodes, frames, provenance, *, mode="training_qualification"):
    """Scientific digest excludes clocks/machine paths. Publication mode needs all 280 sources."""
    document = json.loads(canonical_bytes({
        "schema": SCHEMA, "dataset": "MSU-MFSD", "policy": POLICY, "mode": mode,
        "status": "technical_outputs_complete" if mode == "training_qualification" else "coverage_complete",
        "publication_ready": False, "catalog_sha256": catalog.catalog_sha256,
        "protocol_lock_sha256": catalog.protocol_lock_sha256,
        "projection_sha256": catalog.projection_sha256, "evidence": catalog.evidence,
        "snapshot_sha256": digest(snapshot), "environment_sha256": digest(asdict(environment)),
        "provenance": provenance, "decodes": [asdict(d) for d in decodes], "frames": frames,
        "commands": [portable_commands(d.codec, d.video_id.removeprefix("MSU-MFSD:")) for d in decodes]}))
    validate_manifest(document, catalog, snapshot, environment)
    return document


def validate_manifest(document, catalog, snapshot, environment, output_root=None):
    expected_keys = {"schema", "dataset", "policy", "mode", "status", "publication_ready",
                     "catalog_sha256", "protocol_lock_sha256", "projection_sha256", "evidence",
                     "snapshot_sha256", "environment_sha256", "provenance", "decodes", "frames", "commands"}
    _require(set(document) == expected_keys and document["schema"] == SCHEMA and
             document["dataset"] == "MSU-MFSD" and document["policy"] == POLICY and
             document["publication_ready"] is False, "manifest schema/readiness")
    environment.validate()
    hashes = _snapshot_sources(snapshot, catalog)
    for key, value in (("catalog_sha256", catalog.catalog_sha256),
                       ("protocol_lock_sha256", catalog.protocol_lock_sha256),
                       ("projection_sha256", catalog.projection_sha256),
                       ("snapshot_sha256", digest(snapshot)),
                       ("environment_sha256", digest(asdict(environment))), ("evidence", catalog.evidence)):
        _strict_equal(document[key], value, "manifest provenance binding")
    _require(set(document["provenance"]) == {"policy_document_sha256", "amendment_sha256", "implementation_sha256"},
             "provenance schema")
    for value in document["provenance"].values():
        _sha(value)
    mode = document["mode"]
    _require(mode in ("training_qualification", "publication"), "unsupported manifest mode")
    scope = TRAIN_QUALIFICATION if mode == "training_qualification" else tuple(s.video_id for s in catalog.sources)
    _require(document["status"] == ("technical_outputs_complete" if mode == "training_qualification"
                                    else "coverage_complete"), "manifest status")
    decoded = [_decode_document(d) for d in document["decodes"]]
    _strict_equal(document["commands"], [portable_commands(d.codec, d.video_id.removeprefix("MSU-MFSD:"))
                                         for d in decoded], "command provenance")
    _require(tuple(d.video_id for d in decoded) == scope, "decode scope/order")
    lookup = {s.video_id: s for s in catalog.sources}
    expected_pairs = [(d.video_id, i) for d in decoded for i in sample_ordinals(d.n)]
    _require([(f["group_id"], f["ordinal"]) for f in document["frames"]] == expected_pairs,
             "incomplete/duplicate/unordered frame coverage")
    outputs = [f["output_path"].casefold() for f in document["frames"]]
    _require(len(outputs) == len(set(outputs)), "duplicate output path")
    by_id = {d.video_id: d for d in decoded}
    for d in decoded:
        _require(d.source_sha256 == hashes[d.video_id] and d.snapshot_sha256 == digest(snapshot),
                 "decode source snapshot mismatch")
        _require(d.executable_sha256 == environment.ffmpeg_sha256 and
                 d.probe_executable_sha256 == environment.ffprobe_sha256, "decoder artifact binding")
    for f in document["frames"]:
        video = f["group_id"]
        _frame_validate(f, lookup[video], hashes[video], by_id[video], output_root)
    return digest(document)


def write_manifest(output_root, document, catalog, snapshot, environment, raw_root):
    root = Path(output_root)
    _require(read_json(root / "state.json") == {"schema": SCHEMA, "state": "partial"} and
             not any((root / p).exists() for p in ("failure.json", "completion.json", "manifest.json", "completion.pending.json", "completion.tmp.json")),
             "output failed or already completed")
    verify_snapshot(raw_root, catalog, snapshot)
    sha = validate_manifest(document, catalog, snapshot, environment, root)
    try:
        write_once(root / "completion.pending.json", {"schema": SCHEMA, "state": "pending"})
        write_once(root / "manifest.json", document)
        # Only publish the seal after its payload has been flushed and fsynced.
        temporary = root / "completion.tmp.json"
        write_once(temporary, {"schema": SCHEMA, "manifest_sha256": sha})
        os.replace(temporary, root / "completion.json")
        (root / "completion.pending.json").unlink()
    except BaseException:
        # A pending marker survives failed publication, even if failure recording
        # itself is impossible (e.g. disk full). Never infer success from a seal.
        try:
            write_once(root / "failure.json", {"schema": SCHEMA, "state": "failed", "reason": "publication write failed"})
        except OSError:
            pass
        raise



def read_completed_manifest(output_root, catalog, snapshot, environment):
    root = Path(output_root)
    _require(not any((root / name).exists() for name in
                     ("failure.json", "completion.pending.json", "completion.tmp.json")), "failed or incomplete output")
    document = read_json(root / "manifest.json")
    sha = validate_manifest(document, catalog, snapshot, environment, root)
    _strict_equal(read_json(root / "completion.json"), {"schema": SCHEMA, "manifest_sha256": sha},
                  "completion seal mismatch")
    return document
