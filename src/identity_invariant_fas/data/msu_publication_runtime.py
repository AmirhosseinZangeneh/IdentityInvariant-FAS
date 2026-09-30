"""Explicit Path B transports. No media discovery and no execution on import."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
from types import MappingProxyType
import zipfile

from . import msu_publication_preprocessing as p


@dataclass(frozen=True)
class ApprovedExecutableSpec:
    sha256: str
    size: int
    version: str


APPROVED_EXECUTABLES = MappingProxyType({
    'ffmpeg': ApprovedExecutableSpec(
        '1326dde4c84ff1f96fe6b8916c5bed29e163e9b5dccf995f6f3db069d143ec5e',
        101897728, '8.1.2-essentials_build-www.gyan.dev'),
    'ffprobe': ApprovedExecutableSpec(
        'b49ccc7c6547b141ad5a2f6ec69cc04323d7133d7704d70b331b904c63eecb07',
        101692928, '8.1.2-essentials_build-www.gyan.dev'),
})


@dataclass(frozen=True)
class Executable:
    """Recorded evidence only; manually constructing this object grants no trust."""
    path: str
    sha256: str
    size: int
    version: str

    def verify(self, role):
        return verify_approved_executable(self.path, role)


def _verify_approved_bytes(path, role):
    p._require(role in APPROVED_EXECUTABLES, 'unsupported executable role')
    p._absolute(path)
    spec = APPROVED_EXECUTABLES[role]
    p.verify_file(path, spec.sha256, spec.size)
    return spec


def verify_approved_executable(path, role):
    """Check frozen role identity, then obtain version evidence from those bytes.

    Callers supply only a path, never acceptance criteria. Entry points choose
    the required role, independently of any caller-created evidence object.
    Hashes are rechecked after version execution and after successful media use.
    Path-based process creation cannot eliminate a concurrent replacement race
    between hashing and Windows CreateProcess; protect the local artifact directory.
    """
    spec = _verify_approved_bytes(path, role)
    run = subprocess.run([str(path), '-version'], stdin=subprocess.DEVNULL,
                         capture_output=True, timeout=30, shell=False)
    output = run.stdout.decode('utf-8', errors='strict')
    p._require(run.returncode == 0 and not run.stderr.strip(), repr(run.stderr))
    p._require(re.match(re.escape(role + ' version ' + spec.version) + r'(?:\s|$)', output)
               and 'configuration:' in output and 'libavcodec' in output
               and 'libavformat' in output, 'wrong/incomplete approved executable version')
    _verify_approved_bytes(path, role)
    return Executable(str(path), spec.sha256, spec.size, output)


def extract_executables(archive, destination):
    """Hash before extraction; extract only unambiguous executable members."""
    p.verify_file(archive, p.ARCHIVE_SHA256)
    destination = Path(destination).absolute()
    with zipfile.ZipFile(archive) as z:
        members = {}
        for name in ("ffmpeg", "ffprobe"):
            matches = [i for i in z.infolist() if i.filename.endswith('/bin/' + name + '.exe')]
            p._require(len(matches) == 1, 'ambiguous archive executable')
            members[name] = matches[0]
        destination.mkdir(parents=True, exist_ok=False)
        result = {}
        for name, member in members.items():
            path = destination / (name + '.exe')
            h = hashlib.sha256()
            with z.open(member) as src, path.open('xb') as dst:
                for chunk in iter(lambda: src.read(1024 * 1024), b''):
                    h.update(chunk)
                    dst.write(chunk)
            p.verify_file(path, h.hexdigest(), member.file_size)
            result[name] = verify_approved_executable(path, name)
    return result


def stream_process(command, consume, *, timeout=120, diagnostic_limit=4 * 1024 * 1024):
    """Drain both pipes concurrently; timeout/overflow/parser failure kills and reaps.

    Stdout is delivered in 64 KiB chunks. Diagnostics are retained byte-for-byte
    up to a safety ceiling; exceeding it fails, never truncates into success.
    """
    diagnostics = bytearray()
    errors = []
    timed_out = threading.Event()
    with subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, shell=False) as process:
        def drain():
            try:
                for block in iter(lambda: process.stderr.read(65536), b''):
                    if len(diagnostics) + len(block) > diagnostic_limit:
                        errors.append('stderr safety limit exceeded')
                        process.kill()
                        return
                    diagnostics.extend(block)
            except BaseException as exc:
                errors.append(repr(exc))
                process.kill()

        def expire():
            timed_out.set()
            process.kill()

        reader = threading.Thread(target=drain, daemon=True)
        timer = threading.Timer(timeout, expire)
        reader.start()
        timer.start()
        try:
            for block in iter(lambda: process.stdout.read(65536), b''):
                consume(block)
            code = process.wait()
        except BaseException:
            process.kill()
            process.wait()
            raise
        finally:
            timer.cancel()
            reader.join()
        p._require(not timed_out.is_set(), 'subprocess timeout')
        p._require(not errors, repr(errors))
    stderr = diagnostics.decode('utf-8', errors='strict')
    p._require(code == 0 and not stderr.strip(), f'subprocess exit={code}; stderr={stderr!r}')
    return code, stderr


def probe(executable, source, codec):
    executable = verify_approved_executable(executable.path, 'ffprobe')
    payload = bytearray()
    def collect(block):
        p._require(len(payload) + len(block) <= 128 * 1024 * 1024, 'probe JSON safety limit')
        payload.extend(block)
    code, stderr = stream_process(p.build_probe_command(executable.path, source, codec), collect)
    _verify_approved_bytes(executable.path, 'ffprobe')
    doc = json.loads(payload, object_pairs_hook=p._pairs)
    streams = [s for s in doc['streams'] if s['codec_type'] == 'video']
    p._require(len(streams) == 1, 'ambiguous video streams')
    stream = streams[0]
    p._require(stream['codec_name'] == codec, 'codec mismatch')
    p._require(stream.get('disposition', {}).get('attached_pic') == 0, 'attached picture')
    size = (stream['width'], stream['height'])
    frames = [f for f in doc['frames'] if f.get('media_type') == 'video']
    p._require(frames and all(f['stream_index'] == stream['index'] for f in frames), 'frame stream')
    p._require(stream.get('sample_aspect_ratio') == '1:1' and
               all(f.get('sample_aspect_ratio') == '1:1' for f in frames), 'non-square/unknown SAR')
    p._require(stream.get('field_order') in (None, 'unknown', 'progressive') and
               all(f.get('interlaced_frame') == 0 for f in frames), 'nonprogressive/unknown frames')
    p._require(all((f['width'], f['height']) == size for f in frames), 'frame geometry drift')
    rotations, matrix = [], None
    if 'rotate' in stream.get('tags', {}):
        rotations.append(float(stream['tags']['rotate']))
    for side in stream.get('side_data_list', []):
        if side['side_data_type'] == 'Display Matrix':
            p._require(matrix is None, 'multiple display matrices')
            rows = [list(map(int, line.split(':', 1)[1].split()))
                    for line in side['displaymatrix'].strip().splitlines()]
            p._require(len(rows) == 3 and all(len(row) == 3 for row in rows), 'display matrix shape')
            # ffprobe's signed rotation is -atan2(b, a); its displayed 2x2
            # block therefore already matches the contract's Cartesian CCW block.
            scales = (65536, 65536, 1073741824)
            matrix = tuple(tuple(rows[i][j] / scales[j] for j in range(3)) for i in range(3))
            rotations.append(float(side['rotation']))
    p.normalize_orientation(tuple(rotations), matrix)
    return doc, dict(codec=codec, source_size=size, rotations=tuple(rotations), matrix=matrix,
        probe_stderr=stderr, probe_returncode=code, probe_complete=True,
        probe_frame_sizes=tuple((f['width'], f['height']) for f in frames), progressive=True,
        sample_aspect_ratio=(1, 1), video_stream_count=1, attached_picture=False,
        color_metadata=tuple((name, stream.get(key)) for name, key in
            (('range', 'color_range'), ('matrix', 'color_space'),
             ('primaries', 'color_primaries'), ('transfer', 'color_transfer'))))


@dataclass(frozen=True)
class SelectedFrame:
    """Transient pixels, never part of a JSON manifest."""
    ordinal: int
    oriented_rgb: object


def validate_selected_frames(result, frames):
    result.validate()
    p._require(tuple(f.ordinal for f in frames) == p.sample_ordinals(result.n),
               'selected frame scope/order')
    for frame in frames:
        p._rgb(frame.oriented_rgb)
        p._require((frame.oriented_rgb.shape[1], frame.oriented_rgb.shape[0]) == result.oriented_size
                   and p.pixel_sha256(frame.oriented_rgb) == result.frame_hashes[frame.ordinal],
                   'selected oriented RGB binding')


def decode(executables, root, source, snapshot, snapshot_sha256, codec):
    result, raw_probe, _ = _decode_core(executables, root, source, snapshot, snapshot_sha256,
                                        codec, retain_selected=False)
    return result, raw_probe


def decode_selected_frames(executables, root, source, snapshot, snapshot_sha256, codec):
    return _decode_core(executables, root, source, snapshot, snapshot_sha256,
                        codec, retain_selected=True)


def _decode_core(executables, root, source, snapshot, snapshot_sha256, codec, *, retain_selected):
    """Bind a pre-existing snapshot before either scan and recheck after natural EOF.

    Hash every frame. Optionally retain only the 30 probe-derived sample positions;
    no second media decode, and no successful result before all EOF checks pass.
    """
    # Never trust caller-supplied hashes, roles, version claims or verify methods.
    executable = verify_approved_executable(executables['ffmpeg'].path, 'ffmpeg')
    probe_executable = verify_approved_executable(executables['ffprobe'].path, 'ffprobe')
    p._validate_source(source)
    p._require(p.digest(snapshot) == snapshot_sha256, 'snapshot digest mismatch')
    matches = [r for r in snapshot['sources'] if r['source'] == asdict(source)]
    p._require(len(matches) == 1, 'source absent/ambiguous in snapshot')
    sha = matches[0]['sha256']
    path = p._inside(root, source.filepath).absolute()
    def verify():
        before = path.stat()
        p.verify_file(path, sha, source.byte_size)
        after = path.stat()
        p._require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns),
                   'source drift during verification')
        return after.st_size, after.st_mtime_ns
    initial = verify()
    raw_probe, fields = probe(probe_executable, path, codec)
    selected_ordinals = (p.sample_ordinals(len(fields['probe_frame_sizes']))
                         if retain_selected else ())
    selected_set, selected = set(selected_ordinals), []
    executable = verify_approved_executable(executable.path, 'ffmpeg')
    parser = p.PPMParser(fields['source_size'])
    hashes = []
    angle = p.normalize_orientation(fields['rotations'], fields['matrix'])
    def consume(block):
        for frame in parser.feed(block):
            oriented = p.orient_rgb(frame, angle)
            ordinal = len(hashes)
            hashes.append(p.pixel_sha256(oriented))
            if ordinal in selected_set:
                selected.append(SelectedFrame(ordinal, oriented))
    code, stderr = stream_process(p.build_decode_command(executable.path, path, codec), consume)
    _verify_approved_bytes(executable.path, 'ffmpeg')
    parser.finish()
    p._require(verify() == initial, 'source drift during decode')
    w, h = fields['source_size']
    result = p.DecodeResult(video_id=source.video_id, source_sha256=sha,
        snapshot_sha256=snapshot_sha256, executable_sha256=executable.sha256,
        probe_executable_sha256=probe_executable.sha256, frame_hashes=tuple(hashes),
        oriented_size=(h, w) if angle % 180 else (w, h), n=len(hashes), stderr=stderr,
        returncode=code, complete=True, **fields)
    result.validate()
    if retain_selected:
        validate_selected_frames(result, selected)
    return result, raw_probe, tuple(selected)


def infer_yunet(model, oriented_rgb):
    """Always exec a new Python interpreter, even with cv2 imported in the parent."""
    p._absolute(model)
    p.verify_file(model, p.MODEL_SHA256, 232589)
    p._rgb(oriented_rgb)
    import numpy as np
    header = json.dumps({'model': str(model), 'shape': list(oriented_rgb.shape[:2])}).encode() + b'\n'
    run = subprocess.run([sys.executable, '-m', 'identity_invariant_fas.data.msu_yunet_worker'],
        input=header + np.ascontiguousarray(oriented_rgb).tobytes(), capture_output=True,
        env=p.detector_worker_environment(os.environ), timeout=120, shell=False)
    p._require(run.returncode == 0 and not run.stderr.strip(),
               f'worker exit={run.returncode}; stderr={run.stderr!r}')
    result = json.loads(run.stdout, object_pairs_hook=p._pairs)
    p._require(result['schema'] == 'msu-yunet-worker-v1' and
               result['cv2_absent_before_import'] is True and result['engine_environment'] is None
               and result['cv2_version'] == '4.11.0'
               and result['settings'] == p.DETECTOR and result['threads'] == 1
               and result['opencl'] is False and result['onnx_runtime_used'] is False
               and result['input_size'] == [640, 640] and result['repeat_identical'] is True,
               'worker qualification evidence mismatch')
    return result


def select_inference(result):
    """Scientific selection is separate from successful zero-detection execution."""
    box, score = p.single_detection(result['detections'])
    return p.remap_detection(box, p.Letterbox(**result['geometry'])), score


def verify_retained_yunet(evidence, model):
    """Revalidate local successful evidence against the installed binary fingerprints.

    This is evidence reuse, not a new inference or an authenticity signature.
    """
    import importlib.metadata
    import platform
    import cv2
    import numpy._core._multiarray_umath as np_binary
    import PIL._imaging as pil_binary
    actual = (("Python", platform.python_version()),) + tuple(
        (name, importlib.metadata.version(name)) for name, _ in p.RUNTIME[1:])
    p._require(actual == p.RUNTIME, 'retained evidence runtime version mismatch')
    p.validate_cv2_runtime(cv2)
    p.verify_file(model, p.MODEL_SHA256, 232589)
    record = p.read_json(evidence)
    p._require(record['schema'] == 'msu-yunet-worker-v1' and
        record['settings'] == p.DETECTOR and record['cv2_version'] == cv2.__version__ and
        record['python'] == platform.python_version() and record['cv2_absent_before_import'] is True and
        record['engine_environment'] is None and record['threads'] == 1 and
        record['opencl'] is False and record['onnx_runtime_used'] is False and
        record['repeat_identical'] is True and record['input_size'] == [640, 640] and
        record['detections'] == [] and record['opencv_build'] == cv2.getBuildInformation(),
        'retained YuNet qualification mismatch')
    binaries = list(Path(cv2.__file__).parent.glob('*.pyd'))
    p._require(len(binaries) == 1, 'ambiguous cv2 binary')
    paths = (sys.executable, binaries[0], np_binary.__file__, pil_binary.__file__)
    expected = [[name, p.stream_sha256(path)] for (name, _), path in zip(p.RUNTIME, paths)]
    p._require(record['runtime_fingerprints'] == expected, 'retained binary fingerprint mismatch')
    return record
