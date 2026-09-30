"""Entirely synthetic Path B validation. Never discovers or opens licensed data."""
from copy import deepcopy
from dataclasses import FrozenInstanceError, asdict, replace
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import zipfile

import numpy as np
import pytest
from PIL import Image

from identity_invariant_fas.data import msu_publication_preprocessing as p

_WORKER_INITIALIZER = p.create_detector_in_worker


@pytest.fixture(autouse=True)
def no_external_execution(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("No network/subprocess/artifact inference in Path B tests")
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(p, "create_detector_in_worker", forbidden)


def store(path, value):
    path.write_bytes(p.canonical_bytes(value))


@pytest.fixture
def catalog_files(tmp_path):
    raw = tmp_path / "synthetic-source"
    raw.mkdir()
    (raw / "README.txt").write_text("synthetic metadata only")
    (raw / "train_sub_list.txt").write_text(" ".join(p.TRAIN))
    (raw / "test_sub_list.txt").write_text(" ".join(p.TEST))
    fingerprints = {n: p.stream_sha256(raw / n) for n in
                    ("README.txt", "train_sub_list.txt", "test_sub_list.txt")}
    fingerprints["DecFrames.m"] = "a" * 64  # Historical opaque field; no file required.
    rows = []
    for client in (*p.TRAIN, *p.TEST):
        for camera in p.CAMERAS:
            for attack in (None, *p.MEDIA):
                video = p._video(client, camera, attack)
                rel = video.removeprefix("MSU-MFSD:")
                path = raw / rel
                path.parent.mkdir(exist_ok=True, parents=True)
                path.write_bytes(video.encode())  # Unique tiny non-video bytes.
                rows.append(asdict(p.Source(rel, client, "train" if client in p.TRAIN else "test",
                    int(attack is not None), camera, p.CAMERAS[camera], attack, p.MEDIA.get(attack),
                    "SD", "scene01", path.stat().st_size, video)))
    rows.sort(key=lambda r: r["filepath"])
    for row in rows:
        row.update(annotation_filepath="opaque/not-required.face", annotation_sha256="opaque")
    catalog = {"schema_version": 1, "dataset": "MSU-MFSD", "protocol": "official_public_train_test",
               "train_clients": list(p.TRAIN), "test_clients": list(p.TEST),
               "source_fingerprints": fingerprints, "recordings": rows}
    lock = {"normalized_metadata_sha256": p.digest(catalog), "source_fingerprints": fingerprints,
            "partitions": {"train": {"client_ids": list(p.TRAIN)}, "test": {"client_ids": list(p.TEST)}}}
    cp, lp = tmp_path / "catalog.json", tmp_path / "lock.json"
    store(cp, catalog)
    store(lp, lock)
    return raw, cp, lp


def rewrite_catalog(files, change):
    raw, cp, lp = files
    doc, lock = p.read_json(cp), p.read_json(lp)
    change(doc)
    lock["normalized_metadata_sha256"] = p.digest(doc)
    store(cp, doc)
    store(lp, lock)


def test_source_catalog_no_annotations(catalog_files, monkeypatch):
    original = Path.open
    def guarded(path, *args, **kwargs):
        assert path.suffix != ".face"
        assert path.suffix not in (".mp4", ".mov")  # Validator uses stat only.
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", guarded)
    catalog = p.load_source_catalog(*catalog_files)
    assert len(catalog.sources) == 280
    assert sum(s.pad_partition == "train" for s in catalog.sources) == 120
    assert not list(catalog_files[0].rglob("*.face"))
    assert "annotation" not in p.canonical_bytes([asdict(s) for s in catalog.sources]).decode()


def test_catalog_digest(catalog_files):
    cp = catalog_files[1]
    doc = p.read_json(cp)
    doc["recordings"][0]["annotation_sha256"] = "changed opaque historical field"
    store(cp, doc)
    with pytest.raises(p.ContractError, match="catalog digest"):
        p.load_source_catalog(*catalog_files)


@pytest.mark.parametrize("change", [
    lambda d: d["recordings"].pop(),
    lambda d: d["recordings"].append(d["recordings"][0]),
    lambda d: d["recordings"][0].update(class_label=0),
    lambda d: d["recordings"][0].update(presentation_device="wrong"),
    lambda d: d["recordings"][0].update(filepath="../outside.mov"),
    lambda d: d["recordings"][0].update(video_id="wrong"),
    lambda d: d["train_clients"].reverse(),
])
def test_catalog_semantics(catalog_files, change):
    rewrite_catalog(catalog_files, change)
    with pytest.raises(p.ContractError):
        p.load_source_catalog(*catalog_files)


def test_missing_and_size_drift(catalog_files):
    raw, cp, _ = catalog_files
    path = raw / p.read_json(cp)["recordings"][0]["filepath"]
    path.write_bytes(b"wrong")
    with pytest.raises(p.ContractError, match="size drift"):
        p.load_source_catalog(*catalog_files)
    path.unlink()
    with pytest.raises(p.ContractError, match="missing"):
        p.load_source_catalog(*catalog_files)


def test_hardlink_alias(catalog_files):
    raw, cp, _ = catalog_files
    rows = p.read_json(cp)["recordings"]
    first, second = (raw / rows[i]["filepath"] for i in (0, 1))
    second.unlink()
    os.link(first, second)
    rewrite_catalog(catalog_files, lambda d: d["recordings"][1].update(byte_size=first.stat().st_size))
    with pytest.raises(p.ContractError, match="physical alias"):
        p.load_source_catalog(*catalog_files)


def test_source_snapshot_drift_and_duplicates(catalog_files):
    catalog = p.load_source_catalog(*catalog_files)
    raw = catalog_files[0]
    snapshot = p.source_snapshot(raw, catalog)
    p.verify_snapshot(raw, catalog, snapshot)
    first = raw / catalog.sources[0].filepath
    original = first.read_bytes()
    first.write_bytes(b"!" + original[1:])
    with pytest.raises(p.ContractError, match="snapshot drift"):
        p.verify_snapshot(raw, catalog, snapshot)
    first.write_bytes(original)
    second = raw / catalog.sources[1].filepath
    second.write_bytes(original)
    changed = replace(catalog, sources=(catalog.sources[0], replace(catalog.sources[1], byte_size=len(original)),
                                        *catalog.sources[2:]))
    with pytest.raises(p.ContractError, match="duplicate source content"):
        p.source_snapshot(raw, changed)


def test_streaming_hash_and_artifact_failure(tmp_path):
    path = tmp_path / "artifact"
    payload = b"synthetic" * 300000
    path.write_bytes(payload)
    sha = hashlib.sha256(payload).hexdigest()
    assert p.stream_sha256(path) == sha
    p.verify_file(path, sha, len(payload))
    with pytest.raises(p.ContractError, match="hash"):
        p.verify_file(path, "0" * 64)
    with pytest.raises(p.ContractError, match="size"):
        p.verify_file(path, sha, 1)


@pytest.fixture
def environment():
    return p.EnvironmentLock(p.ARCHIVE_SHA256, p.MODEL_SHA256, "1" * 64, "2" * 64, p.RUNTIME,
        "ffmpeg version 8.1.2-build\nconfiguration: synthetic\nlibavcodec\nlibavformat",
        "ffprobe version 8.1.2-build\nconfiguration: synthetic\nlibavcodec\nlibavformat",
        "General configuration for OpenCV 4.11.0", tuple((n, "3" * 64) for n, _ in p.RUNTIME))


def test_environment_contract_immutable(environment, tmp_path):
    environment.validate()
    with pytest.raises(FrozenInstanceError):
        environment.model_sha256 = "0" * 64
    for bad in (replace(environment, model_sha256="0" * 64),
                replace(environment, versions=()), replace(environment, opencv_build="ONNX Runtime: YES"),
                replace(environment, ffmpeg_version_output="ffmpeg version 8.1.20")):
        with pytest.raises(p.ContractError):
            bad.validate()
    lock = tmp_path / "environment.json"
    p.write_once(lock, asdict(environment))
    with pytest.raises(FileExistsError):
        p.write_once(lock, asdict(environment))


def test_actual_artifact_validation_rejects_synthetic_archive(environment, tmp_path):
    archive = tmp_path / "ffmpeg.zip"
    archive.write_bytes(b"not approved")
    with pytest.raises(p.ContractError, match="hash"):
        p.validate_artifact_files(environment, archive, tmp_path / p.MODEL_NAME,
                                 tmp_path / "ffmpeg.exe", tmp_path / "ffprobe.exe")


def test_archive_executable_binding_synthetic(environment, tmp_path, monkeypatch):
    archive, model = tmp_path/'approved.zip', tmp_path/p.MODEL_NAME
    ffmpeg, ffprobe = tmp_path/'ffmpeg.exe', tmp_path/'ffprobe.exe'
    ffmpeg.write_bytes(b'synthetic executable never executed')
    ffprobe.write_bytes(b'synthetic probe never executed')
    model.write_bytes(b'0'*232589)
    with zipfile.ZipFile(archive,'w') as z:
        z.writestr('release/bin/ffmpeg.exe',ffmpeg.read_bytes())
        z.writestr('release/bin/ffprobe.exe',ffprobe.read_bytes())
    # Trust anchors are replaced only within this synthetic fixture; hashing/ZIP
    # membership checks themselves run for real, without artifacts or execution.
    monkeypatch.setattr(p,'ARCHIVE_SHA256',p.stream_sha256(archive))
    monkeypatch.setattr(p,'MODEL_SHA256',p.stream_sha256(model))
    lock = replace(environment,archive_sha256=p.ARCHIVE_SHA256,model_sha256=p.MODEL_SHA256,
                   ffmpeg_sha256=p.stream_sha256(ffmpeg),ffprobe_sha256=p.stream_sha256(ffprobe))
    p.validate_artifact_files(lock,archive,model,ffmpeg,ffprobe)
    ffmpeg.write_bytes(b'other executable')
    changed = replace(lock,ffmpeg_sha256=p.stream_sha256(ffmpeg))
    with pytest.raises(p.ContractError,match='pinned archive'):
        p.validate_artifact_files(changed,archive,model,ffmpeg,ffprobe)


def test_exact_decode_command(tmp_path):
    exe, source = tmp_path / "ffmpeg.exe", tmp_path / "source.mov"
    cmd = p.build_decode_command(exe, source, "prores")
    assert cmd == (str(exe), "-nostdin", "-hide_banner", "-loglevel", "warning", "-nostats",
        "-xerror", "-hwaccel", "none", "-threads", "1", "-c:v", "prores", "-err_detect",
        "explode", "-noautorotate", "-i", str(source), "-map", "0:v:0", "-an", "-sn", "-dn",
        "-filter_threads", "1", "-fps_mode", "passthrough", "-c:v", "ppm", "-threads", "1",
        "-pix_fmt", "rgb24", "-f", "image2pipe", "pipe:1")
    p.validate_command(cmd, exe, source, "prores")
    for args in (("-ss", "1"), ("-r", "30"), ("-t", "1"), ("-frames:v", "30"),
                 ("-fflags", "+discardcorrupt"), ("-hwaccel", "auto"), ("-vf", "fps=30")):
        with pytest.raises(p.ContractError, match="command"):
            p.validate_command((*cmd, *args), exe, source, "prores")
    probe = p.build_probe_command(tmp_path / "ffprobe.exe", source, "prores")
    assert "-show_frames" in probe and "-show_streams" in probe and "-select_streams" not in probe
    p.validate_command(probe, tmp_path / "ffprobe.exe", source, "prores", probe=True)
    with pytest.raises(p.ContractError, match="absolute"):
        p.build_decode_command("ffmpeg", source, "h264")
    with pytest.raises(p.ContractError, match="codec"):
        p.build_decode_command(exe, source, "auto")


def ppm(image):
    return f"P6\n{image.shape[1]} {image.shape[0]}\n255\n".encode() + image.tobytes()


@pytest.mark.parametrize("chunk_size", [1, 2, 7, 10000])
def test_incremental_ppm(chunk_size):
    images = [np.arange(18, dtype=np.uint8).reshape(2, 3, 3), np.full((1, 4, 3), 10, np.uint8)]
    payload = b"".join(ppm(i) for i in images)
    frames = list(p.parse_ppm(payload[i:i+chunk_size] for i in range(0, len(payload), chunk_size)))
    assert len(frames) == 2
    for actual, expected in zip(frames, images):
        np.testing.assert_array_equal(actual, expected)
        assert actual.flags.c_contiguous and actual.dtype == np.uint8


@pytest.mark.parametrize("payload", [b"P3\n1 1\n255\nabc", b"P6\n0 1\n255\n", b"P6\n-1 1\n255\n",
    b"P6\n1 1\n256\nabc", b"P6\n1 1\n255\nab", b"P6\n1 1\n255\nabcP6\n",
    b"P6\n#comment\n1 1\n255\nabc", b"P6\r\n1 1\r\n255\r\nabc", b"X"*81])
def test_ppm_invalid(payload):
    with pytest.raises(p.ContractError):
        list(p.parse_ppm([payload]))


@pytest.mark.parametrize("angle,k", [(0, 0), (90, 1), (180, 2), (270, 3), (-90, 3), (450, 1)])
def test_rotation(angle, k):
    image = np.repeat(np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint8)[:, :, None], 3, axis=2)
    expected = [np.array([[1, 2, 3], [4, 5, 6]]), np.array([[3, 6], [2, 5], [1, 4]]),
                np.array([[6, 5, 4], [3, 2, 1]]), np.array([[4, 1], [5, 2], [6, 3]])][k]
    np.testing.assert_array_equal(p.orient_rgb(image, angle)[:, :, 0], expected)
    assert p.normalize_orientation((angle,)) == k * 90


@pytest.mark.parametrize("rotations,matrix", [((45,), None), ((float('nan'),), None),
    ((float('inf'),), None), ((0, 90), None), ((), [[-1,0,0],[0,1,0],[0,0,1]]),
    ((), [[1,.1,0],[0,1,0],[0,0,1]]), ((), [[2,0,0],[0,2,0],[0,0,1]]),
    ((180,), [[0,-1,0],[1,0,0],[0,0,1]])])
def test_invalid_orientation(rotations, matrix):
    with pytest.raises(p.ContractError):
        p.normalize_orientation(rotations, matrix)


def test_rotation_matrix_and_tolerance():
    assert p.normalize_orientation() == 0
    assert p.normalize_orientation((90+1e-7,), [[0,-1,0],[1,0,0],[0,0,1]]) == 90


def decoded(video=p.TRAIN_QUALIFICATION[0], image=None, n=30, source_sha="5"*64, snapshot_sha="6"*64):
    if image is None:
        image = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
    size = image.shape[1], image.shape[0]
    return p.DecodeResult(video, source_sha, snapshot_sha, "1"*64, "2"*64, "prores", size, (0,), None,
        (p.pixel_sha256(image),)*n, size, n, "", 0, True, "", 0, True, (size,)*n,
        True, (1, 1), 1, False, tuple((k, None) for k in ("range", "matrix", "primaries", "transfer")))


def test_decode_result_and_ppm_binding():
    image = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
    d = decoded()
    p.validate_decoded_frames(d, p.parse_ppm([ppm(image)*30]))
    with pytest.raises(p.ContractError, match="count"):
        p.validate_decoded_frames(d, [image]*29)
    with pytest.raises(p.ContractError, match="dimensions"):
        p.validate_decoded_frames(d, [np.zeros((1,1,3), np.uint8)]*30)
    with pytest.raises(p.ContractError, match="RGB"):
        p.validate_decoded_frames(d, [image.astype(np.float32)]*30)


@pytest.mark.parametrize("changes", [{"returncode":1}, {"stderr":"warning"}, {"complete":False},
    {"probe_stderr":"corrupt"}, {"probe_returncode":1}, {"probe_complete":False},
    {"n":29}, {"probe_frame_sizes":((1,1),)*30}, {"progressive":False},
    {"sample_aspect_ratio":(2,1)}, {"video_stream_count":2}, {"attached_picture":True},
    {"oriented_size":(2,3)}, {"codec":"auto"}, {"rotations":(45,)}])
def test_decode_fail_closed(changes):
    with pytest.raises(p.ContractError):
        replace(decoded(), **changes).validate()


def test_sampler_exhaustive():
    for n in range(30, 5000):
        positions = p.sample_ordinals(n)
        assert len(positions) == len(set(positions)) == 30
        assert positions[0] == 0 and positions[-1] == n-1
        assert positions == tuple((j*(n-1))//29 for j in range(30))
    assert p.sample_ordinals(10**20)[-1] == 10**20-1
    for invalid in (0, 29, 30.0, True):
        with pytest.raises(p.ContractError):
            p.sample_ordinals(invalid)


@pytest.mark.parametrize("size,resized,padding", [((720,480),(640,427),(0,106,0,107)),
    ((480,720),(427,640),(106,0,107,0)), ((640,640),(640,640),(0,0,0,0)),
    ((3,2),(640,427),(0,106,0,107)), ((1,2000),(1,640),(319,0,320,0))])
def test_letterbox(size, resized, padding):
    g = p.letterbox_geometry(*size)
    assert (g.resized_width,g.resized_height) == resized
    assert (g.left,g.top,g.right,g.bottom) == padding


def test_remap_original_coordinate_crop():
    g = p.letterbox_geometry(720,480)
    assert p.remap_detection((0,106,640,427),g) == (0,0,720,480)
    mapped = p.remap_detection((-4,100,20,30),g)  # Padding overlap allowed if original intersection exists.
    assert mapped[0] < 0 and mapped[1] < 0
    continuous, bounds = p.crop_bounds((.2,.3,10.1,8.2),20,20)
    assert bounds == (0,0,12,10)
    assert continuous == pytest.approx((-.81,-.52,11.31,9.32))
    for box in ((0,0,10,10), (0,106,-1,3), (float('nan'),1,3,4)):
        with pytest.raises(p.ContractError):
            p.remap_detection(box,g)


def detection(box=(160,180,250,200)):
    return [list(box) + [0]*10 + [.95]]


def test_exactly_one_detection():
    assert p.single_detection(detection())[1] == .95
    for rows in (None, [], detection()*2, detection()*3):
        with pytest.raises(p.ContractError, match="exactly one"):
            p.single_detection(rows)


def test_final_crop_original_pixels_and_reload(tmp_path):
    image = np.zeros((9,13,3),np.uint8)
    image[:,:,0] = np.arange(13)*15
    image[:,:,1] = np.arange(9)[:,None]*20
    before = image.copy()
    box = (2.2,1.2,5.1,4.2)
    output = tmp_path / "crop.png"
    metadata = p.save_crop(image,box,output)
    np.testing.assert_array_equal(image,before)
    _, (l,t,r,b) = p.crop_bounds(box,13,9)
    expected = Image.fromarray(image[t:b,l:r]).resize((160,160),Image.Resampling.BILINEAR,reducing_gap=None)
    assert metadata == p.inspect_png(output)
    assert metadata['pixel_sha256'] == p.pixel_sha256(np.asarray(expected))
    with pytest.raises(FileExistsError):
        p.save_crop(image,box,output)


def test_worker_environment_no_mutation():
    original = {"EXAMPLE":"unchanged"}
    child = p.detector_worker_environment(original)
    assert 'OPENCV_FORCE_DNN_ENGINE' not in child and original == {'EXAMPLE':'unchanged'}
    assert p.detector_worker_environment({'OPENCV_FORCE_DNN_ENGINE':'4'}) == {}


def test_worker_refuses_imported_cv2_or_obsolete_engine(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules,'cv2',object())
    monkeypatch.setenv('OPENCV_FORCE_DNN_ENGINE','1')
    with pytest.raises(p.ContractError,match='already imported'):
        _WORKER_INITIALIZER(tmp_path/'unused.onnx')
    monkeypatch.delitem(sys.modules,'cv2')
    with pytest.raises(p.ContractError,match='inapplicable'):
        _WORKER_INITIALIZER(tmp_path/'unused.onnx')


@pytest.mark.parametrize('version', ['5.0.0', '4.12.0', '4.11.1'])
def test_reject_wrong_cv2_version(version):
    from types import SimpleNamespace
    with pytest.raises(p.ContractError, match='cv2 runtime version'):
        p.validate_cv2_runtime(SimpleNamespace(__version__=version))


def test_amended_runtime_and_unchanged_detector_parameters():
    assert dict(p.RUNTIME) == {'Python': '3.12.10', 'opencv-python': '4.11.0.86',
                               'numpy': '2.5.2', 'Pillow': '12.3.0'}
    assert p.DETECTOR == {'model_sha256': p.MODEL_SHA256, 'input_size': [640,640],
        'backend': 3, 'target': 0, 'engine': 'opencv-4.11-native-dnn', 'onnx_runtime': False,
        'score_threshold': .90, 'nms_threshold': .30, 'top_k': 5000}


def test_scope_exact_training_only():
    p.validate_qualification_scope(p.TRAIN_QUALIFICATION)
    assert p.TRAIN_CODEC_GATE == p.TRAIN_QUALIFICATION[:4]
    assert len(p.TEST_CODEC_CHECK) == 4
    for ids in (p.TRAIN_QUALIFICATION[::-1], p.TRAIN_QUALIFICATION[:4],
                (*p.TRAIN_QUALIFICATION,*p.TEST_CODEC_CHECK), (*p.TRAIN_QUALIFICATION[:7],p.TEST_CODEC_CHECK[0])):
        with pytest.raises(p.ContractError):
            p.validate_qualification_scope(ids)


@pytest.mark.parametrize("path", ["C:/secret.png", "/secret.png", "../x", "a/../x", "a\\x", "a//x", "./x"])
def test_unsafe_relative_paths(path):
    with pytest.raises(p.ContractError):
        p.relative_path(path)


def test_output_root_safety(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    for output in (raw, raw / "out", tmp_path):
        with pytest.raises(p.ContractError, match="overlap"):
            p.fresh_output(raw,output)
    out = p.fresh_output(raw,tmp_path/'out')
    with pytest.raises(FileExistsError):
        p.fresh_output(raw,out)
    p.record_failure(out,"synthetic failure")
    assert not (out/'manifest.json').exists()
    with pytest.raises(FileExistsError):
        p.record_failure(out,"cannot overwrite")


@pytest.fixture
def manifest_case(catalog_files, environment, tmp_path):
    catalog = p.load_source_catalog(*catalog_files)
    snapshot = p.source_snapshot(catalog_files[0],catalog,p.TRAIN_QUALIFICATION)
    lookup = {s.video_id:s for s in catalog.sources}
    hashes = {x['source']['video_id']:x['sha256'] for x in snapshot['sources']}
    output = p.fresh_output(catalog_files[0],tmp_path/'output')
    image = np.arange(18,dtype=np.uint8).reshape(2,3,3)
    decodes, frames = [], []
    for j, video in enumerate(p.TRAIN_QUALIFICATION):
        d = decoded(video,image,source_sha=hashes[video],snapshot_sha=p.digest(snapshot))
        decodes.append(d)
        for ordinal in p.sample_ordinals(d.n):
            frames.append(p.make_frame_record(lookup[video],hashes[video],d,ordinal,image,
                           detection(),output,f'{j:02d}/{ordinal:02d}.png'))
    provenance = {k:'4'*64 for k in ('policy_document_sha256','amendment_sha256','implementation_sha256')}
    doc = p.make_manifest(catalog,snapshot,environment,decodes,frames,provenance)
    return catalog_files[0],catalog,snapshot,environment,output,doc


def test_manifest_roundtrip_completion(manifest_case):
    raw,cat,snapshot,env,out,doc = manifest_case
    before = p.digest(doc)
    p.write_manifest(out,doc,cat,snapshot,env,raw)
    loaded = p.read_completed_manifest(out,cat,snapshot,env)
    assert p.digest(loaded) == before and len(loaded['frames']) == 240
    assert loaded['publication_ready'] is False
    assert '.face' not in p.canonical_bytes(doc).decode()
    with pytest.raises(p.ContractError):
        p.write_manifest(out,doc,cat,snapshot,env,raw)
    with pytest.raises(p.ContractError):
        p.record_failure(out,'cannot alter completed run')


def test_manifest_adversarial_mutations(manifest_case):
    raw,cat,snapshot,env,out,doc = manifest_case
    changes = [lambda d:d['frames'].pop(), lambda d:d['frames'].append(d['frames'][0]),
        lambda d:d.update(publication_ready=True), lambda d:d.update(mode='publication'),
        lambda d:d['frames'][0].update(output_path='C:/outside.png'),
        lambda d:d['frames'][0].update(mapped_box=[0,0,1,1]),
        lambda d:d['frames'][0].update(annotation_index=0),
        lambda d:d['frames'][0]['detector'].update(engine=4),
        lambda d:d['decodes'][0].update(complete=False),
        lambda d:d['decodes'][0].update(executable_sha256='9'*64),
        lambda d:d['frames'][0].update(source_sha256='9'*64),
        lambda d:d['frames'][0]['source'].update(pad_partition='test'),
        lambda d:d['commands'][0]['ffmpeg'].extend(['-r','30']),
        lambda d:d.update(timestamp='today')]
    for change in changes:
        altered = deepcopy(doc)
        change(altered)
        with pytest.raises(p.ContractError):
            p.validate_manifest(altered,cat,snapshot,env,out)
    p.record_failure(out,'stop')
    with pytest.raises(p.ContractError):
        p.write_manifest(out,doc,cat,snapshot,env,raw)
    assert not (out/'manifest.json').exists()


def test_completion_missing_or_altered_rejected(manifest_case):
    raw,cat,snapshot,env,out,doc = manifest_case
    p.write_once(out/'manifest.json',doc)
    with pytest.raises(FileNotFoundError):
        p.read_completed_manifest(out,cat,snapshot,env)
    p.write_once(out/'completion.json',{'schema':p.SCHEMA,'manifest_sha256':'0'*64})
    with pytest.raises(p.ContractError,match='seal'):
        p.read_completed_manifest(out,cat,snapshot,env)


def test_png_tamper_and_failed_write(manifest_case, monkeypatch):
    raw,cat,snapshot,env,out,doc = manifest_case
    path = out/doc['frames'][0]['output_path']
    path.write_bytes(b'not a PNG')
    with pytest.raises((ValueError,OSError)):
        p.write_manifest(out,doc,cat,snapshot,env,raw)
    assert not (out/'manifest.json').exists()


def test_output_write_failure_preserves_partial(tmp_path, monkeypatch):
    original = Image.Image.save
    def partial(self, stream, *args, **kwargs):
        stream.write(b'partial')
        raise OSError('synthetic disk failure')
    monkeypatch.setattr(Image.Image,'save',partial)
    output = tmp_path/'partial.png'
    with pytest.raises(OSError,match='disk failure'):
        p.save_crop(np.zeros((4,4,3),np.uint8),(0,0,4,4),output)
    assert output.read_bytes() == b'partial'
    monkeypatch.setattr(Image.Image,'save',original)
    with pytest.raises(FileExistsError):
        p.save_crop(np.zeros((4,4,3),np.uint8),(0,0,4,4),output)


def test_strict_json(tmp_path):
    path = tmp_path/'bad.json'
    for payload in ('{"x":1,"x":2}', '{"x":NaN}'):
        path.write_text(payload)
        with pytest.raises(p.ContractError):
            p.read_json(path)


def test_no_historical_import_or_import_time_cv2():
    # AST checks do not depend on whether another test imported cv2 earlier.
    import ast
    tree = ast.parse(Path(p.__file__).read_text())
    top_imports = [n for n in tree.body if isinstance(n,(ast.Import,ast.ImportFrom))]
    assert not any('cv2' in ast.unparse(n) or 'msu_preprocessing' in ast.unparse(n) or
                   'msu_protocol' in ast.unparse(n) for n in top_imports)


@pytest.mark.parametrize("stage", ["write", "flush", "fsync", "publish", "after_publish"])
def test_completion_io_failures(manifest_case, monkeypatch, stage):
    raw, cat, snapshot, env, out, doc = manifest_case
    original_open, original_fsync, original_replace = Path.open, os.fsync, os.replace
    completion_fd = []

    class FaultyStream:
        def __init__(self, stream):
            self.stream = stream
        def __enter__(self):
            completion_fd.append(self.stream.fileno())
            return self
        def __exit__(self, *args):
            self.stream.close()
            completion_fd.clear()
        def write(self, payload):
            result = self.stream.write(payload)
            if stage == "write":
                raise OSError("completion write")
            return result
        def flush(self):
            self.stream.flush()
            if stage == "flush":
                raise OSError("completion flush")
        def fileno(self):
            return self.stream.fileno()

    def open_file(path, *args, **kwargs):
        stream = original_open(path, *args, **kwargs)
        return FaultyStream(stream) if path.name == "completion.tmp.json" and args == ("xb",) else stream
    def fsync(fd):
        if stage == "fsync" and fd in completion_fd:
            raise OSError("completion fsync")
        return original_fsync(fd)
    def publish(src, dst):
        if stage == "publish":
            raise OSError("completion publish")
        original_replace(src, dst)
        if stage == "after_publish":
            raise OSError("completion after publish")
    monkeypatch.setattr(Path, "open", open_file)
    monkeypatch.setattr(os, "fsync", fsync)
    monkeypatch.setattr(os, "replace", publish)
    with pytest.raises(OSError, match="completion"):
        p.write_manifest(out, doc, cat, snapshot, env, raw)
    assert (out / "failure.json").exists()
    assert (out / "completion.pending.json").exists()
    seal = out / ("completion.json" if stage == "after_publish" else "completion.tmp.json")
    # Even valid JSON is insufficient after any failed publication operation.
    assert p.read_json(seal)["manifest_sha256"] == p.digest(doc)
    with pytest.raises(p.ContractError, match="incomplete"):
        p.read_completed_manifest(out, cat, snapshot, env)


def test_completion_pending_survives_failure_marker_disk_failure(manifest_case, monkeypatch):
    raw, cat, snapshot, env, out, doc = manifest_case
    original_open = Path.open
    def open_file(path, *args, **kwargs):
        if path.name in ("completion.tmp.json", "failure.json") and args == ("xb",):
            raise OSError("disk full")
        return original_open(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", open_file)
    with pytest.raises(OSError):
        p.write_manifest(out, doc, cat, snapshot, env, raw)
    assert (out / "completion.pending.json").exists()
    assert not (out / "completion.json").exists()
    with pytest.raises(p.ContractError):
        p.read_completed_manifest(out, cat, snapshot, env)


def test_completion_failure_marker_conflict(manifest_case):
    raw, cat, snapshot, env, out, doc = manifest_case
    p.write_manifest(out, doc, cat, snapshot, env, raw)
    p.write_once(out / "failure.json", {"state": "failed"})
    with pytest.raises(p.ContractError, match="failed"):
        p.read_completed_manifest(out, cat, snapshot, env)


def test_valid_temporary_seal_is_not_completion(manifest_case):
    raw, cat, snapshot, env, out, doc = manifest_case
    p.write_once(out / "manifest.json", doc)
    p.write_once(out / "completion.tmp.json", {"schema": p.SCHEMA, "manifest_sha256": p.digest(doc)})
    with pytest.raises(p.ContractError, match="incomplete"):
        p.read_completed_manifest(out, cat, snapshot, env)


def test_decode_snapshot_relabeling_rejected(manifest_case):
    raw, cat, snapshot, env, out, doc = manifest_case
    video = doc['decodes'][0]['video_id']
    source = next(s for s in cat.sources if s.video_id == video)
    path = raw / source.filepath
    payload = path.read_bytes()
    path.write_bytes(b'!' + payload[1:])  # Same ID and size; different content.
    new_snapshot = p.source_snapshot(raw, cat, p.TRAIN_QUALIFICATION)
    new_sha = next(r['sha256'] for r in new_snapshot['sources'] if r['source']['video_id'] == video)
    altered = deepcopy(doc)
    altered['snapshot_sha256'] = p.digest(new_snapshot)
    for frame in altered['frames']:
        if frame['group_id'] == video:
            frame['source_sha256'] = new_sha
    with pytest.raises(p.ContractError, match='decode source snapshot'):
        p.validate_manifest(altered, cat, new_snapshot, env, out)
    d = p._decode_document(doc['decodes'][0])
    with pytest.raises(p.ContractError, match='decode source hash'):
        p.make_frame_record(source, new_sha, d, 0, np.arange(18, dtype=np.uint8).reshape(2,3,3),
                            detection(), out, 'must-not-exist.png')
    assert not (out / 'must-not-exist.png').exists()
    # Changing the snapshot binding alone must not mask an old source digest.
    for decode in altered['decodes']:
        decode['snapshot_sha256'] = p.digest(new_snapshot)
    with pytest.raises(p.ContractError, match='decode source snapshot'):
        p.validate_manifest(altered, cat, new_snapshot, env, out)


@pytest.mark.parametrize('field', ['source_sha256', 'snapshot_sha256'])
def test_decode_hash_validation_and_immutability(field):
    for invalid in ('A'*64, '0'*63, 'invalid'):
        with pytest.raises(p.ContractError, match='SHA-256'):
            replace(decoded(), **{field: invalid}).validate()
    with pytest.raises(FrozenInstanceError):
        setattr(decoded(), field, '0'*64)


def test_decode_source_mutation_after_frame_creation(manifest_case):
    raw, cat, snapshot, env, out, doc = manifest_case
    p.validate_manifest(doc, cat, snapshot, env, out)
    altered = deepcopy(doc)
    altered['decodes'][0]['source_sha256'] = '9'*64
    with pytest.raises(p.ContractError, match='decode source snapshot'):
        p.validate_manifest(altered, cat, snapshot, env, out)


def test_ppm_limits():
    parser = p.PPMParser()
    with pytest.raises(p.ContractError, match='bound'):
        parser.feed(b'P6\n99999999999999999999 99999999999999999999\n255\n')
    assert len(parser.buffer) <= parser.MAX_HEADER_BYTES
    with pytest.raises(p.ContractError, match='bound'):
        p.PPMParser(max_frame_bytes=12).feed(b'P6\n3 2\n255\n')
    with pytest.raises(p.ContractError, match='geometry'):
        p.PPMParser(expected_size=(3, 2)).feed(b'P6\n2 3\n255\n')
    for size in ((0, 1), (-1, 2), (True, 1), (1.0, 2)):
        with pytest.raises(p.ContractError):
            p.PPMParser(expected_size=size)


def test_ppm_payload_boundaries_and_bounded_scanning():
    class NoScanningBuffer(bytearray):
        def count(self, *args):
            raise AssertionError('No full buffer scanning')
        def index(self, *args):
            raise AssertionError('No full buffer scanning')
    image = np.array([10, 32, 35, 0, 255, 10], dtype=np.uint8).reshape(1, 2, 3)
    parser = p.PPMParser(expected_size=(2, 1))
    parser.buffer = NoScanningBuffer()
    frames = []
    for byte in ppm(image) * 3:
        frames.extend(parser.feed(bytes([byte])))
        assert len(parser.buffer) <= max(parser.MAX_HEADER_BYTES, 6)
    parser.finish()
    assert len(frames) == 3
    for frame in frames:
        np.testing.assert_array_equal(frame, image)
    with pytest.raises(p.ContractError, match='truncated'):
        list(p.parse_ppm([ppm(image), ppm(image)[:-1]], expected_size=(2, 1)))
    with pytest.raises(p.ContractError):
        list(p.parse_ppm([ppm(image), b'P3\n2 1\n255\n']))
    with pytest.raises(p.ContractError):
        list(p.parse_ppm([ppm(image), b'garbage']))
    parser = p.PPMParser()
    with pytest.raises(p.ContractError, match='header'):
        parser.feed(b'X' * 100000)
    assert len(parser.buffer) == parser.MAX_HEADER_BYTES + 1


@pytest.mark.parametrize('score, accepted', [
    (np.float32(.90), True), (np.nextafter(np.float32(.90), np.float32(0)), False),
    (np.nextafter(np.float32(.90), np.float32(1)), True),
    (np.float32('nan'), False), (np.float32('inf'), False)])
def test_float32_threshold_boundary(score, accepted):
    rows = np.asarray(detection(), dtype=np.float32)
    rows[0, 14] = score
    if accepted:
        assert p.single_detection(rows)[1] == float(score)
    else:
        with pytest.raises(p.ContractError):
            p.single_detection(rows)


def test_threshold_manifest_and_coherent_mutation_seal(manifest_case):
    raw, cat, snapshot, env, out, doc = manifest_case
    doc['frames'][0]['score'] = float(np.float32(.90))
    p.write_manifest(out, doc, cat, snapshot, env, raw)
    altered = deepcopy(doc)
    frame = altered['frames'][0]
    frame['detector_box'][0] += 1
    g = p.letterbox_geometry(*frame['oriented_size'])
    frame['mapped_box'] = p.remap_detection(frame['detector_box'], g)
    frame['expanded_bounds'], frame['crop_bounds'] = p.crop_bounds(frame['mapped_box'], *frame['oriented_size'])
    # Internal geometry consistency is not a substitute for the original seal.
    p.validate_manifest(altered, cat, snapshot, env, out)
    store(out / 'manifest.json', altered)
    with pytest.raises(p.ContractError, match='seal'):
        p.read_completed_manifest(out, cat, snapshot, env)
