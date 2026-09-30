"""Qualify pinned artifacts on newly generated media ONLY. Never discovers MSU data."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np

from identity_invariant_fas.data import msu_publication_preprocessing as p
from identity_invariant_fas.data import msu_publication_runtime as r

FLAGS = ('ffmpeg_archive_verified', 'ffmpeg_executables_verified', 'ffmpeg_runtime_qualified',
         'ffprobe_runtime_qualified', 'ppm_transport_qualified', 'synthetic_h264_qualified',
         'synthetic_prores_qualified', 'yunet_model_verified', 'yunet_worker_isolated',
         'yunet_runtime_qualified', 'repeatability_qualified', 'failure_paths_qualified',
         'rotation_qualified')

IMPLEMENTATION_FILES = (
    'src/identity_invariant_fas/__init__.py',
    'src/identity_invariant_fas/data/__init__.py',
    'src/identity_invariant_fas/data/msu_publication_preprocessing.py',
    'src/identity_invariant_fas/data/msu_publication_runtime.py',
    'src/identity_invariant_fas/data/msu_yunet_worker.py',
    'scripts/qualify_msu_publication_runtime.py',
)


def implementation_provenance():
    root = Path(__file__).resolve().parents[1]
    return {name: p.stream_sha256(root / name) for name in IMPLEMENTATION_FILES}


def portable_summary(report, raw_sha256):
    """Allowlisted portable evidence; implementation provenance is not policy."""
    summary = dict(schema='msu-runtime-portable-summary-v2', policy=report['policy'],
        amendments=['001', '002'], scientific_policy_sha256=report['policy_sha256'],
        implementation_provenance={'files': report['implementation_provenance'],
                                   'raw_qualification_report_sha256': raw_sha256},
        environment=report['environment'], checks=report['checks'],
        runtime_qualification_passed=report['runtime_qualification_passed'],
        licensed_media_accessed=False, eight_video_qualification='not run',
        msu_experiment_ready=False)
    if report['artifacts']:
        a = report['artifacts']
        summary['artifacts'] = dict(archive_sha256=a['archive_sha256'],
            yunet={k: a['model'][k] for k in ('size', 'sha256')},
            executables={name: {k: exe[k] for k in ('sha256', 'size', 'version')}
                         for name, exe in a['executables'].items()})
    summary['synthetic_media'] = {}
    for name in ('h264', 'prores', 'rotation'):
        if name not in report['decoder']:
            continue
        evidence = report['decoder'][name]
        decoded = evidence['decode']
        summary['synthetic_media'][name] = {
            **{k: decoded[k] for k in ('codec', 'n', 'source_size', 'oriented_size',
                                       'rotations', 'matrix', 'frame_hashes', 'returncode',
                                       'probe_returncode')},
            'ordinals': list(p.sample_ordinals(decoded['n'])),
            'clean_diagnostics': not decoded['stderr'].strip() and not decoded['probe_stderr'].strip(),
            'probe_frame_count': len(decoded['probe_frame_sizes']),
        }
        for key in ('repeat_identical', 'sign_axis_verified'):
            if key in evidence:
                summary['synthetic_media'][name][key] = evidence[key]
    summary['failure_paths'] = {name: 'rejected' for name in report['decoder'].get('failure_paths', {})}
    if 'worker' in report['yunet']:
        worker = report['yunet']['worker']
        summary['yunet'] = {k: worker[k] for k in ('cv2_version', 'python', 'settings',
            'cv2_absent_before_import', 'engine_environment', 'threads', 'opencl', 'input_size',
            'detections', 'repeat_identical', 'onnx_runtime_used', 'runtime_fingerprints')}
        summary['yunet']['evidence_reused'] = 'retained_evidence_sha256' in report['yunet']
    return summary


def generate(executable, root, codec):
    camera = 'android' if codec == 'h264' else 'laptop'
    video = p._video('002', camera)
    relative = video.removeprefix('MSU-MFSD:')
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    command = [executable.path, '-nostdin', '-hide_banner', '-loglevel', 'warning', '-nostats',
               '-f', 'lavfi', '-i', 'testsrc2=size=96x64:rate=10', '-frames:v', '35',
               '-threads', '1', '-c:v', 'libx264' if codec == 'h264' else 'prores_ks',
               '-pix_fmt', 'yuv420p' if codec == 'h264' else 'yuv422p10le', str(path)]
    executable = r.verify_approved_executable(executable.path, 'ffmpeg')
    run = subprocess.run(command, capture_output=True, timeout=120)
    p._require(run.returncode == 0 and not run.stderr.strip(), repr(run.stderr))
    source = p.Source(relative, '002', 'train', 0, camera, p.CAMERAS[camera], None,
                      None, 'SD', 'scene01', path.stat().st_size, video)
    catalog = p.SourceCatalog((source,), 'a' * 64, 'b' * 64, ())
    snapshot = p.source_snapshot(root, catalog)
    return source, snapshot, command


def qualify(archive, model, output, yunet_evidence=None):
    output = Path(output).absolute()
    # All videos are created beneath this fresh private directory.
    output.mkdir(parents=True, exist_ok=False)
    provenance = implementation_provenance()
    report = dict(schema='msu-synthetic-runtime-qualification-v2', policy=p.POLICY,
        implementation_provenance=provenance,
        policy_sha256=p.digest({'policy': p.POLICY, 'detector': p.DETECTOR,
                               'runtime': p.RUNTIME, 'archive': p.ARCHIVE_SHA256}),
        environment=dict(python=platform.python_version(), platform=platform.platform(),
            architecture=platform.machine(), versions={n: importlib.metadata.version(n)
                                                       for n, _ in p.RUNTIME[1:]},
            engine_environment=None, engine='opencv-4.11-native-dnn', backend=3, target=0,
            onnx_runtime_used=False),
        artifacts={}, decoder={}, yunet={}, checks=dict.fromkeys(FLAGS, False),
        runtime_qualification_passed=False, qualification_status='failed',
        licensed_media_accessed=False, eight_video_qualification='pending')
    checks = report['checks']
    try:
        p.verify_file(model, p.MODEL_SHA256, 232589)
        checks['yunet_model_verified'] = True
        # Qualify the amended blocker first, even on later reproduction runs.
        import cv2
        report['yunet']['parent_cv2_imported'] = cv2.__version__
        blank = np.zeros((640, 640, 3), dtype=np.uint8)
        if yunet_evidence is not None:
            worker = r.verify_retained_yunet(yunet_evidence, model)
            report['yunet']['retained_evidence_sha256'] = p.stream_sha256(yunet_evidence)
        else:
            worker = r.infer_yunet(str(Path(model).absolute()), blank)
            repeat = r.infer_yunet(str(Path(model).absolute()), blank)
            p._require(worker == repeat, 'worker repeat mismatch')
        p._require(worker['detections'] == [], 'blank zero-detection qualification')
        try:
            r.select_inference(worker)
        except p.ContractError:
            report['yunet']['scientific_selection'] = 'failed: exactly one required (expected for blank)'
        else:
            raise p.ContractError('blank selection unexpectedly passed')
        report['yunet']['worker'] = worker
        checks['yunet_worker_isolated'] = checks['yunet_runtime_qualified'] = True
        p.verify_file(archive, p.ARCHIVE_SHA256)
        checks['ffmpeg_archive_verified'] = True
        executables = r.extract_executables(archive, output / 'bin')
        report['artifacts'] = dict(archive_sha256=p.ARCHIVE_SHA256,
            model={'path': str(model), 'size': 232589, 'sha256': p.MODEL_SHA256},
            executables={k: asdict(v) for k, v in executables.items()})
        checks['ffmpeg_executables_verified'] = True
        for codec in ('h264', 'prores'):
            root = output / codec
            source, snapshot, command = generate(executables['ffmpeg'], root, codec)
            runs = [r.decode(executables, root, source, snapshot, p.digest(snapshot), codec)
                    for _ in range(2)]
            first, second = runs[0][0], runs[1][0]
            p._require(first.n == second.n == 35 and first.source_size == (96, 64),
                       'synthetic count/dimension mismatch')
            ordinals = p.sample_ordinals(35)
            p._require(ordinals[0] == 0 and ordinals[-1] == 34 and len(set(ordinals)) == 30,
                       'sampler endpoints')
            p._require(first.frame_hashes == second.frame_hashes, 'repeat pixel mismatch')
            report['decoder'][codec] = dict(generation_command=command,
                decode=asdict(first), probe=runs[0][1], ordinals=ordinals,
                selected_rgb_sha256=[first.frame_hashes[i] for i in ordinals],
                repeat_identical=True)
            checks['synthetic_' + codec + '_qualified'] = True
        for flag in ('ffmpeg_runtime_qualified', 'ffprobe_runtime_qualified',
                     'ppm_transport_qualified', 'repeatability_qualified'):
            checks[flag] = True
        failures = {}
        invalid = output / 'invalid.bin'
        invalid.write_bytes(b'not a video\x00')
        truncated = output / 'truncated.mp4'
        original = output / 'h264' / report['decoder']['h264']['decode']['video_id'].removeprefix('MSU-MFSD:')
        truncated.write_bytes(original.read_bytes()[:128])
        for name, path in (('nonexistent', output / 'absent.mp4'), ('invalid', invalid),
                           ('truncated', truncated)):
            for tool in ('ffmpeg', 'ffprobe'):
                builder = p.build_decode_command if tool == 'ffmpeg' else p.build_probe_command
                verified = r.verify_approved_executable(executables[tool].path, tool)
                try:
                    r.stream_process(builder(verified.path, path, 'h264'), lambda block: None)
                except p.ContractError as exc:
                    failures[name + '_' + tool] = str(exc)
                else:
                    raise p.ContractError('invalid input unexpectedly succeeded')
        report['decoder']['failure_paths'] = failures
        checks['failure_paths_qualified'] = True
        # Container rotation only: remux the generated H.264 fixture without re-encoding.
        rotated_root = output / 'rotation'
        rotated_path = rotated_root / report['decoder']['h264']['decode']['video_id'].removeprefix('MSU-MFSD:')
        rotated_path.parent.mkdir(parents=True)
        rotation_command = [executables['ffmpeg'].path, '-nostdin', '-hide_banner',
            '-loglevel', 'warning', '-nostats', '-display_rotation:v:0', '90',
            '-i', str(original), '-map', '0:v:0', '-c:v', 'copy', str(rotated_path)]
        r.verify_approved_executable(executables['ffmpeg'].path, 'ffmpeg')
        rotation_run = subprocess.run(rotation_command, capture_output=True, timeout=120)
        p._require(rotation_run.returncode == 0 and not rotation_run.stderr.strip(),
                   repr(rotation_run.stderr))
        relative = rotated_path.relative_to(rotated_root).as_posix()
        rotated_source = p.Source(relative, '002', 'train', 0, 'android', p.CAMERAS['android'],
            None, None, 'SD', 'scene01', rotated_path.stat().st_size, 'MSU-MFSD:' + relative)
        rotated_catalog = p.SourceCatalog((rotated_source,), 'a' * 64, 'b' * 64, ())
        rotated_snapshot = p.source_snapshot(rotated_root, rotated_catalog)
        rotated, metadata = r.decode(executables, rotated_root, rotated_source,
                                    rotated_snapshot, p.digest(rotated_snapshot), 'h264')
        p._require(p.normalize_orientation(rotated.rotations, rotated.matrix) == 90 and
                   rotated.oriented_size == (64, 96), 'rotation normalization')
        expected_hashes = []
        parser = p.PPMParser((96, 64))
        def rotated_expected(block):
            for frame in parser.feed(block):
                expected_hashes.append(p.pixel_sha256(np.ascontiguousarray(np.rot90(frame))))
        r.verify_approved_executable(executables['ffmpeg'].path, 'ffmpeg')
        r.stream_process(p.build_decode_command(executables['ffmpeg'].path, original, 'h264'),
                         rotated_expected)
        parser.finish()
        p._require(tuple(expected_hashes) == rotated.frame_hashes, 'rotation sign/axis mismatch')
        report['decoder']['rotation'] = dict(generation_command=rotation_command,
            decode=asdict(rotated), probe=metadata, sign_axis_verified=True)
        checks['rotation_qualified'] = True
        lock = p.EnvironmentLock(p.ARCHIVE_SHA256, p.MODEL_SHA256,
            executables['ffmpeg'].sha256, executables['ffprobe'].sha256, p.RUNTIME,
            executables['ffmpeg'].version, executables['ffprobe'].version,
            worker['opencv_build'], tuple(tuple(x) for x in worker['runtime_fingerprints']))
        p.validate_artifact_files(lock, archive, model, executables['ffmpeg'].path,
                                  executables['ffprobe'].path)
        report['environment_lock'] = asdict(lock)
        p._require(implementation_provenance() == provenance, 'implementation changed during qualification')
        report['runtime_qualification_passed'] = all(checks.values())
        report['qualification_status'] = 'passed'
    except Exception as exc:
        report['blocker'] = f'{type(exc).__name__}: {exc}'
    p.write_once(output / 'qualification.json', report)
    p.write_once(output / 'qualification-summary.json',
                 portable_summary(report, p.stream_sha256(output / 'qualification.json')))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', required=True, type=Path)
    parser.add_argument('--model', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path, help='Fresh ignored directory')
    parser.add_argument('--yunet-evidence', type=Path, help='Reuse verified local prior inference evidence')
    args = parser.parse_args()
    report = qualify(args.archive.absolute(), args.model.absolute(), args.output, args.yunet_evidence)
    print(json.dumps({k: report[k] for k in ('checks', 'runtime_qualification_passed')}, indent=2))
    if 'blocker' in report:
        print(report['blocker'])
    return 0 if report['runtime_qualification_passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
