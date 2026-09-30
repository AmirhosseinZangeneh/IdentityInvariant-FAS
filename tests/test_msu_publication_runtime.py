"""Transport fixtures are generated locally; never discover licensed datasets."""
import os
from pathlib import Path
import subprocess
import sys
from types import MappingProxyType, SimpleNamespace

import pytest

from identity_invariant_fas.data import msu_publication_preprocessing as p
from identity_invariant_fas.data import msu_publication_runtime as r


class TestPureTransport:
    def test_incremental_large_stdout_and_stderr(self):
        # Alternate enough data on BOTH pipes to fill Windows pipe buffers.
        command = [sys.executable, '-c',
            "import os; [(os.write(1,b'x'*65536),os.write(2,b'w'*65536)) for _ in range(20)]"]
        chunks = []
        with pytest.raises(p.ContractError, match='stderr='):
            r.stream_process(command, lambda b: chunks.append(len(b)))
        assert sum(chunks) == 20 * 65536
        assert max(chunks) <= 65536

    @pytest.mark.parametrize('program,reason', [
        ("import sys; sys.exit(7)", 'exit=7'),
        ("import sys; sys.stderr.write('warning: fixture\\n')", 'warning: fixture'),
        ("import sys; sys.stderr.write('error: fixture\\n')", 'error: fixture'),
    ])
    def test_fail_closed_diagnostics(self, program, reason):
        with pytest.raises(p.ContractError, match=reason):
            r.stream_process([sys.executable, '-c', program], lambda b: None)

    @pytest.mark.parametrize('payload', [b'garbage\n\n\n', b'P6\n2 2\n255\nshort'])
    def test_malformed_stdout(self, payload):
        parser = p.PPMParser()
        with pytest.raises(p.ContractError):
            r.stream_process([sys.executable, '-c', f'import sys; sys.stdout.buffer.write({payload!r})'],
                             parser.feed)
            parser.finish()

    def test_diagnostic_overflow(self):
        with pytest.raises(p.ContractError, match='safety limit'):
            r.stream_process([sys.executable, '-c', "import os; os.write(2,b'w'*131072)"],
                             lambda b: None, diagnostic_limit=1024)

    def test_timeout(self):
        with pytest.raises(p.ContractError, match='timeout'):
            r.stream_process([sys.executable, '-c', 'import time; time.sleep(10)'],
                             lambda b: None, timeout=.2)

    def test_clean_fragmented_ppm(self):
        parser = p.PPMParser((2, 1))
        frames = []
        r.stream_process([sys.executable, '-c',
            "import os; [os.write(1,bytes([v])) for v in b'P6\\n2 1\\n255\\nabcdef']"],
            lambda b: frames.extend(parser.feed(b)))
        parser.finish()
        assert len(frames) == 1 and frames[0].tobytes() == b'abcdef'

    def test_no_path_lookup(self):
        with pytest.raises(p.ContractError):
            r.Executable('ffmpeg.exe', 'a' * 64, 1, '').verify('ffmpeg')

    def test_archive_mismatch_before_extraction(self, tmp_path):
        archive = tmp_path / 'fake.zip'
        archive.write_bytes(b'wrong')
        with pytest.raises(p.ContractError, match='hash mismatch'):
            r.extract_executables(archive, tmp_path / 'bin')
        assert not (tmp_path / 'bin').exists()


class TestApprovedExecutableBinding:
    @pytest.mark.parametrize('role', ['ffmpeg', 'ffprobe'])
    @pytest.mark.parametrize('kind', ['python', 'arbitrary'])
    def test_caller_declared_identity_rejected(self, tmp_path, monkeypatch, role, kind):
        path = Path(sys.executable).absolute() if kind == 'python' else tmp_path / 'arbitrary.exe'
        if kind == 'arbitrary':
            path.write_bytes(b'arbitrary executable with correct caller metadata')
        claimed = r.Executable(str(path), p.stream_sha256(path), path.stat().st_size,
                               role + ' version 8.1.2-essentials_build-www.gyan.dev')
        def forbidden(*args, **kwargs):
            raise AssertionError('unapproved executable was launched')
        monkeypatch.setattr(subprocess, 'run', forbidden)
        with pytest.raises(p.ContractError, match='artifact'):
            claimed.verify(role)
        # Entrypoints must also ignore a caller's no-op verify method.
        forged = SimpleNamespace(path=str(path), verify=lambda *args: None)
        with pytest.raises(p.ContractError, match='artifact'):
            if role == 'ffprobe':
                r.probe(forged, str(tmp_path / 'unused.mp4'), 'h264')
            else:
                r.decode({'ffmpeg': forged}, None, None, None, None, 'h264')

    @pytest.fixture
    def tiny_approved(self, tmp_path, monkeypatch):
        # Explicit test-only trust root, never configurable by the runtime API.
        path = tmp_path / 'fixture.exe'
        path.write_bytes(b'approved fixture')
        version = '8.1.2-essentials_build-www.gyan.dev'
        spec = r.ApprovedExecutableSpec(p.stream_sha256(path), path.stat().st_size, version)
        monkeypatch.setattr(r, 'APPROVED_EXECUTABLES', MappingProxyType({'ffmpeg': spec}))
        return path, spec

    @pytest.mark.parametrize('replacement', [b'changed! fixture', b'short'])
    def test_mutated_bytes_and_wrong_size(self, tiny_approved, replacement, monkeypatch):
        path, _ = tiny_approved
        path.write_bytes(replacement)
        monkeypatch.setattr(subprocess, 'run', lambda *a, **kw: pytest.fail('must reject before launch'))
        with pytest.raises(p.ContractError, match='artifact'):
            r.verify_approved_executable(path, 'ffmpeg')

    @pytest.mark.parametrize('version', ['8.1.3-essentials_build-www.gyan.dev',
        '8.1.2-other-build', '8.1.2-essentials_build-www.gyan.dev-extra'])
    def test_wrong_actual_version_despite_valid_bytes(self, tiny_approved, monkeypatch, version):
        path, spec = tiny_approved
        stdout = f'ffmpeg version {version}\nconfiguration: fixture\nlibavcodec\nlibavformat\n'.encode()
        monkeypatch.setattr(subprocess, 'run', lambda *a, **kw: SimpleNamespace(
            returncode=0, stderr=b'', stdout=stdout))
        claimed = r.Executable(str(path), spec.sha256, spec.size, 'caller claims correct version')
        with pytest.raises(p.ContractError, match='approved executable version'):
            claimed.verify('ffmpeg')

    def test_version_evidence_is_observed_not_claimed(self, tiny_approved, monkeypatch):
        path, spec = tiny_approved
        stdout = f'ffmpeg version {spec.version}\nconfiguration: fixture\nlibavcodec\nlibavformat\n'.encode()
        def run(argv, **kwargs):
            assert argv == [str(path), '-version'] and kwargs['shell'] is False
            return SimpleNamespace(returncode=0, stderr=b'', stdout=stdout)
        monkeypatch.setattr(subprocess, 'run', run)
        verified = r.Executable(str(path), '0'*64, 0, 'invented').verify('ffmpeg')
        assert (verified.sha256, verified.size, verified.version) == (spec.sha256, spec.size, stdout.decode())

    def test_change_during_version_check_rejected(self, tiny_approved, monkeypatch):
        path, spec = tiny_approved
        def run(*args, **kwargs):
            path.write_bytes(b'changed! fixture')
            return SimpleNamespace(returncode=0, stderr=b'', stdout=(
                f'ffmpeg version {spec.version}\nconfiguration: fixture\nlibavcodec\nlibavformat').encode())
        monkeypatch.setattr(subprocess, 'run', run)
        with pytest.raises(p.ContractError, match='artifact hash mismatch'):
            r.verify_approved_executable(path, 'ffmpeg')


@pytest.mark.runtime_artifact
class TestRealArtifactSyntheticMedia:
    def test_approved_binaries_and_role_swap(self, monkeypatch):
        paths = {role: os.environ.get('MSU_RUNTIME_' + role.upper()) for role in ('ffmpeg', 'ffprobe')}
        if not all(paths.values()):
            pytest.skip('Set explicit MSU_RUNTIME_FFMPEG and MSU_RUNTIME_FFPROBE paths')
        checked = {role: r.verify_approved_executable(path, role) for role, path in paths.items()}
        for role, executable in checked.items():
            assert executable.sha256 == r.APPROVED_EXECUTABLES[role].sha256
            assert executable.size == r.APPROVED_EXECUTABLES[role].size
        monkeypatch.setattr(subprocess, 'run', lambda *a, **kw: pytest.fail('role swap must fail before launch'))
        with pytest.raises(p.ContractError, match='artifact'):
            r.probe(checked['ffmpeg'], str(Path('unused.mp4').absolute()), 'h264')
        with pytest.raises(p.ContractError, match='artifact'):
            r.decode({'ffmpeg': checked['ffprobe'], 'ffprobe': checked['ffmpeg']},
                     None, None, None, None, 'h264')

    def test_yunet_candidate(self):
        model = os.environ.get('MSU_RUNTIME_MODEL')
        if not model:
            pytest.skip('Set MSU_RUNTIME_MODEL to the explicit verified 2023mar model')
        retained = os.environ.get('MSU_RUNTIME_YUNET_EVIDENCE')
        if retained:
            result = r.verify_retained_yunet(retained, model)
            assert result['detections'] == [] and result['repeat_identical']
            return
        import cv2
        import numpy as np
        import importlib.metadata as metadata
        assert cv2.__version__ == '4.11.0'
        installed = {d.metadata['Name'].lower() for d in metadata.distributions()}
        assert installed.intersection({'opencv-python', 'opencv-contrib-python',
            'opencv-python-headless', 'opencv-contrib-python-headless'}) == {'opencv-python'}
        assert not any(n.startswith('onnxruntime') for n in installed)
        # A contaminated parent and obsolete parent variable cannot contaminate the child.
        previous = os.environ.get('OPENCV_FORCE_DNN_ENGINE')
        os.environ['OPENCV_FORCE_DNN_ENGINE'] = '4'
        try:
            for shape in ((640,640,3), (61,97,3)):
                image = np.zeros(shape, np.uint8)
                first, second = r.infer_yunet(model, image), r.infer_yunet(model, image)
                assert first == second
                assert first['cv2_absent_before_import'] is True
                assert first['engine_environment'] is None
                assert first['input_size'] == [640,640]
                assert first['settings'] == p.DETECTOR
                assert first['detections'] == []
                assert first['onnx_runtime_used'] is False
                with pytest.raises(p.ContractError, match='exactly one'):
                    r.select_inference(first)
        finally:
            if previous is None:
                os.environ.pop('OPENCV_FORCE_DNN_ENGINE', None)
            else:
                os.environ['OPENCV_FORCE_DNN_ENGINE'] = previous

    def test_qualification(self, tmp_path):
        archive, model = os.environ.get('MSU_RUNTIME_ARCHIVE'), os.environ.get('MSU_RUNTIME_MODEL')
        if not archive or not model:
            pytest.skip('Set MSU_RUNTIME_ARCHIVE and MSU_RUNTIME_MODEL to explicit frozen artifacts; no PATH fallback')
        assert Path(archive).is_absolute() and Path(model).is_absolute()
        script = Path(__file__).resolve().parents[1] / 'scripts' / 'qualify_msu_publication_runtime.py'
        output = Path(os.environ.get('MSU_RUNTIME_QUALIFICATION_OUTPUT', str(tmp_path / 'qualification')))
        command = [sys.executable, str(script), '--archive', archive, '--model', model,
                   '--output', str(output)]
        retained = os.environ.get('MSU_RUNTIME_YUNET_EVIDENCE')
        if retained:
            command.extend(['--yunet-evidence', retained])
        run = subprocess.run(command,
                             capture_output=True, timeout=240)
        assert run.returncode == 0, (run.stdout.decode(), run.stderr.decode())
        report = p.read_json(output / 'qualification.json')
        assert report['runtime_qualification_passed']
        assert all(report['checks'].values())
        assert report['yunet']['worker']['cv2_absent_before_import']
        assert report['yunet']['worker']['detections'] == []
        summary = p.read_json(output / 'qualification-summary.json')
        provenance = summary['implementation_provenance']
        assert provenance['raw_qualification_report_sha256'] == p.stream_sha256(output / 'qualification.json')
        root = Path(__file__).resolve().parents[1]
        assert len(provenance['files']) == 6
        assert all(p.stream_sha256(root / name) == sha for name, sha in provenance['files'].items())
        assert summary['scientific_policy_sha256'] == report['policy_sha256']
        text = p.canonical_bytes(summary).decode()
        assert str(root) not in text and root.as_posix() not in text
        assert summary['scientific_policy_sha256'] == p.digest({
            'policy': p.POLICY, 'detector': p.DETECTOR, 'runtime': p.RUNTIME, 'archive': p.ARCHIVE_SHA256})
