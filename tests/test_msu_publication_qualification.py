"""Synthetic orchestration only. Recording-shaped fixtures are private temp bytes."""
from copy import deepcopy
from dataclasses import asdict, replace
import gc
from pathlib import Path
from types import SimpleNamespace
import weakref

import numpy as np
import pytest

from identity_invariant_fas.data import msu_publication_preprocessing as p
from identity_invariant_fas.data import msu_publication_runtime as r
from identity_invariant_fas.data import msu_publication_qualification as q
from test_msu_publication_preprocessing import decoded, detection, environment


@pytest.fixture(autouse=True)
def recording_boundary(tmp_path, monkeypatch):
    """Any accidental real recording access fails before an OS open/stat/resolve."""
    for method in ('open', 'stat', 'resolve'):
        original = getattr(Path, method)
        def guarded(path, *args, _original=original, **kwargs):
            if path.suffix.lower() in ('.mp4', '.mov', '.face'):
                assert path.is_relative_to(tmp_path), 'licensed or out-of-fixture access'
                assert path.suffix != '.face'
            return _original(path, *args, **kwargs)
        monkeypatch.setattr(Path, method, guarded)


@pytest.fixture
def metadata(tmp_path):
    raw = tmp_path / 'raw'
    raw.mkdir()
    for name, text in (('README.txt', 'synthetic'), ('train_sub_list.txt', ' '.join(p.TRAIN)),
                       ('test_sub_list.txt', ' '.join(p.TEST))):
        (raw / name).write_text(text)
    fingerprints = {n: p.stream_sha256(raw / n) for n in
                    ('README.txt', 'train_sub_list.txt', 'test_sub_list.txt')}
    rows = []
    for client in (*p.TRAIN, *p.TEST):
        for camera in p.CAMERAS:
            for attack in (None, *p.MEDIA):
                video = p._video(client, camera, attack)
                rows.append(asdict(p.Source(video.removeprefix('MSU-MFSD:'), client,
                    'train' if client in p.TRAIN else 'test', int(attack is not None), camera,
                    p.CAMERAS[camera], attack, p.MEDIA.get(attack), 'SD', 'scene01', len(video.encode()), video)))
    catalog = dict(schema_version=1, dataset='MSU-MFSD', protocol='official_public_train_test',
        train_clients=list(p.TRAIN), test_clients=list(p.TEST), source_fingerprints=fingerprints,
        recordings=sorted(rows, key=lambda s: s['filepath']))
    lock = dict(normalized_metadata_sha256=p.digest(catalog), source_fingerprints=fingerprints,
        partitions={s: {'client_ids': list(ids)} for s, ids in (('train', p.TRAIN), ('test', p.TEST))})
    cp, lp = tmp_path / 'catalog.json', tmp_path / 'lock.json'
    p.write_once(cp, catalog)
    p.write_once(lp, lock)
    return raw, cp, lp


def put_sources(raw, sources):
    for source in sources:
        path = raw / source.filepath
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(source.video_id.encode())


def guard_paths(monkeypatch, raw, allowed):
    touched = []
    for method in ('open', 'stat', 'resolve'):
        original = getattr(Path, method)
        def guarded(path, *args, _original=original, **kwargs):
            if path.suffix in ('.mp4', '.mov'):
                assert path in allowed, f'unauthorized recording path: {path}'
                touched.append(path)
            return _original(path, *args, **kwargs)
        monkeypatch.setattr(Path, method, guarded)
    return touched


def test_metadata_no_recording_files_or_path_operations(metadata, monkeypatch):
    guard_paths(monkeypatch, metadata[0], set())
    cat = p.load_source_catalog_metadata(*metadata)
    assert len(cat.sources) == 280 and len([s for s in cat.sources if s.pad_partition == 'train']) == 120
    assert not (metadata[0] / 'scene01').exists()


@pytest.mark.parametrize('change', [
    lambda d: d['recordings'].pop(), lambda d: d['recordings'].append(d['recordings'][0]),
    lambda d: d['train_clients'].reverse(), lambda d: d['recordings'][0].update(video_id='wrong'),
    lambda d: d['recordings'][0].update(capture_model='wrong')])
def test_metadata_semantics_without_media(metadata, monkeypatch, change):
    raw, cp, lp = metadata
    doc, lock = p.read_json(cp), p.read_json(lp)
    change(doc)
    lock['normalized_metadata_sha256'] = p.digest(doc)
    cp.write_bytes(p.canonical_bytes(doc))
    lp.write_bytes(p.canonical_bytes(lock))
    guard_paths(monkeypatch, raw, set())
    with pytest.raises(p.ContractError):
        p.load_source_catalog_metadata(*metadata)


def test_scoped_snapshot_exact_eight_and_full_scope_preserved(metadata, monkeypatch):
    cat = p.load_source_catalog_metadata(*metadata)
    sources = p.qualification_sources(cat, p.TRAIN_QUALIFICATION)
    put_sources(metadata[0], sources)
    allowed = {metadata[0] / s.filepath for s in sources}
    touched = guard_paths(monkeypatch, metadata[0], allowed)
    snap = p.source_snapshot(metadata[0], cat, p.TRAIN_QUALIFICATION)
    assert set(touched) == allowed
    assert tuple(s['source']['video_id'] for s in snap['sources']) == p.TRAIN_QUALIFICATION
    assert len(snap['sources']) == 8 and snap['projection_sha256'] == cat.projection_sha256
    p.verify_snapshot(metadata[0], cat, snap, p.TRAIN_QUALIFICATION)
    # A full-release operation does NOT silently become scoped.
    with pytest.raises(AssertionError, match='unauthorized'):
        p.source_snapshot(metadata[0], cat)


@pytest.mark.parametrize('scope', [p.TRAIN_QUALIFICATION[::-1], p.TRAIN_CODEC_GATE,
    (*p.TRAIN_QUALIFICATION, p.TEST_CODEC_CHECK[0]), (*p.TRAIN_QUALIFICATION[:7], p.TEST_CODEC_CHECK[0])])
def test_wrong_snapshot_scope_fails_before_access(metadata, monkeypatch, scope):
    cat = p.load_source_catalog_metadata(*metadata)
    guard_paths(monkeypatch, metadata[0], set())
    with pytest.raises(p.ContractError, match='exact ordered'):
        p.source_snapshot(metadata[0], cat, scope)


@pytest.mark.parametrize('change', ['missing', 'extra', 'reorder', 'test', 'projection'])
def test_snapshot_mode_mutations(metadata, change):
    cat = p.load_source_catalog_metadata(*metadata)
    sources = p.qualification_sources(cat, p.TRAIN_QUALIFICATION)
    put_sources(metadata[0], sources)
    snap = p.source_snapshot(metadata[0], cat, p.TRAIN_QUALIFICATION)
    p._snapshot_sources(snap, cat, 'training_qualification')
    if change == 'missing': snap['sources'].pop()
    if change == 'extra': snap['sources'].append(snap['sources'][0])
    if change == 'reorder': snap['sources'].reverse()
    if change == 'test': snap['sources'][0]['source'] = asdict(next(s for s in cat.sources if s.pad_partition == 'test'))
    if change == 'projection': snap['projection_sha256'] = '0'*64
    with pytest.raises(p.ContractError):
        p._snapshot_sources(snap, cat, 'training_qualification')


def test_publication_still_280_sources_and_8400_frames(metadata, environment, tmp_path):
    cat = p.load_source_catalog_metadata(*metadata)
    put_sources(metadata[0], cat.sources)  # Synthetic tiny bytes only.
    snap = p.source_snapshot(metadata[0], cat)
    assert len(snap['sources']) == 280
    p._snapshot_sources(snap, cat, 'publication')
    image = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
    out = p.fresh_output(metadata[0], tmp_path / 'sample')
    decodes, frames = [], []
    for index, (source, entry) in enumerate(zip(cat.sources, snap['sources'])):
        d = replace(decoded(source.video_id, image, source_sha=entry['sha256'], snapshot_sha=p.digest(snap)),
                    codec=p.source_codec(source))
        decodes.append(d)
        # One real synthetic crop per video; clone its metadata for coverage validation only.
        record = p.make_frame_record(source, entry['sha256'], d, 0, image, detection(), out, f'{index}.png')
        for ordinal in p.sample_ordinals(d.n):
            frames.append({**record, 'ordinal': ordinal, 'output_path': f'{index}/{ordinal}.png'})
    provenance = {k: '4'*64 for k in ('policy_document_sha256', 'amendment_sha256', 'implementation_sha256')}
    doc = p.make_manifest(cat, snap, environment, decodes, frames, provenance, mode='publication')
    assert len(doc['frames']) == 8400
    for key in ('frames', 'decodes'):
        bad = deepcopy(doc)
        bad[key].pop()
        with pytest.raises(p.ContractError): p.validate_manifest(bad, cat, snap, environment)
    bad = deepcopy(snap)
    bad['sources'].pop()
    with pytest.raises(p.ContractError): p._snapshot_sources(bad, cat, 'publication')


@pytest.fixture
def selected_transport(tmp_path, monkeypatch):
    video = p.TRAIN_QUALIFICATION[0]
    source = p.Source(video.removeprefix('MSU-MFSD:'), '005', 'train', 1, 'laptop', p.CAMERAS['laptop'],
                      'iphone_video', p.MEDIA['iphone_video'], 'SD', 'scene01', len(video.encode()), video)
    put_sources(tmp_path, [source])
    cat = p.SourceCatalog((source,), 'a'*64, 'b'*64, ())
    snap = p.source_snapshot(tmp_path, cat)
    image = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
    fields = dict(codec='prores', source_size=(3,2), rotations=(90,), matrix=None, probe_stderr='',
        probe_returncode=0, probe_complete=True, probe_frame_sizes=((3,2),)*35, progressive=True,
        sample_aspect_ratio=(1,1), video_stream_count=1, attached_picture=False,
        color_metadata=tuple((k,None) for k in ('range','matrix','primaries','transfer')))
    checks = []
    def checked(path, role):
        checks.append(('version', role))
        return r.Executable(str(path), ('1' if role == 'ffmpeg' else '2')*64, 1, 'fixture')
    monkeypatch.setattr(r, 'verify_approved_executable', checked)
    monkeypatch.setattr(r, '_verify_approved_bytes', lambda path, role: checks.append(('bytes', role)))
    monkeypatch.setattr(r, 'probe', lambda *args: ({'synthetic': True}, fields))
    count = [35]
    def stream(command, consume):
        assert tuple(command) == p.build_decode_command(tmp_path/'ffmpeg.exe', tmp_path/source.filepath, 'prores')
        for i in range(count[0]):
            ppm = b'P6\n3 2\n255\n' + (image+i).tobytes()
            for start in range(0, len(ppm), 7): consume(ppm[start:start+7])
        return 0, ''
    monkeypatch.setattr(r, 'stream_process', stream)
    args = ({role: SimpleNamespace(path=tmp_path/(role+'.exe')) for role in ('ffmpeg','ffprobe')},
            tmp_path, source, snap, p.digest(snap), 'prores')
    return args, image, checks, count


def test_shared_decode_selected_exact_binding_orientation_and_memory(selected_transport, monkeypatch):
    args, image, checks, count = selected_transport
    ordinary, probe = r.decode(*args)
    refs = []
    orient = p.orient_rgb
    def observed(*args):
        result = orient(*args)
        refs.append(weakref.ref(result))
        return result
    monkeypatch.setattr(p, 'orient_rgb', observed)
    result, raw, frames = r.decode_selected_frames(*args)
    assert result == ordinary and raw == probe
    assert len(result.frame_hashes) == 35 and len(frames) == 30
    assert tuple(f.ordinal for f in frames) == p.sample_ordinals(35)
    assert (frames[0].ordinal, frames[-1].ordinal) == (0,34)
    for frame in frames:
        assert np.array_equal(frame.oriented_rgb, np.rot90(image+frame.ordinal))
        assert p.pixel_sha256(frame.oriented_rgb) == result.frame_hashes[frame.ordinal]
    gc.collect()
    assert {i for i, ref in enumerate(refs) if ref() is not None} == set(p.sample_ordinals(35))
    assert checks.count(('bytes','ffmpeg')) == 2
    count[0] = 34
    with pytest.raises(p.ContractError, match='count/dimension'):
        r.decode_selected_frames(*args)


@pytest.fixture
def orchestrator(metadata, environment, tmp_path, monkeypatch):
    cat = p.load_source_catalog_metadata(*metadata)
    sources = p.qualification_sources(cat, p.TRAIN_QUALIFICATION)
    put_sources(metadata[0], sources)
    touched = guard_paths(monkeypatch, metadata[0], {metadata[0]/s.filepath for s in sources})
    state = {'head': 'a'*40, 'files': {name:'4'*64 for name in (*q.IMPLEMENTATION_FILES,*q.POLICY_FILES)}}
    monkeypatch.setattr(q, 'implementation_state', lambda *args: state)
    monkeypatch.setattr(q, 'verify_runtime_versions', lambda: p.RUNTIME)
    monkeypatch.setattr(q, 'artifact_preflight', lambda *args: ({}, environment))
    monkeypatch.setattr(q, 'CATALOG_SHA256', cat.catalog_sha256)
    monkeypatch.setattr(q, 'fresh_ignored_output', lambda repo, raw, out, git: p.fresh_output(raw,out))
    calls = []
    def decode(exes, raw, source, snapshot, snapshot_sha, codec):
        calls.append(source.video_id)
        entry = next(x for x in snapshot['sources'] if x['source']['video_id'] == source.video_id)
        p.verify_file(raw/source.filepath, entry['sha256'], source.byte_size)
        image = np.arange(18, dtype=np.uint8).reshape(2,3,3)
        d = replace(decoded(source.video_id,image,n=35,source_sha=entry['sha256'],snapshot_sha=snapshot_sha),codec=codec)
        return d, {'synthetic':True}, tuple(r.SelectedFrame(i,image.copy()) for i in p.sample_ordinals(35))
    monkeypatch.setattr(r, 'decode_selected_frames', decode)
    monkeypatch.setattr(r, 'infer_yunet', lambda *args: {'detections':detection()})
    options = dict(raw_root=metadata[0], catalog=metadata[1], protocol_lock=metadata[2],
        archive=tmp_path/'archive.zip', ffmpeg=tmp_path/'ffmpeg.exe', ffprobe=tmp_path/'ffprobe.exe',
        model=tmp_path/'model.onnx', output=tmp_path/'qualification', expected_head='a'*40, git_executable=tmp_path/'git.exe')
    return options, calls, touched


def test_complete_synthetic_orchestrator(orchestrator):
    options, calls, touched = orchestrator
    report = q.run_qualification(**options)
    assert calls == [*p.TRAIN_CODEC_GATE, *p.TRAIN_QUALIFICATION, *p.TRAIN_QUALIFICATION]
    assert report['run_1_crops'] == report['run_2_crops'] == 240
    root = options['output']
    for name in ('run-1','run-2'):
        doc = p.read_json(root/name/'manifest.json')
        assert len(doc['decodes']) == 8 and len(doc['frames']) == 240 and not doc['publication_ready']
        assert len(p.read_json(root/name/'snapshot.json')['sources']) == 8
        assert (root/name/'completion.json').exists()
        assert len(list((root/name/'crops').rglob('*.png'))) == 240
    assert not (root/'phase-a'/'manifest.json').exists()
    review = p.read_json(root/'review'/'index.json')
    assert len(review['pairs']) == 24 and review['human_technical_review'] == 'pending'
    assert [(pair['video_id'],pair['j']) for pair in review['pairs']] == [(v,j) for v in p.TRAIN_QUALIFICATION for j in (0,14,29)]
    assert len(list((root/'review').glob('*.png'))) == 48
    assert report['technical_qualification'] == 'pending_human_review'
    assert not report['full_preprocessing_authorized']
    assert all('client001_' not in str(path) for path in touched)


@pytest.mark.parametrize('stage,limit', [('gate',1), ('gate',2), ('gate',3), ('gate',4), ('run-1',5), ('run-2',13)])
@pytest.mark.parametrize('bad_detections', [[], detection()+detection()])
def test_failure_stops_no_retry_or_next_phase(orchestrator, monkeypatch, stage, limit, bad_detections):
    options, calls, touched = orchestrator
    def infer(*args):
        return {'detections': bad_detections if len(calls) == limit else detection()}
    monkeypatch.setattr(r, 'infer_yunet', infer)
    with pytest.raises(p.ContractError, match='exactly one'): q.run_qualification(**options)
    assert len(calls) == limit
    root = options['output']
    assert (root/'failure.json').exists() and not (root/'report.json').exists()
    assert not (root/'review'/'index.json').exists()
    phase = 'phase-a' if stage == 'gate' else stage
    assert (root/phase/'failure.json').exists() and not (root/phase/'completion.json').exists()
    if stage == 'gate':
        assert set(touched) <= {options['raw_root']/v.removeprefix('MSU-MFSD:') for v in p.TRAIN_CODEC_GATE}
        assert not (root/'run-1').exists()
    if stage == 'run-1': assert not (root/'run-2').exists()


def test_repeat_detector_landmark_mismatch_blocks_review(orchestrator, monkeypatch):
    options, calls, _ = orchestrator
    def infer(*args):
        rows = detection()
        if len(calls) >= 13: rows[0][4] = 1.0  # Same box/crop, different normalized landmark.
        return {'detections': rows}
    monkeypatch.setattr(r, 'infer_yunet', infer)
    with pytest.raises(p.ContractError, match='repeatability'): q.run_qualification(**options)
    root = options['output']
    mismatch = p.read_json(root/'repeatability-mismatch.json')
    assert mismatch['video_id'] == p.TRAIN_QUALIFICATION[0] and mismatch['j'] == 0 and mismatch['ordinal'] == 0
    assert mismatch['field'].startswith('detections')
    assert not (root/'run-2'/'completion.json').exists() and not (root/'review'/'index.json').exists()


def test_cli_explicit_arguments_and_no_execution_on_missing(monkeypatch):
    monkeypatch.setattr(q,'run_qualification',lambda **kw: pytest.fail('must parse before execution'))
    with pytest.raises(SystemExit): q.main([])
    with pytest.raises(SystemExit): q.main(['--raw-root','explicit'])


def test_wrong_runtime_before_artifact_or_media_access(orchestrator, monkeypatch):
    options, calls, touched = orchestrator
    # Restore the real verifier, with a deterministic wrong installed version.
    monkeypatch.setattr(q, 'verify_runtime_versions', _VERIFY_RUNTIME)
    monkeypatch.setattr(q.platform,'python_version',lambda:'3.12.10')
    monkeypatch.setattr(q.importlib.metadata,'version',lambda name: '5.0.0.93' if name=='opencv-python' else dict(p.RUNTIME)[name])
    monkeypatch.setattr(q,'artifact_preflight',lambda *a: pytest.fail('wrong runtime must stop first'))
    with pytest.raises(p.ContractError, match='wrong isolated runtime'): q.run_qualification(**options)
    assert not calls and not touched


_VERIFY_RUNTIME = q.verify_runtime_versions


def test_preflight_order_and_catalog_failure_no_media(orchestrator, monkeypatch):
    options, calls, touched = orchestrator
    order = []
    for name in ('implementation_state','verify_runtime_versions','artifact_preflight'):
        original = getattr(q,name)
        def call(*args,_name=name,_original=original):
            order.append(_name)
            return _original(*args)
        monkeypatch.setattr(q,name,call)
    monkeypatch.setattr(q,'CATALOG_SHA256','0'*64)
    with pytest.raises(p.ContractError,match='catalog digest'): q.run_qualification(**options)
    assert order == ['verify_runtime_versions','implementation_state','artifact_preflight']
    assert not calls and not touched and not options['output'].exists()


def test_output_safety(tmp_path, monkeypatch):
    repo, raw = tmp_path/'repo', tmp_path/'raw'
    repo.mkdir(); raw.mkdir()
    def git(repo,git_executable,*args): return SimpleNamespace(returncode=0,stdout=b'')
    monkeypatch.setattr(q,'_git',git)
    for out in (repo/'docs'/'review',raw/'out',repo/'outputs'):
        with pytest.raises(p.ContractError): q.fresh_ignored_output(repo,raw,out,tmp_path/'git.exe')
    out = repo/'outputs'/'fresh'
    q.fresh_ignored_output(repo,raw,out,tmp_path/'git.exe')
    with pytest.raises(p.ContractError,match='already exists'): q.fresh_ignored_output(repo,raw,out,tmp_path/'git.exe')
    with pytest.raises(p.ContractError,match='overlap'): q.fresh_ignored_output(repo,repo/'outputs'/'inner',repo/'outputs'/'inner'/'out',tmp_path/'git.exe')
    monkeypatch.setattr(q,'_git',lambda *args: SimpleNamespace(returncode=1,stdout=b''))
    with pytest.raises(p.ContractError): q.fresh_ignored_output(repo,raw,repo/'outputs'/'notignored',tmp_path/'git.exe')


def test_runtime_module_has_no_discovery_or_historical_dependency():
    text = Path(q.__file__).read_text()
    assert 'datasets/MSU' not in text and '.face' not in text
    assert 'msu_preprocessing' not in text and 'msu_protocol' not in text
    assert '.glob(' not in text and '.rglob(' not in text


def test_codec_metadata_extension_disagreement(metadata):
    cat = p.load_source_catalog_metadata(*metadata)
    for source in p.qualification_sources(cat,p.TRAIN_QUALIFICATION):
        assert p.source_codec(source) == ('h264' if source.capture_device=='android' else 'prores')
        with pytest.raises(p.ContractError): p.source_codec(replace(source,filepath=source.filepath+'.wrong'))


def test_four_source_snapshot_access_and_projection(metadata, monkeypatch):
    cat = p.load_source_catalog_metadata(*metadata)
    four = p.qualification_sources(cat, p.TRAIN_QUALIFICATION)[:4]
    put_sources(metadata[0], four)
    allowed = {metadata[0]/s.filepath for s in four}
    touched = guard_paths(monkeypatch, metadata[0], allowed)
    snap = p.codec_gate_snapshot(metadata[0], cat)
    assert set(touched) == allowed
    assert [x['source']['video_id'] for x in snap['sources']] == list(p.TRAIN_CODEC_GATE)
    assert snap['projection_sha256'] == cat.projection_sha256
    with pytest.raises(p.ContractError, match='coverage'):
        p._snapshot_sources(snap, cat, 'training_qualification')


@pytest.mark.parametrize('mutation', ['missing', 'extra', 'reordered', 'test-source', 'wrong-projection'])
def test_qualification_manifest_snapshot_mutations(metadata, environment, mutation):
    cat = p.load_source_catalog_metadata(*metadata)
    sources = p.qualification_sources(cat,p.TRAIN_QUALIFICATION)
    snap = {'projection_sha256':cat.projection_sha256, 'sources':[
        {'source':asdict(s),'sha256':p.digest(asdict(s))} for s in sources]}
    decodes = [replace(decoded(s.video_id,source_sha=e['sha256'],snapshot_sha=p.digest(snap)),codec=p.source_codec(s))
               for s,e in zip(sources,snap['sources'])]
    # Snapshot validation must fail before any frame/output access.
    if mutation == 'missing': snap['sources'].pop()
    if mutation == 'extra': snap['sources'].append(snap['sources'][0])
    if mutation == 'reordered': snap['sources'].reverse()
    if mutation == 'test-source': snap['sources'][0]['source'] = asdict(next(s for s in cat.sources if s.pad_partition=='test'))
    if mutation == 'wrong-projection': snap['projection_sha256'] = '0'*64
    provenance = {k:'4'*64 for k in ('policy_document_sha256','amendment_sha256','implementation_sha256')}
    with pytest.raises(p.ContractError,match='snapshot'):
        p.make_manifest(cat,snap,environment,decodes,[],provenance)


def test_qualification_manifest_requires_ordered_eight_and_240(orchestrator, environment):
    options, _, _ = orchestrator
    q.run_qualification(**options)
    cat = p.load_source_catalog_metadata(options['raw_root'],options['catalog'],options['protocol_lock'])
    root = options['output']/'run-1'
    snap, original = p.read_json(root/'snapshot.json'), p.read_json(root/'manifest.json')
    for key in ('decodes','frames'):
        for action in ('pop','extra','reverse'):
            doc = deepcopy(original)
            if action=='pop': doc[key].pop()
            if action=='extra': doc[key].append(doc[key][0])
            if action=='reverse': doc[key].reverse()
            with pytest.raises(p.ContractError): p.validate_manifest(doc,cat,snap,environment)


@pytest.mark.parametrize('kind', ['scope', 'pixels', 'dtype', 'dimensions'])
def test_selected_frame_integrity_rejects_corruption(selected_transport, kind):
    result, _, frames = r.decode_selected_frames(*selected_transport[0])
    frames = list(frames)
    if kind=='scope': frames.reverse()
    if kind=='pixels': frames[0].oriented_rgb[0,0,0] ^= 1
    if kind=='dtype': frames[0] = r.SelectedFrame(0,frames[0].oriented_rgb.astype(np.float32))
    if kind=='dimensions': frames[0] = r.SelectedFrame(0,frames[0].oriented_rgb.reshape(2,3,3))
    with pytest.raises(p.ContractError): r.validate_selected_frames(result,frames)


def test_artifact_preflight_explicit_paths_and_order(tmp_path, monkeypatch):
    events = []
    def verify(path, sha, size=None):
        events.append(Path(path).name)
    def executable(path, role):
        events.append(role)
        return SimpleNamespace(sha256='1'*64,version='synthetic')
    monkeypatch.setattr(p,'verify_file',verify)
    monkeypatch.setattr(r,'verify_approved_executable',executable)
    def worker(*args):
        events.append('worker')
        return {'opencv_build':'synthetic','runtime_fingerprints':tuple((n,'3'*64) for n,_ in p.RUNTIME)}
    monkeypatch.setattr(r,'infer_yunet',worker)
    monkeypatch.setattr(p,'validate_artifact_files',lambda *args:events.append('archive-membership'))
    paths = [tmp_path/name for name in ('archive.zip','ffmpeg.exe','ffprobe.exe','model.onnx')]
    q.artifact_preflight(*paths)
    assert events == ['archive.zip','ffmpeg','ffprobe','model.onnx','worker','archive-membership']
    for index in range(4):
        bad = list(paths); bad[index] = Path(bad[index].name)
        events.clear()
        with pytest.raises(p.ContractError,match='absolute'): q.artifact_preflight(*bad)
        assert not events


def test_artifact_failure_prevents_metadata_or_media(orchestrator, monkeypatch):
    options,calls,touched = orchestrator
    def fail(*args): raise p.ContractError('unapproved fixture artifact')
    monkeypatch.setattr(q,'artifact_preflight',fail)
    monkeypatch.setattr(p,'load_source_catalog_metadata',lambda *args:pytest.fail('metadata accessed too soon'))
    with pytest.raises(p.ContractError,match='unapproved'): q.run_qualification(**options)
    assert not calls and not touched


@pytest.mark.parametrize('field', ['source_sha256','snapshot_sha256','n','source_size','oriented_size',
                                  'rotations','frame_hashes'])
def test_repeat_comparison_all_decode_fields(field):
    doc = {'decodes':[asdict(decoded())], 'frames':[]}
    first = (None,{},doc,[],[])
    second = deepcopy(first)
    second[2]['decodes'][0][field] = 'different'
    mismatch = q.compare_runs(first,second)
    assert field in mismatch['field'] and mismatch['video_id'] == p.TRAIN_QUALIFICATION[0]


@pytest.mark.parametrize('field', ['ordinal','oriented_rgb_sha256','score','detector_box',
                                  'mapped_box','crop_bounds','output'])
def test_repeat_comparison_all_frame_fields(field):
    frame = dict(group_id=p.TRAIN_QUALIFICATION[0],ordinal=0,oriented_rgb_sha256='a'*64,score=.95,
        detector_box=[1,2,3,4],mapped_box=[1,2,3,4],crop_bounds=[0,1,5,6],
        output={'png_sha256':'b'*64,'pixel_sha256':'c'*64})
    doc = {'decodes':[], 'frames':[frame]}
    first = (None,{},doc,[],[])
    second = deepcopy(first)
    second[2]['frames'][0][field] = 'different'
    mismatch = q.compare_runs(first,second)
    assert field in mismatch['field'] and (mismatch['j'],mismatch['ordinal']) == (0,0)


def test_cli_requires_every_path(monkeypatch,tmp_path):
    names = ('raw-root','catalog','protocol-lock','archive','ffmpeg','ffprobe','model','output','expected-head','git-executable')
    captured = []
    monkeypatch.setattr(q,'run_qualification',lambda **kw:captured.append(kw))
    args = {name: ('a'*40 if name=='expected-head' else str(tmp_path/name)) for name in names}
    for missing in names:
        argv = [arg for name,value in args.items() if name!=missing for arg in ('--'+name,value)]
        with pytest.raises(SystemExit): q.main(argv)
    assert not captured
    q.main([arg for name,value in args.items() for arg in ('--'+name,value)])
    assert captured == [{name.replace('-','_'):value for name,value in args.items()}]


def test_committed_implementation_guard(tmp_path, monkeypatch):
    name = 'module.py'
    (tmp_path/name).write_bytes(b'original\r\n')
    monkeypatch.setattr(q,'IMPLEMENTATION_FILES',(name,))
    monkeypatch.setattr(q,'POLICY_FILES',())
    def git(repo,git_executable,*args):
        return SimpleNamespace(returncode=0,stdout=(b'a'*40+b'\n' if args[0]=='rev-parse' else b'original\n'))
    monkeypatch.setattr(q,'_git',git)
    assert q.implementation_state(tmp_path,'a'*40,tmp_path/'git.exe')['files'][name] == p.stream_sha256(tmp_path/name)
    with pytest.raises(p.ContractError,match='HEAD'): q.implementation_state(tmp_path,'b'*40,tmp_path/'git.exe')
    (tmp_path/name).write_bytes(b'modified\n')
    with pytest.raises(p.ContractError,match='uncommitted'): q.implementation_state(tmp_path,'a'*40,tmp_path/'git.exe')
