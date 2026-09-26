"""Synthetic tooling tests only. Never read licensed candidate pixels or start a real session."""
import copy
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from identity_invariant_fas.data import msu_pilot as p
from identity_invariant_fas.data import msu_pilot_analysis as analysis

REAL_CLOCK_SAMPLE = p.clock_sample

ROOT = Path(__file__).resolve().parents[1]
START = datetime(2030, 1, 1, tzinfo=timezone.utc)
DISPLAY = dict(tk_scaling=1.0, dpi=96.0, screen=[1024, 768], os_scaling_disclosure="synthetic 100%")


def grid():
    return [dict(zoom=z, boundary=list(b), target=list(t),
                 click=[b[i]+z*(t[i]+0.5) for i in range(2)]) for z, b, t in p.GRID_TRIALS]


def eyes(x=10.0, missing=False):
    result = {}
    for side, dx in zip(p.SIDES, (0, 20)):
        px, py = x+dx, 10.0
        result[side] = dict(measurable=True, x=px, y=py, radius=0.5, reason="",
                           conversion=dict(click=[px+0.5, py+0.5], boundary=[0, 0], zoom=1,
                                           viewport=[100, 100], display=DISPLAY))
    if missing:
        result[p.SIDES[0]] = dict(measurable=False, x=None, y=None, radius=None,
                                  reason="synthetic ambiguous pupil", conversion=None)
    return result


@pytest.fixture
def frozen_metadata():
    text = (ROOT / p.PROTOCOL).read_text(encoding="utf-8")
    pack = json.loads((ROOT / "docs/audit/msu_policy_human_pack.json").read_bytes())
    return text, pack


@pytest.fixture
def plan(frozen_metadata, tmp_path, monkeypatch):
    assets, cs = p.inventory(*frozen_metadata)
    # Explicitly synthetic landmarks and dimensions. Never open the real paths.
    for g, asset in assets.items():
        asset.update(path=g+".png", shape=[64, 64, 3], file_sha256="synthetic")
    for c in cs:
        c["eye_coordinates"] = [10.0, 10.0, 30.0, 10.0]
    result = dict(reviewer="Synthetic tester", exposure="Synthetic fixture; no candidate viewing",
                authorization="synthetic unit test", verified=dict(assets=assets, associations=cs,
                protocol_sha256=p.PROTOCOL_SHA, input_fingerprints=[], software={}),
                randomization={str(s): dict(seed=f"synthetic-{s}", algorithm=p.ALGORITHM,
                    order=p.frozen_order(list(assets), f"synthetic-{s}", s)) for s in (1, 2)}, **p.GATES)
    p.write_new(tmp_path / "plan.json", result)
    p.seal_plan(tmp_path, result)
    (tmp_path / "clock").mkdir()
    anchor = dict(epoch="synthetic-clock", utc=START.isoformat(), monotonic_ns=0)
    p.write_new(tmp_path / "clock/anchor.json", dict(sample=anchor,
                plan_digest=p.digest(p.json_bytes(result)), token="synthetic", port=1))
    state = {"now": START}
    def sample(*_):
        now = state["now"]
        return dict(epoch=anchor["epoch"], utc=now.isoformat(),
                    monotonic_ns=round((now-START).total_seconds()*1e9))
    monkeypatch.setattr(p, "clock_sample", sample)
    original_append, original_start = p.Session.append, p.Session.start
    def append(self, kind, data, now=None, **kwargs):
        if now is not None:
            state["now"] = now
        return original_append(self, kind, data, now, **kwargs)
    def start(self, now=None):
        if now is not None:
            state["now"] = now
        return original_start(self, now)
    monkeypatch.setattr(p.Session, "append", append)
    monkeypatch.setattr(p.Session, "start", start)
    monkeypatch.setattr(p, "preflight", lambda _: result["verified"])
    monkeypatch.setattr(p, "output_directory", lambda _: tmp_path)
    return result



def complete(out, plan, number, start, missing_group=None, overrides=None):
    s = p.Session(out, plan, number)
    s.start(start)
    s.append("grid", dict(trials=grid(), display=DISPLAY), start+timedelta(seconds=1))
    tick = 2
    for i, item in enumerate(s.order):
        if i == 8:
            s.append("break", {"resumed": True}, start+timedelta(seconds=tick))
            tick += 1
        s.append("present", {"opaque_id": item["opaque_id"]}, start+timedelta(seconds=tick))
        value = (overrides or {}).get(item["group"], eyes(missing=item["group"] == missing_group))
        s.append("measurement", dict(opaque_id=item["opaque_id"], eyes=value),
                 start+timedelta(seconds=tick+1))
        tick += 2
    s.lock(start+timedelta(seconds=tick))
    return s


def test_protocol_exact_hash_and_16_group_coverage(frozen_metadata):
    assert p.digest((ROOT / p.PROTOCOL).read_bytes()) == p.PROTOCOL_SHA
    assets, cs = p.inventory(*frozen_metadata)
    assert len(assets) == len({a["pixel_sha256"] for a in assets.values()}) == 16
    assert len(cs) == 24 and len({c["case_id"] for c in cs}) == 6
    assert sum(c["group"] == "P3" for c in cs) == 3
    bad = copy.deepcopy(frozen_metadata[1])
    next(c for c in bad["candidates"] if c["case_id"] == cs[0]["case_id"])["pad_partition"] = "test"
    with pytest.raises(ValueError):
        p.inventory(frozen_metadata[0], bad)


def test_hash_preflight_failure_before_any_pixels(tmp_path, monkeypatch):
    path = tmp_path / p.PROTOCOL
    path.parent.mkdir(parents=True)
    path.write_text("tampered protocol")
    monkeypatch.setattr(p, "load_asset", lambda *a: pytest.fail("Must fail before pixels"))
    with pytest.raises(ValueError, match="fingerprint"):
        p.preflight(tmp_path)


def test_asset_hash_dimensions_and_dtype(tmp_path):
    cv2 = pytest.importorskip("cv2")
    image = np.zeros((4, 6, 3), np.uint8)
    raw = cv2.imencode(".png", image)[1].tobytes()
    (tmp_path / "synthetic.png").write_bytes(raw)
    asset = dict(path="synthetic.png", file_sha256=p.digest(raw),
                 pixel_sha256=p.digest(image.tobytes()), shape=[4, 6, 3])
    assert np.array_equal(p.load_asset(tmp_path, asset), image)
    for key, value in (("file_sha256", "bad"), ("pixel_sha256", "bad"), ("shape", [4, 7, 3])):
        with pytest.raises(ValueError):
            p.load_asset(tmp_path, {**asset, key: value})
    image16 = np.zeros((4, 6, 3), np.uint16)
    raw16 = cv2.imencode(".png", image16)[1].tobytes()
    (tmp_path / "synthetic.png").write_bytes(raw16)
    with pytest.raises(ValueError):
        p.load_asset(tmp_path, {**asset, "file_sha256": p.digest(raw16),
                               "pixel_sha256": p.digest(image16.tobytes())})


def test_frozen_randomization_and_masking(plan):
    groups = list(plan["verified"]["assets"])
    a = p.frozen_order(groups, "seed", 1)
    b = p.frozen_order(list(reversed(groups)), "seed", 1)
    assert [i["group"] for i in a] == [i["group"] for i in b]
    assert {i["opaque_id"] for i in a}.isdisjoint(i["opaque_id"] for i in b)
    assert a != p.frozen_order(groups, "seed", 2)
    assert len({x["opaque_id"] for x in a}) == 16
    for item in a:
        assert item["opaque_id"].startswith("image-")
        assert not any(v in item["opaque_id"] for v in ("native", "export", "client", ".mp4", ".mov"))
    with pytest.raises(ValueError):
        p.frozen_order(groups[:-1]+[groups[0]], "seed", 1)


@pytest.mark.parametrize("zoom", [1, 2, 4])
@pytest.mark.parametrize("boundary", [(0, 0), (-21, -17), (30, 40)])
def test_coordinate_conversion_every_synthetic_pixel(zoom, boundary):
    for x in range(64):
        for y in range(64):
            click = [boundary[0]+zoom*(x+0.5), boundary[1]+zoom*(y+0.5)]
            assert p.display_to_image(click, boundary, zoom) == [x, y]


def test_grid_must_cover_all_zooms_and_pan():
    p.validate_grid(grid(), DISPLAY)
    with pytest.raises(ValueError):
        p.validate_grid(grid()[:-1], DISPLAY)
    bad = grid()
    bad[0]["click"][0] += 2
    with pytest.raises(ValueError):
        p.validate_grid(bad, DISPLAY)


def test_unmeasurable_and_uncertainty_validation():
    for e in eyes(missing=True).values():
        p.validate_eye(e, [64, 64, 3])
    bad = eyes(missing=True)[p.SIDES[0]]
    for patch in ({"reason": ""}, {"x": 1}, {"radius": 0.5}):
        with pytest.raises(ValueError):
            p.validate_eye({**bad, **patch}, [64, 64, 3])
    good = eyes()[p.SIDES[0]]
    for patch in ({"radius": 0.49}, {"radius": float("nan")}, {"radius": 2}, {"x": 12}):
        with pytest.raises(ValueError):
            p.validate_eye({**good, **patch}, [64, 64, 3])


def test_grid_blocks_presentation_and_payload_is_opaque(tmp_path, plan, monkeypatch):
    s = p.Session(tmp_path, plan, 1)
    s.start()
    calls = []
    monkeypatch.setattr(p, "load_asset", lambda *a: calls.append(1) or np.zeros((64, 64, 3), np.uint8))
    bridge = p.Presenter(tmp_path, s)
    with pytest.raises(ValueError):
        bridge.current()
    assert calls == []
    bridge.grid(grid(), DISPLAY)
    payload = bridge.current()
    assert set(payload) == {"opaque_id", "pixels"} and calls == [1]
    assert not any(e["kind"] == "measurement" for e in s.read())


def test_locking_append_only_duplicate_and_early_analysis(tmp_path, plan):
    s = p.Session(tmp_path, plan, 1)
    s.start(START)
    with pytest.raises(ValueError):
        s.lock(START+timedelta(seconds=1))
    with pytest.raises(FileNotFoundError):
        p.analysis_sessions(tmp_path, plan)
    s.append("grid", dict(trials=grid(), display=DISPLAY), START+timedelta(seconds=1))
    item = s.order[0]
    s.append("present", {"opaque_id": item["opaque_id"]}, START+timedelta(seconds=2))
    s.append("measurement", dict(opaque_id=item["opaque_id"], eyes=eyes()), START+timedelta(seconds=3))
    with pytest.raises(ValueError):
        s.append("measurement", dict(opaque_id=item["opaque_id"], eyes=eyes()), START+timedelta(seconds=4))
    with pytest.raises(FileExistsError):
        s.start(START)


def test_session2_delay_and_locked_integrity(tmp_path, plan):
    first = complete(tmp_path, plan, 1, START)
    end = p.instant(first.locked()[-1]["utc"])
    with pytest.raises(ValueError, match="48 hours"):
        p.Session(tmp_path, plan, 2).start(end+timedelta(hours=48, microseconds=-1))
    assert not (tmp_path / "session-2").exists()
    with pytest.raises(FileNotFoundError):
        p.analysis_sessions(tmp_path, plan)
    second = complete(tmp_path, plan, 2, end+timedelta(hours=48))
    assert len(p.analysis_sessions(tmp_path, plan)) == 2
    with pytest.raises(ValueError, match="locked"):
        first.append("note", {"text": "late edit"})
    event_path = second.path / "event-0002.json"
    event_path.write_text(event_path.read_text().replace("present", "unknown"))
    with pytest.raises(ValueError, match="integrity"):
        second.locked()


def test_stop_timeout_scaling_change_and_break_required(tmp_path, plan):
    s = p.Session(tmp_path, plan, 1)
    s.start(START)
    s.append("grid", dict(trials=grid(), display=DISPLAY), START+timedelta(seconds=1))
    with pytest.raises(ValueError, match="60 minutes"):
        s.append("present", {"opaque_id": s.order[0]["opaque_id"]}, START+timedelta(minutes=61))
    s.append("present", {"opaque_id": s.order[0]["opaque_id"]}, START+timedelta(seconds=2))
    bad = eyes()
    bad[p.SIDES[0]]["conversion"]["display"] = {**DISPLAY, "dpi": 144}
    with pytest.raises(ValueError, match="scaling"):
        s.append("measurement", dict(opaque_id=s.order[0]["opaque_id"], eyes=bad), START+timedelta(seconds=3))
    s.append("stop", {"reason": "synthetic target exposure"}, START+timedelta(minutes=61))
    with pytest.raises(ValueError):
        s.append("present", {"opaque_id": s.order[0]["opaque_id"]}, START+timedelta(minutes=62))
    with pytest.raises(FileNotFoundError):
        s.locked()


def test_analysis_synthetic_ties_missingness_and_gates(tmp_path, plan):
    original = copy.deepcopy(plan)
    first = complete(tmp_path, plan, 1, START, missing_group="P3")
    complete(tmp_path, plan, 2, p.instant(first.locked()[-1]["utc"])+timedelta(hours=48))
    result = analysis.analyze(tmp_path)
    assert plan == original
    assert all(result[k] == v for k, v in p.GATES.items())
    assert len(result["landmark_measurements"]) == 64
    assert len(result["case_associations"]) == 24 and len(result["case_outcomes"]) == 6
    assert len(result["pairwise_comparisons"]) == 30  # 2*choose(3,2) + 4*choose(4,2)
    assert sum(c["status"] == "identical_pixels" for c in result["causal_assessments"]) == 2
    assert all(c["status"] in ("identical_pixels", "unresolved") for c in result["causal_assessments"])
    assert sum(o["outcome"] == "inconclusive_missing_landmark" for o in result["case_outcomes"]) == 3


def test_analysis_bounds_strict_margin_and_repeatability(tmp_path, plan):
    # Synthetic A1 is closer than A2/A3; all labels attached to A1 must survive.
    overrides = {g: eyes(18.0) for g in plan["verified"]["assets"]}
    overrides["A1"] = eyes(10.0)
    first = complete(tmp_path, plan, 1, START, overrides=overrides)
    complete(tmp_path, plan, 2, p.instant(first.locked()[-1]["utc"])+timedelta(hours=48), overrides=overrides)
    result = analysis.analyze(tmp_path)
    outcome = next(o for o in result["case_outcomes"] if o["case_id"].endswith("mp4:annotation:1"))
    assert outcome["best_group"] == "A1"
    assert set(outcome["labels"]) == {"native_i", "decframes_export_i"}
    assert analysis._residual(eyes(), [10, 10, 30, 10], 0)["upper"] == 0.5
    assert analysis._residual(eyes(missing=True), [10, 10, 30, 10], 0)["mean"] is None


def test_read_plan_rejects_seed_or_software_changes(tmp_path, plan):
    assert p.read_plan(tmp_path, plan["verified"]) == plan
    changed = copy.deepcopy(plan["verified"])
    changed["software"]["tool"] = "changed"
    with pytest.raises(ValueError):
        p.read_plan(tmp_path, changed)
    tampered = copy.deepcopy(plan)
    tampered["randomization"]["1"]["seed"] = "changed"
    (tmp_path / "plan.json").write_text(json.dumps(tampered))
    with pytest.raises(ValueError):
        p.read_plan(tmp_path, plan["verified"])


def test_ui_no_analysis_import_or_provenance_strings():
    text = (ROOT / "src/identity_invariant_fas/data/msu_pilot_ui.py").read_text()
    for forbidden in ("msu_pilot_analysis", "eye_coordinates", "candidate_label", "source_video_id",
                      "native_i", "export_ordinal", "human_review_responses"):
        assert forbidden not in text


def test_tk_synthetic_grid_only_smoke():
    """Hidden local Tk widget test; synthetic grid only, never a real session."""
    tk = pytest.importorskip("tkinter")
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("Tk display unavailable")
    root.withdraw()
    from identity_invariant_fas.data.msu_pilot_ui import ReviewerWindow
    class GridOnly:
        def stop(self, reason):
            self.reason = reason
        def current(self):
            pytest.fail("No candidate presentation permitted in grid smoke test")
    bridge = GridOnly()
    window = ReviewerWindow(root, bridge, "synthetic display")
    root.update_idletasks()
    window.click(SimpleNamespace(x=30, y=32))  # first synthetic target; not a candidate mark
    assert len(window.trials) == 1
    window.stop("synthetic test complete")
    root.destroy()


def test_analysis_command_cannot_score_one_session(tmp_path, plan, monkeypatch):
    complete(tmp_path, plan, 1, START)
    monkeypatch.setattr(p, "preflight", lambda _: plan["verified"])
    monkeypatch.setattr(p, "output_directory", lambda _: tmp_path)
    monkeypatch.setattr(analysis, "analyze", lambda *a: pytest.fail("Scoring must not start"))
    assert p.main(["--root", str(tmp_path), "analyze"]) == 1
    assert not (tmp_path / "analysis.json").exists()


def test_break_is_required_before_ninth_group(tmp_path, plan):
    s = p.Session(tmp_path, plan, 1)
    s.start(START)
    s.append("grid", dict(trials=grid(), display=DISPLAY), START+timedelta(seconds=1))
    for i, item in enumerate(s.order[:8]):
        s.append("present", {"opaque_id": item["opaque_id"]}, START+timedelta(seconds=2+i*2))
        s.append("measurement", dict(opaque_id=item["opaque_id"], eyes=eyes()),
                 START+timedelta(seconds=3+i*2))
    with pytest.raises(ValueError, match="break"):
        s.append("present", {"opaque_id": s.order[8]["opaque_id"]}, START+timedelta(seconds=20))


def test_exact_margin_is_not_a_win_and_origin_sensitivity(tmp_path, plan):
    # Synthetic 2px displacement gives lower=1.5; 0.5+1.0 == 1.5 must NOT pass.
    overrides = {g: eyes(12.0) for g in plan["verified"]["assets"]}
    overrides["A1"] = eyes(10.0)
    first = complete(tmp_path, plan, 1, START, overrides=overrides)
    complete(tmp_path, plan, 2, p.instant(first.locked()[-1]["utc"])+timedelta(hours=48), overrides=overrides)
    result = analysis.analyze(tmp_path)
    outcome = next(o for o in result["case_outcomes"] if o["case_id"].endswith("mp4:annotation:1"))
    assert outcome["best_group"] is None
    assert outcome["outcome"] == "inconclusive"


def test_session_reversal_is_inconclusive_and_not_averaged(tmp_path, plan):
    a = {g: eyes(18.0) for g in plan["verified"]["assets"]}
    a["A1"] = eyes(10.0)
    first = complete(tmp_path, plan, 1, START, overrides=a)
    b = copy.deepcopy(a)
    b["A1"], b["A2"] = eyes(18.0), eyes(10.0)
    complete(tmp_path, plan, 2, p.instant(first.locked()[-1]["utc"])+timedelta(hours=48), overrides=b)
    result = analysis.analyze(tmp_path)
    row = next(r for r in result["pairwise_comparisons"] if r["groups"] == ["A1", "A2"])
    assert row["robust_direction"] is None
    assert {"session_sensitive", "repeatability_concern"} <= set(row["flags"])


def test_tk_synthetic_full_measurement_flow(tmp_path, plan, monkeypatch):
    """One synthetic ledger/UI exercise in tmp_path; not the private real Session 1."""
    tk = pytest.importorskip("tkinter")
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("Tk display unavailable")
    root.withdraw()
    from identity_invariant_fas.data.msu_pilot_ui import ReviewerWindow
    monkeypatch.setattr(p, "load_asset", lambda *a: np.zeros((64, 64, 3), np.uint8))
    s = p.Session(tmp_path, plan, 1)
    bridge = p.Presenter(tmp_path, s)
    bridge.start()
    ui = ReviewerWindow(root, bridge, "synthetic 100%")
    root.update_idletasks()
    for z, b, target in p.GRID_TRIALS:
        ui.click(SimpleNamespace(x=b[0]+z*(target[0]+0.5), y=b[1]+z*(target[1]+0.5)))
    assert ui.payload is not None
    for i in range(16):
        for side, x in zip(p.SIDES, (10, 30)):
            ui.side.set(side)
            ui.click(SimpleNamespace(x=20+x+0.5, y=20+10.5))
        ui.submit()
        if i == 7:
            assert ui.payload is None
            before = s.read()
            ui.click(SimpleNamespace(x=30, y=30))
            assert not ui.ended and s.read() == before and ui.payload is None
            ui.resume_break()
    assert ui.ended and ui.count == 16
    assert s.locked()[-1]["kind"] == "complete"
    root.destroy()

@pytest.mark.parametrize("dx", [-1, 0, 1])
def test_eye_relationship(dx):
    pair = eyes()
    right = pair[p.SIDES[1]]
    right["x"] = 10 + dx
    right["conversion"]["click"][0] = right["x"] + .5
    if dx <= 0:
        with pytest.raises(ValueError, match="image-left"):
            p.validate_eyes(pair, [64, 64, 3])
    else:
        p.validate_eyes(pair, [64, 64, 3])
    p.validate_eyes(eyes(missing=True), [64, 64, 3])


@pytest.mark.parametrize("mutation", ["seed", "coherent_order", "mapping", "truncated", "malformed"])
def test_preparation_seal_tampering(tmp_path, plan, mutation):
    changed = copy.deepcopy(plan)
    r = changed["randomization"]["1"]
    if mutation in ("seed", "coherent_order"):
        r["seed"] = "changed"
        if mutation == "coherent_order":
            r["order"] = p.frozen_order(list(plan["verified"]["assets"]), r["seed"], 1)
    if mutation == "mapping":
        r["order"][0]["opaque_id"] = "image-" + "a"*64
    if mutation in ("truncated", "malformed"):
        (tmp_path / "plan-seal.json").write_text("{" if mutation == "truncated" else "{}")
    else:
        (tmp_path / "plan.json").write_text(json.dumps(changed))
    with pytest.raises((ValueError, KeyError)):
        p.read_plan(tmp_path, plan["verified"])
    with pytest.raises((ValueError, KeyError)):
        p.Session(tmp_path, changed, 1).start(START)
    assert not (tmp_path / "session-1").exists()


def test_prepare_exclusive_and_seeds(tmp_path, plan):
    with pytest.raises(ValueError, match="distinct"):
        p.prepare(tmp_path, plan["verified"], "r", "e", "a", ("same", "same"))
    with pytest.raises(FileExistsError):
        p.prepare(tmp_path, plan["verified"], "r", "e", "a", ("one", "two"))
    changed = copy.deepcopy(plan)
    changed["randomization"]["2"]["seed"] = changed["randomization"]["1"]["seed"]
    changed["randomization"]["2"]["order"] = p.frozen_order(
        list(plan["verified"]["assets"]), changed["randomization"]["1"]["seed"], 2)
    # Even a newly anchored invalid fixture must fail semantic validation.
    (tmp_path / "plan.json").write_text(json.dumps(changed))
    (tmp_path / "plan-seal.json").unlink()
    p.seal_plan(tmp_path, changed)
    with pytest.raises(ValueError, match="seeds"):
        p.read_plan(tmp_path, plan["verified"])


def clock_point(seconds, wall=None, epoch="test"):
    return dict(epoch=epoch, utc=(START+timedelta(seconds=seconds if wall is None else wall)).isoformat(),
                monotonic_ns=round(seconds*1e9))


@pytest.mark.parametrize("seconds,wall,accepted", [(172800,172800,True), (172799,172799,False),
    (10,172810,False), (172800,10,False), (20,-1,False)])
def test_actual_elapsed_gap(seconds, wall, accepted):
    a, b = clock_point(0), clock_point(seconds, wall)
    first, second = dict(clock=a, utc=a["utc"]), dict(clock=b, utc=b["utc"])
    if accepted:
        p.require_gap(first, second)
    else:
        with pytest.raises(ValueError):
            p.require_gap(first, second)


@pytest.mark.parametrize("timestamp", ["bad", "2030-01-01T00:00:00", "2030-01-01T03:30:00+03:30", None])
def test_malformed_non_utc_clock(timestamp):
    with pytest.raises((ValueError, TypeError)):
        p.validate_clock(clock_point(0), {**clock_point(1), "utc": timestamp})


def test_clock_continuity_restart_suspend_and_unavailable(tmp_path, plan):
    for sample in (clock_point(1, epoch="restart"), clock_point(-1)):
        with pytest.raises(ValueError):
            p.validate_clock(clock_point(0), sample)
    with pytest.raises(ValueError, match="suspended"):
        p.validate_clock(clock_point(0), clock_point(11), clock_point(0), continuous=True)
    # Exercise actual socket client, bypassing the synthetic fixture replacement.
    with pytest.raises(p.ClockIntegrityError, match="unavailable"):
        REAL_CLOCK_SAMPLE(tmp_path, plan)  # synthetic anchor has no listening witness
    (tmp_path / "clock/failed.json").write_text("{}")
    with pytest.raises(ValueError, match="failed"):
        p.Session(tmp_path, plan, 1).start(START)


@pytest.mark.parametrize("damage", ["unsealed", "stopped", "zero_gap", "short_gap", "clock_gap", "lock", "incomplete",
    "deleted", "truncated", "reordered", "malformed", "binding"])
def test_direct_analysis_api_refuses_invalid_records(tmp_path, plan, monkeypatch, damage):
    first = complete(tmp_path, plan, 1, START)
    second = complete(tmp_path, plan, 2, p.instant(first.locked()[-1]["utc"])+timedelta(hours=48))
    if damage == "unsealed":
        (tmp_path / "plan-seal.json").unlink()
    elif damage == "lock":
        (second.path / "lock.json").write_text("{}")
    elif damage == "incomplete":
        (second.path / "lock.json").unlink()
    elif damage == "binding":
        changed = copy.deepcopy(plan)
        changed["reviewer"] = "another reviewer"
        (tmp_path / "plan.json").write_text(json.dumps(changed))
    elif damage in ("deleted", "truncated", "malformed", "reordered"):
        path = second.path / "event-0002.json"
        if damage == "deleted":
            path.unlink()
        else:
            path.write_text("{" if damage == "truncated" else "{}" if damage == "malformed"
                            else (second.path / "event-0003.json").read_text())
    else:
        # Coherently rehash tampered records/lock to test semantics, not just digests.
        events = second.read()
        if damage == "stopped":
            events[-1]["kind"], events[-1]["data"] = "stop", {"reason": "target exposure"}
        else:
            for e in events:
                if damage == "clock_gap":
                    e["clock"]["monotonic_ns"] -= 10*10**9
                else:
                    shift = timedelta(microseconds=1) if damage == "short_gap" else timedelta(hours=48)
                    e["utc"] = (p.instant(e["utc"])-shift).isoformat()
                    e["clock"]["utc"] = e["utc"]
                    e["clock"]["monotonic_ns"] -= 1000 if damage == "short_gap" else 48*3600*10**9
        previous = second.plan_digest
        for e in events:
            e.pop("digest")
            e["previous"] = previous
            e["digest"] = previous = p.digest(p.json_bytes(e))
            (second.path / f"event-{e['sequence']:04d}.json").write_text(json.dumps(e))
        (second.path / "lock.json").write_text(json.dumps(dict(plan_digest=second.plan_digest,
            event_count=len(events), events_digest=p.digest(p.json_bytes(events)),
            start_utc=events[0]["utc"], end_utc=events[-1]["utc"], **p.GATES)))
    monkeypatch.setattr(analysis, "_calculate", lambda *a: pytest.fail("Invalid input reached calculation"))
    with pytest.raises((ValueError, OSError, KeyError, TypeError)):
        analysis.analyze(tmp_path)
    with pytest.raises(TypeError):
        analysis.analyze(plan, [])


def test_invalidating_disclosure_stops(tmp_path, plan):
    session = p.Session(tmp_path, plan, 1)
    session.start(START)
    p.Presenter(tmp_path, session).note("synthetic target exposure", invalidating=True)
    assert session.read()[-1]["kind"] == "stop"
    with pytest.raises(ValueError):
        session.lock(START)


def test_live_synthetic_clock_witness(tmp_path, plan, monkeypatch):
    import threading
    import time
    (tmp_path / "clock/anchor.json").unlink()
    (tmp_path / "clock").rmdir()
    offset = [0]
    origin = time.monotonic()
    monkeypatch.setattr(p, "utcnow", lambda: START+timedelta(seconds=time.monotonic()-origin+offset[0]))
    errors = []
    ready = threading.Event()
    write = p.write_new
    def publish(path, value):
        write(path, value)
        if Path(path).name == "anchor.json":
            ready.set()
    monkeypatch.setattr(p, "write_new", publish)
    def run():
        try:
            p.clock_guard(tmp_path, plan)
        except (ValueError, OSError) as error:
            errors.append(error)
    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    try:
        assert ready.wait(30), errors
        sample = REAL_CLOCK_SAMPLE(tmp_path, plan)
        assert sample["epoch"] and sample["monotonic_ns"] > 0
        with pytest.raises(FileExistsError):
            p.clock_guard(tmp_path, plan)
    finally:
        offset[0] = 100  # a forward wall correction must permanently fail the witness
        thread.join(3)
    assert not thread.is_alive() and errors
    assert (tmp_path / "clock/failed.json").exists()
    with pytest.raises(ValueError):
        REAL_CLOCK_SAMPLE(tmp_path, plan)


@pytest.mark.parametrize("ok", [False, True])
def test_windows_dpi_setup_fails_closed(ok):
    from identity_invariant_fas.data.msu_pilot_ui import enable_windows_dpi
    requested = []
    api = SimpleNamespace(SetProcessDpiAwarenessContext=lambda value: requested.append(value.value) or ok)
    if ok:
        enable_windows_dpi(api)
    else:
        with pytest.raises(ValueError, match="DPI"):
            enable_windows_dpi(api)
    assert len(requested) == 1


def test_ui_error_and_break_masking(monkeypatch):
    from identity_invariant_fas.data import msu_pilot_ui as ui
    errors = []
    monkeypatch.setattr(ui.messagebox, "showinfo", lambda *args: errors.append(args))
    window = object.__new__(ui.ReviewerWindow)
    window.ended, window.trials, window.payload = False, grid(), None
    window.guard_display = lambda: pytest.fail("Break click must be ignored before handling")
    window.click(SimpleNamespace(x=10, y=10))
    assert not window.ended and not errors
    window.payload = {"opaque_id": "image-" + "a"*64}
    window.guard_display = lambda: None
    window.image = SimpleNamespace(height=64, width=64)
    window.fields = {s: (SimpleNamespace(get=lambda: False), SimpleNamespace(get=lambda: ".5"),
                         SimpleNamespace(get=lambda: "")) for s in p.SIDES}
    pair = eyes()
    window.centers = {s: {k: pair[other][k] for k in ("x", "y", "conversion")}
                      for s, other in zip(p.SIDES, reversed(p.SIDES))}
    window.presenter = SimpleNamespace(submit=lambda *a: pytest.fail("Reversed marks cannot be appended"))
    window.submit()
    assert errors and not window.ended
    rendered = str(errors)
    for forbidden in ("native", "export", "client002", ".face", "A1", "P3", "residual"):
        assert forbidden not in rendered


def test_clock_anchor_modification_detected(tmp_path, plan):
    session = complete(tmp_path, plan, 1, START)
    path = tmp_path / "clock/anchor.json"
    anchor = json.loads(path.read_bytes())
    anchor["port"] = 2
    path.write_text(json.dumps(anchor))
    with pytest.raises(ValueError, match="anchor"):
        session.locked()


def test_prepare_creates_seal_and_frozen_secret_mappings(tmp_path, frozen_metadata, monkeypatch):
    assets, associations = p.inventory(*frozen_metadata)
    verified = dict(assets=assets, associations=associations)
    out = tmp_path / "fresh-preparation"
    monkeypatch.setattr(p, "output_directory", lambda _: out)
    plan = p.prepare(tmp_path, verified, "Synthetic reviewer", "Synthetic exposure",
                     "Synthetic authorization", ("predictable-1", "predictable-2"))
    assert p.read_plan(out, verified) == plan
    assert p.instant(plan["prepared_utc"]).tzinfo is not None
    a = plan["randomization"]["1"]["order"]
    guesses = p.frozen_order(list(assets), "predictable-1", 1)
    assert [i["group"] for i in a] == [i["group"] for i in guesses]
    assert {i["opaque_id"] for i in a}.isdisjoint(i["opaque_id"] for i in guesses)
    assert {i["opaque_id"] for i in a}.isdisjoint(
        i["opaque_id"] for i in plan["randomization"]["2"]["order"])
    with pytest.raises(FileExistsError):
        p.prepare(tmp_path, verified, "r", "e", "a", ("one", "two"))


@pytest.mark.parametrize("wall_gap_hours,clock_offset_seconds", [(48, 0), (49.5, 0), (49.5, 1)])
def test_summary_gap_hours_from_validated_elapsed_evidence(tmp_path, plan, monkeypatch,
                                                         wall_gap_hours, clock_offset_seconds):
    first = complete(tmp_path, plan, 1, START)
    end = p.instant(first.locked()[-1]["utc"])
    sample = p.clock_sample
    def offset_sample(*args):
        clock = sample(*args)
        clock["monotonic_ns"] -= clock_offset_seconds*10**9
        return clock
    monkeypatch.setattr(p, "clock_sample", offset_sample)
    complete(tmp_path, plan, 2, end+timedelta(hours=wall_gap_hours))
    sessions = p.analysis_sessions(tmp_path, plan)
    expected = (sessions[1][0]["clock"]["monotonic_ns"]
                - sessions[0][-1]["clock"]["monotonic_ns"]) / 3_600_000_000_000
    result = analysis.analyze(tmp_path)
    assert result["sessions"][0]["gap_hours"] is None  # No preceding session.
    assert result["sessions"][1]["gap_hours"] == expected
    assert expected == pytest.approx(wall_gap_hours-clock_offset_seconds/3600)
    assert (p.instant(result["sessions"][1]["start_utc"])
            - p.instant(result["sessions"][0]["end_utc"])).total_seconds()/3600 == wall_gap_hours
    assert all(result[k] == v for k, v in p.GATES.items())
