"""Analysis of already validated, locked sessions; never imported by the reviewer UI."""
from __future__ import annotations

import itertools
import math

from .msu_pilot import GATES, PROTOCOL, SIDES, require
from .serialization import digest, json_bytes


def _residual(eyes, target, offset):
    per_eye = []
    for side, q in zip(SIDES, (target[:2], target[2:])):
        eye = eyes[side]
        if not eye["measurable"]:
            per_eye.append(dict(side=side, signed=None, distance=None, lower=None,
                                upper=None, reason=eye["reason"]))
            continue
        signed = [eye["x"] - (q[0]+offset), eye["y"] - (q[1]+offset)]
        d = math.hypot(*signed)
        per_eye.append(dict(side=side, signed=signed, distance=d,
                            lower=max(0, d-eye["radius"]), upper=d+eye["radius"], reason=None))
    available = all(e["distance"] is not None for e in per_eye)
    values = {k: sum(e[k] for e in per_eye)/2 if available else None
              for k in ("distance", "lower", "upper")}
    iod = math.dist(target[:2], target[2:])
    require(iod > 0, "Invalid inter-eye distance")
    return dict(eyes=per_eye, mean=values["distance"], lower=values["lower"],
                upper=values["upper"], normalized_mean=values["distance"]/iod if available else None,
                available=available, reason=None if available else "missing_landmark")


def analyze(root):
    """Only public analysis entry: fresh preflight, sealed plan, replay and locks."""
    from . import msu_pilot as pilot
    verified = pilot.preflight(root)
    out = pilot.output_directory(root)
    plan = pilot.read_plan(out, verified)
    sessions = pilot.analysis_sessions(out, plan)
    return _calculate(plan, sessions)


def _calculate(plan, sessions):
    """Internal arithmetic; callers must use analyze(root).

    No pixels, decoder calls, PAD metrics, or inferential significance tests.
    """
    require(len(sessions) == 2 and all(s[-1]["kind"] == "complete" for s in sessions),
            "Two completed sessions required")
    verified = plan["verified"]
    marks, measurements = {}, []
    session_records = []
    # Descriptive only: analyze() has already validated both clocks via analysis_sessions().
    gap_hours = (sessions[1][0]["clock"]["monotonic_ns"]
                 - sessions[0][-1]["clock"]["monotonic_ns"]) / 3_600_000_000_000
    for s, events in enumerate(sessions, 1):
        mapping = {i["opaque_id"]: i["group"] for i in plan["randomization"][str(s)]["order"]}
        rows = [e for e in events if e["kind"] == "measurement"]
        require(len(rows) == 16 and len({e["data"]["opaque_id"] for e in rows}) == 16,
                "Incomplete unique measurements")
        for event in rows:
            data = event["data"]
            g = mapping[data["opaque_id"]]
            marks[s, g] = data["eyes"]
            for side in SIDES:
                measurements.append(dict(reviewer=plan["reviewer"], session=s, group=g,
                    image_side=side, opaque_id=data["opaque_id"], utc=event["utc"],
                    pixel_sha256=verified["assets"][g]["pixel_sha256"], **data["eyes"][side]))
        session_records.append(dict(session=s, start_utc=events[0]["utc"], end_utc=events[-1]["utc"],
            gap_hours=None if s == 1 else gap_hours,
            reviewer=plan["reviewer"], randomization=plan["randomization"][str(s)],
            grid=next(e["data"] for e in events if e["kind"] == "grid"),
            events_digest=digest(json_bytes(events)), completion="locked"))
    repeatability, concerns = [], set()
    for g in verified["assets"]:
        for side in SIDES:
            a, b = marks[1, g][side], marks[2, g][side]
            available = a["measurable"] and b["measurable"]
            displacement = math.dist((a["x"], a["y"]), (b["x"], b["y"])) if available else None
            overlap = displacement <= a["radius"]+b["radius"] if available else None
            if overlap is False:
                concerns.add(g)
            repeatability.append(dict(reviewer=plan["reviewer"], group=g, image_side=side,
                session_ids=[1, 2], center_displacement=displacement, disks_overlap=overlap,
                reason=None if available else "missing_landmark"))
    comparisons, outcomes, residuals, causal, changes = [], [], [], [], []
    cases = sorted({c["case_id"] for c in verified["associations"]})
    for case in cases:
        cs = [c for c in verified["associations"] if c["case_id"] == case]
        groups = sorted({c["group"] for c in cs})
        target = cs[0]["eye_coordinates"]
        require(all(c["eye_coordinates"] == target for c in cs), "Conflicting annotation coordinates")
        values, bounds = {}, {}
        for g in groups:
            for convention, offset in (("raw", 0), ("minus_one", -1)):
                for s in (1, 2):
                    v = _residual(marks[s, g], target, offset)
                    values[g, convention, s] = v
                    residuals.append(dict(reviewer=plan["reviewer"], case_id=case, group=g,
                                          convention=convention, session=s, **v))
                a, b = values[g, convention, 1], values[g, convention, 2]
                bounds[g, convention] = (min(a["lower"], b["lower"]), max(a["upper"], b["upper"])) \
                    if a["available"] and b["available"] else None
                changes.append(dict(case_id=case, group=g, convention=convention,
                    residual_change=b["mean"]-a["mean"] if bounds[g, convention] else None,
                    envelope=bounds[g, convention]))
        missing = any(v is None for v in bounds.values())
        advantages, flags = set(), set()
        ties = [{"group": g, "labels": [c["candidate_label"] for c in cs if c["group"] == g]}
                for g in groups if sum(c["group"] == g for c in cs) > 1]
        for tie in ties:
            causal.append(dict(case_id=case, groups=[tie["group"], tie["group"]],
                               labels=tie["labels"], status="identical_pixels", trace=None))
        unresolved = []
        for a, b in itertools.combinations(groups, 2):
            pair_flags, direction = [], None
            complete = all(bounds[g, k] is not None for g in (a, b) for k in ("raw", "minus_one"))
            if complete:
                for g, h in ((a, b), (b, a)):
                    if all(bounds[g, k][1]+1.0 < bounds[h, k][0] for k in ("raw", "minus_one")):
                        direction = g
                        advantages.add((g, h))
                diffs = {(k, s): values[a, k, s]["mean"]-values[b, k, s]["mean"]
                         for k in ("raw", "minus_one") for s in (1, 2)}
                if any(diffs[k, 1]*diffs[k, 2] < 0 for k in ("raw", "minus_one")):
                    pair_flags.append("session_sensitive")
                if any(diffs["raw", s]*diffs["minus_one", s] < 0 for s in (1, 2)):
                    pair_flags.append("coordinate_sensitive")
            else:
                pair_flags.append("missing_landmark")
            if a in concerns or b in concerns:
                pair_flags.append("repeatability_concern")
            if direction is None:
                unresolved.append([a, b])
                if complete:
                    pair_flags.append("operationally_unresolved")
            flags.update(pair_flags)
            comparisons.append(dict(reviewer=plan["reviewer"], case_id=case, groups=[a, b],
                bounds={g: {k: bounds[g, k] for k in ("raw", "minus_one")} for g in (a, b)},
                operational_margin_pixels=1.0, robust_direction=direction, flags=pair_flags))
            causal.append(dict(case_id=case, groups=[a, b], status="unresolved", trace=None,
                               reason="Different arrays; no independent source-frame/PTS trace supplied."))
        winners = [g for g in groups if all((g, h) in advantages for h in groups if h != g)]
        best = winners[0] if len(winners) == 1 and not missing else None
        outcomes.append(dict(case_id=case, reviewer=plan["reviewer"], source_video_id=cs[0]["source_video_id"],
            complete=not missing, outcome="inconclusive_missing_landmark" if missing else
            "robust_geometric_best_group" if best else "inconclusive",
            best_group=best, labels=[c["candidate_label"] for c in cs if c["group"] == best],
            exact_pixel_ties=ties, unresolved_pairs=unresolved, flags=sorted(flags),
            limitations="One previously exposed reviewer/subject; alignment is not historical input identity."))
    return dict(schema="msu-six-case-exploratory-alignment-v1", execution_status="analyzed",
        authorization_reference=plan["authorization"], protocol_path=PROTOCOL,
        protocol_sha256=verified["protocol_sha256"], baseline_head="1ba31f3b794f9c471e352f99a0bab189c2b4cbcd",
        input_fingerprints=verified["input_fingerprints"], software=verified["software"],
        asset_reverification=list(verified["assets"].values()),
        reviewers=[dict(name=plan["reviewer"], exposure=plan["exposure"], participation_status="actual")],
        sessions=session_records, landmark_measurements=measurements,
        case_associations=verified["associations"], residuals=residuals,
        pairwise_comparisons=comparisons, repeatability=repeatability,
        residual_repeatability=changes, case_outcomes=outcomes, causal_assessments=causal,
        deviations=[dict(session=s, **e) for s, events in enumerate(sessions, 1)
                    for e in events if e["kind"] == "note"], stopping_events=[],
        interpretation="Exploratory geometric comparisons only; no global winner or historical correspondence claim.",
        **GATES)
