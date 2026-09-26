"""Frozen pilot preflight and append-only session ledger. No measurement scoring here."""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import re
import secrets
import socket
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .serialization import digest, json_bytes

BASELINE = "0f9dd0f9e122909475027491fbe9b9173828e7de"
PROTOCOL = "docs/MSU_CORRESPONDENCE_PILOT_PROTOCOL.md"
PROTOCOL_SHA = "dfbd88ad785bf32b947e8f2c54aafc3a593d5474112dae6b4eb864cf439b39a4"
MANIFEST = "data_processed/recovery/msu-native-six-20260924-v1/manifest.json"
MANIFEST_SHA = "b93711932627c4d0407b58836e81e1ba5f890a81060e6e96434f4f969d130028"
OUTPUT = "data_processed/recovery/msu-correspondence-pilot-v1"
GATES = dict(policy_state="P4", frozen_policy=None,
             fidelity_status="manual_review_pending", experiment_ready=False)
ALGORITHM = "sha256-sort-v1; UTF-8 seed/session/group; hex lexical order"
SIDES = ("image-left", "image-right")
COVERAGE = dict(case_count=6, candidate_association_count=24,
                distinct_oriented_pixel_groups=16, verified_clean_group_count=16)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def utcnow():
    return datetime.now(timezone.utc)


def instant(value):
    result = datetime.fromisoformat(value)
    require(result.tzinfo is not None and result.utcoffset() == timedelta(0), "UTC required")
    return result


def git(root, *args):
    return subprocess.check_output(["git", *args], cwd=root)


def checked_bytes(path, expected):
    raw = Path(path).read_bytes()
    require(digest(raw) == expected, "Input fingerprint mismatch")
    return raw


def inventory(text, pack):
    """Parse only the hash-pinned protocol; never infer a replacement asset."""
    dirs = dict(re.findall(r"^([APLR])-dir = (.+)$", text, re.M))
    rows = re.findall(r"^\| ([APL][1-6]) \| ([APLR])-dir / `([^`]+)` \| `([0-9a-f]{64})` \|$",
                      text, re.M)
    require(len(rows) == 16 and len({r[0] for r in rows}) == 16
            and len({r[3] for r in rows}) == 16, "Expected sixteen unique pixel groups")
    assets = {g: dict(group=g, path=dirs[d] + f, pixel_sha256=h,
                     shape=[480, 640 if g.startswith("L") else 720, 3])
              for g, d, f, h in rows}
    cases = dict(re.findall(r"^\| ([APL]\d+-case) \| `(MSU-MFSD:[^`]+)` \|$", text, re.M))
    matrix = re.findall(r"^\| ([APL]\d+-case) \| (\d+) → ([APL]\d) \| (\d+) → ([APL]\d)"
                        r" \| (\d+) → ([APL]\d) \| (\d+) → ([APL]\d) \|$", text, re.M)
    require(len(cases) == len(matrix) == 6, "Expected six cases")
    labels = ("native_i", "native_i_minus_1", "native_i_plus_1", "decframes_export_i")
    associations = []
    for row in matrix:
        for j, label in enumerate(labels):
            ordinal, group = row[1 + 2*j:3 + 2*j]
            matches = [c for c in pack["candidates"]
                       if c["case_id"] == cases[row[0]] and c["candidate_label"] == label]
            require(len(matches) == 1, "Ambiguous candidate association")
            c = matches[0]
            require(c["available"] and c["pad_partition"] == "train"
                    and c["client_id"] == "002" and c["rotation_metadata"] == 0
                    and c["candidate_frame_index"] == int(ordinal)
                    and c["oriented_pixels_sha256"] == assets[group]["pixel_sha256"],
                    "Candidate differs from frozen protocol")
            eyes = c["eye_coordinates"]
            require(len(eyes) == 4 and all(math.isfinite(x) for x in eyes)
                    and eyes[0] < eyes[2], "Invalid frozen eye ordering")
            associations.append({**c, "group": group, "image_side_permutation": [1, 2]})
    require(len(associations) == 24 and {c["group"] for c in associations} == set(assets),
            "Incomplete group coverage")
    return assets, associations


def load_asset(root, asset):
    import cv2
    import numpy as np
    raw = checked_bytes(Path(root) / asset["path"], asset["file_sha256"])
    image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_UNCHANGED)
    require(image is not None and image.dtype == np.uint8
            and list(image.shape) == asset["shape"]
            and digest(image.tobytes()) == asset["pixel_sha256"], "Asset pixels mismatch")
    return image


def preflight(root):
    """Read-only: decode retained BMP/PNG arrays for hashing, never video or display."""
    root = Path(root).resolve()
    text = checked_bytes(root / PROTOCOL, PROTOCOL_SHA).decode("utf-8")
    manifest = json.loads(checked_bytes(root / MANIFEST, MANIFEST_SHA))
    require(manifest["asset_only_status"] == "GO"
            and manifest["frozen_case_coverage"] == COVERAGE, "Recovery coverage is not GO")
    fingerprints = [dict(path=PROTOCOL, sha256=PROTOCOL_SHA, hash_kind="exact_bytes"),
                    dict(path=MANIFEST, sha256=MANIFEST_SHA, hash_kind="exact_bytes")]
    paths = git(root, "ls-tree", "-r", "--name-only", BASELINE, "docs/audit",
                "docs/MSU_PITTPATT_PROVENANCE.md", "docs/MSU_POLICY_GATE.md",
                "docs/MSU_HUMAN_REVIEW.md").decode().splitlines()
    require(len(paths) == 29, "Historical evidence set changed")
    for rel in paths:
        blob = git(root, "show", BASELINE + ":" + rel)
        raw = (root / rel).read_bytes()
        require(raw.replace(b"\r\n", b"\n") == blob.replace(b"\r\n", b"\n"),
                "Historical evidence changed")
        fingerprints.append(dict(path=rel, sha256=digest(raw), hash_kind="exact_bytes",
                                 committed_blob_sha256=digest(blob)))
    for rel, hashes in manifest["evidence"].items():
        checked_bytes(root / rel, hashes["working_file_sha256"])
        require(digest(git(root, "show", BASELINE + ":" + rel)) == hashes["committed_blob_sha256"],
                "Historical blob mismatch")
    pack = json.loads((root / "docs/audit/msu_policy_human_pack.json").read_bytes())
    assets, associations = inventory(text, pack)
    for rel in sorted({c["annotation_filepath"] for c in associations}):
        cs = [c for c in associations if c["annotation_filepath"] == rel]
        require(len({c["annotation_sha256"] for c in cs}) == 1, "Annotation fingerprint conflict")
        raw = checked_bytes(root / "datasets/MSU" / rel, cs[0]["annotation_sha256"])
        lines = raw.decode("utf-8").splitlines()
        for c in cs:
            require(lines[c["annotation_line_number"]-1].strip() == c["raw_annotation_line"].strip(),
                    "Original annotation row differs")
        fingerprints.append(dict(path="datasets/MSU/" + rel, sha256=digest(raw), hash_kind="exact_bytes"))
    entries = manifest["frames"] + manifest["existing_representatives"]["records"]
    require(len(entries) == 16 and len({e["group"] for e in entries}) == 16,
            "Recovery inventory differs")
    for group, asset in assets.items():
        entry = next(e for e in entries if e["group"] == group)
        require(entry["status"] == "verified"
                and entry["expected_pixels_sha256"] == asset["pixel_sha256"]
                and entry["expected_shape_hwc"] == asset["shape"], "Recovery association mismatch")
        asset["file_sha256"] = entry.get("file_sha256", entry.get("png_file_sha256"))
        pixels = load_asset(root, asset)
        asset.update(actual_file_sha256=asset["file_sha256"],
                     actual_pixel_sha256=digest(pixels.tobytes()), width=pixels.shape[1],
                     height=pixels.shape[0], channels=3, dtype="uint8", status="verified")
    for name in ("msu_policy_decision", "msu_human_review_validation"):
        state = json.loads((root / f"docs/audit/{name}.json").read_bytes())
        require(all(state[k] == GATES[k] for k in ("policy_state", "frozen_policy", "experiment_ready")),
                "Scientific gate changed")
    require(pack["fidelity_status"] == GATES["fidelity_status"]
            and pack["experiment_ready"] is False, "Fidelity gate changed")
    import cv2
    import numpy as np
    import PIL
    software = dict(python=sys.version, executable=sys.executable, platform=platform.platform(),
                    numpy=np.__version__, opencv=cv2.__version__, pillow=PIL.__version__)
    software["tool_sources"] = {
        p.name: digest(p.read_bytes()) for p in Path(__file__).parent.glob("msu_pilot*.py")}
    return dict(protocol_sha256=PROTOCOL_SHA, input_fingerprints=fingerprints,
                assets=assets, associations=associations, software=software, gates=dict(GATES))


def frozen_order(groups, seed, session, mapping=None):
    require(isinstance(seed, str) and bool(seed.strip()) and session in (1, 2), "Invalid seed/session")
    require(len(groups) == 16 and len(set(groups)) == 16, "Expected sixteen distinct groups")
    def token(kind, group):
        return digest(json_bytes([ALGORITHM, kind, seed, session, group]))
    ordered = sorted(groups, key=lambda g: (token("order", g), g))
    # Reproducible order and secret labels have deliberately separate entropy.
    mapping = mapping if mapping is not None else {g: "image-" + secrets.token_hex(32) for g in groups}
    require(set(mapping) == set(groups) and all(re.fullmatch(r"image-[0-9a-f]{64}", v)
            for v in mapping.values()), "Invalid private mapping")
    result = [dict(opaque_id=mapping[g], group=g) for g in ordered]
    require(len({i["opaque_id"] for i in result}) == 16, "Opaque ID collision")
    return result


def write_new(path, value):
    """Exclusive creation plus fsync: never overwrite a plan, event, lock, or result."""
    with Path(path).open("xb") as stream:
        stream.write(json_bytes(value))
        stream.flush()
        os.fsync(stream.fileno())


def output_directory(root):
    root = Path(root).resolve()
    out = root / OUTPUT
    require(out.resolve().is_relative_to(root / "data_processed"), "Output escapes private directory")
    require(subprocess.run(["git", "check-ignore", "-q", OUTPUT + "/plan.json"], cwd=root).returncode == 0,
            "Execution directory must be Git-ignored")
    return out


def prepare(root, verified, reviewer, exposure, authorization, seeds):
    require(all(isinstance(v, str) and v.strip() for v in (reviewer, exposure, authorization)),
            "Actual reviewer, exposure disclosure and separate authorization required")
    require(len(seeds) == 2 and seeds[0] != seeds[1], "Two distinct frozen seeds required")
    orders = {str(s): dict(seed=seeds[s-1], algorithm=ALGORITHM,
                          order=frozen_order(list(verified["assets"]), seeds[s-1], s)) for s in (1, 2)}
    plan = dict(schema="msu-pilot-private-plan-v1", prepared_utc=utcnow().isoformat(),
                reviewer=reviewer, exposure=exposure, authorization=authorization,
                randomization=orders, verified=verified, **GATES)
    out = output_directory(root)
    out.mkdir(parents=True, exist_ok=False)
    write_new(out / "plan.json", plan)
    seal_plan(out, plan)
    return plan


def seal_plan(out, plan):
    """Separate exclusive anchor; a partial write fails closed, never repaired automatically."""
    write_new(Path(out) / "plan-seal.json", dict(schema="msu-plan-seal-v1",
              plan_sha256=digest(json_bytes(plan))))


def read_plan(out, verified):
    plan = json.loads((Path(out) / "plan.json").read_bytes())
    seal = json.loads((Path(out) / "plan-seal.json").read_bytes())
    require(seal == dict(schema="msu-plan-seal-v1", plan_sha256=digest(json_bytes(plan))),
            "Preparation seal mismatch")
    require(plan["verified"] == verified, "Plan preflight/software changed; do not start")
    require(all(plan.get(k) == v for k, v in GATES.items()), "Plan gate changed")
    for s in (1, 2):
        r = plan["randomization"][str(s)]
        mapping = {i["group"]: i["opaque_id"] for i in r["order"]}
        require(r["algorithm"] == ALGORITHM
                and r["order"] == frozen_order(list(verified["assets"]), r["seed"], s, mapping),
                "Frozen randomization changed")
    require(plan["randomization"]["1"]["seed"] != plan["randomization"]["2"]["seed"],
            "Session seeds must differ")
    return plan


# These are conservative clock-integrity tolerances, not scientific thresholds.
CLOCK_DRIFT_SECONDS = 2.0
CLOCK_POLL_LIMIT_SECONDS = 10.0


class ClockIntegrityError(ValueError):
    """Only fixed, reviewer-safe clock diagnostics; never paths or private labels."""


def clock_require(condition, message):
    if not condition:
        raise ClockIntegrityError(message)


def validate_clock(anchor, sample, previous=None, continuous=False):
    clock_require(set(sample) == {"epoch", "utc", "monotonic_ns"}, "Malformed clock evidence")
    clock_require(sample["epoch"] == anchor["epoch"] and type(sample["monotonic_ns"]) is int,
            "Clock continuity unavailable")
    wall = instant(sample["utc"])
    for baseline in (anchor, previous or anchor):
        elapsed = (sample["monotonic_ns"] - baseline["monotonic_ns"]) / 1e9
        wall_elapsed = (wall - instant(baseline["utc"])).total_seconds()
        clock_require(elapsed >= 0 and wall_elapsed >= 0
                and abs(wall_elapsed - elapsed) <= CLOCK_DRIFT_SECONDS,
                "Clock discontinuity; continuation refused")
    if continuous and previous:
        clock_require(elapsed <= CLOCK_POLL_LIMIT_SECONDS, "Clock witness interrupted/suspended")


def clock_guard(out, plan):
    """One uninterrupted coordinator process. Restart is intentionally unsupported."""
    out = Path(out)
    require(read_plan(out, plan["verified"]) == plan, "Plan binding changed")
    directory = out / "clock"
    directory.mkdir(exist_ok=False)
    epoch, token = secrets.token_hex(32), secrets.token_hex(32)
    def sample():
        return dict(epoch=epoch, utc=utcnow().isoformat(), monotonic_ns=time.monotonic_ns())
    with socket.socket() as server:
        server.bind(("127.0.0.1", 0))
        server.listen(4)
        server.settimeout(0.5)
        anchor = previous = sample()
        write_new(directory / "anchor.json", dict(plan_digest=digest(json_bytes(plan)),
                  sample=anchor, port=server.getsockname()[1], token=token))
        try:
            while True:
                current = sample()
                validate_clock(anchor, current, previous, continuous=True)
                previous = current
                try:
                    connection, _ = server.accept()
                except socket.timeout:
                    continue
                with connection:
                    connection.settimeout(1)
                    request = connection.makefile("rb").readline(512)
                    require(request == (token + "\n").encode(), "Invalid clock request")
                    current = sample()
                    validate_clock(anchor, current, previous, continuous=True)
                    previous = current
                    # Wire framing is one JSON line; ledger serialization is pretty-printed.
                    connection.sendall(json.dumps(current, separators=(",", ":")).encode() + b"\n")
        except BaseException:
            write_new(directory / "failed.json", {"reason": "Clock witness ended or continuity failed"})
            raise


def clock_anchor(out, plan):
    directory = Path(out) / "clock"
    clock_require(not (directory / "failed.json").exists(),
                  "Clock witness failed or lost continuity; restart is unsupported")
    anchor = json.loads((directory / "anchor.json").read_bytes())
    require(anchor["plan_digest"] == digest(json_bytes(plan)), "Clock plan binding changed")
    validate_clock(anchor["sample"], anchor["sample"])
    return anchor


def clock_sample(out, plan):
    anchor = clock_anchor(out, plan)
    # A fresh response requires the original, still-running process. No stale-file fallback.
    try:
        with socket.create_connection(("127.0.0.1", anchor["port"]), timeout=2) as connection:
            connection.sendall((anchor["token"] + "\n").encode())
            sample = json.loads(connection.makefile("rb").readline(2048))
    except (OSError, ValueError):
        raise ClockIntegrityError("Clock witness unavailable; continuity cannot be established") from None
    validate_clock(anchor["sample"], sample)
    return sample


def require_gap(first, second):
    validate_clock(first["clock"], second["clock"])
    clock_require(instant(second["utc"]) - instant(first["utc"]) >= timedelta(hours=48)
            and second["clock"]["monotonic_ns"] - first["clock"]["monotonic_ns"] >= 48*3600*10**9,
            "Session 2 requires 48 hours of verified elapsed time after locked Session 1")


def display_to_image(u, boundary, zoom):
    require(zoom in (1, 2, 4) and all(math.isfinite(v) for v in (*u, *boundary)),
            "Invalid display transform")
    return [(u[i] - boundary[i]) / zoom - 0.5 for i in range(2)]


# Six synthetic targets exercise all zooms, origin and negative panning. No real pixels.
GRID_TRIALS = [(z, b, t) for z in (1, 2, 4)
               for b, t in (((20, 20), (10, 12)), ((-8, -6), (30, 28)))]


def validate_grid(trials, display):
    require(len(trials) == len(GRID_TRIALS), "Complete synthetic conversion check required")
    require(all(k in display for k in ("tk_scaling", "dpi", "screen", "os_scaling_disclosure"))
            and str(display["os_scaling_disclosure"]).strip(), "Display scaling disclosure required")
    require(all(type(display[k]) in (int, float) and math.isfinite(display[k]) and display[k] > 0
                for k in ("tk_scaling", "dpi"))
            and len(display["screen"]) == 2 and all(type(v) is int and v > 0 for v in display["screen"]),
            "Invalid display configuration")
    for row, (z, b, target) in zip(trials, GRID_TRIALS):
        require(row["zoom"] == z and row["boundary"] == list(b)
                and row["target"] == list(target), "Grid trial altered")
        xy = display_to_image(row["click"], b, z)
        require(all(abs(a-c) <= 0.5 for a, c in zip(xy, target)), "Synthetic click missed pixel center")


def validate_eye(eye, shape):
    require(set(eye) == {"measurable", "x", "y", "radius", "reason", "conversion"},
            "Invalid eye record")
    require(type(eye["measurable"]) is bool, "Boolean measurable flag required")
    if not eye["measurable"]:
        require(all(eye[k] is None for k in ("x", "y", "radius", "conversion"))
                and isinstance(eye["reason"], str) and eye["reason"].strip(),
                "Unmeasurable eye requires null coordinates and reason")
        return
    require(all(type(eye[k]) in (int, float) and math.isfinite(eye[k]) for k in ("x", "y", "radius")),
            "Finite eye coordinates/radius required")
    require(0 <= eye["x"] <= shape[1]-1 and 0 <= eye["y"] <= shape[0]-1
            and eye["radius"] >= 0.5, "Eye/radius outside protocol bounds")
    require(isinstance(eye["reason"], str) and (eye["radius"] == 0.5 or eye["reason"].strip()),
            "Explain larger uncertainty radius")
    c = eye["conversion"]
    require(all(k in c for k in ("click", "boundary", "zoom", "viewport", "display")),
            "Conversion metadata required")
    require(len(c["viewport"]) == 2 and all(type(v) is int and v > 0 for v in c["viewport"]),
            "Invalid viewport dimensions")
    xy = display_to_image(c["click"], c["boundary"], c["zoom"])
    require(all(abs(a-b) < 1e-9 for a, b in zip(xy, (eye["x"], eye["y"]))),
            "Click/coordinate mismatch")


def validate_eyes(eyes, shape):
    require(set(eyes) == set(SIDES), "Both image-side records required")
    for eye in eyes.values():
        validate_eye(eye, shape)
    left, right = (eyes[s] for s in SIDES)
    require(not (left["measurable"] and right["measurable"]) or left["x"] < right["x"],
            "Correct eye marks: image-left x must be smaller than image-right x")


class Session:
    """Exclusive sequential event files, hash chained; replay validates every transition.

    Tamper evident, not a security boundary against an owner rewriting the entire folder.
    An interrupted/incomplete session cannot be silently resumed or replaced.
    """
    def __init__(self, out, plan, number):
        require(number in (1, 2), "Invalid session")
        self.out, self.plan, self.number = Path(out), plan, number
        self.path = self.out / f"session-{number}"
        self.order = plan["randomization"][str(number)]["order"]

    @property
    def plan_digest(self):
        return digest(json_bytes(self.plan))

    def read(self):
        paths = sorted(self.path.glob("event-*.json"))
        events, previous = [], self.plan_digest
        for i, path in enumerate(paths):
            require(path.name == f"event-{i:04d}.json", "Event sequence gap")
            event = json.loads(path.read_bytes())
            stored = event.pop("digest")
            require(stored == digest(json_bytes(event)) and event["previous"] == previous
                    and event["sequence"] == i, "Session event integrity failure")
            event["digest"] = stored
            events.append(event)
            previous = stored
        self._validate(events)
        return events

    def _validate(self, events):
        grid, shown, count, took_break, ended = False, None, 0, False, False
        start, last, previous_clock = None, None, None
        anchor = clock_anchor(self.out, self.plan)["sample"] if events else None
        for i, e in enumerate(events):
            now = instant(e["utc"])
            require(e["clock_anchor_sha256"] == digest((self.out / "clock/anchor.json").read_bytes()),
                    "Clock anchor altered")
            require(e["utc"] == e["clock"]["utc"], "Event clock binding changed")
            validate_clock(anchor, e["clock"], previous_clock)
            previous_clock = e["clock"]
            require(not ended and (last is None or now >= last), "Invalid event chronology")
            last = now
            kind, data = e["kind"], e["data"]
            if i == 0:
                require(kind == "start" and data == {"session": self.number,
                        "reviewer": self.plan["reviewer"], "exposure": self.plan["exposure"],
                        "authorization": self.plan["authorization"]}, "Invalid session start")
                start = now
                continue
            require((now - start <= timedelta(minutes=60)
                     and e["clock"]["monotonic_ns"] - events[0]["clock"]["monotonic_ns"] <= 3600*10**9)
                    or kind == "stop", "Session exceeds 60 minutes")
            if kind == "grid":
                require(not grid and count == 0 and shown is None, "Grid already recorded")
                validate_grid(data["trials"], data["display"])
                grid = data
            elif kind == "present":
                require(grid and shown is None and count < 16 and (count < 8 or took_break),
                        "Grid/break/measurement prerequisite missing")
                require(data == {"opaque_id": self.order[count]["opaque_id"]}, "Unexpected presentation")
                shown = data["opaque_id"]
            elif kind == "measurement":
                require(shown and data["opaque_id"] == shown and set(data["eyes"]) == set(SIDES),
                        "Unexpected or duplicate measurement")
                asset = self.plan["verified"]["assets"][self.order[count]["group"]]
                validate_eyes(data["eyes"], asset["shape"])
                for eye in data["eyes"].values():
                    validate_eye(eye, asset["shape"])
                    if eye["measurable"]:
                        require(eye["conversion"]["display"] == grid["display"], "Display scaling changed")
                count += 1
                shown = None
            elif kind == "break":
                require(count == 8 and shown is None and not took_break, "Unexpected break")
                require(data == {"resumed": True}, "Break acknowledgement required")
                took_break = True
            elif kind == "note":
                require(isinstance(data.get("text"), str) and data["text"].strip(), "Empty disclosure")
            elif kind == "complete":
                require(count == 16 and grid and took_break and shown is None and data == {},
                        "Cannot lock incomplete session")
                ended = True
            elif kind == "stop":
                require(isinstance(data.get("reason"), str) and data["reason"].strip(), "Stop reason required")
                ended = True
            else:
                raise ValueError("Unknown event kind")

    def append(self, kind, data, now=None, *, _clock=None):
        require(not (self.path / "lock.json").exists(), "Session is locked")
        events = self.read()
        sample = _clock or clock_sample(self.out, self.plan)
        require(now is None or now.isoformat() == sample["utc"], "Caller time differs from clock witness")
        event = dict(sequence=len(events), previous=events[-1]["digest"] if events else self.plan_digest,
                     utc=sample["utc"], clock=sample,
                     clock_anchor_sha256=digest((self.out / "clock/anchor.json").read_bytes()),
                     kind=kind, data=data)
        event["digest"] = digest(json_bytes(event))
        self._validate(events + [event])
        write_new(self.path / f"event-{len(events):04d}.json", event)

    def start(self, now=None):
        require(read_plan(self.out, self.plan["verified"]) == self.plan, "Plan binding changed")
        sample = clock_sample(self.out, self.plan)
        require(now is None or now.isoformat() == sample["utc"], "Caller time differs from clock witness")
        if self.number == 2:
            first = Session(self.out, self.plan, 1).locked()
            require_gap(first[-1], dict(utc=sample["utc"], clock=sample))
        self.path.mkdir(exist_ok=False)
        self.append("start", dict(session=self.number, reviewer=self.plan["reviewer"],
                                  exposure=self.plan["exposure"], authorization=self.plan["authorization"]), now, _clock=sample)

    def lock(self, now=None):
        self.append("complete", {}, now)
        events = self.read()
        write_new(self.path / "lock.json", dict(plan_digest=self.plan_digest,
                  event_count=len(events), events_digest=digest(json_bytes(events)),
                  start_utc=events[0]["utc"], end_utc=events[-1]["utc"], **GATES))

    def locked(self):
        lock = json.loads((self.path / "lock.json").read_bytes())
        events = self.read()
        require(events and events[-1]["kind"] == "complete"
                and lock == dict(plan_digest=self.plan_digest, event_count=len(events),
                                 events_digest=digest(json_bytes(events)), start_utc=events[0]["utc"],
                                 end_utc=events[-1]["utc"], **GATES), "Invalid session lock")
        return events


def analysis_sessions(out, plan):
    require(read_plan(out, plan["verified"]) == plan, "Plan binding changed")
    first, second = (Session(out, plan, s).locked() for s in (1, 2))
    require_gap(first[-1], second[0])
    return first, second


class Presenter:
    """Narrow reviewer interface. Public image payload contains no provenance or targets."""
    def __init__(self, root, session):
        self._root, self._session = Path(root), session

    def start(self):
        self._session.start()

    def grid(self, trials, display):
        self._session.append("grid", dict(trials=trials, display=display))

    def current(self):
        events = self._session.read()
        require(any(e["kind"] == "grid" for e in events), "Synthetic validation required")
        require(not any(e["kind"] in ("complete", "stop") for e in events), "Session ended")
        count = sum(e["kind"] == "measurement" for e in events)
        require(count < 16, "Session complete")
        item = self._session.order[count]
        if events[-1]["kind"] != "present":
            self._session.append("present", {"opaque_id": item["opaque_id"]})
        image = load_asset(self._root, self._session.plan["verified"]["assets"][item["group"]])
        return {"opaque_id": item["opaque_id"], "pixels": image}

    def submit(self, opaque_id, eyes):
        self._session.append("measurement", dict(opaque_id=opaque_id, eyes=eyes))

    def resume_break(self):
        self._session.append("break", {"resumed": True})

    def note(self, text, invalidating=False):
        self._session.append("stop" if invalidating else "note",
                             {"reason" if invalidating else "text": text})

    def stop(self, reason):
        self._session.append("stop", {"reason": reason})

    def finish(self):
        self._session.lock()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("preflight")
    prep = sub.add_parser("prepare")
    for arg in ("reviewer", "exposure", "authorization", "seed1", "seed2"):
        prep.add_argument("--" + arg, required=True)
    review = sub.add_parser("review")
    review.add_argument("--session", type=int, choices=(1, 2), required=True)
    sub.add_parser("analyze")
    sub.add_parser("clock-guard")
    args = parser.parse_args(argv)
    try:
        verified = preflight(args.root)
        if args.command == "preflight":
            print("Preflight passed: 6 cases, 24 associations, 16 verified groups. No session started.")
        elif args.command == "prepare":
            prepare(args.root, verified, args.reviewer, args.exposure, args.authorization,
                    (args.seed1, args.seed2))
            print("Private plan frozen. No session started. Keep the coordinator directory private.")
        else:
            out = output_directory(args.root)
            plan = read_plan(out, verified)
            if args.command == "clock-guard":
                clock_guard(out, plan)
            elif args.command == "review":
                from .msu_pilot_ui import run
                run(args.root, Session(out, plan, args.session))
            else:
                sessions = analysis_sessions(out, plan)
                from .msu_pilot_analysis import analyze
                result = analyze(args.root)
                write_new(out / "analysis.json", result)
                print("Analysis written to private coordinator output. Scientific gates unchanged.")
    except ClockIntegrityError as error:
        print("STOP: " + str(error), file=sys.stderr)
        return 1
    except (ValueError, OSError, KeyError, TypeError, IndexError, subprocess.CalledProcessError):
        # Do not echo paths, group identities, targets or a traceback into the reviewer view.
        print("STOP: validation or execution failed. No substitution or automatic restart.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
