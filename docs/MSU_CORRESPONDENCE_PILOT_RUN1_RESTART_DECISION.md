# MSU correspondence pilot: Run 1 restart decision

Decision recorded 2026-09-26. **Run 1 classification: `operationally_aborted`.**
This is an operational disposition, not a scientific outcome or an amendment
to Run 1's records. It authorizes neither execution nor retrospective repair.

## Basis and evidence limits

The repository was on `research-reconstruction` at frozen tooling commit
`8ef9a86cebbb67a5f3159d91ab84010779fcc5e2`, with a clean working tree before
this documentation task. The governing methodology was
[protocol v1](MSU_CORRESPONDENCE_PILOT_PROTOCOL.md), with exact-byte SHA-256
`dfbd88ad785bf32b947e8f2c54aafc3a593d5474112dae6b4eb864cf439b39a4`.
The [v1 tooling guide](MSU_CORRESPONDENCE_PILOT_TOOLING.md) required one
uninterrupted local clock witness spanning the inter-session interval.

The operator's execution report establishes the following chronology:

1. Run 1 Session 1 completed successfully and locked successfully under v1.
2. Session 2 was never started. No pilot analysis was run.
3. After Session 1, frequent/daily power outages were recognized as making
   the continuous 48-hour clock-witness requirement operationally infeasible.
4. The witness was intentionally terminated with Ctrl+C. The observed
   termination was `KeyboardInterrupt` in `clock_guard()`.
5. `clock/failed.json` was observed absent after termination, while
   `session-1/lock.json` still existed. The working tree was clean.

Read-only path-existence checks during this documentation task corroborated
the lock's presence, the failure marker's absence, and the absence of
`session-2/` and `analysis.json` in the Run 1 root. These checks did not open
the lock, measurement events, mapping, plan, or images. They are not a new
lock-validation replay, and file absence alone is not proof that an operation
never ran; the no-Session-2/no-analysis statements come from the operator.
Exact execution times are not inferred or fabricated here.

The missing `failed.json` despite the observed KeyboardInterrupt is retained
as an execution fact. Its cause is unresolved. Do not manufacture the marker,
change the exception history, or reinterpret its absence as evidence of
continuity. No retrospective diagnosis or repair is performed by this decision.

## Disposition and preservation

Run 1 is **operationally aborted**, because its timing mechanism is incompatible
with the execution environment. This disposition is unrelated to candidate
measurements or preferences. No scientific result, repeatability result,
geometric ranking, or historical correspondence conclusion is drawn from Run 1.

The existence of a valid Session-1 lock does **not** authorize pairing it with
any future Session 2, under either protocol version. The completed session is
retained as historical execution evidence only; it is not a transferable first
half of another run. Do not score it, import its marks, or use its outcomes to
choose a policy, frames, thresholds, or the next run's ordering.

Preserve the entire private root unchanged:
`data_processed/recovery/msu-correspondence-pilot-v1/`.
Do not alter, delete, restart, repair, reuse, or relabel its records. In
particular, do not restart the v1 witness, synthesize `failed.json`, replace a
lock, or create a Session 2 there. This public decision is separate from the
private ledger and does not write an abort marker into that ledger.

This document includes no private measurement values, ordering, opaque
mapping, candidate identities, or private-record digests. No private contents
were inspected or uploaded in preparing it.

## Prospective restart only

[Protocol v2](MSU_CORRESPONDENCE_PILOT_PROTOCOL_V2.md) proposes a new timing
evidence mechanism while retaining the minimum **48 real elapsed hours**.
It requires a new private root and a complete new sequence:
`prepare -> Session 1 -> >=48h -> Session 2 -> analysis`.
No portion of Run 1 supplies a session for that sequence.

Before a new run, finish and freeze the v2 timing profile, implement and audit
its tooling, and obtain separate execution authorization. The actual reviewer
must disclose the additional Run 1 exposure without reopening its records.
The new run remains exploratory and previously exposed, not an untouched or
independent confirmation. This restart does not erase that exposure.

Preserve exactly: `policy_state=P4`, `frozen_policy=null`,
`fidelity_status=manual_review_pending`, `experiment_ready=false`.
No current pilot result changes those states. V1 methodology, historical
audits and human-review responses remain unchanged.
