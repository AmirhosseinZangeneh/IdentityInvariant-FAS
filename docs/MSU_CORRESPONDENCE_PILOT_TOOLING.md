# Local tooling for the frozen exploratory pilot

This tool implements [protocol version 1](MSU_CORRESPONDENCE_PILOT_PROTOCOL.md).
Implementation and synthetic tests do not authorize a reviewer session.
No real reviewer measurements or candidate analysis were produced by this task.
P4, null frozen policy, pending fidelity and false experiment readiness remain
unchanged. One actual reviewer can perform the two sessions; this is not
independent confirmation or an untouched blinded study.

## Components and private records

`identity_invariant_fas.data.msu_pilot` performs preflight, freezes a plan and
validates the append-only ledger. `msu_pilot_ui` is a local Tkinter interface
using Pillow nearest-neighbor display, without an HTTP server. It does not
import the analysis module. `msu_pilot_analysis` implements section 7 only for
the separately invoked analysis command. Install the project's existing
`video` extra for OpenCV; Pillow is already a core dependency. Windows Python
must include Tkinter. No new runtime package is introduced.

Protocol exact-byte SHA-256:
`dfbd88ad785bf32b947e8f2c54aafc3a593d5474112dae6b4eb864cf439b39a4`.
The original protocol and private manifest are never rewritten. Preflight
checks the fixed manifest digest, 6/24/16/16 coverage, historical files against
commit `0f9dd0f9e122909475027491fbe9b9173828e7de`, manifest-bound evidence hashes,
original selected annotation-file hashes/rows, and all sixteen retained image
file hashes, dimensions, uint8 type and full BGR pixel hashes. It uses image
decoding solely to verify existing PNG/BMP arrays; it has no video decoder,
frame search, export, substitution or recovery path. Exact protocol bytes are
required, including line endings; a mismatch stops rather than rewriting them.

Output is fixed beneath ignored
`data_processed/recovery/msu-correspondence-pilot-v1/`:

- `plan.json`: actual reviewer, disclosure, separate authorization reference,
  input and tool/environment fingerprints, both seeds/orders, and private
  opaque-ID mappings. Preparation creates a fresh directory exclusively.
- `plan-seal.json`: exclusive preparation-time SHA-256 anchor covering the entire
  canonical plan, including protocol/evidence/software fingerprints, reviewer
  identity/disclosure, authorization, preparation time, seeds, orders and mappings.
  A partial/truncated seal fails closed; neither prepare nor read repairs it.
- `clock/anchor.json`: exclusive clock-witness epoch, initial UTC/monotonic
  sample, plan binding and private localhost connection credentials. A failure
  marker permanently blocks continuation. Event digests also bind this anchor.
- `session-1/` and `session-2/`: exclusive numbered JSON event files, chained
  by SHA-256 to the plan; grid clicks, display settings, presentation events,
  measurements, disclosure/stop events, UTC and independent monotonic samples,
  clock-anchor fingerprint and break acknowledgement.
- `lock.json` within each completed session: digest of all events, their count,
  plan digest, start/end UTC and unchanged scientific gates. The lock is
  exclusive; later event writes are refused.
- `analysis.json`: created exclusively by the separate analysis command after
  both sessions and the 48-hour separation pass replay validation.

These files and licensed images must stay private and uncommitted. The tool
does not duplicate the original images. A coordinator should retain backups
and lock digests separately. Hash chains detect accidental editing/truncation
and concurrent writes fail on exclusive creation, but this is not a security
boundary against a malicious filesystem owner who replaces every private
artifact, including the separate seal and clock anchor. The seal detects
ordinary post-prepare changes, including coherent seed/order/mapping edits.
Truthful exposure/reviewer disclosures remain necessary. Clock continuity is
enforced in software as described below, not accepted on a procedural warning.

## Later workflow, only after separate authorization

Run from the repository root with `env_cuda`. The following commands are a
future workflow, not commands to run during implementation review.

1. The coordinator verifies the actual reviewer is willing, records prior
   pack/protocol exposure and a separate execution authorization reference.
   Keep the coordinator terminal and private directory away from the reviewer.
   Read-only preflight may be run without starting a session:

   ```powershell
   .\env_cuda\Scripts\python.exe -m identity_invariant_fas.data.msu_pilot preflight
   ```

2. Choose two different seed strings once, before any presentation. Freeze the
   plan with actual values replacing the quoted placeholders:

   ```powershell
   .\env_cuda\Scripts\python.exe -m identity_invariant_fas.data.msu_pilot prepare --reviewer "ACTUAL REVIEWER" --exposure "ACTUAL PRIOR EXPOSURE DISCLOSURE" --authorization "SEPARATE AUTHORIZATION REFERENCE" --seed1 "FROZEN SESSION ONE SEED" --seed2 "FROZEN SESSION TWO SEED"
   ```

   The algorithm is `sha256-sort-v1`: order the sixteen groups lexically by
   SHA-256 of canonical JSON `[algorithm, "order", seed, session, group]`;
   use group ID only as a collision tiebreaker. Opaque IDs instead use 32 bytes
   from `secrets.token_hex`, with an `image-` prefix, independently for each
   group/session; they are not derived from seeds or group identifiers. Freeze
   both mappings in the sealed private plan. Seeds need not be secret or
   high-entropy to preserve ID secrecy. Record algorithm and realized orders; no Session 1 sequence or marks are loaded into Session
   2's interface. Preparing the plan does not start either session.

3. In a separate private coordinator terminal, start the clock witness before
   Session 1, and keep it running continuously through Session 2 and analysis:

   ```powershell
   .\env_cuda\Scripts\python.exe -m identity_invariant_fas.data.msu_pilot clock-guard
   ```

   It listens only on localhost using a private random credential and provides
   fresh samples from the same process to session events. It polls at most
   every 0.5 seconds while idle, comparing UTC to `time.monotonic_ns()` against
   both the previous poll and the initial anchor. Forward/backward corrections
   exceeding 2 seconds, a backwards clock, or a poll gap exceeding 10 seconds
   fail closed. These conservative integrity tolerances are operational
   choices, not validated scientific thresholds. Small cumulative clock drift
   beyond 2 seconds also stops the witness. Neither tolerance subtracts from
   the required 48 hours: both UTC and monotonic elapsed time must independently
   reach 172800 seconds after Session 1 locks.

   Keep the machine awake; sleep, hibernation, reboot, process termination,
   debugger suspension, or excessive scheduling delays are unsupported. The
   exclusive clock directory prevents restarting the witness; a dead witness
   cannot answer fresh requests. Preserve all records if continuity fails and
   obtain a separately documented restart decision. Do not delete the directory
   to retry. This intentionally conservative implementation requires an awake
   workstation for at least 48 hours; one reviewer is sufficient but cannot
   substitute recollection or a wall-clock correction for this evidence.

   Then start the separately authorized Session 1 from another terminal:

   ```powershell
   .\env_cuda\Scripts\python.exe -m identity_invariant_fas.data.msu_pilot review --session 1
   ```

   Preflight repeats and must match the frozen plan, including software source
   fingerprints. Confirm the actual reviewer identity and disclose Windows
   display scaling, monitor and remote-desktop settings. Failure to establish
   Windows per-monitor DPI awareness stops before any session starts. The session starts
   with six synthetic clicks: two known pixel centers at each of 1x, 2x, 4x,
   including negative viewport offsets. A click must land within the target
   pixel (0.5 original pixel per axis). This is an input-validation tolerance,
   not an added scientific discrimination threshold. Every synthetic pixel
   center is also exercised in automated conversion tests. No real image is
   presented until all six trials pass and their events are persisted.

4. For each opaque image choose image-left, click its visible pupil center,
   enter uncertainty radius, and explain any radius above 0.5 pixel. Repeat
   for image-right. Alternatively mark the eye unmeasurable and give a reason;
   its coordinates, radius and conversion metadata are saved as null. Use
   1x/2x/4x nearest-neighbor zoom and right-drag panning. The UI shows no target
   coordinates, original filename, labels, domains, ordinals, previous
   preferences, mapping, rankings or scores. It shows one image at a time and
   no persistent markers on that image. Clicks remain editable until saving
   both eyes; submitted records cannot be overwritten or silently revisited.
   Display metadata and the transform used for each click are recorded. Both
   measurable eyes must satisfy image-left.x < image-right.x. Reversed or equal
   x marks prompt correction and are never silently swapped or target-matched.

5. After eight images the image is removed and a break is required; click
   Resume after break when ready. Image clicks on the blank break screen are
   ignored; only Resume advances. Disclose recognition with the disclosure
   button; its target-exposure question stops an invalidated session. Use Stop
   for annotation-target exposure, fatigue, technical failure
   or withdrawal. A display-scaling change or 60-minute session limit stops
   the session. The limit includes the synthetic check and break. Finishing
   all sixteen images locks the session automatically without calculating any
   scores. Keep that digest privately. Incomplete/invalid sessions are not
   resumable in this deliberately small tool. There is no force/restart option;
   preserve records and obtain a documented restart decision before designing
   a fresh run. Never delete or edit records to bypass a failed prerequisite.

6. At least 48 hours after the valid locked Session 1 end time:

   ```powershell
   .\env_cuda\Scripts\python.exe -m identity_invariant_fas.data.msu_pilot review --session 2
   ```

   The command refuses an early, incomplete, invalid or altered first session.
   Session 2 has its own fresh synthetic check, opaque IDs and frozen order.
   Do not view Session 1 marks/order, old overlays or preferences between
   sessions. Reading old records is not prevented for the same filesystem
   owner; disclose any access instead of claiming perfect blinding.

7. Only after both valid sessions are locked, the coordinator may separately
   invoke analysis under its execution authorization:

   ```powershell
   .\env_cuda\Scripts\python.exe -m identity_invariant_fas.data.msu_pilot analyze
   ```

   This reruns preflight and verifies both event chains, coverage, locks and
   separation before calculation. The public `msu_pilot_analysis.analyze(root)`
   API independently performs these same checks; it accepts no raw session
   lists. Each session summary includes `gap_hours`: null for Session 1, and
   for Session 2 the validated monotonic elapsed time from Session 1 completion
   to Session 2 start, in hours. This is descriptive only; the existing dual-clock
   eligibility check remains authoritative. UTC start/end timestamps are retained.
   The arithmetic core is private. Keep the clock witness running until
   analysis finishes; a witness failure conservatively blocks later analysis. Outputs include both fixed
   origin conventions, uncertainty bounds, per-eye repeatability, exact label
   ties, all pairwise distinct-group comparisons, missingness and six case
   outcomes. Cross-case images reuse the same marks. No group duplicates are
   counted as new observations. Causal statuses default to unresolved except
   exact pixel ties; the tool makes no decoder-only or temporal attribution.
   No global winner or scientific gate promotion is produced.

The reviewer interface masks labels, not recognizable content or prior human
exposure. It is not protection against a reviewer inspecting source files or
the coordinator's records. If the reviewer is also the coordinator, disclose
that limitation. The synthetic gate must be completed by the actual reviewer
on the actual display before either session; automated UI tests cannot certify
that future display configuration.

## Verification without pilot execution

`tests/test_msu_pilot.py` uses synthetic arrays and synthetic session records in
pytest temporary directories, never licensed candidate pixels or real marks.
It tests frozen metadata coverage, exact ties, hash/type/dimension failures,
opaque payloads, deterministic order, conversion, missing landmarks,
append/lock validation, early Session 2, analysis gates, and P4 preservation.
A Tk smoke test uses a hidden synthetic grid when a display is available;
headless hosts skip that GUI check. The preflight command verifies real asset
integrity without displaying, measuring or scoring them.

Additional synthetic regressions cover plan/seal tampering, same seeds,
cryptographic ID independence, actual elapsed-time versus wall-clock jumps,
malformed timestamps, unsupported continuity loss, reversed/coincident marks,
break clicks, event/lock corruption and direct analysis API bypass attempts.
No test establishes physical display calibration on a future Windows monitor;
DPI setup, stable display metadata and the actual-reviewer synthetic check
remain necessary at execution. Remote-desktop scaling changes require stopping.

The independent clock uses Python's monotonic counter, whose reference is
unaffected by system clock updates; see the [Python time documentation](https://docs.python.org/3.12/library/time.html#time.monotonic).
The implementation requires a single uninterrupted witness process rather than
assuming a counter remains comparable after a reboot or service restart.
