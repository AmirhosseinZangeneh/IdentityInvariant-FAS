# Frozen six-case exploratory annotation-alignment pilot

Protocol version 1, drafted 2026-09-25 against `research-reconstruction` at
`1ba31f3b794f9c471e352f99a0bab189c2b4cbcd`. **Design only; execution status:
`not_executed`.** This document specifies a prospective measurement procedure
after prior visual review and asset recovery. It is not an untouched blinded
confirmation, an externally registered study, or authorization to execute.
Record this document's exact byte hash before any separately authorized run;
later amendments must be dated and must not silently replace this version.

Preserve `policy_state=P4`, `frozen_policy=null`,
`fidelity_status=manual_review_pending`, and `experiment_ready=false` regardless
of pilot outcome. Asset-only GO means image availability and integrity, not
PittPatt correspondence. No PAD accuracy, official TEST observations, training,
policy selection, or scientific readiness promotion is part of this protocol.

## 1. Question, hypotheses, and unit of analysis

Exploratory question: within these six previously reviewed training cases, can
repeat manual eye localization distinguish geometric agreement with the
published `.face` eye coordinates among the frozen image candidates, at the
resolution of the reviewer's stated uncertainty?

Competing descriptive hypotheses are H-N (native ordinal i aligns best), H-N−
(native i−1), H-N+ (native i+1), and H-E (DecFrames export ordinal i). H-T permits
exact pixel ties or operationally unresolved differences; H-M permits different
case outcomes; H-U permits unmeasurable or unstable results. None asserts that
PittPatt received the winning pixels. The candidate set may omit its actual
input, and the manual landmark definition may differ from the SDK's definition.

All cases belong to **one training subject, client 002**, across three source
videos. Six annotation rows, sixteen unique pixel arrays, and twenty-four
candidate associations are not independent human subjects or independent
acquisitions. Adjacent P cases share images and annotation context. Report
case-level descriptive comparisons only; no pooled significance tests,
confidence intervals based on independent cases, majority-vote policy, or
release-wide accuracy claim. Review sessions below are measurement sessions;
they do not establish dataset acquisition sessions (`session_id` remains null).

## 2. Evidence and design-time checks

Read together with [PittPatt provenance](MSU_PITTPATT_PROVENANCE.md),
[policy gate](MSU_POLICY_GATE.md), [human review](MSU_HUMAN_REVIEW.md), and
[committed human pack](audit/msu_policy_human_pack.json). Retained-export
metadata is in [bundled-export evidence](audit/msu_index_bundled_export.json)
and [export calibration](audit/msu_policy_export_calibration.json).

The working tree was clean at design entry; branch and full HEAD matched the
baseline above. All 29 checked historical files (the tracked files under
`docs/audit/` plus the three named MSU documents) matched HEAD, allowing checkout
LF/CRLF conversion. The six
evidence files fingerprinted by the recovery manifest also matched their
recorded working-file and committed-blob SHA-256 values. The final response
retains its exact-byte hash
`385fe217e56e95d172d0b91d7f0237e1a9462a77073ae3297f34a01adf89afa4`.
The human-pack committed-blob SHA-256 is
`b0ad8ef0ae6e48615c842f1be5cbe98c3ace1f2f2d1498890532d38c6edd10d9`.

Private execution evidence: `data_processed/recovery/msu-native-six-20260924-v1/manifest.json`.
Its design-time exact-byte SHA-256 is
`b93711932627c4d0407b58836e81e1ba5f890a81060e6e96434f4f969d130028`.
It reports `asset_only_status=GO` and `frozen_case_coverage` of 6 cases,
24 candidate associations, 16 distinct oriented-pixel groups, and 16 verified
clean groups. The manifest is **private execution evidence, not a committed
scientific audit**. This document does not publish its absolute workstation
paths or promote its contents into historical provenance.

The manifest records selected `env_cuda` Python 3.12.10, NumPy 2.5.2, OpenCV
5.0.0, FFmpeg component versions and binary fingerprints. Six recovered images
matched frozen full BGR uint8 pixel hashes and PNG reload hashes. Ten retained
BMP representatives were separately reverified on Windows after recovery.
Its execution-order note explicitly records that the precise ten-file inventory
and request to check it first arrived after six-frame recovery had finished.
Do not rewrite that sequence as an all-sixteen pre-recovery verification.

The prior read-only Ubuntu asset inventory identified the cases and
groups below; no separate committed six-case recovery design was found.
The mapping below was checked against committed human-pack metadata and the
Windows private manifest. At design time all sixteen paths were checked for
existence only: **no candidate image was opened, decoded, inspected, or measured**.
Pixel verification statements refer to the prior private recovery execution.
Fresh byte/pixel verification is a prerequisite of any later authorized run.

## 3. Frozen cases and all candidate associations

Case aliases below are unambiguous abbreviations for these complete IDs:

| Alias | Exact case ID |
| --- | --- |
| A1-case | `MSU-MFSD:scene01/real/real_client002_android_SD_scene01.mp4:annotation:1` |
| A150-case | `MSU-MFSD:scene01/real/real_client002_android_SD_scene01.mp4:annotation:150` |
| P115-case | `MSU-MFSD:scene01/attack/attack_client002_android_SD_printed_photo_scene01.mp4:annotation:115` |
| P116-case | `MSU-MFSD:scene01/attack/attack_client002_android_SD_printed_photo_scene01.mp4:annotation:116` |
| P117-case | `MSU-MFSD:scene01/attack/attack_client002_android_SD_printed_photo_scene01.mp4:annotation:117` |
| L119-case | `MSU-MFSD:scene01/real/real_client002_laptop_SD_scene01.mov:annotation:119` |

N=`native_i`, N−=`native_i_minus_1`, N+=`native_i_plus_1`,
E=`decframes_export_i`. Each matrix cell is one association, written
`domain ordinal → pixel group`. There are exactly 24 cells.

| Case | N (native) | N− (native) | N+ (native) | E (export) |
| --- | --- | --- | --- | --- |
| A1-case | 1 → A1 | 0 → A2 | 2 → A3 | 1 → A1 |
| A150-case | 150 → A4 | 149 → A5 | 151 → A6 | 150 → A5 |
| P115-case | 115 → P1 | 114 → P2 | 116 → P3 | 115 → P4 |
| P116-case | 116 → P3 | 115 → P1 | 117 → P5 | 116 → P2 |
| P117-case | 117 → P5 | 116 → P3 | 118 → P6 | 117 → P1 |
| L119-case | 119 → L1 | 118 → L2 | 120 → L3 | 119 → L4 |

For lossless reconstruction of all candidate IDs, let S be the case ID with
`:annotation:i` removed. A native candidate ID is `S:native:j`. An export ID
is `S:domain:decframes_export:ordinal:i:policy:D`, where D is
`92a81eefbe44dd485d3baa9955b1fa8f7f15f029865069e349051792774bd318`.
Preserve the original annotation i independently from j and export filename
number i+1. A filename number is never treated as a native ordinal.

A1 and A5 are exact within-case label ties. P1, P2, P3, and P5 are reused
across cases. Measure each group once per reviewer/session, then reuse its
coordinates unchanged for every association. Cross-case residuals can differ
because the annotation targets differ; this does not create new measurements.

## 4. Clean-image inventory

All paths here are repository-relative. Directory aliases concatenate with
the exact filenames below; they are not new directories to create:

```text
A-dir = data_processed/MSU-MFSD/index-evidence/bundled-export-v2/31225a6160c602538da5baa907a1c42b3edcadb299ac2345d2dd80d0ecc84a8b/
P-dir = data_processed/MSU-MFSD/policy-gate/calibration-v2/retained/a29653c14464fca59ce54f38fc6dca62b2e3e242a3a6a6a668aafd11762f0a8b/
L-dir = data_processed/MSU-MFSD/policy-gate/calibration-v2/retained/5b073f6562039a8415cfdfc531c947435c8acff7960070da5d880305bf846a2a/
R-dir = data_processed/recovery/msu-native-six-20260924-v1/
```

Groups A1–A6 and P1–P6 are 720×480; L1–L4 are 640×480 (width×height).
All have three BGR uint8 channels and recorded rotation 0 degrees. Hashes
below cover the full oriented pixel array in C byte order, not compressed file
bytes. File-byte hashes and verification statuses are retained in the private
manifest. No overlays, crops, screenshots, or JPEG replacements qualify.

| Group | Directory and exact filename | Expected oriented-pixel SHA-256 |
| --- | --- | --- |
| A1 | A-dir / `real_client002_android_SD_scene01_002.bmp` | `88072c5959f977fbb06fc5a353eb7ae6c162880ae7bcd5b196fde8c47e70ea0c` |
| A2 | A-dir / `real_client002_android_SD_scene01_001.bmp` | `93f29d815fcb317fbc8abd212e9326a247a6bd1c33a53f8d7a8ab8b87d2e543a` |
| A3 | A-dir / `real_client002_android_SD_scene01_003.bmp` | `11a1848ebd7f4df405b11b808ac0fe31921627745280f202956266f2d40a69ef` |
| A4 | A-dir / `real_client002_android_SD_scene01_152.bmp` | `95bf96fdb9997d111722e65539169eb8e49e32fb2d71338c0f7d3d4cd6db0236` |
| A5 | A-dir / `real_client002_android_SD_scene01_151.bmp` | `8f3ba09de9e5160ceacd4408d19f072f4369f3709d570ac7c411276d4b1f5a40` |
| A6 | A-dir / `real_client002_android_SD_scene01_153.bmp` | `76750361bdaba8dc4b9a4b232b4862953875f9003cd02872ecb33afaf68b0676` |
| P1 | P-dir / `attack_client002_android_SD_printed_photo_scene01_118.bmp` | `5d2c81fd2675b0831a8d7e0005d3094b47294be65456c28aa9d55b66ceab8872` |
| P2 | P-dir / `attack_client002_android_SD_printed_photo_scene01_117.bmp` | `f5ee08a85f2d2402d8ccdf81b8b7a9925407fb159af6e496e41963fb3bf25f1b` |
| P3 | R-dir / `attack_client002_android_SD_printed_photo_scene01_native_000116.png` | `1a2a4734a68d1563050d5cb0ccc3e9bbf1d6bfcdf00315bd6ee9ad0ae97334c9` |
| P4 | P-dir / `attack_client002_android_SD_printed_photo_scene01_116.bmp` | `bffa05cbc687e6c5f0091d9a8bb8f8234ecd6073a326a14d97f86d6c3a356b64` |
| P5 | R-dir / `attack_client002_android_SD_printed_photo_scene01_native_000117.png` | `e2a57c70be80e8c894ef72e19fd6ad3046e8a539b623b61e59c7810369681271` |
| P6 | R-dir / `attack_client002_android_SD_printed_photo_scene01_native_000118.png` | `1afa8bdee4e29d45f37e98dc793bc319b47713d0a0574d71ebd1c28dbc2a2701` |
| L1 | R-dir / `real_client002_laptop_SD_scene01_native_000119.png` | `c8dd94c2e526cdd1ddb7bbce34c2c647e766709a87e563a0b29e314071caf68e` |
| L2 | R-dir / `real_client002_laptop_SD_scene01_native_000118.png` | `23a3f0484fc44c4e9b298303253e463c8a66d6fcb001b04b4cdd2294c20629fc` |
| L3 | R-dir / `real_client002_laptop_SD_scene01_native_000120.png` | `6358e8a8ed7914b3c8809971f9709648025f5e0264e51208ffc133b623d10cd3` |
| L4 | L-dir / `real_client002_laptop_SD_scene01_120.bmp` | `965dd4c91e700000b5264e9e7c9e87cdee4ed700c044b00ba6f2c69dab21934a` |

Keep licensed images and future reviewer records private in ignored storage.
The manifest's two recovery-source byte hashes are
`6b176489e0774f9a87addf3d35cdc03cf8ed35f8dbd34b837c29a90025613d61`
for P and `0284ab9e573c65f4bbda340d01b1f82a9c4c20945436acb63530835961595861`
for L. It does not supply a new full-video hash for A; no such claim is made.

## 5. Coordinates and landmark definition

Use the complete oriented image without mirroring or additional rotation.
Operational image coordinates place the top-left pixel center at (0,0), x
increases rightward and y downward; the bottom-right center is (W−1,H−1).
Retain floating-point coordinates without rounding annotation values. This
pixel-center convention is a declared measurement convention, not recovered
PittPatt provenance. README names eye coordinates but does not specify their
anatomical target, coordinate origin, or subject-versus-image laterality.

Parse the original `.face` row as index, left/top/right/bottom box coordinates,
then eye pair 1=(field 6,field 7), eye pair 2=(field 8,field 9), counting fields
from one. Preserve the raw row, line number, and annotation-file SHA-256 from
the human pack. Assign the smaller-x annotation pair to **image-left** and the
larger-x pair to **image-right**, recording the pair permutation. In all six
frozen rows pair 1 has smaller x. Never relabel using presumed anatomical
left/right or choose an assignment that minimizes a candidate's error.
Equal x, nonfinite values, or nonpositive inter-eye distance is a metadata
failure; do not resolve it by inspecting candidates.

| Case | `.face` line | Image-left raw (x,y) | Image-right raw (x,y) |
| --- | --- | --- | --- |
| A1-case | 2 | (301.93,253.55) | (372.62,245.30) |
| A150-case | 151 | (282.95,258.15) | (354.63,249.91) |
| P115-case | 116 | (336.30,225.10) | (432.32,231.87) |
| P116-case | 117 | (335.45,225.75) | (431.85,231.82) |
| P117-case | 118 | (334.71,226.17) | (430.18,232.80) |
| L119-case | 120 | (362.49,189.80) | (418.58,189.38) |

The manual landmark is the center of the visible pupil region in each eye,
excluding a specular highlight when its surrounding boundary is visible.
For P it is the pupil depicted on the printed face, not any live presenter.
If the pupil cannot be distinguished, record that eye as unmeasurable; do not
substitute iris center, eyelid midpoint, face-box center, model output, or
the published annotation. This deliberately conservative definition can make
the entire pilot inconclusive. It is a geometric proxy, not a claim about the
SDK's eye landmark semantics.

For each measurable eye record a chosen center and a nonnegative uncertainty
radius in original pixels, enclosing positions the reviewer considers plausible.
Minimum radius is 0.5 pixel; record why a larger radius is needed. An eye for
which no meaningful finite region can be assigned is unmeasurable, with null
coordinates/radius and a reason (blur, occlusion, ambiguous pupil, or other).
These radii are subjective uncertainty regions, not calibrated probabilities.
Do not force a maximum radius to make comparisons decisive.

Future measurement software must show clean images, hide annotation marks and
numeric targets, and validate click-to-image conversion on a synthetic grid
before candidate viewing. Allow 1×, 2×, and 4× nearest-neighbor display zoom
and panning only; record scale, viewport origin, display size and OS scaling.
With displayed image top-left boundary b and scale z, convert a click u by
`x=(u_x−b_x)/z−0.5` and likewise y, after mapping event coordinates into the
same display units. Validate that every displayed pixel center maps correctly.
No smoothing, enhancement, automatic registration, alignment, or altered image
files. A transient cursor is permitted during a future authorized session;
do not create annotated media as an output of this design task.

Primary comparisons use raw annotation coordinates unchanged. A mandatory
coordinate-origin sensitivity check subtracts (1,1) from both annotation eyes
to represent a possible one-based origin. It never changes measured centers.
Report both fixed conventions; do not select the convention that favors a
label. Agreement across these checks does not settle SDK landmark semantics,
half-pixel conventions, or face-box endpoint inclusivity. No box measurements
or box-derived error scores enter this pilot.

## 6. Feasible reviewer procedure and exposure

The actual historical reviewer is **Amirhossein Zangeneh**, documented in the
human-review record. His participation in this pilot is proposed, not confirmed.
No additional reviewer is currently established. One willing reviewer can
realistically perform two sessions of sixteen images: 32 image presentations
and at most 64 eye-center/radius records. Reserve up to 60 minutes per session,
with a break after eight images; incomplete sessions remain incomplete.
These are workload planning choices, not measured timing estimates.

After separate authorization, identify the actual participant and disclose
prior pack review, protocol/mapping access, candidate recognition and any
knowledge of old preferences. Use opaque image IDs and a fresh randomized
order in each session. Freeze and record the randomization algorithm and seeds
before viewing; retain the mapping privately. Show each group once in each
session, with no side-by-side candidate comparisons or numeric annotation
targets. Lock each session's records before scoring or revealing mappings.

Separate sessions by at least 48 hours. Hide session-one marks and results in
session two. Do not look back at old overlays/preferences between sessions;
record accidental exposure. Label masking reduces cues but cannot erase the
original review or the present protocol's exposed inventory. If the same
person prepares and measures the assets, declare that additional limitation.
The 48-hour interval, session cap, zoom choices, and uncertainty floor are
**operational choices, not empirically validated scientific thresholds**.

With only the original reviewer, conclusions are restricted to that person's
exploratory geometric alignment and within-reviewer repeatability under
disclosed prior exposure. Do not report inter-rater agreement, consensus, or
independent confirmation. If a second reviewer becomes available, identify
them and freeze a dated extension before their first view; report their
measurements separately. Do not fabricate a three-reviewer design, average
away disagreement, or treat two sessions as two independent reviewers.

## 7. Prespecified calculations and outcome rules

Only after both sessions are locked, join measurements to the frozen matrix.
Let q(c,e,k) be annotation eye e for case c under coordinate convention k
(raw or minus-one); let m(g,e,s) and r(g,e,s) be measured center and radius
for group g, eye e, session s. Compute Euclidean distance d=||m−q|| in original
pixels, and the two-eye mean residual R=(d_left+d_right)/2. Also report R
divided by the case's raw annotated inter-eye distance as a descriptive scale
normalization; it does not drive decisions. Preserve signed x/y residuals.

For each eye, distance bounds are [max(0,d−r), d+r]. Average the two lower
bounds and the two upper bounds to obtain [L,U] for a group/case/session.
These are conservative geometric bounds conditional on subjective radii,
not statistical confidence intervals and not uncertainty estimates for the
SDK's coordinates. Report session-to-session center displacement for each eye,
the change in R, and whether each pair of stated uncertainty disks overlaps.
For a conservative across-session interval use
`L*=min(L_session1,L_session2)` and `U*=max(U_session1,U_session2)`.
Do not average sessions to remove disagreement or replace an inconvenient mark.

Use the following rules in order:

1. Identical group/hash within a case is an **exact pixel tie**. Reuse the same
   measurements and residuals for all its labels, regardless of numerical rank.
2. If either eye in either session is unmeasurable for any distinct group in a
   case, mark that case `inconclusive_missing_landmark`. Retain available
   descriptive residuals but make no complete-case winner claim. Do not remove
   the difficult candidate, impute a mark, or acquire replacement frames.
3. For measurable cases, a group g has a robust geometric advantage over h only
   when `U*(g)+1.0 < L*(h)` for **both** fixed coordinate conventions. The 1.0
   original-pixel margin is an operational resolution choice, not a validated
   scientific threshold. Equality does not pass. Record every unordered pair
   once (within-case distinct groups only), with direction or unresolved status.
4. A case has `robust_geometric_best_group` only if one group satisfies rule 3
   against every other distinct group. Report all labels attached to that group;
   an A1 or A5 outcome cannot distinguish its N/E or N−/E labels.
5. For remaining comparisons, separately flag nominal ordering reversals across
   sessions as `session_sensitive`, reversals across origin conventions as
   `coordinate_sensitive`, and nonoverlapping eye uncertainty disks across
   sessions as `repeatability_concern`. Preserve all applicable flags. Absent a
   robust best group, the case is inconclusive, with those flags as reasons.
   Otherwise label unresolved differences `operationally_unresolved`, never
   statistical equivalence. Pairwise unresolved relations need not be transitive;
   do not collapse them into invented equivalence groups.

Report all six case outcomes, full per-group/session residuals and missingness.
Keep each source video and the adjacent P triplet visible in reporting. No
global winner, minimum win count, significance claim, post-hoc threshold search,
or early stopping after favorable cases is permitted. A robust advantage can
coexist with repeatability concerns; show them rather than hiding them in a
single label. Every result, including unanimous geometric preference, retains
the scientific gate unchanged.

## 8. Pixel difference versus temporal difference

An identical oriented-pixel hash proves an exact retained-array tie within the
specified dimensions/type, not identical timestamps or annotation-generation
history. Unequal hashes prove different arrays; they do not identify the cause.
Native successful-read ordinals identify decoder output positions. Export
ordinals identify resampled output positions. Different native ordinals alone
do not supply original capture times, and export/native ordinal equality does
not establish a common frame. Candidate labels are hypotheses, not causes.

Use a separate causal-status field: `identical_pixels`,
`decoder_only_supported`, `temporal_difference_supported`, or `unresolved`.
Beyond an exact hash tie, default to `unresolved`. A decoder-only claim requires
an independent trace to the same source presentation frame/PTS and a controlled
comparison isolating decoder/color-conversion differences. A temporal claim
requires an independent source-frame/PTS trace demonstrating distinct temporal
inputs, including any export duplication/drop mapping. Cite exact trace
locators, timebase and ambiguity for either claim. Existing zero-MAE comparisons
can establish an array association, not missing historical timestamps.

This pilot contains no new decoding, frame search, optical flow, image-error
threshold to infer time, or controlled decoder comparison. Apparent motion or
color differences noticed by a reviewer are qualitative observations only.
Where suitable independent trace evidence is absent, report `unresolved` even
if geometric ranking is decisive. Any new causal experiment needs separate
authorization and a separate design; do not retrofit it into this pilot.

## 9. Failure and stopping conditions

Before future execution, stop on absent authorization, unavailable reviewer,
changed protocol hash without a documented amendment, provenance/manifest
mismatch, any missing or mismatched asset file/pixel hash, unexpected
dimensions/type/orientation, a changed historical artifact, non-training
case, case/association/group counts other than 6/24/16, or contradictory
candidate-to-group metadata. Do not regenerate, substitute, search nearby
frames, change decoders, or alter historical evidence to pass a check.

Stop the affected session on invalid coordinate conversion, transformed or
annotated input, lost records, software malfunction, or participant fatigue or
withdrawal. Preserve partial records and the reason. Exposure to hidden
annotation targets during marking makes that session invalid for the planned
masked comparison; disclose and require a separately documented restart
decision rather than quietly discarding or replacing it. Recognition of
previously reviewed faces alone is expected and must be disclosed, not falsely
treated as successful blinding. A session gap shorter than 48 hours or failure
to complete two valid sessions prevents the planned repeatability decision.

Unmeasurable eyes and unresolved differences are legitimate outcomes, not
grounds for trying other landmarks, frames, or thresholds. Finish other valid
frozen groups when feasible, report all missingness, and do not claim a complete
pilot where a prerequisite failed. All future outputs remain separate from
historical audits, responses and the recovery manifest.

## 10. Future result schema (empty; not execution evidence)

The following is a schema-shaped empty template, not a completed result.
Arrays remain empty in this design. A future authorized run writes a new private
file with explicit protocol and input hashes. Do not fill outcomes from old
human preferences, asset-only GO, or the existence of this document.

```json
{
  "schema": "msu-six-case-exploratory-alignment-v1",
  "execution_status": "not_executed",
  "authorization_reference": null,
  "protocol_path": "docs/MSU_CORRESPONDENCE_PILOT_PROTOCOL.md",
  "protocol_sha256": null,
  "baseline_head": "1ba31f3b794f9c471e352f99a0bab189c2b4cbcd",
  "input_fingerprints": [],
  "asset_reverification": [],
  "reviewers": [],
  "sessions": [],
  "landmark_measurements": [],
  "case_associations": [],
  "residuals": [],
  "pairwise_comparisons": [],
  "repeatability": [],
  "case_outcomes": [],
  "causal_assessments": [],
  "deviations": [],
  "stopping_events": [],
  "interpretation": null,
  "policy_state": "P4",
  "frozen_policy": null,
  "fidelity_status": "manual_review_pending",
  "experiment_ready": false
}
```

Required record fields when execution is separately authorized:

| Array | Required fields and constraints |
| --- | --- |
| input_fingerprints | Repository-relative path, SHA-256, hash kind (exact bytes or Git blob), role; include protocol, human pack, annotation evidence, response, private manifest, and software/version configuration. |
| asset_reverification | Group, relative path, expected/actual file SHA-256, expected/actual pixel SHA-256, width/height/channels/dtype, status; sixteen unique groups. |
| reviewers | Actual ID/name, participation status, historical-review role, prior exposure disclosure; proposed people cannot supply records. |
| sessions | Reviewer ID, session ID, UTC start/end, gap hours, opaque group ordering/mapping reference, randomization algorithm/seed, display and software settings, synthetic conversion check, exposure events, completion/lock status and record digest. |
| landmark_measurements | Unique (reviewer, session, group, image-side) key; pixel hash, opaque ID, x/y/radius or null, measurable flag, reason, zoom/viewport, timestamp. Maximum 64 eye records per two-session reviewer, including unmeasurable records. |
| case_associations | Exact case ID, source ID, original annotation index/raw row/line/file hash, original eye pair order, image-side permutation, exact candidate ID/label/domain/ordinal, group/hash; exactly the frozen 24 associations. |
| residuals | Reviewer/session/case/group/convention, signed per-eye residuals, per-eye distances and bounds, mean residual/bounds, normalized mean, availability/reason. Null numerical values for missing measurements. |
| pairwise_comparisons | Reviewer, case, unordered group pair, both conventions' bounds/margins, robust direction or null, exact-tie labels and unresolved/sensitivity flags. Never count duplicate-label comparisons as extra observations. |
| repeatability | Reviewer/group/eye, two session IDs, center displacement, disk-overlap flag; case/group residual change and interval envelope, or null with missingness reason. |
| case_outcomes | Six exact case IDs per reviewer, completion, outcome, best group or null, all attached labels, unresolved pair list, sensitivity/repeatability flags, limitations. |
| causal_assessments | Case/group pair, status enum from section 8, independent trace references/timebase or null, reason and unresolved alternatives. No ordinal-only causal assertion. |
| deviations / stopping_events | Timestamp, affected records, rule, reason, authorization/amendment reference or null, disposition; never overwrite original marks. |

## 11. Interpretation, outstanding gates, and later confirmation

Geometric agreement with published `.face` coordinates is **not proof of the
historical PittPatt input identity**. The generation decoder, stream/timebase,
rotation stage, SDK target/configuration and index-writing loop remain unknown.
The selected cases and reviewer have prior exposure; one subject cannot
represent the release. Exact ties, limited resolution, decoder differences,
unknown landmark semantics and subjective uncertainty can defeat discrimination.
Even a repeated unique geometric ranking cannot close these provenance gaps.

Geometry remains a separate gate: the historical review contains 14 cases with
at least one uncertain geometry response and one negative face-box case
(real Android client 008, annotation 298, chin truncation). That case overlaps
the uncertainty set through a different field; its face-box response is `no`.
This eye-only pilot does not repair boxes, validate endpoints or approve crops.
Those concerns require separate review and explicit fidelity approval.

Codec handling remains a separate gate: eight previously recorded codec-error
recordings remain unresolved under the policy gate. This pilot neither
investigates them nor certifies general decoding integrity; timestamp warnings
and absent captured diagnostics are not silently reclassified as clean EOF.
No test-cohort evidence is used to select, tune, or confirm a mapping here.

A later confirmation requires a separately authorized, prospectively frozen
protocol using **previously unreviewed official TRAINING subjects**, not merely
new rows of client 002 or the already reviewed clients. Establish eligibility
from the full historical review/exposure records before seeing candidate
images; no specific new subjects or images are selected here. Predefine source
and row sampling, candidate domains, exclusions, group deduplication, reviewer
exposure controls, sample size rationale, measurement definition, missingness
and decision rules before outcomes are available. Freeze thresholds without
tuning on confirmation data; pilot-derived choices must be disclosed as such.

An independent reviewer would strengthen a later check but must actually be
recruited and named before claiming independence. With only the original
reviewer, new-subject work can test that person's prospective reproducibility;
it still cannot establish inter-rater validity or independent human confirmation.
No pilot or confirmation preference can replace attributable original-author
annotation-generation evidence. P4 and the other scientific gates remain
unchanged until their separate requirements are met.
