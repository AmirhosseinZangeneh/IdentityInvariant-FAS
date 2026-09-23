# MSU frame-domain policy and human-fidelity gate

**Decision: P4 (unresolved); `experiment_ready=false`.** Neither successful
export nor a plausible overlay establishes the stream used to generate PittPatt
annotations. No canonical preprocessing change, annotation remapping, experiment,
training, raw-file change, or human approval is made by this task.

## Evidence domains and source findings

The immutable [raw protocol lock](audit/msu_protocol_lock.json) remains the
source of recording identity, partition and annotation provenance. This task
reuses the [index-semantics audit](MSU_INDEX_SEMANTICS.md), including its full
280-recording native observations, independent decoder diagnostics, source
analysis and original eight-crop review. Those audits were not rerun or rewritten.

Three different numbers must remain distinct:

| Domain | Meaning |
| --- | --- |
| Annotation | Original first column of `.face`; never rewritten |
| Native decoder | Zero-based ordinal of successful sequential reads in the recorded OpenCV/FFmpeg build |
| DecFrames export | Zero-based ordinal of the rounded-rate image2 output; ordinal `i` has filename number `i+1` |

The bundled README, `DecFrames.m`, its two scene entry scripts and bundled FFmpeg
documentation/binaries were inspected and fingerprinted in
[source evidence](audit/msu_index_source_evidence.json). All seven fingerprints
were revalidated locally. README does **not** explicitly document zero/one-based
native numbering or the PittPatt input stream/timebase. `DecFrames.m` reads
`fidx=M(:,1)` and names the eyes sidecar using `fidx(t)+1`; `t` selects an
annotation row. It does not seek a video frame using `i`, `i-1`, or `i+1`.
MATLAB row indexing does not establish video indexing. Its BMP export uses
`avg_frame_rate` rounded to two decimals and `-r`; it detects rotation 180 but
does not apply that flag in the export code. README's referenced `decode.m` is
absent. Same-numbered sidecars and images support a file association, **not proof
that PittPatt generated its annotations from those exported pixels**.

## Export counter calibration

The exact bundled Windows binary is fingerprinted, not replaced with a newer
FFmpeg. Original image2 export uses:

```text
ffmpeg.exe -i SOURCE -r ROUNDED_RATE STEM_%03d.bmp -progress pipe:1
```

The disk-free counter retains the image2/BMP encoder and default synchronization:

```text
ffmpeg.exe -i SOURCE -r ROUNDED_RATE -y -vcodec bmp -f image2 -updatefirst 1 NUL -progress pipe:1
```

No `-xerror`, synchronization override or orientation correction is added. The
counter still decodes/encodes; it avoids retaining every BMP. This is a Windows
implementation, not a cross-platform or cross-version equivalence claim.

The existing [eight-training-video calibration](audit/msu_policy_export_calibration.json)
matched actual numbered BMP counts, contiguous numbering from 1, final progress,
exit status and duplication/drop counters. It includes rotation 0/180, both
cameras, real access, all three attack media, internal gaps, problematic endpoints
and a codec-error recording. All its measured drop counters were zero.

To close that specific gap, a separate
[nonzero-drop calibration](audit/msu_policy_drop_calibration.json) used training
`attack_client022_android_SD_printed_photo_scene01.mp4`, the training audit's
only nonzero-drop recording. Actual BMP export and the counter both produced
**267 images, 3 duplicated frames, 1 dropped frame**, with contiguous numbering.
The eight-video artifact and completed cohort audits were preserved. This ninth
calibration is supplementary validation, performed after those audits; it does
not change the previously recorded policy state or pretend to predate test
observation. Only its annotation-0 BMP was retained, without adding a human
correspondence case or selecting a mapping.

Together these measured cases validate the relevant count/dup/drop behavior of
this counter/build. They do not prove pixel identity for arbitrary builds or
annotation correspondence. `validate_export_calibration()` checks measurements,
not only a stored `passed` flag, and rejects missing counts, incomplete progress,
nonzero exit status, mismatched counters or file counts. Matching counts may
still coexist with codec diagnostics.

Export configuration digest:
`92a81eefbe44dd485d3baa9955b1fa8f7f15f029865069e349051792774bd318`.
It covers the bundled executable hashes, rounded-rate rule, image2/BMP settings,
counter transport, orientation assumption and correspondence caveat. Per-source
reports retain rational and rounded rates. Canonical JSON hashes and identifiers
exclude absolute paths and timestamps. Runtime diagnostics are observations;
their exact text/order is not promised to be identical across builds/runs.

## Completed release availability audit

[Training](audit/msu_policy_train_export.json) and
[test](audit/msu_policy_test_export.json) records cover **all 280 locked videos**,
with unique source IDs and unchanged client/device/class/annotation provenance.
The [release summary](audit/msu_policy_export_release.json) links both by digest.
These are count-only export observations, not 77,953 approved crops.

| Official cohort | Videos | Annotation rows | Export ordinals | Duplications | Drops | Codec-diagnostic recordings |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 120 | 33,593 | 33,604 | 21 | 1 | 4 |
| test | 160 | 44,280 | 44,349 | 80 | 40 | 4 |
| total | 280 | 77,873 | 77,953 | 101 | 41 | 8 |

All annotations have an available export ordinal under this calibrated counting
model; no unknown availability or nonzero process exit was recorded. Fifty
recordings report duplication, eight report dropping. Every export count equals
its annotation maximum plus one. The 80 export positions without annotations
correspond to the existing 80 internal annotation gaps across 12 recordings.
Original maxima are <=299 for 185 recordings, 300 for 93, and >300 for two
(301 and 310). None of these observations permits renumbering or filling gaps.

In contrast, the reused native audit observed **277 unavailable annotated
indices across 44 recordings**. Its stopping evidence distinguishes 212
independent decodes consistent with clean EOF, 60 completed decodes with mux
timestamp diagnostics, and eight codec-error recordings. A failed read alone
does not establish EOF. Export resampling can duplicate/drop decoded frames;
export availability cannot repair native correspondence by a universal offset.

## Eight codec-error recordings

All eight are laptop attacks. Counts below are observations, not a content or
fidelity certificate. Original-style export returns zero for all eight despite
the recorded ProRes/decoding errors. Absence of captured native errors for 005
and 007 does not certify their pixels as clean.

| Client | Cohort | Medium | Native reads | Export count | Captured error evidence |
| --- | --- | --- | ---: | ---: | --- |
| 005 | train | iPhone video | 301 | 301 | bundled export |
| 007 | train | iPad video | 301 | 301 | bundled export |
| 008 | train | iPad video | 127 | 300 | native and bundled export |
| 023 | test | iPhone video | 247 | 301 | native and bundled export |
| 028 | test | printed photo | 205 | 205 | native and bundled export |
| 049 | test | printed photo | 213 | 213 | native and bundled export |
| 051 | test | printed photo | 211 | 211 | native and bundled export |
| 053 | train | iPad video | 301 | 301 | native and bundled export |

The release artifact preserves full canonical IDs and diagnostics, including the
prior strict bundled checks separately. All eight remain fail-closed pending
documented codec resolution. No automatic exclusion, alternate-decoder approval
or recovery of apparently missing native frames is authorized by these counts.

## Training-only policy evidence and human review

The [decision](audit/msu_policy_decision.json) consumes bundled/source evidence
and the official **training** audit only. It was recorded before the new test
export observations; the test audit stores that decision's digest. Policy
generation rejects test-cohort input. Existing test observations were known from
the previous index audit, but are not policy-selection inputs. The decision was
not altered after viewing new test availability. Since no policy was frozen,
test results are descriptive/exploratory checks, **not validation of a frozen
policy** and not model-selection evidence.

The [training cases](audit/msu_policy_training_cases.json) retain eight sources:
real Android 002/003/008, real laptop 002, Android 002 iPad/iPhone/print attacks,
and laptop 008 iPad attack. Seven original annotation indices per source cover
early, neighboring middle and endpoint positions. Rotations 0 and 180, gaps,
ordinary/problematic endpoints and a codec error are represented. Each of 56
cases compares native `i`, `i-1`, `i+1`, and export ordinal `i` without changing
the original annotation index. Of 224 candidates, **184 are available and 40
are explicitly unavailable**; no substitute frame is used.

The [human pack](audit/msu_policy_human_pack.json) preserves source metadata,
original raw annotation/box/eyes, each candidate's domain/index/identity,
rotation assumption, export provenance, overlay path/hash and availability.
It also includes the eight historical crops and their full-frame overlays:
**64 review forms** in total. The [review template](audit/msu_policy_review_template.json)
has null reviewer and decision fields. Automated observations are never human
approval. Overlay coordinates use rounded raw values, explicit metadata rotation
and a half-open box assumption, all still unverified.

Open the self-contained local page:

```text
data_processed/MSU-MFSD/policy-gate/human-review-v1/review.html
```

On the audited workstation:
`C:\Users\NOVIN\Desktop\IdentityInvariant-FAS\data_processed\MSU-MFSD\policy-gate\human-review-v1\review.html`.
Use the case selector and full-resolution overlay links. Review upright
orientation, eyes, face bounds, temporal/source correspondence and half-open
endpoints; select uncertain/indistinguishable when appropriate. Enter reviewer
identity and comments, then download the response JSON before closing. The page
has no network submission or automatic approval; downloaded responses also keep
readiness false until separately assessed. Licensed BMPs/PNGs and the page remain
under ignored `data_processed/`, not in Git. Only metadata/templates are proposed
for version control. Visual similarity of neighboring frames is not a temporal
mapping determination.

## APIs and validation without repeating decoding

Evidence-only APIs are in `identity_invariant_fas.data.msu_policy_evidence`:

- `calibrate_counter(root, lock, cases, fresh_output)` compares actual BMP export
  with the counter using explicit training video IDs/original annotation indices.
- `count_cohort(root, lock, calibration, partition)` observes one official
  cohort; test is marked ineligible for selection.
- `unresolved_policy(protocol, training_audit)` produces P4, never approval.
- `write_human_pack(root, lock, cases, calibration, calibration_root,
  historical_review, historical_root, fresh_output)` produces review assets.
- `validate_local_evidence(...)` checks the existing lock, bundled files,
  annotations, calibration, codec classifications and local review hashes without
  decoding/exporting again. Focused tests separately check cohort scope, digest
  links, training/test separation and unavailable candidates without licensed data.

Run the read-only local validation with the repository installed in the Python
environment; all paths are workspace-relative:

```python
from identity_invariant_fas.data.msu_policy_evidence import validate_local_evidence

report = validate_local_evidence(
    "datasets/MSU", "docs/audit/msu_protocol_lock.json", "docs/audit",
    "data_processed/MSU-MFSD/policy-gate/human-review-v1",
    "data_processed/MSU-MFSD/policy-gate/calibration-v2",
    "data_processed/MSU-MFSD/policy-gate/drop-calibration-v1",
)
```

[Recorded local validation](audit/msu_policy_validation.json) verifies seven
bundled files, all 280 locked entries/77,873 annotation rows, nine training
calibration sources, eight codec classifications and 64 review cases. This
validation is reproducible without rerunning release decoding; input digests
bind it to the retained artifacts. A successful integrity check does not grant
scientific fidelity approval.

## Decision and exact remaining experiment gate

P1 means native correspondence established; P2 means DecFrames export
correspondence established; P3 means another explicit domain established; P4
means unresolved/multiple plausible domains. **P4 is the strongest justified
state here.** The export-domain candidate has stronger availability/file-name
evidence, but its annotation-generation provenance remains missing. No speculative
migration is designed or applied. Canonical preprocessing stays fail-closed and
`fidelity_status=manual_review_pending`, `experiment_ready=false`.

Before the first controlled MSU experiment, independently establish annotation
correspondence from PittPatt/source-generation evidence plus training-only
controlled review; record actual human geometry and temporal review; resolve and
lock handling of all codec-error recordings; then freeze an evidence-supported
frame-domain policy that retains original annotation numbers separately from any
explicit processed-frame index. Validate complete selected-frame/crop coverage
and lock its preprocessing digest. Define development/model selection using only
official training clients before using the official test cohort for final
evaluation. All these conditions are required; availability alone satisfies none
of the missing approvals. Sessions remain unverified/absent and frames remain
members of their canonical source-video groups.
