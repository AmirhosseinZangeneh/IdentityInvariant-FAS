# MSU annotation-to-decoder indexing evidence

**Decision: category 4 — source/release/native-decoder correspondence remains
unresolved.** No mapping or preprocessing correction is adopted. Original
annotation indices, raw files, protocol lock, sampling, training code and
scientific results are unchanged. `experiment_ready` remains false and fidelity
review remains pending. This document separates annotation observations, decoder
observations, bundled-source evidence and visual evidence; none uses PAD outcomes.

## Source evidence

The [source evidence artifact](audit/msu_index_source_evidence.json) fingerprints
the bundled README, all three MATLAB files, FFmpeg README and executable tools.
The current canonical protocol/preprocessing code and documentation, and the
repository's `.face` parsing/decoding call sites, were also inspected.

The README (lines 103–106) calls the first field “Frame index”, identifies
PittPatt 5.0.2 and describes coordinates relative to rotated frames. It **does
not explicitly say zero-based or one-based**, identify the annotation-generation
decoder/timebase or define box endpoint inclusivity. Every observed annotation
file starts at index 0; that observation alone does not establish its frame domain.

`DecFrames.m` establishes the following directly:

- Lines 122–128 import the `.face` matrix, assign `fidx=M(:,1)`, and write an eye
  sidecar named with `fidx(t)+1`. The variable `t` iterates annotation rows;
  `eyes(t,...)` selects the same row's coordinates. The filename uses the stored
  annotation number, not the row number. Internal gaps therefore remain gaps.
- There is **no annotation-index frame access**, `index-1` operation, time seek,
  or MATLAB video-frame array indexing. The `+1` transforms the sidecar filename
  number; it does not seek to native decoder frame `i+1`.
- Lines 91–105 read `avg_frame_rate` via FFprobe and run FFmpeg with `-r` rounded
  to two decimals for Android/laptop exports, writing `<stem>_%03d.bmp`.
  FFmpeg performs stream decoding/export; MATLAB does not iterate decoded frames.
- Lines 81–88 detect `TAG:rotate=180` and print a notice. The `rotated` variable
  does not drive any rotation operation later in this file. The README references
  `decode.m`, which is absent from this local release.
- Existing directories containing at least 100 BMPs are reused. The sidecar loop
  does not check that each corresponding BMP exists.

`DecFrames_real_scene01.m` and `DecFrames_attack_scene01.m` call this helper for
both official subject lists. Neither supplies additional indexing semantics.
Bundled FFmpeg is `N-50911-g9efcfbe` (2013 build). Its image2 help describes
`start_number`, and the controlled exports actually begin at `_001.bmp` without
an explicit start-number argument. Thus the **source sidecar-to-export filename
association is annotation i → exported file i+1**. A zero-based exported-image
ordinal is supported; a zero-based native decoder position is not established
by that association or by MATLAB's one-based array indexing.

## Release-wide decoder observations

The [boundary audit](audit/msu_index_boundaries.json) covers all **280 canonical
recordings**, both devices, both partitions and every class/attack type. A timing
sample of the client 002 real Android/laptop videos took approximately 0.142 /
0.447 seconds for 300 / 239 sequential reads, making full coverage practical.
Timings are explanatory estimates, not part of deterministic evidence payloads.

OpenCV 5.0.0 with FFmpeg, AUTO=0, read sequentially from position zero to the first
unsuccessful read: **77,671 successful reads**. All 280 were attempted; this does
not claim that every stream decoded cleanly. A repeat diagnostic pass preserved
the same per-recording counts. The original observations were retained and then
enriched, rather than discarded or reinterpreted. The complete annotation
inventory remains **77,873 rows**, zero malformed rows or duplicates, 80 internal
gaps in 12 recordings. Each row of the boundary audit preserves the canonical
source ID, client, partition, device, class, attack/presentation metadata, source
and annotation provenance, annotation range/count, availability differences,
reported stream metadata and successful sequential read range.

Annotation maxima are distributed as follows; the artifact contains the exact
histogram for every maximum value:

| Annotation maximum | Recordings |
| --- | ---: |
| 190–298 | 95 |
| 299 | 90 |
| 300 | 93 |
| 301 | 1 |
| 310 | 1 |

The maxima above 300 belong to test laptop attacks: client 023 iPad (`max=310`,
311 native reads) and client 026 iPhone (`max=301`, 302 reads). Their endpoint
availability is observed, not inferred from the annotation counts.

**44 recordings have 277 annotated indices unavailable under candidate A
(annotation i → native i).** These numbers describe availability, not a mapping
correction or a universal offset:

| Breakdown | Videos audited | Videos with unavailable annotations | Unavailable indices |
| --- | ---: | ---: | ---: |
| Train | 120 | 17 | 191 |
| Test | 160 | 27 | 86 |
| Android | 140 | 39 | 51 |
| Laptop | 140 | 5 | 226 |
| Bona fide | 70 | 12 | 12 |
| Attack | 210 | 32 | 265 |
| iPad video attack | 70 | 9 | 182 |
| iPhone video attack | 70 | 6 | 54 |
| Printed-photo attack | 70 | 17 | 29 |

### Termination and codec diagnostics

A failed OpenCV read is **never labeled EOF by itself**. In-process stderr fd
redirection proved unreliable for this Windows OpenCV build (different C runtime);
empty captured messages do not establish error-free decoding. The audit therefore
adds an independent bundled FFmpeg pass for all 280 sources using
`-v error -xerror -map 0:v:0 -vsync 0 -f null - -progress pipe:1`.
This writes no images, disables output frame duplication, captures process stderr,
exit code and final progress, and compares final count with the native observation.
All 68 sources with independent diagnostics were additionally rechecked in isolated
OpenCV subprocesses so their native codec messages could be attributed correctly.

| Stop assessment | Recordings | Interpretation |
| --- | ---: | --- |
| Consistent with clean EOF in independent decode | 212 | Successful independent termination, equal counts, no captured error messages |
| Completed with mux timestamp diagnostics | 60 | Equal counts and completed process, but non-monotonic DTS messages from null output; not called unqualified clean EOF |
| Codec-error evidence | 8 | ProRes/decoding errors in one or both decoder paths; completion/counts alone cannot certify clean data |

The eight codec-error records are laptop attacks: client 005 iPhone, 007 iPad,
008 iPad, 023 iPhone, 028 print, 049 print, 051 print, and 053 iPad. Five strict
bundled decodes exited nonzero. Some native decodes continued despite codec
messages; no conclusion about physical corruption versus decoder-version behavior
is manufactured.

Two native shortfalls have codec-error evidence: training client 008 iPad stops
after 127 reads versus a reported 300 frames (173 unavailable annotations); test
client 023 iPhone stops after 247 versus 301 (49 unavailable annotations, after
accounting for gaps). These account for **222** unavailable indices. The other
**55** occur in 42 recordings: 48 in 37 clean-EOF-consistent cases and 7 in five
completed timestamp-warning cases. Codec failure and annotation endpoint mismatch
are therefore separate phenomena. No indices are dropped or remapped.

## Controlled bundled-export correspondence

[Bundled-export evidence](audit/msu_index_bundled_export.json) reproduces the
actual `DecFrames.m` image export for the four original training review sources:
client 002 real Android, client 003 real Android, client 002 real laptop, and
client 002 Android iPad attack. Only output directories were redirected outside
raw data. The exact relative argument vectors, rounded rates, tool hashes,
export counts, filename endpoints and selected BMP hashes are retained.

Both real Android exports contain **301 BMPs from 300 native frames** and FFmpeg
reports `dup=1`. The laptop export has 239 BMPs / 239 native frames; the iPad
attack has 300 / 300. All supplied annotation sidecar numbers have corresponding
BMPs in these four reproduced exports. This reproduces export behavior, **not
the missing PittPatt annotation-generation process**.

Comparisons of exported BMP pixels with nearby native frames provide stronger
temporal evidence than availability alone:

- Client 002 Android: exported ordinal 0/1/2 exactly matches native 0/1/2;
  exported ordinal 150 exactly matches native 149; ordinal 300 matches native 299.
- Client 003 Android: ordinals 0/1 match native 0/1; ordinal 2 matches native 1;
  ordinal 150 matches native 149; ordinal 300 matches native 299.
- Client 002 iPad attack: all five inspected exported ordinals match native i
  exactly. Laptop comparisons favor native i at all five inspected positions,
  with nonzero pixel differences between decoder versions (about 0.15 mean
  absolute 8-bit channel difference).

These are full-frame decoder-pixel comparisons, not PAD metrics or geometric
fidelity scores. A universal i−1 would break early exact export matches; a
universal i would miss later duplicated-export correspondence. Neither result
licenses a piecewise mapping for the annotation files without generation evidence.

## Candidate visual review and the existing eight crops

[Explicit cases](audit/msu_index_review_cases.json) cover eight videos: the four
above, training client 002 Android print, training client 008 Android real with
gaps, and the two test laptop sources with maxima 310 and 301. No test PAD outcome
was inspected. For each, seven sorted annotated positions were selected: first
two, three around the annotation-list midpoint, and last two. These are evidence
cases, not a change to the production sampler.

[Candidate metadata](audit/msu_index_correspondence.json) records 56 annotation
indices × mappings A (i), B (i−1), C (i+1): **168 candidates, 144 available**.
The other 24 remain explicitly unavailable, including negative candidates; no
clamping or substitution occurs. Every case retains original annotation index,
candidate decoder index, raw box/eyes/line, source/client/partition/device/class,
rotation, overlay path/hash and mapping label. There are 56 three-column sheets
and 144 full-resolution overlays. All licensed images/BMPs remain ignored under
`data_processed/MSU-MFSD/index-evidence/`.

Open `data_processed/MSU-MFSD/index-evidence/candidates-v1/review.html` locally.
It also presents the **existing eight crops** and their original full-frame
overlays. The [eight-crop review record](audit/msu_index_existing_crop_review.json)
preserves the historical manifest hash, crop/frame metadata and five pending
checks per crop: upright orientation, eyes, face box, source-frame correspondence
and half-open endpoints. Reviewer is null; no approval is recorded or implied.

The [visual notes](audit/msu_index_visual_notes.json) identify four candidate
sheets viewed by the assistant: real Android clients 002/003 at annotation 150,
real laptop client 002 at 119, and Android print client 002 at 231. Interior
neighbors look very similar and their overlays broadly cover face/eyes; this
display-scale observation does not distinguish temporal offset. At the print
endpoint only B is available, which is not a valid reason to adopt B. These notes
are qualitative assistant observations, **not human geometry approval**.

## APIs, reproducibility and remaining gate

The new `data.msu_index_evidence` module is evidence-only. Existing preprocessing
and its v2 schema are unchanged. Future reruns must use fresh output locations:

```powershell
python -m identity_invariant_fas.data.msu_index_evidence --root datasets/MSU --output manifests/generated/new_boundaries.json
python -m identity_invariant_fas.data.msu_index_evidence --root datasets/MSU --enrich-existing manifests/generated/new_boundaries.json --output manifests/generated/new_diagnostics.json
python -m identity_invariant_fas.data.msu_index_evidence --root datasets/MSU --cases docs/audit/msu_index_review_cases.json --output data_processed/MSU-MFSD/new-index-review
```

`boundary_audit(root, lock, video_ids=None)` supports explicit partial scope and
never labels it release-wide. `enrich_stop_evidence(root, lock, prior_report)`
preserves observations and checks source metadata against the raw lock.
`write_review(root, lock, cases, output)` produces candidate JSON/PNGs.
`audit_bundled_export(root, lock, video_ids, output)` reproduces original BMP export
for explicit IDs; it does not apply a correction. Outputs contain relative paths,
no wall-clock timestamps, and software/source hashes. Runtime frame/progress and
codec diagnostics are observations, not a cross-build bit-exactness guarantee.

Before the first controlled MSU experiment: obtain or reconstruct with independent
evidence the PittPatt input stream/export/timebase and annotation numbering
convention; investigate codec-error recordings without silently omitting them;
complete and record human geometry/correspondence review; then propose any
versioned mapping policy preserving both original annotation number and explicit
decoder index. Validate complete selected coverage under an accepted policy and
define model selection using official training clients only. None of those
approvals is supplied by this audit. Sessions remain absent and frames remain
members of their original source-video groups.
