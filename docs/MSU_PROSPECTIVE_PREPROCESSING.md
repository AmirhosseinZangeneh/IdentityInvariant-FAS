# MSU prospective publication preprocessing

**Runtime update (2026-09-28): [Amendment 002](PUBLICATION_EXPERIMENT_MATRIX_AMENDMENT_002.md)
supersedes the OpenCV 5.0.0.93 pin and its engine-variable requirement below with
opencv-python 4.11.0.86 / native OpenCV DNN, CPU, one thread, OpenCL disabled.
Python/NumPy/Pillow, model, detector parameters and all scientific geometry remain
unchanged. Historical runtime-attempt sections below are retained as provenance.**

Date: 2026-09-27. Reviewed repository baseline:
`534d7d91655d5fafe1b5ce2d862164545033c785`.
**MSU_PROSPECTIVE_PREPROCESSING_SELECTED = true.** This selects a scientific
design, not a qualified implementation or permission to process/train.
Technical qualification: **not executed; publication processing blocked**.
Authority for the matrix change is [Amendment 001](PUBLICATION_EXPERIMENT_MATRIX_AMENDMENT_001.md).

## 1. Decision and historical boundary

Path A reconstructs preprocessing using historical `.face` indices and eye
geometry. Its blockers remain: unknown PittPatt input stream/index convention;
native/export endpoint and resampling discrepancies; unresolved orientation,
coordinate and crop correspondence; and eight recordings with codec diagnostics.
Manual correspondence evidence established neither historical input identity nor
an approved geometry policy. Successful decoding alone cannot clear these gates.
The [historical gate](MSU_POLICY_GATE.md), [provenance audit](MSU_PITTPATT_PROVENANCE.md),
human reviews and correspondence/timing documents remain unchanged.

Path B is selected for publication: original release video -> complete sequential
decode -> uniform decoded-position sampling -> independent face detection ->
unaligned crop -> RGB 160x160 input. This is defensible because client provenance
and official cohorts do not depend on reconstructing PittPatt pixels, and every
arm/cohort receives the same outcome-independent preprocessing. Detector/crop
bias remains a limitation, not proof of identity neutrality. This policy makes
no historical preprocessing-equivalence claim and may fail technical qualification.

Historical state remains exactly `policy_state=P4`, `frozen_policy=null`,
`fidelity_status=manual_review_pending`, `experiment_ready=false`. Do not change
these fields or reinterpret an old processed manifest as approved. A future
Path-B-specific qualification record must gate the new path independently;
historical readers/runners remain blocked. There is no timestamp/pilot dependency.

## 2. Source-only protocol binding

Use only the 280 source videos covered by the committed
[protocol lock](audit/msu_protocol_lock.json) and [client protocol](MSU_PROTOCOL.md):
120 videos from 15 official training clients, 160 from 20 official test clients.
Preserve canonical client ID, official partition, PAD label, `MSU-MFSD:` video
ID, relative source path, capture camera, attack medium and acquisition slot.
Never move a video/client between cohorts or infer sessions from `scene01`.

The existing live protocol loader checks annotations; do not use it as a Path B
annotation prerequisite. The next implementation needs a small source-only
validator. First verify the retained normalized recording catalog against the
lock's `normalized_metadata_sha256` using its existing canonical serialization,
then project its video metadata. Annotation fields in that catalog are opaque
historical provenance, never geometry/frame-selection inputs. If that catalog
is unavailable or mismatched, stop; do not invent a replacement inventory.
The retained `manifests/generated/msu_recordings.json` was checked read-only in
this task: 280 records, canonical SHA-256
`cfba71ee3ffc798ea1d21a3a859dfa29f0c83210316177850dd8de562454b838`, matching the
committed lock. This check did not read raw media or annotations.
Verify the README and official list fingerprints, exact 35-client sets and eight
slots/client, paths, media sizes, uniqueness and absence of physical aliases.
Freeze the projected catalog digest and parent lock digest before decoding.

The original lock did not hash video contents. Record each video's full SHA-256
before/after its processing; reject drift. These new hashes bind the prospective
source bytes, not retrospectively prove byte identity with the historical audit.
Require all source hashes before publication execution and check byte-identical
video aliases across distinct IDs/cohorts; any such conflict stops qualification.

`.face` files remain preserved provenance/history only. Path B does not read
them, require their presence, select their indices or use their coordinates.
An opaque parent-lock digest is not a scientific dependency on annotation
contents. No raw annotation or historical artifact is deleted or rewritten.

## 3. Decoder and artifact policy

Freeze **FFmpeg/ffprobe 8.1.2, Gyan Windows x64 static essentials ZIP build**:
[versioned archive](https://www.gyan.dev/ffmpeg/builds/packages/ffmpeg-8.1.2-essentials_build.zip).
Published [archive SHA-256](https://www.gyan.dev/ffmpeg/builds/packages/ffmpeg-8.1.2-essentials_build.zip.sha256):
`db580001caa24ac104c8cb856cd113a87b0a443f7bdf47d8c12b1d740584a2ec`.
This is an artifact selection, not a claim it has been installed or tested here.
Before any MSU decode, verify the archive, record extracted executable hashes,
version/build configuration and library versions in a write-once environment
lock, and retain the archive. Use absolute executable paths, never PATH lookup
or a floating download. Any mismatch stops; another build requires an amendment.

Why this route: standalone exit status and complete diagnostics allow a clearer
failure contract than treating an OpenCV failed read as EOF. No modern standalone
FFmpeg/ffprobe was found on PATH. The historical bundled 2013 executable is not
selected. Installed OpenCV is `opencv-python==5.0.0.93` with bundled avcodec
61.19.100 / avformat 61.7.100; its native reads do not establish clean termination
or equivalence to another decoder. No new binary was run in this design task.

Required execution semantics (implementation must test these before media use):

- Exactly one video stream; reject ambiguity/attached-picture streams. Native
  software `h264` or `prores` decoder according to stream codec, CPU only,
  one decoder thread; unsupported codec fails. No hardware acceleration.
- Decode from start to natural EOF; no seek, time trim, frame limit, frame-rate
  override, interpolation, frame dropping or duplication. Use
  `-fps_mode passthrough`, `-noautorotate`, `-err_detect explode`, `-xerror`,
  warning-level captured stderr and no progress/stats text. Any warning/error
  diagnostic, nonzero exit or incomplete pipe output fails, even if frames exist.
  Never use discard-corrupt/ignore-error recovery options.
- Frame payload is lossless P6 PPM (`image2pipe`, `ppm`, `rgb24`), with parsed
  per-frame dimensions and complete bytes; no JPEG. Use one filter/output
  thread. The pinned build performs native color conversion to RGB24 using
  source metadata and its fixed defaults when unspecified; record color range,
  matrix/primaries/transfer tags including absence. No per-file color overrides,
  enhancement, deinterlacing or tone mapping. Unsupported non-8-bit output fails.
- Require constant positive decoded dimensions, progressive frames and square
  sample aspect ratio. A full sequential ffprobe frame-metadata scan with the
  same software decoder/thread/error policy checks these properties and emitted
  frame count against the FFmpeg pass. Probe diagnostics/nonzero exit also fail.
  No metadata `nb_frames` or nominal FPS value substitutes for actual counts.
- Orientation comes only from container display metadata. Missing rotation
  means zero; supported pure rotations are multiples of 90 degrees, with
  absolute tolerance 1e-6 degree. Conflicting tags, reflection, skew, scaling or
  unsupported matrices fail. Interpret ffprobe display rotation as signed
  counterclockwise, normalize modulo 360, and rotate the RGB array by exact
  integer quarter-turns once, without interpolation. Record original metadata,
  applied rotation and oriented dimensions. Synthetic tests must establish
  sign/axis behavior. Do not infer rotation by `.face` fit or manual test review.

The [FFmpeg option documentation](https://ffmpeg.org/ffmpeg.html) defines the
passthrough, error and autorotation controls. Freezing versions does not by
itself prove pixel equivalence: hash oriented RGB pixels, pin software/CPU
execution settings, and require repeatability in training qualification below.
If the chosen build cannot implement this contract, stop before MSU processing;
do not silently translate it into a weaker EOF/error policy.

## 4. Exactly 30 sampled positions

Let N be the number of complete, valid frames emitted in presentation order by
the successful full decode. Buffer/cache that single decode in fresh private
staging until EOF and checks succeed. Ordinals are zero-based. For j=0..29 use:

`ordinal[j] = (j * (N - 1)) // 29`

Require N >= 30. This includes ordinal 0 and N-1 and yields 30 distinct positions;
use integer arithmetic. It is uniform in decoded position, not necessarily in
seconds for variable-frame-rate sources. A failed/truncated decode is not a
shorter valid sequence. Never pad/repeat frames, deduplicate naturally identical
pixels, use annotation positions, or search adjacent frames. Distinct positions
can legitimately contain identical pixel content. Apply the same rule to all
videos, cohorts, tuning/final arms and probe inputs.

## 5. Independent detector and crop

Select **OpenCV FaceDetectorYN / YuNet**, CPU DNN backend, through the installed
`opencv-python==5.0.0.93`; Python 3.12.10, NumPy 2.5.2, Pillow 12.3.0.
Pin package/binary fingerprints and CPU settings before qualification. Use
`cv2.setNumThreads(1)`, disable OpenCL and explicitly pass
`cv2.dnn.DNN_BACKEND_OPENCV` (3) / `cv2.dnn.DNN_TARGET_CPU` (0).
Freeze the classic native engine: launch a fresh process with
`OPENCV_FORCE_DNN_ENGINE=1` (`ENGINE_CLASSIC`) before importing cv2; reject
conflicting settings. ONNX Runtime is explicitly not used, nor are automatic
engine selection, GPU or target fallback. The installed build reports
`ONNX Runtime: NO`; engine forcing is specified by the
[OpenCV 5.0.0 importer](https://github.com/opencv/opencv/blob/5.0.0/modules/dnn/src/onnx/onnx_importer.cpp)
and [engine enum](https://github.com/opencv/opencv/blob/5.0.0/modules/dnn/include/opencv2/dnn/dnn.hpp).

Use only FP32 `face_detection_yunet_2023mar.onnx`, 232589 bytes, SHA-256
`8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4`, from the
[official model pointer](https://raw.githubusercontent.com/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx).
Acquire and verify the actual ONNX bytes, not the LFS pointer; retain model and
license locally. No executable detector has been qualified. On 2026-09-28 the
official ONNX bytes were read into memory, hash-verified and metadata-parsed
without inference or saving weights: input `input` is float32 NCHW
**[1,3,640,640]**. Freeze `FaceDetectorYN.create(..., input_size=(640,640), ...)`
and its input tensor at this size; no source-dependent `setInputSize` calls.
The [Zoo notes](https://github.com/opencv/opencv_zoo/blob/main/models/face_detection_yunet/README.md)
distinguish this fixed-shape model from the newer dynamic model. Keep 2023mar.

Rationale: the existing runtime exposes FaceDetectorYN; no committed independent
face detector/weights were found. MTCNN/facenet-pytorch, TensorFlow, ONNX Runtime,
InsightFace and RetinaFace packages were absent in the selected environment.
YuNet avoids introducing another inference framework. This is a minimal
dependency choice, not evidence that its crops outperform alternatives or work
on every attack. The [official OpenCV detector documentation](https://docs.opencv.org/5.0/javadoc/org/opencv/objdetect/FaceDetectorYN.html)
specifies score/NMS/top-K controls; all choices below are prospective operational
settings, not empirically validated scientific thresholds.

- Detect once per selected oriented frame, using a **640x640 letterboxed
  detector copy**. Preserve the original RGB raster for cropping. For its width
  W and height H, let M=max(W,H); compute resized dimensions with exact integer
  half-up rounding: `Wr=max(1,(1280*W+M)//(2*M))` and analogously Hr using H.
  Convert RGB to contiguous BGR uint8 and resize to (Wr,Hr) using
  `cv2.resize(..., interpolation=cv2.INTER_LINEAR)`; omit resizing if unchanged.
  Pad with BGR (0,0,0): left L=(640-Wr)//2, top T=(640-Hr)//2; right/bottom
  receive the remainder. No image stretching beyond integer rounding, tiling,
  augmentation, adaptive detector resolution or test-driven size selection.
  FaceDetectorYN's blob conversion uses float32 BGR, scale 1, zero mean and no
  channel swap; 640 is divisible by its stride-padding divisor 32, so there is
  no additional internal padding. See the
  [pinned detector implementation](https://github.com/opencv/opencv/blob/5.0.0/modules/objdetect/src/face_detect.cpp).
- Explicit score threshold 0.90, NMS IoU threshold 0.30, top-K 5000. Use the
  pinned implementation's post-NMS returned detections. Exactly one is required.
  Zero or multiple detections stop the video and study; never choose by label,
  face size, score preference, temporal tracking or visual/manual correction.
- Count returned detections before any padding/crop filtering; the exactly-one
  rule above is unchanged. Map its floating detector box (xd,yd,wd,hd) to the
  original oriented frame using `x=(xd-L)*W/Wr`, `y=(yd-T)*H/Hr`,
  `w=wd*W/Wr`, `h=hd*H/Hr`, evaluated in float64 without intermediate rounding
  or clipping. These are box-edge coordinates; no half-pixel offset is added.
  Record W/H, Wr/Hr, padding, detector box and mapped box in the manifest.
- No landmark alignment, eye rotation, mirroring or `.face` geometry. Ignore
  predicted landmarks for crop construction. Use the mapped original-frame box
  `(x,y,w,h)`; all values must be finite, w/h positive, and its unexpanded box
  must intersect the image. Expand by **10% of w on each horizontal side and
  10% of h on each vertical side**, in original-frame coordinates before any
  rounding/clipping or final Pillow resize. No square-padding step for the crop.
- Bounds are `[floor(x-.1w), ceil(x+1.1w))` horizontally and analogous y/h
  vertically, clipped to `[0,W)` / `[0,H)`. Record unclipped/clipped bounds;
  reject an empty crop. Coordinates refer to the oriented original raster.
- Slice RGB uint8 pixels; resize once to 160x160 using Pillow 12.3.0
  `Image.Resampling.BILINEAR`, default `reducing_gap=None`, no aspect-preserving
  padding. Save RGB lossless PNG and verify dimension/dtype/pixel hash on reload.
  Model input is ToTensor (RGB values /255) with no further geometric operation,
  augmentation or normalization fitting. This defines MSU's realization of the
  matrix's common 160x160 input; NUAA transforms are not amended.

## 6. Fail-closed handling and manifest

| Condition | Required action |
| --- | --- |
| Source/artifact/catalog drift; ambiguous metadata | Stop before publishing any successful manifest |
| Decoder/probe diagnostic, premature end, inconsistent counts/dimensions | Stop; retain logs and partial evidence, no success certification |
| N < 30 | Stop; no repeats, reduced count or replacement source |
| Detector exception, no face, multiple faces, nonfinite box | Stop; no retry with altered threshold/model/frame |
| Empty/invalid crop or wrong output shape/dtype | Stop; no manual repair |
| Write failure, reload/hash mismatch, interrupted run | Stop; retain partial outputs, never overwrite/relabel them complete |

Any required video failure blocks the main MSU study; do not exclude difficult
videos/clients, alter official cohorts or average over fewer frames. A future
policy revision needs a dated amendment and training-only rationale; test-side
failures cannot justify trying alternatives on those test images. Report any
prior test exposure if a later study is redesigned. No adaptive fallback chain.

Use a fresh ignored root, separate from historical outputs/private pilots, e.g.
`data_processed/MSU-MFSD/prospective-publication-v1/`. Reject existing destinations.
Keep licensed pixels private. A completed manifest requires all 280 videos and
8400 selected crops, including 1200 bona-fide test frames for M2. Qualification
outputs are explicitly partial technical evidence, never an experiment dataset.
All arms consume one frozen complete manifest, never independently recrop.

Record policy/document/implementation digests; environment/artifact lock; source
catalog and official-list hashes; full source hashes and labels/group metadata;
decoder/probe commands, logs, status and N; source/oriented dimensions, dtype,
color/orientation metadata; 30 ordinals and oriented RGB pixel hashes; detector
weights/settings/count/score/box; crop bounds, output path, encoded-file and
reloaded RGB pixel hashes; failure/completeness status. Pixel hashes use SHA-256
of contiguous row-major HxWx3 uint8 RGB bytes with shape/color separately bound
in the manifest digest. Never reuse historical BGR hash labels for RGB hashes.

## 7. Bounded training-only technical qualification (future authorization)

Before any media, implement/test only a separate Path B validator and bounded
processor, acquire the pinned artifacts, and freeze their local fingerprints.
Synthetic tests must cover error diagnostics despite exit zero, corrupt/short
streams, frame counts/endpoints, no duplication, rotation/dimension changes,
RGB/BGR and crop rounding, zero/multiple detections, failed writes and ignored
annotation files. Do not modify the historical annotation preprocessor.

Then qualify exactly these **eight official training videos**, in this order
(paths relative to `datasets/MSU`; video IDs add `MSU-MFSD:`):

1. `scene01/attack/attack_client005_laptop_SD_iphone_video_scene01.mov`
2. `scene01/attack/attack_client007_laptop_SD_ipad_video_scene01.mov`
3. `scene01/attack/attack_client008_laptop_SD_ipad_video_scene01.mov`
4. `scene01/attack/attack_client053_laptop_SD_ipad_video_scene01.mov`
5. `scene01/real/real_client002_android_SD_scene01.mp4`
6. `scene01/real/real_client003_android_SD_scene01.mp4`
7. `scene01/real/real_client002_laptop_SD_scene01.mov`
8. `scene01/attack/attack_client002_android_SD_printed_photo_scene01.mp4`

These cover both cameras, both PAD classes, all three attack media, documented
Android orientation cases and all four training codec cases. First qualify
decoding of 1-4; any failure stops before proceeding. Then apply the identical
full policy to all eight: at most 240 unique selected crop records. Repeat the
same eight-source run once into another fresh root and require identical N,
ordinals, oriented pixel hashes, detector outputs/bounds and crop pixel hashes.
This is repeatability checking, not an alternative-setting search.

One named technical reviewer checks the fixed sample slots j=0,14,29 for each
video (24 frame/crop pairs): correct upright orientation, the intended visible
face retained rather than background, and no gross truncation caused by the
crop rule. Record pass/fail reasons and manifest/log checks, not scientific
preferences or identity/PAD scores. Any failure blocks qualification; no manual
crop correction. All 240 records receive automated checks. Passing this small
set establishes feasibility, not a guarantee of all-video detector accuracy.
No annotation overlay, correspondence ranking or large human-review study.

## 8. Codec gate and execution boundary

The [historical eight-codec table](MSU_POLICY_GATE.md#eight-codec-error-recordings)
is risk evidence, not a result under the selected decoder. All eight prospective
statuses are **not tested**. All are laptop attacks:

| Cohort | Client / medium | Qualification stage |
| --- | --- | --- |
| Train | 005 / iphone_video; 007 / ipad_video; 008 / ipad_video; 053 / ipad_video | First four sources above |
| Test | 023 / iphone_video; 028 / printed_photo; 049 / printed_photo; 051 / printed_photo | Only after training qualification and final artifact/policy freeze |

Test paths use the same locked `scene01/attack/attack_clientID_laptop_SD_MEDIUM_scene01.mov`
grammar. A later separately authorized four-test-source check applies exactly
the same decode/detection policy mechanically. No test crop-quality preference,
PAD/probe feedback, neighboring-frame search or per-file decoder change. A
failure blocks processing; a success does not erase old diagnostics. Remaining
videos must also pass during a later authorized release-wide run. No success
or exclusion is inferred from historical counts, current versions or this plan.

Before a small validation run: acquire/hash the selected artifacts, verify the
retained source catalog, implement the separate contract and pass synthetic
tests, then obtain execution authorization. Before training: training technical
qualification, test mechanical conformance and full 280-video/8400-crop coverage,
independent Path B readiness checks and the matrix's training/probe adapters are
still required. None was executed here. No full pytest is needed for this prose.

## 9. Publication interpretation

Allowed after qualification/execution: results under this project-defined
preprocessing; unseen documented-client PAD generalization; and frozen-feature
identity decodability under this same pipeline and camera/source-video grouping.
Do not claim session separation or verified pixel-level human correspondence
beyond supported release metadata. A pretrained detector may itself affect
identity/attack representations; comparisons estimate intervention effects
conditional on this shared pipeline, not detector-independent invariance.
Disclose the external pretrained detector and its documented training provenance;
do not fine-tune it on MSU or claim verified pretraining-identity disjointness
without evidence. Encoder training-client separation is a distinct claim.

Not allowed: faithful original MSU/PittPatt preprocessing reproduction, proof of
annotation/frame correspondence, historical `.face` fidelity, or attributing
differences from literature solely to the model when preprocessing differs.
RQs, client cohorts, three-arm gradient interventions, tuning/test separation,
seeds, metrics and statistics remain those of the publication matrix.

## 10. Contract implementation status (2026-09-28)

`identity_invariant_fas.data.msu_publication_preprocessing` implements the separate
Path B contract; it does not import the historical annotation preprocessor or
protocol loader. Its source-only catalog validator reads the retained metadata,
README and official lists and stats source paths. Content hashing is a separate
explicit operation. No annotation file is opened or required.

The module provides pinned command builders, incremental strict P6 parsing,
decode/probe result validation, orientation and sampling, letterbox/remapping,
crop/PNG helpers, artifact/environment lock validation, exact qualification
scope, and deterministic manifest validation with write-once completion seals.
The synthetic tests use temporary binary/RGB fixtures only, prohibit network
and subprocess execution, and exercise the real hashing/geometry/PNG logic.
They require no production-version match and no FFmpeg/YuNet installation.
Validation on 2026-09-28: **88 focused synthetic tests passed; 548 full-suite
tests passed** using `env_cuda/Scripts/python.exe`. No real decode, detector
inference, qualification run or scientific training was executed.

Transport is deliberately separated from the contract: no ffmpeg/ffprobe
subprocess launcher or release-wide runner is provided here. The later bounded
qualification harness must capture complete stdout/stderr and exit status,
normalize ffprobe metadata into `DecodeResult`, and run `validate_decoded_frames`
on all PPM frames before releasing selected frames. Its orientation matrix input
is a normalized 3x3 Cartesian counterclockwise matrix, not ffprobe's raw fixed-point
display-matrix text; the harness must validate/convert that representation and
retain diagnostics. It must not assert completion on timeout, pipe failure or
truncated JSON. ffprobe has no ffmpeg `-xerror` switch: its exact command plus
strict stderr/exit/completion/frame-metadata validation supplies the probe gate.
No alternative decode or detector settings are permitted.

The detector factory is only for a fresh interpreter launched with
`detector_worker_environment`; it refuses pre-imported cv2 or a missing/conflicting
engine setting. It verifies runtime versions/model bytes before importing cv2,
then configures classic CPU inference and the frozen letterbox. No worker was
started. Production environment locks require executable build evidence,
runtime binary fingerprints, and executable membership in the approved archive;
supplied build evidence must be captured and verified during real qualification.

Manifests retain portable command references and complete source/group metadata,
bind catalog/snapshot/environment/policy/implementation digests, and recompute
geometry rather than trusting stored boxes. Scientific digests contain no wall
clock timestamps or absolute host paths. Qualification mode requires the ordered
eight sources and 240 crops; publication mode requires all 280 sources and 8400
crops. Neither mode sets `publication_ready=true`. A completion seal certifies
only validated output coverage, not reviewer approval or experiment readiness.
Readers require the seal, all PNG checks and no failure record. Partial writes
are preserved and cannot be silently retried into an existing destination.

**Real artifact qualification: NOT RUN. Training-video qualification: NOT RUN.
Publication processing: NOT READY.** Historical P4/readiness states and Amendment
001 are unchanged. The next separately authorized task is the bounded transport
integration/artifact qualification followed by the prescribed training-only checks;
passing synthetic contract tests does not demonstrate decoder/detector feasibility
on licensed recordings.

## 11. Runtime integration attempt (2026-09-28)

The separate runtime module and synthetic-only qualification script now provide
explicit executable verification, ffprobe adaptation, incremental PPM transport
with concurrent bounded stderr draining, source snapshot binding, and a fresh
Python interpreter for the existing YuNet factory. These are **unqualified**.
The frozen contract helpers and Amendment 001 have not changed.

The official 2023mar model matched 232589 bytes and the frozen SHA-256. In the
matching Python 3.12.10 / opencv-python 5.0.0.93 / NumPy 2.5.2 / Pillow 12.3.0
environment, the fresh child passed the factory's pre-import engine and model
checks but failed inside `FaceDetectorYN.create` with OpenCV's
`Input shape redefinition is not allowed` (`tgtName` is `input`). No inference
completed. This triggers the explicit stop condition; no alternate engine,
model, geometry, runtime version, or scientific policy was attempted.

FFmpeg acquisition was interrupted after that stop condition; no archive or
executable qualification is claimed. H.264/ProRes real subprocess qualification
and runtime repeatability remain unexecuted. Eight-video MSU qualification and
full publication preprocessing remain pending and blocked. No licensed MSU
media was accessed. Two historical tests that automatically accessed a locally
installed release are now explicitly skipped in the synthetic suite; historical
preprocessing implementations are unchanged.

Validation: 108 existing focused contract tests passed; 11 subprocess-fixture
tests passed; the artifact test skipped without explicit complete artifacts.
The full synthetic suite passed with 577 passed and 3 skipped. `git diff --check`
passed. These results do not qualify the real decoder or detector runtime.

For a later authorized reproduction, use the pinned environment and explicit
local artifacts (the output directory must be fresh and ignored):

```powershell
env_cuda/Scripts/python.exe scripts/qualify_msu_publication_runtime.py --archive <absolute-frozen-zip> --model <absolute-2023mar-onnx> --output <fresh-ignored-directory>
```

This command is a synthetic runtime qualification attempt, never an MSU runner.
It fails closed and does not resolve the observed detector blocker.

## 12. Synthetic runtime qualification completed (2026-09-30)

Amendment 002's isolated OpenCV 4.11 runtime resolves the earlier constructor
blocker. Retained successful YuNet inference/isolation evidence was verified
against the current model and runtime binary fingerprints without rerunning
inference. The main `env_cuda` environment remains unchanged.

The frozen FFmpeg ZIP matched its SHA-256; fresh extracted ffmpeg/ffprobe 8.1.2
executables were fingerprinted and their complete build/library output retained.
Generated H.264 and ProRes fixtures each produced 35 progressive square-SAR
96x64 frames, with clean diagnostics, matching sequential probe/decode counts
and identical repeat RGB hashes. A +90-degree display-matrix remux matched exact
quarter-turn RGB hashes. Invalid, absent and truncated inputs were rejected.
Synthetic pipe-stress, timeout, malformed PPM and diagnostic tests also passed.

Validation on resume: 123 focused tests passed (2 artifact tests deselected);
2 real-artifact tests passed in the isolated environment (11 fixtures deselected);
the full `env_cuda` suite passed with 581 passed and 4 skipped. The latter skips
are the two licensed-release checks and the two separately executed artifact
tests. `git diff --check` passed.

Local evidence is ignored under `outputs/msu-runtime/qualification-20260930/`.
Reproduce in the isolated runtime with a fresh output directory:

```powershell
$env:PYTHONPATH = (Resolve-Path src).Path
outputs/msu-runtime/opencv411-env/Scripts/python.exe scripts/qualify_msu_publication_runtime.py --archive outputs/msu-runtime/artifacts/ffmpeg-8.1.2-essentials_build.zip --model outputs/msu-runtime/artifacts/face_detection_yunet_2023mar.onnx --yunet-evidence outputs/msu-runtime/yunet411-qualified.json --output outputs/msu-runtime/fresh-qualification
```

Omit `--yunet-evidence` only when a new inference qualification is intended.
Eight-video training qualification still requires separate authorization and has
not run. Full publication preprocessing remains pending. No licensed MSU media
was accessed; runtime qualification is not MSU experiment readiness.

### Runtime identity enforcement and implementation provenance

Runtime entry points independently select the required executable role and
verify absolute paths against these approved identities from the frozen archive:

| Role | Bytes | SHA-256 |
| --- | ---: | --- |
| ffmpeg | 101897728 | `1326dde4c84ff1f96fe6b8916c5bed29e163e9b5dccf995f6f3db069d143ec5e` |
| ffprobe | 101692928 | `b49ccc7c6547b141ad5a2f6ec69cc04323d7133d7704d70b331b904c63eecb07` |

Both must actually report `8.1.2-essentials_build-www.gyan.dev`. Caller-created
evidence objects cannot supply acceptance criteria; swapped roles fail. The
runtime checks bytes before/after version execution and after successful media
execution. Path-based Windows process creation leaves an unavoidable interval
between hashing and opening the executable; the local artifact directory must
remain protected from concurrent modification. No scientific command changed.

Qualification records now fingerprint the contract, runtime, worker, qualification
script and the two executed package initializers. The portable summary also
binds the raw report hash. These implementation/provenance hashes are separate
from the scientific policy digest. Previous local evidence is preserved;
the fresh binding-enforced qualification has a sanitized summary at
`docs/audit/msu_publication_runtime_qualification.json` and raw logs under
`outputs/msu-runtime/qualification-binding-20260930/`.

Targeted fix validation: 22 focused runtime tests passed (3 artifact tests
deselected); 3 real-artifact tests passed (22 fixtures deselected); the full
suite passed with 592 passed and 5 skipped (two licensed-data checks and three
separately executed artifact tests). `git diff --check` passed. H.264, ProRes,
rotation, repeatability, failure paths and fresh YuNet inference/isolation all
passed under the enforced executable identities. No licensed media was accessed.

### Bounded orchestrator implementation status

The eight-source orchestrator is implemented in
`msu_publication_qualification.py`, with an explicit CLI at
`scripts/run_msu_eight_video_qualification.py`. It validates the full catalog as
metadata without accessing recording paths, snapshots only the four codec-gate
sources for Phase A, then snapshots the exact eight training sources for each of
two independent runs. Both decoder APIs share one transport core; selected decode
retains only the frozen 30 sample frames. Existing detector and crop helpers
produce the technical manifests, followed by exact repeatability comparison and
24 fixed review pairs with human review pending.

The CLI requires explicit artifact/metadata/output paths, an absolute
`--git-executable`, and `--expected-head`. It refuses uncommitted implementation
changes, a mismatched isolated runtime, unapproved artifacts, and unsafe or reused
output roots before recording access. Thus this implementation must be audited
and committed separately before a future authorized real run.

Synthetic orchestration tests passed. Real eight-video qualification is **NOT
RUN**; no licensed MSU recording was accessed during implementation. Full MSU
preprocessing remains unauthorized. The earlier artifact qualification report
describes its recorded implementation hashes; it is not evidence that this new
orchestrator has processed real media. Frozen scientific rules and Amendments
001/002 are unchanged.
