# MSU preprocessing and provenance

`identity_invariant_fas.data.msu_preprocessing` is the canonical preprocessing
API. It verifies the reviewed [raw protocol lock](audit/msu_protocol_lock.json)
before reading annotations or decoding. It never regenerates that lock from
processed outputs. This is deterministic, label-preserving representation work;
it fits no statistics, uses no PAD outcomes, selects no model hyperparameters,
and creates no development partition. Reading official test annotations is not
model selection. Future validation and model selection must use training clients.

**The first controlled experiment remains blocked.** A new decoder/annotation
endpoint discrepancy and manual geometry checks remain unresolved, as described
below. The complete release has not been cropped or declared geometrically valid.

## Migration and compatibility

The command `scripts/preprocess_msu.py` now delegates to `run_preprocessing` and
`load_msu_protocol`. The reviewed `read_msu_subjects` handles numeric list token
`02` as `002`, consistent with `DecFrames.m`. No second normalizer was introduced,
and arbitrary identifiers are not normalized. Synthetic integration tests cover
the two-digit lists and three-digit source filenames through actual PNG writing.

The legacy `data.msu.index_msu_videos` remains available with unchanged parsing
and a `FutureWarning`. It still has the documented padding limitation and uses
filename stems as groups; it is not a publication API. The preprocessing command
no longer uses it. Old script helper functions and its implicit JPEG/CSV output
are retired. Existing training interfaces, generic `FASSample` / `SampleRecord`,
models, checkpoints and result files are unchanged. No training adapter is added.

## Commands and API

Install the project's `video` extra. Run from the repository root; local evidence
was obtained with `env_cuda/Scripts/python.exe` (Python 3.12.10, OpenCV 5.0.0,
NumPy 2.5.2, FFmpeg backend). Always choose a **fresh** output directory outside
the raw root. Reusing a directory fails, even for a dry run.

Full annotation/index validation, **without decoding** (the default mode):

```powershell
python scripts/preprocess_msu.py --root datasets/MSU --mode plan --output-root data_processed/MSU-MFSD/preprocessing-plan-v1
```

Small orientation verification, exactly the four training sources used here:

```powershell
python scripts/preprocess_msu.py --root datasets/MSU --mode verify --output-root data_processed/MSU-MFSD/preprocessing-review-v1 --video-id MSU-MFSD:scene01/real/real_client002_android_SD_scene01.mp4 --video-id MSU-MFSD:scene01/real/real_client003_android_SD_scene01.mp4 --video-id MSU-MFSD:scene01/real/real_client002_laptop_SD_scene01.mov --video-id MSU-MFSD:scene01/attack/attack_client002_android_SD_ipad_video_scene01.mp4
```

`verify` selects only the first two annotated indices of each explicit training
video (at most six videos). This distinct selection policy is included in its
configuration digest. It writes crops, `manifest.json`, overlays under `review/`,
and `audit.json`. It does not grant visual approval.

Reproduce the separate endpoint check; it sequentially decodes the two explicit
sources through the first failed read and writes observations, not crops:

```powershell
python scripts/preprocess_msu.py --root datasets/MSU --mode boundary --output-root data_processed/MSU-MFSD/preprocessing-boundary-v1 --video-id MSU-MFSD:scene01/real/real_client002_android_SD_scene01.mp4 --video-id MSU-MFSD:scene01/real/real_client003_android_SD_scene01.mp4
```

Future extraction uses `--mode process --frames-per-video 30 --margin 0.0`;
optional `--output-size WIDTH HEIGHT` resizes. This command currently fails if a
selected annotation references an unavailable decoded frame. Do not circumvent
the endpoint discrepancy by dropping rows or tuning the sampler. No full
`process` run was performed for the local release in this task.

Python entry points:

```python
from identity_invariant_fas.data.msu_preprocessing import (
    PreprocessingConfig, run_preprocessing, read_processed_manifest,
    audit_decoder_boundaries,
)

audit = run_preprocessing(raw_root, lock_path, fresh_output,
                         PreprocessingConfig(frames_per_video=30), mode="plan")
# For a completed process/verify output:
frames = read_processed_manifest(manifest_path, raw_root, lock_path)
assert frames.processing_status == "complete"
assert frames.fidelity_status == "manual_review_pending"
assert not frames.experiment_ready
observations = [frame.probe_observation() for frame in frames]
```

The reader rechecks the raw lock, annotation bytes/line numbers/values, selected
indices against the exact algorithm, crop hashes, and decoded crop dimensions.
It does not prove crop-to-video geometry without a source decoder review.
`validate_processed_manifest(document, protocol, lock_digest)` is the lower-level
structural/provenance validator and requires an already validated raw protocol.

## Structured manifest and deterministic identity

`manifest.json` has schema `msu-processed-v2`, configuration and SHA-256 digest,
raw protocol-lock digest, exact per-video selected indices, and typed
`ProcessedMSUFrame` records. It is published only after complete selected-frame
coverage. Failed runs retain `audit.json` with partial written-frame metadata
and failure accounting, but no successful manifest. Partial output is not a
dataset for experiments.

The ambiguous manifest `status="complete"` is replaced by separate fields:
`processing_status="complete"`, `fidelity_status="manual_review_pending"`, and
`experiment_ready=false`. Completion describes extraction/provenance checks,
not fidelity approval. Both validators enforce these fields and return a
`ProcessedMSUManifest` sequence with the statuses retained alongside its frames.
Both accept `require_experiment_ready=True`, which currently rejects every
manifest: this schema has no supported approved state or approval mechanism.
A separate evidence/approval task is needed before adding one. Missing statuses,
forged approval/readiness, legacy status-only manifests, partial and failed
processing are rejected. Existing unreviewed v1 outputs are not silently upgraded.
Plan audits retain `status="index_validated_only"` with
`processing_status="not_processed"`; failed audits have
`processing_status="failed"`. Neither produces a successful processed manifest.

Each frame retains:

- `sample_id`, dataset, canonical client, official PAD partition and label;
- `source_video_id`, portable source filepath, and source byte size;
- integer zero-based source `frame_index`, capture device, attack type,
  presentation device, and `session_id=null`;
- annotation filepath, SHA-256, one-based physical line number, raw line and all
  eight annotation values (rectangle and both eyes);
- actual clipped integer crop box, raw/oriented frame dimensions, metadata
  rotation applied, and output dimensions;
- crop-relative filepath, crop-content SHA-256 and preprocessing config digest.

`source_video_id` is **exactly** `MSURecording.video_id`, including its
`MSU-MFSD:` namespace and full relative source path with extension. It never
becomes a filename stem or frame ID. The processed sample ID is
`<source_video_id>:frame:<original_index>:prep:<config_sha256>`.
Crop paths are `crops/<sha256(source_video_id UTF-8)>/<index:06d>.png` relative to
the output root. No absolute local roots or timestamps enter canonical payloads.
Repeated generation in a relocated synthetic tree is byte-identical.
`probe_observation()` uses the sample ID, canonical client, canonical video
group and a null session. More frames never create more independent groups.

The raw lock records media sizes, not full video-content hashes. Same-size raw
video edits are therefore not detected by that lock; it remains a documented
limitation. Crop hashes certify written crop bytes, not original media identity.

## Selection, annotations and crop policy

For sorted unique annotated indices `a` of length `n`, choose `k=min(n, requested)`.
If `k=1`, take `a[0]`; otherwise choose
`a[floor(i*(n-1)/(k-1))]` for integer `i=0..k-1`. Integer arithmetic avoids
floating-point rounding variability. Sampling is uniform over annotation
positions, not elapsed time. Sparse original indices stay sparse. The sampler
rejects duplicate/negative/noninteger indices, empty input and invalid counts;
it never pads or repeats frames when fewer than requested are available.

The parser requires a nonnegative integer index and exactly eight finite
coordinates. It accepts comma-separated or whitespace-separated rows. Rectangle
extent must be positive. UTF-8 errors, missing/extra fields, invalid geometry or
fractional indices fail. Blank lines are counted and ignored. Duplicate frame
indices fail even if the entire rows are identical. All requested annotations
are scanned before writing any crop; malformed and duplicate rows are reported
with source ID and original line number. Valid index gaps are reported, not
filled. The sampler only selects existing annotations; missing selected
annotations or unavailable decoded frames fail explicitly.

The versioned crop convention treats `(left, top, right, bottom)` as half-open,
expands each side by `margin * original_extent`, floors left/top, ceils
right/bottom, and clips to decoded dimensions. Empty intersections fail. This
endpoint convention is a **project assumption requiring review**, because the
README does not specify endpoint inclusivity. Raw values and the actual applied
box are both retained; eyes are preserved but are not used to align the crop.

Defaults are zero expansion, native crop dimensions (no resize), lossless PNG
with compression 3; optional resize uses OpenCV `INTER_LINEAR`. This is an
explicit migration from undocumented JPEG encoding. Nothing is selected using
test outcomes. The digest hashes sorted, indented UTF-8/LF JSON of schema,
sampling, crop, orientation, decode, resize, encoding and error policies plus
Python/NumPy/OpenCV and available FFmpeg library versions. This records the
software environment but does not promise identical decoder pixels across
different machines/builds. Changing these settings produces a different digest.

## Orientation and controlled local evidence

The bundled README states that some Android videos are upside-down and their
annotations refer to rotated frames. The old script enabled AUTO without
checking its effect. The new decoder forces FFmpeg, disables and verifies AUTO,
reads sequentially from index zero, checks the reported next frame position and
applies only recorded 0/180-degree rotation explicitly. Other angles/backends
fail. The implemented assumption is that annotation coordinates refer to rotated
frames, so boxes are not rotated a second time. Annotation/frame alignment and
geometry are unverified. The configuration records exactly:
`FFmpeg; AUTO=0; explicit metadata rotation 0 or 180 only; assume annotation coordinates refer to rotated frames; alignment and geometry unverified`.
Sequential counter checks are machine checks, not proof of annotation alignment.

An initial header-only survey inspected all 15 training bona-fide Android clips
and found 180-degree metadata for clients `003,005,006,007,008,009`; the other
nine (`002,011,012,021,022,034,053,054,055`) reported zero. Headers for client 002's
laptop bona-fide and Android iPad-attack clips reported zero. No frames were
decoded in that survey. It motivated inclusion of both metadata cases in the
fixed review scope; no crop setting was tuned.

The [full index audit](audit/msu_preprocessing_plan.json) inspected 77,873
annotation rows across 280 sources: zero malformed rows, duplicate indices or
blank rows. There are 80 unannotated indices between each file's first and last
annotation, across 12 videos; these are reported and never filled. It planned
8,400 frames (3,600 train / 4,800 test; 2,100 bona fide /
6,300 attack; 240 per client). **It wrote zero crops and did not verify those
indices against decoded media.** Its config digest is
`04a398e472b7f102911bdc3595b14aee6c5f65415755ed52c954a9442a4c47b2`.

The [controlled crop audit](audit/msu_preprocessing_verification.json) covers
exactly frames 0 and 1 of the four named training videos: eight crops, six bona
fide and two attack, from clients 002 (six) and 003 (two). The other 276 sources
were not selected. All eight AUTO-versus-explicit oriented arrays were
pixel-identical; eyes fell within applied boxes, crop writes succeeded and group
IDs were preserved. Source annotations, crop hashes and dimensions passed the
canonical reader. Its distinct verification config digest is
`0a6558f34d19c6ac5d43c4485532add18d59f333a0f9ca374d58d1aa68e56a53`.

These two reviewed audits received a metadata-only status-schema and assumption
wording correction. Their config digests consequently changed; each retains its
previous schema/digest in `metadata_correction`. All measured observations,
counts and selected indices are unchanged. No crops or decoder evidence were
regenerated for the correction, and no fidelity approval was performed. The
local ignored v1 crop manifest remains historical and is rejected by the v2
reader; the successful reader check above describes the original run.

Android decoded dimensions were 720x480; laptop dimensions were 640x480.
Local overlays/crops are in ignored `data_processed/MSU-MFSD/preprocessing-review-v1`;
licensed imagery is not added to Git. A local contact sheet of the four frame-0
overlays was viewed during implementation: faces appeared upright and overlays
approximately aligned. This limited qualitative inspection is not human review
or a fidelity certification. Human review must inspect all eight original-sized
overlays/crops, confirm eye/face placement and box endpoints, and independently
establish the annotation-to-decoder frame convention.

The [endpoint audit](audit/msu_preprocessing_decoder_boundary.json) is a separate
controlled read-through of **only** client 002 and 003 bona-fide Android sources:
300 successful sequential reads each, indices 0–299, but 301 annotation rows
covering 0–300. The first subsequent read failed in both. No boundary crops were
written and intermediate decoded frames were not visually inspected. Repeating
the audit through the committed API produced the same bytes. This establishes a
discrepancy with this decoder, not which tool or annotation is responsible.
For exactly these two checked training videos, the uniform sampler includes
index 300 and must reject its unavailability. This decoder finding is not a
release-wide result. The full 280-video annotation plan has maxima ranging from
190 to 310, including two sources exceeding 300, and 80 internal index gaps
across 12 sources. Those annotation observations alone establish neither decoder
availability nor an off-by-one mapping or valid correction. Do not drop, clamp,
renumber, shift, reinterpret or automatically remap official annotation indices,
or change sampling to avoid a discrepancy. The index-only
audit is intentionally not a claim of complete extraction readiness.

## Gate before experiments

The subsequent [index-semantics evidence audit](MSU_INDEX_SEMANTICS.md) now covers
all 280 sources, separates codec errors from endpoint availability, and reproduces
the bundled rounded-rate export on four cases. No production mapping or fidelity
approval resulted; the historical two-video evidence above retains its original
scope. Current measurements and the remaining evidence gate are in that report.

The subsequent [final policy and human-fidelity gate](MSU_POLICY_GATE.md) reuses
that evidence and completes calibrated export-domain counting across all 280
sources. Its state remains P4 (unresolved), with human review pending and
`experiment_ready=false`. It supplies a training-only review page combining the
original eight crops with native/export candidate overlays. Export availability
does not authorize a canonical preprocessing change or an annotation remapping.

Undertake a separate evidence task to determine annotation-to-decoder indexing
semantics from source evidence (including the original extraction convention)
and controlled decoding/visual checks. Review the eight orientation/geometry overlays
and endpoint assumptions, and lock any evidence-supported policy revision.
Then validate coverage and crop provenance across the full release under the
accepted settings before integrating a training adapter. No full extraction or
model training has been done here. Keep the raw protocol lock as the source of
truth; do not regenerate it to excuse a processing inconsistency.

RQ1 still has only two bona-fide video groups per client, one per camera, with
equal identity classes required on both probe sides. Session-disjoint claims
remain unsupported; `scene01`, devices and frames are not sessions. RQ3 retains
the verified official 15/20 disjoint client cohorts. A future development/model
selection rule must use training clients only and be documented before the first
controlled PAD experiment. The test cohort must not determine preprocessing or
model settings.
