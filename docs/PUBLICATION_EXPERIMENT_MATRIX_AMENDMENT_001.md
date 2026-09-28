# Publication matrix amendment 001: prospective MSU preprocessing

Date: 2026-09-27. Parent matrix at commit
`534d7d91655d5fafe1b5ce2d862164545033c785`.
**Prospective design amendment; implementation/qualification not executed.**
No publication scientific outcomes have been observed and no publication
experiment has run, as recorded in the task's execution state. Historical
technical/manual evidence exists and is not being described as untouched data.

## Reason and exact scope

The original matrix section 2 prescribed 30 crops using uniform annotation
positions and zero margin, conditional on approving historical frame/geometry
correspondence. That historical reconstruction remains blocked. The paper's
verified-client questions can instead be studied under an explicit shared
project pipeline, without making a PittPatt reproduction claim.

This amendment supersedes **only MSU preprocessing and the corresponding
publication readiness dependency** in:

- Section 2's paragraph starting "For MSU, plan 30 crops per source video" and
  its MSU realization of the common input transform.
- Section 1's M1/M2 "MSU gate" references: these now require the new Path B gate.
- Section 6's M2 frame inputs: use the same Path B manifest as M1.
- Section 9 steps 2-3 and the final MSU-gate condition: qualify Path B and its
  separate manifest/reader, rather than require resolution of historical P4.

The original text remains visible in the parent matrix as amendment history;
it is no longer the operative annotation-based MSU sampling requirement. Historical
preprocessing documents remain authoritative descriptions of Path A only.

## Replacement contract

[MSU_PROSPECTIVE_PREPROCESSING.md](MSU_PROSPECTIVE_PREPROCESSING.md) is the normative
Path B specification, including artifact hashes, source binding, qualification
set and failure conditions. In brief:

1. Use the same locked 280 original videos and official 15-train/20-test clients;
   source-only metadata verification has no `.face` input requirement.
2. FFmpeg/ffprobe 8.1.2, versioned Gyan Windows essentials archive pinned by
   SHA-256; native CPU decode, complete sequence, strict diagnostic failure,
   no seeking/resampling, explicit container rotation and RGB24 conversion.
3. For N >= 30 complete decoded frames, select zero-based ordinals
   `floor(j*(N-1)/29)` for j=0..29, endpoints included, no repetition/replacement.
4. OpenCV 5.0.0.93 CPU YuNet FP32 2023mar model pinned by SHA-256, score 0.90,
   NMS 0.30, top-K 5000; exactly one detection, no landmark alignment; box
   expansion 10% per side; floor/ceil bounds clipped to the oriented raster.
   Runtime clarification (2026-09-28): `DNN_BACKEND_OPENCV` / `DNN_TARGET_CPU`,
   forced `ENGINE_CLASSIC` (no ONNX Runtime), fixed float32 NCHW [1,3,640,640].
   Use the normative specification's 640x640 bilinear/black-letterbox copy and
   inverse box scaling to original oriented coordinates before the 10% expansion;
   crop the original raster, never the detector copy. No adaptive resolution.
5. Pillow 12.3.0 bilinear RGB resize to 160x160, lossless PNG/reload hash checks,
   then ToTensor without another geometric operation. Shared frozen manifest
   for all MSU arms and probes; 30 crops/video and equal video contribution remain.
6. Any required decode/detection/crop/coverage failure stops the study. First
   qualify eight specified training videos, including four codec cases; only
   then permit frozen-policy mechanical checks on the four test codec cases.
   No test preferences/performance select alternatives; no silent exclusions.

This intentionally does **not** reproduce PittPatt preprocessing. `.face` files
remain preserved historical provenance, not publication preprocessing inputs.
Historical P4 stays unresolved with `frozen_policy=null`,
`fidelity_status=manual_review_pending`, `experiment_ready=false`. The prospective
path has a separate unqualified status; this document grants no execution readiness.

## Unchanged scope and claims

RQ1-RQ3, core/optional experiment roles, exact client cohorts, controlled
spoof-only/positive/adversarial interventions, architecture, lambda/beta/schedule,
warm-up, seeds, training-only tuning, checkpoint and threshold policies, video
aggregation, camera/source-video-separated probes and statistical reporting are
unchanged. NUAA is unaffected. Replay remains conditional; its own future input
compatibility qualification must reference this amended MSU source pipeline.

MSU can support verified-client generalization and identity decodability under
this prospective pipeline after its gates pass. It cannot establish historical
PittPatt correspondence, faithful original preprocessing, or model-only causes
of differences from papers using different preprocessing. No historical output,
policy decision, human review or private pilot is rewritten or rescued.

Next task is bounded implementation and synthetic validation of Path B, with
artifact acquisition/fingerprint verification. Subsequent technical media
qualification and full processing require separate authorization. No GPU
training, probe, PAD inference, video decode or crop generation occurred here.
