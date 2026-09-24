# MSU PittPatt annotation-generation provenance investigation

Investigation date: 2026-09-24. Baseline: `research-reconstruction`,
`4feb7fc5254b626247dff37b5dd595d523fee7ed`; working tree initially clean.
No applicable `AGENTS.md` was found. This is an evidence investigation only.

**Conclusion: retain P4.** No inspected evidence independently establishes the
complete stream, timebase, orientation processing and numbering used to generate
the released `.face` files. Preserve `policy_state=P4`, `frozen_policy=null`,
`fidelity_status=manual_review_pending`, `experiment_ready=false`.

## Precise question and evidence scope

For each original `.face` row with stored index `i`, which exact image was
supplied to PittPatt 5.0.2, through which decoder/export pipeline, at which
timestamp or ordinal, after which orientation transform, and how was `i`
assigned? The question concerns the original annotation-producing run, not a
later consumer's plausible association with an image.

The [protocol](MSU_PROTOCOL.md), [index semantics](MSU_INDEX_SEMANTICS.md),
[policy gate](MSU_POLICY_GATE.md), [human review](MSU_HUMAN_REVIEW.md) and
[preprocessing history](MSU_PREPROCESSING.md) were read alongside their relevant
committed JSON evidence. Below, **documented** means an explicit source statement
or code operation; **observed** means a reproduced measurement, with prior runs
distinguished from this investigation; **inference** is interpretation;
**unresolved** identifies an absent link. Local release paths are under ignored
`datasets/MSU/`; line numbers refer to those fingerprinted files.

## Evidence table

| Source / exact locator | Finding and evidentiary strength | Limit |
| --- | --- | --- |
| `datasets/MSU/README.txt`, lines 90–106 | **Documented:** acquisition formats, approximately 30 fps, PittPatt 5.0.2 face/eye detection, first field described as frame index, missing detections, coordinates relative to rotated frames for affected Android videos. | Acquisition frame rate is not the annotation-generation timebase. No explicit numeric base, decoder invocation, resampling rule or box endpoint convention. |
| Same README, lines 22–39, 132–147 | **Documented:** release includes decoding scripts and subject lists; access requires approval; identifies Wen/Han/Jain's 2015 paper. | Download instructions and citation do not specify annotation generation. |
| `datasets/MSU/DecFrames.m`, lines 81–105, 119–129 | **Documented:** probes rotation; exports BMPs with output `-r` derived from `avg_frame_rate` rounded to two decimals; reads an existing `.face` matrix; names eye sidecars with `fidx(t)+1`. | Consumes annotations, never calls PittPatt or writes `.face`. Row `t` is not frame index `i`. `rotated` only drives a notice, not an image transform. |
| `DecFrames.m`, lines 100–115; `DecFrames_real_scene01.m` and `DecFrames_attack_scene01.m`, lines 1–5 each | **Documented:** reuses directories with at least 100 BMPs; scene scripts invoke the helper for train/test lists. | No sidecar/BMP existence check in the sidecar loop; neither entry script adds generation semantics. These scripts were not executed here. |
| `datasets/MSU/ffmpeg/README.txt`, lines 1–15; [source evidence](audit/msu_index_source_evidence.json), `files`, `direct_source_facts`, `missing_referenced_helper` | **Documented:** bundled Win64 build identifies git `9efcfbe`, March 2013. **Observed now:** all seven recorded file sizes/SHA-256 values match; only the three named MATLAB files exist; README's `decode.m` is absent. | Fingerprints identify available tools, not proof that they generated the annotations. No annotation wrapper, PittPatt configuration or generation log was found in the inspected bundled code/text. Absence is local, not a claim that none ever existed. |
| [Native boundary evidence](audit/msu_index_boundaries.json), `records[].annotation_*`, `successful_reads`, `libraries`; [protocol lock](audit/msu_protocol_lock.json), `source_fingerprints`, `normalized_metadata_sha256` | **Observed now:** all 280 annotation hashes match the boundary artifact; text recount confirms 77,873 rows, all minima zero, 80 internal gaps in 12 files. **Observed previously:** decoder successful-read counts and ranges. | Neither raw index statistics nor lock integrity identifies PittPatt input images. The protocol lock does not hash full video contents. |
| [Bundled exports](audit/msu_index_bundled_export.json), `records[].relative_command`, `comparisons`, `ffmpeg_final_progress` | **Observed previously:** actual filenames, duplication and selected full-frame pixel comparisons for four training sources. | Reproduces the supplied export command, not annotation generation. |
| [Eight-source calibration](audit/msu_policy_export_calibration.json) and [supplementary drop calibration](audit/msu_policy_drop_calibration.json), `records`, `scope_limit`, `config_digest` | **Observed previously:** agreement between actual numbered BMP exports and the disk-free image2/BMP counter for nine training sources in the recorded build. | Count/dup/drop equivalence is not pixel correspondence or a cross-build guarantee. |
| [Final human validation](audit/msu_human_review_validation.json), `cases`, `temporal_comparison_counts`, `negative_geometry_cases`, `provenance` | **Observed previously:** 64 training-case observations were recorded and validated, including 56 comparisons. | Preferences and plausible overlays do not reveal the historical PittPatt input stream. |

Source hashes above are anchored by `msu-index-source-evidence-v1`; for example,
`DecFrames.m` SHA-256 is
`9fef6faaddb174cbbd4e43aece752def946e7ffbff38021edc549788985615bc`.
This investigation rechecked bytes and text only; it did not repeat any decoding,
exports, crop generation or review-asset generation.

## The four domains

| Domain | Established | Missing relationship |
| --- | --- | --- |
| **A — original annotation index** | The stored first column, preserved exactly; observed values begin at zero and retain gaps. README defines coordinate fields and rotated-coordinate intent. | Whether the index counts decoder outputs, resampled images, SDK inputs or another sequence; when the counter increments relative to failed detections. No per-row timestamp is supplied by the documented format. |
| **B — native successful-read ordinal** | Project-defined zero-based sequence of successful OpenCV/FFmpeg reads, with AUTO disabled in the recorded audit. Build is recorded in `msu_index_boundaries.json: libraries`; current read loop is `src/identity_invariant_fas/data/msu_preprocessing.py`, lines 143–165. | This is a decoder observation, not a source timestamp or an established PittPatt input index. Explicit 0/180-degree review rotation is a project assumption, not recovered generation code. |
| **C — DecFrames export ordinal / filename** | Reproduced numbering starts at `_001.bmp`; zero-based export ordinal `i` has filename number `i+1`. Code associates annotation `i` with the same-numbered eye sidecar. Rounded-rate export can duplicate/drop frames. | That file association does not establish A=C or C=B, or identify the pixels/orientation originally given to PittPatt. |
| **D — actual PittPatt input** | README attributes detections to SDK 5.0.2 and gives rotated-coordinate intent. | Actual decoder/API, selected video stream, timestamps/timebase, resampling, rotation stage, SDK configuration and `.face` index-writing logic remain unknown. |

**Inference:** the sidecar convention makes export correspondence a plausible
hypothesis, not established provenance. The historical source artifact's phrase
“exported BMP ordinal i+1” refers to the one-based filename number; this document
uses **ordinal** exclusively for zero-based C. The artifact is not rewritten.
Never equate DecFrames filename `i+1` with native frame `i`.

## Exact limits of previous calibration and review

**Prior observations:** the eight-source calibration matched counts, contiguous
numbering, completion/exit status and duplication/drop counters, but measured no
drops. The supplementary training source
`MSU-MFSD:scene01/attack/attack_client022_android_SD_printed_photo_scene01.mp4`
matched 267 images, 3 duplications and 1 drop. Both artifacts share configuration
digest `92a81eefbe44dd485d3baa9955b1fa8f7f15f029865069e349051792774bd318`.
The [training availability audit](audit/msu_policy_train_export.json) reports
120 sources and no unavailable annotation ordinals under that counting model.
These findings establish availability under C, not correspondence to D.

The four-source pixel audit is stronger than counts but still compares B with C.
For `MSU-MFSD:scene01/real/real_client002_android_SD_scene01.mp4`, its
`comparisons` report zero pixel MAE between export ordinal 0 and native 0,
and between export ordinal 150 and native 149. **Inference:** even this one
training example prevents a universal B=C or B=C−1 claim for those exports;
it does not establish which sequence generated A.

The final human audit records export/native/native−1/native+1/indistinguishable
preferences of **30/6/6/7/7** among 56 comparisons. Eight historical crop forms
are excluded from those temporal totals. One negative face-box case and 14 cases
with uncertain geometry remain. **Inference limit:** the 30 export preferences
cannot identify D; candidates can share pixels and neighboring frames can look
similar. Endpoint plausibility is not proof of box inclusivity.

The historical [policy decision](audit/msu_policy_decision.json) predates human
ingestion; its reason mentioning absent human review is retained as history,
not a claim that review is still unperformed. The current distinction is
`human_observation_status=recorded_and_validated` versus fidelity approval still
pending. Existing official-test observations are descriptive only and were not
used here to choose a policy. Codec errors were not investigated or resolved.

## External original-author evidence and access limits

Searches on the investigation date covered the original MSU release path,
MSU/PittPatt, `DecFrames.m`, `decode.m`, SDK 5.0.2 and the cited paper.

- **Documented, search-indexed author source:** the MSU
  [database catalog](https://biometrics.cse.msu.edu/pubs/databases.html) lists the
  release; the [project page](https://biometrics.cse.msu.edu/projects/face_recog_spoofing.html)
  identifies MFSD and links the Wen/Han/Jain publication/database. The retrieved
  text supplies no annotation-generation procedure.
- **Documented, search-indexed author paper excerpt:** the
  [2015 author-hosted manuscript](https://biometrics.cse.msu.edu/Publications/Face/WenHanJain_FaceSpoofDetection_TIFS15.pdf),
  manuscript page 15, conclusion and reference [26], identifies the public
  35-subject subset and cites PittPatt's SDK. That excerpt does not connect a
  particular decoding pipeline to the released `.face` rows.
- **Unresolved access:** direct browser retrieval of the
  [original release page](https://biometrics.cse.msu.edu/Publications/Databases/MSUMobileFaceSpoofing/index.htm)
  failed; direct manuscript/project-page retrieval timed out. HTTP retries for
  the release page/paper resolved to HTTPS and also failed. Indexed excerpts
  are not a complete inspection of the live release site or paper. No claim is
  made that inaccessible material lacks the missing evidence, or that the
  local README is byte-identical to a currently hosted copy.

No original-author generation script, execution log or explicit frame/timebase
statement was recovered. Third-party decoder conventions cannot fill this gap.

## Missing chain, policy conclusion and smallest next request

The missing chain is: identified release video → exact decoder/stream and
timestamp handling → any resampling and orientation processing → exact images
submitted to PittPatt → detector configuration/output → `.face` index assignment.
The available consumer scripts start downstream of that final annotation step.

Under [policy definitions](MSU_POLICY_GATE.md), lines 234–240, **P1** needs proven
native correspondence, **P2** proven DecFrames correspondence, and **P3** another
explicitly established domain. None is established. **P4 remains in force**;
matching endpoint counts, filenames or reviewer preferences cannot close the
missing links.

The smallest next evidence request is an attributable original-author/release
record of the actual annotation wrapper and invocation (including the referenced
`decode.m`, if used), its decoder/SDK versions and index-writing loop. Ask it to
specify direct-video versus exported-image input, stream/PTS/timebase and `-r`/
dup/drop behavior, rotation before detection versus coordinate transformation,
and how missing detections affect numbering. Request a traceable original input
image hash or generation-log entry tied to one **training** `.face` row where B
and C differ: client 002 real Android annotation 150 is an existing discriminating
case. An attributable archived script plus that trace is stronger than recollection
or a new visual preference. Ask whether the same pipeline covered both cameras
and the full public release; one example alone cannot establish universal scope.

If such records are recovered, a separately authorized, narrowly scoped check
could compare that original input identity and index trace with retained candidate
evidence, without full-video re-export. If independent generation evidence remains
unavailable, another count run or overlay vote cannot resolve historical provenance.
This request has **not** been sent and no controlled rerun is authorized here.

**No preprocessing-policy change, preprocessing execution, fidelity approval,
training or PAD experiment authorization is granted.** Original annotations,
raw locks, source review JSON and all committed audit artifacts remain unchanged.
