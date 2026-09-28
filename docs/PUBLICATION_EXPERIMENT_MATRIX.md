# Prospective publication experiment matrix

**Design status: PUBLICATION_MATRIX_FROZEN = true. Execution: not yet executed.**
Baseline: `529ef7adc1a7cd36b0639d6e23968b137cb695d2`. This is the prospective
selection/reporting contract, not execution authorization or a readiness waiver.
Changes require a dated amendment before affected outcomes are inspected; retain
the original plan and disclose deviations. No historical result selects a setting.

**MSU amendment (2026-09-27):** [Amendment 001](PUBLICATION_EXPERIMENT_MATRIX_AMENDMENT_001.md)
supersedes the annotation-based MSU preprocessing clause and its publication
readiness dependency with prospective Path B. The original wording below is
retained as history; historical P4 remains unresolved. All other matrix rules
remain unchanged, and Path B is not yet technically qualified.

Authority: [NUAA role](NUAA_SCIENTIFIC_ROLE.md), [MSU client protocol](MSU_PROTOCOL.md),
[MSU preprocessing](MSU_PREPROCESSING.md), [MSU gate](MSU_POLICY_GATE.md),
[Replay protocol](REPLAY_PROTOCOL.md), [scientific audit](SCIENTIFIC_AUDIT.md),
[identity probing](IDENTITY_PROBING.md), and [reproducibility](REPRODUCIBILITY.md).

## 1. Research questions and minimum package

- **RQ1:** How much verified-client identity information is linearly recoverable
  from frozen FAS representations on clients unseen during encoder training,
  under camera/source-video-separated closed-set probing?
- **RQ2:** What effects do matched positive versus adversarial identity-gradient
  interventions have on PAD performance and identity decodability, relative to
  spoof-only training? Encouragement/suppression describes the objective, not an
  assumed measured outcome or proof that identity is the sole causal mediator.
- **RQ3:** Does identity-adversarial training improve PAD generalization to
  unseen verified clients? Does any benefit transfer to an external dataset
  when independently validated target data become available?

The contribution is controlled evidence about identity dependence, not novelty
of GRL. Negative or inconclusive results remain reportable. The core is a minimum
defensible study, not a guarantee of journal acceptance or benchmark superiority.

| ID / priority | Experiment and arms | Can support | Cannot support |
| --- | --- | --- | --- |
| N1 / CORE | NUAA official source lists; all three controlled arms, five seeds | ECNN-inspired PAD sanity check and matched auxiliary-proxy effects under disclosed validation | Verified-human suppression, unseen-human generalization, or faithful external-paper reproduction |
| M1 / CORE, blocked on MSU gate | Official MSU 15-train / 20-test clients; three arms, five seeds | PAD generalization to documented held-out clients; matched intervention comparisons | Universal identity invariance, session independence, or external generalization by itself |
| M2 / CORE, same gate | Frozen M1 encoders; verified-client probes in both camera directions | Linear client decodability across separate source videos/cameras | Open-set identity recognition, session-disjoint claims, or absence of all identity information |
| N2 / OMIT from minimum package | NUAA custom folder-token k-fold | Only exploratory proxy generalization if separately preregistered later | Human-subject-disjoint or human-identity suppression claims |
| R1 / CONDITIONAL | MSU -> Replay zero-shot, same M1 checkpoints | External transfer under a validated target protocol | Current readiness or superiority across all datasets |
| R2 / OPTIONAL, conditional | Replay within-dataset three-arm replication; reverse transfer | Independent verified-client replication after a separate frozen plan | A required prerequisite for the core MSU study |
| F1 / OPTIONAL | Faithful external-paper reproduction | Reproduction only after architecture/classifier/preprocessing/protocol fidelity audit | Retroactive certification of current `PaperECNN` or historical outputs |

N2 is omitted prospectively, not retained because code exists. Counts derived
only from the committed NUAA source audit and current seed-42 split rules show
outer fold 5 has 118 bona-fide versus 1,198 attack images, and fold-1 validation
has 118 versus 789. Token/class association and unverified human mapping further
limit interpretation. Do not search split seeds, rebalance folds after outcomes,
or use this official-test-source custom study to select N1 settings. A later
supplement requires its own frozen imbalance treatment and claim limits; it
must not delay core execution or be included selectively after favorable results.

## 2. Backbone and matched training contract

Publication name: **ECNN-inspired controlled backbone**. `PaperECNN` is an API
name, not a fidelity certificate: the committed audit leaves external-paper
architecture, preprocessing and training correspondence unverified. The encoder
has six convolutions, pooling/batch normalization and a 256-dimensional feature
output; the project uses a learned two-logit linear spoof head. No new external
reproduction is required for the causal comparison; F1 is a useful optional
comparator if a separate fidelity audit and execution are justified.

Use `ControlledIdentityECNN` for every core arm, including the dormant subject
head in `spoof_only`. One shared architecture/state-dict layout, identical
initial encoder/spoof/subject weights and batch permutations within each matched
seed/fold; verify initialization digests before fitting. The number of auxiliary
classes is the number of training clients/tokens only (MSU inner: 10; final: 15;
N1 fitting: 7), never padded using test identities.

| Arm | Objective | Encoder identity scale, epochs 1..5 | Epoch 6 onward |
| --- | --- | --- | --- |
| `spoof_only` | Spoof CE only | 0; subject objective disabled | 0 |
| `identity_positive` | Spoof CE + 0.10 * identity CE | 0; subject head trains | +lambda |
| `identity_adversarial` | Spoof CE + 0.10 * identity CE | 0; subject head trains | -lambda |

Subject-head gradients retain ordinary beta weighting, never lambda scaling.
Freeze **beta=0.10**, **fixed schedule**, **warm-up=5**, no beta/schedule search.
The effective identity contribution to the encoder is beta times the signed
routing scale, not an additional loss multiplier on the subject classifier.

All core fits: random initialization, AdamW LR=0.001, betas=(0.9,0.999),
eps=1e-8, weight_decay=0.01; batch size 32; no LR schedule, no class weighting,
no augmentation, no early stopping; num_workers=0; shuffle training only;
drop_last=false. RGB, shared 160x160 resize and ToTensor [0,1], no learned
normalization or additional preprocessing. Pin library/environment versions and
resize behavior. Identity CE is required for every training sample in identity
arms; spoof-only ignores those labels. Validation/test use spoof scores only.

For MSU, plan 30 crops per source video using the existing uniform-over-annotation
positions sampler, zero margin, approved orientation/crop/frame-domain policy,
and lossless provenance-bearing extraction, then the common model transform.
This does NOT approve the current unverified mapping/geometry. Missing frames,
failed codecs or incomplete coverage must stop the main study, not trigger
silent replacement/exclusion, sampler changes or reduced test cohorts. If the
approved path cannot satisfy this contract, amend before tuning; never choose
preprocessing using held-out PAD/probe performance. Train on the same frozen
frame manifest for every arm; all videos have equal planned frame counts.

## 3. MSU training-only selection and final fitting

Official training clients are exactly
`002,003,005,006,007,008,009,011,012,021,022,034,053,054,055`.
Freeze three project-defined inner validation groups by round-robin allocation
of this sorted list (no random partition seed and no official development set):

| Inner fold | Validation clients | Fitting clients | Initialization/data-order seed |
| --- | --- | --- | ---: |
| 1 | `002,006,009,021,053` | Other 10 official training clients | 1730 |
| 2 | `003,007,011,022,054` | Other 10 official training clients | 1731 |
| 3 | `005,008,012,034,055` | Other 10 official training clients | 1732 |

Keep every frame/video of a client on one side. Each validation group contains
10 bona-fide and 30 attack videos under the locked inventory. Fit each fold for
30 epochs. Evaluate validation **video ACER** after every epoch at the fixed
threshold below; only epochs **6..30** are eligible for selection.

The prospective lambda grid is **{0.01, 0.05, 0.10}**. These are new candidate
choices, not evidence-based favorites from the invalid historical sweep. For
each fold, fit positive and adversarial at all three values plus one spoof-only
reference: **21 tuning fits** total. The spoof-only run is reused in the
selection arithmetic, not rerun per lambda. No exploratory expansion of the grid.

Let A(f,a,l,e) be validation video ACER. Select the single pair (lambda*, E*)
minimizing the equal-weight mean over the three folds and three arms at a
common eligible epoch; A for spoof-only is independent of lambda. This treats
positive/adversarial symmetrically and includes the baseline in epoch choice.
Exact ties: smaller lambda, then earlier epoch. Do not independently select
arm-specific lambda or epochs, nor use identity probe results for selection.
Publish the complete training-side selection table, including unsuccessful
candidates. Nonfinite/missing metrics or failed fits stop selection; no seed
replacement, dropping a fold, or extending the epoch budget after seeing trends.

Freeze lambda*, E*, input/code/environment digests and the selection record
before final fitting. Reinitialize **all 15 training clients** for each final
arm/seed, run exactly E* epochs with the same five-epoch warm-up, and retain
only the predetermined final-epoch checkpoint for reporting. No final validation
holdout, per-seed best checkpoint, or official-test checkpoint selection.
Training.epochs/Trainer total_epochs must equal the actual run budget (30 in
tuning; E* in refits). Fixed schedule avoids rescaling when the budget changes.

Final evaluation clients are exactly
`001,013,014,023,024,026,028,029,030,032,033,035,036,037,039,042,048,049,050,051`.
Their PAD scores/probe outcomes are opened only in the frozen final reporting
stage after all choices and checkpoints are locked. Prior test metadata/codec
audits already exist: do not call the media historically untouched or this an
untouched blinded confirmation. No test observation may select model, threshold,
epoch or preprocessing. Technical validation uses the already frozen policy;
an unresolved test failure stops reporting rather than motivating a favorable fix.

## 4. NUAA official-source-list design

Use `client_train_face.txt` and `imposter_train_face.txt` only for fitting and
project-defined validation; `client_test_face.txt` and `imposter_test_face.txt`
are final reporting only. Preserve source labels and crop contents. Validation
tokens are fixed at **0004,0007**, the current 0.20 split of sorted training
tokens with partition seed 42; fitting tokens are **0001,0002,0003,0005,0006,0008,0009**.
This is token separation, not verified-human separation. Official train/test
tokens overlap. No official testing image enters training-side selection.

Fit all three arms for 30 epochs for each final seed, using **lambda=0.05 fixed
prospectively**, beta=0.10 and the common settings. This value is a design
constant, not a claimed optimum, not selected from old results or MSU/test
performance; there is no NUAA grid. For each seed, choose one common epoch in
6..30 minimizing mean validation image ACER across all three arms, ties to the
earlier epoch. Report the three checkpoints at that epoch, without refitting on
validation tokens. Disclose reserved validation and actual fitting counts;
this is official-source-list evaluation with project-defined validation, not
an exact reproduction of a paper that trained on every listed training image.

N1 supplies PAD/backbone sanity evidence and proxy intervention comparisons,
not verified-human conclusions. No separate redundant PaperECNNClassifier vs
controlled-spoof-only comparison is required: their spoof architectures are the
same. Historical outputs remain untouched and cannot choose any setting here.

## 5. Seeds, endpoints and threshold policy

Final model seeds for every core arm/dataset: **[11,23,37,53,71]**. Seed the model
and an explicitly recorded data-order generator with that seed; match both
across arms. These five repeats quantify optimization variability, not five
different official test partitions. Inner MSU seeds above are tuning-only;
NUAA partition seed 42 is independent of model seed. Probe seed=42;
bootstrap seed=20260927, generator NumPy PCG64. Pin software versions. Do not
replace an unfavorable seed; reproduce technical failures under the same seed
or declare the planned package incomplete.

**Primary PAD metric: ACER** at a fixed threshold, image-level for NUAA and
video-level for MSU/Replay. The primary M1 contrast is adversarial minus
spoof-only ACER; adversarial minus positive is the prespecified sign comparison.
Report positive minus spoof-only as well. Lower ACER is better. Do not average
image-level NUAA and video-level MSU into one cross-dataset score.

Code audit: labels 0=bona fide, 1=attack; APCER=FN/(FN+TP), BPCER=FP/(FP+TN),
ACER=(APCER+BPCER)/2. Attack score is softmax(logits)[1]. For video results take
the arithmetic mean of the 30 frame attack scores per canonical source video,
then classify; no vote, max-score or median search. Each video has equal weight.
Pooled-attack APCER is the primary definition; additionally report each MSU
attack-medium APCER descriptively, without selecting the best medium.

Freeze **attack iff score >= 0.5** for every dataset, validation and reporting.
No threshold is tuned. The current Trainer uses argmax (exact ties favor bona
fide), whereas video aggregation uses >=0.5. A small tested reporting/validation
adapter must enforce this single tie rule before execution; do not mix policies.
Secondary PAD metrics: APCER, BPCER, ROC-AUC, diagnostic EER and accuracy, with
counts and evaluation unit. EER uses the repository's nearest empirical ROC
point minimizing |FPR-FNR| and averages the two errors there; record that it is
discrete, not interpolated. Its test-derived threshold is descriptive only and
must never become the operating threshold or a selection input. HTER is not
an additional core endpoint; under the same pooled binary errors it duplicates
ACER here. Do not label fixed-0.5 Replay results as its development-EER-threshold
benchmark protocol. Any such benchmark comparator needs a separate plan.

Both PAD classes must be present in every selection/reporting unit. Current
metrics use nanmean for a missing class, which is not acceptable evidence of
ACER; the future adapter must reject such inputs. Preserve raw score precision,
report rates as percentages and contrasts as percentage points; never select
using rounded display values.

## 6. Verified-client identity probing and representation evidence

For each of the 15 frozen M1 checkpoints, use `extract_features` in eval/no-grad
mode: 256-dimensional encoder output, before either classifier. No encoder or
batch-normalization updates. Use only the 40 official-test bona-fide source
videos, 20 clients x two cameras, with the same approved 30 frames per video.
Do not include attacks, auxiliary-head logits, or real/attack class indicators.

Two explicit closed-set `ProbeSplit`s on the same frozen table:
**Android -> laptop**, and **laptop -> Android**. All source-video frames stay
together, both sides have the same 20 verified clients, and source-video IDs
are disjoint within each direction. `session_disjoint=false`; cameras and
scene01 are not sessions. Clients are unseen to the PAD encoder but necessarily
known to the probe classifier. Fitting this prespecified probe on one camera
of test-client features is permitted measurement, not PAD fitting or permission
to tune encoders on test-client outcomes.

Freeze train-camera-only StandardScaler plus balanced L2 logistic regression,
C=1.0, lbfgs, max_iter=5000, seed=42, repository multiclass behavior in a pinned
scikit-learn version. No probe grid, evaluation-side scaling or feature selection.
Convergence failure invalidates that probe; no adaptive test-informed settings.
Use the group-aware API, never the legacy NUAA sample-CV command.

Primary identity endpoint: **balanced accuracy** (mean recall over all 20 client
classes), reported separately by direction and averaged equally within each
model seed. With exactly 30 evaluation frames/client it equals ordinary
accuracy; retain accuracy and macro-F1 as secondary metrics and uniform chance
5% as a descriptive reference. Primary identity contrast: adversarial minus
positive, with spoof-only comparisons also reported. Fewer than all prescribed
clients/videos or incomplete frame coverage stops the core probe.

Decodability measures information accessible to this classifier and acquisition
shift; a weak score does not prove information-theoretic invariance. Joint PAD
and probe contrasts are the necessary representation evidence. Core PAD
discrimination/AUC already measures task utility: no extra table is required
merely to show separation. Fisher ratios, silhouette, distances and t-SNE are
optional descriptive supplements requiring a separately fixed sampling/grouping
recipe; t-SNE is never evidence of suppression. No outcome-dependent choice of
layer, camera direction, checkpoint or representation plot is allowed.

## 7. Reporting and uncertainty

Report all five seeds, mean and sample SD (ddof=1), and per-seed paired arm
differences. Do not publish only the best seed or treat frames/folds/camera
directions as independent replication. Always show both M2 directions; they
reuse recordings in different roles and are not ten independent repeats.

For M1, use 5,000 paired two-axis bootstrap draws: independently resample five
seed indices and 20 test-client indices with replacement; use the same draws
for every arm, retain all eight videos within each sampled client (multiplicity
as weights), recompute video metrics per sampled seed and average over seeds.
Report percentile 2.5/97.5% intervals for arm ACER and paired differences.
This respects source clustering and estimates uncertainty over client/seed
sampling conditional on the frozen training cohort and selection procedure;
it does not include uncertainty from alternative training cohorts or tuning.

For M2, retain per-frame probe predictions and use the same paired seed/client
resampling, carrying both camera directions and all frames within each sampled
client. Weight each client's recall by its sampled multiplicity and compute
the equal-direction mean; do not refit
probes or duplicate identity classes as new labels. Intervals are conditional
on the fitted probes/20-class task, not unseen-population identity enrollment.
Small samples (five seeds, 20 clients) limit precision. Emphasize effect sizes,
intervals and limitations; no confirmatory p-values or significance claims.
NUAA core reports seed variation and counts only, without image-iid confidence
intervals. Existing sample-stratified bootstrap is not the MSU cluster method.

## 8. Conditional external study

R1 is only **MSU -> Replay official PAD test cohort**, all three M1 arms and
five frozen seed checkpoints separately, no ensemble selection. Freeze a
target input policy after catalog/inventory validation and before target
performance is computed: source-compatible RGB/crop semantics and 160x160
transform, deterministic 30-frame/video sampling, mean video attack score,
same >=0.5 threshold. No target train/devel/test score, identity probe, label
or threshold result may select the source model or preprocessing settings;
labels are for protocol validation and final scoring only. No fine-tuning,
target normalization fitting or recalibration. If crop/acquisition compatibility
cannot be justified without tuning, R1 stays blocked. State fixed-threshold
cross-dataset protocol explicitly, not official within-Replay reproduction.

Licensed data, authentic official client/recording/cohort and enrollment/access
mapping, source fingerprints, reconciled inventory, approved preprocessing and
group validation are prerequisites. Current Replay metadata tests are synthetic,
not acquisition evidence. R2/reverse transfer require a separate preregistration
and are omitted from the minimum package. Replay never blocks M1/M2.

## 9. Execution order and remaining gates

1. Review/freeze this matrix and code baseline; retain historical results as
   historical only. No planned result or selected lambda*/E* exists yet.
2. Resolve the existing MSU preprocessing/execution gate through its authorized
   evidence process. Preserve P4, frozen_policy=null,
   fidelity_status=manual_review_pending and experiment_ready=false until that
   process actually clears them. This plan authorizes no timing infrastructure,
   new correspondence experiment, alternate decoder, exclusion or gate bypass.
3. Before any core GPU training, implement only the small adapters needed to
   execute this contract: approved MSU manifest/readiness enforcement, explicit
   inner-client partitions, separated split/model seeds, shared selection,
   final-epoch refit, uniform threshold ties/video validation, both-class checks,
   and probe predictions/balanced accuracy/cluster reporting. Synthetic tests
   must verify these. Lock environment, manifests, sampling and configuration
   snapshots; verify class/client/video coverage and matched initialization.
4. Separately authorize and run only the 21 MSU training-side tuning fits.
5. Freeze lambda*, E* and full selection evidence without consulting test
   performance. No grid expansion or best-arm-specific tuning.
6. Separately authorize the 15 final M1 fits; lock all final checkpoints before
   the single final PAD reporting stage. Keep failed/partial runs auditable.
7. Run the 30 fixed M2 probe fits (15 checkpoints x two directions), with no
   feedback into PAD selection or preprocessing.
8. Run the 15 N1 official-list fits under the independently frozen NUAA rule;
   lock its selected checkpoints before final NUAA reporting. N2 stays omitted.
9. Add R1 only if its acquisition/protocol/compatibility gates pass; otherwise
   omit it and narrow the paper's external-generalization claims.
10. Apply prespecified statistics/figures, report null findings and deviations,
    then write claims within the matrix bounds.

Core budget: **21 MSU tuning fits + 15 MSU final fits + 15 NUAA fits**, at most
30 epochs each; **30 linear probe fits**, not additional encoder training.
No optional reproduction, extra backbone, larger sweep or dataset collection
blocks this package. If MSU cannot clear its gate, the central verified-client
paper remains blocked; NUAA proxy work is not an interchangeable substitute.

Existing YAMLs are implementation examples, not executable definitions of this
new matrix: controlled NUAA currently has 100 epochs/custom folds; grl_sweep is
NUAA-only; MSU/Replay configs reference legacy generic manifests. The current
official NUAA runner couples split/model seeds and chooses checkpoints per arm;
it needs the explicit fixed split/shared-epoch adapter above. Current Trainer
evaluation is frame/image-level, probe output lacks per-observation predictions
and balanced accuracy, and existing bootstrap is sample-based. No runner or
config was changed, made ready, or executed by this design task. No orchestration
framework or machine-readable execution plan is required to freeze this prose.
