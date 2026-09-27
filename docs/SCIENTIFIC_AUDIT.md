# Scientific Audit

This refactor identified several issues that matter for a publication-quality
evaluation. They are documented explicitly rather than silently hidden.

## 1. Historical ablation was not architecture-matched

The historical `AblationECNN` used a 128-dimensional representation and did
not contain the final `fc3` layer of the 256-dimensional ECNN encoder used by
II-ECNN. Therefore the previous ablation comparison changed both the encoder
and the presence of GRL.

The legacy `AblationECNN` was subsequently architecture-matched to II-ECNN,
but architecture matching alone was insufficient: the comparison was not
gradient/schedule matched. Identity encoder gradients differed in sign,
magnitude (+1 versus -lambda), and onset (epoch 1 versus after warm-up).

**Consequence:** legacy AblationECNN versus II-ECNN results cannot support the
final causal identity-suppression claim. Retraining those legacy arms does
not resolve the mismatch. Prospective causal comparisons must use the new
controlled three-arm implementation: `spoof_only`, `identity_positive`, and
`identity_adversarial` (see below). Historical outputs remain untouched;
legacy classes are not retroactively reinterpreted.

## 2. Representation analyses used a different image size

Several historical leakage/t-SNE scripts resized NUAA images to 64x64, while
the NUAA folder-token k-fold training scripts used 160x160 inputs. This introduces an
avoidable preprocessing mismatch.

The refactored identity leakage experiment defaults to the same 160x160
evaluation transform as training.

**Consequence:** re-run identity leakage and representation analyses using the
same preprocessing as the corresponding trained model.

## 3. GRL sweep and main II-ECNN training were not schedule-matched

The historical main II-ECNN script used a five-epoch subject-loss warm-up,
whereas the historical GRL sweep applied the subject loss from the first
epoch. That means lambda was not isolated as the only changed variable.

The refactored implementation also had two critical bugs: the publication
factory omitted `training.epochs`, leaving Trainer's default of 100, and the
trainer replaced configured lambda with an unrelated unit-amplitude schedule.
Its `epoch < warmup_epochs` check gave only four zero-lambda epochs when called
with one-based epochs and warmup=5. Shared configuration alone did not make
those runs a valid lambda sensitivity experiment.

**Prospective contract:** `model.grl_lambda` is the configured target, passed
explicitly into an immutable GRL configuration independently of model state.
Trainer requires an explicit epoch budget. Public epochs are 1..total_epochs;
warmup N means exactly epochs 1..N have lambda zero. Subject loss remains
enabled: during GRL warm-up its head can learn, but its encoder gradient is
zero. This is not a subject-loss disabling warm-up.

`training.grl_schedule: fixed` is the default and is explicit in the current
sensitivity config. Epoch N+1 onward uses the exact configured target.
Optional `dann` (`progressive` alias) uses target times
`2/(1+exp(-10*p))-1`, where `p=(epoch-N)/(total_epochs-N)` after warm-up.
It begins above zero in the first active epoch and approaches, but does not
equal, the target at the last epoch. Invalid targets, modes and epoch ranges
fail; GRL all-warmup runs are unsupported. Non-GRL models are not scheduled.
The legacy zero-based script explicitly passes epoch+1 to Trainer.

All sweep runs share folds, seeds, split construction, optimizer, learning
rate, weight decay, subject-loss weight, warm-up, epoch budget, early-stopping
policy, preprocessing and augmentation; only target and output location vary.
The sweep uses a fresh v2 output root and refuses an existing root. Per-epoch
effective coefficients are retained in EpochResult and runner histories.
No sweep or new scientific experiment was executed to validate this fix.

**Consequence:** any results produced under the previous semantics cannot
support final claims about the effect of configured lambda values (including
0.01/0.05/0.10 or an optimum at 0.05). Preserve historical outputs unchanged;
this prospective fix does not repair or validate them retrospectively.

**Outstanding ablation gate:** AblationECNN remains architecture-matched but
is not schedule/gradient-magnitude matched. Its subject branch contributes
positive encoder gradients from epoch 1 without lambda scaling. II-ECNN now
contributes zero subject encoder gradient during warm-up and a negative,
lambda-scaled gradient afterward (both also use subject-loss weight). This
GRL-fix task did not alter historical ablation behavior. The prospective
controlled implementation below supplies a separate matched comparison;
the historical classes and result directories retain their original meaning.

### Prospective controlled identity ablation

`configs/controlled_identity_ablation.yaml` and
`experiments/controlled_identity_ablation.py` define three explicit arms of
one `ControlledIdentityECNN` architecture. This execution path has not been
run to produce scientific results. It delegates to the existing custom NUAA
folder-token k-fold runner; it does not change dataset labels or splits.
On NUAA this is implementation-valid proxy intervention, not a verified
human-identity intervention; see [NUAA scientific role](NUAA_SCIENTIFIC_ROLE.md).

| Arm | Objective | Encoder identity scale, epochs 1..N | Scale after N |
| --- | --- | --- | --- |
| Spoof-only | spoof CE only | 0, subject objective disabled | 0 |
| Identity-positive | spoof CE + beta * identity CE | 0, subject head trains | +effective lambda |
| Identity-adversarial | spoof CE + beta * identity CE | 0, subject head trains | -effective lambda |

The signed gradient control is immediately **before** the subject classifier.
Its forward pass is the identity. Subject-head gradients have the ordinary
beta weighting and are never multiplied by lambda or its sign. Both identity
arms use exactly the corrected fixed/DANN schedule above, with the same
nonnegative target, warm-up, total epochs and beta. The Trainer freezes the
target separately from mutable model routing state. `identity_history.json`
records the explicit arm, actual signed `identity_encoder_scale`, subject
objective enabled/disabled, and beta; controlled records do not label a
positive routing scale as `grl_lambda`.

All three arms instantiate the same encoder, spoof head, and subject head in
the same order, with identical state-dict keys and parameter counts. The
spoof-only subject head is dormant: its forward branch is skipped, its loss
is disabled even if labels exist, and its gradients remain None, so AdamW
does not apply updates or weight decay to those parameters. Its allocated
capacity includes the unused head; its active optimization capacity does
not. The two identity arms have identical active capacity. At identical
weights/input, spoof features, logits and spoof-only gradients are identical.

The canonical driver varies only the explicit arm and output path. Dataset,
fold seed and subject split, per-fold initialization seed, transforms,
augmentation, batch order, optimizer, learning rate, weight decay, epochs,
early-stopping configuration, lowest-validation-ACER model-selection rule,
and evaluation metrics all use the same existing runner. Model selection
can select different epochs as an outcome of the intervention. It reserves
a fresh output root and retains each generated configuration. Lambda 0.05
is a configurable example, **not** a scientifically selected final value.
No historical sweep is used to choose it. The existing config key
`grl_schedule` now also names the shared absolute schedule for controlled
arms; `identity_target_lambda` is their nonnegative target.

Identity-positive means that the auxiliary objective encourages
identity-discriminative shared representations; it does not assume that
measured identity information necessarily increases. Opposite, equal-scale
gradients are guaranteed at the same parameters and input, not identical
gradient norms along subsequently diverging optimization trajectories.

**Historical limit:** old AblationECNN versus II-ECNN outputs changed sign,
magnitude (+1 versus -lambda), and onset (epoch 1 versus after warm-up).
They cannot support a final causal claim attributing PAD or representation
differences to identity-gradient direction or suppression. They remain
historical/debug evidence only, alongside the invalid old lambda-sensitivity
results described above. No historical output is deleted, rewritten, or
retroactively repaired. Legacy model/config paths remain available for
compatibility and are not the new controlled publication comparison.

**Before execution:** freeze an approved dataset/protocol and prospective
hyperparameter-selection plan, and separately authorize experiments. Human
identity claims require verified provenance; NUAA proxy-only experiments must
follow the scientific-role decision. The canonical configuration
retains the existing custom repartitioning of NUAA's test partition; it is
not an official-protocol result. Synthetic autograd tests validate the
implementation, not identity suppression, scientific performance, or
publication readiness. No MSU state or pilot evidence is involved.

## 4. APCER/BPCER naming was inconsistent in one historical sweep

The project label convention is:

- 0 = bona fide
- 1 = attack

For this convention:

- APCER = attack predicted bona fide = FN / (FN + TP)
- BPCER = bona fide predicted attack = FP / (FP + TN)

The historical `train_iecnn_grl_sweep.py` swapped the APCER and BPCER names.
ACER was unaffected because it averages the two rates.

The refactored metrics module has one centralized implementation and tests.

## 5. NUAA folder-token k-fold protocol is a custom proxy protocol

The historical k-fold experiments loaded the NUAA `test` partition and
then re-partitioned raw folder tokens into five folder-token-disjoint folds.
This supports only custom proxy generalization, not verified unseen-human
generalization. Cross-class human correspondence remains unresolved. It is **not** the
official NUAA train/test protocol and should not be numerically compared to a
paper reporting the official protocol without a clear qualification.

For the manuscript, report separately:

1. official-source-list ECNN-inspired backbone evaluation (external-paper
   reproduction requires a separate fidelity audit), and
2. custom folder-token-disjoint proxy evaluation, only if separately planned;
   the minimum publication matrix omits this optional experiment.

## 6. Validation must respect the audited grouping semantics

Historical scripts split validation samples randomly from the outer training
pool, which allows the same folder tokens to occur in train and validation.
The outer test fold remained folder-token-disjoint; that does not certify
human disjointness or rule out correlated content. Model selection can still
exploit token-associated validation signals.

The NUAA runner reserves validation **folder tokens**, not samples. Verified
human-subject-disjoint claims require independently documented human mappings.

## 7. Identity leakage probe semantics

Closed-set identity probing requires the same identity classes to be represented
in probe training and testing. The publication path now requires independent
groups as well as disjoint samples; the sample-level utility is retained only
as legacy/exploratory. Identities may be unseen to the PAD encoder but must be
known to the identity classifier. Genuinely unseen classifier identities need
verification, retrieval, or open-set identification rather than ordinary
closed-set multiclass evaluation.

For video datasets, use session/video groups to prevent near-duplicate frames
from crossing probe folds. See [identity probing](IDENTITY_PROBING.md) for the
validated group-aware API and the current metadata limitations.

## Publication status

The refactored code is an implementation base, not completed publication
evidence. The [prospective publication matrix](PUBLICATION_EXPERIMENT_MATRIX.md)
defines required versus optional experiments and remaining execution adapters
and gates. Use the ECNN-inspired controlled-backbone name until external-paper
fidelity is verified. Historical sweeps and unmatched ablations remain invalid
for final claims; their existence does not require repeating every old study.
