# Scientific Audit

This refactor identified several issues that matter for a publication-quality
evaluation. They are documented explicitly rather than silently hidden.

## 1. Historical ablation was not architecture-matched

The historical `AblationECNN` used a 128-dimensional representation and did
not contain the final `fc3` layer of the 256-dimensional ECNN encoder used by
II-ECNN. Therefore the previous ablation comparison changed both the encoder
and the presence of GRL.

The public implementation fixes this: `AblationECNN` now uses the exact same
ECNN encoder, spoof head, and subject head as II-ECNN, with GRL as the intended
single causal difference.

**Consequence:** historical ablation numbers must not be presented as the final
controlled GRL ablation. Re-train the corrected ablation before manuscript
submission.

## 2. Representation analyses used a different image size

Several historical leakage/t-SNE scripts resized NUAA images to 64x64, while
the subject-k-fold training scripts used 160x160 inputs. This introduces an
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
task does not alter ablation behavior. A separate scientific design decision
is required before claiming a sign-only causal comparison.

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

## 5. NUAA subject-k-fold protocol is a custom protocol

The historical subject-k-fold experiments loaded the NUAA `test` partition and
then re-partitioned those identities into five subject-disjoint folds. This is
a valid custom identity-generalization experiment, but it is **not** the
official NUAA train/test protocol and should not be numerically compared to a
paper reporting the official protocol without a clear qualification.

For the manuscript, report separately:

1. official-protocol baseline reproduction, and
2. custom subject-disjoint evaluation used to test identity generalization.

## 6. Validation should also be identity-disjoint for the strict protocol

Historical scripts split validation samples randomly from the outer training
pool, which allows the same subjects to occur in train and validation. The
outer test fold remained subject-disjoint, so this does not contaminate test
samples, but model selection can still exploit subject-specific validation
signals.

The refactored strict protocol reserves validation **subjects**, not samples.

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

The refactored code is suitable as the implementation base for the next
experiments, but the corrected ablation, GRL sweep, and representation analyses
should be re-run before final paper claims are frozen.
