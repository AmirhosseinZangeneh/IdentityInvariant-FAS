# Group-aware closed-set identity probing

The recommended API is `evaluation.identity_probe`. It evaluates externally
supplied frozen features with a linear identity classifier. It does not train
the PAD encoder, infer identities or groups, read datasets, or run experiments
by itself.

## Two different notions of unseen identity

An identity can be **unseen to the PAD encoder**: none of that person's data
was used to train the representation. Its frozen features can still be used
in a closed-set identity probe, provided the **identity classifier** receives
independent training observations of that person.

An identity **unseen to the identity classifier** is absent from its training
classes. Ordinary closed-set multiclass accuracy is not a valid evaluation
of that task. Use a separately specified verification, retrieval, or open-set
identification protocol for genuinely unseen classes.

Closed-set probing therefore keeps exactly the same identity classes on both
sides, while holding out independent observation groups. Encoder-training
identity provenance is a separate audit; this API cannot establish it from
features alone.

## Metadata and API

Each `ProbeObservation` corresponds to exactly one row of the feature matrix:

| Field | Meaning |
| --- | --- |
| `sample_id` | Unique observation/frame identifier, not a class or group label. |
| `identity_id` | Verified identity-class label for the probe, not the PAD real/attack label. |
| `group_id` | Independent source unit such as a video; all its frames share this ID. |
| `session_id` | Optional higher-level recording session containing one or more groups. |

IDs are non-empty opaque strings with no surrounding whitespace; missing
sessions use `None`. Sample IDs must be unique across the supplied table.
Group and session IDs must be globally scoped within that table. If a source
uses local video/session numbers, supply an audited composite key that
distinguishes different recordings. Conversely, do not prefix a shared
recording with identity labels to split it into artificial independent groups.
Multiple identities can legitimately share a group or session.

```python
from identity_invariant_fas.evaluation import (
    ProbeObservation,
    make_closed_set_probe_splits,
    validate_closed_set_probe_split,
    evaluate_closed_set_identity_probe,
)

# metadata and frozen_features must have exactly the same observation order.
observations = [
    ProbeObservation(
        sample_id=row["sample_id"],
        identity_id=row["verified_identity_id"],
        group_id=row["source_video_id"],
        session_id=row.get("recording_session_id"),
    )
    for row in metadata
]
splits = make_closed_set_probe_splits(observations, n_splits=3, seed=42)
for split in splits:
    validate_closed_set_probe_split(observations, split)
result = evaluate_closed_set_identity_probe(
    frozen_features, observations, splits=splits, seed=42,
)
```

Set `session_disjoint=True` consistently on generation, validation, and
evaluation to hold out entire sessions in addition to groups. Every sample
then needs a session ID, and each group must belong to exactly one session.
Group-only mode makes no claim of session independence.

`ProbeSplit(train_indices, eval_indices)` also supports externally defined
protocols. The validator enforces integer indices in bounds, non-empty sides,
no repeated indices, a complete partition of the observation table, exact
train/evaluation identity-class equality, group disjointness, and requested
session disjointness. At least two identity classes are required. Subset the
table and features first if a protocol uses a smaller cohort. The evaluator
validates **all** supplied splits before fitting any classifier.

## Splitting and reporting

Generation requires an explicit integer seed and `n_splits >= 2`. Every
identity needs at least `n_splits` distinct groups, or sessions in session
mode. Thus a two-way probe needs at least two independent units per identity;
many frames from one video do not meet this requirement. Generated folds use
each observation for evaluation exactly once and keep all frames of a unit
together.

Single-identity units are sorted, seeded-shuffled, and distributed round-robin
within identity, with a seeded fold offset. Assignment balances unit counts,
not frame counts. Shared units use `StratifiedGroupKFold` on unique
(unit, identity) pairs, never ordinary sample-level `StratifiedKFold`. Every
candidate is checked against the exact closed-set invariants. With complex
shared units, this heuristic can fail even when another valid assignment
exists; failure raises an error, without searching seeds, dropping identities,
or falling back to sample splitting. A separately designed protocol may then
be submitted for explicit validation. Record the scikit-learn/NumPy versions
alongside any later experiment; reproducibility is within a fixed environment.

The classifier retains the existing probe's train-fitted `StandardScaler`
and balanced logistic regression (`lbfgs`, default `max_iter=5000`). Features
must be finite and have shape `[n_observations, n_features]`. There is no
feature selection or hyperparameter tuning on evaluation observations.

Results report observation-level accuracy and macro-F1 per split, unweighted
means across splits, and sample standard deviations across splits. One
explicit holdout is allowed and has `None` for standard deviation. Explicit
repeated holdouts may reuse samples across *different* splits, but never
across the two sides of one split. `n_splits` applies only to generated folds.
Sample IDs, group IDs, identity counts, and (in session mode) session IDs are
returned for membership auditing. `uniform_chance_accuracy = 1 / n_identities`
describes uniform random guessing, not an empirical majority-class baseline.
These are sample-weighted within-fold scores, not video-aggregated scores or
confidence intervals; many correlated evaluation frames must not be treated
as independent observations for statistical inference.

The API checks declared provenance, not pixels or human identity. Different
IDs assigned to duplicated frames, incorrect video boundaries, or unverified
identity mappings cannot be detected here. Such metadata needs a separate
audit before making publication claims.

## Compatibility and current metadata limits

`cross_validated_identity_probe(features, subjects, ...)` remains available
with its historical signature, calculation, and return schema, but emits a
`FutureWarning` and is explicitly **legacy/exploratory**. It uses sample-level
stratification and cannot check video/session leakage. It is not the
recommended publication API.

The sole existing caller, `experiments/identity_leakage_nuaa.py`, now refuses
to run unless `--allow-legacy-sample-cv` is supplied. Opt-in retains the old
calculation and adds output markers identifying the exploratory protocol and
the absence of group-leakage checks. No source groups are guessed for NUAA.

| Existing metadata | Video-group-disjoint probing | Session-disjoint probing |
| --- | --- | --- |
| `FASSample` | Has `path`, `subject`, and optional `video_id`; sufficient to represent grouping when populated and audited. Integer `subject_id` is a training index, not independent identity provenance. | No session field; `camera` is a device attribute, not a session identifier. |
| MSU indexing/preprocessing | The [canonical MSU audit](MSU_PROTOCOL.md) verifies two bona-fide videos per client, one per camera. Legacy preprocessing preserves video stems but has a documented subject-ID normalization blocker; canonical frame integration is pending. | No verified session IDs; camera separation must not be called session separation. |
| NUAA loader | Leaves `video_id` and `camera` empty; current caller extracts only features and subjects. Not sufficient. | Filename session-like tokens are documented locally but not extracted or verified as independent recording keys. Not sufficient. |
| Canonical `SampleRecord` CSV | Has a unique `sample_id` and source `subject_id`, but no group/video field. A separate audited observation table is required. | No session field. |
| Existing external-manifest adapter | Can carry producer-supplied `video_id`, but its presence alone does not establish independent groups. No new dataset integration is performed here. | No explicit session field. |

NUAA cross-class human-identity correspondence also remains unresolved in
[the provenance audit](NUAA_SOURCE_MANIFEST.md). Its folder-derived
`subject_id` must not be promoted to verified human identity without evidence.
The subsequent MSU audit establishes metadata readiness for video-group
probing, subject to its preprocessing limitations. No scientific probe
experiment was run.
