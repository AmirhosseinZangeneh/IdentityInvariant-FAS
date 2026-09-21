"""Group-aware closed-set identity probing of frozen, externally supplied features."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class ProbeObservation:
    """One feature row; all IDs are explicit, globally scoped, opaque strings."""

    sample_id: str
    identity_id: str
    group_id: str
    session_id: str | None = None


@dataclass(frozen=True)
class ProbeSplit:
    """Indices into a single aligned feature/observation table."""

    train_indices: tuple[int, ...]
    eval_indices: tuple[int, ...]


def _validate_observations(observations: Sequence[ProbeObservation], session_disjoint: bool) -> None:
    if not observations:
        raise ValueError("At least two identities and non-empty observations are required.")
    seen = set()
    group_sessions = {}
    for row in observations:
        for name in ("sample_id", "identity_id", "group_id"):
            value = getattr(row, name)
            if not isinstance(value, str) or not value or value != value.strip():
                raise ValueError(f"{name} must be a non-empty string without surrounding whitespace.")
        if row.sample_id in seen:
            raise ValueError(f"Duplicate sample_id: {row.sample_id}")
        seen.add(row.sample_id)
        if row.session_id is not None and (
            not isinstance(row.session_id, str)
            or not row.session_id
            or row.session_id != row.session_id.strip()
        ):
            raise ValueError("session_id must be a non-empty string or None.")
        if session_disjoint:
            if row.session_id is None:
                raise ValueError("Session-disjoint probing requires session metadata for every sample.")
            previous = group_sessions.setdefault(row.group_id, row.session_id)
            if previous != row.session_id:
                raise ValueError(f"Group {row.group_id} spans multiple sessions; metadata is inconsistent.")
    if len({row.identity_id for row in observations}) < 2:
        raise ValueError("At least two identities are required for identity probing.")


def _indices(values: Sequence[int], n: int) -> set[int]:
    if not len(values):
        raise ValueError("Probe train and evaluation must both be non-empty.")
    if any(isinstance(i, (bool, np.bool_)) or not isinstance(i, Integral) for i in values):
        raise ValueError("Probe indices must be integers.")
    result = set(values)
    if len(result) != len(values) or any(i < 0 or i >= n for i in result):
        raise ValueError("Probe indices must be unique and within observation bounds.")
    return result


def validate_closed_set_probe_split(
    observations: Sequence[ProbeObservation],
    split: ProbeSplit,
    *,
    session_disjoint: bool = False,
) -> None:
    """Reject sample/group leakage, unequal class sets, and incomplete partitions.

    Both sides must contain every identity in the supplied observation table.
    To select a cohort, subset that table and its feature rows before splitting.
    """
    _validate_observations(observations, session_disjoint)
    train = _indices(split.train_indices, len(observations))
    evaluation = _indices(split.eval_indices, len(observations))
    if train & evaluation:
        raise ValueError("Sample overlap between probe train and evaluation.")
    if train | evaluation != set(range(len(observations))):
        raise ValueError("Probe split must partition all supplied observations.")
    train_ids = {observations[i].identity_id for i in train}
    eval_ids = {observations[i].identity_id for i in evaluation}
    if eval_ids - train_ids:
        raise ValueError(f"Unseen evaluation identities: {sorted(eval_ids - train_ids)}")
    if train_ids - eval_ids:
        raise ValueError(f"Training identities absent from evaluation: {sorted(train_ids - eval_ids)}")
    for field in ("group_id", "session_id") if session_disjoint else ("group_id",):
        overlap = {getattr(observations[i], field) for i in train} & {
            getattr(observations[i], field) for i in evaluation
        }
        if overlap:
            raise ValueError(f"{field} overlap between probe train and evaluation: {sorted(overlap)}")


def make_closed_set_probe_splits(
    observations: Sequence[ProbeObservation],
    *,
    seed: int,
    n_splits: int = 5,
    session_disjoint: bool = False,
) -> tuple[ProbeSplit, ...]:
    """Generate validated group-disjoint folds using an explicit seed.

    Each identity needs at least n_splits groups (sessions in session mode).
    Single-identity units are distributed per identity; shared units use a
    stratified-group candidate which is rejected if exact class coverage fails.
    No fallback to sample splitting or automatic seed search is performed.
    """
    _validate_observations(observations, session_disjoint)
    if isinstance(n_splits, bool) or not isinstance(n_splits, Integral) or n_splits < 2:
        raise ValueError("n_splits must be an integer of at least two.")
    if isinstance(seed, bool) or not isinstance(seed, Integral) or not 0 <= seed < 2**32:
        raise ValueError("seed must be an explicit integer in [0, 2**32).")
    field = "session_id" if session_disjoint else "group_id"
    units: dict[str, set[str]] = {}
    for row in observations:
        units.setdefault(getattr(row, field), set()).add(row.identity_id)
    identities = sorted({row.identity_id for row in observations})
    for identity in identities:
        count = sum(identity in members for members in units.values())
        if count < n_splits:
            raise ValueError(
                f"Identity {identity!r} has {count} separable {field} values; "
                f"at least n_splits={n_splits} are required."
            )

    unit_folds: dict[str, int] = {}
    if all(len(members) == 1 for members in units.values()):
        rng = np.random.RandomState(seed)
        for identity in identities:
            ordered = sorted(unit for unit, members in units.items() if identity in members)
            rng.shuffle(ordered)
            offset = int(rng.randint(n_splits))
            unit_folds.update((unit, (i + offset) % n_splits) for i, unit in enumerate(ordered))
    else:
        # One row per (unit, identity), so many frames do not dominate assignment.
        pairs = [(unit, identity) for unit in sorted(units) for identity in sorted(units[unit])]
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for fold, (_, evaluation) in enumerate(splitter.split(
            np.zeros(len(pairs)), [p[1] for p in pairs], groups=[p[0] for p in pairs]
        )):
            for i in evaluation:
                unit_folds[pairs[i][0]] = fold

    splits = []
    for fold in range(n_splits):
        split = ProbeSplit(
            tuple(i for i, row in enumerate(observations) if unit_folds[getattr(row, field)] != fold),
            tuple(i for i, row in enumerate(observations) if unit_folds[getattr(row, field)] == fold),
        )
        try:
            validate_closed_set_probe_split(observations, split, session_disjoint=session_disjoint)
        except ValueError as error:
            raise ValueError(
                f"Cannot produce valid closed-set fold {fold}: {error} "
                "Use an explicitly validated protocol; no sample-level fallback is allowed."
            ) from error
        splits.append(split)
    return tuple(splits)


def evaluate_closed_set_identity_probe(
    features,
    observations: Sequence[ProbeObservation],
    *,
    seed: int,
    n_splits: int = 5,
    session_disjoint: bool = False,
    max_iter: int = 5000,
    splits: Sequence[ProbeSplit] | None = None,
) -> dict:
    """Fit train-only scaling and linear classifiers after validating every split.

    Explicit splits may be a single holdout or repeated holdouts. Each must be
    a complete partition; repeated evaluation across different splits is allowed.
    n_splits applies only when generating folds. No encoder is trained here.
    """
    observations = tuple(observations)
    _validate_observations(observations, session_disjoint)
    if isinstance(seed, bool) or not isinstance(seed, Integral) or not 0 <= seed < 2**32:
        raise ValueError("seed must be an explicit integer in [0, 2**32).")
    features = np.asarray(features, dtype=float)
    if features.ndim != 2 or features.shape[0] != len(observations) or features.shape[1] == 0:
        raise ValueError("features must have shape [n_observations, n_features] with nonzero width.")
    if not np.isfinite(features).all():
        raise ValueError("features must contain only finite values.")
    if splits is None:
        splits = make_closed_set_probe_splits(
            observations, seed=seed, n_splits=n_splits, session_disjoint=session_disjoint
        )
    splits = tuple(splits)
    if not splits:
        raise ValueError("At least one probe split is required.")
    # Validate all folds before fitting any classifier.
    for split in splits:
        validate_closed_set_probe_split(observations, split, session_disjoint=session_disjoint)
    labels = np.asarray([row.identity_id for row in observations])
    classes = sorted(set(labels.tolist()))
    fold_results = []
    for split in splits:
        train, evaluation = list(split.train_indices), list(split.eval_indices)
        classifier = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(
                max_iter=max_iter, solver="lbfgs", class_weight="balanced", random_state=seed
            )),
        ])
        classifier.fit(features[train], labels[train])
        predictions = classifier.predict(features[evaluation])
        result = {
            "accuracy": float(accuracy_score(labels[evaluation], predictions)),
            "macro_f1": float(f1_score(labels[evaluation], predictions, labels=classes, average="macro")),
        }
        for name, indices in (("train", train), ("eval", evaluation)):
            result[name + "_sample_ids"] = [observations[i].sample_id for i in indices]
            result[name + "_group_ids"] = sorted({observations[i].group_id for i in indices})
            result[name + "_identity_counts"] = {
                identity: sum(observations[i].identity_id == identity for i in indices)
                for identity in classes
            }
            if session_disjoint:
                result[name + "_session_ids"] = sorted({observations[i].session_id for i in indices})
        fold_results.append(result)
    summary = {
        "protocol": "closed_set_identity_probe",
        "grouping": "session" if session_disjoint else "group",
        "seed": int(seed),
        "n_identities": len(classes),
        "identity_classes": classes,
        "uniform_chance_accuracy": 1.0 / len(classes),
        "folds": fold_results,
    }
    for metric in ("accuracy", "macro_f1"):
        values = np.asarray([fold[metric] for fold in fold_results])
        summary[metric + "_mean"] = float(values.mean())
        summary[metric + "_std"] = float(values.std(ddof=1)) if len(values) > 1 else None
    return summary
