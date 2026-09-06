"""Linear probing for subject-identity leakage in frozen representations."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def cross_validated_identity_probe(
    features,
    subjects,
    *,
    n_splits: int = 5,
    seed: int = 42,
    max_iter: int = 5000,
) -> dict:
    """Measure linearly decodable identity using held-out images.

    Each fold contains examples from the same identity classes in training and
    testing; otherwise closed-set identity accuracy is undefined. The split is
    sample-disjoint, not subject-disjoint. For frame-based datasets, a
    session/video-group split should be preferred when group identifiers exist.
    """
    features = np.asarray(features)
    subjects = np.asarray(subjects)
    if features.ndim != 2:
        raise ValueError("features must have shape [n_samples, n_features].")
    if len(features) != len(subjects):
        raise ValueError("features and subjects must contain the same number of samples.")

    unique_subjects, counts = np.unique(subjects, return_counts=True)
    if len(unique_subjects) < 2:
        raise ValueError("At least two subjects are required for identity probing.")
    if counts.min() < n_splits:
        raise ValueError("Every subject must have at least n_splits samples.")

    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_results = []

    for train_index, test_index in splitter.split(features, subjects):
        classifier = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=max_iter,
                        solver="lbfgs",
                        class_weight="balanced",
                    ),
                ),
            ]
        )
        classifier.fit(features[train_index], subjects[train_index])
        predictions = classifier.predict(features[test_index])

        fold_results.append(
            {
                "accuracy": float(accuracy_score(subjects[test_index], predictions)),
                "macro_f1": float(
                    f1_score(subjects[test_index], predictions, average="macro")
                ),
            }
        )

    accuracies = np.asarray([row["accuracy"] for row in fold_results])
    macro_f1 = np.asarray([row["macro_f1"] for row in fold_results])

    return {
        "n_subjects": int(len(unique_subjects)),
        "chance_accuracy": float(1.0 / len(unique_subjects)),
        "accuracy_mean": float(accuracies.mean()),
        "accuracy_std": float(accuracies.std(ddof=1)),
        "macro_f1_mean": float(macro_f1.mean()),
        "macro_f1_std": float(macro_f1.std(ddof=1)),
        "folds": fold_results,
    }
