"""Paired stratified bootstrap utilities for sample-level PAD comparisons."""

from __future__ import annotations

import numpy as np

from .metrics import compute_classification_metrics


def stratified_paired_bootstrap(
    labels,
    predictions_by_model: dict[str, np.ndarray],
    *,
    repeats: int = 5000,
    seed: int = 42,
) -> dict:
    labels = np.asarray(labels)
    real_indices = np.flatnonzero(labels == 0)
    attack_indices = np.flatnonzero(labels == 1)
    if len(real_indices) == 0 or len(attack_indices) == 0:
        raise ValueError("Both bona fide and attack samples are required.")

    rng = np.random.default_rng(seed)
    samples = {name: np.asarray(predictions) for name, predictions in predictions_by_model.items()}
    distributions = {name: [] for name in samples}

    for _ in range(repeats):
        chosen = np.concatenate(
            [
                rng.choice(real_indices, size=len(real_indices), replace=True),
                rng.choice(attack_indices, size=len(attack_indices), replace=True),
            ]
        )
        y = labels[chosen]
        for name, predictions in samples.items():
            metrics = compute_classification_metrics(y, predictions[chosen])
            distributions[name].append(metrics["acer"])

    return {
        name: {
            "acer_mean": float(np.mean(values)),
            "acer_ci95": [
                float(np.percentile(values, 2.5)),
                float(np.percentile(values, 97.5)),
            ],
        }
        for name, values in distributions.items()
    }
