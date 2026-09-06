"""Paired statistical summaries for repeated/fold-level experiments."""

from __future__ import annotations

import itertools

import numpy as np
from scipy.stats import wilcoxon


def paired_cohens_dz(a, b) -> float:
    differences = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    if differences.size < 2:
        return float("nan")
    std = differences.std(ddof=1)
    return float(differences.mean() / std) if std > 0 else float("inf")


def exact_sign_flip_test(a, b) -> float:
    """Exact two-sided paired randomization test on the mean difference."""
    differences = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    if differences.shape[0] > 20:
        raise ValueError("Exact sign-flip test is limited to 20 pairs.")

    observed = abs(float(differences.mean()))
    permuted = []
    for signs in itertools.product((-1.0, 1.0), repeat=len(differences)):
        permuted.append(abs(float(np.mean(differences * np.asarray(signs)))))
    return float(np.mean(np.asarray(permuted) >= observed - 1e-15))


def paired_fold_comparison(a, b) -> dict:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("Paired inputs must have identical shape.")
    if a.ndim != 1:
        raise ValueError("Paired inputs must be one-dimensional.")

    try:
        wilcoxon_result = wilcoxon(a, b, alternative="two-sided", method="auto")
        wilcoxon_p = float(wilcoxon_result.pvalue)
    except ValueError:
        wilcoxon_p = float("nan")

    return {
        "n_pairs": int(len(a)),
        "mean_a": float(a.mean()),
        "mean_b": float(b.mean()),
        "mean_difference_a_minus_b": float((a - b).mean()),
        "cohens_dz": paired_cohens_dz(a, b),
        "wilcoxon_p": wilcoxon_p,
        "exact_sign_flip_p": exact_sign_flip_test(a, b) if len(a) <= 20 else float("nan"),
    }
