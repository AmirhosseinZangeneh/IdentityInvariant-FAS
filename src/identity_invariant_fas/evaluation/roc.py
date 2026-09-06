"""ROC, AUC, and equal-error-rate utilities."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def compute_roc(labels, scores) -> dict:
    fpr, tpr, thresholds = roc_curve(labels, scores)
    return {
        "fpr": fpr,
        "tpr": tpr,
        "thresholds": thresholds,
        "auc": float(roc_auc_score(labels, scores)),
    }


def compute_eer(labels, scores) -> dict[str, float]:
    fpr, tpr, thresholds = roc_curve(labels, scores)
    fnr = 1.0 - tpr
    index = int(np.nanargmin(np.abs(fpr - fnr)))
    return {
        "eer": float((fpr[index] + fnr[index]) / 2.0),
        "threshold": float(thresholds[index]),
    }
