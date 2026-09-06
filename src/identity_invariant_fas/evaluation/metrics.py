"""ISO-style binary PAD metrics using 0=bona fide and 1=attack."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score


def confusion_counts(y_true, y_pred) -> tuple[int, int, int, int]:
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()
    return int(tn), int(fp), int(fn), int(tp)


def compute_apcer_bpcer(y_true, y_pred) -> tuple[float, float]:
    """Return APCER and BPCER for the fixed label convention.

    Attack samples have label 1, so attack errors are false negatives.
    Bona fide samples have label 0, so bona fide errors are false positives.
    """
    tn, fp, fn, tp = confusion_counts(y_true, y_pred)
    apcer = fn / (fn + tp) if (fn + tp) else float("nan")
    bpcer = fp / (fp + tn) if (fp + tn) else float("nan")
    return float(apcer), float(bpcer)


def compute_acer(y_true, y_pred) -> float:
    apcer, bpcer = compute_apcer_bpcer(y_true, y_pred)
    return float(np.nanmean([apcer, bpcer]))


def compute_classification_metrics(y_true, y_pred, attack_scores=None) -> dict:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    tn, fp, fn, tp = confusion_counts(y_true, y_pred)
    apcer, bpcer = compute_apcer_bpcer(y_true, y_pred)
    result = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "apcer": apcer,
        "bpcer": bpcer,
        "acer": float(np.nanmean([apcer, bpcer])),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }

    if attack_scores is not None:
        result["auc"] = float(roc_auc_score(y_true, np.asarray(attack_scores)))

    return result
