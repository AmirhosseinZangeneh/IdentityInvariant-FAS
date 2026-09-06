"""Representation quality metrics used alongside visual embeddings."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import silhouette_score


def binary_fisher_ratio(features, labels) -> float:
    features = np.asarray(features, dtype=float)
    labels = np.asarray(labels)
    classes = np.unique(labels)
    if len(classes) != 2:
        raise ValueError("binary_fisher_ratio requires exactly two classes.")

    first = features[labels == classes[0]]
    second = features[labels == classes[1]]
    mean_distance = np.sum((first.mean(axis=0) - second.mean(axis=0)) ** 2)
    within = np.sum(first.var(axis=0)) + np.sum(second.var(axis=0))
    return float(mean_distance / max(within, 1e-12))


def silhouette(features, labels) -> float:
    return float(silhouette_score(np.asarray(features), np.asarray(labels)))
