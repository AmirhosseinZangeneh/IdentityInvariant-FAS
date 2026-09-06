"""Frame-to-video score aggregation for video-based PAD datasets."""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from .metrics import compute_classification_metrics


def aggregate_video_scores(
    labels,
    scores,
    video_ids,
    *,
    method: str = "mean",
    threshold: float = 0.5,
) -> dict:
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=float)
    video_ids = np.asarray(video_ids)

    grouped_scores: dict[str, list[float]] = defaultdict(list)
    grouped_labels: dict[str, set[int]] = defaultdict(set)

    for label, score, video_id in zip(labels, scores, video_ids):
        if not video_id:
            raise ValueError("Video-level evaluation requires non-empty video_id values.")
        grouped_scores[str(video_id)].append(float(score))
        grouped_labels[str(video_id)].add(int(label))

    video_labels = []
    video_scores = []
    ordered_ids = sorted(grouped_scores)

    for video_id in ordered_ids:
        if len(grouped_labels[video_id]) != 1:
            raise ValueError(f"Inconsistent labels within video {video_id}.")
        values = np.asarray(grouped_scores[video_id])
        if method == "mean":
            aggregated = float(values.mean())
        elif method == "median":
            aggregated = float(np.median(values))
        else:
            raise ValueError("method must be 'mean' or 'median'.")

        video_labels.append(next(iter(grouped_labels[video_id])))
        video_scores.append(aggregated)

    video_labels = np.asarray(video_labels)
    video_scores = np.asarray(video_scores)
    video_predictions = (video_scores >= threshold).astype(int)

    return {
        "aggregation": method,
        "threshold": float(threshold),
        "video_ids": ordered_ids,
        "labels": video_labels.tolist(),
        "scores": video_scores.tolist(),
        "predictions": video_predictions.tolist(),
        "metrics": compute_classification_metrics(
            video_labels,
            video_predictions,
            video_scores,
        ),
    }
