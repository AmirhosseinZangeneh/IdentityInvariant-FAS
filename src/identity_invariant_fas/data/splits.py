"""Leakage-resistant subject-level splitting utilities."""

from __future__ import annotations

import random

from collections import defaultdict

from .manifest import FASSample


def get_subjects(samples: list[FASSample]) -> list[str]:
    """Return sorted unique subject identifiers."""

    return sorted(
        {
            sample.subject
            for sample in samples
        }
    )


def assert_disjoint(
    *subject_sets: set[str],
) -> None:
    """Ensure that provided subject sets do not overlap."""

    for index, left in enumerate(subject_sets):
        for right in subject_sets[index + 1 :]:
            overlap = left & right

            if overlap:
                raise ValueError(
                    f"Subject leakage detected: {sorted(overlap)}"
                )


def assign_subject_ids(
    samples: list[FASSample],
) -> list[FASSample]:
    """Assign deterministic integer IDs to subjects."""

    mapping = build_subject_mapping(samples)

    return [
        FASSample(
            path=sample.path,
            label=sample.label,
            subject=sample.subject,
            dataset=sample.dataset,
            split=sample.split,
            subject_id=mapping[sample.subject],
            video_id=sample.video_id,
            attack_type=sample.attack_type,
            camera=sample.camera,
            bbox=sample.bbox,
        )
        for sample in samples
    ]


def build_subject_mapping(
    samples: list[FASSample],
) -> dict[str, int]:
    """Create deterministic subject-to-index mapping."""

    return {
        subject: index
        for index, subject in enumerate(
            get_subjects(samples)
        )
    }


def filter_by_split(
    samples: list[FASSample],
    split: str,
) -> list[FASSample]:
    """Filter samples by official dataset split."""

    return [
        sample
        for sample in samples
        if sample.split == split
    ]


def subject_kfold_split(
    samples: list[FASSample],
    n_folds: int = 5,
    seed: int = 42,
) -> list[dict]:
    """
    Create deterministic subject-disjoint cross-validation folds.
    """

    subjects = get_subjects(samples)

    if n_folds < 2 or n_folds > len(subjects):
        raise ValueError(
            "n_folds must be between 2 and number of subjects."
        )

    rng = random.Random(seed)

    shuffled_subjects = subjects.copy()
    rng.shuffle(shuffled_subjects)

    buckets = [
        []
        for _ in range(n_folds)
    ]

    for index, subject in enumerate(shuffled_subjects):
        buckets[index % n_folds].append(subject)

    all_subjects = set(shuffled_subjects)

    folds = []

    for fold_index, bucket in enumerate(
        buckets,
        start=1,
    ):

        test_subjects = set(bucket)
        train_subjects = (
            all_subjects -
            test_subjects
        )

        assert_disjoint(
            train_subjects,
            test_subjects,
        )

        folds.append(
            {
                "fold": fold_index,
                "train_subjects": train_subjects,
                "test_subjects": test_subjects,
                "train": [
                    sample
                    for sample in samples
                    if sample.subject in train_subjects
                ],
                "test": [
                    sample
                    for sample in samples
                    if sample.subject in test_subjects
                ],
            }
        )

    return folds


def subject_train_val_split(
    samples: list[FASSample],
    val_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[list[FASSample], list[FASSample]]:
    """
    Split samples into subject-disjoint train and validation sets.
    """

    if not 0.0 < val_ratio < 1.0:
        raise ValueError(
            "val_ratio must be in (0, 1)."
        )

    subjects = get_subjects(samples)

    if len(subjects) < 2:
        raise ValueError(
            "At least two subjects are required."
        )

    rng = random.Random(seed)

    shuffled_subjects = subjects.copy()
    rng.shuffle(shuffled_subjects)

    n_val = max(
        1,
        int(round(len(subjects) * val_ratio)),
    )

    n_val = min(
        n_val,
        len(subjects) - 1,
    )

    val_subjects = set(
        shuffled_subjects[:n_val]
    )

    train_subjects = set(
        shuffled_subjects[n_val:]
    )

    assert_disjoint(
        train_subjects,
        val_subjects,
    )

    train = [
        sample
        for sample in samples
        if sample.subject in train_subjects
    ]

    val = [
        sample
        for sample in samples
        if sample.subject in val_subjects
    ]

    return train, val