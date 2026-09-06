"""NUAA dataset indexing for the face-detector-output format."""

from __future__ import annotations

import os

from pathlib import Path

from ..constants import (
    ATTACK_LABEL,
    BONA_FIDE_LABEL,
)

from .manifest import FASSample


_FILE_SPECS = {
    "train": [
        (
            "client_train_face.txt",
            "ClientFace",
            BONA_FIDE_LABEL,
        ),
        (
            "imposter_train_face.txt",
            "ImposterFace",
            ATTACK_LABEL,
        ),
    ],
    "test": [
        (
            "client_test_face.txt",
            "ClientFace",
            BONA_FIDE_LABEL,
        ),
        (
            "imposter_test_face.txt",
            "ImposterFace",
            ATTACK_LABEL,
        ),
    ],
}


def _parse_line(
    line: str,
) -> tuple[str, tuple[float, ...]]:
    """Parse NUAA annotation line."""

    parts = line.split()

    relative_path = parts[0]

    bbox = tuple(
        float(value)
        for value in parts[1:]
    )

    return relative_path, bbox


def _subject_and_name(
    listed_path: str,
) -> tuple[str, str]:
    """Extract subject identifier and filename."""

    normalized = (
        listed_path
        .replace("\\", os.sep)
        .replace("/", os.sep)
    )

    normalized_path = Path(normalized)

    return (
        normalized_path.parent.name,
        normalized_path.name,
    )


def load_nuaa_samples(
    root: str | Path,
    partition: str = "all",
    *,
    verify_files: bool = True,
) -> list[FASSample]:
    """
    Load NUAA samples from official protocol files.

    Supported partitions:
        train: official training lists
        test: official testing lists
        all: combine all official lists

    Subject-disjoint evaluation should be generated
    separately using subject-level split utilities.
    """

    root = Path(root)

    if partition not in {
        "train",
        "test",
        "all",
    }:
        raise ValueError(
            "partition must be one of: train, test, all."
        )

    partitions = (
        ["train", "test"]
        if partition == "all"
        else [partition]
    )

    samples: list[FASSample] = []

    for split_name in partitions:

        for (
            list_name,
            folder,
            label,
        ) in _FILE_SPECS[split_name]:

            list_path = root / list_name

            if not list_path.exists():
                raise FileNotFoundError(
                    f"Missing NUAA list file: {list_path}"
                )

            with list_path.open(
                "r",
                encoding="utf-8",
                errors="replace",
            ) as handle:

                for raw_line in handle:

                    raw_line = raw_line.strip()

                    if not raw_line:
                        continue

                    listed_path, bbox = _parse_line(
                        raw_line
                    )

                    subject, filename = (
                        _subject_and_name(
                            listed_path
                        )
                    )

                    image_path = (
                        root
                        / folder
                        / subject
                        / filename
                    )

                    if (
                        verify_files
                        and not image_path.exists()
                    ):
                        raise FileNotFoundError(
                            f"Missing NUAA image: {image_path}"
                        )

                    samples.append(
                        FASSample(
                            path=str(image_path),
                            label=label,
                            subject=subject,
                            dataset="NUAA",
                            split=split_name,
                            bbox=bbox,
                        )
                    )

    return samples