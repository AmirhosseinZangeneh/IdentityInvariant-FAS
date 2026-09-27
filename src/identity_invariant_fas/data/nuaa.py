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


def nuaa_protocol_metadata(protocol: str, *, source_partition: str | None = None) -> dict:
    """Fixed publication labels; folder names cannot certify human identity.

    This describes existing protocols without altering sample grouping.
    No caller-supplied identity-verification override is accepted.
    """
    profiles = {
        "custom_kfold": ("NUAA custom folder-token-disjoint k-fold", "folder_token_disjoint"),
        "official": ("NUAA official train/test", "official_source_lists"),
        "legacy_probe": ("legacy_exploratory_sample_level_cv", "sample_level_cv"),
    }
    if protocol not in profiles:
        raise ValueError("Unknown NUAA protocol; verified-human split claims are unsupported")
    if protocol == "official":
        if source_partition is not None:
            raise ValueError("Official NUAA evaluation uses train and test source lists")
        source_partition = "official_train_and_test"
    elif source_partition not in _FILE_SPECS and source_partition != "all":
        raise ValueError("Specify the NUAA source partition: train, test, or all")
    label, split_semantics = profiles[protocol]
    result = {
        "protocol": label,
        "source_partition": source_partition,
        "identity_semantics": "folder_token_unverified",
        "identity_provenance_status": "unresolved",
        "human_identity_verified": False,
        "split_semantics": split_semantics,
        "folder_token_grouping": "raw_token_shared_across_class_directories",
    }
    if protocol != "legacy_probe":
        result["validation_split_semantics"] = "project_defined_folder_token_disjoint"
    return result


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
    """Extract the raw folder token and filename, not a verified human ID."""

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

    The compatibility field ``subject`` is a raw folder token. Custom
    folder-token-disjoint evaluation uses the subject-level split utilities;
    it is not verified human-subject-disjoint evaluation.
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
