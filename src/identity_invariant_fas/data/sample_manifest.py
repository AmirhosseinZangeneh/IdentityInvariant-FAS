"""Validated sample metadata, independent of training and protocol assignment."""

from __future__ import annotations

import csv
from dataclasses import dataclass, fields
from pathlib import Path, PurePosixPath, PureWindowsPath

from ..constants import ATTACK_LABEL, BONA_FIDE_LABEL


@dataclass(frozen=True)
class SampleRecord:
    """One sample's source metadata; subject_id is an opaque source identifier."""

    sample_id: str
    filepath: str
    subject_id: str
    class_label: int
    attack_type: str
    split: str
    preprocessing_version: str


REQUIRED_COLUMNS = tuple(field.name for field in fields(SampleRecord))
UNASSIGNED_SPLIT = "unassigned"


def load_sample_manifest(path: str | Path) -> list[SampleRecord]:
    """Read UTF-8 CSV metadata in file order, rejecting invalid rows.

    Filepaths are portable paths relative to a caller-selected dataset root.
    Both separator styles are accepted and returned with forward slashes.
    Images are neither opened nor checked. See docs/SAMPLE_MANIFEST.md for
    source split, unknown metadata, and legacy compatibility semantics.
    """
    records: list[SampleRecord] = []
    seen: set[str] = set()
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, strict=True)
        columns = reader.fieldnames or []
        missing = [name for name in REQUIRED_COLUMNS if name not in columns]
        if missing:
            raise ValueError(f"Missing required columns: {', '.join(missing)}")
        if len(columns) != len(set(columns)):
            raise ValueError("Duplicate column names in sample manifest")

        for row in reader:
            context = f"Manifest line {reader.line_num}"
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"{context}: row length does not match header")
            for name in ("sample_id", "filepath", "subject_id", "split"):
                if not row[name].strip():
                    raise ValueError(f"{context}: {name} must be non-empty")
            for name in ("sample_id", "subject_id", "split"):
                if row[name] != row[name].strip():
                    raise ValueError(f"{context}: {name} must not have leading or trailing whitespace")

            sample_id = row["sample_id"]
            if sample_id in seen:
                raise ValueError(f"{context}: duplicate sample_id {sample_id!r}")

            label = row["class_label"]
            allowed = {str(BONA_FIDE_LABEL), str(ATTACK_LABEL)}
            if label not in allowed:
                expected = " or ".join(sorted(allowed))
                raise ValueError(f"{context}: invalid class_label {label!r}; expected {expected}")

            filepath = row["filepath"].replace("\\", "/")
            portable = PurePosixPath(filepath)
            windows = PureWindowsPath(filepath)
            if (
                portable.is_absolute()
                or windows.drive
                or ".." in portable.parts
                or not portable.parts
                or ":" in filepath
                or "\x00" in filepath
            ):
                raise ValueError(f"{context}: filepath must be a portable dataset-relative path")

            records.append(
                SampleRecord(
                    sample_id=sample_id,
                    filepath=portable.as_posix(),
                    subject_id=row["subject_id"],
                    class_label=int(label),
                    attack_type=row["attack_type"],
                    split=row["split"],
                    preprocessing_version=row["preprocessing_version"],
                )
            )
            seen.add(sample_id)
    return records
