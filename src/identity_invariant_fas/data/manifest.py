"""Manifest-based dataset primitives shared by all FAS datasets."""

from __future__ import annotations

import csv

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Iterable

from PIL import Image
from torch.utils.data import Dataset


@dataclass(frozen=True)
class FASSample:
    """Normalized sample representation shared across FAS datasets."""

    path: str
    label: int
    subject: str
    dataset: str

    split: str = ""

    subject_id: int = -1

    video_id: str = ""
    attack_type: str = ""
    camera: str = ""

    bbox: tuple[float, ...] | None = None


class ManifestDataset(Dataset):
    """Load RGB images from normalized FAS sample records."""

    def __init__(
        self,
        samples: list[FASSample],
        transform=None,
    ) -> None:
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        sample = self.samples[index]

        with Image.open(sample.path) as image:
            image = image.convert("RGB")

            if self.transform is not None:
                image = self.transform(image)

        return {
            "image": image,
            "label": sample.label,
            "subject": sample.subject,
            "subject_id": sample.subject_id,
            "path": sample.path,
            "dataset": sample.dataset,
            "split": sample.split,
            "video_id": sample.video_id,
            "attack_type": sample.attack_type,
            "camera": sample.camera,
            "bbox": sample.bbox,
        }


_FIELDNAMES = [
    field.name
    for field in fields(FASSample)
]


def write_manifest(
    samples: Iterable[FASSample],
    path: str | Path,
) -> None:
    """Write sample records to a CSV manifest file."""

    path = Path(path)
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:

        writer = csv.DictWriter(
            handle,
            fieldnames=_FIELDNAMES,
        )

        writer.writeheader()

        for sample in samples:
            row = asdict(sample)

            if row["bbox"] is not None:
                row["bbox"] = ",".join(
                    map(str, row["bbox"])
                )

            writer.writerow(row)


def read_manifest(
    path: str | Path,
) -> list[FASSample]:
    """Read sample records from a CSV manifest file."""

    path = Path(path)

    samples: list[FASSample] = []

    with path.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as handle:

        for row in csv.DictReader(handle):

            row["label"] = int(row["label"])
            row["subject_id"] = int(
                row.get("subject_id", -1)
            )

            bbox = row.get("bbox", "")

            if bbox:
                row["bbox"] = tuple(
                    float(value)
                    for value in bbox.split(",")
                )
            else:
                row["bbox"] = None

            samples.append(
                FASSample(
                    path=row["path"],
                    label=int(row["label"]),
                    subject=row["subject"],
                    dataset=row["dataset"],
                    split=row.get("split", ""),
                    subject_id=int(row.get("subject_id", -1)),
                    video_id=row.get("video_id", ""),
                    attack_type=row.get("attack_type", ""),
                    camera=row.get("camera", ""),
                    bbox=row["bbox"],
                )
            )

    return samples