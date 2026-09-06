"""Training dataset wrappers that attach fold-specific subject labels."""

from __future__ import annotations

from PIL import Image
from torch.utils.data import Dataset

from ..data.manifest import FASSample


class SubjectAwareDataset(Dataset):
    def __init__(
        self,
        samples: list[FASSample],
        subject_mapping: dict[str, int],
        transform=None,
    ) -> None:
        self.samples = samples
        self.subject_mapping = subject_mapping
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        with Image.open(sample.path) as image:
            image = image.convert("RGB")
            if self.transform is not None:
                image = self.transform(image)

        return {
            "image": image,
            "label": sample.label,
            "subject_label": self.subject_mapping.get(
                sample.subject,
                -1,
            ),
            "subject": sample.subject,
            "path": sample.path,
            "dataset": sample.dataset,
            "split": sample.split,
            "video_id": sample.video_id,
            "attack_type": sample.attack_type,
            "camera": sample.camera,
            "bbox": sample.bbox,
        }
