from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from identity_invariant_fas.data.nuaa import load_nuaa_samples
from identity_invariant_fas.data.splits import assign_subject_ids
from identity_invariant_fas.data.manifest import ManifestDataset


DATA_ROOT = Path("datasets/NUAA")


def main():

    samples = load_nuaa_samples(
        DATA_ROOT,
        partition="all",
    )

    samples = assign_subject_ids(samples)

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )

    dataset = ManifestDataset(
        samples=samples,
        transform=transform,
    )

    loader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        num_workers=0,
    )

    batch = next(iter(loader))

    print("Dataset size:")
    print(len(dataset))

    print("\nBatch keys:")
    print(batch.keys())

    print("\nImage:")
    print(batch["image"].shape)

    print("\nLabels:")
    print(batch["label"])

    print("\nSubject IDs:")
    print(batch["subject_id"])

    print("\nDatasets:")
    print(batch["dataset"])


if __name__ == "__main__":
    main()