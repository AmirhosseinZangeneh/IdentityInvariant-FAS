from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from identity_invariant_fas.data.nuaa import load_nuaa_samples
from identity_invariant_fas.data.splits import assign_subject_ids
from identity_invariant_fas.data.manifest import ManifestDataset

from identity_invariant_fas.models.paper_ecnn import PaperECNNClassifier
from identity_invariant_fas.models.identity_invariant_ecnn import IdentityInvariantECNN
from identity_invariant_fas.models.ablation_ecnn import AblationECNN


DATA_ROOT = Path("datasets/NUAA")


def get_batch():

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
        samples,
        transform=transform,
    )

    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0,
    )

    return next(iter(loader))


def test_model(name, model, batch):

    print("\n" + "=" * 50)
    print(name)
    print("=" * 50)

    model.train()

    images = batch["image"]
    labels = batch["label"]
    subject_ids = batch["subject_id"]

    outputs = model(
        images
    )

    if isinstance(outputs, tuple):

        print("Output type: tuple")

        for index, output in enumerate(outputs):
            print(
                f"Output {index}:",
                output.shape
            )

        loss = (
            outputs[0].mean()
            +
            outputs[1].mean()
        )

    else:

        print(
            "Output:",
            outputs.shape
        )

        loss = outputs.mean()


    loss.backward()

    print("Backward: OK")


def main():

    batch = get_batch()

    print(
        "Input batch:",
        batch["image"].shape
    )

    models = [

        (
            "PaperECNNClassifier",
            PaperECNNClassifier()
        ),

        (
            "IdentityInvariantECNN",
            IdentityInvariantECNN(
                num_subjects=16
            )
        ),

        (
            "AblationECNN",
            AblationECNN(
                num_subjects=16
            )
        ),
    ]


    for name, model in models:

        test_model(
            name,
            model,
            batch
        )


if __name__ == "__main__":
    main()