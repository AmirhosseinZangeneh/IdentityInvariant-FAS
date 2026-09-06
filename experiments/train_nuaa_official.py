"""Train/evaluate on the NUAA official train/test lists."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from identity_invariant_fas.data.nuaa import load_nuaa_samples
from identity_invariant_fas.data.splits import build_subject_mapping, subject_train_val_split
from identity_invariant_fas.training.checkpoint import load_checkpoint, save_checkpoint
from identity_invariant_fas.training.datasets import SubjectAwareDataset
from identity_invariant_fas.data.transforms import build_eval_transform, build_train_transform
from identity_invariant_fas.utils.config import load_config
from identity_invariant_fas.utils.io import write_json
from identity_invariant_fas.utils.logging import configure_logging
from identity_invariant_fas.utils.reproducibility import make_generator, seed_everything
from torch.utils.data import DataLoader

from _common import build_model, build_optimizer_and_trainer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    seed = int(config["training"].get("seed", 42))
    seed_everything(seed)

    root = config["dataset"]["root"]
    train_pool = load_nuaa_samples(root, partition="train")
    official_test = load_nuaa_samples(root, partition="test")
    train_samples, val_samples = subject_train_val_split(
        train_pool,
        val_ratio=float(config["training"].get("val_subject_ratio", 0.2)),
        seed=seed,
    )

    subject_mapping = build_subject_mapping(train_samples)
    image_size = int(config["dataset"].get("image_size", 160))
    batch_size = int(config["training"].get("batch_size", 32))
    num_workers = int(config["training"].get("num_workers", 0))

    train_loader = DataLoader(
        SubjectAwareDataset(
            train_samples,
            subject_mapping,
            build_train_transform(image_size),
        ),
        batch_size=batch_size,
        shuffle=True,
        generator=make_generator(seed),
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        SubjectAwareDataset(val_samples, subject_mapping, build_eval_transform(image_size)),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        SubjectAwareDataset(official_test, subject_mapping, build_eval_transform(image_size)),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config["model"]["name"], len(subject_mapping), config["model"]).to(device)
    optimizer, trainer = build_optimizer_and_trainer(model, config, device)

    output_dir = Path(config["training"].get("output_dir", "outputs/nuaa_official"))
    logger = configure_logging(output_dir / "training.log")
    checkpoint_path = output_dir / "best_model.pt"
    best_acer = float("inf")

    for epoch in range(1, int(config["training"].get("epochs", 30)) + 1):
        train_result = trainer.train_epoch(train_loader, epoch)
        validation = trainer.evaluate(val_loader)
        logger.info(
            "epoch=%d train_loss=%.6f train_acc=%.4f val_acer=%.4f",
            epoch,
            train_result.loss,
            train_result.accuracy,
            validation["acer"],
        )
        if validation["acer"] < best_acer:
            best_acer = validation["acer"]
            save_checkpoint(
                checkpoint_path,
                model,
                optimizer=optimizer,
                epoch=epoch,
                best_metric=best_acer,
                metadata={"protocol": "NUAA official train/test", "subject_mapping": subject_mapping},
            )

    load_checkpoint(checkpoint_path, model, map_location=device)
    result = trainer.evaluate(test_loader)
    write_json(result, output_dir / "official_test_results.json")
    print({key: value for key, value in result.items() if key != "predictions"})


if __name__ == "__main__":
    main()
