"""Train a configured model under the subject-disjoint NUAA protocol."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import torch

from identity_invariant_fas.data.nuaa import load_nuaa_samples
from identity_invariant_fas.data.splits import subject_kfold_split
from identity_invariant_fas.training.checkpoint import load_checkpoint, save_checkpoint
from identity_invariant_fas.utils.config import load_config
from identity_invariant_fas.utils.io import write_json
from identity_invariant_fas.utils.logging import configure_logging
from identity_invariant_fas.utils.reproducibility import seed_everything

from _common import build_model, build_optimizer_and_trainer, make_fold_loaders


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    seed = int(config["training"].get("seed", 42))
    seed_everything(seed)

    output_root = Path(config["training"].get("output_dir", "outputs/nuaa_kfold"))
    logger = configure_logging(output_root / "training.log")

    partition = config["dataset"].get("partition", "test")
    samples = load_nuaa_samples(config["dataset"]["root"], partition=partition)
    folds = subject_kfold_split(
        samples,
        n_folds=int(config["training"].get("folds", 5)),
        seed=seed,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fold_results = []

    for fold in folds:
        fold_id = int(fold["fold"])
        fold_seed = seed + fold_id
        seed_everything(fold_seed)

        train_loader, val_loader, test_loader, subject_mapping = make_fold_loaders(
            fold["train"],
            fold["test"],
            config,
            fold_seed,
        )
        model = build_model(config["model"]["name"], len(subject_mapping), config["model"]).to(device)
        optimizer, trainer = build_optimizer_and_trainer(model, config, device)

        fold_dir = output_root / f"fold_{fold_id}"
        checkpoint_path = fold_dir / "best_model.pt"
        best_acer = float("inf")

        for epoch in range(1, int(config["training"].get("epochs", 30)) + 1):
            train_result = trainer.train_epoch(train_loader, epoch)
            validation = trainer.evaluate(val_loader)

            logger.info(
                "fold=%d epoch=%d train_loss=%.6f train_acc=%.4f val_acer=%.4f",
                fold_id,
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
                    metadata={
                        "model": config["model"],
                        "dataset": config["dataset"],
                        "fold": fold_id,
                        "subject_mapping": subject_mapping,
                    },
                )

        load_checkpoint(checkpoint_path, model, map_location=device)
        test_result = trainer.evaluate(test_loader)
        write_json(test_result["predictions"], fold_dir / "predictions.json")

        summary_row = {
            "fold": fold_id,
            **{key: value for key, value in test_result.items() if key != "predictions"},
        }
        fold_results.append(summary_row)
        logger.info("fold=%d test=%s", fold_id, summary_row)

    summary = {
        "model": config["model"]["name"],
        "protocol": "NUAA custom subject-disjoint k-fold",
        "partition": partition,
        "acer_mean": float(np.mean([row["acer"] for row in fold_results])),
        "acer_std": float(np.std([row["acer"] for row in fold_results], ddof=1)),
        "accuracy_mean": float(np.mean([row["accuracy"] for row in fold_results])),
        "accuracy_std": float(np.std([row["accuracy"] for row in fold_results], ddof=1)),
        "fold_results": fold_results,
    }
    write_json(summary, output_root / "summary.json")
    logger.info("final=%s", summary)


if __name__ == "__main__":
    main()
