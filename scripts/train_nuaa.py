"""Train NUAA FAS models."""

from __future__ import annotations

import argparse
import json

from pathlib import Path

import torch

from torch.utils.data import DataLoader


from identity_invariant_fas.utils.config import load_config

from identity_invariant_fas.utils.reproducibility import (
    seed_everything,
    make_generator,
)

from identity_invariant_fas.data.nuaa import load_nuaa_samples

from identity_invariant_fas.data.splits import (
    subject_train_val_split,
    build_subject_mapping,
)

from identity_invariant_fas.data.transforms import (
    build_train_transform,
    build_eval_transform,
)

from identity_invariant_fas.training.datasets import SubjectAwareDataset

from identity_invariant_fas.training.engine import Trainer

from identity_invariant_fas.training.checkpoint import save_checkpoint

from identity_invariant_fas.models import (
    IdentityInvariantECNN,
    PaperECNNClassifier,
    AblationECNN,
)



def build_model(config: dict):

    model_cfg = config["model"]

    name = model_cfg["name"]


    if name == "IdentityInvariantECNN":

        return IdentityInvariantECNN(
            num_subjects=model_cfg["num_subjects"],
            grl_lambda=model_cfg.get(
                "grl_lambda",
                0.05,
            ),
            subject_hidden_dim=model_cfg.get(
                "subject_hidden_dim",
                128,
            ),
        )


    if name == "PaperECNNClassifier":

        return PaperECNNClassifier()


    if name == "AblationECNN":

        return AblationECNN(
            num_subjects=model_cfg["num_subjects"]
        )


    raise ValueError(
        f"Unknown model: {name}"
    )



def main(config_path: str):

    config = load_config(config_path)


    seed_everything(
        config.get(
            "seed",
            42
        )
    )


    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


    print(
        "Device:",
        device
    )



    samples = load_nuaa_samples(
        config["dataset"]["root"],
        partition="all",
    )


    train_samples, val_samples = subject_train_val_split(
        samples,
        val_ratio=0.2,
        seed=config["seed"],
    )


    mapping = build_subject_mapping(
        samples
    )



    train_dataset = SubjectAwareDataset(
        samples=train_samples,
        subject_mapping=mapping,
        transform=build_train_transform(
            config["dataset"]["image_size"]
        ),
    )


    val_dataset = SubjectAwareDataset(
        samples=val_samples,
        subject_mapping=mapping,
        transform=build_eval_transform(
            config["dataset"]["image_size"]
        ),
    )



    generator = make_generator(
        config["seed"]
    )



    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        generator=generator,
        num_workers=config["dataset"]["num_workers"],
    )


    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["dataset"]["num_workers"],
    )



    model = build_model(
        config
    ).to(device)



    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"]["lr"],
        weight_decay=config["training"]["weight_decay"],
    )


    scheduler = None

    if config["training"].get("scheduler") == "cosine":

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config["training"]["epochs"],
        )



    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        device=device,
        subject_weight=config["training"]["subject_weight"],
        warmup_epochs=config["training"]["warmup_epochs"],
        total_epochs=config["training"]["epochs"],
    )



    output_dir = Path(
        config["output_dir"]
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


    best_acer = float("inf")

    patience = 20

    patience_counter = 0


    history = []



    for epoch in range(
        config["training"]["epochs"]
    ):


        train_result = trainer.train_epoch(
            train_loader,
            epoch,
        )


        val_metrics = trainer.evaluate(
            val_loader
        )


        acer = val_metrics.get(
            "acer",
            float("inf")
        )


        if scheduler is not None:

            scheduler.step()



        record = {

            "epoch": epoch + 1,

            "train_loss": train_result.loss,

            "train_accuracy": train_result.accuracy,

            "val_acer": acer,

            "grl_lambda": train_result.grl_lambda,

        }


        history.append(
            record
        )



        print(
            f"Epoch {epoch+1}: "
            f"loss={train_result.loss:.4f} "
            f"acc={train_result.accuracy:.4f} "
            f"ACER={acer:.4f}"
        )



        if acer < best_acer:


            best_acer = acer

            patience_counter = 0


            save_checkpoint(

                output_dir / "best.pt",

                model=model,

                optimizer=optimizer,

                epoch=epoch,

                best_metric=best_acer,

                metadata={

                    "metrics": val_metrics,

                    "config": config,

                },

            )


            print(
                "Best model updated."
            )



        else:

            patience_counter += 1



        if patience_counter >= patience:

            print(
                "Early stopping triggered."
            )

            break



    with open(
        output_dir / "history.json",
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            history,
            file,
            indent=2,
        )



if __name__ == "__main__":


    parser = argparse.ArgumentParser()


    parser.add_argument(
        "--config",
        required=True,
    )


    args = parser.parse_args()


    main(
        args.config
    )