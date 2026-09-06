from __future__ import annotations

from pathlib import Path

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

from identity_invariant_fas.data.nuaa import load_nuaa_samples
from identity_invariant_fas.data.splits import (
    build_subject_mapping,
    subject_kfold_split,
    subject_train_val_split,
)
from identity_invariant_fas.data.transforms import build_eval_transform, build_train_transform
from identity_invariant_fas.models import AblationECNN, IdentityInvariantECNN, PaperECNNClassifier
from identity_invariant_fas.training.datasets import SubjectAwareDataset
from identity_invariant_fas.training.engine import Trainer
from identity_invariant_fas.utils.reproducibility import make_generator


def build_model(name: str, num_subjects: int, model_config: dict):
    normalized = name.lower()
    if normalized == "paper_ecnn":
        return PaperECNNClassifier()
    if normalized == "ii_ecnn":
        return IdentityInvariantECNN(
            num_subjects=num_subjects,
            grl_lambda=float(model_config.get("grl_lambda", 0.05)),
        )
    if normalized == "ablation_ecnn":
        return AblationECNN(num_subjects=num_subjects)
    raise ValueError(f"Unknown model: {name}")


def make_fold_loaders(train_pool, test_samples, config: dict, fold_seed: int):
    image_size = int(config["dataset"].get("image_size", 160))
    val_ratio = float(config["training"].get("val_subject_ratio", 0.2))
    batch_size = int(config["training"].get("batch_size", 32))

    train_samples, val_samples = subject_train_val_split(
        train_pool,
        val_ratio=val_ratio,
        seed=fold_seed,
    )
    subject_mapping = build_subject_mapping(train_samples)

    train_dataset = SubjectAwareDataset(
        train_samples,
        subject_mapping,
        build_train_transform(
            image_size=image_size,
            horizontal_flip=bool(config["dataset"].get("horizontal_flip", False)),
        ),
    )
    val_dataset = SubjectAwareDataset(
        val_samples,
        subject_mapping,
        build_eval_transform(image_size=image_size),
    )
    test_dataset = SubjectAwareDataset(
        test_samples,
        subject_mapping,
        build_eval_transform(image_size=image_size),
    )

    generator = make_generator(fold_seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=int(config["training"].get("num_workers", 0)),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=int(config["training"].get("num_workers", 0)),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=int(config["training"].get("num_workers", 0)),
    )
    return train_loader, val_loader, test_loader, subject_mapping


def build_optimizer_and_trainer(model, config: dict, device):
    optimizer = Adam(model.parameters(), lr=float(config["training"].get("learning_rate", 1e-3)))
    trainer = Trainer(
        model,
        optimizer,
        device,
        subject_weight=float(config["training"].get("subject_loss_weight", 0.1)),
        warmup_epochs=int(config["training"].get("warmup_epochs", 5)),
    )
    return optimizer, trainer
