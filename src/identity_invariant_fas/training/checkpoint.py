"""Checkpoint I/O utilities with reproducibility support."""

from __future__ import annotations

import os
import random

from pathlib import Path

from typing import Any

import numpy as np

import torch

from torch import nn


def normalize_legacy_state_dict(
    state_dict: dict[str, Any],
) -> dict[str, Any]:
    """Normalize known historical checkpoint key names."""

    normalized: dict[str, Any] = {}

    for key, value in state_dict.items():

        if key.startswith("module."):
            key = key[len("module."):]

        if key.startswith("fas_classifier."):
            key = (
                "spoof_classifier."
                +
                key[len("fas_classifier."):]
            )

        normalized[key] = value

    return normalized


def get_rng_state() -> dict[str, Any]:
    """Capture random states for exact experiment recovery."""

    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }

    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()

    return state


def restore_rng_state(
    state: dict[str, Any],
) -> None:
    """Restore random states from checkpoint."""

    if "python" in state:
        random.setstate(
            state["python"]
        )

    if "numpy" in state:
        np.random.set_state(
            state["numpy"]
        )

    if "torch" in state:
        torch.set_rng_state(
            state["torch"]
        )

    if (
        "cuda" in state
        and torch.cuda.is_available()
    ):
        torch.cuda.set_rng_state_all(
            state["cuda"]
        )


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    *,
    optimizer=None,
    scheduler=None,
    epoch: int | None = None,
    best_metric: float | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:

    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


    payload: dict[str, Any] = {

        "model_state_dict":
            model.state_dict(),

        "epoch":
            epoch,

        "best_metric":
            best_metric,

        "metadata":
            metadata or {},

        "rng_state":
            get_rng_state(),
    }


    if optimizer is not None:

        payload[
            "optimizer_state_dict"
        ] = optimizer.state_dict()


    if scheduler is not None:

        payload[
            "scheduler_state_dict"
        ] = scheduler.state_dict()


    temporary = path.with_suffix(
        path.suffix + ".tmp"
    )


    torch.save(
        payload,
        temporary,
    )


    os.replace(
        temporary,
        path,
    )


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    *,
    optimizer=None,
    scheduler=None,
    map_location: str | torch.device = "cpu",
    strict: bool = True,
    restore_rng: bool = True,
) -> dict[str, Any]:

    payload = torch.load(
        path,
        map_location=map_location,
        weights_only=False,
    )


    if (
        isinstance(payload, dict)
        and
        "model_state_dict" in payload
    ):

        state_dict = payload[
            "model_state_dict"
        ]

        metadata = payload

    else:

        state_dict = payload

        metadata = {
            "legacy_state_dict_only": True
        }


    state_dict = normalize_legacy_state_dict(
        state_dict
    )


    model.load_state_dict(
        state_dict,
        strict=strict,
    )


    if (
        optimizer is not None
        and
        "optimizer_state_dict" in metadata
    ):

        optimizer.load_state_dict(
            metadata[
                "optimizer_state_dict"
            ]
        )


    if (
        scheduler is not None
        and
        "scheduler_state_dict" in metadata
    ):

        scheduler.load_state_dict(
            metadata[
                "scheduler_state_dict"
            ]
        )


    rng_state = metadata.get("rng_state")

    if (
        restore_rng
        and
        isinstance(rng_state, dict)
    ):

        restore_rng_state(
            rng_state
        )


    return metadata