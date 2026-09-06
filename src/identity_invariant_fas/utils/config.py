"""YAML configuration loading and validation."""

from __future__ import annotations

from pathlib import Path

from typing import Any

import yaml



_REQUIRED_SECTIONS = (
    "dataset",
    "model",
    "training",
)



def load_config(
    path: str | Path,
) -> dict[str, Any]:

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Config file not found: {path}"
        )


    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:

        config = yaml.safe_load(handle)


    if not isinstance(config, dict):

        raise ValueError(
            "Configuration file must contain a YAML mapping."
        )


    for section in _REQUIRED_SECTIONS:

        if section not in config:

            raise ValueError(
                f"Missing required configuration section: {section}"
            )


    config.setdefault(
        "seed",
        42,
    )

    config.setdefault(
        "output_dir",
        "outputs",
    )


    training = config["training"]

    if not isinstance(training, dict):

        raise ValueError(
            "training section must be a mapping."
        )


    training.setdefault(
        "epochs",
        50,
    )

    training.setdefault(
        "batch_size",
        32,
    )

    training.setdefault(
        "lr",
        1e-4,
    )


    return config