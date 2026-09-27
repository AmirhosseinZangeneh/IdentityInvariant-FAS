"""Run a controlled GRL-lambda sweep with an otherwise identical protocol."""

from __future__ import annotations

import argparse
import copy
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml
from identity_invariant_fas.training.engine import GRLConfiguration


def build_sweep_configs(config: dict) -> list[dict]:
    """Only the target and output location vary; never mutate the source config."""
    base = copy.deepcopy(config)
    values = [float(value) for value in base.pop("grl_values")]
    if not values or len(set(values)) != len(values):
        raise ValueError("GRL sweep requires distinct target coefficients")
    if base["model"]["name"] != "ii_ecnn":
        raise ValueError("GRL sweep requires ii_ecnn")
    training = base["training"]
    schedule = training["grl_schedule"]  # The publication sweep must be explicit.
    runs = []
    for target in values:
        GRLConfiguration(target, schedule, training.get("warmup_epochs", 5), training["epochs"])
        run = copy.deepcopy(base)
        run["model"]["grl_lambda"] = target
        run["training"]["output_dir"] = str(Path(training["output_dir"]) / f"lambda_{target:g}")
        runs.append(run)
    return runs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/grl_sweep.yaml")
    args = parser.parse_args()

    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    runs = build_sweep_configs(config)
    if Path(config["training"]["output_dir"]).exists():
        raise FileExistsError("Sweep output root exists; choose a fresh root, preserving prior results")

    for run_config in runs:

        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".yaml",
            encoding="utf-8",
            delete=False,
        ) as handle:
            yaml.safe_dump(run_config, handle, sort_keys=False)
            temporary_config = handle.name

        try:
            subprocess.run(
                [
                    sys.executable,
                    "experiments/train_nuaa_kfold.py",
                    "--config",
                    temporary_config,
                ],
                check=True,
            )
        finally:
            Path(temporary_config).unlink()


if __name__ == "__main__":
    main()
