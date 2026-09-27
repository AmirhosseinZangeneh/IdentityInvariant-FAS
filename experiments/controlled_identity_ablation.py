"""Prospective matched three-arm experiment. Importing does not execute it."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
import subprocess
import sys

import yaml

from identity_invariant_fas.models.controlled_identity_ecnn import IDENTITY_MODES
from identity_invariant_fas.training.engine import GRLConfiguration
from identity_invariant_fas.training.losses import MultiTaskFASLoss


def build_ablation_configs(config: dict) -> list[dict]:
    """Only the explicit arm and output destination vary across matched runs."""
    base = copy.deepcopy(config)
    if base.pop("identity_arms") != list(IDENTITY_MODES):
        raise ValueError("Exactly the three controlled identity arms are required")
    if base["model"]["name"] != "controlled_identity_ecnn":
        raise ValueError("Use the shared controlled_identity_ecnn architecture")
    if "identity_mode" in base["model"] or "grl_lambda" in base["model"]:
        raise ValueError("Specify identity_arms and one nonnegative identity_target_lambda")
    training = base["training"]
    if training["initialization_policy"] != "same_fold_seed":
        raise ValueError("Controlled arms require the same initialization seed per fold")
    if training["optimizer"] != "adamw" or training["model_selection"] != "lowest_validation_acer":
        raise ValueError("Unsupported optimizer/model-selection contract")
    GRLConfiguration(base["model"]["identity_target_lambda"], training["grl_schedule"],
                     training["warmup_epochs"], training["epochs"])
    runs = []
    for mode in IDENTITY_MODES:
        MultiTaskFASLoss(training["subject_loss_weight"], identity_mode=mode)
        run = copy.deepcopy(base)
        run["model"]["identity_mode"] = mode
        run["training"]["output_dir"] = str(Path(training["output_dir"]) / mode)
        runs.append(run)
    return runs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/controlled_identity_ablation.yaml")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    runs = build_ablation_configs(config)
    root = Path(config["training"]["output_dir"])
    # Atomic reservation: interrupted/previous runs must never be overwritten.
    root.mkdir(parents=True, exist_ok=False)
    for run in runs:
        path = root / (run["model"]["identity_mode"] + ".yaml")
        with path.open("x", encoding="utf-8") as handle:
            yaml.safe_dump(run, handle, sort_keys=False)
        subprocess.run([sys.executable, str(Path(__file__).with_name("train_nuaa_kfold.py")),
                        "--config", str(path)], check=True)


if __name__ == "__main__":
    main()
