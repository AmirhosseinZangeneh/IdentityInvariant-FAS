"""Run a controlled GRL-lambda sweep with an otherwise identical protocol."""

from __future__ import annotations

import argparse
import copy
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/grl_sweep.yaml")
    args = parser.parse_args()

    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    values = config.pop("grl_values")

    for value in values:
        run_config = copy.deepcopy(config)
        run_config["model"]["grl_lambda"] = float(value)
        run_config["training"]["output_dir"] = str(
            Path(config["training"]["output_dir"]) / f"lambda_{value}"
        )

        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".yaml",
            encoding="utf-8",
            delete=False,
        ) as handle:
            yaml.safe_dump(run_config, handle, sort_keys=False)
            temporary_config = handle.name

        subprocess.run(
            [
                sys.executable,
                "experiments/train_nuaa_kfold.py",
                "--config",
                temporary_config,
            ],
            check=True,
        )


if __name__ == "__main__":
    main()
