"""Compare paired fold-level results without over-claiming significance."""

from __future__ import annotations

import argparse
import json

from identity_invariant_fas.evaluation.statistics import paired_fold_comparison


def load_metric(path: str, metric: str):
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    return [row[metric] for row in data["fold_results"]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", required=True)
    parser.add_argument("--b", required=True)
    parser.add_argument("--metric", default="acer")
    args = parser.parse_args()

    result = paired_fold_comparison(
        load_metric(args.a, args.metric),
        load_metric(args.b, args.metric),
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
