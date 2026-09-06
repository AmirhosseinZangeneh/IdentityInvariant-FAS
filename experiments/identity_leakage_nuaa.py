"""Evaluate linearly decodable identity leakage from frozen NUAA features."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from identity_invariant_fas.data.manifest import ManifestDataset
from identity_invariant_fas.data.nuaa import load_nuaa_samples
from identity_invariant_fas.data.transforms import build_eval_transform
from identity_invariant_fas.evaluation.identity_leakage import cross_validated_identity_probe
from identity_invariant_fas.models import AblationECNN, IdentityInvariantECNN, PaperECNNClassifier
from identity_invariant_fas.training.checkpoint import load_checkpoint
from identity_invariant_fas.utils.io import write_json


@torch.no_grad()
def extract_features(model, loader, device):
    model.eval()
    feature_batches = []
    subjects = []

    for batch in loader:
        images = batch["image"].to(device)
        feature_batches.append(model.extract_features(images).cpu().numpy())
        subjects.extend(batch["subject"])

    return np.concatenate(feature_batches, axis=0), np.asarray(subjects)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["paper_ecnn", "ii_ecnn", "ablation_ecnn"], required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--nuaa-root", default="datasets")
    parser.add_argument("--partition", default="test", choices=["train", "test", "all"])
    parser.add_argument("--num-subjects", type=int, default=None)
    parser.add_argument("--grl-lambda", type=float, default=0.05)
    parser.add_argument("--image-size", type=int, default=160)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    samples = load_nuaa_samples(args.nuaa_root, partition=args.partition)
    loader = DataLoader(
        ManifestDataset(samples, transform=build_eval_transform(args.image_size)),
        batch_size=args.batch_size,
        shuffle=False,
    )

    if args.model == "paper_ecnn":
        model = PaperECNNClassifier()
    else:
        if args.num_subjects is None:
            raw = torch.load(args.checkpoint, map_location="cpu")
            state = raw.get("model_state_dict", raw) if isinstance(raw, dict) else raw
            key = "subject_classifier.2.weight"
            if key not in state:
                raise ValueError("--num-subjects is required when it cannot be inferred.")
            args.num_subjects = int(state[key].shape[0])

        model = (
            IdentityInvariantECNN(args.num_subjects, grl_lambda=args.grl_lambda)
            if args.model == "ii_ecnn"
            else AblationECNN(args.num_subjects)
        )

    model.to(device)
    load_checkpoint(args.checkpoint, model, map_location=device)
    features, subjects = extract_features(model, loader, device)
    result = cross_validated_identity_probe(features, subjects)
    result["feature_dim"] = int(features.shape[1])
    result["image_size"] = args.image_size
    write_json(result, args.output)
    print(result)


if __name__ == "__main__":
    main()
