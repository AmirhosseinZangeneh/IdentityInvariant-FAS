"""Evaluate a trained image model on a manifest, optionally at video level."""

from __future__ import annotations

import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader

from identity_invariant_fas.data.manifest import ManifestDataset, read_manifest
from identity_invariant_fas.data.transforms import build_eval_transform
from identity_invariant_fas.evaluation.aggregation import aggregate_video_scores
from identity_invariant_fas.evaluation.metrics import compute_classification_metrics
from identity_invariant_fas.models import AblationECNN, IdentityInvariantECNN, PaperECNNClassifier
from identity_invariant_fas.training.checkpoint import load_checkpoint
from identity_invariant_fas.utils.io import write_json


def infer_num_subjects(checkpoint: str) -> int:
    raw = torch.load(checkpoint, map_location="cpu")
    state = raw.get("model_state_dict", raw) if isinstance(raw, dict) else raw
    if "subject_classifier.2.weight" not in state:
        raise ValueError("Cannot infer subject head size from checkpoint.")
    return int(state["subject_classifier.2.weight"].shape[0])


def build_model(name: str, checkpoint: str, grl_lambda: float):
    if name == "paper_ecnn":
        return PaperECNNClassifier()
    num_subjects = infer_num_subjects(checkpoint)
    if name == "ii_ecnn":
        return IdentityInvariantECNN(num_subjects, grl_lambda=grl_lambda)
    if name == "ablation_ecnn":
        return AblationECNN(num_subjects)
    raise ValueError(f"Unknown model: {name}")


@torch.no_grad()
def predict(model, loader, device):
    labels, predictions, scores, video_ids = [], [], [], []
    model.eval()

    for batch in loader:
        images = batch["image"].to(device)
        output = model(images)
        logits = output[0] if isinstance(output, tuple) else output
        attack_scores = torch.softmax(logits, dim=1)[:, 1]
        predicted = logits.argmax(dim=1)

        labels.extend(batch["label"].tolist())
        predictions.extend(predicted.cpu().tolist())
        scores.extend(attack_scores.cpu().tolist())
        video_ids.extend(batch["video_id"])

    return (
        np.asarray(labels),
        np.asarray(predictions),
        np.asarray(scores),
        np.asarray(video_ids),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--model", choices=["paper_ecnn", "ii_ecnn", "ablation_ecnn"], required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--grl-lambda", type=float, default=0.05)
    parser.add_argument("--image-size", type=int, default=160)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--split", default="")
    parser.add_argument("--video-level", action="store_true")
    parser.add_argument("--aggregation", choices=["mean", "median"], default="mean")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    samples = read_manifest(args.manifest)
    if args.split:
        samples = [sample for sample in samples if sample.split == args.split]

    loader = DataLoader(
        ManifestDataset(samples, build_eval_transform(args.image_size)),
        batch_size=args.batch_size,
        shuffle=False,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args.model, args.checkpoint, args.grl_lambda).to(device)
    load_checkpoint(args.checkpoint, model, map_location=device)

    labels, predictions, scores, video_ids = predict(model, loader, device)
    result = {
        "frame_metrics": compute_classification_metrics(labels, predictions, scores),
    }
    if args.video_level:
        result["video_level"] = aggregate_video_scores(
            labels,
            scores,
            video_ids,
            method=args.aggregation,
            threshold=args.threshold,
        )

    write_json(result, args.output)
    print(result)


if __name__ == "__main__":
    main()
