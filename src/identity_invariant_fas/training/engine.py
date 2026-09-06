"""Reusable training and evaluation engine."""

from __future__ import annotations

import math

from dataclasses import dataclass

import numpy as np

import torch

from torch import nn

from ..evaluation.metrics import compute_classification_metrics

from .losses import LossOutput, MultiTaskFASLoss


@dataclass
class EpochResult:
    loss: float
    spoof_loss: float
    subject_loss: float | None
    accuracy: float
    grl_lambda: float


def compute_grl_lambda(
    epoch: int,
    total_epochs: int,
) -> float:
    """Compute progressive GRL coefficient."""

    progress = epoch / max(
        total_epochs - 1,
        1,
    )

    return (
        2.0 /
        (1.0 + math.exp(-10 * progress))
    ) - 1.0


class Trainer:

    def __init__(
        self,
        model: nn.Module,
        optimizer,
        device: str | torch.device,
        *,
        subject_weight: float = 0.1,
        warmup_epochs: int = 0,
        total_epochs: int = 100,
    ) -> None:

        self.model = model
        self.optimizer = optimizer
        self.device = torch.device(device)

        self.loss_fn: MultiTaskFASLoss = MultiTaskFASLoss(
            subject_weight=subject_weight
        )

        self.warmup_epochs = int(
            warmup_epochs
        )

        self.total_epochs = int(
            total_epochs
        )

        self.current_grl_lambda = 0.0


    def _update_grl_lambda(
        self,
        epoch: int,
    ) -> None:
        """
        Update GRL coefficient when supported by the model.
        """

        grl_setter = getattr(
            self.model,
            "set_grl_lambda",
            None,
        )

        if not callable(grl_setter):
            self.current_grl_lambda = 0.0
            return

        if epoch < self.warmup_epochs:
            value = 0.0

        else:
            value = compute_grl_lambda(
                epoch,
                self.total_epochs,
            )

        grl_setter(value)

        self.current_grl_lambda = value


    def _forward(
        self,
        images: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:

        output = self.model(images)

        if isinstance(output, tuple):

            spoof_logits = output[0]

            subject_logits = (
                output[1]
                if len(output) > 1
                else None
            )

        else:

            spoof_logits = output
            subject_logits = None

        return (
            spoof_logits,
            subject_logits,
        )


    def train_epoch(
        self,
        loader,
        epoch: int,
    ) -> EpochResult:

        self.model.train()

        self._update_grl_lambda(
            epoch
        )

        total_count = 0

        total_loss = 0.0
        total_spoof_loss = 0.0
        total_subject_loss = 0.0

        subject_count = 0

        total_correct = 0


        for batch in loader:

            images = batch["image"].to(
                self.device
            )

            labels = batch["label"].to(
                self.device
            )

            subject_labels = batch.get(
                "subject_label"
            )

            if subject_labels is not None:

                subject_labels = subject_labels.to(
                    self.device
                )


            self.optimizer.zero_grad(
                set_to_none=True
            )


            spoof_logits, subject_logits = self._forward(
                images
            )


            losses: LossOutput = self.loss_fn(
                spoof_logits=spoof_logits,
                labels=labels,
                subject_logits=subject_logits,
                subject_labels=subject_labels,
                subject_loss_enabled=True,
            )


            losses.total.backward()

            self.optimizer.step()


            count = labels.size(0)

            total_count += count


            total_loss += (
                losses.total.item()
                *
                count
            )


            total_spoof_loss += (
                losses.spoof.item()
                *
                count
            )


            if losses.subject is not None:

                total_subject_loss += (
                    losses.subject.item()
                    *
                    count
                )

                subject_count += count


            total_correct += (
                (
                    spoof_logits.argmax(dim=1)
                    ==
                    labels
                )
                .sum()
                .item()
            )


        if total_count == 0:

            raise RuntimeError(
                "Training loader produced no samples."
            )


        return EpochResult(

            loss=(
                total_loss /
                total_count
            ),

            spoof_loss=(
                total_spoof_loss /
                total_count
            ),

            subject_loss=(
                total_subject_loss /
                subject_count
                if subject_count
                else None
            ),

            accuracy=(
                total_correct /
                total_count
            ),

            grl_lambda=self.current_grl_lambda,
        )


    @torch.no_grad()
    def evaluate(
        self,
        loader,
    ) -> dict:

        self.model.eval()

        labels_all: list[int] = []
        predictions_all: list[int] = []
        scores_all: list[float] = []

        paths_all: list[str] = []
        subjects_all: list[str] = []


        for batch in loader:

            images = batch["image"].to(
                self.device
            )

            labels = batch["label"].to(
                self.device
            )


            output = self.model(
                images
            )


            spoof_logits = (
                output[0]
                if isinstance(output, tuple)
                else output
            )


            scores = torch.softmax(
                spoof_logits,
                dim=1,
            )[:, 1]


            predictions = spoof_logits.argmax(
                dim=1
            )


            labels_all.extend(
                labels.cpu().tolist()
            )

            predictions_all.extend(
                predictions.cpu().tolist()
            )

            scores_all.extend(
                scores.cpu().tolist()
            )


            paths_all.extend(
                batch.get(
                    "path",
                    [""] * labels.size(0),
                )
            )

            subjects_all.extend(
                batch.get(
                    "subject",
                    [""] * labels.size(0),
                )
            )


        metrics = compute_classification_metrics(
            np.asarray(labels_all),
            np.asarray(predictions_all),
            np.asarray(scores_all),
        )


        metrics["predictions"] = [

            {
                "path": path,
                "subject": subject,
                "label": int(label),
                "prediction": int(prediction),
                "score_attack": float(score),
                "correct": bool(
                    label == prediction
                ),
            }

            for path, subject, label, prediction, score

            in zip(
                paths_all,
                subjects_all,
                labels_all,
                predictions_all,
                scores_all,
            )

        ]


        return metrics