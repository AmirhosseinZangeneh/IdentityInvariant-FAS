"""Reusable training and evaluation engine."""

from __future__ import annotations

import math

from dataclasses import dataclass

import numpy as np

import torch

from torch import nn

from ..evaluation.metrics import compute_classification_metrics
from ..models.controlled_identity_ecnn import validate_identity_mode

from .losses import LossOutput, MultiTaskFASLoss


@dataclass
class EpochResult:
    loss: float
    spoof_loss: float
    subject_loss: float | None
    accuracy: float
    grl_lambda: float | None
    identity_mode: str | None = None
    identity_encoder_scale: float | None = None
    subject_loss_enabled: bool | None = None
    subject_weight: float | None = None

    def history_record(self, epoch: int) -> dict:
        """Legacy GRL history stays compatible; controlled arms use signed names."""
        if self.identity_mode is None:
            return {"epoch": epoch, "grl_lambda": self.grl_lambda}
        return {
            "epoch": epoch, "identity_mode": self.identity_mode,
            "identity_encoder_scale": self.identity_encoder_scale,
            "subject_loss_enabled": self.subject_loss_enabled,
            "subject_weight": self.subject_weight,
        }


def compute_grl_lambda(
    epoch: int,
    total_epochs: int,
) -> float:
    """Unit DANN factor at a one-based active epoch (after warm-up)."""
    if type(total_epochs) is not int or total_epochs < 1:
        raise ValueError("total_epochs must be an integer >= 1")
    if type(epoch) is not int or not 1 <= epoch <= total_epochs:
        raise ValueError("epoch must be in 1..total_epochs")
    progress = epoch / total_epochs

    return (
        2.0 /
        (1.0 + math.exp(-10 * progress))
    ) - 1.0


@dataclass(frozen=True)
class GRLConfiguration:
    """Prospective contract; the target is independent of mutable model state."""

    target: float
    schedule: str
    warmup_epochs: int
    total_epochs: int

    def __post_init__(self):
        if not math.isfinite(self.target) or self.target < 0:
            raise ValueError("GRL target must be finite and >= 0")
        if self.schedule not in {"fixed", "dann", "progressive"}:
            raise ValueError("Unknown GRL schedule")
        if type(self.total_epochs) is not int or self.total_epochs < 1:
            raise ValueError("total_epochs must be an integer >= 1")
        if type(self.warmup_epochs) is not int or not 0 <= self.warmup_epochs < self.total_epochs:
            raise ValueError("GRL warmup_epochs must be an integer in 0..total_epochs-1")

    def coefficient(self, epoch: int) -> float:
        if type(epoch) is not int or not 1 <= epoch <= self.total_epochs:
            raise ValueError("epoch must be in 1..total_epochs")
        if epoch <= self.warmup_epochs:
            return 0.0
        if self.schedule == "fixed":
            return self.target
        return self.target * compute_grl_lambda(
            epoch - self.warmup_epochs, self.total_epochs - self.warmup_epochs
        )


class Trainer:

    def __init__(
        self,
        model: nn.Module,
        optimizer,
        device: str | torch.device,
        *,
        subject_weight: float = 0.1,
        warmup_epochs: int = 0,
        total_epochs: int,
        grl_target_lambda: float | None = None,
        grl_schedule: str = "fixed",
        identity_mode: str | None = None,
        identity_target_lambda: float | None = None,
    ) -> None:

        self.model = model
        self.optimizer = optimizer
        self.device = torch.device(device)

        self.identity_mode = identity_mode
        model_mode = getattr(model, "identity_mode", None)
        if identity_mode is not None:
            validate_identity_mode(identity_mode)
            if model_mode != identity_mode or not callable(getattr(model, "set_identity_encoder_scale", None)):
                raise ValueError("Trainer and model must explicitly agree on identity_mode")
            if grl_target_lambda is not None:
                raise ValueError("Controlled arms use identity_target_lambda, not grl_target_lambda")
        elif model_mode is not None or identity_target_lambda is not None:
            raise ValueError("Controlled identity training requires an explicit identity_mode")
        self.loss_fn: MultiTaskFASLoss = MultiTaskFASLoss(
            subject_weight=subject_weight, identity_mode=identity_mode
        )

        if type(total_epochs) is not int or total_epochs < 1:
            raise ValueError("total_epochs must be an integer >= 1")
        if type(warmup_epochs) is not int or warmup_epochs < 0:
            raise ValueError("warmup_epochs must be an integer >= 0")
        if grl_schedule not in {"fixed", "dann", "progressive"}:
            raise ValueError("Unknown GRL schedule")
        if grl_target_lambda is not None and (
            not math.isfinite(grl_target_lambda) or grl_target_lambda < 0
        ):
            raise ValueError("GRL target must be finite and >= 0")
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.grl_config = None
        self.identity_config = None
        if identity_mode is not None:
            if identity_target_lambda is None:
                raise ValueError("Controlled arms require an explicit identity target")
            self.identity_config = GRLConfiguration(
                identity_target_lambda, grl_schedule, warmup_epochs, total_epochs
            )
        if callable(getattr(model, "set_grl_lambda", None)):
            if grl_target_lambda is None:
                raise ValueError("GRL models require an explicit configured target")
            self.grl_config = GRLConfiguration(
                grl_target_lambda, grl_schedule, warmup_epochs, total_epochs
            )

        self.current_grl_lambda = 0.0
        self.current_identity_encoder_scale = None


    def _update_grl_lambda(
        self,
        epoch: int,
    ) -> None:
        """
        Update GRL coefficient when supported by the model.
        """

        if type(epoch) is not int or not 1 <= epoch <= self.total_epochs:
            raise ValueError("epoch must be in 1..total_epochs")
        if self.identity_mode is not None:
            magnitude = self.identity_config.coefficient(epoch)
            sign = {"spoof_only": 0, "identity_positive": 1, "identity_adversarial": -1}[self.identity_mode]
            scale = sign * magnitude
            self.model.set_identity_encoder_scale(scale)
            self.current_identity_encoder_scale = scale
            self.current_grl_lambda = None
            return
        grl_setter = getattr(
            self.model,
            "set_grl_lambda",
            None,
        )

        if not callable(grl_setter):
            self.current_grl_lambda = 0.0
            return

        value = self.grl_config.coefficient(epoch)

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
                subject_loss_enabled=self.identity_mode != "spoof_only",
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
            identity_mode=self.identity_mode,
            identity_encoder_scale=self.current_identity_encoder_scale,
            subject_loss_enabled=self.identity_mode != "spoof_only",
            subject_weight=self.loss_fn.subject_weight,
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
