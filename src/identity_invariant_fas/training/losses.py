"""Loss functions for baseline and identity-aware FAS models."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class LossOutput:
    total: Tensor
    spoof: Tensor
    subject: Tensor | None


class MultiTaskFASLoss(nn.Module):
    """Cross-entropy FAS loss with an optional adversarial subject objective."""

    def __init__(self, subject_weight: float = 0.1) -> None:
        super().__init__()
        if subject_weight < 0:
            raise ValueError("subject_weight must be non-negative.")
        self.subject_weight = float(subject_weight)
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(
        self,
        spoof_logits: Tensor,
        labels: Tensor,
        subject_logits: Tensor | None = None,
        subject_labels: Tensor | None = None,
        *,
        subject_loss_enabled: bool = True,
    ) -> LossOutput:
        spoof_loss = self.cross_entropy(spoof_logits, labels)
        subject_loss: Tensor | None = None

        if (
            subject_loss_enabled
            and subject_logits is not None
            and subject_labels is not None
        ):
            valid = subject_labels >= 0
            if torch.any(valid):
                subject_loss = self.cross_entropy(subject_logits[valid], subject_labels[valid])

        total = spoof_loss
        if subject_loss is not None:
            total = total + self.subject_weight * subject_loss

        return LossOutput(total=total, spoof=spoof_loss, subject=subject_loss)
