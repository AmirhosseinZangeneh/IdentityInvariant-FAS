"""Scientifically controlled ablation model for the GRL contribution."""

from __future__ import annotations

from torch import Tensor, nn

from .paper_ecnn import PaperECNN


class AblationECNN(nn.Module):
    """ECNN with the same two heads as II-ECNN but without gradient reversal.

    The encoder, feature dimensionality, spoof head, and subject head match the
    identity-invariant model. The only intended causal difference is the absence
    of gradient reversal in the subject branch.
    """

    def __init__(self, num_subjects: int, subject_hidden_dim: int = 128) -> None:
        super().__init__()
        if num_subjects < 2:
            raise ValueError("num_subjects must be at least 2.")

        self.encoder = PaperECNN()
        self.feature_dim = self.encoder.feature_dim
        self.spoof_classifier = nn.Linear(self.feature_dim, 2)
        self.subject_classifier = nn.Sequential(
            nn.Linear(self.feature_dim, subject_hidden_dim),
            nn.ReLU(),
            nn.Linear(subject_hidden_dim, num_subjects),
        )

    def extract_features(self, x: Tensor) -> Tensor:
        return self.encoder(x)

    def extract_spoof_features(self, x: Tensor) -> Tensor:
        return self.extract_features(x)

    def forward(self, x: Tensor, return_feature: bool = False):
        features = self.extract_features(x)
        spoof_logits = self.spoof_classifier(features)

        if return_feature:
            return spoof_logits, features

        subject_logits = self.subject_classifier(features)
        return spoof_logits, subject_logits
