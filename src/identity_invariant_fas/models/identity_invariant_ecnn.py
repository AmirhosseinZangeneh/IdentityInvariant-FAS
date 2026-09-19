"""Identity-invariant ECNN with adversarial subject suppression."""

from __future__ import annotations

from torch import Tensor, nn

from .grl import GradientReversalLayer
from .paper_ecnn import PaperECNN


class IdentityInvariantECNN(nn.Module):
    """Learn spoof-discriminative features while suppressing subject identity.

    A subject classifier receives encoder features through a gradient reversal
    layer. Its classification loss therefore trains the subject head normally
    while driving the shared encoder away from linearly subject-discriminative
    representations.
    """

    def __init__(
        self,
        num_subjects: int,
        grl_lambda: float = 0.05,
        subject_hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        if num_subjects < 2:
            raise ValueError("num_subjects must be at least 2.")

        self.encoder = PaperECNN()
        self.feature_dim = self.encoder.feature_dim
        self.grl_lambda = float(grl_lambda)

        self.spoof_classifier = nn.Linear(self.feature_dim, 2)
        self.grl = GradientReversalLayer(coefficient=self.grl_lambda)
        self.subject_classifier = nn.Sequential(
            nn.Linear(self.feature_dim, subject_hidden_dim),
            nn.ReLU(),
            nn.Linear(subject_hidden_dim, num_subjects),
        )

    def set_grl_lambda(self, value: float) -> None:
        self.grl_lambda = float(value)
        self.grl.set_coefficient(value)

    def extract_features(self, x: Tensor) -> Tensor:
        return self.encoder(x)

    def forward(
        self,
        x: Tensor,
        return_feature: bool = False
    ):

        features = self.extract_features(x)

        spoof_logits = self.spoof_classifier(features)

        subject_logits = self.subject_classifier(
            self.grl(features)
        )

        if return_feature:
            return spoof_logits, subject_logits, features

        return spoof_logits, subject_logits
