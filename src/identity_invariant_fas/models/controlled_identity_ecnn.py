"""Matched identity interventions; historical model classes remain unchanged."""

from __future__ import annotations

import math

from torch import nn

from .grl import GradientScaleLayer
from .paper_ecnn import PaperECNN


IDENTITY_MODES = ("spoof_only", "identity_positive", "identity_adversarial")


def validate_identity_mode(mode: str) -> None:
    if mode not in IDENTITY_MODES:
        raise ValueError(f"identity_mode must be one of {IDENTITY_MODES}")


class ControlledIdentityECNN(nn.Module):
    """One architecture, including a dormant subject head for spoof_only.

    The configured arm is explicit and the encoder scale is supplied by the
    Trainer's frozen schedule. The control precedes the subject classifier,
    so it never scales that classifier's parameter gradients.
    """

    def __init__(self, num_subjects: int, *, identity_mode: str,
                 subject_hidden_dim: int = 128) -> None:
        super().__init__()
        validate_identity_mode(identity_mode)
        if num_subjects < 2:
            raise ValueError("num_subjects must be >= 2")
        self._identity_mode = identity_mode
        self.encoder = PaperECNN()
        self.feature_dim = self.encoder.feature_dim
        self.spoof_classifier = nn.Linear(self.feature_dim, 2)
        self.identity_gradient = GradientScaleLayer(0.0)
        self.subject_classifier = nn.Sequential(
            nn.Linear(self.feature_dim, subject_hidden_dim), nn.ReLU(),
            nn.Linear(subject_hidden_dim, num_subjects),
        )

    @property
    def identity_mode(self) -> str:
        return self._identity_mode

    def set_identity_encoder_scale(self, scale: float) -> None:
        if not math.isfinite(scale):
            raise ValueError("identity encoder scale must be finite")
        if ((self.identity_mode == "spoof_only" and scale != 0)
                or (self.identity_mode == "identity_positive" and scale < 0)
                or (self.identity_mode == "identity_adversarial" and scale > 0)):
            raise ValueError("encoder scale contradicts explicit identity_mode")
        self.identity_gradient.set_scale(scale)

    def extract_features(self, x):
        return self.encoder(x)

    def forward(self, x, return_feature: bool = False):
        features = self.extract_features(x)
        spoof_logits = self.spoof_classifier(features)
        subject_logits = None
        if self.identity_mode != "spoof_only":
            subject_logits = self.subject_classifier(self.identity_gradient(features))
        if return_feature:
            return spoof_logits, subject_logits, features
        return spoof_logits, subject_logits
