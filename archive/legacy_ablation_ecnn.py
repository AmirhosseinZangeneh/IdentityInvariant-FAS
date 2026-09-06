"""Legacy ablation architecture retained only for loading historical checkpoints.

Do not use this class for the final controlled ablation study: unlike II-ECNN,
the historical model used a 128-dimensional encoder and therefore confounded
the effect of gradient reversal with an architecture change.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class LegacyAblationECNN(nn.Module):
    def __init__(self, num_subjects: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.conv2 = nn.Conv2d(32, 32, 3)
        self.pool1 = nn.MaxPool2d(3, 1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 16, 3)
        self.conv4 = nn.Conv2d(16, 16, 3)
        self.pool2 = nn.MaxPool2d(3, 1)
        self.bn2 = nn.BatchNorm2d(16)
        self.conv5 = nn.Conv2d(16, 8, 3)
        self.conv6 = nn.Conv2d(8, 8, 3)
        self.global_pool = nn.AdaptiveMaxPool2d((1, 1))
        self.bn3 = nn.BatchNorm1d(8)
        self.fc1 = nn.Linear(8, 256)
        self.fc2 = nn.Linear(256, 128)
        self.feature_dim = 128
        self.spoof_classifier = nn.Linear(128, 2)
        self.subject_classifier = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_subjects),
        )

    def extract_features(self, x: Tensor) -> Tensor:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.bn1(self.pool1(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.bn2(self.pool2(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = torch.flatten(self.global_pool(x), start_dim=1)
        x = self.bn3(x)
        x = F.relu(self.fc1(x))
        return F.relu(self.fc2(x))

    def forward(self, x: Tensor, return_feature: bool = False):
        features = self.extract_features(x)
        spoof_logits = self.spoof_classifier(features)
        if return_feature:
            return spoof_logits, features
        return spoof_logits, self.subject_classifier(features)
