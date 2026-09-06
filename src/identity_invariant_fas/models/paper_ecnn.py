"""ECNN baseline used by the identity-invariant experiments."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class PaperECNN(nn.Module):
    """ECNN feature encoder with a 256-dimensional output representation."""

    feature_dim = 256

    def __init__(self) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=0)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=0)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=0)
        self.conv4 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=0)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=1)
        self.bn2 = nn.BatchNorm2d(16)

        self.conv5 = nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=0)
        self.conv6 = nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=0)

        self.global_max_pool = nn.AdaptiveMaxPool2d((1, 1))
        self.bn3 = nn.BatchNorm1d(8)

        self.fc1 = nn.Linear(8, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, self.feature_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool1(x)
        x = self.bn1(x)

        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool2(x)
        x = self.bn2(x)

        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = self.global_max_pool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.bn3(x)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.relu(self.fc3(x))


class PaperECNNClassifier(nn.Module):
    """Binary face presentation-attack classifier using the ECNN encoder."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = PaperECNN()
        self.classifier = nn.Linear(self.encoder.feature_dim, 2)

    @property
    def feature_dim(self) -> int:
        return self.encoder.feature_dim

    def extract_features(self, x: Tensor) -> Tensor:
        return self.encoder(x)

    def forward(self, x: Tensor, return_feature: bool = False):
        features = self.extract_features(x)
        logits = self.classifier(features)
        if return_feature:
            return logits, features
        return logits
