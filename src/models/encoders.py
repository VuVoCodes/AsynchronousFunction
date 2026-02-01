"""
Modality-specific encoders for multimodal learning.
"""

import torch
import torch.nn as nn
import torchvision.models as models


class VisualEncoder(nn.Module):
    """Visual encoder using ResNet backbone."""

    def __init__(
        self,
        backbone: str = "resnet18",
        pretrained: bool = True,
        feature_dim: int = 512,
    ):
        super().__init__()
        self.feature_dim = feature_dim

        # Load backbone
        if backbone == "resnet18":
            self.backbone = models.resnet18(pretrained=pretrained)
            in_features = 512
        elif backbone == "resnet50":
            self.backbone = models.resnet50(pretrained=pretrained)
            in_features = 2048
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

        # Remove classification head
        self.backbone.fc = nn.Identity()

        # Project to feature_dim if needed
        if in_features != feature_dim:
            self.projector = nn.Linear(in_features, feature_dim)
        else:
            self.projector = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Visual input tensor of shape (B, C, H, W)
        Returns:
            Feature tensor of shape (B, feature_dim)
        """
        features = self.backbone(x)
        return self.projector(features)


class AudioEncoder(nn.Module):
    """Audio encoder for spectrogram inputs using ResNet backbone."""

    def __init__(
        self,
        backbone: str = "resnet18",
        pretrained: bool = True,
        feature_dim: int = 512,
    ):
        super().__init__()
        self.feature_dim = feature_dim

        # Load backbone
        if backbone == "resnet18":
            self.backbone = models.resnet18(pretrained=pretrained)
            in_features = 512
        elif backbone == "resnet50":
            self.backbone = models.resnet50(pretrained=pretrained)
            in_features = 2048
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

        # Modify first conv layer for single-channel spectrogram input
        self.backbone.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

        # Remove classification head
        self.backbone.fc = nn.Identity()

        # Project to feature_dim if needed
        if in_features != feature_dim:
            self.projector = nn.Linear(in_features, feature_dim)
        else:
            self.projector = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Audio spectrogram tensor of shape (B, 1, H, W)
        Returns:
            Feature tensor of shape (B, feature_dim)
        """
        features = self.backbone(x)
        return self.projector(features)


class TextEncoder(nn.Module):
    """Text encoder using simple embedding + LSTM."""

    def __init__(
        self,
        vocab_size: int = 10000,
        embed_dim: int = 300,
        hidden_dim: int = 256,
        feature_dim: int = 512,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.feature_dim = feature_dim

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.projector = nn.Linear(hidden_dim * 2, feature_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Token indices of shape (B, seq_len)
        Returns:
            Feature tensor of shape (B, feature_dim)
        """
        embedded = self.embedding(x)
        _, (hidden, _) = self.lstm(embedded)
        # Concatenate forward and backward hidden states
        hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        return self.projector(hidden)
