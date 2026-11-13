from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


METHOD_ATTENTION = "Attention-1DCNN"
IMPLEMENTATION_ROLE = "paper-inspired adapted implementation"


@dataclass
class ModelBundle:
    model: nn.Module
    model_config: dict
    learning_rate: float
    lr_gamma: float = 0.9
    lr_period: int = 10


class ChannelAttention1D(nn.Module):
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.gate = nn.Sequential(
            nn.Conv1d(channels, hidden, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gate(self.pool(x))


class SelfFeatureBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super().__init__()
        skip_channels = max(8, out_channels // 4)
        self.main = nn.Sequential(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            ),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            ChannelAttention1D(out_channels, reduction=4),
        )
        self.skip = nn.Sequential(
            nn.Conv1d(in_channels, skip_channels, kernel_size=1),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.BatchNorm1d(skip_channels),
            nn.ReLU(inplace=True),
        )
        self.out_channels = out_channels + skip_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.main(x), self.skip(x)], dim=1)


class SpectralAttentionBranch(nn.Module):
    def __init__(self):
        super().__init__()

        self.block1 = SelfFeatureBlock(1, 64, kernel_size=9)
        self.block2 = SelfFeatureBlock(self.block1.out_channels, 128, kernel_size=5)
        self.pool = nn.AdaptiveAvgPool1d(32)
        self.out_channels = self.block2.out_channels

    @staticmethod
    def spectrum(x: torch.Tensor) -> torch.Tensor:
        magnitude = torch.abs(torch.fft.rfft(x, dim=-1))
        minimum = magnitude.amin(dim=-1, keepdim=True)
        maximum = magnitude.amax(dim=-1, keepdim=True)
        return (magnitude - minimum) / (maximum - minimum).clamp_min(1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.spectrum(x)
        return self.pool(self.block2(self.block1(x)))


class Attention1DCNN(nn.Module):

    def __init__(self, class_count: int):
        super().__init__()
        self.branch_a = SpectralAttentionBranch()
        self.branch_b = SpectralAttentionBranch()
        channels = self.branch_a.out_channels
        self.mutual_fusion = nn.Sequential(
            nn.Conv2d(channels, 128, kernel_size=(2, 3), padding=(0, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(64, class_count)

    def forward(self, x_a: torch.Tensor, x_b: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.extract_features(x_a, x_b))

    def extract_features(self, x_a: torch.Tensor, x_b: torch.Tensor) -> torch.Tensor:
        feature_a = self.branch_a(x_a)
        feature_b = self.branch_b(x_b)
        feature_map = torch.stack([feature_a, feature_b], dim=2)
        return self.mutual_fusion(feature_map).flatten(1)


def build_bundle(method, data, config):
    if method != METHOD_ATTENTION:
        raise ValueError(f"Unknown deep comparison method: {method}")
    return ModelBundle(
        model=Attention1DCNN(data.class_count),
        model_config={
            "Name": METHOD_ATTENTION,
            "Implementation_Role": IMPLEMENTATION_ROLE,
            "Preprocessing": "one-sided DFT magnitude and per-sample min-max normalization",
            "Spectral_Grid_Before_Convolution": None,
            "Self_Feature_Blocks_Per_Source": 2,
            "Attention": "channel squeeze-excitation",
            "Mutual_Fusion": "feature-map stacking followed by 2D and pointwise convolutions",
            "Fusion_Grid": [2, 32],
        },
        learning_rate=1e-3,
        lr_gamma=0.9,
        lr_period=10,
    )
