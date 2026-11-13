from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn


METHOD_ECA = "ECA-MSCNN"
IMPLEMENTATION_ROLE = "paper-inspired adapted implementation"


@dataclass
class ModelBundle:
    model: nn.Module
    model_config: dict
    learning_rate: float
    lr_gamma: float = 0.9
    lr_period: int = 10


class ECABlock(nn.Module):
    def __init__(self, channels: int, gamma: int = 2, bias: int = 1):
        super().__init__()
        kernel = int(abs((math.log2(channels) / gamma) + (bias / gamma)))
        kernel = kernel if kernel % 2 else kernel + 1
        kernel = max(1, kernel)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.conv = nn.Conv1d(
            1, 1, kernel_size=kernel, padding=(kernel - 1) // 2, bias=False
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = self.pool(x).transpose(1, 2)
        weights = self.sigmoid(self.conv(weights)).transpose(1, 2)
        return x * weights


class MultiScaleLayer(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, kernels: tuple[int, int, int]
    ):
        super().__init__()
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(
                        in_channels,
                        out_channels,
                        kernel_size=kernel,
                        padding=kernel // 2,
                    ),
                    nn.BatchNorm1d(out_channels),
                    nn.ReLU(inplace=True),
                    nn.MaxPool1d(kernel_size=2, stride=2),
                )
                for kernel in kernels
            ]
        )
        self.out_channels = out_channels * len(kernels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([branch(x) for branch in self.branches], dim=1)


class ECAMSCNNBranch(nn.Module):
    def __init__(self, kernels: tuple[int, int, int], stem_stride: int):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=96, stride=stem_stride, padding=47),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        self.eca = ECABlock(64)
        self.layer1 = MultiScaleLayer(64, 64, kernels)
        self.layer2 = MultiScaleLayer(self.layer1.out_channels, 128, kernels)
        self.layer3 = MultiScaleLayer(self.layer2.out_channels, 128, kernels)
        self.pool = nn.AdaptiveMaxPool1d(1)

    @staticmethod
    def _zscore(x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True, unbiased=False).clamp_min(1e-6)
        return (x - mean) / std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.eca(self.stem(self._zscore(x)))
        x = self.layer3(self.layer2(self.layer1(x)))
        return self.pool(x).flatten(1)


class ECAMSCNNFusion(nn.Module):
    def __init__(
        self,
        class_count: int,
        kernels_a: tuple[int, int, int],
        kernels_b: tuple[int, int, int],
        stride_a: int,
        stride_b: int,
    ):
        super().__init__()
        self.branch_a = ECAMSCNNBranch(kernels_a, stride_a)
        self.branch_b = ECAMSCNNBranch(kernels_b, stride_b)
        self.reduce = nn.AdaptiveAvgPool1d(64)
        self.classifier = nn.Linear(64, class_count)

    def forward(self, x_a: torch.Tensor, x_b: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.extract_features(x_a, x_b))

    def extract_features(self, x_a: torch.Tensor, x_b: torch.Tensor) -> torch.Tensor:
        fused = torch.cat([self.branch_a(x_a), self.branch_b(x_b)], dim=1)
        return self.reduce(fused.unsqueeze(1)).squeeze(1)


def build_bundle(method, data, config):
    if method != METHOD_ECA:
        raise ValueError(f"Unknown deep comparison method: {method}")
    vibration_kernels = (3, 5, 7)
    ae_kernels = (7, 9, 15)
    kernels_a, kernels_b = vibration_kernels, ae_kernels
    stride_a, stride_b = 8, 16
    model = ECAMSCNNFusion(
        data.class_count,
        kernels_a=kernels_a,
        kernels_b=kernels_b,
        stride_a=stride_a,
        stride_b=stride_b,
    )
    return ModelBundle(
        model=model,
        model_config={
            "Name": METHOD_ECA,
            "Implementation_Role": IMPLEMENTATION_ROLE,
            "Stem_Kernel": 96,
            "Stem_Stride_A": stride_a,
            "Stem_Stride_B": stride_b,
            "Kernels_A": list(kernels_a),
            "Kernels_B": list(kernels_b),
            "MultiScale_Channels": [64, 128, 128],
            "Attention": "ECA within each source branch",
            "Cross_Source_Fusion": "direct latent-feature concatenation",
            "Classifier": "adaptive average pooling plus linear Softmax logits",
        },
        learning_rate=1e-4,
        lr_gamma=0.9,
        lr_period=1,
    )
