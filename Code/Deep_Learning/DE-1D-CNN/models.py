from __future__ import annotations

import json
import math
from pathlib import Path
from dataclasses import dataclass

import torch
import torch.nn as nn


METHOD_DE = "DE-1D-CNN"
IMPLEMENTATION_ROLE = "paper-inspired adapted implementation"


@dataclass
class ModelBundle:
    model: nn.Module
    model_config: dict
    learning_rate: float
    lr_gamma: float = 0.9
    lr_period: int = 10


class DE1DCNN(nn.Module):

    def __init__(
        self, class_count: int, base_channels: int, dropout: float, hidden: int
    ):
        super().__init__()
        widths = [
            base_channels,
            base_channels,
            base_channels * 2,
            base_channels * 2,
            base_channels * 2,
        ]
        kernels = [7, 5, 5, 3, 3]
        blocks = []
        in_channels = 1
        for width, kernel in zip(widths, kernels):
            blocks.extend(
                [
                    nn.Conv1d(
                        in_channels, width, kernel_size=kernel, padding=kernel // 2
                    ),
                    nn.BatchNorm1d(width),
                    nn.ReLU(inplace=True),
                    nn.MaxPool1d(kernel_size=2, stride=2),
                ]
            )
            in_channels = width
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool1d(8)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(float(dropout)),
            nn.Linear(widths[-1] * 8, int(hidden)),
            nn.ReLU(inplace=True),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden), int(hidden)),
            nn.ReLU(inplace=True),
            nn.Linear(int(hidden), class_count),
        )

    @staticmethod
    def _standardize(x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True, unbiased=False).clamp_min(1e-6)
        return (x - mean) / std

    def forward(self, x_a: torch.Tensor, x_b: torch.Tensor) -> torch.Tensor:

        fused = torch.cat([self._standardize(x_a), self._standardize(x_b)], dim=-1)
        return self.classifier(self.pool(self.features(fused)))

    def extract_features(self, x_a: torch.Tensor, x_b: torch.Tensor) -> torch.Tensor:
        fused = torch.cat([self._standardize(x_a), self._standardize(x_b)], dim=-1)
        return self.pool(self.features(fused)).flatten(1)


def load_de_parameters(path, dataset="MY_DATA"):

    path = Path(path).expanduser().resolve()
    record = json.loads(path.read_text(encoding="utf-8-sig"))
    if record.get("Dataset") != dataset or record.get("Method") != METHOD_DE:
        raise ValueError(f"Frozen DE parameter dataset/method mismatch: {path}")
    params = record.get("Best_Parameters", {})
    bounds = {
        "base_channels": (32, 64),
        "dropout": (0.1, 0.5),
        "hidden": (300, 400),
        "learning_rate": (1e-4, 10.0**-2.5),
        "lr_gamma": (0.1, 0.95),
        "lr_period": (1, 10),
    }
    if set(params) != set(bounds):
        raise ValueError(f"Frozen DE parameters have missing/unexpected fields: {path}")
    for name, (lower, upper) in bounds.items():
        value = params[name]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not lower <= value <= upper
            or name in {"base_channels", "hidden", "lr_period"}
            and int(value) != value
        ):
            raise ValueError(f"Invalid frozen DE parameter {name}: {value}")
    return dict(params)


def build_bundle(method, data, config):
    if method != METHOD_DE:
        raise ValueError(f"Unknown deep comparison method: {method}")
    if not config.get("de_parameters_file"):
        raise ValueError("de_parameters_file is required; DE search is disabled")
    params = load_de_parameters(config["de_parameters_file"], data.condition.dataset)
    model = DE1DCNN(
        data.class_count, params["base_channels"], params["dropout"], params["hidden"]
    )
    return ModelBundle(
        model=model,
        model_config={
            "Name": METHOD_DE,
            "Implementation_Role": IMPLEMENTATION_ROLE,
            "Fusion": "standardize_each_source_then_temporal_concatenation",
            "Convolution_Blocks": 5,
            **params,
        },
        learning_rate=params["learning_rate"],
        lr_gamma=params["lr_gamma"],
        lr_period=params["lr_period"],
    )
