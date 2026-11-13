from dataclasses import dataclass
from pathlib import Path
import random

import numpy as np
import pandas as pd
import torch

from common.data_columns import align_modal_dataframe_columns

SPLIT_NAMES = ("train", "validation", "calibration", "test")
SPLIT_RATIOS = (0.60, 0.075, 0.075, 0.25)
DEFAULT_PREPROCESSING = {
    "noise_power_centered": True,
    "ae_train_dc_removal": True,
    "window_normalization": "zscore",
    "pooled_train_zscore": False,
}


@dataclass(frozen=True)
class Condition:
    rpm: str
    snr: int | None
    channel: str
    path_a: Path
    path_b: Path
    dataset: str = "MY_DATA"
    window_a: int = 512
    step_a: int = 51
    window_b: int = 16384
    step_b: int = 16384


@dataclass
class PreparedData:
    condition: Condition
    class_names: list[str]
    splits: dict[str, dict[str, np.ndarray]]

    @property
    def class_count(self):
        return len(self.class_names)


def condition_for(config, rpm, snr, channel="CH_06"):
    root = Path(config["laboratory_root"])
    if snr is not None and (isinstance(snr, bool) or int(snr) != float(snr)):
        raise ValueError("SNR must be an integer or null for Clean")
    return Condition(
        str(rpm),
        None if snr is None else int(snr),
        str(channel),
        root
        / "Data_Vibration"
        / str(rpm)
        / channel
        / f"data_Vibration_1k_{rpm}_{channel}_4c.csv",
        root / "Data_AE" / str(rpm) / f"data_AE_2000k_{rpm}_4c.csv",
    )


def chronological_boundaries(length):
    bounds = [
        0,
        int(round(length * SPLIT_RATIOS[0])),
        int(round(length * (SPLIT_RATIOS[0] + SPLIT_RATIOS[1]))),
        int(round(length * (SPLIT_RATIOS[0] + SPLIT_RATIOS[1] + SPLIT_RATIOS[2]))),
        length,
    ]
    if any(b <= a for a, b in zip(bounds, bounds[1:])):
        raise ValueError("Every chronological split must contain raw samples")
    return bounds


def normalize_windows(windows, mode):
    if mode == "none":
        return windows
    x = np.asarray(windows, dtype=np.float64)
    if x.ndim != 2 or x.shape[-1] < 2 or not np.isfinite(x).all():
        raise ValueError("Expected finite 2D windows")
    if mode == "zscore":
        numerator = x - x.mean(axis=-1, keepdims=True)
        scale = x.std(axis=-1, keepdims=True, ddof=0)
    elif mode == "rms":
        numerator = x
        scale = np.sqrt(np.mean(x * x, axis=-1, keepdims=True))
    else:
        raise ValueError("window_normalization must be none, zscore or rms")
    if not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError("Cannot normalize a zero-scale window")
    return numerator / (scale + 1e-12)


def prepare_source(frame, window, step, snr, preprocessing, is_ae):
    records = []
    for column in frame.columns:
        raw = frame[column].to_numpy(dtype=np.float64)
        if raw.ndim != 1 or not np.isfinite(raw).all():
            raise ValueError(f"Non-finite raw signal: {column}")
        bounds = chronological_boundaries(len(raw))
        records.append(
            {name: raw[bounds[i] : bounds[i + 1]] for i, name in enumerate(SPLIT_NAMES)}
        )
    remove_dc = preprocessing["ae_train_dc_removal"] and is_ae
    offsets = [
        float(np.mean(record["train"])) if remove_dc else 0.0 for record in records
    ]
    trains = [
        record["train"] - offset if remove_dc else record["train"]
        for record, offset in zip(records, offsets)
    ]
    pooled = np.concatenate(trains)
    mean, std = float(np.mean(pooled)), float(np.std(pooled, ddof=0))
    if preprocessing["pooled_train_zscore"] and (not np.isfinite(std) or std <= 0):
        raise ValueError("Pooled TRAIN standard deviation must be finite and positive")
    result = {name: [] for name in SPLIT_NAMES}
    for record, offset in zip(records, offsets):
        segments = {name: values.copy() for name, values in record.items()}
        for name in SPLIT_NAMES:
            if remove_dc:
                segments[name] = segments[name] - offset
            if preprocessing["pooled_train_zscore"]:
                segments[name] = (segments[name] - mean) / (std + 1e-12)
        for name in SPLIT_NAMES:
            if snr is not None:
                signal = segments[name]
                noise = np.random.randn(len(signal))
                power = (
                    np.mean((signal - np.mean(signal)) ** 2)
                    if preprocessing["noise_power_centered"]
                    else np.mean(signal**2)
                )
                if not np.isfinite(power) or power <= 0:
                    raise ValueError(
                        "Finite-SNR noise requires positive finite signal power"
                    )
                noise_power = np.mean(noise**2)
                if noise_power > 0:
                    segments[name] = (
                        signal
                        + np.sqrt((power / (10 ** (float(snr) / 10.0))) / noise_power)
                        * noise
                    )
        for name, signal in segments.items():
            windows = np.asarray(
                [
                    signal[i : i + window]
                    for i in range(0, len(signal) - window + 1, step)
                ]
            )
            if not len(windows):
                raise ValueError(f"No {name} windows; window={window}, step={step}")
            result[name].append(
                normalize_windows(
                    windows, preprocessing["window_normalization"]
                ).astype(np.float32)
            )
    return result


def prepare_condition(config, rpm, snr, seed, channel="CH_06"):
    condition = condition_for(config, rpm, snr, channel)
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed(int(seed))
        torch.cuda.manual_seed_all(int(seed))
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    frame_a, frame_b, names = align_modal_dataframe_columns(
        pd.read_csv(condition.path_a), pd.read_csv(condition.path_b)
    )
    names = [str(name) for name in names]
    if len(names) != config.get("expected_class_count", 4):
        raise ValueError(f"Unexpected fault-class count: {names}")
    preprocessing = dict(config.get("preprocessing", DEFAULT_PREPROCESSING))
    for key in ("noise_power_centered", "ae_train_dc_removal", "pooled_train_zscore"):
        if type(preprocessing.get(key)) is not bool:
            raise ValueError(f"preprocessing.{key} must be boolean")
    a = prepare_source(
        frame_a,
        condition.window_a,
        condition.step_a,
        condition.snr,
        preprocessing,
        False,
    )
    b = prepare_source(
        frame_b,
        condition.window_b,
        condition.step_b,
        condition.snr,
        preprocessing,
        True,
    )
    splits = {}
    for name in SPLIT_NAMES:
        xs_a, xs_b, labels = [], [], []
        for class_index, (values_a, values_b) in enumerate(zip(a[name], b[name])):
            count = min(len(values_a), len(values_b))
            ia = (
                np.arange(count)
                if len(values_a) == count
                else np.round(np.linspace(0, len(values_a) - 1, count)).astype(int)
            )
            ib = (
                np.arange(count)
                if len(values_b) == count
                else np.round(np.linspace(0, len(values_b) - 1, count)).astype(int)
            )
            xs_a.append(values_a[ia])
            xs_b.append(values_b[ib])
            labels.append(np.full(count, class_index, dtype=np.int64))
        splits[name] = {
            "x_a": np.concatenate(xs_a),
            "x_b": np.concatenate(xs_b),
            "y": np.concatenate(labels),
        }
    return PreparedData(condition, names, splits)
