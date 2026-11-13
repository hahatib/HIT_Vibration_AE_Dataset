from __future__ import annotations
import numpy as np

VERSION = 3
KEYS = ("noise_power_centered", "ae_train_dc_removal", "pooled_train_zscore")
WINDOW_MODES = ("none", "zscore", "rms")
WINDOW_EPSILON = 1e-12


def protocol(value=None):
    value = {} if value is None else value
    if not isinstance(value, dict) or set(value) - set(KEYS) - {
        "version",
        "window_normalization",
    }:
        raise ValueError(
            "preprocessing accepts only declared boolean switches and window_normalization"
        )
    if "version" in value and (
        type(value["version"]) is not int or value["version"] != VERSION
    ):
        raise ValueError("Unsupported preprocessing protocol version")
    result = {"version": VERSION}
    for key in KEYS:
        flag = value.get(key, key == "pooled_train_zscore")
        if type(flag) is not bool:
            raise ValueError(f"preprocessing.{key} must be boolean")
        result[key] = flag
    mode = value.get("window_normalization", "none")
    if type(mode) is not str or mode not in WINDOW_MODES:
        raise ValueError("window_normalization must be none, zscore or rms")
    result["window_normalization"] = mode
    return result


def validate(value):
    if not isinstance(value, dict) or set(value) != {
        "version",
        "window_normalization",
        *KEYS,
    }:
        raise ValueError("Missing saved preprocessing protocol; use fresh results")
    checked = protocol(value)
    if checked != value:
        raise ValueError("Invalid saved preprocessing protocol")
    return checked


def signal_power(signal, centered=False):
    if centered:
        return np.mean((signal - np.mean(signal)) ** 2)
    return np.mean(signal**2)


def require_noise_power(power):
    if not np.isfinite(power) or power <= 0:
        raise ValueError(
            "Finite-SNR noise requires positive finite reference signal power"
        )
    return power


def noise_record(clean, observed, snr, preprocessing, **context):
    saved = validate(preprocessing)
    total = float(signal_power(clean))
    ac = float(signal_power(clean, True))
    noise = float(np.mean((observed - clean) ** 2))

    def db(power):
        return float(10 * np.log10(power / noise)) if power > 0 and noise > 0 else None

    actual_total, actual_ac = (db(total), db(ac))
    basis = actual_ac if saved["noise_power_centered"] else actual_total
    error = None if snr is None or basis is None else basis - float(snr)
    if snr is not None and (error is None or abs(error) > 1e-07):
        raise ValueError(
            f"Actual SNR does not match the selected power definition: {context}, error={error}"
        )
    return {
        **context,
        "Protocol_Version": VERSION,
        **{key: saved[key] for key in KEYS},
        "window_normalization": saved["window_normalization"],
        "Sample_Count": int(len(clean)),
        "Clean": snr is None,
        "Target_SNR_dB": None if snr is None else float(snr),
        "Signal_Total_Power": total,
        "Signal_AC_Power": ac,
        "Noise_Power": noise,
        "Actual_Total_SNR_dB": actual_total,
        "Actual_AC_SNR_dB": actual_ac,
        "Target_Basis_SNR_dB": basis,
        "SNR_Error_dB": error,
    }


def normalize_windows(windows, preprocessing):
    saved = validate(preprocessing)
    mode = saved["window_normalization"]
    if mode == "none":
        return windows
    x = np.asarray(windows, dtype=np.float64)
    if x.ndim != 2 or x.shape[-1] < 2 or (not np.isfinite(x).all()):
        raise ValueError("Expected finite 2D windows")
    if mode == "zscore":
        centered = x - x.mean(axis=-1, keepdims=True)
        scale = x.std(axis=-1, keepdims=True, ddof=0)
        numerator = centered
    else:
        numerator = x
        scale = np.sqrt(np.mean(x * x, axis=-1, keepdims=True))
    if not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError("Cannot normalize a zero-scale window")
    return numerator / (scale + WINDOW_EPSILON)


def window_record(windows, preprocessing, **context):
    saved = validate(preprocessing)
    x = np.asarray(windows, dtype=np.float32).astype(np.float64)
    means = x.mean(-1)
    std = x.std(-1)
    rms = np.sqrt(np.mean(x * x, axis=-1))
    return {
        **context,
        "Protocol_Version": VERSION,
        **{key: saved[key] for key in KEYS},
        "window_normalization": saved["window_normalization"],
        "Window_Count": len(x),
        "Window_Length": x.shape[-1],
        "Mean_Abs_Max": float(np.max(np.abs(means))),
        "STD_Min": float(std.min()),
        "STD_Max": float(std.max()),
        "RMS_Min": float(rms.min()),
        "RMS_Max": float(rms.max()),
    }
