import numpy as np
from common.primary import preprocessing_protocol as legacy

WINDOW_EPSILON = legacy.WINDOW_EPSILON
VERSION = 4
KEYS = legacy.KEYS


def protocol(value=None):
    value = {} if value is None else dict(value)
    if "ae_window_normalization" not in value and value.get("version", 3) == 3:
        return legacy.protocol(value)
    if value.get("version", VERSION) != VERSION:
        raise ValueError("Unsupported AE DC preprocessing version")
    mode = value.pop("ae_window_normalization", None)
    if mode != "std_only":
        raise ValueError("AE DC supplement requires ae_window_normalization=std_only")
    value.pop("version", None)
    result = legacy.protocol(value)
    if (
        result["ae_train_dc_removal"]
        or result["pooled_train_zscore"]
        or (not result["noise_power_centered"])
        or (result["window_normalization"] != "zscore")
    ):
        raise ValueError(
            "AE DC supplement must preserve raw AE offset and original AC SNR/vibration zscore"
        )
    return dict(result, version=VERSION, ae_window_normalization=mode)


def validate(value):
    if not isinstance(value, dict) or protocol(value) != value:
        raise ValueError("Missing or inconsistent saved preprocessing")
    return dict(value)


def legacy_view(value):
    saved = validate(value)
    return legacy.protocol({k: saved[k] for k in (*KEYS, "window_normalization")})


def normalize_windows(windows, preprocessing, modality=None):
    saved = validate(preprocessing)
    if saved.get("ae_window_normalization") == "std_only" and modality == "AE":
        x = np.asarray(windows, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] < 2 or (not np.isfinite(x).all()):
            raise ValueError("Expected finite 2D AE windows")
        scale = x.std(axis=-1, keepdims=True, ddof=0)
        if not np.isfinite(scale).all() or np.any(scale <= 0):
            raise ValueError("Cannot normalize a zero-scale AE window")
        return x / (scale + WINDOW_EPSILON)
    return legacy.normalize_windows(windows, legacy_view(saved))


def noise_record(clean, observed, snr, preprocessing, **context):
    saved = validate(preprocessing)
    row = legacy.noise_record(clean, observed, snr, legacy_view(saved), **context)
    row.update(
        Protocol_Version=saved["version"],
        ae_window_normalization=saved.get(
            "ae_window_normalization", saved["window_normalization"]
        ),
    )
    return row


def window_record(windows, preprocessing, **context):
    saved = validate(preprocessing)
    row = legacy.window_record(windows, legacy_view(saved), **context)
    row.update(
        Protocol_Version=saved["version"],
        ae_window_normalization=saved.get(
            "ae_window_normalization", saved["window_normalization"]
        ),
    )
    row["Effective_Window_Normalization"] = (
        row["ae_window_normalization"]
        if context.get("Source_Modality") == "AE"
        else saved["window_normalization"]
    )
    return row
