from __future__ import annotations
import copy
import numpy as np

NORMALIZATION = "rpm_modality_train_preprocessing"
SPLIT_NAMES = ("train", "validation", "calibration", "test")
SPLIT_RATIOS = (0.6, 0.075, 0.075, 0.25)
EPSILON = 1e-12
SPEC = {
    "name": NORMALIZATION,
    "version": 3,
    "scope": "one_RPM_one_modality_all_classes_TRAIN_after_optional_record_DC",
    "order": [
        "chronological_raw_split",
        "optional_per_record_AE_TRAIN_DC_removal",
        "optional_pooled_corrected_TRAIN_fit_and_transform",
        "configured_AWGN",
        "window",
    ],
    "formula": "corrected=raw-optional_saved_record_TRAIN_offset; pooled_enabled ? (corrected-pooled_TRAIN_mean)/(pooled_TRAIN_std+1e-12) : corrected",
    "std_ddof": 0,
    "epsilon": EPSILON,
    "per_window_standardization": False,
    "fit_validation_calibration_or_test": False,
    "mix_RPMs_or_modalities": False,
}


def chronological_boundaries(length):
    if type(length) is not int or length < 1:
        raise ValueError("Raw signal length must be a positive integer")
    bounds = [
        0,
        int(round(length * SPLIT_RATIOS[0])),
        int(round(length * (SPLIT_RATIOS[0] + SPLIT_RATIOS[1]))),
        int(round(length * (SPLIT_RATIOS[0] + SPLIT_RATIOS[1] + SPLIT_RATIOS[2]))),
        length,
    ]
    if any((b <= a for a, b in zip(bounds, bounds[1:]))):
        raise ValueError("Every chronological split must contain raw samples")
    return bounds


def split_raw_signals(frame):
    if len(frame.columns) == 0 or frame.columns.has_duplicates:
        raise ValueError("Expected unique fault-class columns in one modality table")
    records = []
    for column in frame.columns:
        raw = frame[column].to_numpy(dtype=np.float64)
        if raw.ndim != 1 or not np.isfinite(raw).all():
            raise ValueError(f"Non-finite or invalid raw class signal: {column}")
        bounds = chronological_boundaries(len(raw))
        segments = {
            name: raw[bounds[i] : bounds[i + 1]] for i, name in enumerate(SPLIT_NAMES)
        }
        records.append((str(column), bounds, segments))
    return records


def fit_pooled_training_stats(raw_splits, remove_dc=False, pooled_train_zscore=True):
    if not raw_splits:
        raise ValueError("No class TRAIN segments")
    names, trains = ([], [])
    for name, bounds, segments in raw_splits:
        train = np.asarray(segments["train"], dtype=np.float64)
        if (
            train.ndim != 1
            or len(train) != bounds[1] - bounds[0]
            or len(train) == 0
            or (not np.isfinite(train).all())
        ):
            raise ValueError(f"Invalid raw TRAIN segment: {name}")
        names.append(str(name))
        trains.append(train)
    if len(set(names)) != len(names):
        raise ValueError("Duplicate TRAIN class records")
    if type(remove_dc) is not bool or type(pooled_train_zscore) is not bool:
        raise ValueError("remove_dc and pooled_train_zscore must be boolean")
    offsets = {
        name: float(np.mean(train)) if remove_dc else 0.0
        for name, train in zip(names, trains)
    }
    if remove_dc:
        trains = [train - offsets[name] for name, train in zip(names, trains)]
    pooled = np.concatenate(trains)
    mean = float(np.mean(pooled))
    std = float(np.std(pooled, ddof=0))
    mode = (
        "pooled_train_zscore"
        if pooled_train_zscore
        else "record_train_dc_only" if remove_dc else "identity"
    )
    value = {
        "name": NORMALIZATION,
        "version": 3,
        "scope": SPEC["scope"],
        "mode": mode,
        "pooled_train_zscore": pooled_train_zscore,
        "mean": mean if pooled_train_zscore else 0.0,
        "std": std if pooled_train_zscore else 1.0,
        "ddof": 0,
        "epsilon": EPSILON,
        "diagnostic_pooled_train_mean": mean,
        "diagnostic_pooled_train_std": std,
        "pooled_disabled_transform": "identity_without_epsilon",
        "class_columns": names,
        "training_counts": [len(x) for x in trains],
        "record_boundaries": {
            str(name): [int(x) for x in bounds] for name, bounds, _ in raw_splits
        },
        "training_sample_count": int(pooled.size),
        "split_ratios": list(SPLIT_RATIOS),
        "dc_removal": remove_dc,
        "dc_offsets": offsets,
        "dc_fit_scope": "per_original_record_raw_TRAIN_only",
    }
    validate_stats(value)
    return value


def validate_stats(stats):
    if (
        not isinstance(stats, dict)
        or stats.get("name") != NORMALIZATION
        or stats.get("version") != 3
    ):
        raise ValueError(
            "Expected v3 TRAIN preprocessing statistics; legacy weights/caches cannot be reused"
        )
    if (
        stats.get("scope") != SPEC["scope"]
        or stats.get("ddof") != 0
        or stats.get("epsilon") != EPSILON
    ):
        raise ValueError("Standardization scope, ddof or epsilon differs")
    if stats.get("split_ratios") != list(SPLIT_RATIOS):
        raise ValueError("Chronological split ratios differ")
    if (
        not np.isfinite(stats.get("mean", np.nan))
        or not np.isfinite(stats.get("std", np.nan))
        or stats["std"] <= 0
    ):
        raise ValueError("Pooled TRAIN standard deviation must be finite and positive")
    enabled = stats.get("pooled_train_zscore")
    mode = (
        "pooled_train_zscore"
        if enabled
        else "record_train_dc_only" if stats.get("dc_removal") else "identity"
    )
    diagnostic_mean = stats.get("diagnostic_pooled_train_mean", np.nan)
    diagnostic_std = stats.get("diagnostic_pooled_train_std", np.nan)
    if (
        type(enabled) is not bool
        or stats.get("mode") != mode
        or stats.get("pooled_disabled_transform") != "identity_without_epsilon"
        or (not np.isfinite(diagnostic_mean))
        or (not np.isfinite(diagnostic_std))
        or (diagnostic_std < 0)
        or (
            enabled
            and (stats["mean"] != diagnostic_mean or stats["std"] != diagnostic_std)
        )
        or (not enabled and (stats["mean"] != 0.0 or stats["std"] != 1.0))
    ):
        raise ValueError(
            "Pooled TRAIN mode, diagnostics or effective identity statistics differ"
        )
    columns, counts = (stats.get("class_columns", []), stats.get("training_counts", []))
    if (
        not columns
        or len(set(columns)) != len(columns)
        or len(counts) != len(columns)
        or any((type(n) is not int or n < 1 for n in counts))
        or (sum(counts) != stats.get("training_sample_count"))
    ):
        raise ValueError("Invalid pooled TRAIN sample accounting")
    bounds = stats.get("record_boundaries")
    if (
        not isinstance(bounds, dict)
        or set(bounds) != set(columns)
        or any(
            (
                not isinstance(bounds[name], list)
                or len(bounds[name]) != 5
                or any((type(x) is not int for x in bounds[name]))
                or (bounds[name] != chronological_boundaries(bounds[name][-1]))
                or (bounds[name][1] - bounds[name][0] != count)
                for name, count in zip(columns, counts)
            )
        )
    ):
        raise ValueError("Saved raw-record split boundaries differ")
    offsets = stats.get("dc_offsets")
    if (
        type(stats.get("dc_removal")) is not bool
        or not isinstance(offsets, dict)
        or set(offsets) != set(columns)
        or (not all((np.isfinite(x) for x in offsets.values())))
        or (stats.get("dc_fit_scope") != "per_original_record_raw_TRAIN_only")
        or (not stats["dc_removal"] and any((x != 0.0 for x in offsets.values())))
    ):
        raise ValueError("Invalid saved per-record TRAIN DC offsets")
    return stats


def apply_record_zscore(signal, stats, record):
    validate_stats(stats)
    if record not in stats["dc_offsets"]:
        raise ValueError(f"Unknown original record: {record}")
    corrected = np.asarray(signal, dtype=np.float64)
    if stats["dc_removal"]:
        corrected = corrected - stats["dc_offsets"][record]
    return apply_pooled_zscore(corrected, stats)


def apply_pooled_zscore(signal, stats):
    validate_stats(stats)
    if not stats["pooled_train_zscore"]:
        return np.asarray(signal, dtype=np.float64)
    return (np.asarray(signal, dtype=np.float64) - stats["mean"]) / (
        stats["std"] + EPSILON
    )


def stats_from_audit(audit):
    stats = audit.attrs.get("Input_Normalization")
    validate_stats(stats)
    if not audit.Normalization.eq(NORMALIZATION).all():
        raise ValueError("Preprocessing audit is not pooled TRAIN Z-score")
    if (
        not audit.Train_Mean.eq(stats["mean"]).all()
        or not audit.Train_Std.eq(stats["std"]).all()
    ):
        raise ValueError("Fault classes do not share the fitted modality statistics")
    if (
        not audit.Pooled_Train_ZScore.eq(stats["pooled_train_zscore"]).all()
        or not audit.Normalization_Mode.eq(stats["mode"]).all()
        or (
            not audit.Diagnostic_Pooled_Train_Mean.eq(
                stats["diagnostic_pooled_train_mean"]
            ).all()
        )
        or (
            not audit.Diagnostic_Pooled_Train_Std.eq(
                stats["diagnostic_pooled_train_std"]
            ).all()
        )
    ):
        raise ValueError("Pooled TRAIN audit mode or diagnostics differ")
    if (
        not audit.Explicit_DC_Removal.eq(stats["dc_removal"]).all()
        or list(audit.Class_Column) != stats["class_columns"]
        or any(
            (
                row.Record_Train_DC_Offset != stats["dc_offsets"][row.Class_Column]
                for row in audit.itertuples()
            )
        )
    ):
        raise ValueError("Per-record DC audit differs from saved TRAIN offsets")
    for row in audit.itertuples():
        bounds = stats["record_boundaries"][row.Class_Column]
        if (
            [
                row.Train_Raw_Start,
                row.Train_Raw_End_Exclusive,
                row.Validation_Raw_End_Exclusive,
                row.Calibration_Raw_End_Exclusive,
                row.Test_Raw_End_Exclusive,
            ]
            != bounds
            or row.Validation_Raw_Start != bounds[1]
            or row.Calibration_Raw_Start != bounds[2]
            or (row.Test_Raw_Start != bounds[3])
            or (row.Raw_Length != bounds[-1])
        ):
            raise ValueError("Raw-record boundary audit differs from saved TRAIN fit")
    return copy.deepcopy(stats)


def snapshot_from_data(data):
    return {
        "Source_A": stats_from_audit(data.source_a_audit),
        "Source_B": stats_from_audit(data.source_b_audit),
    }
