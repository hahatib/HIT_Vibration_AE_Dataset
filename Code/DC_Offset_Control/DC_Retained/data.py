"""AE DC-retained input preparation; CNN computations are shared unchanged."""

import numpy as np
import pandas as pd
import os
from common.primary import input_standardization as input_norm
from common.data_columns import align_modal_dataframe_columns
from . import preprocessing as pp
from common.primary.branches import (
    SPLIT_NAMES,
    TRAIN_RATIO,
    VALIDATION_RATIO,
    CALIBRATION_RATIO,
    TEST_RATIO,
    SPLIT_COUNT_COLUMNS,
    PreparedTwoSourceData,
    set_reproducibility,
    sampling,
    add_gaussian_noise,
    _append_split,
    align_four_splits_by_source_order,
)

PREPROCESSING = pp.protocol()
WINDOW_OBSERVER = None


def prepare_four_way_split(
    dataframe,
    step_size,
    window_size,
    number_of_classes,
    snr=None,
    normalize=True,
    preprocessing=None,
    modality=None,
    source_file=None,
):
    containers = {name: [None, None] for name in SPLIT_NAMES}
    audit_rows = []
    noise_rows = []
    window_rows = []
    preprocessing = pp.protocol(
        PREPROCESSING if preprocessing is None else preprocessing
    )
    remove_dc = preprocessing["ae_train_dc_removal"] and modality == "AE"
    if type(normalize) is not bool:
        raise ValueError("normalize must be boolean")
    if not normalize and preprocessing["pooled_train_zscore"]:
        raise ValueError(
            "Disable pooled TRAIN Z-score explicitly in preprocessing before normalize=False"
        )
    if (
        TRAIN_RATIO,
        VALIDATION_RATIO,
        CALIBRATION_RATIO,
        TEST_RATIO,
    ) != input_norm.SPLIT_RATIOS:
        raise ValueError("Main and shared preprocessing split ratios differ")
    raw_splits = input_norm.split_raw_signals(dataframe)
    if len(raw_splits) != number_of_classes:
        raise ValueError("Class count does not match the raw modality table")
    fitted = input_norm.fit_pooled_training_stats(
        raw_splits,
        remove_dc=remove_dc,
        pooled_train_zscore=preprocessing["pooled_train_zscore"],
    )
    for class_index, (column, boundaries, raw_segments) in enumerate(raw_splits):
        segments = {name: signal.copy() for name, signal in raw_segments.items()}
        train_min = float(np.min(segments["train"]))
        train_max = float(np.max(segments["train"]))
        train_mean = fitted["mean"]
        train_std = fitted["std"]
        for split_name in SPLIT_NAMES:
            segments[split_name] = input_norm.apply_record_zscore(
                segments[split_name], fitted, column
            )
        for split_name in SPLIT_NAMES:
            clean = segments[split_name]
            if snr is not None:
                segments[split_name] = add_gaussian_noise(
                    clean, snr, preprocessing["noise_power_centered"]
                )
            noise_rows.append(
                pp.noise_record(
                    clean,
                    segments[split_name],
                    snr,
                    preprocessing,
                    Source_Modality=modality or "unspecified",
                    Source_File=str(source_file or ""),
                    Class_Column=column,
                    Split=split_name,
                    Stage="reference_fixed",
                    Epoch=0,
                )
            )
        row = {
            "Class_Index": int(class_index),
            "Class_Column": str(column),
            "Raw_Length": int(boundaries[-1]),
            "Train_Raw_Start": int(boundaries[0]),
            "Train_Raw_End_Exclusive": int(boundaries[1]),
            "Validation_Raw_Start": int(boundaries[1]),
            "Validation_Raw_End_Exclusive": int(boundaries[2]),
            "Calibration_Raw_Start": int(boundaries[2]),
            "Calibration_Raw_End_Exclusive": int(boundaries[3]),
            "Test_Raw_Start": int(boundaries[3]),
            "Test_Raw_End_Exclusive": int(boundaries[4]),
            "Normalization": input_norm.NORMALIZATION,
            "Normalization_Mode": fitted["mode"],
            "Pooled_Train_ZScore": fitted["pooled_train_zscore"],
            "Train_Mean": train_mean,
            "Train_Std": train_std,
            "Diagnostic_Pooled_Train_Mean": fitted["diagnostic_pooled_train_mean"],
            "Diagnostic_Pooled_Train_Std": fitted["diagnostic_pooled_train_std"],
            "Normalization_Scope": input_norm.SPEC["scope"],
            "Normalization_Training_Samples": fitted["training_sample_count"],
            "Normalization_Class_Count": len(raw_splits),
            "Standardization_ddof": 0,
            "Explicit_DC_Removal": remove_dc,
            "Record_Train_DC_Offset": fitted["dc_offsets"][column],
            "Noise_Power_Centered": preprocessing["noise_power_centered"],
            "Train_Min": train_min,
            "Train_Max": train_max,
        }
        for split_name, segment in segments.items():
            raw_windows = sampling(segment, step_size, window_size)
            samples = pp.normalize_windows(raw_windows, preprocessing, modality)
            if WINDOW_OBSERVER is not None:
                WINDOW_OBSERVER(modality, column, split_name, raw_windows, samples)
            if len(samples) == 0:
                raise ValueError(
                    f"No {split_name} windows for class={column}, window={window_size}, step={step_size}."
                )
            window_rows.append(
                pp.window_record(
                    samples,
                    preprocessing,
                    Source_Modality=modality,
                    Class_Column=column,
                    Split=split_name,
                )
            )
            labels = np.zeros((len(samples), number_of_classes), dtype=np.float32)
            labels[:, class_index] = 1.0
            containers[split_name][0], containers[split_name][1] = _append_split(
                containers[split_name][0], containers[split_name][1], samples, labels
            )
            row[SPLIT_COUNT_COLUMNS[split_name]] = int(len(samples))
        audit_rows.append(row)
    prepared = {
        name: {"x": values[0].astype(np.float32), "y": values[1].astype(np.float32)}
        for name, values in containers.items()
    }
    audit = pd.DataFrame(audit_rows)
    audit.attrs["Input_Normalization"] = fitted
    audit.attrs["Preprocessing"] = preprocessing
    audit.attrs["Noise_Audit"] = noise_rows
    audit.attrs["Window_Audit"] = window_rows
    return (prepared, audit)


def prepare_two_source_data(
    csv_a,
    csv_b,
    hp_a,
    hp_b,
    seed,
    snr=None,
    expected_class_count=None,
    source_a_name="Source A",
    source_b_name="Source B",
    display_class_names=None,
):
    if not os.path.isfile(csv_a):
        raise FileNotFoundError(f"{source_a_name} CSV not found: {csv_a}")
    if not os.path.isfile(csv_b):
        raise FileNotFoundError(f"{source_b_name} CSV not found: {csv_b}")
    set_reproducibility(seed)
    frame_a = pd.read_csv(csv_a)
    frame_b = pd.read_csv(csv_b)
    frame_a, frame_b, detected_names = align_modal_dataframe_columns(
        frame_a, frame_b, f"{source_a_name} table", f"{source_b_name} table"
    )
    detected_names = [str(value) for value in detected_names]
    if expected_class_count is not None and len(detected_names) != int(
        expected_class_count
    ):
        raise ValueError(
            f"Expected {expected_class_count} aligned classes, found {len(detected_names)}: {detected_names}"
        )
    prepared_a, audit_a = prepare_four_way_split(
        frame_a,
        hp_a["step"],
        hp_a["window"],
        len(detected_names),
        snr=snr,
        normalize=True,
        preprocessing=PREPROCESSING,
        modality=source_a_name,
        source_file=csv_a,
    )
    prepared_b, audit_b = prepare_four_way_split(
        frame_b,
        hp_b["step"],
        hp_b["window"],
        len(detected_names),
        snr=snr,
        normalize=True,
        preprocessing=PREPROCESSING,
        modality=source_b_name,
        source_file=csv_b,
    )
    paired, pairing_audit = align_four_splits_by_source_order(
        prepared_a, audit_a, prepared_b, audit_b
    )
    if display_class_names is None:
        display_names = list(detected_names)
    else:
        display_names = [str(value) for value in display_class_names]
        if len(display_names) != len(detected_names):
            raise ValueError(
                "display_class_names length does not match detected class count."
            )
    return PreparedTwoSourceData(
        source_a=prepared_a,
        source_b=prepared_b,
        paired=paired,
        source_a_audit=audit_a,
        source_b_audit=audit_b,
        pairing_audit=pairing_audit,
        detected_class_names=detected_names,
        display_class_names=display_names,
    )
