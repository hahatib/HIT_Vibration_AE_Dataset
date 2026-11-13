"""Original CNN models, raw-split preprocessing and training calculations.

DEVICE and PREPROCESSING are selected once by each experiment adapter.
"""

from __future__ import annotations
import copy
import os
import random
from dataclasses import dataclass
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tqdm import tqdm
from . import input_standardization as input_norm
from . import preprocessing_protocol as pp
from common.data_columns import align_modal_dataframe_columns

PREPROCESSING = pp.protocol()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32

TRAIN_RATIO = 0.6

VALIDATION_RATIO = 0.075

CALIBRATION_RATIO = 0.075

TEST_RATIO = 0.25

SPLIT_NAMES = ("train", "validation", "calibration", "test")

PAIRING_RULE = "source_order_even_subset"

VIBRATION_HP = {"window": 512, "step": 51}

AE_HP = {"window": 16384, "step": 16384}


def set_reproducibility(seed: int):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def sampling(signal, step_size, window_size):
    samples = []
    for start in range(0, len(signal) - window_size + 1, step_size):
        samples.append(signal[start : start + window_size])
    return np.asarray(samples)


def add_gaussian_noise(signal, snr, centered=False):
    noise = np.random.randn(len(signal))
    signal_power = pp.require_noise_power(pp.signal_power(signal, centered))
    noise_power = np.mean(noise**2)
    if noise_power <= 0:
        return signal.copy()
    target_noise_power = signal_power / 10 ** (float(snr) / 10.0)
    return signal + np.sqrt(target_noise_power / noise_power) * noise


def classification_metrics(labels, predictions):
    labels = np.asarray(labels, dtype=int)
    predictions = np.asarray(predictions, dtype=int)
    return {
        "Accuracy": 100.0 * accuracy_score(labels, predictions),
        "Precision": 100.0
        * precision_score(labels, predictions, average="weighted", zero_division=0),
        "Recall": 100.0
        * recall_score(labels, predictions, average="weighted", zero_division=0),
        "F1": 100.0
        * f1_score(labels, predictions, average="weighted", zero_division=0),
        "Macro_F1": 100.0
        * f1_score(labels, predictions, average="macro", zero_division=0),
    }


SPLIT_COUNT_COLUMNS = {
    "train": "Train_Samples",
    "validation": "Validation_Samples",
    "calibration": "Calibration_Samples",
    "test": "Test_Samples",
}


def _append_split(container_x, container_y, samples, labels):
    if container_x is None:
        return (samples, labels)
    return (
        np.append(container_x, samples, axis=0),
        np.append(container_y, labels, axis=0),
    )


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
            samples = pp.normalize_windows(
                sampling(segment, step_size, window_size), preprocessing
            )
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


def evenly_spaced_indices(count, target):
    if target <= 0 or count < target:
        raise ValueError(
            f"Invalid source-order pairing count={count}, target={target}."
        )
    if count == target:
        return np.arange(count, dtype=int)
    return np.round(np.linspace(0, count - 1, target)).astype(int)


def align_four_splits_by_source_order(prepared_a, audit_a, prepared_b, audit_b):
    audit_a = audit_a.reset_index(drop=True)
    audit_b = audit_b.reset_index(drop=True)
    if len(audit_a) != len(audit_b):
        raise ValueError("The two sources contain different class counts.")
    paired = {}
    audit_rows = []
    for split_name, count_column in SPLIT_COUNT_COLUMNS.items():
        indices_a, indices_b = ([], [])
        offset_a = 0
        offset_b = 0
        for class_index in range(len(audit_a)):
            count_a = int(audit_a.loc[class_index, count_column])
            count_b = int(audit_b.loc[class_index, count_column])
            target = min(count_a, count_b)
            indices_a.append(offset_a + evenly_spaced_indices(count_a, target))
            indices_b.append(offset_b + evenly_spaced_indices(count_b, target))
            audit_rows.append(
                {
                    "Split": split_name,
                    "Class_Index": int(class_index),
                    "Source_A_Class_Column": str(
                        audit_a.loc[class_index, "Class_Column"]
                    ),
                    "Source_B_Class_Column": str(
                        audit_b.loc[class_index, "Class_Column"]
                    ),
                    "Source_A_Original": count_a,
                    "Source_B_Original": count_b,
                    "Paired_Count": target,
                    "Pairing_Rule": PAIRING_RULE,
                }
            )
            offset_a += count_a
            offset_b += count_b
        index_a = np.concatenate(indices_a)
        index_b = np.concatenate(indices_b)
        paired[split_name] = {
            "x_a": prepared_a[split_name]["x"][index_a],
            "y_a": prepared_a[split_name]["y"][index_a],
            "x_b": prepared_b[split_name]["x"][index_b],
            "y_b": prepared_b[split_name]["y"][index_b],
        }
        labels_a = np.argmax(paired[split_name]["y_a"], axis=1)
        labels_b = np.argmax(paired[split_name]["y_b"], axis=1)
        if not np.array_equal(labels_a, labels_b):
            raise AssertionError(f"Paired labels differ for split={split_name}.")
    return (paired, pd.DataFrame(audit_rows))


class ClassificationDataset(Dataset):

    def __init__(self, features, labels):
        self.features = torch.from_numpy(np.asarray(features)).float().unsqueeze(1)
        self.labels = torch.from_numpy(
            np.argmax(labels, axis=1).astype(np.int64)
        ).long()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return (self.features[index], self.labels[index])


class PairedDataset(Dataset):

    def __init__(self, features_a, features_b, labels_a, labels_b):
        class_a = np.argmax(labels_a, axis=1).astype(np.int64)
        class_b = np.argmax(labels_b, axis=1).astype(np.int64)
        if not np.array_equal(class_a, class_b):
            raise ValueError("Paired source labels are not identical.")
        self.features_a = torch.from_numpy(np.asarray(features_a)).float().unsqueeze(1)
        self.features_b = torch.from_numpy(np.asarray(features_b)).float().unsqueeze(1)
        self.labels = torch.from_numpy(class_a).long()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return (self.features_a[index], self.features_b[index], self.labels[index])


def make_loader(dataset, shuffle, seed):
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=bool(shuffle),
        generator=generator,
        num_workers=0,
        pin_memory=DEVICE.type == "cuda",
    )


class SmallKernelCNN1D(nn.Module):

    def __init__(self, window_size, num_classes):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, stride=2), nn.ReLU(), nn.MaxPool1d(2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=3, stride=2), nn.ReLU(), nn.MaxPool1d(2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3, stride=2), nn.ReLU(), nn.MaxPool1d(2)
        )
        self.layer4 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, stride=2), nn.ReLU(), nn.MaxPool1d(2)
        )
        self.flatten = nn.Flatten()
        with torch.no_grad():
            dummy = torch.randn(1, 1, int(window_size))
            hidden = self.layer4(self.layer3(self.layer2(self.layer1(dummy))))
            flattened_size = int(hidden.view(1, -1).size(1))
        self.fc_layers = nn.Sequential(
            nn.Linear(flattened_size, 100),
            nn.ReLU(),
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, int(num_classes)),
        )
        self.feature_dim = 50

    def extract_features(self, inputs):
        feature_map = self.layer1(inputs)
        feature_map = self.layer2(feature_map)
        feature_map = self.layer3(feature_map)
        feature_map = self.layer4(feature_map)
        flattened = self.flatten(feature_map)
        embedding = self.fc_layers[:-1](flattened)
        logits = self.fc_layers[-1](embedding)
        return (embedding, logits, feature_map)

    def forward(self, inputs):
        return self.extract_features(inputs)[1]


class SqueezeExcitation1D(nn.Module):

    def __init__(self, channels, ratio=8):
        super().__init__()
        hidden = max(channels // ratio, 8)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, hidden, 1),
            nn.GELU(),
            nn.Conv1d(hidden, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, inputs):
        return inputs * self.gate(inputs)


class DilatedResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation):
        super().__init__()
        self.convolution_1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            bias=False,
        )
        self.norm_1 = nn.BatchNorm1d(out_channels)
        self.convolution_2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size,
            padding=dilation * (kernel_size // 2),
            dilation=dilation,
            bias=False,
        )
        self.norm_2 = nn.BatchNorm1d(out_channels)
        self.skip = (
            nn.Conv1d(in_channels, out_channels, 1, stride=stride, bias=False)
            if in_channels != out_channels or stride != 1
            else nn.Identity()
        )
        self.channel_attention = SqueezeExcitation1D(out_channels)

    def forward(self, inputs):
        features = F.gelu(self.norm_1(self.convolution_1(inputs)))
        features = self.norm_2(self.convolution_2(features))
        return F.gelu(self.channel_attention(features) + self.skip(inputs))


class VibrationMultiScaleDilatedCNN(nn.Module):
    feature_dim = 128

    def __init__(self, num_classes):
        super().__init__()
        self.multi_scale_stem = nn.ModuleList(
            [
                nn.Conv1d(1, 16, kernel, stride=2, padding=kernel // 2, bias=False)
                for kernel in (15, 31, 63)
            ]
        )
        self.stem_norm = nn.BatchNorm1d(48)
        self.block1 = DilatedResidualBlock(48, 64, 7, 2, 2)
        self.block2 = DilatedResidualBlock(64, 96, 5, 2, 4)
        self.block3 = DilatedResidualBlock(96, 128, 3, 2, 8)
        self.embedding = nn.Sequential(
            nn.Linear(256, self.feature_dim),
            nn.LayerNorm(self.feature_dim),
            nn.GELU(),
            nn.Dropout(0.2),
        )
        self.classifier = nn.Linear(self.feature_dim, int(num_classes))

    def extract_features(self, inputs):
        features = torch.cat([layer(inputs) for layer in self.multi_scale_stem], dim=1)
        features = F.gelu(self.stem_norm(features))
        features = self.block1(features)
        features = self.block2(features)
        feature_map = self.block3(features)
        pooled = torch.cat([feature_map.mean(dim=-1), feature_map.amax(dim=-1)], dim=1)
        embedding = self.embedding(pooled)
        logits = self.classifier(embedding)
        return (embedding, logits, feature_map)

    def forward(self, inputs):
        return self.extract_features(inputs)[1]


def evaluate_branch(model, loader):
    model.eval()
    labels_all, probabilities_all, features_all = ([], [], [])
    nll_sum = 0.0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)
            features, logits, _ = model.extract_features(inputs)
            nll_sum += F.cross_entropy(logits, labels, reduction="sum").item()
            labels_all.append(labels.cpu().numpy())
            probabilities_all.append(F.softmax(logits, dim=1).cpu().numpy())
            features_all.append(features.cpu().numpy())
    labels_all = np.concatenate(labels_all).astype(np.int64)
    probabilities_all = np.concatenate(probabilities_all).astype(np.float64)
    features_all = np.concatenate(features_all).astype(np.float32)
    predictions = probabilities_all.argmax(axis=1)
    return {
        "Labels": labels_all,
        "Probabilities": probabilities_all,
        "Features": features_all,
        "NLL": float(nll_sum / len(labels_all)),
        **classification_metrics(labels_all, predictions),
    }


def evaluate_paired_branches(model_a, model_b, loader):
    model_a.eval()
    model_b.eval()
    labels_all = []
    probabilities_a_all, probabilities_b_all = ([], [])
    features_a_all, features_b_all = ([], [])
    with torch.no_grad():
        for inputs_a, inputs_b, labels in loader:
            inputs_a = inputs_a.to(DEVICE, non_blocking=True)
            inputs_b = inputs_b.to(DEVICE, non_blocking=True)
            features_a, logits_a, _ = model_a.extract_features(inputs_a)
            features_b, logits_b, _ = model_b.extract_features(inputs_b)
            labels_all.append(labels.numpy())
            probabilities_a_all.append(F.softmax(logits_a, dim=1).cpu().numpy())
            probabilities_b_all.append(F.softmax(logits_b, dim=1).cpu().numpy())
            features_a_all.append(features_a.cpu().numpy())
            features_b_all.append(features_b.cpu().numpy())
    return {
        "Labels": np.concatenate(labels_all).astype(np.int64),
        "Probabilities_A": np.concatenate(probabilities_a_all).astype(np.float64),
        "Probabilities_B": np.concatenate(probabilities_b_all).astype(np.float64),
        "Features_A": np.concatenate(features_a_all).astype(np.float32),
        "Features_B": np.concatenate(features_b_all).astype(np.float32),
    }


def _is_better(accuracy, nll, epoch, best):
    if best is None or accuracy > best["Accuracy"] + 1e-12:
        return True
    if np.isclose(accuracy, best["Accuracy"], rtol=0.0, atol=1e-12):
        if nll < best["NLL"] - 1e-12:
            return True
        if np.isclose(nll, best["NLL"], rtol=0.0, atol=1e-12):
            return epoch < best["Epoch"]
    return False


def train_branch(model, train_loader, validation_loader, epochs, model_name):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    best = None
    history = []
    progress = tqdm(range(1, int(epochs) + 1), desc=f"Train {model_name}", unit="epoch")
    for epoch in progress:
        model.train()
        running_loss = 0.0
        sample_count = 0
        for inputs, labels in train_loader:
            inputs = inputs.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)
            optimizer.zero_grad()
            logits = model(inputs)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(labels)
            sample_count += len(labels)
        validation = evaluate_branch(model, validation_loader)
        train_loss = running_loss / max(sample_count, 1)
        history.append(
            {
                "Epoch": int(epoch),
                "Train_Loss": float(train_loss),
                "Validation_Accuracy": float(validation["Accuracy"]),
                "Validation_NLL": float(validation["NLL"]),
            }
        )
        if _is_better(validation["Accuracy"], validation["NLL"], epoch, best):
            best = {
                "Accuracy": float(validation["Accuracy"]),
                "NLL": float(validation["NLL"]),
                "Epoch": int(epoch),
                "State": copy.deepcopy(model.state_dict()),
            }
        progress.set_postfix(
            train_loss=f"{train_loss:.4f}", val_acc=f"{validation['Accuracy']:.2f}"
        )
    if best is None:
        raise RuntimeError(f"No checkpoint selected for {model_name}.")
    model.load_state_dict(best["State"])
    return (best, history)


@dataclass
class PreparedTwoSourceData:
    source_a: dict
    source_b: dict
    paired: dict
    source_a_audit: pd.DataFrame
    source_b_audit: pd.DataFrame
    pairing_audit: pd.DataFrame
    detected_class_names: list[str]
    display_class_names: list[str]


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


def native_branch_loaders(data: PreparedTwoSourceData, seed):
    return {
        "train_a": make_loader(
            ClassificationDataset(
                data.source_a["train"]["x"], data.source_a["train"]["y"]
            ),
            True,
            seed + 1,
        ),
        "validation_a": make_loader(
            ClassificationDataset(
                data.source_a["validation"]["x"], data.source_a["validation"]["y"]
            ),
            False,
            seed + 2,
        ),
        "train_b": make_loader(
            ClassificationDataset(
                data.source_b["train"]["x"], data.source_b["train"]["y"]
            ),
            True,
            seed + 3,
        ),
        "validation_b": make_loader(
            ClassificationDataset(
                data.source_b["validation"]["x"], data.source_b["validation"]["y"]
            ),
            False,
            seed + 4,
        ),
    }


def paired_loader(data: PreparedTwoSourceData, split, seed):
    values = data.paired[split]
    dataset = PairedDataset(values["x_a"], values["x_b"], values["y_a"], values["y_b"])
    split_seed = {"calibration": 31, "test": 41, "train": 11, "validation": 21}[split]
    return make_loader(dataset, shuffle=False, seed=seed + split_seed)
