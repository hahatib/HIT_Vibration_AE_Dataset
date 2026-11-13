"""Eight scientific comparisons and the four-component subset."""

from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import precision_score, recall_score, f1_score
from .ds import build_discounted_bpa, combine_dempster, probability_nll
from .fitting import predict

METHODS = (
    "Full method",
    "Standard D-S",
    "Calibration only",
    "Reliability only",
    "Calibrated averaging",
    "Reliability-weighted averaging",
    "Vibration branch",
    "AE branch",
)


def evaluate_heads(
    observations, models, make_inputs, config, output, source_labels=("Vibration", "AE")
):
    """Evaluate frozen heads; this function performs no fitting or selection."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    packs, offsets = [], [0]
    for obs in observations:
        data = obs["test"]
        pack = make_inputs(data["A"], data["B"], obs["RPM"], obs["SNR"])
        packs.append(pack)
        offsets.append(offsets[-1] + len(pack[0]))
    pack = tuple(torch.cat([p[i] for p in packs]) for i in range(3))
    predicted = {
        m: predict(models[m], pack, config["device"])
        for m in ("Full", "Calibration only", "Reliability only")
    }
    method_names = METHODS[:-2] + tuple(label + " branch" for label in source_labels)
    rows = []
    for index, obs in enumerate(observations):
        sl = slice(offsets[index], offsets[index + 1])
        full, co, ro = (
            {k: v[sl] for k, v in predicted[m].items()}
            for m in ("Full", "Calibration only", "Reliability only")
        )
        data = obs["test"]
        a, b, labels = data["A"], data["B"], data["Labels"]
        standard = combine_dempster(
            build_discounted_bpa(a, 1.0), build_discounted_bpa(b, 1.0)
        ).probabilities
        probabilities = dict(
            zip(
                method_names,
                (
                    full["logp"],
                    standard,
                    co["logp"],
                    ro["logp"],
                    (full["loga"] + full["logb"]) / 2,
                    (full["r"][:, :1] * full["loga"] + full["r"][:, 1:] * full["logb"])
                    / full["r"].sum(1, keepdims=True),
                    a,
                    b,
                ),
            )
        )
        arrays = {
            "Labels": labels,
            "Raw_A": a,
            "Raw_B": b,
            "Full": full["logp"],
            "Full_Calibrated_A": full["loga"],
            "Full_Calibrated_B": full["logb"],
            "Full_Reliability": full["r"],
            "Full_Temperatures": full["logt"],
            "Full_Bias": full["bias"],
        }
        for method, p in probabilities.items():
            prediction = p.argmax(1)
            correct, count = int((prediction == labels).sum()), len(labels)
            rows.append(
                dict(
                    Method=method,
                    RPM=obs["RPM"],
                    SNR=obs["SNR"],
                    Seed=obs["Seed"],
                    Accuracy=100 * correct / count,
                    Correct_Count=correct,
                    Sample_Count=count,
                    Error_Count=count - correct,
                    NLL=probability_nll(p, labels),
                    Precision=100
                    * precision_score(
                        labels, prediction, average="weighted", zero_division=0
                    ),
                    Recall=100
                    * recall_score(
                        labels, prediction, average="weighted", zero_division=0
                    ),
                    F1=100
                    * f1_score(labels, prediction, average="weighted", zero_division=0),
                )
            )
            if method not in ("Full method", *method_names[-2:]):
                arrays[method + "_Probabilities"] = p
        target = output / "Predictions" / str(obs["RPM"]) / str(obs["SNR"])
        target.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(target / f'seed_{obs["Seed"]}.npz', **arrays)
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "Run_Metrics.csv", index=False)
    frame[frame.Method.isin(METHODS[:4])].to_csv(
        output / "Component_Metrics.csv", index=False
    )
    # Equal RPM means within each seed, then the mean and sample SD across seeds.
    by_seed = (
        frame.groupby(["Method", "SNR", "Seed"], sort=False)
        .Accuracy.mean()
        .reset_index()
    )
    summary = (
        by_seed.groupby(["Method", "SNR"], sort=False)
        .Accuracy.agg(["mean", "std"])
        .reset_index()
    )
    summary.rename(columns={"mean": "Accuracy_Mean", "std": "Accuracy_SD"}).to_csv(
        output / "Summary.csv", index=False
    )
    return frame
