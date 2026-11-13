"""Original DC-removed V8_1 experiment, with 8-unit reliability heads."""

from pathlib import Path
import itertools
import numpy as np
import torch
from common.primary import branches as core
from common.primary.partition import partition_observations
from common.primary.fitting import fit_heads
from common.primary.evaluation import evaluate_heads
from .neural_fusion import SerialFusion, objective as original_objective

windows = {"A": (512, 51), "B": (16384, 16384)}
source_labels = ("Vibration", "AE")


def configure(config):
    core.DEVICE = torch.device(config["device"])
    core.PREPROCESSING = core.pp.protocol(config["preprocessing"])


def conditions(config):
    root = Path(config["laboratory_root"])
    return [
        dict(
            RPM=str(rpm),
            SNR=snr,
            Source_A=str(
                root
                / "Data_Vibration"
                / str(rpm)
                / "CH_06"
                / f"data_Vibration_1k_{rpm}_CH_06_4c.csv"
            ),
            Source_B=str(root / "Data_AE" / str(rpm) / f"data_AE_2000k_{rpm}_4c.csv"),
        )
        for rpm, snr in itertools.product(config["rpms"], config["snrs"])
    ]


def prepare(task, seed, config):
    configure(config)
    return core.prepare_two_source_data(
        task["Source_A"],
        task["Source_B"],
        core.VIBRATION_HP,
        core.AE_HP,
        seed,
        snr=task["SNR"],
        expected_class_count=4,
        source_a_name="Vibration",
        source_b_name="AE",
    )


def build_models(class_count, device):
    if class_count != 4:
        raise ValueError("Laboratory data require four classes")
    return (
        core.VibrationMultiScaleDilatedCNN(class_count).to(device),
        core.SmallKernelCNN1D(16384, class_count).to(device),
    )


def inputs(a, b, rpm, snr):
    a, b = (np.asarray(p, dtype=np.float64) for p in (a, b))
    if any(
        p.ndim != 2
        or not np.isfinite(p).all()
        or np.any(p < 0)
        or np.any(p.sum(1) <= 0)
        for p in (a, b)
    ):
        raise ValueError("Expected finite nonnegative probability matrices")
    a, b = a / a.sum(1, keepdims=True), b / b.sum(1, keepdims=True)
    if a.shape != b.shape or a.shape[1] != 4:
        raise ValueError("Expected paired four-class probabilities")
    clean = snr is None or str(snr) == "Clean"
    context = np.tile(
        [
            (int(rpm) - 1900) / 1100,
            ((6 if clean else float(snr)) + 6) / 12,
            float(clean),
        ],
        (len(a), 1),
    )
    return tuple(torch.as_tensor(x, dtype=torch.float32) for x in (a, b, context))


def fit_and_evaluate(observations, config, output):
    output = Path(output) / "Neural_CRD_DS"
    recipe = config["neural_modules"]
    if recipe.get("refinement", {}).get("enabled"):
        raise ValueError("The V8_1 DC reference has no refinement stage")

    def make_model(method, device):
        torch.manual_seed(recipe["initialization_seed"])
        return SerialFusion(method, recipe["heads"][method]["hidden"]).to(device)

    def objective(model, values, labels, weights, bce=0.05):
        if bce != 0.05:
            raise ValueError("The V8_1 reliability loss is fixed at 0.05")
        return original_objective(model, values, labels, weights)

    fit, selection, metadata = partition_observations(
        observations, windows, inputs, output
    )
    models = fit_heads(
        fit,
        selection,
        metadata,
        make_model=make_model,
        objective=objective,
        recipe=recipe,
        output=output,
        device=config["device"],
    )
    return evaluate_heads(observations, models, inputs, config, output, source_labels)
