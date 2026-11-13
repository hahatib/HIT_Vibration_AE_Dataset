"""CWRU DE/FE experiment: ten classes, unchanged 1024-point branch CNNs."""

from pathlib import Path
import itertools
import numpy as np
import torch
from common.primary import branches as core
from common.primary.partition import partition_observations
from common.primary.fitting import fit_heads
from common.primary.evaluation import evaluate_heads
from .neural_fusion import SerialFusion, objective

windows = {"A": (1024, 1024), "B": (1024, 1024)}
source_labels = ("DE", "FE")


def configure(config):
    core.DEVICE = torch.device(config["device"])
    core.PREPROCESSING = core.pp.protocol(
        dict(config["preprocessing"], ae_train_dc_removal=False)
    )


def conditions(config):
    root = Path(config["cwru_root"])
    return [
        dict(
            RPM=str(rpm),
            SNR=snr,
            Source_A=str(root / f"data_12k_{rpm}_10c_DE.csv"),
            Source_B=str(root / f"data_12k_{rpm}_10c_FE.csv"),
        )
        for rpm, snr in itertools.product(config["rpms"], config["snrs"])
    ]


def prepare(task, seed, config):
    configure(config)
    hp = {"window": 1024, "step": 1024}
    return core.prepare_two_source_data(
        task["Source_A"],
        task["Source_B"],
        hp,
        hp,
        seed,
        snr=task["SNR"],
        expected_class_count=10,
        source_a_name="DE",
        source_b_name="FE",
    )


def build_models(class_count, device):
    if class_count != 10:
        raise ValueError("CWRU requires ten classes")
    return tuple(core.SmallKernelCNN1D(1024, class_count).to(device) for _ in range(2))


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
    if a.shape != b.shape or a.shape[1] != 10:
        raise ValueError("Expected paired ten-class probabilities")
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

    def make_model(method, device):
        torch.manual_seed(recipe["initialization_seed"])
        hp = recipe["heads"][method]
        return SerialFusion(
            method, hp["calibration_hidden"], hp["reliability_hidden"], 10
        ).to(device)

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
        allow_clean_only=True,
    )
    return evaluate_heads(observations, models, inputs, config, output, source_labels)
