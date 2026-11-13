"""Laboratory vibration/AE experiment using the current serial MLP method."""

from pathlib import Path
import numpy as np
import torch
from common.primary import branches
from common.primary.partition import partition_observations
from common.primary.fitting import fit_heads
from common.primary.evaluation import evaluate_heads
from .neural_fusion import SerialFusion, objective


def inputs(a, b, rpm, snr):
    arrays = []
    for p in (a, b):
        p = np.asarray(p, dtype=np.float64)
        if (
            p.ndim != 2
            or p.shape[1] != 4
            or not np.isfinite(p).all()
            or np.any(p < 0)
            or np.any(p.sum(1) <= 0)
        ):
            raise ValueError("Expected finite four-class probabilities")
        arrays.append(p / p.sum(1, keepdims=True))
    a, b = arrays
    if a.shape != b.shape:
        raise ValueError("Paired source shapes differ")
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


class Experiment:
    core = branches
    windows = {"A": (512, 51), "B": (16384, 16384)}
    source_labels = ("Vibration", "AE")

    def configure(self, config):
        branches.DEVICE = torch.device(config["device"])
        branches.PREPROCESSING = branches.pp.protocol(config["preprocessing"])

    def conditions(self, config):
        if len(config["channels"]) != 1:
            raise ValueError("Select one vibration channel")
        lab = Path(config["laboratory_root"])
        channel = config["channels"][0]
        for rpm in config["rpms"]:
            for snr in config["snrs"]:
                yield dict(
                    RPM=rpm,
                    SNR=None if snr is None or str(snr) == "Clean" else int(snr),
                    Source_A=lab
                    / "Data_Vibration"
                    / str(rpm)
                    / channel
                    / f"data_Vibration_1k_{rpm}_{channel}_4c.csv",
                    Source_B=lab / "Data_AE" / str(rpm) / f"data_AE_2000k_{rpm}_4c.csv",
                )

    def prepare(self, task, seed, config):
        return branches.prepare_two_source_data(
            task["Source_A"],
            task["Source_B"],
            dict(window=512, step=51),
            dict(window=16384, step=16384),
            seed,
            snr=task["SNR"],
            expected_class_count=4,
            source_a_name="Vibration",
            source_b_name="AE",
        )

    def build_models(self, class_count, device):
        return (
            branches.VibrationMultiScaleDilatedCNN(class_count).to(device),
            branches.SmallKernelCNN1D(16384, class_count).to(device),
        )

    def fit_and_evaluate(self, observations, config, output):
        output = Path(output) / "Neural_CRD_DS"
        recipe = config["neural_modules"]
        fit, selection, metadata = partition_observations(
            observations, self.windows, inputs, output
        )

        def make_model(method, device):
            torch.manual_seed(recipe["initialization_seed"])
            head = recipe["heads"][method]
            return SerialFusion(
                method, head["calibration_hidden"], head["reliability_hidden"]
            ).to(device)

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
        return evaluate_heads(
            observations, models, inputs, config, output, self.source_labels
        )
