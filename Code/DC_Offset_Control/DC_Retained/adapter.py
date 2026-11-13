"""DC-retained control: std-only AE, original Full head and window-mean baseline."""

from contextlib import contextmanager
from pathlib import Path
import torch
from common.primary import branches as core
from common.primary.partition import partition_observations
from common.primary.fitting import fit_heads
from DC_Offset_Control.DC_Removed.adapter import (
    conditions,
    build_models as build_models,
    inputs,
    windows,
    source_labels as source_labels,
)
from . import data, preprocessing
from .neural_fusion import SerialFusion, objective as original_objective


def check_reference(config):
    reference = Path(config["reference_results_root"])
    expected = [reference / "Neural_CRD_DS/Models/Full.pt"]
    for task in conditions(config):
        snr = "Clean" if task["SNR"] is None else str(task["SNR"])
        for seed in config["seeds"]:
            branch = reference / "Branches" / str(task["RPM"]) / snr / f"seed_{seed}"
            expected.extend(
                [
                    branch / "models.pt",
                    branch / "test.npz",
                    reference
                    / "Neural_CRD_DS/Predictions"
                    / str(task["RPM"])
                    / snr
                    / f"seed_{seed}.npz",
                ]
            )
    missing = next((path for path in expected if not path.is_file()), None)
    if missing is not None:
        raise FileNotFoundError(
            f"Run DC_Removed first; missing matched reference: {missing}"
        )


def configure(config):
    core.DEVICE = torch.device(config["device"])
    data.PREPROCESSING = preprocessing.protocol(config["preprocessing"])


def prepare(task, seed, config, protocol=None, observer=None):
    configure(config)
    previous, callback = data.PREPROCESSING, data.WINDOW_OBSERVER
    try:
        if protocol is not None:
            data.PREPROCESSING = preprocessing.protocol(protocol)
        data.WINDOW_OBSERVER = observer
        return data.prepare_two_source_data(
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
    finally:
        data.PREPROCESSING, data.WINDOW_OBSERVER = previous, callback


@contextmanager
def neural_runtime():
    matmul, cudnn = (
        torch.backends.cuda.matmul.allow_tf32,
        torch.backends.cudnn.allow_tf32,
    )
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        with torch.random.fork_rng():
            torch.use_deterministic_algorithms(True)
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            yield
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = (
            matmul,
            cudnn,
        )
        torch.use_deterministic_algorithms(deterministic, warn_only=warn)


def fit_and_evaluate(observations, config, output):
    from .controls import evaluate_controls

    folder = Path(output) / "Neural_CRD_DS"
    recipe = config["neural_modules"]
    if list(recipe["heads"]) != ["Full"] or recipe.get("refinement", {}).get("enabled"):
        raise ValueError("DC-retained fits only the original V8_1 Full head")

    def make_model(method, device):
        torch.manual_seed(recipe["initialization_seed"])
        return SerialFusion(method, recipe["heads"][method]["hidden"]).to(device)

    def objective(model, values, labels, weights, bce=0.05):
        if bce != 0.05:
            raise ValueError("The V8_1 reliability loss is fixed at 0.05")
        return original_objective(model, values, labels, weights)

    fit, selection, metadata = partition_observations(
        observations, windows, inputs, folder
    )
    with neural_runtime():
        models = fit_heads(
            fit,
            selection,
            metadata,
            make_model=make_model,
            objective=objective,
            recipe=recipe,
            output=folder,
            device=config["device"],
        )
    return evaluate_controls(observations, models["Full"], config, Path(output))
