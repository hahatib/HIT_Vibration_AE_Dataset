"""Local configuration and device selection, independent of the working directory."""

import json
from pathlib import Path

import torch


def load_config(path, output=None, device=None):
    path = Path(path).expanduser().resolve()
    config = json.loads(path.read_text(encoding="utf-8-sig"))
    base = path.parent

    def absolute(value):
        value = Path(value).expanduser()
        return str((value if value.is_absolute() else base / value).resolve())

    config["output_root"] = absolute(output or config["output_root"])
    config["data_root"] = absolute(config["data_root"])
    data_key = "cwru_root" if config.get("datasets") == ["CWRU"] else "laboratory_root"
    config[data_key] = config["data_root"]
    for key in ("reference_results_root", "de_parameters_file"):
        if config.get(key):
            config[key] = absolute(config[key])
    choice = device or config.get("device", "auto")
    if choice not in ("auto", "cpu", "cuda"):
        raise ValueError("device must be auto, cpu, or cuda")
    if choice == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was selected but is not available; use --device cpu")
    config["device"] = (
        ("cuda" if torch.cuda.is_available() else "cpu") if choice == "auto" else choice
    )
    torch.set_num_threads(int(config.get("cpu_threads", 4)))
    for key in ("rpms", "snrs", "seeds"):
        values = config[key]
        if not isinstance(values, list) or not values:
            raise ValueError(f"{key} must be a nonempty list")
        if key == "rpms":
            if any(
                type(v) not in (int, str) or not str(v).isdigit() or int(v) <= 0
                for v in values
            ):
                raise ValueError(
                    "rpms must contain positive integers or integer strings"
                )
            identifiers = [int(v) for v in values]
        else:
            if any(
                type(v) is not int and not (key == "snrs" and v is None) for v in values
            ):
                raise ValueError(
                    f"{key} must contain integers"
                    + (" or null for Clean" if key == "snrs" else "")
                )
            identifiers = values
        if len(set(identifiers)) != len(identifiers):
            raise ValueError(f"{key} must be a nonempty list without duplicates")
    return config
