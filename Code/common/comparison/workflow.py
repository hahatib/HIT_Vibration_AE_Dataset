from pathlib import Path
import argparse
import importlib.util
import itertools
import json
import sys

from common.runtime import load_config

METHOD_KEYS = {
    "DE-1D-CNN": "DE_1D_CNN",
    "Attention-1DCNN": "ATTENTION_1DCNN",
    "ECA-MSCNN": "ECA_MSCNN",
    "ML baselines": "ML",
}


def _load_module(path):
    name = "_comparison_" + path.parent.name.replace("-", "_") + "_" + path.stem
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _write_json(path, value):
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _arguments(root, method, argv):
    parser = argparse.ArgumentParser(
        description=f"Train {method} on the selected experiment grid."
    )
    parser.add_argument("--config", type=Path, default=root / "experiment.json")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    config = load_config(args.config, output=args.output, device=args.device)
    if config.get("methods") != [METHOD_KEYS[method]] or config.get("datasets") != [
        "MY_DATA"
    ]:
        raise ValueError(
            f"This entry requires methods={[METHOD_KEYS[method]]} and datasets=['MY_DATA']"
        )
    grid = list(
        itertools.product(
            config["seeds"],
            config["rpms"],
            config["snrs"],
            config.get("channels", ["CH_06"]),
        )
    )
    if args.dry_run:
        print(
            json.dumps(
                {"method": method, "conditions": len(grid), "config": config},
                ensure_ascii=False,
                indent=2,
            )
        )
        return None, grid
    output = Path(config["output_root"])
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(
            f"Output is not empty; use --output with a new folder: {output}"
        )
    output.mkdir(parents=True, exist_ok=True)
    _write_json(
        output / "experiment.json",
        {k: v for k, v in config.items() if not k.startswith("_")},
    )
    return config, grid


def _condition_folder(config, seed, rpm, snr, channel):
    folder = (
        Path(config["output_root"])
        / f"seed_{seed}"
        / str(rpm)
        / channel
        / ("Clean" if snr is None else f"SNR_{snr}dB")
    )
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def _metrics(labels, predictions, probabilities):
    import numpy as np
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    probabilities = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-12, 1.0)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return {
        "Accuracy": 100 * accuracy_score(labels, predictions),
        "Precision": 100
        * precision_score(labels, predictions, average="weighted", zero_division=0),
        "Recall": 100
        * recall_score(labels, predictions, average="weighted", zero_division=0),
        "F1": 100 * f1_score(labels, predictions, average="weighted", zero_division=0),
        "Macro_F1": 100
        * f1_score(labels, predictions, average="macro", zero_division=0),
        "NLL": float(-np.log(probabilities[np.arange(len(labels)), labels]).mean()),
        "Sample_Count": int(len(labels)),
    }


def deep_main(root, method, argv=None):
    config, grid = _arguments(Path(root), method, argv)
    if config is None:
        return
    import numpy as np
    import pandas as pd
    import torch
    from .data import prepare_condition
    from . import training

    models = _load_module(Path(root) / "models.py")
    torch.set_num_threads(int(config["cpu_threads"]))
    rows = []
    for index, (seed, rpm, snr, channel) in enumerate(grid, 1):
        print(
            f"[{index}/{len(grid)}] {method}: seed={seed}, rpm={rpm}, SNR={snr}",
            flush=True,
        )
        data = prepare_condition(config, rpm, snr, seed, channel)
        folder = _condition_folder(config, seed, rpm, snr, channel)
        training.set_reproducibility(seed)
        bundle = models.build_bundle(method, data, config)
        model, metadata = training.train_model(
            bundle,
            data,
            seed,
            config["epochs"],
            config["comparison_batch_size"],
            config["device"],
        )
        state = metadata.pop("state_dict_cpu")
        torch.save(
            {
                "state_dict": state,
                "class_names": data.class_names,
                "model_config": bundle.model_config,
                "training_metadata": {
                    key: value for key, value in metadata.items() if key != "history"
                },
            },
            folder / "model.pth",
        )
        pd.DataFrame(metadata["history"]).to_csv(
            folder / "training_history.csv", index=False
        )
        test = training.infer(
            model,
            data.splits["test"],
            config["comparison_batch_size"],
            config["device"],
        )
        np.savez_compressed(folder / "test_predictions.npz", **test)
        row = {
            "Method": method,
            "Seed": seed,
            "RPM": str(rpm),
            "SNR": "Clean" if snr is None else snr,
            "Channel": channel,
            **_metrics(test["labels"], test["predictions"], test["probabilities"]),
            "Best_Epoch": metadata["best_epoch"],
            "Best_Validation_Accuracy": metadata["best_validation_accuracy"],
        }
        rows.append(row)
        pd.DataFrame(rows).to_csv(
            Path(config["output_root"]) / "metrics.csv", index=False
        )
        del model, bundle, data
    print(
        f'Completed {len(rows)} conditions. Results: {config["output_root"]}',
        flush=True,
    )


def ml_main(root, argv=None):
    config, grid = _arguments(Path(root), "ML baselines", argv)
    if config is None:
        return
    import joblib
    import numpy as np
    import pandas as pd
    import torch
    from sklearn.preprocessing import StandardScaler
    from .data import prepare_condition

    ml = _load_module(Path(root) / "baselines.py")
    ml.device = torch.device(config["device"])
    torch.set_num_threads(int(config["cpu_threads"]))
    rows = []
    for index, (seed, rpm, snr, channel) in enumerate(grid, 1):
        print(
            f"[{index}/{len(grid)}] ML: seed={seed}, rpm={rpm}, SNR={snr}", flush=True
        )
        data = prepare_condition(config, rpm, snr, seed, channel)
        folder = _condition_folder(config, seed, rpm, snr, channel)
        train, test = data.splits["train"], data.splits["test"]
        emd_train, wpd_train = ml.get_feature_matrices(train["x_a"])
        emd_test, wpd_test = ml.get_feature_matrices(test["x_a"])
        features = {"EMD": (emd_train, emd_test), "WPD": (wpd_train, wpd_test)}
        if any(not np.isfinite(x).all() for pair in features.values() for x in pair):
            raise ValueError("Non-finite EMD/WPD features")
        for feature, classifier in itertools.product(
            ("EMD", "WPD"), ("ANN", "DNN", "SVM")
        ):
            name = f"{feature}_{classifier}"
            method_folder = folder / name
            method_folder.mkdir()
            scaler = StandardScaler()
            x_train = scaler.fit_transform(features[feature][0])
            x_test = scaler.transform(features[feature][1])
            result = ml.train_evaluate_ml(
                x_train,
                train["y"],
                x_test,
                classifier,
                data.class_count,
                epochs=config["epochs"],
                seed=seed,
            )
            fitted = {
                "feature": feature,
                "classifier": classifier,
                "scaler": scaler,
                "input_size": x_train.shape[1],
                "class_count": data.class_count,
                "class_names": data.class_names,
            }
            if classifier == "SVM":
                fitted["model"] = ml.model
            else:
                fitted["state_dict"] = {
                    key: value.detach().cpu().numpy()
                    for key, value in ml.model.state_dict().items()
                }
            joblib.dump(fitted, method_folder / "model.joblib")
            np.savez_compressed(
                method_folder / "test_predictions.npz",
                labels=test["y"],
                predictions=result["preds"],
                probabilities=result["probabilities"],
                features=result["features"],
            )
            row = {
                "Method": name,
                "Seed": seed,
                "RPM": str(rpm),
                "SNR": "Clean" if snr is None else snr,
                "Channel": channel,
                **_metrics(test["y"], result["preds"], result["probabilities"]),
            }
            rows.append(row)
            pd.DataFrame(rows).to_csv(
                Path(config["output_root"]) / "metrics.csv", index=False
            )
    print(f'Completed {len(rows)} models. Results: {config["output_root"]}', flush=True)
