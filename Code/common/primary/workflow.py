"""Serial CNN -> saved calibration -> frozen fusion -> evaluation workflow."""

import gc
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch


def _json(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


def _snr(value):
    return (
        "Clean" if value is None or str(value) in ("Clean", "raw") else str(int(value))
    )


def _calibration(outputs, data, windows):
    starts_a, starts_b, ordinal, labels = [], [], [], []
    for row in data.pairing_audit.loc[
        data.pairing_audit.Split == "calibration"
    ].itertuples():
        n = int(row.Paired_Count)
        starts_a.extend(
            np.round(np.linspace(0, row.Source_A_Original - 1, n)).astype(int)
            * windows["A"][1]
        )
        starts_b.extend(
            np.round(np.linspace(0, row.Source_B_Original - 1, n)).astype(int)
            * windows["B"][1]
        )
        ordinal.extend(range(n))
        labels.extend([int(row.Class_Index)] * n)
    if not np.array_equal(outputs["Labels"], labels):
        raise ValueError("Calibration pairing and labels differ")
    return dict(
        A=outputs["Probabilities_A"],
        B=outputs["Probabilities_B"],
        Labels=np.asarray(labels, dtype=np.int64),
        Start_A=np.asarray(starts_a, dtype=np.int64),
        Start_B=np.asarray(starts_b, dtype=np.int64),
        Ordinal=np.asarray(ordinal, dtype=np.int64),
    )


def dry_run(config, adapter):
    tasks = list(adapter.conditions(config))
    return dict(
        conditions_per_seed=len(tasks),
        seeds=config["seeds"],
        total_conditions=len(tasks) * len(config["seeds"]),
        epochs=config["epochs"],
        snrs=config["snrs"],
        device=config["device"],
        output_root=config["output_root"],
    )


def run_experiment(config, adapter):
    """Adapter supplies dataset preparation, branch models and fusion fitting.

    Required methods: configure(config), conditions(config), prepare(task,seed,config),
    build_models(class_count,device), fit_and_evaluate(observations,config,output).
    Required attributes: core (CNN functions), windows {'A':(size,step),'B':...}.
    Tasks contain RPM, SNR and the adapter's input fields. Each observation retains
    calibration probabilities and physical window starts for the purged holdout.
    """
    output = Path(config["output_root"])
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {output}; choose another --output"
        )
    output.mkdir(parents=True, exist_ok=True)
    adapter.configure(config)
    core = adapter.core
    torch.set_num_threads(int(config.get("cpu_threads", 1)))
    tasks = list(adapter.conditions(config))
    records = []
    for seed in config["seeds"]:
        pending = []
        for task in tasks:
            rpm, snr = int(task["RPM"]), _snr(task["SNR"])
            folder = output / "Branches" / str(rpm) / snr / f"seed_{seed}"
            folder.mkdir(parents=True, exist_ok=True)
            print(f"[CNN] seed={seed}, RPM={rpm}, SNR={snr}", flush=True)
            data = adapter.prepare(task, seed, config)
            core.set_reproducibility(seed)
            a, b = adapter.build_models(
                len(data.detected_class_names), config["device"]
            )
            loaders = core.native_branch_loaders(data, seed)
            labels = getattr(adapter, "source_labels", ("Vibration", "AE"))
            best_a, history_a = core.train_branch(
                a,
                loaders["train_a"],
                loaders["validation_a"],
                config["epochs"],
                labels[0],
            )
            best_b, history_b = core.train_branch(
                b,
                loaders["train_b"],
                loaders["validation_b"],
                config["epochs"],
                labels[1],
            )
            normalization = {
                f"Source_{tag}": audit.attrs["Input_Normalization"]
                for tag, audit in [
                    ("A", data.source_a_audit),
                    ("B", data.source_b_audit),
                ]
            }
            payload = dict(
                Source_A_Branch={
                    k: v.detach().cpu() for k, v in a.state_dict().items()
                },
                Source_B_Branch={
                    k: v.detach().cpu() for k, v in b.state_dict().items()
                },
                preprocessing=config["preprocessing"],
                normalization=normalization,
                class_names=data.detected_class_names,
                source_labels=labels,
                RPM=rpm,
                SNR=snr,
                Seed=seed,
                epochs=config["epochs"],
                selected_epochs=[best_a["Epoch"], best_b["Epoch"]],
            )
            torch.save(payload, folder / "models.pt")
            pd.DataFrame(history_a).to_csv(folder / "history_a.csv", index=False)
            pd.DataFrame(history_b).to_csv(folder / "history_b.csv", index=False)
            calibration = _calibration(
                core.evaluate_paired_branches(
                    a, b, core.paired_loader(data, "calibration", seed)
                ),
                data,
                adapter.windows,
            )
            np.savez_compressed(folder / "calibration.npz", **calibration)
            pending.append((task, folder, normalization, data.detected_class_names))
            del data, a, b, loaders, calibration, payload
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        # Match the original second pass: reconstruct identical noisy splits,
        # reload frozen CNN weights and evaluate their test probabilities.
        for task, folder, normalization, class_names in pending:
            data = adapter.prepare(task, seed, config)
            saved = torch.load(
                folder / "models.pt", map_location=config["device"], weights_only=False
            )
            a, b = adapter.build_models(len(class_names), config["device"])
            a.load_state_dict(saved["Source_A_Branch"])
            b.load_state_dict(saved["Source_B_Branch"])
            test = core.evaluate_paired_branches(
                a, b, core.paired_loader(data, "test", seed)
            )
            test = dict(
                A=test["Probabilities_A"],
                B=test["Probabilities_B"],
                Labels=test["Labels"],
            )
            np.savez_compressed(folder / "test.npz", **test)
            with np.load(folder / "calibration.npz", allow_pickle=False) as z:
                calibration = {k: z[k] for k in z.files}
            records.append(
                dict(
                    RPM=int(task["RPM"]),
                    SNR=_snr(task["SNR"]),
                    Seed=int(seed),
                    calibration=calibration,
                    test=test,
                    normalization=normalization,
                )
            )
            del data, a, b, saved
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    # Preserve the original pooled MLP ordering: RPM, SNR, seed.
    rpms = list(dict.fromkeys(int(t["RPM"]) for t in tasks))
    snrs = list(dict.fromkeys(_snr(t["SNR"]) for t in tasks))
    records.sort(
        key=lambda x: (
            rpms.index(x["RPM"]),
            snrs.index(x["SNR"]),
            config["seeds"].index(x["Seed"]),
        )
    )
    snapshot = {k: v for k, v in config.items() if not k.startswith("_")}
    (output / "experiment.json").write_text(
        json.dumps(snapshot, indent=2, default=_json), encoding="utf-8"
    )
    result = adapter.fit_and_evaluate(records, config, output)
    print(f"[complete] {output}", flush=True)
    return result
