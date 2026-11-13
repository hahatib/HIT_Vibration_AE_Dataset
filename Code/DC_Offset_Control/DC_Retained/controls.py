"""Matched DC controls on the same physical windows and original noise draws."""

from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from common.primary import input_standardization as normalization
from common.primary.fitting import predict
from DC_Offset_Control.DC_Removed.neural_fusion import SerialFusion as ReferenceFusion
from . import adapter


def require(value, message):
    if not value:
        raise ValueError(message)


def relative(obs):
    return Path(str(obs["RPM"])) / str(obs["SNR"]) / f"seed_{obs['Seed']}"


def load_arrays(path):
    with np.load(path, allow_pickle=False) as values:
        return {key: values[key].copy() for key in values.files}


def test_geometry(prepared):
    """Original source-order subset indices, including class offsets."""
    rows, offset_a, offset_b = [], 0, 0
    for label in range(4):
        count_a = int(prepared.source_a_audit.loc[label, "Test_Samples"])
        count_b = int(prepared.source_b_audit.loc[label, "Test_Samples"])
        count = min(count_a, count_b)
        a = adapter.core.evenly_spaced_indices(count_a, count)
        b = adapter.core.evenly_spaced_indices(count_b, count)
        for ordinal, (ia, ib) in enumerate(zip(a, b)):
            rows.append(
                dict(
                    Class_Index=label,
                    Ordinal=ordinal,
                    Native_Index_A=offset_a + int(ia),
                    Native_Index_B=offset_b + int(ib),
                )
            )
        offset_a += count_a
        offset_b += count_b
    return pd.DataFrame(rows)


def metric(obs, name, labels, probability):
    p = np.asarray(probability, dtype=np.float64)
    p = p / p.sum(1, keepdims=True)
    predicted = p.argmax(1)
    return dict(
        RPM=obs["RPM"],
        SNR=obs["SNR"],
        Seed=obs["Seed"],
        Method=name,
        Sample_Count=len(labels),
        Accuracy=100 * accuracy_score(labels, predicted),
        Precision=100
        * precision_score(labels, predicted, average="weighted", zero_division=0),
        Recall=100
        * recall_score(labels, predicted, average="weighted", zero_division=0),
        F1=100 * f1_score(labels, predicted, average="weighted", zero_division=0),
        NLL=float(-np.log(np.clip(p[np.arange(len(labels)), labels], 1e-12, 1)).mean()),
    )


def window_mean(task, obs, config, folder):
    means = {split: [] for split in normalization.SPLIT_NAMES}

    def observer(modality, column, split, raw, normalized):
        if modality == "AE":
            means[split].append(raw.mean(axis=1))

    prepared = adapter.prepare(task, obs["Seed"], config, observer=observer)
    means = {split: np.concatenate(values) for split, values in means.items()}
    labels = prepared.source_b["train"]["y"].argmax(1).astype(np.int64)
    model = GaussianNB(
        var_smoothing=config["window_mean"]["var_smoothing"], priors=None
    )
    model.fit(means["train"][:, None], labels)
    require(
        np.array_equal(model.classes_, np.arange(4)),
        "Window-mean class ordering changed",
    )
    geometry = test_geometry(prepared)
    x = means["test"][geometry.Native_Index_B.to_numpy()][:, None]
    p = model.predict_proba(x).astype(np.float64)
    require(
        np.array_equal(geometry.Class_Index.to_numpy(), obs["test"]["Labels"]),
        "Window-mean test ordering changed",
    )
    folder.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        folder / "Window_Mean_Model.npz",
        classes=model.classes_,
        class_count=model.class_count_,
        class_prior=model.class_prior_,
        theta=model.theta_,
        var=model.var_,
        epsilon=np.array(model.epsilon_),
    )
    geometry.assign(Window_Mean=x[:, 0], Prediction=p.argmax(1)).to_csv(
        folder / "Window_Mean.csv", index=False
    )
    return p


def fixed_model_equivalence(task, obs, config, folder):
    """Keep both original CNNs and the V8_1 Full head fixed while toggling AE DC."""
    reference = Path(config["reference_results_root"])
    checkpoint = torch.load(
        reference / "Branches" / relative(obs) / "models.pt",
        map_location="cpu",
        weights_only=False,
    )
    require(
        checkpoint["epochs"] == config["epochs"],
        "Reference CNN training epochs differ",
    )
    require(
        all(checkpoint[key] == obs[key] for key in ("RPM", "SNR", "Seed")),
        "Reference CNN condition differs",
    )
    reference_test = load_arrays(reference / "Branches" / relative(obs) / "test.npz")
    reference_predictions = load_arrays(
        reference
        / "Neural_CRD_DS/Predictions"
        / str(obs["RPM"])
        / str(obs["SNR"])
        / f"seed_{obs['Seed']}.npz"
    )
    require(
        np.array_equal(reference_test["Labels"], reference_predictions["Labels"]),
        "Reference branch and Full test order differ",
    )
    require(
        np.array_equal(reference_test["B"], reference_predictions["Raw_B"]),
        "Reference branch and Full AE probabilities differ",
    )
    reference_head = torch.load(
        reference / "Neural_CRD_DS/Models/Full.pt",
        map_location="cpu",
        weights_only=False,
    )
    require(
        not reference_head["recipe"].get("refinement", {}).get("enabled"),
        "Reference must be V8_1, without refinement",
    )
    require(
        reference_head["recipe"]["heads"]["Full"]["hidden"] == 8,
        "Reference Full width must be 8",
    )
    original = dict(
        ae_train_dc_removal=True,
        noise_power_centered=True,
        pooled_train_zscore=False,
        window_normalization="zscore",
    )
    require(
        checkpoint["preprocessing"] == original,
        "Reference must use original DC removal and window zscore",
    )
    captured = {}

    def capture(modality, column, split, raw, normalized):
        if modality == "AE":
            captured[column, split] = (raw.copy(), normalized.copy())

    first = adapter.prepare(task, obs["Seed"], config, original, capture)
    expected_norm = normalization.snapshot_from_data(first)
    require(
        checkpoint["normalization"] == expected_norm,
        "Reference raw splits or TRAIN statistics differ",
    )
    require(
        checkpoint["class_names"] == first.detected_class_names,
        "Reference class ordering differs",
    )
    models = adapter.build_models(4, config["device"])
    for model, key in zip(models, ("Source_A_Branch", "Source_B_Branch")):
        model.load_state_dict(checkpoint[key], strict=True)
        model.eval()
    full = ReferenceFusion("Full", 8).to(config["device"])
    full.load_state_dict(reference_head["model"], strict=True)
    full.eval()
    differences = []
    offsets = expected_norm["Source_B"]["dc_offsets"]

    def compare(modality, column, split, raw, normalized):
        if modality != "AE":
            return
        raw0, norm0 = captured.pop((column, split))
        x0, x1 = norm0.astype(np.float32), normalized.astype(np.float32)
        differences.append(
            dict(
                Class_Column=column,
                Split=split,
                Window_Count=len(x0),
                Max_Input_Difference_Float64=float(np.max(np.abs(norm0 - normalized))),
                Max_Input_Difference_Float32=float(
                    np.max(np.abs(x0.astype(np.float64) - x1))
                ),
                Max_Noise_And_Shift_Residual=float(
                    np.max(np.abs(raw - raw0 - offsets[column]))
                ),
                Input_Allclose=bool(
                    np.allclose(
                        x0,
                        x1,
                        atol=config["equivalence"]["input_atol"],
                        rtol=config["equivalence"]["input_rtol"],
                    )
                ),
            )
        )

    second = adapter.prepare(
        task, obs["Seed"], config, dict(original, ae_train_dc_removal=False), compare
    )
    require(not captured, "Missing equivalence windows")
    for split in normalization.SPLIT_NAMES:
        require(
            np.array_equal(first.source_a[split]["x"], second.source_a[split]["x"]),
            "DC toggle changed vibration",
        )
        require(
            np.array_equal(first.paired[split]["y_a"], second.paired[split]["y_a"]),
            "DC toggle changed paired labels",
        )
    pd.testing.assert_frame_equal(
        test_geometry(first), test_geometry(second), check_exact=True
    )
    current = adapter.prepare(task, obs["Seed"], config)
    pd.testing.assert_frame_equal(
        test_geometry(first), test_geometry(current), check_exact=True
    )
    require(
        np.array_equal(current.paired["test"]["y_a"].argmax(1), obs["test"]["Labels"]),
        "Retained and removed groups do not share the same paired test observations",
    )
    values = []
    for prepared in (first, second):
        prediction = adapter.core.evaluate_paired_branches(
            *models, adapter.core.paired_loader(prepared, "test", obs["Seed"])
        )
        require(
            np.array_equal(prediction["Labels"], reference_test["Labels"]),
            "Reference test ordering differs",
        )
        with adapter.neural_runtime():
            fused = predict(
                full,
                adapter.inputs(
                    prediction["Probabilities_A"],
                    prediction["Probabilities_B"],
                    obs["RPM"],
                    obs["SNR"],
                ),
                config["device"],
            )
        values.append(dict(AE=prediction["Probabilities_B"], Full=fused["logp"]))
    rows = []
    for method in ("AE", "Full"):
        a, b = values[0][method], values[1][method]
        rows.append(
            dict(
                Method=method,
                Input_Allclose=all(r["Input_Allclose"] for r in differences),
                Max_Probability_Difference=float(np.max(np.abs(a - b))),
                Probability_Allclose=bool(
                    np.allclose(
                        a,
                        b,
                        atol=config["equivalence"]["probability_atol"],
                        rtol=config["equivalence"]["probability_rtol"],
                    )
                ),
                Prediction_Changed_Count=int(
                    np.count_nonzero(a.argmax(1) != b.argmax(1))
                ),
                Accuracy_Difference_pp=float(
                    100
                    * (
                        (b.argmax(1) == reference_test["Labels"]).mean()
                        - (a.argmax(1) == reference_test["Labels"]).mean()
                    )
                ),
            )
        )
    folder.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(differences).to_csv(folder / "Input_Equivalence.csv", index=False)
    pd.DataFrame(rows).to_csv(folder / "Fixed_Model_Equivalence.csv", index=False)
    historical = {
        "AE": reference_predictions["Raw_B"],
        "Full": reference_predictions["Full"],
    }
    history = [
        dict(
            Method=name,
            Max_Probability_Difference=float(
                np.max(np.abs(values[0][name] - historical[name]))
            ),
            Prediction_Changed_Count=int(
                np.count_nonzero(
                    values[0][name].argmax(1) != historical[name].argmax(1)
                )
            ),
        )
        for name in historical
    ]
    pd.DataFrame(history).to_csv(folder / "Reference_Replay.csv", index=False)
    np.savez_compressed(
        folder / "Fixed_Model_Equivalence.npz",
        Labels=reference_test["Labels"],
        Original_AE=values[0]["AE"],
        Original_Full=values[0]["Full"],
        DC_Off_ZScore_AE=values[1]["AE"],
        DC_Off_ZScore_Full=values[1]["Full"],
    )
    return historical, rows


def evaluate_controls(observations, full, config, output):
    tasks = {
        (str(t["RPM"]), "Clean" if t["SNR"] is None else str(t["SNR"])): t
        for t in adapter.conditions(config)
    }
    rows, equivalence = [], []
    for obs in observations:
        task = tasks[str(obs["RPM"]), str(obs["SNR"])]
        folder = output / "Controls" / relative(obs)
        original, checks = fixed_model_equivalence(task, obs, config, folder)
        for check in checks:
            equivalence.append(
                dict(RPM=obs["RPM"], SNR=obs["SNR"], Seed=obs["Seed"], **check)
            )
        mean = window_mean(task, obs, config, folder)
        with adapter.neural_runtime():
            predicted = predict(
                full,
                adapter.inputs(
                    obs["test"]["A"], obs["test"]["B"], obs["RPM"], obs["SNR"]
                ),
                config["device"],
            )
        probabilities = {
            "AE bias retained": obs["test"]["B"],
            "Full CRD-DS bias retained": predicted["logp"],
            "Window-mean": mean,
            "V8_1 AE reference": original["AE"],
            "V8_1 Full CRD-DS reference": original["Full"],
        }
        labels = obs["test"]["Labels"]
        arrays = {"Labels": labels}
        for name, p in probabilities.items():
            rows.append(metric(obs, name, labels, p))
            arrays[name + "_Probabilities"] = p
            arrays[name + "_Prediction"] = p.argmax(1).astype(np.int64)
        np.savez_compressed(folder / "Predictions.npz", **arrays)
    frame = pd.DataFrame(rows)
    reference_names = {
        "Full CRD-DS bias retained": "V8_1 Full CRD-DS reference",
        "V8_1 Full CRD-DS reference": "V8_1 Full CRD-DS reference",
    }
    frame["Reference_Method"] = frame.Method.map(
        lambda name: reference_names.get(name, "V8_1 AE reference")
    )
    lookup = frame.set_index(["RPM", "SNR", "Seed", "Method"]).Accuracy.to_dict()
    frame["Accuracy_Difference_pp"] = [
        row.Accuracy - lookup[row.RPM, row.SNR, row.Seed, row.Reference_Method]
        for row in frame.itertuples()
    ]
    frame.to_csv(output / "Comparison_Metrics.csv", index=False)
    by_seed = frame.groupby(["Method", "SNR", "Seed"], sort=False)[
        ["Accuracy", "Accuracy_Difference_pp"]
    ].mean()
    by_seed.groupby(["Method", "SNR"], sort=False).agg(["mean", "std"]).to_csv(
        output / "Comparison_Summary.csv"
    )
    pd.DataFrame(equivalence).to_csv(
        output / "Fixed_Model_Equivalence.csv", index=False
    )
    return frame
