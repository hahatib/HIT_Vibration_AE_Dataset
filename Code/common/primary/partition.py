"""Purged physical-window calibration holdout, independent of scores or labels beyond class balance."""

from fractions import Fraction
import numpy as np
import pandas as pd


def require(condition, message):
    if not condition:
        raise ValueError(message)


def roles_for_count(frame, count, windows):
    roles = pd.Series(index=frame.index, dtype="object")
    for _, group in frame.groupby("Class_Index", sort=True):
        group = group.sort_values("Ordinal")
        require(0 < count < len(group), "Empty calibration role")
        selected = group.tail(count)
        for i, row in group.iterrows():
            overlap = any(
                (
                    (
                        (
                            int(row["Absolute_Start_" + tag])
                            < selected["Absolute_End_" + tag]
                        )
                        & (
                            int(row["Absolute_End_" + tag])
                            > selected["Absolute_Start_" + tag]
                        )
                    ).any()
                    for tag in windows
                )
            )
            roles.loc[i] = (
                "Selection" if i in selected.index else "Purged" if overlap else "Fit"
            )
    return roles


def plan_physical(physical, windows, classes, fraction=0.2):
    require(fraction == 0.2, "The declared calibration holdout is 20%")
    rpms = sorted((int(x) for x in physical.RPM.unique()))
    require(
        bool(rpms) and physical.index.is_unique, "Empty/duplicate physical window rows"
    )
    candidates = {}
    for rpm in rpms:
        group = physical[physical.RPM.eq(rpm)]
        sizes = group.groupby("Class_Index").size()
        require(
            set(sizes.index) == set(classes) and sizes.nunique() == 1,
            "Unequal calibration classes",
        )
        options = {}
        for k in range(1, int(sizes.iloc[0])):
            roles = roles_for_count(group, k, windows)
            counts = (
                group.assign(Role=roles)
                .groupby(["Class_Index", "Role"])
                .size()
                .unstack(fill_value=0)
            )
            if (
                "Fit" not in counts
                or "Selection" not in counts
                or counts["Fit"].min() < 1
            ):
                continue
            if not counts.nunique().eq(1).all():
                continue
            fit = int(roles.eq("Fit").sum())
            selection = int(roles.eq("Selection").sum())
            purged = int(roles.eq("Purged").sum())
            options[k, fit, selection, purged] = roles
        require(bool(options), "Insufficient disjoint balanced calibration windows")
        candidates[rpm] = options
    common = set.intersection(*(set(candidates[r]) for r in rpms))
    require(
        bool(common),
        "No common per-RPM fit/selection/purge quota; inspect physical window layouts",
    )
    quota = min(
        common,
        key=lambda x: (
            abs(Fraction(x[2], x[1] + x[2]) - Fraction(str(fraction))),
            -x[1],
            x[0],
        ),
    )
    selected_per_class, fit_count, selection_count, purge_count = quota
    result = physical.copy()
    result["Role"] = pd.concat([candidates[r][quota] for r in rpms]).reindex(
        result.index
    )
    require(result.Role.notna().all(), "Unassigned calibration window")
    per_rpm = result.groupby(["RPM", "Role"]).size().unstack(fill_value=0)
    require(
        per_rpm.nunique().eq(1).all(), "Per-RPM calibration roles have unequal counts"
    )
    overlap_rows = []
    for tag in windows:
        for record, group in result.groupby("Record_" + tag):
            fit, selection = (
                group[group.Role.eq("Fit")],
                group[group.Role.eq("Selection")],
            )
            overlaps = sum(
                (
                    int(
                        (
                            (
                                int(r["Absolute_Start_" + tag])
                                < selection["Absolute_End_" + tag]
                            )
                            & (
                                int(r["Absolute_End_" + tag])
                                > selection["Absolute_Start_" + tag]
                            )
                        ).sum()
                    )
                    for _, r in fit.iterrows()
                )
            )
            require(overlaps == 0, "Physical fit/selection interval overlap")
            overlap_rows.append(
                dict(
                    Modality=tag,
                    Record=record,
                    Fit_Windows=len(fit),
                    Selection_Windows=len(selection),
                    Overlap_Pairs=overlaps,
                )
            )
    counts = result.Role.value_counts().to_dict()
    plan = dict(
        protocol="physical_calibration_holdout_uniform_rpm_v2",
        nominal_fraction=fraction,
        quota_policy="uniform_per_rpm_nearest_fraction",
        fraction_denominator="Fit + Selection after physical-overlap purge",
        actual_retained_fraction=counts["Selection"]
        / (counts["Fit"] + counts["Selection"]),
        partition_seed=None,
        rpm_order=rpms,
        physical_pairs=counts,
        uniform_counts_per_rpm=dict(
            Fit=fit_count, Selection=selection_count, Purged=purge_count
        ),
        selected_pairs_per_class_by_rpm={str(r): selected_per_class for r in rpms},
        selection_uses_probabilities_or_scores=False,
        repeated_SNR_seed_copies_bound=True,
        fit_selection_physical_overlap=0,
        refit_after_selection=False,
    )
    return (result, pd.DataFrame(overlap_rows), plan)


def partition_observations(observations, windows, make_inputs, output):
    """Use physical calibration locations, binding repeated SNR/seed copies."""
    import json
    import torch
    from pathlib import Path

    packs, labels, metadata, physical = [], [], [], {}
    offset = 0
    for obs in observations:
        data = obs["calibration"]
        count = len(data["Labels"])
        classes = range(data["A"].shape[1])
        packs.append(make_inputs(data["A"], data["B"], obs["RPM"], obs["SNR"]))
        labels.append(torch.as_tensor(data["Labels"], dtype=torch.long))
        rows = []
        for label in classes:
            indices = np.flatnonzero(data["Labels"] == label)
            require(
                np.array_equal(data["Ordinal"][indices], np.arange(len(indices))),
                "Invalid calibration ordinals",
            )
            for i in indices:
                row = dict(
                    RPM=int(obs["RPM"]),
                    Class_Index=label,
                    Ordinal=int(data["Ordinal"][i]),
                )
                for tag, (window, step) in windows.items():
                    stats = obs["normalization"]["Source_" + tag]
                    column = stats["class_columns"][label]
                    bounds = stats["record_boundaries"][column]
                    original_count = (bounds[3] - bounds[2] - window) // step + 1
                    expected = (
                        np.round(
                            np.linspace(0, original_count - 1, len(indices))
                        ).astype(np.int64)
                        * step
                    )
                    require(
                        np.array_equal(data["Start_" + tag][indices], expected),
                        "Calibration pairing differs",
                    )
                    start = bounds[2] + int(data["Start_" + tag][i])
                    require(
                        bounds[2] <= start and start + window <= bounds[3],
                        "Window crosses calibration boundary",
                    )
                    row.update(
                        {
                            f"Record_{tag}": f'{int(obs["RPM"])}:{column}',
                            f"Class_Column_{tag}": column,
                            f"Absolute_Start_{tag}": start,
                            f"Absolute_End_{tag}": start + window,
                        }
                    )
                rows.append(row)
        frame = pd.DataFrame(rows)
        rpm = int(obs["RPM"])
        if rpm in physical:
            pd.testing.assert_frame_equal(physical[rpm], frame, check_exact=True)
        else:
            physical[rpm] = frame
        metadata.append(
            pd.DataFrame(
                dict(
                    Row=np.arange(offset, offset + count),
                    RPM=rpm,
                    SNR=str(obs["SNR"]),
                    Seed=int(obs["Seed"]),
                    Labels=data["Labels"],
                    Ordinal=data["Ordinal"],
                    Start_A=data["Start_A"],
                    Start_B=data["Start_B"],
                )
            )
        )
        offset += count
    pack = tuple(torch.cat([p[i] for p in packs]) for i in range(3)) + (
        torch.cat(labels),
    )
    metadata = pd.concat(metadata, ignore_index=True)
    physical, overlaps, plan = plan_physical(
        pd.concat(list(physical.values()), ignore_index=True), windows, classes
    )
    lookup = physical.set_index(["RPM", "Class_Index", "Ordinal"]).Role.to_dict()
    metadata["Role"] = [
        lookup[int(r), int(y), int(o)]
        for r, y, o in zip(metadata.RPM, metadata.Labels, metadata.Ordinal)
    ]
    require(
        (metadata.groupby(["RPM", "Labels", "Ordinal"]).Role.nunique() == 1).all(),
        "Physical copies crossed roles",
    )
    folder = Path(output) / "Selection"
    folder.mkdir(parents=True, exist_ok=True)
    physical.to_csv(folder / "Physical_Partition.csv", index=False)
    overlaps.to_csv(folder / "Physical_Overlap_Checks.csv", index=False)
    metadata.to_csv(folder / "Partition.csv", index=False)
    (folder / "Partition_Plan.json").write_text(
        json.dumps(plan, indent=2), encoding="utf-8"
    )
    fit = tuple(x[metadata.Role.eq("Fit").to_numpy()] for x in pack)
    selection = tuple(x[metadata.Role.eq("Selection").to_numpy()] for x in pack)
    return (
        fit,
        selection,
        metadata.loc[metadata.Role.eq("Selection")].reset_index(drop=True),
    )
