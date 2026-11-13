"""Calibration-only MLP fitting and epoch selection.

The optimizer, accumulation order, initialization and strict accuracy tie rule
are retained from the original experiments. No test arrays enter this module.
"""

import copy
import math
from pathlib import Path
import numpy as np
import pandas as pd
import torch


def require(condition, message):
    if not condition:
        raise ValueError(message)


def state_copy(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def predict(model, pack, device, batch=16384):
    require(len(pack) == 3, "Inference accepts probabilities/context only")
    model.eval()
    parts = {k: [] for k in ("logp", "loga", "logb", "r", "logt", "bias")}
    start = 0
    with torch.no_grad():
        while start < len(pack[0]):
            try:
                out = model(*(x[start : start + batch].to(device) for x in pack))
                current = {}
                for key in parts:
                    value = (
                        out[key].exp()
                        if key in ("logp", "loga", "logb", "logt")
                        else out[key]
                    )
                    current[key] = value.cpu().numpy().astype(np.float64)
                for key in parts:
                    parts[key].append(current[key])
                start += batch
            except torch.cuda.OutOfMemoryError:
                if batch == 1:
                    raise
                batch = max(1, batch // 2)
                torch.cuda.empty_cache()
    result = {k: np.concatenate(v) for k, v in parts.items()}
    for key in ("logp", "loga", "logb"):
        result[key] /= result[key].sum(1, keepdims=True)
    return result


class SelectionScore:
    """Equal RPM/SNR/seed accuracy; CWRU permits a Clean-only fallback."""

    def __init__(self, pack, metadata, device, batch=16384, *, allow_clean_only=False):
        self.pack, self.device, self.batch = pack, device, batch
        sizes = (
            metadata.groupby(["RPM", "SNR", "Seed"]).RPM.transform("size").to_numpy()
        )
        lcm = math.lcm(*(int(x) for x in np.unique(sizes)))
        included = metadata.SNR.ne("Clean").to_numpy()
        if allow_clean_only and not included.any():
            included = np.ones(len(metadata), dtype=bool)
        self.weights = torch.as_tensor(
            np.where(included, lcm // sizes, 0), dtype=torch.int64
        )
        self.total = int(self.weights.sum())
        self.raw_n = int(included.sum())
        require(
            self.total > 0, "At least one noisy SNR is required for epoch selection"
        )

    def __call__(self, model):
        model.eval()
        weighted, raw, start = 0, 0, 0
        with torch.no_grad():
            while start < len(self.pack[0]):
                sl = slice(start, start + self.batch)
                try:
                    output = model(*(x[sl].to(self.device) for x in self.pack[:3]))
                    ok = (
                        output["logp"].exp().argmax(1)
                        == self.pack[3][sl].to(self.device)
                    ).cpu()
                    weights = self.weights[sl]
                    weighted += int((ok * weights).sum())
                    raw += int((ok & (weights > 0)).sum())
                    start += self.batch
                except torch.cuda.OutOfMemoryError:
                    if self.batch == 1:
                        raise
                    self.batch = max(1, self.batch // 2)
                    torch.cuda.empty_cache()
        return dict(
            Selection_Score_Integer=weighted,
            Selection_Weight_Total=self.total,
            Selection_Accuracy=100 * weighted / self.total,
            Selection_Raw_Correct=raw,
            Selection_Raw_Count=self.raw_n,
        )


def fit_head(
    method,
    pack,
    scorer,
    *,
    make_model,
    objective,
    recipe,
    device,
    phase="base",
    parent=None,
):
    model = make_model(method, device)
    if parent is not None:
        model.load_state_dict(parent.state_dict())
    refine = recipe.get("refinement", {})
    if phase == "refinement" and method == "Full":
        for module in (model.cal_a, model.cal_b):
            for parameter in module.parameters():
                parameter.requires_grad_(False)
    lr = refine["learning_rate"] if phase == "refinement" else recipe["learning_rate"]
    bce = refine["reliability_bce_weight"] if phase == "refinement" else 0.05
    epochs = (
        refine["epochs"] if phase == "refinement" else recipe["heads"][method]["epochs"]
    )
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=recipe["weight_decay"],
    )
    batch, history, best = len(pack[0]), [], None
    if phase == "refinement":
        require(
            parent is not None and method != "Calibration only",
            "Invalid refinement parent",
        )
        best = dict(epoch=0, score=scorer(model), model=state_copy(model))
        history = [
            dict(Method=method, Phase=phase, Epoch=0, Total_Loss=None, **best["score"])
        ]
    for epoch in range(1, epochs + 1):
        before_model = state_copy(model)
        before_optimizer = copy.deepcopy(optimizer.state_dict())
        while True:
            model.train()
            optimizer.zero_grad(set_to_none=True)
            total = 0.0
            try:
                for start in range(0, len(pack[0]), batch):
                    a, b, context, y = (
                        x[start : start + batch].to(device) for x in pack
                    )
                    out = model(a, b, context)
                    scale = len(y) / len(pack[0])
                    loss = (
                        objective(model, out, y, torch.ones(len(y), device=device), bce)
                        * scale
                    )
                    require(bool(torch.isfinite(loss)), "Nonfinite objective")
                    loss.backward()
                    total += float(loss.detach())
                norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), recipe["gradient_clip"]
                )
                require(bool(torch.isfinite(norm)), "Nonfinite gradient")
                optimizer.step()
                break
            except torch.cuda.OutOfMemoryError:
                model.load_state_dict(before_model)
                optimizer.load_state_dict(before_optimizer)
                optimizer.zero_grad(set_to_none=True)
                if batch == 1:
                    raise
                batch = max(1, batch // 2)
                torch.cuda.empty_cache()
        score = scorer(model)
        # A strict improvement retains the earliest epoch when accuracy ties.
        if (
            best is None
            or score["Selection_Score_Integer"]
            > best["score"]["Selection_Score_Integer"]
        ):
            best = dict(epoch=epoch, score=score, model=state_copy(model))
        history.append(
            dict(
                Method=method,
                Phase=phase,
                Epoch=epoch,
                Total_Loss=total,
                Accumulation_Chunk=batch,
                **score,
            )
        )
        if epoch % 120 == 0 or epoch == epochs:
            print(
                f'[MLP] {phase} {method} {epoch}/{epochs}; best={best["epoch"]}; '
                f'accuracy={best["score"]["Selection_Accuracy"]:.6f}',
                flush=True,
            )
    model.load_state_dict(best["model"])
    require(scorer(model) == best["score"], "Selected calibration score changed")
    if phase == "refinement" and method == "Full":
        require(
            all(
                torch.equal(v, parent.state_dict()[k])
                for k, v in model.state_dict().items()
                if k.startswith(("cal_a.", "cal_b."))
            ),
            "Refinement changed calibrators",
        )
    return model, best, history


def fit_heads(
    fit,
    selection,
    metadata,
    *,
    make_model,
    objective,
    recipe,
    output,
    device,
    allow_clean_only=False,
):
    """Fit independent component heads and freeze them before test evaluation."""
    output = Path(output)
    (output / "Models").mkdir(parents=True, exist_ok=True)
    (output / "Selection").mkdir(exist_ok=True)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    scorer = SelectionScore(
        selection,
        metadata,
        device,
        recipe.get("inference_batch", 16384),
        allow_clean_only=allow_clean_only,
    )
    base, final, chosen, rows, curves = {}, {}, {}, [], []
    for method in recipe["heads"]:
        base[method], chosen[method], history = fit_head(
            method,
            fit,
            scorer,
            make_model=make_model,
            objective=objective,
            recipe=recipe,
            device=device,
        )
        curves.extend(history)
    for method in recipe["heads"]:
        phase = "base"
        if recipe.get("refinement", {}).get("enabled") and method != "Calibration only":
            phase = "refinement"
            final[method], chosen[method], history = fit_head(
                method,
                fit,
                scorer,
                make_model=make_model,
                objective=objective,
                recipe=recipe,
                device=device,
                phase=phase,
                parent=base[method],
            )
            curves.extend(history)
        else:
            final[method] = base[method]
        best = chosen[method]
        stem = method.replace(" ", "_")
        torch.save(
            dict(
                model=state_copy(final[method]),
                epoch=best["epoch"],
                phase=phase,
                selection_score=best["score"],
                recipe=recipe,
            ),
            output / "Models" / f"{stem}.pt",
        )
        rows.append(
            dict(
                Method=method,
                Phase=phase,
                Selected_Epoch=best["epoch"],
                **best["score"],
            )
        )
    pd.DataFrame(curves).to_csv(output / "Selection/Training_History.csv", index=False)
    pd.DataFrame(rows).to_csv(output / "Selection/Epoch_Selection.csv", index=False)
    return final
