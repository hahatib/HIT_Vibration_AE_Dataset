from __future__ import annotations
import random
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


def set_reproducibility(seed):
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed(int(seed))
        torch.cuda.manual_seed_all(int(seed))
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class PairedArrayDataset(Dataset):

    def __init__(self, values):
        self.x_a = values["x_a"]
        self.x_b = values["x_b"]
        self.y = values["y"]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return (
            torch.from_numpy(self.x_a[index]).unsqueeze(0),
            torch.from_numpy(self.x_b[index]).unsqueeze(0),
            torch.tensor(int(self.y[index]), dtype=torch.long),
        )


def make_loader(values, batch_size, shuffle, seed, device):
    if int(batch_size) < 1 or len(values["y"]) < 1:
        raise ValueError("Batch size and split sample count must be positive")
    generator = torch.Generator().manual_seed(int(seed))
    return DataLoader(
        PairedArrayDataset(values),
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        generator=generator,
        num_workers=0,
        pin_memory=torch.device(device).type == "cuda",
        drop_last=False,
    )


def _validation_metrics(model, loader, device):
    model.eval()
    losses = []
    correct = count = 0
    with torch.no_grad():
        for x_a, x_b, labels in loader:
            x_a = x_a.to(device, non_blocking=True)
            x_b = x_b.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                logits = model(x_a, x_b)
                loss = F.cross_entropy(logits, labels)
            correct += int((logits.argmax(dim=1) == labels).sum().item())
            count += len(labels)
            losses.append(float(loss.detach().cpu()))
    result = float(np.mean(losses))
    if not np.isfinite(result):
        raise FloatingPointError("Non-finite validation loss")
    return (result, 100.0 * correct / count)


def train_model(bundle, data, seed, epochs, batch_size=64, device="cuda"):
    device = torch.device(device)
    if int(epochs) < 1:
        raise ValueError("Epochs must be positive")
    model = bundle.model.to(device)
    train_loader = make_loader(
        data.splits["train"], batch_size, True, int(seed) + 1, device
    )
    validation_loader = make_loader(
        data.splits["validation"], batch_size, False, int(seed) + 2, device
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=float(bundle.learning_rate))
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=max(1, int(bundle.lr_period)), gamma=float(bundle.lr_gamma)
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best_loss = float("inf")
    best_accuracy = -1.0
    best_epoch = -1
    best_state = None
    history = []
    start_time = time.perf_counter()
    for epoch in range(int(epochs)):
        print(f"[epoch] epoch={epoch + 1}/{int(epochs)}", flush=True)
        model.train()
        running = []
        epoch_lr = float(optimizer.param_groups[0]["lr"])
        for x_a, x_b, labels in train_loader:
            x_a = x_a.to(device, non_blocking=True)
            x_b = x_b.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                logits = model(x_a, x_b)
                loss = F.cross_entropy(logits, labels)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite training loss at epoch {epoch}")
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running.append(float(loss.detach().cpu()))
        validation_loss, validation_accuracy = _validation_metrics(
            model, validation_loader, device
        )
        scheduler.step()
        if validation_accuracy > best_accuracy:
            best_accuracy = validation_accuracy
            best_loss = validation_loss
            best_epoch = epoch + 1
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            }
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": float(np.mean(running)),
                "validation_loss": validation_loss,
                "validation_accuracy": validation_accuracy,
                "best_validation_accuracy": best_accuracy,
                "learning_rate": epoch_lr,
                "best_validation_loss": best_loss,
            }
        )
        if epoch == 0 or (epoch + 1) % 10 == 0:
            print(
                f"  epoch={epoch + 1:03d} train={np.mean(running):.5f} val={validation_loss:.5f} val_acc={validation_accuracy:.5f} best_acc={best_accuracy:.5f}",
                flush=True,
            )
    if best_state is None:
        raise RuntimeError("Training did not produce a finite validation checkpoint")
    model.load_state_dict(best_state)
    model.to(device)
    return (
        model,
        {
            "best_epoch": best_epoch,
            "best_validation_loss": best_loss,
            "best_validation_accuracy": best_accuracy,
            "selection_metric": "validation_accuracy",
            "selection_tie_break": "earliest",
            "early_stopping": False,
            "training_seconds": time.perf_counter() - start_time,
            "epochs_completed": len(history),
            "history": history,
            "state_dict_cpu": best_state,
            "optimizer": "Adam",
            "learning_rate": float(bundle.learning_rate),
            "lr_gamma": float(bundle.lr_gamma),
            "lr_period": int(bundle.lr_period),
            "selection_split": "validation",
            "training_split": "train",
            "mixed_precision": device.type == "cuda",
        },
    )


def infer(model, values, batch_size=64, device="cuda"):
    device = torch.device(device)
    model = model.to(device)
    model.eval()
    logits_all, labels_all, features_all = ([], [], [])
    loader = make_loader(values, batch_size, False, 0, device)
    with torch.no_grad():
        for x_a, x_b, labels in loader:
            x_a = x_a.to(device, non_blocking=True)
            x_b = x_b.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                features = model.extract_features(x_a, x_b)
                logits = model.classifier(features)
            logits_all.append(logits.float().cpu())
            features_all.append(features.float().cpu())
            labels_all.append(labels)
    logits = torch.cat(logits_all).numpy()
    labels = torch.cat(labels_all).numpy().astype(np.int64)
    features = torch.cat(features_all).numpy()
    if not np.isfinite(logits).all() or not np.isfinite(features).all():
        raise FloatingPointError("Non-finite inference logits or features")
    shifted = logits - logits.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return {
        "labels": labels,
        "probabilities": probabilities,
        "predictions": np.argmax(probabilities, axis=1).astype(np.int64),
        "logits": logits,
        "features": features,
    }
