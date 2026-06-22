"""10-way angle linear probe with validation, early stopping, and balanced metrics."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


class FeatureNormalizer:
    """Standardize features using train-set mean/std only."""

    def __init__(self):
        self.mean: Optional[torch.Tensor] = None
        self.std: Optional[torch.Tensor] = None

    def fit(self, X: torch.Tensor) -> "FeatureNormalizer":
        self.mean = X.mean(dim=0)
        self.std = X.std(dim=0).clamp_min(1e-6)
        return self

    def transform(self, X: torch.Tensor) -> torch.Tensor:
        if self.mean is None or self.std is None:
            raise RuntimeError("Normalizer not fitted")
        return (X - self.mean) / self.std


class AngleProbe(nn.Module):
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


@dataclass
class ClassificationMetrics:
    accuracy: float
    balanced_accuracy: float
    macro_recall: float
    n_samples: int
    per_class_accuracy: Dict[str, float]
    per_class_count: Dict[str, int]
    confusion_matrix: List[List[int]] = field(default_factory=list)


def _confusion_matrix(
    y_true: torch.Tensor, y_pred: torch.Tensor, num_classes: int
) -> List[List[int]]:
    cm = torch.zeros(num_classes, num_classes, dtype=torch.long)
    for t, p in zip(y_true.view(-1), y_pred.view(-1)):
        cm[int(t), int(p)] += 1
    return cm.tolist()


def compute_classification_metrics(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    class_to_angle: Dict[int, float],
) -> ClassificationMetrics:
    num_classes = len(class_to_angle)
    cm = _confusion_matrix(y_true, y_pred, num_classes)

    correct = (y_true == y_pred).float()
    acc = correct.mean().item()

    per_class: Dict[str, float] = {}
    per_class_n: Dict[str, int] = {}
    recalls = []

    for cls_idx, angle in class_to_angle.items():
        mask = y_true == cls_idx
        n = int(mask.sum().item())
        per_class_n[str(int(angle))] = n
        if n > 0:
            rec = (y_pred[mask] == y_true[mask]).float().mean().item()
            per_class[str(int(angle))] = rec
            recalls.append(rec)
        else:
            per_class[str(int(angle))] = 0.0

    # Balanced accuracy = mean of per-class recalls (fair when classes are balanced)
    balanced_acc = sum(recalls) / len(recalls) if recalls else 0.0
    macro_recall = balanced_acc

    return ClassificationMetrics(
        accuracy=acc,
        balanced_accuracy=balanced_acc,
        macro_recall=macro_recall,
        n_samples=len(y_true),
        per_class_accuracy=per_class,
        per_class_count=per_class_n,
        confusion_matrix=cm,
    )


@dataclass
class TrainResult:
    model: AngleProbe
    best_val_loss: float
    best_epoch: int
    history: List[dict]


def train_angle_probe(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    *,
    num_classes: int,
    epochs: int = 200,
    lr: float = 1e-2,
    weight_decay: float = 1e-2,
    batch_size: int = 128,
    patience: int = 20,
    device: str | None = None,
) -> TrainResult:
    """
    Train with AdamW, LR schedule on val loss, early stopping, best checkpoint.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = AngleProbe(X_train.shape[1], num_classes=num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-5
    )
    criterion = nn.CrossEntropyLoss()  # classes are balanced in data

    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )

    X_val_d = X_val.to(device)
    y_val_d = y_val.to(device)

    best_state = copy.deepcopy(model.state_dict())
    best_val_loss = float("inf")
    best_epoch = 0
    stale = 0
    history: List[dict] = []

    pbar = tqdm(range(1, epochs + 1), desc="Training probe", unit="epoch")
    for epoch in pbar:
        model.train()
        train_loss = 0.0
        n_batches = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
            n_batches += 1

        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_d)
            val_loss = criterion(val_logits, y_val_d).item()

        scheduler.step(val_loss)
        train_loss_avg = train_loss / max(n_batches, 1)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss_avg,
                "val_loss": val_loss,
                "lr": optimizer.param_groups[0]["lr"],
            }
        )

        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1

        pbar.set_postfix(
            train=f"{train_loss_avg:.3f}",
            val=f"{val_loss:.3f}",
            best=best_epoch,
            stale=stale,
            refresh=False,
        )
        if stale >= patience:
            pbar.set_postfix(
                train=f"{train_loss_avg:.3f}",
                val=f"{val_loss:.3f}",
                best=best_epoch,
                stopped=True,
                refresh=True,
            )
            break

    model.load_state_dict(best_state)
    return TrainResult(
        model=model,
        best_val_loss=best_val_loss,
        best_epoch=best_epoch,
        history=history,
    )


@torch.no_grad()
def evaluate_angle_probe(
    model: AngleProbe,
    X: torch.Tensor,
    y: torch.Tensor,
    class_to_angle: Dict[int, float],
    device: str | None = None,
) -> ClassificationMetrics:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    X = X.to(device)
    logits = model(X)
    preds = logits.argmax(dim=-1).cpu()
    return compute_classification_metrics(y.cpu(), preds, class_to_angle)
