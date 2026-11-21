"""Shared CPU training/evaluation engine used by centralized, local-only and federated training."""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.data.preprocessing import make_loader


def make_optimizer(model: nn.Module, cfg: dict) -> torch.optim.Optimizer:
    """Adam optimizer configured from ``cfg['training']``."""
    t = cfg["training"]
    return torch.optim.Adam(model.parameters(), lr=t["learning_rate"], weight_decay=t.get("weight_decay", 0.0))


def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer) -> float:
    """Run one epoch of mini-batch training and return the sample-weighted mean BCE loss.

    Raises:
        FloatingPointError: If the loss becomes NaN or infinite.
    """
    criterion = nn.BCEWithLogitsLoss()
    model.train()
    total, n = 0.0, 0
    for xb, yb in loader:
        optimizer.zero_grad()
        loss = criterion(model(xb).squeeze(-1), yb)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite training loss: {loss.item()}")
        loss.backward()
        optimizer.step()
        total += loss.item() * xb.shape[0]
        n += xb.shape[0]
    return total / max(n, 1)


@torch.no_grad()
def evaluate_loss(model: nn.Module, ds: TensorDataset, batch_size: int = 256) -> float:
    """Mean BCE loss of ``model`` on a dataset (eval mode, no dropout)."""
    if len(ds) == 0:
        return math.nan
    criterion = nn.BCEWithLogitsLoss(reduction="sum")
    model.eval()
    total = 0.0
    for xb, yb in make_loader(ds, batch_size, shuffle=False):
        total += criterion(model(xb).squeeze(-1), yb).item()
    return total / len(ds)


@torch.no_grad()
def predict_proba(model: nn.Module, ds: TensorDataset) -> tuple[np.ndarray, np.ndarray]:
    """Return (y_true, y_prob) for a dataset."""
    X, y = ds.tensors
    return y.numpy().astype(np.int64), model.predict_proba(X).numpy()


def fit(
    model: nn.Module,
    train_ds: TensorDataset,
    val_ds: TensorDataset | None,
    epochs: int,
    cfg: dict,
    seed: int,
) -> dict[str, list[float]]:
    """Train for a fixed number of epochs (no early stopping, no test-set access).

    Returns:
        History with per-epoch ``train_loss`` and ``val_loss``.
    """
    optimizer = make_optimizer(model, cfg)
    history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}
    for epoch in range(epochs):
        loader = make_loader(train_ds, cfg["training"]["batch_size"], shuffle=True, seed=seed + epoch)
        history["train_loss"].append(train_one_epoch(model, loader, optimizer))
        history["val_loss"].append(evaluate_loss(model, val_ds) if val_ds is not None else math.nan)
    return history
