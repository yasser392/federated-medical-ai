"""Experiment A (centralized learning) and Experiment B (local-only hospital learning)."""

from __future__ import annotations

import logging
from typing import Any

import torch

from src.data.dataset import train_val_split
from src.data.partition import PartitionMode
from src.data.pipeline import ExperimentData, prepare_data
from src.data.preprocessing import Standardizer, to_tensor_dataset
from src.evaluation.metrics import compute_metrics
from src.models.neural_network import build_model
from src.models.training import fit, predict_proba
from src.utils.reproducibility import get_device, set_seed

logger = logging.getLogger(__name__)


def _init_model(cfg: dict, seed: int) -> torch.nn.Module:
    """Build a model with seed-determined initial weights (identical across experiments)."""
    torch.manual_seed(seed)
    return build_model(cfg)


def run_centralized(cfg: dict, seed: int | None = None) -> dict[str, Any]:
    """Experiment A: pool all training data in one place and train one model.

    The scaler is fitted on the centralized training split only; the held-out test set is used once,
    for final evaluation.
    """
    seed = cfg["seed"] if seed is None else seed
    set_seed(seed)
    get_device(cfg["runtime"].get("num_threads"))
    data = prepare_data(cfg, "iid", seed)  # partition mode is irrelevant here: only pool/test are used
    train, val = train_val_split(data.pool, cfg["data"]["val_fraction"], seed)
    scaler = Standardizer.fit(train.X)
    train_ds, val_ds, test_ds = (to_tensor_dataset(d, scaler) for d in (train, val, data.test))

    model = _init_model(cfg, seed)
    history = fit(model, train_ds, val_ds, cfg["centralized"]["epochs"], cfg, seed)
    metrics = compute_metrics(*predict_proba(model, test_ds), cfg["training"]["threshold"])
    logger.info("Centralized | acc=%.4f f1=%.4f auc=%.4f", metrics["accuracy"], metrics["f1"], metrics["roc_auc"])
    return {
        "name": "Centralized",
        "seed": seed,
        "n_train": len(train),
        "n_val": len(val),
        "history": history,
        "metrics": metrics,
    }


def run_local_only(cfg: dict, mode: PartitionMode = "non_iid", seed: int | None = None,
                   data: ExperimentData | None = None) -> list[dict[str, Any]]:
    """Experiment B: each hospital trains alone on its private data, with no communication.

    Each hospital fits its own scaler on its own training data (it has no access to anyone else's
    statistics) and is evaluated on the common held-out test set.
    """
    seed = cfg["seed"] if seed is None else seed
    set_seed(seed)
    get_device(cfg["runtime"].get("num_threads"))
    data = data or prepare_data(cfg, mode, seed)
    results = []
    for i, h in enumerate(data.hospitals):
        scaler = Standardizer.fit(h.train.X)
        train_ds, val_ds, test_ds = (to_tensor_dataset(d, scaler) for d in (h.train, h.val, data.test))
        model = _init_model(cfg, seed)
        history = fit(model, train_ds, val_ds, cfg["local_only"]["epochs"], cfg, seed + 100 * (i + 1))
        metrics = compute_metrics(*predict_proba(model, test_ds), cfg["training"]["threshold"])
        logger.info("Local-only %s (%s) | n=%d acc=%.4f f1=%.4f auc=%.4f", h.name, mode, len(h.train),
                    metrics["accuracy"], metrics["f1"], metrics["roc_auc"])
        results.append({
            "name": f"{h.name} (local)",
            "hospital": h.name,
            "mode": mode,
            "seed": seed,
            "n_train": len(h.train),
            "n_val": len(h.val),
            "history": history,
            "metrics": metrics,
        })
    return results
