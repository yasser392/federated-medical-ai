"""Classification metrics for binary diagnosis (positive class = malignant)."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

METRIC_KEYS = ("accuracy", "precision", "recall", "f1", "roc_auc")


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict[str, Any]:
    """Compute accuracy, precision, recall, F1, ROC-AUC, log-loss, confusion matrix and ROC curve.

    Args:
        y_true: Ground-truth labels in {0, 1}.
        y_prob: Predicted probabilities of class 1.
        threshold: Decision threshold.

    Returns:
        JSON-serializable dictionary of metrics.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.clip(np.asarray(y_prob, dtype=np.float64), 1e-7, 1 - 1e-7)
    if y_true.shape != y_prob.shape:
        raise ValueError("y_true and y_prob must have the same shape")
    y_pred = (y_prob >= threshold).astype(int)
    both_classes = len(np.unique(y_true)) == 2
    fpr, tpr, _ = roc_curve(y_true, y_prob) if both_classes else (np.array([]), np.array([]), None)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)) if both_classes else float("nan"),
        "test_loss": float(log_loss(y_true, y_prob, labels=[0, 1])),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
        "roc_curve": {"fpr": fpr.tolist(), "tpr": tpr.tolist()},
        "n_samples": int(y_true.size),
    }


def summarize_runs(runs: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """Mean and standard deviation of scalar metrics across several runs (e.g. seeds)."""
    out: dict[str, dict[str, float]] = {}
    for k in METRIC_KEYS:
        vals = np.array([r[k] for r in runs], dtype=float)
        out[k] = {"mean": float(np.nanmean(vals)), "std": float(np.nanstd(vals)), "n": int(vals.size)}
    return out
