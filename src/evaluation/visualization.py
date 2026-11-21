"""Matplotlib figures for the experiment results (saved as PNG under ``results/figures``)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless backend
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# Validated categorical palette (fixed order; color follows the entity, never its rank).
PALETTE = {
    "Centralized": "#2a78d6",
    "Federated IID": "#eb6834",
    "Federated Non-IID": "#1baf7a",
    "Hospital A (local)": "#eda100",
    "Hospital B (local)": "#e87ba4",
    "Hospital C (local)": "#008300",
}
SEQUENTIAL_CMAP = "Blues"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3df"
METRIC_LABELS = {"accuracy": "Accuracy", "precision": "Precision", "recall": "Recall", "f1": "F1-score",
                 "roc_auc": "ROC-AUC"}

plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "font.size": 10,
    "axes.edgecolor": GRID,
    "axes.labelcolor": TEXT_2,
    "axes.titlecolor": TEXT,
    "axes.titleweight": "bold",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.axisbelow": True,
    "figure.constrained_layout.use": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "xtick.color": TEXT_2,
    "ytick.color": TEXT_2,
    "legend.frameon": False,
    "lines.linewidth": 2.0,
    "lines.markersize": 5,
})


def color_for(name: str) -> str:
    return PALETTE.get(name, "#52514e")


def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    return path


def plot_training_curves(results: dict[str, Any], out: Path) -> Path:
    """Train/validation loss: centralized and local per epoch, federated per round (separate axes)."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    ax = axes[0]
    c = results["centralized"]
    ep = np.arange(1, len(c["history"]["train_loss"]) + 1)
    ax.plot(ep, c["history"]["train_loss"], color=color_for("Centralized"), label="Centralized - train")
    ax.plot(ep, c["history"]["val_loss"], color=color_for("Centralized"), ls="--", label="Centralized - validation")
    ax.set(title="Centralized learning", xlabel="Epoch", ylabel="BCE loss")
    ax.legend()

    ax = axes[1]
    for r in results["local_only"]:
        col = color_for(r["name"])
        ep = np.arange(1, len(r["history"]["train_loss"]) + 1)
        ax.plot(ep, r["history"]["train_loss"], color=col, label=f"{r['hospital']} - train")
        ax.plot(ep, r["history"]["val_loss"], color=col, ls="--", label=f"{r['hospital']} - validation")
    ax.set(title="Local-only learning (non-IID hospitals)", xlabel="Epoch", ylabel="BCE loss")
    ax.legend(fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.16))

    ax = axes[2]
    for key in ("federated_iid", "federated_non_iid"):
        r = results[key]
        rounds = [h["round"] for h in r["history"]]
        col = color_for(r["name"])
        ax.plot(rounds, [h["train_loss"] for h in r["history"]], color=col, marker="o", label=f"{r['name']} - train")
        ax.plot(rounds, [h["val_loss"] for h in r["history"]], color=col, marker="o", ls="--",
                label=f"{r['name']} - validation")
    ax.set(title="Federated learning (FedAvg)", xlabel="Communication round", ylabel="BCE loss (sample-weighted)")
    ax.xaxis.get_major_locator().set_params(integer=True)
    ax.legend(fontsize=8)
    fig.suptitle("Training and validation loss", fontweight="bold", color=TEXT)
    return _save(fig, out)


def plot_accuracy_comparison(rows: list[dict[str, Any]], robustness: dict[str, Any] | None, out: Path) -> Path:
    """Horizontal bars of test accuracy (seed 42) and, if available, mean +/- std over seeds."""
    ncols = 2 if robustness else 1
    fig, axes = plt.subplots(1, ncols, figsize=(6.5 * ncols, 3.8), squeeze=False)
    names = [r["name"] for r in rows]
    y = np.arange(len(names))[::-1]

    ax = axes[0, 0]
    vals = [r["accuracy"] for r in rows]
    ax.barh(y, vals, color=[color_for(n) for n in names], height=0.6, edgecolor="white", linewidth=2)
    for yi, v in zip(y, vals):
        ax.text(v + 0.003, yi, f"{v:.3f}", va="center", fontsize=9, color=TEXT)
    ax.set_yticks(y, names)
    ax.set(xlim=(min(0.85, min(vals) - 0.02), 1.02), xlabel="Test accuracy", title=f"Test accuracy (seed {rows[0]['seed']})")
    ax.grid(axis="y", visible=False)

    if robustness:
        ax = axes[0, 1]
        rn = [n for n in names if n in robustness]
        means = [robustness[n]["accuracy"]["mean"] for n in rn]
        stds = [robustness[n]["accuracy"]["std"] for n in rn]
        yy = np.arange(len(rn))[::-1]
        ax.barh(yy, means, xerr=stds, color=[color_for(n) for n in rn], height=0.6, edgecolor="white", linewidth=2,
                error_kw={"ecolor": TEXT_2, "capsize": 3, "lw": 1})
        for yi, m, s in zip(yy, means, stds):
            ax.text(m + s + 0.004, yi, f"{m:.3f} ± {s:.3f}", va="center", fontsize=9, color=TEXT)
        ax.set_yticks(yy, rn)
        n_seeds = robustness[rn[0]]["accuracy"]["n"]
        ax.set(xlim=(min(0.85, min(means) - 0.05), 1.04), xlabel="Test accuracy",
               title=f"Mean ± std over {n_seeds} seeds")
        ax.grid(axis="y", visible=False)
    return _save(fig, out)


def plot_metric_comparison(rows: list[dict[str, Any]], out: Path) -> Path:
    """Grouped bars: every method on every metric."""
    metrics = list(METRIC_LABELS)
    names = [r["name"] for r in rows]
    width = 0.8 / len(names)
    x = np.arange(len(metrics))
    fig, ax = plt.subplots(figsize=(12, 4.2))
    for i, r in enumerate(rows):
        ax.bar(x + (i - (len(names) - 1) / 2) * width, [r[m] for m in metrics], width * 0.92,
               color=color_for(r["name"]), label=r["name"])
    ax.set_xticks(x, [METRIC_LABELS[m] for m in metrics])
    ax.set(ylim=(0.7, 1.01), ylabel="Score on held-out test set", title="All metrics by training paradigm")
    ax.grid(axis="x", visible=False)
    ax.legend(ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.32))
    return _save(fig, out)


def plot_roc_curves(results: dict[str, Any], out: Path) -> Path:
    """ROC curves: global models (left) and local hospital models vs. federated non-IID (right)."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    groups = [
        ("Global models", [results["centralized"], results["federated_iid"], results["federated_non_iid"]]),
        ("Local hospital models vs. Federated Non-IID", results["local_only"] + [results["federated_non_iid"]]),
    ]
    for ax, (title, runs) in zip(axes, groups):
        ax.plot([0, 1], [0, 1], color=GRID, ls=":", lw=1.2)
        for r in runs:
            roc = r["metrics"]["roc_curve"]
            ax.plot(roc["fpr"], roc["tpr"], color=color_for(r["name"]),
                    label=f"{r['name']} (AUC={r['metrics']['roc_auc']:.3f})")
        ax.set(title=title, xlabel="False positive rate", ylabel="True positive rate", xlim=(-0.01, 1.01),
               ylim=(-0.01, 1.01), aspect="equal")
        ax.legend(loc="lower right", fontsize=8)
    return _save(fig, out)


def plot_confusion_matrices(runs: list[dict[str, Any]], out: Path) -> Path:
    """One confusion matrix per method (rows = true class, cols = predicted class)."""
    n = len(runs)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 3.6 * nrows), squeeze=False)
    labels = ["Benign", "Malignant"]
    for ax, r in zip(axes.ravel(), runs):
        cm = np.array(r["metrics"]["confusion_matrix"])
        ax.imshow(cm, cmap=SEQUENTIAL_CMAP, vmin=0, vmax=cm.sum())
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, str(v), ha="center", va="center", fontsize=13,
                    color="white" if v > cm.sum() * 0.45 else TEXT)
        ax.set_xticks([0, 1], labels)
        ax.set_yticks([0, 1], labels)
        ax.set(xlabel="Predicted", ylabel="True", title=r["name"])
        ax.grid(False)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.suptitle("Confusion matrices on the held-out test set", fontweight="bold", color=TEXT)
    return _save(fig, out)


def plot_federated_convergence(rounds_runs: dict[str, dict[str, Any]], centralized: dict[str, Any], out: Path) -> Path:
    """Per-round global test accuracy, ROC-AUC and loss for IID vs. non-IID (longest run)."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.0))
    panels = [("test_accuracy", "Test accuracy", "accuracy"), ("test_roc_auc", "Test ROC-AUC", "roc_auc"),
              ("test_loss", "Test BCE loss", "test_loss")]
    for ax, (key, label, ckey) in zip(axes, panels):
        for name, r in rounds_runs.items():
            rounds = [h["round"] for h in r["history"]]
            ax.plot(rounds, [h[key] for h in r["history"]], marker="o", color=color_for(name), label=name)
        ax.axhline(centralized["metrics"][ckey], color=color_for("Centralized"), ls="--", lw=1.5,
                   label="Centralized (final)")
        ax.set(title=label, xlabel="Communication round", ylabel=label)
        ax.xaxis.get_major_locator().set_params(integer=True)
        ax.legend(fontsize=8)
    fig.suptitle("Federated convergence of the global model", fontweight="bold", color=TEXT)
    return _save(fig, out)


def plot_iid_vs_noniid(robustness: dict[str, Any], out: Path) -> Path:
    """Grouped bars (mean +/- std over seeds): Federated IID vs. Federated Non-IID."""
    metrics = list(METRIC_LABELS)
    x = np.arange(len(metrics))
    fig, ax = plt.subplots(figsize=(9, 4.0))
    for i, name in enumerate(["Federated IID", "Federated Non-IID"]):
        means = [robustness[name][m]["mean"] for m in metrics]
        stds = [robustness[name][m]["std"] for m in metrics]
        ax.bar(x + (i - 0.5) * 0.38, means, 0.36, yerr=stds, color=color_for(name), label=name,
               error_kw={"ecolor": TEXT_2, "capsize": 3, "lw": 1})
    n_seeds = robustness["Federated IID"]["accuracy"]["n"]
    ax.set_xticks(x, [METRIC_LABELS[m] for m in metrics])
    ax.set(ylim=(0.8, 1.02), ylabel="Test score", title=f"IID vs. Non-IID federated learning (mean ± std, {n_seeds} seeds)")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="lower right")
    return _save(fig, out)


def plot_rounds_comparison(rounds_table: list[dict[str, Any]], out: Path) -> Path:
    """Final test metrics after 3 / 5 / 10 rounds for both partitions."""
    metrics = ["accuracy", "f1", "roc_auc"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(13, 3.8))
    for ax, m in zip(axes, metrics):
        for mode_name in ("Federated IID", "Federated Non-IID"):
            rows = sorted([r for r in rounds_table if r["name"] == mode_name], key=lambda r: r["num_rounds"])
            ax.plot([r["num_rounds"] for r in rows], [r[m] for r in rows], marker="o", color=color_for(mode_name),
                    label=mode_name)
        ax.set(title=METRIC_LABELS[m], xlabel="Communication rounds", ylabel=METRIC_LABELS[m])
        ax.set_xticks(sorted({r["num_rounds"] for r in rounds_table}))
        ax.legend(fontsize=8)
    fig.suptitle("Effect of the number of communication rounds", fontweight="bold", color=TEXT)
    return _save(fig, out)


def plot_partitions(partitions: dict[str, list[dict[str, Any]]], out: Path) -> Path:
    """Stacked bars of benign/malignant counts per hospital for each partition scheme."""
    fig, axes = plt.subplots(1, len(partitions), figsize=(5.5 * len(partitions), 3.8), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (title, rows) in zip(axes, partitions.items()):
        names = [r["hospital"] for r in rows]
        ben = [r["n_benign"] for r in rows]
        mal = [r["n_malignant"] for r in rows]
        ax.bar(names, ben, 0.6, color="#9ec5ef", label="Benign", edgecolor="white", linewidth=2)
        ax.bar(names, mal, 0.6, bottom=ben, color="#1c5cab", label="Malignant", edgecolor="white", linewidth=2)
        for i, r in enumerate(rows):
            ax.text(i, r["n_total"] + 4, f"n={r['n_total']}\n{r['malignant_ratio']:.0%} malignant", ha="center",
                    fontsize=8.5, color=TEXT)
        ax.set(title=title, ylabel="Samples (train + validation)")
        ax.set_ylim(0, max(r["n_total"] for rs in partitions.values() for r in rs) * 1.25)
        ax.grid(axis="x", visible=False)
        ax.legend(loc="upper right")
    fig.suptitle("Hospital data distributions", fontweight="bold", color=TEXT)
    return _save(fig, out)
