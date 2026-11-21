"""Build comparison tables and figures from the saved metrics (no retraining).

Usage:
    python -m experiments.compare_results
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from src.data.pipeline import partition_summary, prepare_data
from src.evaluation import visualization as viz
from src.utils.config import load_config, resolve_path

logger = logging.getLogger(__name__)


def _load(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found - run `python -m experiments.run_all` first")
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def comparison_rows(main: dict[str, Any]) -> list[dict[str, Any]]:
    """One row per method: Centralized, the three local models, Federated IID, Federated Non-IID."""
    runs = [main["centralized"], *main["local_only"], main["federated_iid"], main["federated_non_iid"]]
    rows = []
    for r in runs:
        m = r["metrics"]
        tn, fp, fn, tp = (v for row in m["confusion_matrix"] for v in row)
        rows.append({
            "name": r["name"], "seed": r["seed"],
            "accuracy": m["accuracy"], "precision": m["precision"], "recall": m["recall"], "f1": m["f1"],
            "roc_auc": m["roc_auc"], "test_loss": m["test_loss"], "TN": tn, "FP": fp, "FN": fn, "TP": tp,
        })
    return rows


def rounds_rows(rounds_runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{"name": r["name"], "num_rounds": r["num_rounds"], **{k: r["metrics"][k] for k in
             ("accuracy", "precision", "recall", "f1", "roc_auc", "test_loss")},
             "kilobytes_total": r["communication"]["kilobytes_total"]} for r in rounds_runs]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    metrics_dir = resolve_path(cfg, "metrics_dir")
    fig_dir = resolve_path(cfg, "figures_dir")

    main_res = _load(metrics_dir / "main_results.json")
    rounds_runs = _load(metrics_dir / "rounds_comparison.json")
    rob_path = metrics_dir / "robustness.json"
    robustness = _load(rob_path)["summary"] if rob_path.is_file() else None

    # ---- tables ----
    rows = comparison_rows(main_res)
    df = pd.DataFrame(rows)
    df.to_csv(metrics_dir / "comparison_table.csv", index=False)
    cols = ["name", "accuracy", "precision", "recall", "f1", "roc_auc", "test_loss", "TN", "FP", "FN", "TP"]
    (metrics_dir / "comparison_table.md").write_text(df[cols].to_markdown(index=False, floatfmt=".4f"), encoding="utf-8")

    rr = pd.DataFrame(rounds_rows(rounds_runs))
    rr.to_csv(metrics_dir / "rounds_comparison.csv", index=False)
    (metrics_dir / "rounds_comparison.md").write_text(rr.to_markdown(index=False, floatfmt=".4f"), encoding="utf-8")

    if robustness:
        rob_rows = [{"name": n, **{f"{k}": f"{v['mean']:.4f} ± {v['std']:.4f}" for k, v in s.items()}}
                    for n, s in robustness.items()]
        pd.DataFrame(rob_rows).to_csv(metrics_dir / "robustness_table.csv", index=False)
        (metrics_dir / "robustness_table.md").write_text(pd.DataFrame(rob_rows).to_markdown(index=False),
                                                         encoding="utf-8")

    comm = {k: main_res[k]["communication"] for k in ("federated_iid", "federated_non_iid")}
    with (metrics_dir / "communication_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(comm, fh, indent=2)

    # ---- figures ----
    runs = [main_res["centralized"], *main_res["local_only"], main_res["federated_iid"], main_res["federated_non_iid"]]
    longest = max(cfg["federated"]["rounds_comparison"])
    conv = {r["name"]: r for r in rounds_runs if r["num_rounds"] == longest}
    seed = main_res["seed"]
    partitions = {"IID partition": partition_summary(prepare_data(cfg, "iid", seed)),
                  f"Non-IID partition (Dirichlet α={cfg['partition']['dirichlet_alpha']})":
                      partition_summary(prepare_data(cfg, "non_iid", seed))}

    produced = [
        viz.plot_training_curves(main_res, fig_dir / "training_loss_curves.png"),
        viz.plot_accuracy_comparison(rows, robustness, fig_dir / "accuracy_comparison.png"),
        viz.plot_metric_comparison(rows, fig_dir / "metrics_comparison.png"),
        viz.plot_roc_curves(main_res, fig_dir / "roc_curves.png"),
        viz.plot_confusion_matrices(runs, fig_dir / "confusion_matrices.png"),
        viz.plot_federated_convergence(conv, main_res["centralized"], fig_dir / "federated_convergence.png"),
        viz.plot_rounds_comparison(rounds_rows(rounds_runs), fig_dir / "rounds_comparison.png"),
        viz.plot_partitions(partitions, fig_dir / "partition_distribution.png"),
    ]
    if robustness:
        produced.append(viz.plot_iid_vs_noniid(robustness, fig_dir / "iid_vs_noniid.png"))

    print("\nComparison table (held-out test set, seed %d):\n" % seed)
    print(df[cols].to_markdown(index=False, floatfmt=".4f"))
    print("\nCommunication rounds comparison:\n")
    print(rr.to_markdown(index=False, floatfmt=".4f"))
    if robustness:
        print("\nRobustness (mean ± std over seeds):\n")
        print(pd.DataFrame(rob_rows).to_markdown(index=False))
    print("\nFigures written:")
    for p in produced:
        print("  -", p.relative_to(resolve_path(cfg, "results_dir").parent))


if __name__ == "__main__":
    main()
