"""Run every experiment (A-E + multi-seed robustness) and write metrics, logs and figures.

Usage:
    python -m experiments.run_all [--config config.yaml] [--skip-robustness]
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path
from typing import Any

import torch

from experiments import compare_results
from src.centralized.train import run_centralized, run_local_only
from src.evaluation.metrics import summarize_runs
from src.federated.simulation import run_federated
from src.utils.config import load_config, override, resolve_path, setup_logging


def _write_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2)


def _strip_log(run: dict[str, Any]) -> dict[str, Any]:
    """Drop the per-message communication log (saved separately) from a federated result."""
    return {k: v for k, v in run.items() if k != "communication_log"}


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="Path to a YAML config (default: config.yaml)")
    parser.add_argument("--skip-robustness", action="store_true", help="Skip the multi-seed repetition")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    log = setup_logging(cfg, "run_all")
    metrics_dir = resolve_path(cfg, "metrics_dir")
    logs_dir = resolve_path(cfg, "logs_dir")
    seed = cfg["seed"]
    t0 = time.time()
    log.info("Python %s | torch %s | CUDA available: %s (not used) | device: cpu", platform.python_version(),
             torch.__version__, torch.cuda.is_available())

    # ---- Experiments A-D (main seed) -------------------------------------------------------------
    log.info("=== Experiment A: centralized learning ===")
    centralized = run_centralized(cfg, seed)
    log.info("=== Experiment B: local-only learning ===")
    local_non_iid = run_local_only(cfg, "non_iid", seed)
    local_iid = run_local_only(cfg, "iid", seed)
    log.info("=== Experiment C: federated learning, IID ===")
    fed_iid = run_federated(cfg, "iid", seed=seed)
    log.info("=== Experiment D: federated learning, non-IID ===")
    fed_non_iid = run_federated(cfg, "non_iid", seed=seed)

    _write_json(fed_non_iid["communication_log"], logs_dir / "communication_log_non_iid.json")
    _write_json(fed_iid["communication_log"], logs_dir / "communication_log_iid.json")
    main_results = {
        "seed": seed,
        "config": cfg,
        "centralized": centralized,
        "local_only": local_non_iid,
        "local_only_iid": local_iid,
        "federated_iid": _strip_log(fed_iid),
        "federated_non_iid": _strip_log(fed_non_iid),
    }
    _write_json(main_results, metrics_dir / "main_results.json")

    # ---- Experiment E: number of communication rounds ---------------------------------------------
    log.info("=== Experiment E: communication rounds %s ===", cfg["federated"]["rounds_comparison"])
    rounds_runs = []
    for mode in ("iid", "non_iid"):
        for n_rounds in cfg["federated"]["rounds_comparison"]:
            r = run_federated(cfg, mode, num_rounds=n_rounds, seed=seed, audit=False)
            rounds_runs.append(_strip_log(r))
    _write_json(rounds_runs, metrics_dir / "rounds_comparison.json")

    # ---- Aggregation backend cross-check (Flower FedAvg vs. native formula) ------------------------
    native = run_federated(override(cfg, **{"federated.aggregation_backend": "native"}), "non_iid", seed=seed,
                           audit=False)
    backend_check = {
        "flower_metrics": {k: fed_non_iid["metrics"][k] for k in ("accuracy", "f1", "roc_auc", "test_loss")},
        "native_metrics": {k: native["metrics"][k] for k in ("accuracy", "f1", "roc_auc", "test_loss")},
    }
    backend_check["max_abs_difference"] = max(
        abs(backend_check["flower_metrics"][k] - backend_check["native_metrics"][k]) for k in backend_check["flower_metrics"]
    )
    log.info("Backend cross-check: max |flower - native| over final metrics = %.2e", backend_check["max_abs_difference"])
    _write_json(backend_check, metrics_dir / "aggregation_backend_check.json")

    # ---- Multi-seed robustness ---------------------------------------------------------------------
    if not args.skip_robustness:
        seeds = cfg.get("robustness", {}).get("seeds", [seed])
        log.info("=== Robustness over seeds %s ===", seeds)
        per_seed: dict[str, list[dict[str, Any]]] = {}

        def add(name: str, metrics: dict[str, Any]) -> None:
            per_seed.setdefault(name, []).append({k: metrics[k] for k in ("accuracy", "precision", "recall", "f1", "roc_auc")})

        for s in seeds:
            add("Centralized", run_centralized(cfg, s)["metrics"])
            for r in run_local_only(cfg, "non_iid", s):
                add(r["name"], r["metrics"])
            add("Federated IID", run_federated(cfg, "iid", seed=s, audit=False)["metrics"])
            add("Federated Non-IID", run_federated(cfg, "non_iid", seed=s, audit=False)["metrics"])
        _write_json({"seeds": seeds, "per_seed": per_seed,
                     "summary": {name: summarize_runs(runs) for name, runs in per_seed.items()}},
                    metrics_dir / "robustness.json")

    elapsed = time.time() - t0
    _write_json({"elapsed_seconds": round(elapsed, 1), "python": platform.python_version(),
                 "torch": torch.__version__, "device": "cpu"}, metrics_dir / "run_info.json")
    log.info("All experiments finished in %.1f s", elapsed)

    compare_results.main([] if args.config is None else ["--config", args.config])
    return main_results


if __name__ == "__main__":
    main()
