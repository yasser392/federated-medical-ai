"""Experiments C, D, E: federated simulation of several hospitals with FedAvg."""

from __future__ import annotations

import logging
from typing import Any, Callable

import numpy as np
import torch

from src.data.partition import PartitionMode
from src.data.pipeline import ExperimentData, partition_summary, prepare_data
from src.data.preprocessing import Standardizer, to_tensor_dataset
from src.evaluation.metrics import compute_metrics
from src.federated.client import HospitalClient
from src.federated.server import FederatedServer
from src.models.neural_network import build_model, count_parameters, get_parameters, set_parameters
from src.models.training import evaluate_loss, predict_proba
from src.privacy.communication import CommunicationTracker, audit_payload
from src.utils.reproducibility import get_device, set_seed

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[int, int, dict[str, Any]], None]


def run_federated(
    cfg: dict,
    mode: PartitionMode = "non_iid",
    num_rounds: int | None = None,
    seed: int | None = None,
    progress_callback: ProgressCallback | None = None,
    data: ExperimentData | None = None,
    audit: bool = True,
) -> dict[str, Any]:
    """Run a full federated training simulation.

    Steps: (1) hospitals compute local sufficient statistics and the server builds a global
    standardizer from them; (2) the server initializes the global model; (3) for each round the
    server distributes parameters, hospitals train locally, and the server aggregates with
    sample-weighted FedAvg; (4) the final global model is evaluated on the held-out test set.

    Args:
        cfg: Configuration dictionary.
        mode: ``"iid"`` or ``"non_iid"`` partitioning.
        num_rounds: Communication rounds (defaults to ``cfg['federated']['num_rounds']``).
        seed: Random seed (defaults to ``cfg['seed']``).
        progress_callback: Called as ``callback(round, num_rounds, round_record)`` after each round.
        data: Pre-built experiment data (optional).
        audit: Whether to audit every client->server payload against the raw local data.

    Returns:
        A JSON-serializable result dictionary.
    """
    seed = cfg["seed"] if seed is None else seed
    num_rounds = int(num_rounds or cfg["federated"]["num_rounds"])
    set_seed(seed)
    get_device(cfg["runtime"].get("num_threads"))
    data = data or prepare_data(cfg, mode, seed)

    # (1) federated standardization from (n, sum, sum_sq) only
    scaler = Standardizer.fit_federated([h.train.X for h in data.hospitals])
    test_ds = to_tensor_dataset(data.test, scaler)

    clients, raw = {}, {}
    for i, h in enumerate(data.hospitals):
        hc = HospitalClient(h.name, to_tensor_dataset(h.train, scaler), to_tensor_dataset(h.val, scaler), cfg,
                            seed=seed + 100 * (i + 1))
        clients[h.name] = hc.to_client()
        raw[h.name] = (h.train.X, h.train.y, scaler.transform(h.train.X))

    # (2) global model initialization (same seed-determined weights as the centralized baseline)
    torch.manual_seed(seed)
    global_model = build_model(cfg)
    threshold = cfg["training"]["threshold"]

    def evaluate_fn(server_round: int, parameters: list[np.ndarray], config: dict) -> tuple[float, dict]:
        set_parameters(global_model, parameters)
        m = compute_metrics(*predict_proba(global_model, test_ds), threshold)
        return evaluate_loss(global_model, test_ds), {k: m[k] for k in ("accuracy", "precision", "recall", "f1", "roc_auc")}

    tracker = CommunicationTracker()

    def audit_fn(cid: str, server_round: int, arrays: list[np.ndarray]) -> None:
        X_raw, y_raw, X_scaled = raw[cid]
        for X in (X_raw, X_scaled):
            tracker.audits.append(audit_payload(arrays, X, y_raw, cid, server_round))

    server = FederatedServer(get_parameters(global_model), clients, cfg, evaluate_fn, tracker,
                             audit_fn if audit else None)

    # (3) communication rounds
    history: list[dict[str, Any]] = []
    for rnd in range(1, num_rounds + 1):
        record = server.run_round(rnd)
        history.append(record)
        if progress_callback is not None:
            progress_callback(rnd, num_rounds, record)

    # (4) final evaluation of the global model
    set_parameters(global_model, server.global_ndarrays)
    metrics = compute_metrics(*predict_proba(global_model, test_ds), threshold)
    raw_bytes = {h.name: int(h.train.X.nbytes + h.train.y.nbytes + h.val.X.nbytes + h.val.y.nbytes)
                 for h in data.hospitals}
    comm = tracker.summary(len(clients), num_rounds, raw_bytes)
    comm["model_parameters"] = count_parameters(global_model)
    logger.info("Federated %s (%d rounds) | acc=%.4f f1=%.4f auc=%.4f | %.1f KB exchanged, audits passed=%s",
                mode, num_rounds, metrics["accuracy"], metrics["f1"], metrics["roc_auc"], comm["kilobytes_total"],
                comm["audits_passed"])
    return {
        "name": f"Federated {'IID' if mode == 'iid' else 'Non-IID'}",
        "mode": mode,
        "seed": seed,
        "num_rounds": num_rounds,
        "local_epochs": cfg["federated"]["local_epochs"],
        "aggregation_backend": server.backend,
        "partition": partition_summary(data),
        "history": history,
        "metrics": metrics,
        "communication": comm,
        "communication_log": tracker.to_dict(),
    }
