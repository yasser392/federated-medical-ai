"""Integration tests for federated training, baselines and the privacy audit."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from flwr.common import FitIns, ndarrays_to_parameters, parameters_to_ndarrays

from src.centralized.train import run_centralized, run_local_only
from src.data.pipeline import prepare_data
from src.data.preprocessing import Standardizer, to_tensor_dataset
from src.federated.aggregation import fedavg
from src.federated.client import HospitalClient
from src.federated.simulation import run_federated
from src.models.neural_network import build_model, get_parameters
from src.privacy.communication import audit_payload


@pytest.mark.parametrize("mode", ["iid", "non_iid"])
def test_federated_training_runs_on_cpu(fast_cfg: dict, mode: str) -> None:
    res = run_federated(fast_cfg, mode, seed=42)
    assert len(res["history"]) == fast_cfg["federated"]["num_rounds"]
    for h in res["history"]:
        assert math.isfinite(h["train_loss"]) and math.isfinite(h["val_loss"])
    m = res["metrics"]
    assert 0.0 <= m["accuracy"] <= 1.0 and 0.0 <= m["roc_auc"] <= 1.0
    assert m["n_samples"] == 114
    assert res["communication"]["audits_passed"] is True
    assert res["communication"]["raw_data_bytes_transmitted"] == 0
    assert not torch.cuda.is_initialized()


def test_client_returns_only_parameters_and_counts(fast_cfg: dict) -> None:
    data = prepare_data(fast_cfg, "non_iid", 42)
    h = data.hospitals[0]
    scaler = Standardizer.fit(h.train.X)
    client = HospitalClient(h.name, to_tensor_dataset(h.train, scaler), to_tensor_dataset(h.val, scaler), fast_cfg, 1)
    params, n, metrics = client.fit(get_parameters(build_model(fast_cfg)), {"server_round": 1, "local_epochs": 1})
    assert n == len(h.train)
    assert [p.shape for p in params] == [p.shape for p in get_parameters(build_model(fast_cfg))]
    assert set(metrics) == {"train_loss", "client"}
    assert math.isfinite(metrics["train_loss"])


def test_one_round_global_model_is_weighted_average(fast_cfg: dict) -> None:
    """Re-run the clients of round 1 manually and check the server's model equals FedAvg of them."""
    cfg = fast_cfg
    from src.utils.config import override
    # dropout disabled so local training is fully determined by the seeded data shuffling
    cfg1 = override(cfg, **{"federated.num_rounds": 1, "model.dropout": 0.0})
    data = prepare_data(cfg1, "non_iid", 42)
    scaler = Standardizer.fit_federated([h.train.X for h in data.hospitals])
    torch.manual_seed(42)
    init = get_parameters(build_model(cfg1))
    updates = []
    for i, h in enumerate(data.hospitals):
        c = HospitalClient(h.name, to_tensor_dataset(h.train, scaler), to_tensor_dataset(h.val, scaler), cfg1,
                           seed=42 + 100 * (i + 1))
        res = c.to_client().fit(FitIns(ndarrays_to_parameters(init), {"server_round": 1, "local_epochs": 1}))
        updates.append((parameters_to_ndarrays(res.parameters), res.num_examples))
    expected = fedavg(updates)

    from src.federated.server import FederatedServer
    clients = {}
    for i, h in enumerate(data.hospitals):
        clients[h.name] = HospitalClient(h.name, to_tensor_dataset(h.train, scaler), to_tensor_dataset(h.val, scaler),
                                         cfg1, seed=42 + 100 * (i + 1)).to_client()
    server = FederatedServer(init, clients, cfg1)
    server.run_round(1)
    got = server.global_ndarrays
    for g, e in zip(got, expected):
        assert g.shape == e.shape
        np.testing.assert_allclose(g, e, rtol=1e-5, atol=1e-6)
    # the aggregated model differs from the initial one (training happened)
    assert any(not np.allclose(g, i0) for g, i0 in zip(got, init))


def test_baselines_run(fast_cfg: dict) -> None:
    c = run_centralized(fast_cfg, 42)
    assert all(math.isfinite(v) for v in c["history"]["train_loss"])
    locs = run_local_only(fast_cfg, "non_iid", 42)
    assert len(locs) == fast_cfg["partition"]["num_clients"]
    assert all(r["metrics"]["n_samples"] == 114 for r in locs)


def test_audit_detects_raw_data_leak() -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20, 30))
    y = rng.integers(0, 2, size=20).astype(float)
    clean = [rng.normal(size=(64, 30)), rng.normal(size=64)]
    assert audit_payload(clean, X, y, "A", 1)["passed"]
    leaky = [X.copy(), rng.normal(size=5)]
    rep = audit_payload(leaky, X, y, "A", 1)
    assert not rep["passed"] and rep["raw_rows_found_in_payload"] == 20


def test_federated_is_reproducible(fast_cfg: dict) -> None:
    a = run_federated(fast_cfg, "non_iid", seed=42, audit=False)
    b = run_federated(fast_cfg, "non_iid", seed=42, audit=False)
    assert a["metrics"]["accuracy"] == b["metrics"]["accuracy"]
    assert a["history"][-1]["train_loss"] == pytest.approx(b["history"][-1]["train_loss"])
