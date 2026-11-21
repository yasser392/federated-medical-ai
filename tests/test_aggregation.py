"""Tests for FedAvg aggregation (native formula and Flower's strategy)."""

from __future__ import annotations

import numpy as np
import pytest
from flwr.common import Code, FitRes, Status, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.strategy import FedAvg

from src.federated.aggregation import fedavg, weighted_average


def _updates() -> list[tuple[list[np.ndarray], int]]:
    rng = np.random.default_rng(0)
    return [([rng.normal(size=(4, 3)).astype(np.float32), rng.normal(size=3).astype(np.float32)], n)
            for n in (10, 30, 60)]


def test_fedavg_matches_formula() -> None:
    ups = _updates()
    agg = fedavg(ups)
    total = sum(n for _, n in ups)
    for layer in range(2):
        expected = sum(n * p[layer].astype(np.float64) for p, n in ups) / total
        np.testing.assert_allclose(agg[layer], expected, rtol=1e-6)
        assert agg[layer].dtype == np.float32


def test_fedavg_simple_example() -> None:
    a = ([np.array([0.0, 0.0])], 1)
    b = ([np.array([4.0, 8.0])], 3)
    np.testing.assert_allclose(fedavg([a, b])[0], [3.0, 6.0])


def test_fedavg_identical_clients_is_identity() -> None:
    p = [np.ones((2, 2)), np.full(2, 3.0)]
    out = fedavg([(p, 5), (p, 50)])
    for x, y in zip(out, p):
        np.testing.assert_allclose(x, y)


def test_fedavg_input_validation() -> None:
    with pytest.raises(ValueError):
        fedavg([])
    with pytest.raises(ValueError):
        fedavg([([np.ones(2)], 0)])
    with pytest.raises(ValueError):
        fedavg([([np.ones(2)], 1), ([np.ones(3)], 1)])


def test_flower_fedavg_equals_native() -> None:
    ups = _updates()
    results = [(None, FitRes(Status(Code.OK, ""), ndarrays_to_parameters(p), n, {})) for p, n in ups]
    params, _ = FedAvg(inplace=False).aggregate_fit(1, results, [])  # type: ignore[arg-type]
    flower = parameters_to_ndarrays(params)
    for f, nat in zip(flower, fedavg(ups)):
        np.testing.assert_allclose(f, nat, rtol=1e-6, atol=1e-7)


def test_weighted_average() -> None:
    assert weighted_average([(1, 0.0), (3, 1.0)]) == pytest.approx(0.75)
