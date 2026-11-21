"""Native implementation of sample-weighted Federated Averaging (FedAvg).

    W_global = sum_k(n_k * W_k) / sum_k(n_k)

where ``W_k`` are the parameters returned by client ``k`` and ``n_k`` its number of training samples.
This is used (a) as an alternative backend to Flower's ``FedAvg`` strategy and (b) in the unit tests
to verify that Flower's aggregation is numerically identical to the formula.
"""

from __future__ import annotations

import numpy as np

ClientUpdate = tuple[list[np.ndarray], int]


def fedavg(updates: list[ClientUpdate]) -> list[np.ndarray]:
    """Aggregate client parameters with sample-weighted averaging.

    Args:
        updates: List of ``(parameters, num_examples)``; each ``parameters`` is a list of arrays with
            identical shapes across clients.

    Returns:
        The aggregated parameter list (same dtypes as the first client's arrays).

    Raises:
        ValueError: On empty input, non-positive total weight, or mismatched shapes.
    """
    if not updates:
        raise ValueError("fedavg requires at least one client update")
    if any(n < 0 for _, n in updates):
        raise ValueError("num_examples must be non-negative")
    total = sum(n for _, n in updates)
    if total <= 0:
        raise ValueError("Total number of examples must be positive")
    reference = updates[0][0]
    for params, _ in updates[1:]:
        if len(params) != len(reference) or any(a.shape != b.shape for a, b in zip(params, reference)):
            raise ValueError("All client updates must have identical parameter shapes")

    aggregated = []
    for layer in range(len(reference)):
        acc = np.zeros(reference[layer].shape, dtype=np.float64)
        for params, n in updates:
            acc += n * params[layer].astype(np.float64)
        aggregated.append((acc / total).astype(reference[layer].dtype))
    return aggregated


def weighted_average(values: list[tuple[int, float]]) -> float:
    """Sample-weighted mean of scalar metrics ``[(n_k, v_k), ...]``."""
    total = sum(n for n, _ in values)
    return float(sum(n * v for n, v in values) / total) if total else float("nan")
