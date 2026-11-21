"""Tests for IID and non-IID hospital partitioning."""

from __future__ import annotations

import numpy as np
import pytest

from src.data.partition import dirichlet_partition, iid_partition
from src.data.pipeline import prepare_data


@pytest.mark.parametrize("mode", ["iid", "non_iid"])
def test_every_training_sample_in_exactly_one_hospital(cfg: dict, mode: str) -> None:
    data = prepare_data(cfg, mode, cfg["seed"])
    ids = np.concatenate([np.concatenate([h.train.indices, h.val.indices]) for h in data.hospitals])
    assert len(ids) == len(set(ids.tolist()))  # no duplicates
    assert set(ids.tolist()) == set(data.pool.indices.tolist())  # full coverage
    assert len(data.hospitals) == cfg["partition"]["num_clients"]


@pytest.mark.parametrize("mode", ["iid", "non_iid"])
def test_no_test_sample_in_hospital_data(cfg: dict, mode: str) -> None:
    data = prepare_data(cfg, mode, cfg["seed"])
    test_ids = set(data.test.indices.tolist())
    for h in data.hospitals:
        assert not test_ids & set(h.train.indices.tolist())
        assert not test_ids & set(h.val.indices.tolist())


def test_iid_partition_is_balanced() -> None:
    y = np.array([0] * 300 + [1] * 150)
    parts = iid_partition(y, 3, seed=42)
    sizes = [len(p) for p in parts]
    assert max(sizes) - min(sizes) <= 1
    ratios = [y[p].mean() for p in parts]
    assert max(ratios) - min(ratios) < 0.02


def test_non_iid_partition_is_skewed_and_valid() -> None:
    y = np.array([0] * 285 + [1] * 170)
    parts = dirichlet_partition(y, 3, alpha=0.5, seed=42, min_samples_per_class=5)
    assert sum(len(p) for p in parts) == len(y)
    assert len(np.unique(np.concatenate(parts))) == len(y)
    for p in parts:
        assert (y[p] == 0).sum() >= 5 and (y[p] == 1).sum() >= 5
    ratios = [y[p].mean() for p in parts]
    assert max(ratios) - min(ratios) > 0.15  # clearly different label distributions


def test_partitions_are_reproducible() -> None:
    y = np.array([0] * 200 + [1] * 120)
    a = dirichlet_partition(y, 3, 0.5, seed=7)
    b = dirichlet_partition(y, 3, 0.5, seed=7)
    for pa, pb in zip(a, b):
        np.testing.assert_array_equal(pa, pb)


def test_configurable_number_of_clients(cfg: dict) -> None:
    from src.utils.config import override
    cfg5 = override(cfg, **{"partition.num_clients": 5, "partition.client_names": []})
    data = prepare_data(cfg5, "iid", 42)
    assert len(data.hospitals) == 5


def test_invalid_mode_raises(cfg: dict) -> None:
    with pytest.raises(ValueError):
        prepare_data(cfg, "bogus", 42)  # type: ignore[arg-type]
