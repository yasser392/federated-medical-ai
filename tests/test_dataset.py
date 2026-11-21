"""Tests for dataset loading, splitting and preprocessing."""

from __future__ import annotations

import numpy as np
from sklearn.preprocessing import StandardScaler

from src.data.dataset import load_dataset, train_test_split_dataset
from src.data.pipeline import prepare_data
from src.data.preprocessing import Standardizer


def test_dataset_shape_and_labels() -> None:
    ds = load_dataset()
    assert ds.X.shape == (569, 30)
    assert ds.y.shape == (569,)
    assert set(np.unique(ds.y)) == {0, 1}
    assert int(ds.y.sum()) == 212  # 212 malignant (positive class) / 357 benign
    assert len(ds.feature_names) == 30


def test_stratified_split_is_disjoint_and_balanced(cfg: dict) -> None:
    ds = load_dataset()
    pool, test = train_test_split_dataset(ds, cfg["data"]["test_size"], cfg["seed"])
    assert len(pool) + len(test) == 569
    assert not set(pool.indices) & set(test.indices)
    assert abs(pool.y.mean() - test.y.mean()) < 0.02  # stratification preserves class ratio


def test_split_is_reproducible(cfg: dict) -> None:
    ds = load_dataset()
    _, t1 = train_test_split_dataset(ds, 0.2, 42)
    _, t2 = train_test_split_dataset(ds, 0.2, 42)
    np.testing.assert_array_equal(t1.indices, t2.indices)


def test_test_set_identical_across_partition_modes(cfg: dict) -> None:
    a = prepare_data(cfg, "iid", 42)
    b = prepare_data(cfg, "non_iid", 42)
    np.testing.assert_array_equal(a.test.indices, b.test.indices)


def test_standardizer_matches_sklearn_and_is_fit_on_train_only() -> None:
    ds = load_dataset()
    pool, test = train_test_split_dataset(ds, 0.2, 42)
    s = Standardizer.fit(pool.X)
    ref = StandardScaler().fit(pool.X)
    np.testing.assert_allclose(s.mean, ref.mean_, rtol=1e-6)
    np.testing.assert_allclose(s.std, ref.scale_, rtol=1e-5)
    # statistics differ from a scaler fit on all data -> the test set was not used
    assert not np.allclose(s.mean, ds.X.mean(axis=0))


def test_federated_standardizer_equals_pooled_fit(cfg: dict) -> None:
    data = prepare_data(cfg, "non_iid", 42)
    fed = Standardizer.fit_federated([h.train.X for h in data.hospitals])
    pooled = Standardizer.fit(np.concatenate([h.train.X for h in data.hospitals]))
    np.testing.assert_allclose(fed.mean, pooled.mean, rtol=1e-9)
    np.testing.assert_allclose(fed.std, pooled.std, rtol=1e-6)
