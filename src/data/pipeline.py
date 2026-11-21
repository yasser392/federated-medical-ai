"""End-to-end data preparation shared by all experiments (single source of truth for splits)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.data.dataset import RawDataset, load_dataset, train_test_split_dataset
from src.data.partition import HospitalData, PartitionMode, build_hospitals
from src.utils.config import client_names


@dataclass(frozen=True)
class ExperimentData:
    """Training pool, independent test set and the hospital partition for one seed and mode."""

    pool: RawDataset
    test: RawDataset
    hospitals: list[HospitalData]
    mode: PartitionMode
    seed: int


def prepare_data(cfg: dict, mode: PartitionMode, seed: int) -> ExperimentData:
    """Load the dataset, hold out the test set, and partition the remaining pool across hospitals.

    The test split depends only on ``seed`` (not on ``mode``), so IID and non-IID experiments with the
    same seed are evaluated on exactly the same test samples.
    """
    full = load_dataset()
    pool, test = train_test_split_dataset(full, cfg["data"]["test_size"], seed)
    hospitals = build_hospitals(pool, mode, cfg, client_names(cfg), seed)
    _assert_no_leakage(pool, test, hospitals)
    return ExperimentData(pool=pool, test=test, hospitals=hospitals, mode=mode, seed=seed)


def _assert_no_leakage(pool: RawDataset, test: RawDataset, hospitals: list[HospitalData]) -> None:
    """Guarantee disjoint, complete partitions and no test sample in any hospital dataset."""
    test_idx = set(test.indices.tolist())
    seen: list[int] = []
    for h in hospitals:
        for part in (h.train, h.val):
            ids = part.indices.tolist()
            if test_idx.intersection(ids):
                raise AssertionError(f"Test samples leaked into {h.name}")
            seen.extend(ids)
    if len(seen) != len(set(seen)):
        raise AssertionError("A sample was assigned to more than one hospital split")
    if set(seen) != set(pool.indices.tolist()):
        raise AssertionError("Hospital partitions do not cover the full training pool")


def partition_summary(data: ExperimentData) -> list[dict]:
    """Per-hospital sample counts and malignant ratios."""
    rows = []
    for h in data.hospitals:
        y = np.concatenate([h.train.y, h.val.y])
        rows.append(
            {
                "hospital": h.name,
                "n_train": len(h.train),
                "n_val": len(h.val),
                "n_total": int(y.size),
                "n_malignant": int(y.sum()),
                "n_benign": int(y.size - y.sum()),
                "malignant_ratio": round(float(y.mean()), 4),
            }
        )
    return rows
