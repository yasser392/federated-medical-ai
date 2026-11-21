"""Partitioning the training pool across simulated hospitals (IID and non-IID)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

import numpy as np

from src.data.dataset import RawDataset, train_val_split

logger = logging.getLogger(__name__)

PartitionMode = Literal["iid", "non_iid"]


@dataclass(frozen=True)
class HospitalData:
    """Private data owned by one hospital: a local training set and a local validation set."""

    name: str
    train: RawDataset
    val: RawDataset

    @property
    def n_train(self) -> int:
        return len(self.train)


def iid_partition(y: np.ndarray, num_clients: int, seed: int) -> list[np.ndarray]:
    """Stratified IID split: every client receives (almost) the same size and class ratio.

    Returns:
        A list of positional index arrays, one per client, that are disjoint and cover all samples.
    """
    if num_clients < 1:
        raise ValueError("num_clients must be >= 1")
    rng = np.random.default_rng(seed)
    parts: list[list[int]] = [[] for _ in range(num_clients)]
    offset = 0
    for cls in np.unique(y):
        idx = rng.permutation(np.flatnonzero(y == cls))
        for j, chunk in enumerate(np.array_split(idx, num_clients)):
            # rotate which client gets the larger chunk so sizes stay balanced
            parts[(j + offset) % num_clients].extend(chunk.tolist())
        offset += 1
    return [np.sort(np.asarray(p, dtype=np.int64)) for p in parts]


def dirichlet_partition(
    y: np.ndarray,
    num_clients: int,
    alpha: float,
    seed: int,
    min_samples_per_class: int = 5,
    max_retries: int = 1000,
) -> list[np.ndarray]:
    """Non-IID label-skew partition using a Dirichlet distribution over clients for each class.

    For each class ``c`` a proportion vector ``p_c ~ Dir(alpha)`` is sampled and the samples of ``c``
    are split across clients according to ``p_c``. A small ``alpha`` yields strongly skewed hospitals
    (e.g. a referral oncology centre with mostly malignant cases vs. a screening clinic with mostly
    benign cases) and different hospital sizes. Draws are repeated until every client has at least
    ``min_samples_per_class`` samples of every class, so each hospital can still validate locally.

    Raises:
        RuntimeError: If no valid partition is found within ``max_retries`` draws.
    """
    if alpha <= 0:
        raise ValueError("alpha must be > 0")
    classes = np.unique(y)
    rng = np.random.default_rng(seed)
    for attempt in range(max_retries):
        parts: list[list[int]] = [[] for _ in range(num_clients)]
        for cls in classes:
            idx = rng.permutation(np.flatnonzero(y == cls))
            props = rng.dirichlet(alpha * np.ones(num_clients))
            cuts = (np.cumsum(props) * len(idx)).astype(int)[:-1]
            for client, chunk in enumerate(np.split(idx, cuts)):
                parts[client].extend(chunk.tolist())
        ok = all(
            (np.asarray(p, dtype=np.int64).size > 0)
            and all(int((y[np.asarray(p, dtype=np.int64)] == c).sum()) >= min_samples_per_class for c in classes)
            for p in parts
        )
        if ok:
            logger.debug("Dirichlet partition found after %d attempt(s)", attempt + 1)
            return [np.sort(np.asarray(p, dtype=np.int64)) for p in parts]
    raise RuntimeError(
        f"Could not find a Dirichlet partition with >= {min_samples_per_class} samples per class "
        f"per client after {max_retries} attempts; increase alpha or reduce min_samples_per_class."
    )


def partition_indices(y: np.ndarray, mode: PartitionMode, cfg: dict, seed: int) -> list[np.ndarray]:
    """Dispatch to the IID or non-IID partitioner using configuration values."""
    p = cfg["partition"]
    if mode == "iid":
        return iid_partition(y, p["num_clients"], seed)
    if mode == "non_iid":
        return dirichlet_partition(
            y,
            p["num_clients"],
            alpha=p["dirichlet_alpha"],
            seed=seed,
            min_samples_per_class=p.get("min_samples_per_class", 5),
            max_retries=p.get("max_retries", 1000),
        )
    raise ValueError(f"Unknown partition mode: {mode!r} (expected 'iid' or 'non_iid')")


def build_hospitals(
    train_pool: RawDataset, mode: PartitionMode, cfg: dict, names: list[str], seed: int
) -> list[HospitalData]:
    """Partition the training pool and split each hospital's share into local train/validation."""
    parts = partition_indices(train_pool.y, mode, cfg, seed)
    hospitals = []
    for i, (name, idx) in enumerate(zip(names, parts)):
        local = train_pool.subset(idx)
        tr, va = train_val_split(local, cfg["data"]["val_fraction"], seed + i)
        hospitals.append(HospitalData(name=name, train=tr, val=va))
    return hospitals
