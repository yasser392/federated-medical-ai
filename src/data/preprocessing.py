"""Feature standardization fitted on training data only (centralized or federated)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.data.dataset import RawDataset
from src.utils.reproducibility import make_generator

_EPS = 1e-8


@dataclass(frozen=True)
class Standardizer:
    """Z-score standardizer: ``(x - mean) / std``."""

    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, X: np.ndarray) -> "Standardizer":
        """Fit on a (training) feature matrix. Uses the population std, like sklearn's StandardScaler."""
        if X.ndim != 2 or X.shape[0] == 0:
            raise ValueError("Standardizer.fit expects a non-empty 2-D array")
        mean = X.mean(axis=0, dtype=np.float64)
        std = X.std(axis=0, dtype=np.float64)
        return cls(mean=mean, std=np.where(std < _EPS, 1.0, std))

    @classmethod
    def fit_federated(cls, client_X: list[np.ndarray]) -> "Standardizer":
        """Fit from per-client sufficient statistics only (n, sum, sum of squares).

        Each hospital shares three aggregate vectors instead of raw rows. The result is exactly
        equal to fitting on the union of all client training sets.
        """
        stats = [local_sufficient_statistics(X) for X in client_X]
        n = sum(s[0] for s in stats)
        if n == 0:
            raise ValueError("No samples provided to fit_federated")
        total = np.sum([s[1] for s in stats], axis=0)
        total_sq = np.sum([s[2] for s in stats], axis=0)
        mean = total / n
        var = np.maximum(total_sq / n - mean**2, 0.0)
        std = np.sqrt(var)
        return cls(mean=mean, std=np.where(std < _EPS, 1.0, std))

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply standardization."""
        return ((X - self.mean) / self.std).astype(np.float32)


def local_sufficient_statistics(X: np.ndarray) -> tuple[int, np.ndarray, np.ndarray]:
    """Return (n, sum, sum of squares) of a local feature matrix (computed in float64)."""
    X64 = X.astype(np.float64)
    return int(X64.shape[0]), X64.sum(axis=0), (X64**2).sum(axis=0)


def to_tensor_dataset(ds: RawDataset, scaler: Standardizer) -> TensorDataset:
    """Scale a raw dataset and wrap it in a ``TensorDataset`` (features float32, labels float32)."""
    X = torch.from_numpy(scaler.transform(ds.X))
    y = torch.from_numpy(ds.y.astype(np.float32))
    return TensorDataset(X, y)


def make_loader(ds: TensorDataset, batch_size: int, shuffle: bool, seed: int = 0) -> DataLoader:
    """Create a CPU DataLoader with seeded shuffling and no worker processes (low RAM)."""
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=make_generator(seed) if shuffle else None,
        num_workers=0,
    )
