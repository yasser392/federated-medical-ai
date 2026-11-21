"""Loading of the Breast Cancer Wisconsin (Diagnostic) dataset and the global train/test split."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class RawDataset:
    """Unscaled dataset with stable sample indices.

    Attributes:
        X: Feature matrix, shape (n_samples, 30).
        y: Binary labels. Note: sklearn encodes 0 = malignant, 1 = benign. We re-encode so that
            1 = malignant (the positive / clinically critical class) and 0 = benign.
        feature_names: The 30 feature names.
        indices: Original row indices into the full dataset (used to prove disjointness).
    """

    X: np.ndarray
    y: np.ndarray
    feature_names: list[str]
    indices: np.ndarray

    def __len__(self) -> int:
        return int(self.y.shape[0])

    def subset(self, idx: np.ndarray) -> "RawDataset":
        """Return a subset by positional index."""
        return RawDataset(self.X[idx], self.y[idx], self.feature_names, self.indices[idx])


def load_dataset() -> RawDataset:
    """Load the full dataset (569 samples, 30 features) with malignant = 1."""
    bunch = load_breast_cancer()
    X = bunch.data.astype(np.float32)
    y = (bunch.target == 0).astype(np.int64)  # malignant -> 1
    return RawDataset(X=X, y=y, feature_names=list(bunch.feature_names), indices=np.arange(len(y)))


def train_test_split_dataset(ds: RawDataset, test_size: float, seed: int) -> tuple[RawDataset, RawDataset]:
    """Stratified split into a training pool and an independent, held-out test set.

    The test set is never used for training, validation, early stopping or scaling.
    """
    pos = np.arange(len(ds))
    tr, te = train_test_split(pos, test_size=test_size, stratify=ds.y, random_state=seed)
    return ds.subset(np.sort(tr)), ds.subset(np.sort(te))


def train_val_split(ds: RawDataset, val_fraction: float, seed: int) -> tuple[RawDataset, RawDataset]:
    """Split a dataset into train/validation, stratified when every class has >= 2 samples."""
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be in (0, 1)")
    pos = np.arange(len(ds))
    counts = np.bincount(ds.y, minlength=2)
    stratify = ds.y if counts.min() >= 2 else None
    tr, va = train_test_split(pos, test_size=val_fraction, stratify=stratify, random_state=seed)
    return ds.subset(np.sort(tr)), ds.subset(np.sort(va))


def dataset_summary(ds: RawDataset) -> dict[str, int | float]:
    """Return sample counts and class balance."""
    n = len(ds)
    malignant = int(ds.y.sum())
    return {
        "n_samples": n,
        "n_malignant": malignant,
        "n_benign": n - malignant,
        "malignant_ratio": round(malignant / n, 4) if n else 0.0,
    }
