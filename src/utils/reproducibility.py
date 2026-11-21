"""Reproducibility and CPU-only device helpers."""

from __future__ import annotations

import os
import random

import numpy as np
import torch

# Hide any GPU from PyTorch before it is used anywhere in the project.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch RNGs and enable deterministic algorithms."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def get_device(num_threads: int | None = None) -> torch.device:
    """Return the CPU device (CUDA is never used) and optionally limit torch threads."""
    if num_threads is not None and num_threads > 0:
        torch.set_num_threads(int(num_threads))
    return torch.device("cpu")


def make_generator(seed: int) -> torch.Generator:
    """Create a seeded torch generator (used for DataLoader shuffling)."""
    g = torch.Generator()
    g.manual_seed(seed)
    return g
