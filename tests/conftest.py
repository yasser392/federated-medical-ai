"""Shared pytest fixtures."""

from __future__ import annotations

import pytest

from src.utils.config import load_config, override


@pytest.fixture(scope="session")
def cfg() -> dict:
    """Project configuration."""
    return load_config()


@pytest.fixture(scope="session")
def fast_cfg(cfg: dict) -> dict:
    """A configuration with fewer epochs/rounds so training tests run in seconds."""
    return override(cfg, **{"federated.num_rounds": 2, "federated.local_epochs": 1,
                            "centralized.epochs": 2, "local_only.epochs": 2})
