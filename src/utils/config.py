"""Configuration loading and project path helpers."""

from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH: Path = PROJECT_ROOT / "config.yaml"

_REQUIRED_SECTIONS = ("seed", "data", "partition", "model", "training", "federated", "paths")


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load the YAML configuration file.

    Args:
        path: Optional path to a YAML file. Defaults to ``config.yaml`` at the project root.

    Returns:
        The parsed configuration dictionary.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
        ValueError: If required sections are missing or values are inconsistent.
    """
    cfg_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    if not isinstance(cfg, dict):
        raise ValueError(f"Configuration file {cfg_path} is empty or malformed")
    missing = [s for s in _REQUIRED_SECTIONS if s not in cfg]
    if missing:
        raise ValueError(f"Configuration is missing required sections: {missing}")
    _validate(cfg)
    return cfg


def _validate(cfg: dict[str, Any]) -> None:
    """Basic sanity checks on configuration values."""
    part = cfg["partition"]
    if part["num_clients"] < 1:
        raise ValueError("partition.num_clients must be >= 1")
    names = part.get("client_names") or []
    if names and len(names) != part["num_clients"]:
        raise ValueError("partition.client_names must have num_clients entries")
    if not 0.0 < cfg["data"]["test_size"] < 1.0:
        raise ValueError("data.test_size must be in (0, 1)")
    if cfg["federated"]["num_rounds"] < 1:
        raise ValueError("federated.num_rounds must be >= 1")
    if cfg["federated"].get("aggregation_backend", "flower") not in ("flower", "native"):
        raise ValueError("federated.aggregation_backend must be 'flower' or 'native'")


def client_names(cfg: dict[str, Any]) -> list[str]:
    """Return the display names of the simulated hospitals."""
    n = cfg["partition"]["num_clients"]
    names = cfg["partition"].get("client_names") or []
    if len(names) == n:
        return list(names)
    return [f"Hospital {chr(ord('A') + i)}" if i < 26 else f"Hospital {i}" for i in range(n)]


def override(cfg: dict[str, Any], **updates: Any) -> dict[str, Any]:
    """Return a deep copy of ``cfg`` with dotted-key overrides, e.g. ``{"federated.num_rounds": 10}``."""
    new = copy.deepcopy(cfg)
    for dotted, value in updates.items():
        keys = dotted.split(".")
        node = new
        for k in keys[:-1]:
            node = node.setdefault(k, {})
        node[keys[-1]] = value
    _validate(new)
    return new


def resolve_path(cfg: dict[str, Any], key: str) -> Path:
    """Resolve ``cfg['paths'][key]`` relative to the project root and create it."""
    p = PROJECT_ROOT / cfg["paths"][key]
    p.mkdir(parents=True, exist_ok=True)
    return p


def setup_logging(cfg: dict[str, Any] | None = None, name: str = "run", level: int = logging.INFO) -> logging.Logger:
    """Configure root logging to the console and to ``results/logs/<name>.log``."""
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if cfg is not None:
        log_dir = resolve_path(cfg, "logs_dir")
        handlers.append(logging.FileHandler(log_dir / f"{name}.log", mode="w", encoding="utf-8"))
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        handlers=handlers,
        force=True,
    )
    # Flower is chatty; keep its logs at WARNING.
    logging.getLogger("flwr").setLevel(logging.WARNING)
    return logging.getLogger(name)
