"""Tests for the PyTorch model."""

from __future__ import annotations

import numpy as np
import torch

from src.models.neural_network import MedicalMLP, build_model, count_parameters, get_parameters, set_parameters


def test_model_initialization(cfg: dict) -> None:
    model = build_model(cfg)
    assert isinstance(model, MedicalMLP)
    # 30*64+64 + 64*32+32 + 32*1+1
    assert count_parameters(model) == 4097
    assert all(p.device.type == "cpu" for p in model.parameters())


def test_forward_shape_and_probabilities(cfg: dict) -> None:
    model = build_model(cfg)
    x = torch.randn(8, 30)
    assert model(x).shape == (8, 1)
    p = model.predict_proba(x)
    assert p.shape == (8,)
    assert torch.all((p >= 0) & (p <= 1))


def test_parameter_roundtrip(cfg: dict) -> None:
    torch.manual_seed(0)
    a = build_model(cfg)
    torch.manual_seed(1)
    b = build_model(cfg)
    set_parameters(b, get_parameters(a))
    for pa, pb in zip(get_parameters(a), get_parameters(b)):
        np.testing.assert_array_equal(pa, pb)


def test_seeded_initialization_is_reproducible(cfg: dict) -> None:
    torch.manual_seed(42)
    a = get_parameters(build_model(cfg))
    torch.manual_seed(42)
    b = get_parameters(build_model(cfg))
    for x, y in zip(a, b):
        np.testing.assert_array_equal(x, y)


def test_cuda_is_not_required() -> None:
    # the project never moves tensors to CUDA; it must run when CUDA is unavailable
    import src.utils.reproducibility as r
    assert r.get_device().type == "cpu"
