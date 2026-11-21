"""Lightweight multilayer perceptron for binary diagnosis, plus parameter (de)serialization helpers."""

from __future__ import annotations

from collections import OrderedDict
from typing import Sequence

import numpy as np
import torch
from torch import nn


class MedicalMLP(nn.Module):
    """MLP: Linear(30,64) -> ReLU -> Dropout(0.2) -> Linear(64,32) -> ReLU -> Linear(32,1).

    ``forward`` returns raw logits (trained with ``BCEWithLogitsLoss``, which is binary cross
    entropy applied to the sigmoid output, computed in a numerically stable way).
    ``predict_proba`` returns sigmoid probabilities of the malignant class.
    """

    def __init__(self, input_dim: int = 30, hidden_dims: Sequence[int] = (64, 32), dropout: float = 0.2) -> None:
        super().__init__()
        if len(hidden_dims) != 2:
            raise ValueError("hidden_dims must contain exactly two layer sizes")
        h1, h2 = hidden_dims
        self.net = nn.Sequential(
            nn.Linear(input_dim, h1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return logits of shape (batch, 1)."""
        return self.net(x)

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return malignant-class probabilities of shape (batch,)."""
        self.eval()
        return torch.sigmoid(self.forward(x)).squeeze(-1)


def build_model(cfg: dict) -> MedicalMLP:
    """Instantiate the model from the ``model`` section of the configuration (on CPU)."""
    m = cfg["model"]
    return MedicalMLP(m["input_dim"], tuple(m["hidden_dims"]), m["dropout"]).to("cpu")


def get_parameters(model: nn.Module) -> list[np.ndarray]:
    """Return model weights as a list of NumPy arrays (state_dict order)."""
    return [v.detach().cpu().numpy().copy() for v in model.state_dict().values()]


def set_parameters(model: nn.Module, parameters: list[np.ndarray]) -> None:
    """Load a list of NumPy arrays into the model (state_dict order)."""
    keys = list(model.state_dict().keys())
    if len(keys) != len(parameters):
        raise ValueError(f"Expected {len(keys)} parameter arrays, got {len(parameters)}")
    state = OrderedDict((k, torch.as_tensor(np.asarray(v))) for k, v in zip(keys, parameters))
    model.load_state_dict(state, strict=True)


def count_parameters(model: nn.Module) -> int:
    """Number of trainable scalar parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
