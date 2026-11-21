"""Flower client representing one hospital. Raw data never leaves this object."""

from __future__ import annotations

import logging

import numpy as np
from flwr.client import NumPyClient
from flwr.common import Scalar
from torch.utils.data import TensorDataset

from src.data.preprocessing import make_loader
from src.evaluation.metrics import compute_metrics
from src.models.neural_network import build_model, get_parameters, set_parameters
from src.models.training import evaluate_loss, make_optimizer, predict_proba, train_one_epoch

logger = logging.getLogger(__name__)


class HospitalClient(NumPyClient):
    """A hospital participating in federated training.

    It owns a private local training set and validation set (already standardized with the
    federated scaler) and a local PyTorch model. It only ever returns model parameters, the number
    of local training examples and a few scalar metrics.
    """

    def __init__(self, name: str, train_ds: TensorDataset, val_ds: TensorDataset, cfg: dict, seed: int) -> None:
        self.name = name
        self._train_ds = train_ds  # private
        self._val_ds = val_ds  # private
        self.cfg = cfg
        self.seed = seed
        self.model = build_model(cfg)

    @property
    def num_train(self) -> int:
        return len(self._train_ds)

    @property
    def num_val(self) -> int:
        return len(self._val_ds)

    def get_parameters(self, config: dict[str, Scalar]) -> list[np.ndarray]:
        """Return the current local model parameters."""
        return get_parameters(self.model)

    def fit(self, parameters: list[np.ndarray], config: dict[str, Scalar]) -> tuple[list[np.ndarray], int, dict[str, Scalar]]:
        """Load the global model, train it locally for ``local_epochs`` and return the new weights."""
        set_parameters(self.model, parameters)
        server_round = int(config.get("server_round", 0))
        local_epochs = int(config.get("local_epochs", self.cfg["federated"]["local_epochs"]))
        optimizer = make_optimizer(self.model, self.cfg)  # fresh optimizer state each round
        losses = []
        for epoch in range(local_epochs):
            loader = make_loader(self._train_ds, self.cfg["training"]["batch_size"], shuffle=True,
                                 seed=self.seed + 1000 * server_round + epoch)
            losses.append(train_one_epoch(self.model, loader, optimizer))
        logger.debug("%s round %d: local train loss %.4f", self.name, server_round, losses[-1])
        return get_parameters(self.model), self.num_train, {"train_loss": float(losses[-1]), "client": self.name}

    def evaluate(self, parameters: list[np.ndarray], config: dict[str, Scalar]) -> tuple[float, int, dict[str, Scalar]]:
        """Evaluate the global model on the private local validation set."""
        set_parameters(self.model, parameters)
        loss = evaluate_loss(self.model, self._val_ds)
        m = compute_metrics(*predict_proba(self.model, self._val_ds), self.cfg["training"]["threshold"])
        return float(loss), self.num_val, {"accuracy": m["accuracy"], "client": self.name}
