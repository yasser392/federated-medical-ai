"""Federated server: Flower ``FedAvg`` strategy driven in-process (no network, no Ray).

Design note
-----------
Flower 1.44's ``start_simulation`` relies on Ray (not installed, heavy for a laptop) and its newer
``run_simulation`` requires a packaged ServerApp/ClientApp project. To stay lightweight while still
using Flower's real components, this module drives Flower's own objects directly:

* ``flwr.server.strategy.FedAvg``: ``configure_fit``, ``aggregate_fit``, ``configure_evaluate``,
  ``aggregate_evaluate`` and ``evaluate`` (server-side evaluation).
* ``flwr.server.client_manager.SimpleClientManager`` for client registration and sampling.
* ``flwr.client.NumPyClient`` hospitals wrapped as ``flwr.client.Client`` via ``to_client()``.
* ``flwr.common.Parameters``: every message is serialized exactly as Flower would send it.

Only the network transport is replaced by in-process function calls (``InProcessClientProxy``).
"""

from __future__ import annotations

import logging
from typing import Any, Callable

import numpy as np
from flwr.client import Client
from flwr.common import (
    Code,
    DisconnectRes,
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    GetParametersIns,
    GetParametersRes,
    GetPropertiesIns,
    GetPropertiesRes,
    Parameters,
    ReconnectIns,
    Status,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_manager import SimpleClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

from src.federated.aggregation import fedavg, weighted_average
from src.privacy.communication import CommunicationTracker

logger = logging.getLogger(__name__)

EvaluateFn = Callable[[int, list[np.ndarray], dict], tuple[float, dict[str, Any]] | None]


class InProcessClientProxy(ClientProxy):
    """A ``ClientProxy`` that forwards Flower messages to a local ``Client`` by function call."""

    def __init__(self, cid: str, client: Client) -> None:
        super().__init__(cid)
        self._client = client

    def get_properties(self, ins: GetPropertiesIns, timeout: float | None, group_id: int | None) -> GetPropertiesRes:
        return self._client.get_properties(ins)

    def get_parameters(self, ins: GetParametersIns, timeout: float | None, group_id: int | None) -> GetParametersRes:
        return self._client.get_parameters(ins)

    def fit(self, ins: FitIns, timeout: float | None, group_id: int | None) -> FitRes:
        return self._client.fit(ins)

    def evaluate(self, ins: EvaluateIns, timeout: float | None, group_id: int | None) -> EvaluateRes:
        return self._client.evaluate(ins)

    def reconnect(self, ins: ReconnectIns, timeout: float | None, group_id: int | None) -> DisconnectRes:
        return DisconnectRes(reason="in-process")


def _weighted_metric(key: str) -> Callable[[list[tuple[int, dict]]], dict]:
    def agg(metrics: list[tuple[int, dict]]) -> dict:
        return {key: weighted_average([(n, float(m[key])) for n, m in metrics if key in m])}
    return agg


class FederatedServer:
    """Central server orchestrating FedAvg rounds over registered hospital clients."""

    def __init__(
        self,
        initial_parameters: list[np.ndarray],
        clients: dict[str, Client],
        cfg: dict,
        evaluate_fn: EvaluateFn | None = None,
        tracker: CommunicationTracker | None = None,
        audit_fn: Callable[[str, int, list[np.ndarray]], None] | None = None,
    ) -> None:
        fed = cfg["federated"]
        n = len(clients)
        self.backend: str = fed.get("aggregation_backend", "flower")
        self.local_epochs: int = int(fed["local_epochs"])
        self.tracker = tracker or CommunicationTracker()
        self.audit_fn = audit_fn
        self.parameters: Parameters = ndarrays_to_parameters(initial_parameters)
        self.strategy = FedAvg(
            fraction_fit=fed.get("fraction_fit", 1.0),
            fraction_evaluate=1.0,
            min_fit_clients=max(1, int(round(fed.get("fraction_fit", 1.0) * n))),
            min_evaluate_clients=n,
            min_available_clients=n,
            evaluate_fn=evaluate_fn,
            on_fit_config_fn=lambda rnd: {"server_round": rnd, "local_epochs": self.local_epochs},
            on_evaluate_config_fn=lambda rnd: {"server_round": rnd},
            fit_metrics_aggregation_fn=_weighted_metric("train_loss"),
            evaluate_metrics_aggregation_fn=_weighted_metric("accuracy"),
            initial_parameters=self.parameters,
            accept_failures=False,
            inplace=False,
        )
        self.client_manager = SimpleClientManager()
        for cid, client in clients.items():
            self.client_manager.register(InProcessClientProxy(cid, client))

    @property
    def global_ndarrays(self) -> list[np.ndarray]:
        return parameters_to_ndarrays(self.parameters)

    def run_round(self, server_round: int) -> dict[str, Any]:
        """Execute one communication round: distribute, train locally, aggregate, evaluate."""
        # 1-2. select clients and distribute the current global parameters
        instructions = self.strategy.configure_fit(server_round, self.parameters, self.client_manager)
        results: list[tuple[ClientProxy, FitRes]] = []
        for proxy, ins in instructions:
            self.tracker.record(server_round, proxy.cid, "server->client", ins.parameters)
            res = proxy.fit(ins, timeout=None, group_id=server_round)
            if res.status.code != Code.OK:
                raise RuntimeError(f"Client {proxy.cid} failed in round {server_round}: {res.status.message}")
            # 3. receive client parameters (and audit the payload content)
            self.tracker.record(server_round, proxy.cid, "client->server", res.parameters, res.metrics)
            if self.audit_fn is not None:
                self.audit_fn(proxy.cid, server_round, parameters_to_ndarrays(res.parameters))
            results.append((proxy, res))

        # 4-5. sample-weighted FedAvg and update of the global model
        if self.backend == "native":
            new = fedavg([(parameters_to_ndarrays(r.parameters), r.num_examples) for _, r in results])
            fit_metrics = _weighted_metric("train_loss")([(r.num_examples, r.metrics) for _, r in results])
            self.parameters = ndarrays_to_parameters(new)
        else:
            aggregated, fit_metrics = self.strategy.aggregate_fit(server_round, results, [])
            if aggregated is None:
                raise RuntimeError(f"Aggregation failed in round {server_round}")
            self.parameters = aggregated

        # federated evaluation of the new global model on each hospital's private validation set
        eval_ins = self.strategy.configure_evaluate(server_round, self.parameters, self.client_manager)
        eval_results = []
        for proxy, ins in eval_ins:
            self.tracker.record(server_round, proxy.cid, "server->client", ins.parameters)
            eval_results.append((proxy, proxy.evaluate(ins, timeout=None, group_id=server_round)))
        val_loss, val_metrics = self.strategy.aggregate_evaluate(server_round, eval_results, [])

        # server-side evaluation on the held-out test set (reporting only; never used for training)
        server_eval = self.strategy.evaluate(server_round, self.parameters)
        test_loss, test_metrics = server_eval if server_eval is not None else (float("nan"), {})

        record = {
            "round": server_round,
            "train_loss": float(fit_metrics.get("train_loss", float("nan"))),
            "val_loss": float(val_loss) if val_loss is not None else float("nan"),
            "val_accuracy": float(val_metrics.get("accuracy", float("nan"))),
            "test_loss": float(test_loss),
            **{f"test_{k}": float(v) for k, v in test_metrics.items() if isinstance(v, (int, float))},
            "client_train_loss": {p.cid: float(r.metrics["train_loss"]) for p, r in results},
            "client_num_examples": {p.cid: int(r.num_examples) for p, r in results},
            "client_val_loss": {p.cid: float(r.loss) for p, r in eval_results},
        }
        logger.info("Round %d | train_loss=%.4f val_loss=%.4f test_acc=%.4f test_auc=%.4f", server_round,
                    record["train_loss"], record["val_loss"], record.get("test_accuracy", float("nan")),
                    record.get("test_roc_auc", float("nan")))
        return record
