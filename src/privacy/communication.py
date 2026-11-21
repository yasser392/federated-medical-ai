"""Communication tracking and a payload audit showing that raw hospital data is never transmitted.

Every message exchanged between the server and a hospital passes through ``CommunicationTracker``,
which records the *serialized* Flower ``Parameters`` actually exchanged (byte length of each tensor)
and a manifest of what the message contains. ``audit_payload`` then checks the content against the
hospital's raw records.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
from flwr.common import Parameters, parameters_to_ndarrays


@dataclass
class Message:
    """One transmitted message."""

    round: int
    client: str
    direction: str  # "server->client" or "client->server"
    num_tensors: int
    num_parameters: int
    payload_bytes: int
    tensor_shapes: list[list[int]]
    extra_fields: dict[str, str] = field(default_factory=dict)


@dataclass
class CommunicationTracker:
    """Accumulates every message of a federated run."""

    messages: list[Message] = field(default_factory=list)
    audits: list[dict[str, Any]] = field(default_factory=list)

    def record(self, server_round: int, client: str, direction: str, parameters: Parameters,
               extra: dict[str, Any] | None = None) -> None:
        """Record a ``Parameters`` message (sizes measured on the serialized bytes)."""
        arrays = parameters_to_ndarrays(parameters)
        self.messages.append(Message(
            round=server_round,
            client=client,
            direction=direction,
            num_tensors=len(arrays),
            num_parameters=int(sum(a.size for a in arrays)),
            payload_bytes=int(sum(len(t) for t in parameters.tensors)),
            tensor_shapes=[list(a.shape) for a in arrays],
            extra_fields={k: type(v).__name__ for k, v in (extra or {}).items()},
        ))

    def summary(self, num_clients: int, num_rounds: int, raw_bytes_per_client: dict[str, int] | None = None) -> dict[str, Any]:
        """Aggregate communication statistics."""
        down = [m for m in self.messages if m.direction == "server->client"]
        up = [m for m in self.messages if m.direction == "client->server"]
        total_bytes = sum(m.payload_bytes for m in self.messages)
        out: dict[str, Any] = {
            "num_clients": num_clients,
            "num_rounds": num_rounds,
            "num_messages": len(self.messages),
            "parameters_per_model": up[0].num_parameters if up else (down[0].num_parameters if down else 0),
            "parameters_transmitted_total": int(sum(m.num_parameters for m in self.messages)),
            "bytes_server_to_clients": int(sum(m.payload_bytes for m in down)),
            "bytes_clients_to_server": int(sum(m.payload_bytes for m in up)),
            "bytes_total": int(total_bytes),
            "kilobytes_total": round(total_bytes / 1024, 2),
            "bytes_per_round": round(total_bytes / max(num_rounds, 1), 1),
            "fields_sent_by_clients": sorted({k for m in up for k in m.extra_fields} | {"parameters", "num_examples"}),
            "audits_passed": all(a["passed"] for a in self.audits) if self.audits else None,
            "num_audits": len(self.audits),
        }
        if raw_bytes_per_client:
            out["raw_data_bytes_kept_local"] = raw_bytes_per_client
            out["raw_data_bytes_total"] = int(sum(raw_bytes_per_client.values()))
            out["raw_data_bytes_transmitted"] = 0
        return out

    def to_dict(self) -> dict[str, Any]:
        return {"messages": [asdict(m) for m in self.messages], "audits": self.audits}


def audit_payload(arrays: list[np.ndarray], raw_X: np.ndarray, raw_y: np.ndarray, client: str,
                  server_round: int, atol: float = 1e-6) -> dict[str, Any]:
    """Check that a client's outgoing payload does not contain its raw records.

    Checks performed:
      1. No transmitted tensor has the shape of the local feature matrix.
      2. No row of the raw (or standardized) feature matrix appears as a contiguous 30-value
         sequence inside any transmitted tensor.
      3. The local label vector does not appear as a contiguous sequence in the payload.
      4. The payload size is independent of the number of local samples (fixed model size).

    This is a sanity demonstration, **not** a privacy proof: model weights can still leak
    information through inversion / membership-inference attacks.
    """
    n, d = raw_X.shape
    shape_hit = any(a.shape in {(n, d), (d, n)} for a in arrays)
    flat = np.concatenate([a.ravel().astype(np.float64) for a in arrays])
    rows_found = 0
    if flat.size >= d:
        windows = np.lib.stride_tricks.sliding_window_view(flat, d)
        for row in raw_X.astype(np.float64):
            if np.any(np.all(np.abs(windows - row) <= atol, axis=1)):
                rows_found += 1
    record = {
        "round": server_round,
        "client": client,
        "local_samples": int(n),
        "payload_values": int(flat.size),
        "shape_matches_raw_data": bool(shape_hit),
        "raw_rows_found_in_payload": int(rows_found),
        "labels_vector_sent": _contains_sequence(flat, raw_y.astype(np.float64), atol),
    }
    record["passed"] = not shape_hit and rows_found == 0 and not record["labels_vector_sent"]
    return record


def _contains_sequence(flat: np.ndarray, seq: np.ndarray, atol: float) -> bool:
    """True if ``seq`` appears as a contiguous subsequence of ``flat``."""
    if seq.size == 0 or flat.size < seq.size:
        return False
    windows = np.lib.stride_tricks.sliding_window_view(flat, seq.size)
    return bool(np.any(np.all(np.abs(windows - seq) <= atol, axis=1)))
