"""
On-line feature construction.

The experiment runner and the FastAPI inference service must produce exactly
the same feature vector as `simulator.dataset_generator.add_temporal_features`,
otherwise the model is served out of distribution. This module implements the
streaming counterpart of that batch transformation using a per-node ring buffer.
"""

from __future__ import annotations

import json
import os
from collections import deque
from typing import Deque, Dict, List, Optional

import numpy as np

from ..simulator.config import FEATURE_WINDOW
from ..simulator.dataset_generator import ROLLING_FIELDS
from ..simulator.failure_generator import TELEMETRY_FIELDS

RATIO_FIELDS = ["latency_ms", "packet_loss", "cpu_usage"]
_MEDIAN_PATH = os.path.join("data", "results", "node_type_latency_median.json")


def load_latency_medians(path: str = _MEDIAN_PATH) -> Dict[str, float]:
    if os.path.exists(path):
        with open(path) as fh:
            return json.load(fh)
    return {}


class OnlineFeatureBuilder:
    """Streaming replica of the batch temporal feature transformation."""

    def __init__(
        self,
        window: int = FEATURE_WINDOW,
        latency_medians: Optional[Dict[str, float]] = None,
    ) -> None:
        self.window = window
        self.buffers: Dict[str, Deque[Dict[str, float]]] = {}
        self.ramp = np.arange(window) - (window - 1) / 2.0
        self.denom = float((self.ramp ** 2).sum())
        self.latency_medians = latency_medians or load_latency_medians()

    def reset(self) -> None:
        self.buffers.clear()

    def push(self, node_id: str, telemetry: Dict[str, float]) -> None:
        buf = self.buffers.setdefault(node_id, deque(maxlen=self.window))
        buf.append({k: float(telemetry[k]) for k in TELEMETRY_FIELDS})

    def features(self, node_id: str, node_type: str) -> Dict[str, float]:
        buf = self.buffers[node_id]
        current = buf[-1]
        row: Dict[str, float] = dict(current)
        row["node_type"] = node_type

        for field in ROLLING_FIELDS:
            series = np.array([obs[field] for obs in buf], dtype=float)
            n = len(series)
            row[f"{field}_mean{self.window}"] = float(series.mean()) if n >= 2 else np.nan
            row[f"{field}_std{self.window}"] = float(series.std(ddof=1)) if n >= 2 else 0.0
            if n == self.window:
                centred = series - series.mean()
                row[f"{field}_slope{self.window}"] = float(np.dot(centred, self.ramp) / self.denom)
            else:
                row[f"{field}_slope{self.window}"] = 0.0
            row[f"{field}_delta"] = float(series[-1] - series[-2]) if n >= 2 else 0.0

        for field in RATIO_FIELDS:
            base = row[f"{field}_mean{self.window}"]
            if base is None or (isinstance(base, float) and (np.isnan(base) or base == 0.0)):
                row[f"{field}_ratio"] = 1.0
            else:
                row[f"{field}_ratio"] = float(min(20.0, max(0.0, current[field] / base)))

        median_latency = self.latency_medians.get(node_type, max(1e-6, current["latency_ms"]))
        row["degradation_index"] = float(
            0.35 * min(1.0, current["packet_loss"])
            + 0.25 * (current["latency_ms"] / median_latency)
            + 0.20 * (1.0 - current["availability"])
            + 0.20 * current["retx_rate"]
        )
        return row

    def batch(self, node_types: Dict[str, str]) -> List[Dict[str, float]]:
        return [self.features(nid, ntype) for nid, ntype in node_types.items() if nid in self.buffers]
