"""
Digital-twin core.

The twin maintains a *synchronised, observation-only* replica of the physical
network. It never reads the latent health of the environment; it consumes the
same telemetry vector that a real OAM/NWDAF interface would expose, adds the
model-derived risk score r_n(t) = P(failure within H | telemetry window), and
exposes the derived quantities used by the orchestrator:

    * per-node operational state (observed, with detection delay)
    * per-node and per-link residual capacity
    * a time-varying cost graph  (Eq. 12)
    * path KPI evaluation        (Eq. 13-15)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import networkx as nx

from ..simulator.config import W_LATENCY, W_LOAD, W_LOSS, W_RISK

LATENCY_REF_MS = 25.0
OBSERVED_FAIL_LOSS = 0.50
OBSERVED_FAIL_AVAIL = 0.08


@dataclass
class TwinNode:
    node_id: str
    node_type: str
    capacity_mbps: float
    telemetry: Dict[str, float] = field(default_factory=dict)
    risk: float = 0.0
    observed_up: bool = True
    load_mbps: float = 0.0
    consecutive_bad: int = 0
    latency_baseline: float = 0.0

    @property
    def utilisation(self) -> float:
        return min(0.985, self.load_mbps / max(1e-6, self.capacity_mbps))


class DigitalTwin:
    """Observation-driven replica of the hybrid network."""

    def __init__(self, graph: nx.Graph, detection_delay: int = 1) -> None:
        self.graph = graph
        self.detection_delay = detection_delay
        self.nodes: Dict[str, TwinNode] = {
            nid: TwinNode(nid, data["node_type"], data["capacity_mbps"])
            for nid, data in graph.nodes(data=True)
        }
        self.link_load: Dict[Tuple[str, str], float] = {}
        # Bandwidth reserved for standby paths (reserved 1:1 protection only).
        self.link_reserved: Dict[Tuple[str, str], float] = {}
        self.node_reserved: Dict[str, float] = {}
        self.t = 0
        self._graph_cache: Dict[Tuple, nx.Graph] = {}
        self._path_cache: Dict[Tuple, Optional[List[str]]] = {}

    # ------------------------------------------------------------- ingestion
    def ingest(self, snapshot: Dict[str, Dict[str, float]]) -> None:
        """Consume one telemetry snapshot (no access to latent state)."""
        for nid, obs in snapshot.items():
            node = self.nodes[nid]
            node.telemetry = {k: v for k, v in obs.items() if not k.startswith("_")}
            bad = (
                obs["packet_loss"] >= OBSERVED_FAIL_LOSS
                and obs["availability"] <= OBSERVED_FAIL_AVAIL
            )
            node.consecutive_bad = node.consecutive_bad + 1 if bad else 0
            node.observed_up = node.consecutive_bad < self.detection_delay + 1
            latency = float(obs.get("latency_ms", 0.0))
            if node.latency_baseline <= 0.0:
                node.latency_baseline = latency
            elif not bad:
                node.latency_baseline = 0.97 * node.latency_baseline + 0.03 * latency
        self.t += 1

    def set_risk(self, risk: Dict[str, float]) -> None:
        for nid, value in risk.items():
            if nid in self.nodes:
                self.nodes[nid].risk = float(value)
        self.invalidate()

    def baseline_latency(self, nid: str) -> float:
        """Slowly adapting latency reference used by the non-learning alarm rule."""
        node = self.nodes[nid]
        return node.latency_baseline or node.telemetry.get("latency_ms", 1.0)

    def invalidate(self) -> None:
        """Drop the per-slot cost-graph and path caches."""
        self._graph_cache.clear()
        self._path_cache.clear()

    def reset_load(self) -> None:
        for node in self.nodes.values():
            node.load_mbps = 0.0
        self.link_load = {}
        self.link_reserved = {}
        self.node_reserved = {}

    def reserve(self, path: Sequence[str], demand: float) -> None:
        for nid in path:
            self.node_reserved[nid] = self.node_reserved.get(nid, 0.0) + demand
        for u, v in zip(path[:-1], path[1:]):
            key = (u, v) if u <= v else (v, u)
            self.link_reserved[key] = self.link_reserved.get(key, 0.0) + demand

    def move_flow_load(self, old: Sequence[str], new: Sequence[str], demand: float) -> None:
        """Keep the capacity ledger exact within a slot when a flow changes path."""
        if old:
            self.add_flow_load(old, -demand)
        if new:
            self.add_flow_load(new, demand)
        self.invalidate()

    def add_flow_load(self, path: Sequence[str], demand: float) -> None:
        for nid in path:
            self.nodes[nid].load_mbps += demand
        for u, v in zip(path[:-1], path[1:]):
            key = (u, v) if u <= v else (v, u)
            self.link_load[key] = self.link_load.get(key, 0.0) + demand

    def load_map(self) -> Dict[str, float]:
        return {nid: node.utilisation for nid, node in self.nodes.items()}

    # ------------------------------------------------------------ cost model
    def node_penalty(self, nid: str, risk_aware: bool) -> float:
        node = self.nodes[nid]
        tel = node.telemetry
        latency = tel.get("latency_ms", 10.0)
        loss = tel.get("packet_loss", 0.01)
        cost = W_LATENCY * (latency / LATENCY_REF_MS) + W_LOSS * loss
        cost += W_LOAD * max(0.0, node.utilisation - 0.75) ** 2
        if risk_aware:
            cost += W_RISK * node.risk ** 2
        return cost

    def cost_graph(
        self,
        risk_aware: bool,
        excluded: Optional[Set[str]] = None,
        demand: float = 0.0,
    ) -> nx.Graph:
        """Time-varying weighted graph used for constrained shortest paths."""
        excluded = frozenset(excluded or ())
        key = (risk_aware, excluded, round(demand, 2))
        cached = self._graph_cache.get(key)
        if cached is not None:
            return cached
        g = nx.Graph()
        for nid, node in self.nodes.items():
            if nid in excluded or not node.observed_up:
                continue
            g.add_node(nid)
        for u, v, data in self.graph.edges(data=True):
            if u not in g or v not in g:
                continue
            key = (u, v) if u <= v else (v, u)
            residual = (data["capacity_mbps"] - self.link_load.get(key, 0.0)
                        - self.link_reserved.get(key, 0.0))
            if demand > 0.0 and residual < demand:
                continue
            weight = (
                W_LATENCY * (data["prop_latency_ms"] / LATENCY_REF_MS)
                + 0.5 * (self.node_penalty(u, risk_aware) + self.node_penalty(v, risk_aware))
            )
            g.add_edge(u, v, weight=weight)
        self._graph_cache[key] = g
        return g

    def shortest_path(
        self,
        source: str,
        target: str,
        risk_aware: bool,
        excluded: Optional[Set[str]] = None,
        demand: float = 0.0,
    ) -> Optional[List[str]]:
        key = ("sp", source, target, risk_aware, frozenset(excluded or ()), round(demand, 2))
        if key in self._path_cache:
            return self._path_cache[key]
        g = self.cost_graph(risk_aware, excluded, demand)
        result: Optional[List[str]]
        if source not in g or target not in g:
            result = None
        else:
            try:
                result = nx.shortest_path(g, source, target, weight="weight")
            except nx.NetworkXNoPath:
                result = None
        self._path_cache[key] = result
        return result

    def k_shortest_paths(
        self, source: str, target: str, risk_aware: bool, k: int = 3
    ) -> List[List[str]]:
        key = ("ksp", source, target, risk_aware, k)
        if key in self._path_cache:
            return self._path_cache[key]  # type: ignore[return-value]
        g = self.cost_graph(risk_aware)
        paths: List[List[str]] = []
        if source in g and target in g:
            try:
                gen = nx.shortest_simple_paths(g, source, target, weight="weight")
                for _ in range(k):
                    paths.append(next(gen))
            except (nx.NetworkXNoPath, StopIteration):
                pass
        self._path_cache[key] = paths  # type: ignore[assignment]
        return paths

    # --------------------------------------------------------- path analysis
    def path_risk(self, path: Sequence[str]) -> float:
        """R(path) = 1 - prod(1 - r_n)  (Eq. 16)."""
        product = 1.0
        for nid in path:
            product *= 1.0 - min(0.999, self.nodes[nid].risk)
        return 1.0 - product

    def path_risk_max(self, path: Sequence[str]) -> float:
        """R_inf(path) = max_n r_n: the bottleneck-risk aggregation used to
        trigger pre-emptive migration (Eq. 17)."""
        return max((self.nodes[nid].risk for nid in path), default=0.0)

    def path_predicted_kpi(self, path: Sequence[str]) -> Tuple[float, float]:
        latency = sum(self.nodes[n].telemetry.get("latency_ms", 10.0) for n in path)
        for u, v in zip(path[:-1], path[1:]):
            latency += self.graph[u][v]["prop_latency_ms"]
        keep = 1.0
        for nid in path:
            keep *= 1.0 - min(1.0, self.nodes[nid].telemetry.get("packet_loss", 0.0))
        return latency, 1.0 - keep

    def residual_capacity(self, path: Sequence[str], include_reserved: bool = True) -> float:
        margins = []
        for u, v in zip(path[:-1], path[1:]):
            key = (u, v) if u <= v else (v, u)
            margins.append(self.graph[u][v]["capacity_mbps"] - self.link_load.get(key, 0.0)
                           - (self.link_reserved.get(key, 0.0) if include_reserved else 0.0))
        for nid in path:
            node = self.nodes[nid]
            margins.append(node.capacity_mbps - node.load_mbps
                           - (self.node_reserved.get(nid, 0.0) if include_reserved else 0.0))
        return min(margins) if margins else 0.0

    def state_vector(self) -> Dict[str, Dict[str, float]]:
        return {
            nid: {
                "node_type": node.node_type,
                "risk": node.risk,
                "observed_up": node.observed_up,
                "utilisation": node.utilisation,
                **node.telemetry,
            }
            for nid, node in self.nodes.items()
        }
