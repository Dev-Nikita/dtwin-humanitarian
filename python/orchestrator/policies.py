"""
Service orchestration policies.

Four policies are evaluated in the paper:

    static          : shortest-latency path chosen at admission, never revised.
                      Represents a statically engineered emergency network.
    reactive        : industry-standard restoration - the path is recomputed
                      only after the twin *observes* that a node on it is down.
    proactive_ai    : the proposed mechanism - risk-augmented path cost,
                      service-differentiated risk budgets and pre-emptive
                      migration of flows whose path risk exceeds the budget.
    oracle          : same machinery, but the risk score is replaced by the
                      ground-truth future failure indicator. It bounds the
                      achievable gain and quantifies how much of that bound the
                      learned predictor recovers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set

import numpy as np

from ..simulator.config import (
    HYSTERESIS_STEPS,
    PROACTIVE_RECONFIG_STEPS,
    REACTIVE_RECONFIG_STEPS,
    RISK_EXCLUDE,
)
from ..simulator.traffic_generator import Flow
from .digital_twin import DigitalTwin

# Service-differentiated risk budget tau_c (Eq. 17). The budgets are expressed
# as multiples of the operating threshold theta* selected on the validation
# episodes, so that the control loop inherits the statistical operating point of
# the predictor instead of using an arbitrary constant. Life-critical classes get
# a fraction of theta* (they migrate on weaker evidence); delay-tolerant classes
# tolerate several times theta* before paying the cost of a migration.
RISK_BUDGET_FACTOR: Dict[int, float] = {5: 0.80, 4: 1.00, 3: 1.30, 2: 1.80, 1: 2.50}
EXCLUDE_FACTOR: float = 2.50
DEFAULT_THETA: float = 0.22

# Backwards-compatible absolute budgets (used by the sensitivity sweep).
RISK_BUDGET: Dict[int, float] = {
    p: min(0.95, f * DEFAULT_THETA) for p, f in RISK_BUDGET_FACTOR.items()
}

POLICIES = ("static", "reactive", "proactive_threshold", "degradation_aware",
            "proactive_ai", "oracle")

# Non-learning proactive baseline (P4). The trigger is the same family of static
# KPI limits that production alarm systems use, but applied *before* the element
# fails rather than after. Comparing it with the learned policy isolates the
# incremental value of the predictor from the value of acting early at all.
KPI_ALARM = {
    "packet_loss": 0.12,
    "latency_ratio": 1.9,
    "cpu_usage": 88.0,
    "availability": 0.80,
}


@dataclass
class Decision:
    flow_id: int
    step: int
    kind: str            # admit | reactive_reroute | proactive_reroute | drop
    path: List[str]
    path_risk: float
    reason: str


class Orchestrator:
    """Risk-aware, SLA-constrained service orchestrator."""

    def __init__(
        self,
        policy: str,
        twin: DigitalTwin,
        k_candidates: int = 3,
        rng: Optional[np.random.Generator] = None,
        theta: float = DEFAULT_THETA,
        budget_scale: float = 1.0,
        migration_cost: Optional[tuple] = None,
    ) -> None:
        if policy not in POLICIES:
            raise ValueError(f"unknown policy {policy}")
        self.policy = policy
        self.twin = twin
        self.k = k_candidates
        self.rng = rng if rng is not None else np.random.default_rng(0)
        self.theta = theta
        self.budget = {
            p: float(min(0.95, budget_scale * f * theta))
            for p, f in RISK_BUDGET_FACTOR.items()
        }
        self.exclude_limit = float(min(0.95, EXCLUDE_FACTOR * theta))
        self.migration_cost = tuple(migration_cost or PROACTIVE_RECONFIG_STEPS)
        self.lead_times: List[int] = []          # slots between trigger and failure
        self.trigger_log: List[Dict[str, float]] = []
        self.decisions: List[Decision] = []
        self.counters: Dict[str, int] = {
            "admitted": 0,
            "rejected": 0,
            "reactive_reroute": 0,
            "proactive_reroute": 0,
            "reroute_failed": 0,
        }

    # ------------------------------------------------------------- selection
    @property
    def risk_aware(self) -> bool:
        """Policies whose path cost includes a risk term. P5 uses the observable
        degradation index in place of a predicted failure probability, so it is
        risk-aware in the routing sense but performs no anticipation."""
        return self.policy in ("proactive_ai", "oracle", "degradation_aware")

    @property
    def proactive(self) -> bool:
        return self.policy in ("proactive_ai", "oracle", "proactive_threshold",
                               "degradation_aware")

    def kpi_alarm(self, path: Sequence[str]) -> bool:
        """Non-learning trigger: any element on the path violates a static limit."""
        for nid in path:
            tel = self.twin.nodes[nid].telemetry
            if not tel:
                continue
            baseline = max(1e-6, self.twin.baseline_latency(nid))
            if (
                tel.get("packet_loss", 0.0) > KPI_ALARM["packet_loss"]
                or tel.get("latency_ms", 0.0) / baseline > KPI_ALARM["latency_ratio"]
                or tel.get("cpu_usage", 0.0) > KPI_ALARM["cpu_usage"]
                or tel.get("availability", 1.0) < KPI_ALARM["availability"]
            ):
                return True
        return False

    def _hard_exclusions(self, priority: int) -> Set[str]:
        if not self.risk_aware:
            return set()
        limit = max(self.exclude_limit, self.budget[priority])
        return frozenset(nid for nid, n in self.twin.nodes.items() if n.risk >= limit)

    def _select_path(self, flow: Flow) -> Optional[List[str]]:
        """Choose an SLA-feasible path with the lowest composite score."""
        excluded = self._hard_exclusions(flow.priority)
        candidates: List[List[str]] = []

        primary = self.twin.shortest_path(
            flow.source, flow.destination, self.risk_aware, excluded, flow.demand_mbps
        )
        if primary:
            candidates.append(primary)
        for path in self.twin.k_shortest_paths(
            flow.source, flow.destination, self.risk_aware, self.k
        ):
            if path not in candidates:
                candidates.append(path)
        if not candidates:
            fallback = self.twin.shortest_path(flow.source, flow.destination, False)
            if fallback:
                candidates.append(fallback)
        if not candidates:
            return None

        best, best_score = None, float("inf")
        for path in candidates:
            latency, loss = self.twin.path_predicted_kpi(path)
            risk = self.twin.path_risk(path) if self.risk_aware else 0.0
            capacity_ok = self.twin.residual_capacity(path) >= flow.demand_mbps
            score = latency / flow.max_latency_ms + 4.0 * loss / max(1e-6, flow.max_loss)
            if self.risk_aware:
                score += (3.0 + 0.6 * flow.priority) * risk
            if not capacity_ok:
                score += 2.5
            if latency > flow.max_latency_ms:
                score += 1.5
            if loss > flow.max_loss:
                score += 1.5
            if score < best_score:
                best, best_score = path, score
        return best

    # ----------------------------------------------------------------- admit
    def admit(self, flow: Flow, step: int) -> bool:
        path = self._select_path(flow)
        if path is None:
            self.counters["rejected"] += 1
            self.decisions.append(Decision(flow.flow_id, step, "drop", [], 1.0, "no_path"))
            return False
        flow.path = path
        flow.last_reroute_step = step
        self.counters["admitted"] += 1
        self.decisions.append(
            Decision(flow.flow_id, step, "admit", path, self.twin.path_risk(path), self.policy)
        )
        return True

    # --------------------------------------------------------------- reroute
    def path_broken(self, path: Sequence[str]) -> bool:
        return any(not self.twin.nodes[n].observed_up for n in path)

    def maybe_reroute(self, flow: Flow, step: int) -> Optional[str]:
        """Return the kind of re-route performed, if any."""
        if self.policy == "static":
            return None
        if step - flow.last_reroute_step < HYSTERESIS_STEPS:
            return None

        broken = self.path_broken(flow.path)
        proactive = False
        if not broken and self.risk_aware:
            proactive = self.twin.path_risk_max(flow.path) > self.budget[flow.priority]
        elif not broken and self.policy == "proactive_threshold":
            proactive = self.kpi_alarm(flow.path)
        if not broken and not proactive:
            return None

        new_path = self._select_path(flow)
        if new_path is None or new_path == flow.path:
            if broken:
                self.counters["reroute_failed"] += 1
            return None

        kind = "reactive_reroute" if broken else "proactive_reroute"
        flow.path = new_path
        flow.last_reroute_step = step
        flow.reroutes += 1
        if kind == "proactive_reroute":
            flow.proactive_reroutes += 1
            delay = int(self.rng.integers(*self.migration_cost, endpoint=True))
            self.trigger_log.append(
                {"step": step, "flow": flow.flow_id, "priority": flow.priority,
                 "risk": self.twin.path_risk_max(flow.path)}
            )
        else:
            delay = int(self.rng.integers(*REACTIVE_RECONFIG_STEPS, endpoint=True))
        flow.reconfig_until = step + delay
        self.counters[kind] += 1
        self.decisions.append(
            Decision(flow.flow_id, step, kind, new_path, self.twin.path_risk(new_path), self.policy)
        )
        return kind
