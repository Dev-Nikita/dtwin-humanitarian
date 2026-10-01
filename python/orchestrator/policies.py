"""
Service orchestration policies.

Every policy is a combination of a small number of mechanisms, so that the
contribution of each can be read off a controlled comparison (Section 6.3):

    risk source      none | kpi (static alarm) | index (contemporaneous
                     degradation index) | predictor (calibrated P(fail<=H)) |
                     oracle_band (monotone map of the latent health, the
                     original P5) | oracle_prob (Monte Carlo P(fail<=H | h),
                     the calibrated perfect-state reference)
    differentiation  criticality (class-dependent budgets b_c, criticality
                     term 0.6*chi_c in Eq. 11, migrations processed in
                     decreasing criticality) | uniform (b_c = 1 for every class,
                     a constant risk coefficient equal to the class-average one,
                     arrival-order processing)
    restoration      reactive (break-before-make after detection, cost C_r) |
                     protection (pre-computed node-disjoint backup, switched to
                     after detection at cost C_m, reactive fallback)

Named policies:

    static                      P0   path fixed at admission
    reactive                    P1   reactive restoration (reference)
    protection                  P1b  1:1 pre-computed protection + fast switch-over
    proactive_threshold         P2   static KPI alarm, nominal operator limits
    proactive_threshold_matched P2m  same rule, limits relaxed so that its per-element
                                     alarm rate equals the predictor's (validation runs)
    degradation_aware_uniform   P3u  index risk, uniform treatment of classes
    degradation_aware           P3   index risk, criticality differentiation
    proactive_ai_uniform        P4u  predicted risk, uniform treatment of classes
    proactive_ai                P4   predicted risk, criticality differentiation (proposed)
    oracle                      P5   latent-health band map (original P5)
    oracle_calibrated           P5c  exact P(fail within H | h) from the generative model
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set

import numpy as np

from ..simulator.config import (
    HYSTERESIS_STEPS,
    PROACTIVE_RECONFIG_STEPS,
    REACTIVE_RECONFIG_STEPS,
)
from ..simulator.traffic_generator import Flow
from .digital_twin import DigitalTwin

# Service-differentiated risk budget tau_c = min(0.95, kappa * b_c * theta*) (Eq. 10).
RISK_BUDGET_FACTOR: Dict[int, float] = {5: 0.80, 4: 1.00, 3: 1.30, 2: 1.80, 1: 2.50}
UNIFORM_BUDGET_FACTOR: Dict[int, float] = {5: 1.0, 4: 1.0, 3: 1.0, 2: 1.0, 1: 1.0}
BUDGET_PROFILES: Dict[str, Dict[int, float]] = {
    "nominal": RISK_BUDGET_FACTOR,
    "flat": UNIFORM_BUDGET_FACTOR,
    "shallow": {5: 0.90, 4: 1.00, 3: 1.15, 2: 1.40, 1: 1.75},
    "steep": {5: 0.50, 4: 0.75, 3: 1.25, 2: 2.50, 1: 4.00},
}
EXCLUDE_FACTOR: float = 2.50
# First-submission behaviour: the hysteresis interval also delayed the
# restoration of a flow whose path had been observed down. Kept only as a switch
# for the attribution analysis of the revision; False in every reported run.
LEGACY_RESTORATION_HYSTERESIS: bool = False
DEFAULT_THETA: float = 0.22

# Protection model (revision R1b). "feasible": the standby path is pre-signalled
# but its capacity is not reserved, and a switch-over is admitted only if the
# standby path has enough residual capacity at the moment of the switch (flows
# affected in the same slot are served in a fixed order and see each other's
# load); otherwise the flow falls back to reactive restoration. "reserved": the
# standby bandwidth is reserved in advance, so it is unavailable to other flows
# and the switch-over is always feasible. "optimistic": no capacity check at the
# switch-over (the first-revision model, kept as an upper bound).
PROTECTION_MODE: str = "feasible"
PROTECTION_MODES = ("feasible", "reserved", "optimistic")
# Standby refresh discipline of predictive protection: at most one refresh per
# flow per REFRESH_COOLDOWN slots, and a replacement is adopted only if it is
# observed up, capacity-feasible and lowers the bottleneck risk of the standby
# path by at least REFRESH_MIN_GAIN (or the current standby path is down).
REFRESH_COOLDOWN: int = 6
REFRESH_MIN_GAIN: float = 0.05

# Coefficients of the candidate-selection objective, Eq. (11):
# J = D/Dmax + LOSS * p/pmax + (RISK0 + RISK1 * chi_c) * R_prod + Pen
SCORE_COEF: Dict[str, float] = {"loss": 4.0, "risk0": 3.0, "risk1": 0.6}

# Backwards-compatible absolute budgets (used by the sensitivity sweep).
RISK_BUDGET: Dict[int, float] = {
    p: min(0.95, f * DEFAULT_THETA) for p, f in RISK_BUDGET_FACTOR.items()
}

# Static operator KPI limits (P2). An element is in alarm when its KPI
# severity  s = max(loss/0.12, latency_ratio/1.9, cpu/88, 0.80/availability)
# exceeds the scale factor (1.0 for the nominal operator limits).
KPI_ALARM = {
    "packet_loss": 0.12,
    "latency_ratio": 1.9,
    "cpu_usage": 88.0,
    "availability": 0.80,
}


@dataclass(frozen=True)
class PolicySpec:
    risk: str = "none"               # none | kpi | index | predictor | oracle_band | oracle_prob
    differentiation: str = "criticality"
    restoration: str = "reactive"    # reactive | protection | none
    kpi_scale_key: str = ""          # "" = nominal limits, "matched" = validation-matched
    migrate: bool = True             # risk may trigger make-before-break migrations of live flows
    backup_refresh: bool = False     # risk re-selects the standby (backup) path, no interruption
    primary_risk: bool = True        # risk enters primary-path selection (ablation switch)


POLICY_SPECS: Dict[str, PolicySpec] = {
    "static": PolicySpec(restoration="none"),
    "reactive": PolicySpec(),
    "protection": PolicySpec(restoration="protection"),
    "proactive_threshold": PolicySpec(risk="kpi"),
    "proactive_threshold_matched": PolicySpec(risk="kpi", kpi_scale_key="matched"),
    "degradation_aware_uniform": PolicySpec(risk="index", differentiation="uniform"),
    "degradation_aware": PolicySpec(risk="index"),
    "proactive_ai_uniform": PolicySpec(risk="predictor", differentiation="uniform"),
    "proactive_ai": PolicySpec(risk="predictor"),
    "oracle": PolicySpec(risk="oracle_band"),
    "oracle_calibrated": PolicySpec(risk="oracle_prob"),
    # Predictive protection (revision): the risk estimate is spent on actions that
    # interrupt nothing -- choosing a low-risk primary at admission and keeping the
    # standby path away from elements whose risk exceeds the budget -- while the
    # switch-over itself stays detection-driven, as in 1:1 protection.
    "protection_index": PolicySpec(risk="index", restoration="protection",
                                   migrate=False, backup_refresh=True),
    "protection_predictive": PolicySpec(risk="predictor", restoration="protection",
                                        migrate=False, backup_refresh=True),
    "protection_oracle": PolicySpec(risk="oracle_prob", restoration="protection",
                                    migrate=False, backup_refresh=True),
    "protection_predictive_mig": PolicySpec(risk="predictor", restoration="protection",
                                            migrate=True, backup_refresh=True),
    # Post-confirmation exploratory ablation of P6 (does not alter any policy above):
    # risk used only for the primary path (standby as in P1b), or only for the standby.
    "protection_predictive_route": PolicySpec(risk="predictor", restoration="protection",
                                              migrate=False, backup_refresh=False),
    "protection_predictive_standby": PolicySpec(risk="predictor", restoration="protection",
                                                migrate=False, backup_refresh=True,
                                                primary_risk=False),
}
POLICIES = tuple(POLICY_SPECS)

# Paper labels.
POLICY_LABEL: Dict[str, str] = {
    "static": "P0 static",
    "reactive": "P1 reactive restoration",
    "protection": "P1b 1:1 protection",
    "proactive_threshold": "P2 threshold-proactive",
    "proactive_threshold_matched": "P2m threshold-proactive, matched rate",
    "degradation_aware_uniform": "P3u degradation-aware, uniform",
    "degradation_aware": "P3 degradation-aware",
    "proactive_ai_uniform": "P4u predictive, uniform",
    "proactive_ai": "P4 proposed (predictive)",
    "oracle": "P5 perfect-state, band map",
    "oracle_calibrated": "P5c perfect-state, calibrated",
    "protection_index": "P6i protection, index-refreshed backup",
    "protection_predictive": "P6 predictive protection",
    "protection_oracle": "P6o protection, oracle-refreshed backup",
    "protection_predictive_mig": "P6m predictive protection + migration",
    "protection_predictive_route": "P6-route risk-aware primary only",
    "protection_predictive_standby": "P6-standby risk-aware standby only",
}


@dataclass
class Decision:
    flow_id: int
    step: int
    kind: str            # admit | reactive_reroute | protection_switch | proactive_reroute | drop
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
        budget_factors: Optional[Dict[int, float]] = None,
        score_coef: Optional[Dict[str, float]] = None,
        kpi_scale: float = 1.0,
        protection_mode: Optional[str] = None,
        duration_fn=None,
    ) -> None:
        if policy not in POLICY_SPECS:
            raise ValueError(f"unknown policy {policy}")
        self.policy = policy
        self.spec = POLICY_SPECS[policy]
        self.twin = twin
        self.k = k_candidates
        self.rng = rng if rng is not None else np.random.default_rng(0)
        self.theta = theta
        self.kpi_scale = float(kpi_scale)
        self.score_coef = dict(SCORE_COEF, **(score_coef or {}))
        if self.spec.differentiation == "uniform":
            factors = UNIFORM_BUDGET_FACTOR
        else:
            factors = budget_factors or RISK_BUDGET_FACTOR
        self.budget = {
            p: float(min(0.95, budget_scale * f * theta)) for p, f in factors.items()
        }
        self.exclude_limit = float(min(0.95, EXCLUDE_FACTOR * theta))
        self.migration_cost = tuple(migration_cost or PROACTIVE_RECONFIG_STEPS)
        self.trigger_log: List[Dict[str, float]] = []
        self.decisions: List[Decision] = []
        self.backup: Dict[int, List[str]] = {}
        self.protection_mode = protection_mode or PROTECTION_MODE
        assert self.protection_mode in PROTECTION_MODES
        self.duration_fn = duration_fn
        self.last_refresh: Dict[int, int] = {}
        self.counters: Dict[str, int] = {
            "path_computations": 0,
            "switch_infeasible": 0,
            "admitted": 0,
            "rejected": 0,
            "reactive_reroute": 0,
            "protection_switch": 0,
            "backup_refresh": 0,
            "proactive_reroute": 0,
            "reroute_failed": 0,
        }

    # ------------------------------------------------------------- selection
    @property
    def risk_aware(self) -> bool:
        """Policies whose path cost contains the risk term of Eq. (7)."""
        return self.spec.risk in ("index", "predictor", "oracle_band", "oracle_prob")

    @property
    def proactive(self) -> bool:
        return self.spec.risk != "none"

    @property
    def differentiated(self) -> bool:
        return self.spec.differentiation == "criticality"

    def kpi_severity(self, nid: str) -> float:
        tel = self.twin.nodes[nid].telemetry
        if not tel:
            return 0.0
        baseline = max(1e-6, self.twin.baseline_latency(nid))
        return max(
            tel.get("packet_loss", 0.0) / KPI_ALARM["packet_loss"],
            (tel.get("latency_ms", 0.0) / baseline) / KPI_ALARM["latency_ratio"],
            tel.get("cpu_usage", 0.0) / KPI_ALARM["cpu_usage"],
            KPI_ALARM["availability"] / max(1e-6, tel.get("availability", 1.0)),
        )

    def kpi_alarm_node(self, nid: str) -> bool:
        return self.kpi_severity(nid) > self.kpi_scale

    def kpi_alarm(self, path: Sequence[str]) -> bool:
        """Non-learning trigger: any element on the path violates a static limit."""
        return any(self.kpi_alarm_node(n) for n in path)

    def _hard_exclusions(self, priority: int) -> Set[str]:
        if self.spec.risk == "kpi":
            # P2 does not route onto elements that are themselves in alarm; this is
            # the analogue of the risk exclusion of the risk-aware policies.
            return frozenset(n for n in self.twin.nodes if self.kpi_alarm_node(n))
        if not self.risk_aware or not self.spec.primary_risk:
            return frozenset()
        limit = max(self.exclude_limit, self.budget[priority])
        return frozenset(nid for nid, n in self.twin.nodes.items() if n.risk >= limit)

    def _risk_coef(self, priority: int) -> float:
        c = self.score_coef
        chi = priority if self.differentiated else 3.0
        return c["risk0"] + c["risk1"] * chi

    def _select_path(self, flow: Flow, avoid: Optional[Set[str]] = None) -> Optional[List[str]]:
        """Choose an SLA-feasible path with the lowest composite score (Eq. 11)."""
        excluded = set(self._hard_exclusions(flow.priority)) | set(avoid or ())
        excluded -= {flow.source, flow.destination}
        excluded = frozenset(excluded)
        candidates: List[List[str]] = []
        ra = self.risk_aware and self.spec.primary_risk

        primary = self.twin.shortest_path(
            flow.source, flow.destination, ra, excluded, flow.demand_mbps
        )
        if primary:
            candidates.append(primary)
        for path in self.twin.k_shortest_paths(
            flow.source, flow.destination, ra, self.k
        ):
            if path in candidates:
                continue
            if self.spec.risk == "kpi" and (set(path) & excluded):
                continue
            candidates.append(path)
        if not candidates and not avoid:
            fallback = self.twin.shortest_path(flow.source, flow.destination, False)
            if fallback:
                candidates.append(fallback)
        if not candidates:
            return None

        best, best_score = None, float("inf")
        for path in candidates:
            latency, loss = self.twin.path_predicted_kpi(path)
            risk = self.twin.path_risk(path) if ra else 0.0
            capacity_ok = self.twin.residual_capacity(path) >= flow.demand_mbps
            score = (latency / flow.max_latency_ms
                     + self.score_coef["loss"] * loss / max(1e-6, flow.max_loss))
            if ra:
                score += self._risk_coef(flow.priority) * risk
            if not capacity_ok:
                score += 2.5
            if latency > flow.max_latency_ms:
                score += 1.5
            if loss > flow.max_loss:
                score += 1.5
            if score < best_score:
                best, best_score = path, score
        return best

    def _candidate_backup(self, flow: Flow) -> Optional[List[str]]:
        """Node-disjoint standby path. Predictive protection keeps it off elements
        whose risk exceeds the class budget and uses the risk-augmented cost; under
        the feasible and reserved models a capacity-feasible path is preferred."""
        self.counters["path_computations"] += 1
        inner = set(flow.path[1:-1])
        if self.spec.backup_refresh:
            risky = {n for n, st in self.twin.nodes.items()
                     if st.risk > self.budget[flow.priority]}
            inner |= (risky - {flow.source, flow.destination})
        risk_aware = self.spec.backup_refresh and self.risk_aware
        demand = flow.demand_mbps if self.protection_mode != "optimistic" else 0.0
        path = self.twin.shortest_path(flow.source, flow.destination, risk_aware,
                                       frozenset(inner), demand)
        if path is None and demand > 0.0:
            path = self.twin.shortest_path(flow.source, flow.destination, risk_aware,
                                           frozenset(inner), 0.0)
        if path is None and self.spec.backup_refresh:
            path = self.twin.shortest_path(flow.source, flow.destination, False,
                                           frozenset(flow.path[1:-1]), 0.0)
        if path and path != flow.path:
            return path
        return None

    def _compute_backup(self, flow: Flow) -> None:
        if self.spec.restoration != "protection" or not flow.path:
            return
        path = self._candidate_backup(flow)
        if path:
            self.backup[flow.flow_id] = path
        else:
            self.backup.pop(flow.flow_id, None)

    def _maybe_refresh_backup(self, flow: Flow, step: int) -> None:
        if step - self.last_refresh.get(flow.flow_id, -10_000) < REFRESH_COOLDOWN:
            return
        bk = self.backup.get(flow.flow_id)
        bk_down = bk is None or self.path_broken(bk)
        bk_risk = 1.0 if bk is None else self.twin.path_risk_max(bk)
        if not bk_down and bk_risk <= self.budget[flow.priority]:
            return
        new = self._candidate_backup(flow)
        self.last_refresh[flow.flow_id] = step
        if new is None or new == bk or self.path_broken(new):
            return
        if not bk_down and self.twin.path_risk_max(new) > bk_risk - REFRESH_MIN_GAIN:
            return
        if (self.protection_mode != "optimistic"
                and self.twin.residual_capacity(new) < flow.demand_mbps and not bk_down):
            return
        self.backup[flow.flow_id] = new
        self.counters["backup_refresh"] += 1

    def _duration(self, kind: str, flow: Flow, step: int, old_path) -> tuple:
        """Interruption length in slots and its cause label. The assessment layer
        supplies a keyed draw (common random numbers per physical event); without
        it the controller stream is used."""
        if self.duration_fn is not None:
            return self.duration_fn(kind, flow, step, old_path)
        if kind in ("proactive_reroute", "protection_switch"):
            d = int(self.rng.integers(*self.migration_cost, endpoint=True))
        else:
            d = int(self.rng.integers(*REACTIVE_RECONFIG_STEPS, endpoint=True))
        return d, ("migration" if kind == "proactive_reroute" else "restoration")

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
        self.twin.move_flow_load([], path, flow.demand_mbps)
        self._compute_backup(flow)
        self.decisions.append(
            Decision(flow.flow_id, step, "admit", path, self.twin.path_risk(path), self.policy)
        )
        return True

    # --------------------------------------------------------------- reroute
    def path_broken(self, path: Sequence[str]) -> bool:
        return any(not self.twin.nodes[n].observed_up for n in path)

    def maybe_reroute(self, flow: Flow, step: int) -> Optional[str]:
        """Return the kind of re-route performed, if any."""
        if self.spec.restoration == "none":
            return None
        broken = self.path_broken(flow.path)
        # Standby-path refresh interrupts no traffic; it has its own cooldown.
        if not broken and self.spec.backup_refresh:
            self._maybe_refresh_backup(flow, step)
        # Hysteresis bounds proactive churn only; it never delays the restoration
        # of a flow whose path has been observed to be down.
        if ((not broken or LEGACY_RESTORATION_HYSTERESIS)
                and step - flow.last_reroute_step < HYSTERESIS_STEPS):
            return None

        proactive = False
        self.last_trigger_node = None
        if not self.spec.migrate:
            pass
        elif not broken and self.risk_aware:
            proactive = self.twin.path_risk_max(flow.path) > self.budget[flow.priority]
        elif not broken and self.spec.risk == "kpi":
            proactive = self.kpi_alarm(flow.path)
        if not broken and not proactive:
            return None
        if proactive:
            if self.risk_aware:
                self.last_trigger_node = max(flow.path, key=lambda n: self.twin.nodes[n].risk)
            else:
                self.last_trigger_node = max(flow.path, key=self.kpi_severity)

        new_path = None
        kind = "proactive_reroute" if not broken else "reactive_reroute"
        if broken and self.spec.restoration == "protection":
            backup = self.backup.get(flow.flow_id)
            if backup and not self.path_broken(backup):
                feasible = (self.protection_mode != "feasible"
                            or self.twin.residual_capacity(backup) >= flow.demand_mbps)
                if feasible:
                    new_path, kind = backup, "protection_switch"
                else:
                    self.counters["switch_infeasible"] += 1
        if new_path is None:
            new_path = self._select_path(flow)
        if new_path is None or new_path == flow.path:
            if broken:
                self.counters["reroute_failed"] += 1
            return None

        old_path = list(flow.path)
        flow.path = new_path
        flow.last_reroute_step = step
        flow.reroutes += 1
        self.twin.move_flow_load(old_path, new_path, flow.demand_mbps)
        if kind == "proactive_reroute":
            flow.proactive_reroutes += 1
            self.trigger_log.append(
                {"step": step, "flow": flow.flow_id, "priority": flow.priority,
                 "risk": self.twin.path_risk_max(flow.path)}
            )
        delay, label = self._duration(kind, flow, step, old_path)
        # Half-open accounting: an action in slot t with duration C interrupts
        # the flow in slots t, ..., t + C - 1, i.e. exactly C slots (C = 0: none).
        flow.reconfig_until = step + delay - 1
        flow.reconfig_kind = label
        self._compute_backup(flow)
        self.counters[kind] += 1
        self.decisions.append(
            Decision(flow.flow_id, step, kind, new_path, self.twin.path_risk(new_path), self.policy)
        )
        return kind
