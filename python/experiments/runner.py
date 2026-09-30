"""
Closed-loop experiment runner.

One run = (scenario, policy, seed). The loop couples four sub-systems:

    physical environment  ->  telemetry  ->  digital twin  ->  AI risk
                                                     |
                                              orchestrator
                                                     |
                                       routed flows -> offered load
                                                     |
                                   (feeds back into the hazard model)

Because utilisation feeds the hazard rate, orchestration decisions change the
future failure process: the evaluation is a genuine closed loop rather than a
replay of a fixed trace.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd

from ..orchestrator.digital_twin import DigitalTwin
from ..orchestrator.policies import POLICY_SPECS, Orchestrator
from ..predictor.online_features import OnlineFeatureBuilder
from ..simulator.config import (DETECTION_DELAY, EPISODE_STEPS, PREDICTION_HORIZON,
                                REACTIVE_RECONFIG_STEPS, SCENARIOS)
from ..simulator.failure_generator import STATUS_FAILED, NetworkEnvironment
from ..simulator.network_topology import NODE_CATALOGUE, SINKS, TERMINALS, build_topology
from ..simulator.traffic_generator import Flow, TrafficGenerator
from .oracle_table import oracle_risk
from .metrics import MetricCollector
from .resilience import (
    LeadTimeTracker,
    SurvivalRecord,
    normalised_resilience,
    resilience_loss,
)

ORACLE_LOW, ORACLE_HIGH = 0.18, 0.35   # health band mapped to oracle risk


class ExperimentRun:
    def __init__(
        self,
        scenario: str,
        policy: str,
        seed: int,
        steps: int = EPISODE_STEPS,
        model=None,
        feature_columns: Optional[List[str]] = None,
        latency_medians: Optional[Dict[str, float]] = None,
        risk_scale: float = 1.0,
        theta: float = 0.22,
        budget_scale: float = 1.0,
        migration_cost: Optional[tuple] = None,
        topology: str = "reference",
        load_feedback: bool = True,
        hazard_scale: float = 1.0,
        sigma_scale: float = 1.0,
        feedback_gain_scale: float = 1.0,
        horizon: int = PREDICTION_HORIZON,
        budget_factors: Optional[Dict[int, float]] = None,
        score_coef: Optional[Dict[str, float]] = None,
        load_trace: Optional[List[Dict[str, float]]] = None,
        kpi_scale: Optional[float] = None,
        family: str = "jump_diffusion",
        obs_family: str = "nominal",
    ) -> None:
        self.scenario = scenario
        self.spec = POLICY_SPECS[policy]
        self.horizon = int(horizon)
        self.load_trace_in = load_trace
        self.load_trace: List[Dict[str, float]] = []
        self.policy = policy
        self.seed = seed
        self.steps = steps
        self.model = model
        self.feature_columns = feature_columns
        self.risk_scale = risk_scale

        # Separate reproducible streams: exogenous traffic, controller-internal
        # randomness (actuation delays) and the environment (seeded inside
        # NetworkEnvironment). Traffic therefore never depends on how many draws
        # the controller consumed, so arrivals, demands, holding times and
        # endpoints are identical across policies for a given seed (Section 6.4).
        ss_traffic, ss_control, ss_acct = np.random.SeedSequence([int(seed), 0x7A]).spawn(3)
        self.rng = np.random.default_rng(ss_control)
        self.acct_rng = np.random.default_rng(ss_acct)
        self.traffic_rng = np.random.default_rng(ss_traffic)
        self.load_feedback = load_feedback
        self.topology_class = topology
        destroyed, region_map = None, None
        terminals, sinks, arrival_factor = TERMINALS, SINKS, 1.0
        if topology == "reference":
            self.graph = build_topology()
            catalogue = dict(NODE_CATALOGUE)
        elif topology == "large":
            from ..simulator.large_topology import (
                LARGE_ARRIVAL_FACTOR, LARGE_CATALOGUE, LARGE_REGION, LARGE_SINKS,
                LARGE_TERMINALS, build_large, large_destroyed,
            )
            self.graph = build_large()
            catalogue = dict(LARGE_CATALOGUE)
            destroyed, region_map = large_destroyed(scenario), LARGE_REGION
            terminals, sinks, arrival_factor = LARGE_TERMINALS, LARGE_SINKS, LARGE_ARRIVAL_FACTOR
        else:
            from ..simulator.topology_variants import build_variant
            self.graph = build_variant(topology)
            catalogue = {n: t for n, t in NODE_CATALOGUE.items() if n in self.graph}
        self.env = NetworkEnvironment(
            catalogue, scenario, seed=seed,
            hazard_scale=hazard_scale, sigma_scale=sigma_scale,
            feedback_gain_scale=feedback_gain_scale,
            destroyed=destroyed, region_map=region_map,
            family=family, obs_family=obs_family,
        )
        self.region_map = region_map
        self.twin = DigitalTwin(self.graph, detection_delay=DETECTION_DELAY)
        self.theta = theta
        kpi_scale = 1.0 if kpi_scale is None else float(kpi_scale)
        if self.spec.kpi_scale_key == "matched" and kpi_scale == 1.0:
            kpi_scale = _load_scale("kpi_alarm_scale.json", "kpi_scale_at_operating_point", 1.0)
        self.orch = Orchestrator(
            policy, self.twin, rng=self.rng, theta=theta, budget_scale=budget_scale,
            migration_cost=migration_cost, budget_factors=budget_factors,
            score_coef=score_coef, kpi_scale=kpi_scale,
        )
        self._deg_scale = max(1e-3, _load_scale(
            "degradation_index_scale.json", "index_at_operating_point", 0.10))
        self.lead_tracker = LeadTimeTracker(theta=theta, horizon=self.horizon)
        self.survival: List[SurvivalRecord] = []
        self.survival_sustained: List[SurvivalRecord] = []
        self.quality: List[float] = []
        self.traffic = TrafficGenerator(
            [t for t in terminals if t in self.graph],
            [s for s in sinks if s in self.graph],
            self.traffic_rng,
            load_multiplier=SCENARIOS[scenario].load_multiplier * arrival_factor,
        )
        self.features = OnlineFeatureBuilder(latency_medians=latency_medians)
        self.collector = MetricCollector()
        self.active: List[Flow] = []
        self.all_flows: List[Flow] = []
        self.node_types = {nid: ntype for nid, ntype in catalogue.items()
                           if nid in self.graph}
        self.timeline: List[Dict[str, float]] = []
        self.dead_path_migrations = 0
        self.trigger_truth = {"already_failed": 0, "fails_within_H": 0, "no_failure_within_H": 0}
        self._pending: List = []
        self._fail_onsets: Dict[str, List[int]] = {}
        self._prev_failed: Dict[str, bool] = {}

    # ------------------------------------------------------------------ risk
    def degradation_index(self, nid: str) -> float:
        """Contemporaneous degradation index of P3: a weighted combination of
        the current observable KPIs with no reference to the future."""
        tel = self.twin.nodes[nid].telemetry
        if not tel:
            return 0.0
        baseline = max(1e-6, self.twin.baseline_latency(nid))
        return (0.45 * min(1.0, tel.get("packet_loss", 0.0) / 0.20)
                + 0.25 * min(1.0, max(0.0, tel.get("latency_ms", 0.0) / baseline - 1.0))
                + 0.20 * (1.0 - tel.get("availability", 1.0))
                + 0.10 * min(1.0, tel.get("retx_rate", 0.0) / 0.30))

    def _risk(self) -> Dict[str, float]:
        src = self.spec.risk
        if src in ("none", "kpi"):
            return {nid: 0.0 for nid in self.node_types}
        if src == "index":
            # Matched operating point: the index is rescaled so that the value at
            # which it fires as often as the predictor maps onto theta*. The scale
            # is estimated on validation runs only and frozen (Section 6.3).
            return {nid: float(np.clip(self.degradation_index(nid) / self._deg_scale * self.theta,
                                       0.0, 1.0)) for nid in self.node_types}
        if src == "oracle_band":
            risk = {}
            for nid, node in self.env.nodes.items():
                r = (ORACLE_HIGH - node.health) / (ORACLE_HIGH - ORACLE_LOW)
                risk[nid] = float(np.clip(r, 0.0, 1.0))
            return risk
        if src == "oracle_prob":
            return {nid: (1.0 if node.status == STATUS_FAILED else
                          oracle_risk(self.scenario, node.node_type, node.health, self.horizon,
                                      in_region=(self.env.region_map.get(nid) in
                                                 SCENARIOS[self.scenario].common_cause_regions)))
                    for nid, node in self.env.nodes.items()}
        return self._predicted_risk()

    def _predicted_risk(self) -> Dict[str, float]:
        rows = []
        ids = []
        for nid, ntype in self.node_types.items():
            if nid in self.features.buffers:
                rows.append(self.features.features(nid, ntype))
                ids.append(nid)
        if not rows:
            return {nid: 0.0 for nid in self.node_types}
        frame = pd.DataFrame(rows)
        if self.feature_columns:
            for col in self.feature_columns:
                if col not in frame.columns:
                    frame[col] = np.nan
            frame = frame[self.feature_columns]
        prob = self.model.predict_proba(frame)[:, 1]
        prob = np.clip(prob * self.risk_scale, 0.0, 1.0)
        return {nid: float(p) for nid, p in zip(ids, prob)}

    # ------------------------------------------------------------- evaluation
    def _evaluate(self, snapshot: Dict[str, Dict[str, float]], step: int) -> Dict[str, float]:
        outage_flows = 0
        sla_ok_flows = 0
        latencies: List[float] = []
        for flow in self.active:
            path = flow.path
            physical_break = any(self.env.nodes[n].status == STATUS_FAILED for n in path)
            reconfiguring = step <= flow.reconfig_until
            broken = physical_break or reconfiguring
            if broken:
                if flow.first_outage_step is None:
                    flow.outage_cause = ("migration" if (not physical_break and
                                                         flow.reconfig_kind == "migration")
                                         else "failure")
                flow.outage_steps += 1
                flow.outage_run += 1
                if flow.outage_run == 2 and flow.first_sustained_at is None:
                    flow.first_sustained_at = step - 1
                if flow.first_interrupt_at is None:
                    flow.first_interrupt_at = step
                if flow.first_outage_step is None:
                    flow.first_outage_step = step
                self.collector.record_step(flow, True, False, 0.0, 1.0)
                outage_flows += 1
                continue

            flow.outage_run = 0
            if flow.first_outage_step is not None:
                dur = step - flow.first_outage_step
                flow.recovery_steps.append(dur)
                (flow.recovery_migration if flow.outage_cause == "migration"
                 else flow.recovery_failure).append(dur)
                flow.first_outage_step = None

            latency = sum(snapshot[n]["latency_ms"] for n in path)
            for u, v in zip(path[:-1], path[1:]):
                latency += self.graph[u][v]["prop_latency_ms"]
            keep = 1.0
            for n in path:
                keep *= 1.0 - min(1.0, snapshot[n]["packet_loss"])
            loss = 1.0 - keep
            congested = self.twin.residual_capacity(path) < 0.0
            sla_ok = (
                latency <= flow.max_latency_ms
                and loss <= flow.max_loss
                and not congested
            )
            if not sla_ok:
                flow.sla_violation_steps += 1
            else:
                flow.delivered_steps += 1
                sla_ok_flows += 1
            latencies.append(latency)
            self.collector.record_step(flow, False, sla_ok, latency, loss)

        n_active = max(1, len(self.active))
        self.quality.append(1.0 - outage_flows / n_active)
        return {
            "step": step,
            "active_flows": len(self.active),
            "outage_fraction": outage_flows / n_active,
            "sla_fraction": sla_ok_flows / n_active,
            "mean_latency_ms": float(np.mean(latencies)) if latencies else float("nan"),
            "failed_nodes": len(self.env.failed_nodes()),
            "mean_risk": float(np.mean([n.risk for n in self.twin.nodes.values()])),
        }

    # -------------------------------------------------------------------- run
    def run(self) -> Dict[str, float]:
        for step in range(self.steps):
            # 1. offered load from the currently placed flows
            self.twin.reset_load()
            for flow in self.active:
                if flow.path:
                    self.twin.add_flow_load(flow.path, flow.demand_mbps)
            load_map = self.twin.load_map()
            self.load_trace.append(load_map)

            # 2. physical environment advances under that load. Two ways of
            #    breaking the policy -> load -> degradation path are supported:
            #    load_feedback=False replaces the load by the synthetic profile
            #    (original ablation); load_trace replays a recorded utilisation
            #    trajectory, typically that of P1 on the same seed, so the load
            #    level is unchanged but no longer responds to the policy.
            if self.load_trace_in is not None:
                env_load = self.load_trace_in[min(step, len(self.load_trace_in) - 1)]
            elif self.load_feedback:
                env_load = load_map
            else:
                env_load = None
            snapshot = self.env.step(env_load)

            for nid, node in self.env.nodes.items():
                f = node.status == STATUS_FAILED
                if f and not self._prev_failed.get(nid, False):
                    self._fail_onsets.setdefault(nid, []).append(step)
                self._prev_failed[nid] = f

            # 3. twin synchronisation (telemetry only)
            self.twin.ingest(snapshot)
            for nid, obs in snapshot.items():
                self.features.push(nid, obs)

            # 4. AI risk estimation
            risk = self._risk()
            self.twin.set_risk(risk)
            if self.spec.risk in ("index", "predictor", "oracle_band", "oracle_prob"):
                self.lead_tracker.update(
                    step, risk,
                    {nid: n.status == STATUS_FAILED for nid, n in self.env.nodes.items()},
                )

            # 5. admission of new flows
            for flow in self.traffic.arrivals(step):
                self.all_flows.append(flow)
                if self.orch.admit(flow, step):
                    self.active.append(flow)
                else:
                    self.collector.record_drop(flow)

            # 6. re-routing pass; differentiated policies process flows in
            #    decreasing criticality so that scarce residual capacity goes to
            #    the most consequential services first (stable sort).
            order = (sorted(self.active, key=lambda f: -f.priority)
                     if self.orch.differentiated else self.active)
            for flow in order:
                old_path = list(flow.path) if flow.path else []
                kind = self.orch.maybe_reroute(flow, step)
                if (kind == "proactive_reroute" and DEAD_PATH_MIGRATION_COST
                        and any(self.env.nodes[n].status == STATUS_FAILED for n in old_path)):
                    # Physical accounting (revision): a "migration" away from a path
                    # that is already physically down cannot be make-before-break,
                    # because there is no working path to keep the service on while
                    # the new one is established. The interruption is therefore the
                    # restoration delay C_r, not the migration cost C_m. The
                    # controller itself cannot see the physical state; only the
                    # assessment of the interruption depends on it.
                    delay = int(self.acct_rng.integers(*REACTIVE_RECONFIG_STEPS, endpoint=True))
                    flow.reconfig_until = step + delay
                    flow.reconfig_kind = "restoration"
                    self.dead_path_migrations += 1
                if kind == "proactive_reroute" and self.orch.last_trigger_node:
                    # Ground-truth classification of what the migration reacted to
                    # (assessment layer only; never visible to the controller).
                    node = self.env.nodes[self.orch.last_trigger_node]
                    if node.status == STATUS_FAILED:
                        self.trigger_truth["already_failed"] += 1
                    else:
                        self._pending.append((self.orch.last_trigger_node, step))

            # 7. evaluation on the true physical state
            self.timeline.append(self._evaluate(snapshot, step))

            # 8. flow expiry
            still_active = []
            for flow in self.active:
                if step >= flow.end_step:
                    self.collector.finalise_flow(flow)
                else:
                    still_active.append(flow)
            self.active = still_active

        for flow in self.active:
            self.collector.finalise_flow(flow)

        for outcome_flow in self.all_flows:
            observed = min(self.steps, outcome_flow.end_step) - outcome_flow.start_step
            if outcome_flow.first_interrupt_at is not None:
                self.survival.append(SurvivalRecord(
                    max(0, outcome_flow.first_interrupt_at - outcome_flow.start_step), 1))
            else:
                self.survival.append(SurvivalRecord(max(0, observed), 0))
            if outcome_flow.first_sustained_at is not None:
                self.survival_sustained.append(SurvivalRecord(
                    max(0, outcome_flow.first_sustained_at - outcome_flow.start_step), 1))
            else:
                self.survival_sustained.append(SurvivalRecord(max(0, observed), 0))

        summary = self.collector.summary()
        summary.update(
            {
                "scenario": self.scenario,
                "policy": self.policy,
                "topology": self.topology_class,
                "load_feedback": self.load_feedback,
                "seed": self.seed,
                "steps": self.steps,
                "admitted": self.orch.counters["admitted"],
                "rejected": self.orch.counters["rejected"],
                "reactive_reroutes": self.orch.counters["reactive_reroute"],
                "protection_switches": self.orch.counters["protection_switch"],
                "backup_refreshes": self.orch.counters["backup_refresh"],
                "horizon": self.horizon,
                "generated_flows": len(self.all_flows),
                "proactive_reroutes_total": self.orch.counters["proactive_reroute"],
                "reroute_failures": self.orch.counters["reroute_failed"],
                "node_failure_slots": int(
                    sum(1 for r in self.timeline for _ in range(int(r["failed_nodes"])))
                ),
                "resilience_loss": resilience_loss(self.quality),
                "normalised_resilience": normalised_resilience(self.quality),
                "mean_quality": float(np.mean(self.quality)) if self.quality else float("nan"),
                "min_quality": float(np.min(self.quality)) if self.quality else float("nan"),
            }
        )
        summary.update(self.lead_tracker.summary())
        # retrospective resolution of migrations triggered on non-failed elements
        H = self.horizon
        within = sum(1 for nid, t in self._pending
                     if any(t < u <= t + H for u in self._fail_onsets.get(nid, [])))
        summary.update({
            "mig_trigger_already_failed": self.trigger_truth["already_failed"],
            "dead_path_migrations": self.dead_path_migrations,
            "mig_trigger_fails_within_H": within,
            "mig_trigger_no_failure_within_H": len(self._pending) - within,
        })
        rf = [d for f in self.all_flows for d in f.recovery_failure]
        rm = [d for f in self.all_flows for d in f.recovery_migration]
        summary.update({
            "failure_interruptions": len(rf),
            "migration_interruptions": len(rm),
            "mean_recovery_failure": float(np.mean(rf)) if rf else 0.0,
            "mean_recovery_migration": float(np.mean(rm)) if rm else 0.0,
            "outage_slots_failure": int(sum(rf)),
            "outage_slots_migration": int(sum(rm)),
        })
        return summary

    def timeline_frame(self) -> pd.DataFrame:
        df = pd.DataFrame(self.timeline)
        df["scenario"] = self.scenario
        df["policy"] = self.policy
        df["seed"] = self.seed
        return df


# Charge C_r (not C_m) when a proactive migration leaves a path that is already
# physically broken. False reproduces the accounting of the first submission.
DEAD_PATH_MIGRATION_COST: bool = True


def _load_scale(name: str, key: str, default: float) -> float:
    path = os.path.join("data", "results", name)
    if os.path.exists(path):
        import json as _json
        with open(path) as fh:
            return float(_json.load(fh)[key])
    return default


def load_predictor(results_dir: str = os.path.join("data", "results")):
    """Load the calibrated model selected during training."""
    import json

    with open(os.path.join(results_dir, "best_model.json")) as fh:
        meta = json.load(fh)
    model = joblib.load(os.path.join(results_dir, f"{meta['best_model']}_model.joblib"))
    return model, meta
