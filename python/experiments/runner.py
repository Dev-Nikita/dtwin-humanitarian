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
from ..orchestrator.policies import Orchestrator
from ..predictor.online_features import OnlineFeatureBuilder
from ..simulator.config import DETECTION_DELAY, EPISODE_STEPS, SCENARIOS
from ..simulator.failure_generator import STATUS_FAILED, NetworkEnvironment
from ..simulator.network_topology import NODE_CATALOGUE, SINKS, TERMINALS, build_topology
from ..simulator.traffic_generator import Flow, TrafficGenerator
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
    ) -> None:
        self.scenario = scenario
        self.policy = policy
        self.seed = seed
        self.steps = steps
        self.model = model
        self.feature_columns = feature_columns
        self.risk_scale = risk_scale

        self.rng = np.random.default_rng(seed)
        self.load_feedback = load_feedback
        self.env = NetworkEnvironment(
            NODE_CATALOGUE, scenario, seed=seed,
            hazard_scale=hazard_scale, sigma_scale=sigma_scale,
            feedback_gain_scale=feedback_gain_scale,
        )
        self.topology_class = topology
        if topology == "reference":
            self.graph = build_topology()
        else:
            from ..simulator.topology_variants import build_variant
            self.graph = build_variant(topology)
        self.twin = DigitalTwin(self.graph, detection_delay=DETECTION_DELAY)
        self.theta = theta
        self.orch = Orchestrator(
            policy, self.twin, rng=self.rng, theta=theta, budget_scale=budget_scale,
            migration_cost=migration_cost,
        )
        self._deg_scale = 0.10
        scale_path = os.path.join("data", "results", "degradation_index_scale.json")
        if os.path.exists(scale_path):
            import json as _json
            self._deg_scale = max(1e-3, float(
                _json.load(open(scale_path))["index_at_operating_point"]))
        self.lead_tracker = LeadTimeTracker(theta=theta)
        self.survival: List[SurvivalRecord] = []
        self.survival_sustained: List[SurvivalRecord] = []
        self.quality: List[float] = []
        self.traffic = TrafficGenerator(
            [t for t in TERMINALS if t in self.graph],
            [s for s in SINKS if s in self.graph],
            self.rng,
            load_multiplier=SCENARIOS[scenario].load_multiplier,
        )
        self.features = OnlineFeatureBuilder(latency_medians=latency_medians)
        self.collector = MetricCollector()
        self.active: List[Flow] = []
        self.all_flows: List[Flow] = []
        self.node_types = {nid: ntype for nid, ntype in NODE_CATALOGUE.items()
                           if nid in self.graph}
        self.timeline: List[Dict[str, float]] = []

    # ------------------------------------------------------------------ risk
    def _risk(self) -> Dict[str, float]:
        if self.policy in ("static", "reactive"):
            return {nid: 0.0 for nid in self.node_types}
        if self.policy == "degradation_aware":
            # P5: a purely contemporaneous degradation score built from the same
            # observable KPIs, with no reference to the future. It isolates the
            # value of anticipation from the value of avoiding degraded elements.
            risk = {}
            for nid in self.node_types:
                tel = self.twin.nodes[nid].telemetry
                if not tel:
                    risk[nid] = 0.0
                    continue
                baseline = max(1e-6, self.twin.baseline_latency(nid))
                score = (0.45 * min(1.0, tel.get("packet_loss", 0.0) / 0.20)
                         + 0.25 * min(1.0, max(0.0, tel.get("latency_ms", 0.0) / baseline - 1.0))
                         + 0.20 * (1.0 - tel.get("availability", 1.0))
                         + 0.10 * min(1.0, tel.get("retx_rate", 0.0) / 0.30))
                # Matched operating point: the index is rescaled so that the value
                # at which it fires as often as the predictor maps onto theta*.
                # Without this the two baselines would be compared at different
                # alarm rates and the difference would be an artefact of scaling.
                risk[nid] = float(np.clip(score / self._deg_scale * self.theta, 0.0, 1.0))
            return risk
        if self.policy == "oracle":
            risk = {}
            for nid, node in self.env.nodes.items():
                h = node.health
                r = (ORACLE_HIGH - h) / (ORACLE_HIGH - ORACLE_LOW)
                risk[nid] = float(np.clip(r, 0.0, 1.0))
            return risk
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
                flow.recovery_steps.append(step - flow.first_outage_step)
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

            # 2. physical environment advances under that load. Disabling the
            #    feedback breaks the policy -> load -> hazard path while leaving
            #    the observation model untouched.
            snapshot = self.env.step(load_map if self.load_feedback else None)

            # 3. twin synchronisation (telemetry only)
            self.twin.ingest(snapshot)
            for nid, obs in snapshot.items():
                self.features.push(nid, obs)

            # 4. AI risk estimation
            risk = self._risk()
            self.twin.set_risk(risk)
            if self.policy in ("proactive_ai", "oracle"):
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

            # 6. re-routing pass
            for flow in self.active:
                self.orch.maybe_reroute(flow, step)

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
        return summary

    def timeline_frame(self) -> pd.DataFrame:
        df = pd.DataFrame(self.timeline)
        df["scenario"] = self.scenario
        df["policy"] = self.policy
        df["seed"] = self.seed
        return df


def load_predictor(results_dir: str = os.path.join("data", "results")):
    """Load the calibrated model selected during training."""
    import json

    with open(os.path.join(results_dir, "best_model.json")) as fh:
        meta = json.load(fh)
    model = joblib.load(os.path.join(results_dir, f"{meta['best_model']}_model.joblib"))
    return model, meta
