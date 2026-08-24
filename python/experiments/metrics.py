"""Resilience metric definitions (Section 9 of the paper)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

from ..simulator.traffic_generator import Flow


@dataclass
class FlowOutcome:
    flow_id: int
    service_type: str
    priority: int
    active_steps: int = 0
    outage_steps: int = 0
    sla_violation_steps: int = 0
    delivered_steps: int = 0
    latency_sum: float = 0.0
    loss_sum: float = 0.0
    latency_samples: int = 0
    reroutes: int = 0
    proactive_reroutes: int = 0
    recovery_times: List[int] = field(default_factory=list)
    dropped: bool = False

    @property
    def availability(self) -> float:
        if self.active_steps == 0:
            return 0.0
        return 1.0 - self.outage_steps / self.active_steps

    @property
    def success_rate(self) -> float:
        if self.active_steps == 0:
            return 0.0
        return self.delivered_steps / self.active_steps

    @property
    def mean_latency(self) -> float:
        return self.latency_sum / self.latency_samples if self.latency_samples else float("nan")

    @property
    def mean_loss(self) -> float:
        return self.loss_sum / self.latency_samples if self.latency_samples else float("nan")


class MetricCollector:
    """Aggregates per-flow outcomes into run-level resilience indicators."""

    def __init__(self) -> None:
        self.outcomes: Dict[int, FlowOutcome] = {}
        self.timeline: List[Dict[str, float]] = []

    def ensure(self, flow: Flow) -> FlowOutcome:
        if flow.flow_id not in self.outcomes:
            self.outcomes[flow.flow_id] = FlowOutcome(
                flow.flow_id, flow.service_type, flow.priority
            )
        return self.outcomes[flow.flow_id]

    def record_drop(self, flow: Flow) -> None:
        outcome = self.ensure(flow)
        outcome.dropped = True
        outcome.active_steps += 1
        outcome.outage_steps += 1

    def record_step(
        self,
        flow: Flow,
        outage: bool,
        sla_ok: bool,
        latency: float,
        loss: float,
    ) -> None:
        outcome = self.ensure(flow)
        outcome.active_steps += 1
        if outage:
            outcome.outage_steps += 1
        else:
            outcome.latency_sum += latency
            outcome.loss_sum += loss
            outcome.latency_samples += 1
            if sla_ok:
                outcome.delivered_steps += 1
            else:
                outcome.sla_violation_steps += 1

    def finalise_flow(self, flow: Flow) -> None:
        outcome = self.ensure(flow)
        outcome.reroutes = flow.reroutes
        outcome.proactive_reroutes = flow.proactive_reroutes
        outcome.recovery_times.extend(flow.recovery_steps)

    # ------------------------------------------------------------ aggregation
    def summary(self) -> Dict[str, float]:
        outs = list(self.outcomes.values())
        if not outs:
            return {}
        active = np.array([o.active_steps for o in outs], dtype=float)
        outage = np.array([o.outage_steps for o in outs], dtype=float)
        delivered = np.array([o.delivered_steps for o in outs], dtype=float)
        violation = np.array([o.sla_violation_steps for o in outs], dtype=float)
        latencies = [o.mean_latency for o in outs if o.latency_samples]
        losses = [o.mean_loss for o in outs if o.latency_samples]
        recoveries = [r for o in outs for r in o.recovery_times]

        result = {
            "flows": len(outs),
            "dropped_flows": sum(1 for o in outs if o.dropped),
            "service_availability": 1.0 - outage.sum() / max(1.0, active.sum()),
            "sla_compliance": delivered.sum() / max(1.0, active.sum()),
            "sla_violation_rate": violation.sum() / max(1.0, active.sum()),
            "mean_latency_ms": float(np.mean(latencies)) if latencies else float("nan"),
            "p95_latency_ms": float(np.percentile(latencies, 95)) if latencies else float("nan"),
            "mean_packet_loss": float(np.mean(losses)) if losses else float("nan"),
            "mean_recovery_steps": float(np.mean(recoveries)) if recoveries else 0.0,
            "p95_recovery_steps": float(np.percentile(recoveries, 95)) if recoveries else 0.0,
            "outage_events": len(recoveries),
            "total_reroutes": int(sum(o.reroutes for o in outs)),
            "proactive_reroutes": int(sum(o.proactive_reroutes for o in outs)),
        }

        for stype in sorted({o.service_type for o in outs}):
            sub = [o for o in outs if o.service_type == stype]
            act = sum(o.active_steps for o in sub)
            result[f"avail_{stype}"] = 1.0 - sum(o.outage_steps for o in sub) / max(1, act)
            result[f"success_{stype}"] = sum(o.delivered_steps for o in sub) / max(1, act)
            result[f"slots_{stype}"] = int(act)
        return result
