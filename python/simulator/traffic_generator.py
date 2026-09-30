"""
Humanitarian traffic model.

Service requests arrive as a marked Poisson process. Each mark is a service
class from Table 2 with its own priority, latency budget, loss budget and
bandwidth demand. Flows have a random holding time, so the offered load is a
superposition of on-off sources rather than a memoryless per-slot demand.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .config import SERVICE_PROFILES, SERVICE_TYPES


@dataclass
class Flow:
    """A single admitted or pending humanitarian service flow."""

    flow_id: int
    service_type: str
    priority: int
    source: str
    destination: str
    demand_mbps: float
    max_latency_ms: float
    max_loss: float
    start_step: int
    duration: int
    path: List[str] = field(default_factory=list)
    active: bool = True
    reroutes: int = 0
    proactive_reroutes: int = 0
    last_reroute_step: int = -10_000
    reconfig_until: int = -1          # slot until which the flow is being re-established
    outage_steps: int = 0
    sla_violation_steps: int = 0
    delivered_steps: int = 0
    first_outage_step: Optional[int] = None
    first_interrupt_at: Optional[int] = None   # slot of the first interruption
    first_sustained_at: Optional[int] = None   # first interruption lasting >= 2 slots
    outage_run: int = 0
    recovery_steps: List[int] = field(default_factory=list)

    @property
    def end_step(self) -> int:
        return self.start_step + self.duration


class TrafficGenerator:
    """Marked Poisson arrivals of humanitarian service flows."""

    def __init__(
        self,
        sources: List[str],
        sinks: List[str],
        rng: np.random.Generator,
        base_rate: float = 2.0,
        load_multiplier: float = 1.0,
    ) -> None:
        self.sources = sources
        self.sinks = sinks
        self.rng = rng
        self.base_rate = base_rate * load_multiplier
        self._next_id = 0
        self._weights = np.array([SERVICE_PROFILES[s].arrival_weight for s in SERVICE_TYPES])
        self._weights = self._weights / self._weights.sum()

    def arrivals(self, step: int, surge: float = 1.0) -> List[Flow]:
        """Return the flows arriving in slot `step`."""
        n = self.rng.poisson(self.base_rate * surge)
        flows: List[Flow] = []
        for _ in range(int(n)):
            stype = str(self.rng.choice(SERVICE_TYPES, p=self._weights))
            profile = SERVICE_PROFILES[stype]
            demand = profile.min_bandwidth_mbps * float(self.rng.uniform(1.0, 1.8))
            duration = int(self.rng.integers(*profile.duration_steps))
            flow = Flow(
                flow_id=self._next_id,
                service_type=stype,
                priority=profile.priority,
                source=str(self.rng.choice(self.sources)),
                destination=str(self.rng.choice(self.sinks)),
                demand_mbps=demand,
                max_latency_ms=profile.max_latency_ms,
                max_loss=profile.max_loss,
                start_step=step,
                duration=max(1, duration),
            )
            self._next_id += 1
            flows.append(flow)
        return flows

    def summary(self) -> Dict[str, float]:
        return {"generated_flows": self._next_id, "arrival_rate": self.base_rate}
