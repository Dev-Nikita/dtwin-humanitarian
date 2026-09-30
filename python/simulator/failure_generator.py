"""
Stochastic degradation and observation model.

Every node carries a *latent health* h(t) in [0, 1] that is never observed by
the predictor. Health evolves as a mean-reverting jump-diffusion

    dh = mu * (1 - h) dt  -  lambda(t) dt  +  sigma dW  -  J dN            (Eq. 3)

where

    mu        : self-healing / maintenance rate
    lambda(t) : instantaneous hazard, a function of the node profile, the
                scenario stress programme and the current utilisation rho(t)
    sigma dW  : Wiener noise (micro-variability, weather, interference)
    J dN      : compound Poisson shocks (shelling, power loss, satellite
                handover failure, ...)

A node is FAILED when h < FAIL_HEALTH; it then enters a repair phase of
random duration before being restored to a partially healthy state.

Observable telemetry (latency, jitter, loss, throughput, CPU, memory, RSSI,
availability) is generated from (h, rho) through a deterministic observation
map corrupted by multiplicative log-normal noise (Eq. 5-10). Because the map
is noisy and non-invertible, recovering h - and therefore anticipating a
failure - is a genuine statistical learning problem rather than a threshold
lookup.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional

import numpy as np

from .config import (
    ALPHA_LAT_HEALTH,
    ALPHA_LAT_LOAD,
    BETA_LOSS_HEALTH,
    BETA_LOSS_LOAD,
    DEGRADED_HEALTH,
    FAIL_HEALTH,
    GAMMA_SIGNAL,
    HANDOVER_FAIL_HEALTH_DROP,
    HANDOVER_JITTER_GAIN,
    HANDOVER_LATENCY_GAIN,
    HANDOVER_LOSS_ADD,
    NODE_PROFILES,
    NODE_REGION,
    OBS_NOISE,
    QUEUE_TERM_CAP,
    REGION_SUSCEPTIBILITY,
    REPAIR_STEPS,
    RHO_CAP,
    SCENARIOS,
    Scenario,
)

# --- dynamics coefficients (Section 4.2) ------------------------------------
MU_RECOVERY = 0.050          # mu   : mean reversion towards full health
SIGMA_HEALTH = 0.045         # sigma: diffusion coefficient
LAMBDA_BASE = 0.0050         # baseline hazard per step at nominal load
LOAD_HAZARD_GAIN = 3.0       # hazard amplification from utilisation
LOAD_HAZARD_KNEE = 0.65      # utilisation above which hazard grows
HAZARD_MULT_EXPONENT = 1.00  # sub-linear scenario amplification
POST_REPAIR_HEALTH = (0.70, 0.92)

STATUS_HEALTHY = "healthy"
STATUS_DEGRADED = "degraded"
STATUS_FAILED = "failed"
STATUS_RECOVERING = "recovering"

TELEMETRY_FIELDS: List[str] = [
    "latency_ms",
    "jitter_ms",
    "packet_loss",
    "throughput_mbps",
    "cpu_usage",
    "memory_usage",
    "signal_strength",
    "availability",
    "utilisation",
    "queue_depth",
    "retx_rate",
    "energy_headroom",
]



def _lat_override(scenario, node_type: str):
    """Externally anchored latency parameters for a node type, if the scenario
    declares any. Returns (base_ms, jitter_ms); zeros mean 'use the profile'."""
    for ntype, base, jitter in getattr(scenario, "latency_override", ()) or ():
        if ntype == node_type:
            return float(base), float(jitter)
    return 0.0, 0.0


@dataclass
class NodeState:
    """Latent + observable state of a single network element."""

    node_id: str
    node_type: str
    health: float = 1.0
    status: str = STATUS_HEALTHY
    repair_timer: int = 0
    utilisation: float = 0.10
    ewma_availability: float = 0.999
    time_since_failure: int = 10_000
    handover_active: bool = False
    latency_base_override: float = 0.0   # externally anchored base latency, 0 = use profile
    jitter_base_override: float = 0.0
    history: Deque[Dict[str, float]] = field(default_factory=lambda: deque(maxlen=64))

    @property
    def profile(self):
        return NODE_PROFILES[self.node_type]

    @property
    def is_up(self) -> bool:
        return self.status != STATUS_FAILED

    # ---------------------------------------------------------------- hazard
    def hazard(self, scenario: Scenario, lambda_scale: float = 1.0,
               gain_scale: float = 1.0) -> float:
        """lambda(t): instantaneous hazard rate (Eq. 2)."""
        mult = scenario.hazard_multiplier.get(self.node_type, 1.0)
        scen_term = mult ** HAZARD_MULT_EXPONENT
        load_term = 1.0 + gain_scale * LOAD_HAZARD_GAIN * max(
            0.0, self.utilisation - LOAD_HAZARD_KNEE) ** 1.5
        return LAMBDA_BASE * lambda_scale * self.profile.hazard_scale * scen_term * load_term

    # ----------------------------------------------------------------- step
    def advance(self, scenario: Scenario, rng: np.random.Generator,
                lambda_scale: float = 1.0, sigma_scale: float = 1.0,
                gain_scale: float = 1.0) -> None:
        """Advance the latent health process by one slot."""
        if self.status == STATUS_FAILED:
            self.repair_timer -= 1
            self.health = min(FAIL_HEALTH, max(0.0, self.health + rng.normal(0.0, 0.005)))
            if self.repair_timer <= 0:
                self.health = float(rng.uniform(*POST_REPAIR_HEALTH))
                self.status = STATUS_RECOVERING
            self.time_since_failure = 0
            return

        drift = MU_RECOVERY * (1.0 - self.health) - self.hazard(scenario, lambda_scale, gain_scale)
        self.health += drift + SIGMA_HEALTH * sigma_scale * rng.normal()

        # compound Poisson shocks
        if scenario.shock_prob > 0.0 and self.node_type in scenario.shock_targets:
            if rng.random() < scenario.shock_prob:
                self.health -= scenario.shock_magnitude * rng.uniform(0.6, 1.2)

        self.health = float(np.clip(self.health, 0.0, 1.0))
        self.time_since_failure += 1

        if self.health < FAIL_HEALTH:
            self.status = STATUS_FAILED
            self.repair_timer = int(rng.integers(*REPAIR_STEPS))
        elif self.health < DEGRADED_HEALTH:
            self.status = STATUS_DEGRADED
        elif self.status == STATUS_RECOVERING and self.health > 0.90:
            self.status = STATUS_HEALTHY
        elif self.status != STATUS_RECOVERING:
            self.status = STATUS_HEALTHY

    # ----------------------------------------------------------- observation
    def observe(self, rng: np.random.Generator) -> Dict[str, float]:
        """Observation map g(h, rho) + multiplicative noise (Eq. 5-10)."""
        p = self.profile
        h = self.health
        rho = float(np.clip(self.utilisation, 0.0, RHO_CAP))
        deficit = 1.0 - h

        if self.status == STATUS_FAILED:
            obs = {
                "latency_ms": p.base_latency_ms * rng.uniform(6.0, 12.0),
                "jitter_ms": p.base_jitter_ms * rng.uniform(5.0, 10.0),
                "packet_loss": float(rng.uniform(0.55, 1.0)),
                "throughput_mbps": p.capacity_mbps * rng.uniform(0.0, 0.05),
                "cpu_usage": float(rng.uniform(0.0, 12.0)),
                "memory_usage": float(rng.uniform(0.0, 20.0)),
                "signal_strength": (p.base_signal_dbm - GAMMA_SIGNAL) if p.wireless else 0.0,
                "availability": 0.0,
                "utilisation": 0.0,
                "queue_depth": float(rng.uniform(180.0, 320.0)),
                "retx_rate": float(rng.uniform(0.5, 0.95)),
                "energy_headroom": float(rng.uniform(0.0, 0.3)) if p.battery_backed else 1.0,
            }
            self.ewma_availability = 0.90 * self.ewma_availability
            self.history.append(obs)
            return obs

        queue_term = min(QUEUE_TERM_CAP, rho / max(1e-3, 1.0 - rho))
        base_latency = self.latency_base_override or p.base_latency_ms
        base_jitter = self.jitter_base_override or p.base_jitter_ms
        latency = base_latency * (
            1.0 + ALPHA_LAT_HEALTH * deficit ** 2 + ALPHA_LAT_LOAD * queue_term
        )
        jitter = base_jitter * (1.0 + 3.1 * deficit + 1.4 * queue_term)
        loss = (
            p.base_loss
            + BETA_LOSS_HEALTH * deficit ** 3
            + BETA_LOSS_LOAD * max(0.0, rho - 0.80) ** 2
        )
        throughput = p.capacity_mbps * rho * (0.35 + 0.65 * h)
        cpu = 100.0 * np.clip(0.12 + 0.78 * rho + 0.18 * deficit, 0.0, 0.995)
        memory = 100.0 * np.clip(0.20 + 0.55 * rho + 0.22 * deficit ** 1.5, 0.0, 0.995)
        signal = (p.base_signal_dbm - GAMMA_SIGNAL * deficit ** 1.2) if p.wireless else 0.0
        queue_depth = 12.0 * queue_term + 40.0 * deficit ** 2

        # Transient handover reconfiguration (S5). The element is not damaged, but
        # its observable KPIs degrade sharply for the duration of the event, which
        # is exactly the signature that a threshold alarm mistakes for a failure.
        if self.handover_active:
            latency *= rng.uniform(*HANDOVER_LATENCY_GAIN)
            jitter *= rng.uniform(*HANDOVER_JITTER_GAIN)
            loss = min(1.0, loss + rng.uniform(*HANDOVER_LOSS_ADD))
            queue_depth *= 1.8
            throughput *= 0.45

        retx = np.clip(0.6 * loss + 0.12 * deficit ** 2 + 0.05 * max(0.0, rho - 0.85), 0.0, 1.0)
        energy = np.clip(1.0 - 0.55 * deficit - 0.25 * rho, 0.0, 1.0) if p.battery_backed else 1.0

        inst_availability = float(np.clip(0.55 + 0.45 * h - 0.10 * max(0.0, rho - 0.9), 0.0, 1.0))
        self.ewma_availability = 0.85 * self.ewma_availability + 0.15 * inst_availability

        raw = {
            "latency_ms": latency,
            "jitter_ms": jitter,
            "packet_loss": loss,
            "throughput_mbps": throughput,
            "cpu_usage": cpu,
            "memory_usage": memory,
            "signal_strength": signal,
            "availability": self.ewma_availability,
            "utilisation": rho,
            "queue_depth": queue_depth,
            "retx_rate": retx,
            "energy_headroom": energy,
        }

        # multiplicative log-normal noise on strictly positive quantities
        noisy: Dict[str, float] = {}
        for key, value in raw.items():
            if key in ("signal_strength",):
                noisy[key] = float(value + rng.normal(0.0, 1.8)) if value != 0.0 else 0.0
            elif key in ("availability", "utilisation", "energy_headroom"):
                noisy[key] = float(np.clip(value + rng.normal(0.0, 0.012), 0.0, 1.0))
            elif key == "packet_loss":
                noisy[key] = float(np.clip(value * rng.lognormal(0.0, OBS_NOISE * 1.6), 0.0, 1.0))
            elif key == "retx_rate":
                noisy[key] = float(np.clip(value * rng.lognormal(0.0, OBS_NOISE * 1.4), 0.0, 1.0))
            else:
                noisy[key] = float(max(0.0, value * rng.lognormal(0.0, OBS_NOISE)))
        self.history.append(noisy)
        return noisy


class NetworkEnvironment:
    """Container advancing every node of the topology under a given scenario."""

    def __init__(
        self,
        node_catalogue: Dict[str, str],
        scenario_name: str,
        rng: Optional[np.random.Generator] = None,
        seed: int = 0,
        hazard_scale: float = 1.0,
        sigma_scale: float = 1.0,
        feedback_gain_scale: float = 1.0,
    ) -> None:
        self.scenario = SCENARIOS[scenario_name]
        self.scenario_name = scenario_name
        self.rng = rng if rng is not None else np.random.default_rng(seed)
        self.t = 0
        self.hazard_scale = hazard_scale
        self.sigma_scale = sigma_scale
        self.feedback_gain_scale = feedback_gain_scale
        self.last_common_cause: Optional[tuple] = None
        self.nodes: Dict[str, NodeState] = {
            nid: NodeState(node_id=nid, node_type=ntype, health=float(self.rng.uniform(0.86, 1.0)),
                           latency_base_override=_lat_override(self.scenario, ntype)[0],
                           jitter_base_override=_lat_override(self.scenario, ntype)[1])
            for nid, ntype in node_catalogue.items()
        }
        # Elements physically destroyed before the observation window starts.
        # They are unrecoverable within the episode (repair timer >> horizon).
        for nid in self.scenario.destroyed_nodes:
            if nid in self.nodes:
                node = self.nodes[nid]
                node.health = 0.0
                node.status = STATUS_FAILED
                node.repair_timer = 10 ** 6

    # ------------------------------------------------------------------ load
    def apply_load(self, load_map: Optional[Dict[str, float]] = None) -> None:
        """Set per-node utilisation; falls back to a synthetic diurnal profile."""
        if load_map is not None:
            for nid, rho in load_map.items():
                if nid in self.nodes:
                    self.nodes[nid].utilisation = float(np.clip(rho, 0.0, RHO_CAP))
            return

        base = 0.34 * (0.55 + 0.45 * self.scenario.load_multiplier)
        diurnal = 0.14 * np.sin(2.0 * np.pi * self.t / 240.0)
        burst = 0.0
        if self.scenario.burst_period and (self.t % self.scenario.burst_period) < 18:
            burst = 0.22
        for node in self.nodes.values():
            idio = self.rng.normal(0.0, 0.06)
            capacity_relief = 0.10 * (node.profile.capacity_mbps / 1400.0)
            rho = base + diurnal + burst + idio - capacity_relief
            node.utilisation = float(np.clip(rho, 0.02, RHO_CAP))

    # ------------------------------------------------- exogenous disturbances
    def _apply_handovers(self) -> None:
        """S5: periodic non-terrestrial reconfiguration events."""
        scen = self.scenario
        if scen.handover_period <= 0:
            return
        phase = (self.t - scen.handover_phase) % scen.handover_period
        active = phase < scen.handover_duration
        for node in self.nodes.values():
            if node.node_type != "LEO":
                continue
            node.handover_active = active and node.status != STATUS_FAILED
            # A handover that fails to complete leaves the element in a worse
            # physical state; this is the only path by which S5 damages hardware.
            if active and phase == 0 and node.status != STATUS_FAILED:
                if self.rng.random() < scen.handover_fail_prob:
                    node.health = float(
                        max(0.0, node.health - self.rng.uniform(*HANDOVER_FAIL_HEALTH_DROP))
                    )

    def _apply_common_cause(self) -> None:
        """S6: shared regional shock hitting every element of one region."""
        scen = self.scenario
        if scen.common_cause_rate <= 0.0 or not scen.common_cause_regions:
            return
        if self.rng.random() >= scen.common_cause_rate:
            return
        region = str(self.rng.choice(scen.common_cause_regions))
        intensity = float(self.rng.gamma(2.0, scen.common_cause_magnitude / 2.0))
        self.last_common_cause = (self.t, region, intensity)
        for node in self.nodes.values():
            if NODE_REGION.get(node.node_id) != region or node.status == STATUS_FAILED:
                continue
            susceptibility = REGION_SUSCEPTIBILITY.get(node.node_type, 1.0)
            node.health = float(max(0.0, node.health - intensity * susceptibility))
            if node.health < FAIL_HEALTH:
                node.status = STATUS_FAILED
                node.repair_timer = int(self.rng.integers(*REPAIR_STEPS))

    # ------------------------------------------------------------------ step
    def step(self, load_map: Optional[Dict[str, float]] = None) -> Dict[str, Dict[str, float]]:
        """Advance one slot and return the telemetry snapshot for all nodes."""
        self.apply_load(load_map)
        self._apply_handovers()
        self._apply_common_cause()
        snapshot: Dict[str, Dict[str, float]] = {}
        for node in self.nodes.values():
            node.advance(self.scenario, self.rng, self.hazard_scale,
                         self.sigma_scale, self.feedback_gain_scale)
            obs = node.observe(self.rng)
            obs["_status"] = node.status
            obs["_health"] = node.health
            snapshot[node.node_id] = obs
        self.t += 1
        return snapshot

    def failed_nodes(self) -> List[str]:
        return [n.node_id for n in self.nodes.values() if n.status == STATUS_FAILED]
