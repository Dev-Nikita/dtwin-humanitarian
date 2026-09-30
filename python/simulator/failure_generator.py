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
GAMMA_WEAR_SHAPE = 0.25      # shape of the per-slot wear increment (bursty for < 1)
GAMMA_WEAR_GAIN = 0.20       # mean increment = gain * nu_n(t); gives a failure incidence comparable to S4
OBS_LAG = 4                  # telemetry lag of the misspecified observation map [slots]

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


# Layout of the per-element, per-slot random block (common random numbers).
N_NORMAL = 15      # z0 diffusion, z1 failed-state jitter, z2 load idiosyncrasy, z3.. obs noise
N_UNIFORM = 22     # see the U_* indices below
Z_DIFF, Z_FAILJIT, Z_IDIO, Z_OBS0 = 0, 1, 2, 3
(U_SHOCK, U_SHOCKMAG, U_REPAIR, U_POSTREP, U_HOFAIL, U_HODROP, U_CCREPAIR) = range(7)
U_FAILOBS0 = 7     # 12 uniforms for the failed-state observation
U_HO_LAT, U_HO_JIT, U_HO_LOSS = 19, 20, 21


def _unif(u: float, lo: float, hi: float) -> float:
    return float(lo + (hi - lo) * u)


def _randint(u: float, lo: int, hi: int) -> int:
    """Equivalent of Generator.integers(lo, hi) driven by a fixed uniform."""
    return int(lo + min(hi - lo - 1, int(u * (hi - lo))))


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
    # Common-random-number block for the current slot (see RandomBlock). Every
    # exogenous random quantity of this element in slot t is read from a fixed
    # position of this block, so two policies evaluated on the same seed see the
    # same diffusion increments, jumps, repair durations and observation noise
    # for element n at slot t, whatever the policy did before.
    _z: Optional[np.ndarray] = None
    _u: Optional[np.ndarray] = None
    _g: float = 0.0                      # unit-scale gamma draw (wear-out family only)
    family: str = "jump_diffusion"       # degradation family: jump_diffusion | gamma_wear
    obs_family: str = "nominal"          # observation map: nominal | misspecified
    health_hist: Deque[float] = field(default_factory=lambda: deque(maxlen=12))

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
        """Advance the latent health process by one slot.

        All random quantities are read from the element's common-random-number
        block for this slot; `rng` is kept in the signature for compatibility."""
        z, u = self._z, self._u
        if self.status == STATUS_FAILED:
            self.repair_timer -= 1
            self.health = min(FAIL_HEALTH, max(0.0, self.health + 0.005 * z[Z_FAILJIT]))
            if self.repair_timer <= 0:
                lo, hi = POST_REPAIR_HEALTH if self.family != "gamma_wear" else (0.90, 1.0)
                self.health = _unif(u[U_POSTREP], lo, hi)
                self.status = STATUS_RECOVERING
            self.time_since_failure = 0
            return

        if self.family == "gamma_wear":
            # Misspecified family (Section 7.12): monotone wear-out with bursty
            # gamma-distributed increments and no mean reversion. The mean
            # increment is tied to the same degradation intensity nu_n(t).
            self.health -= GAMMA_WEAR_GAIN * self.hazard(scenario, lambda_scale, gain_scale) * self._g
        else:
            drift = MU_RECOVERY * (1.0 - self.health) - self.hazard(scenario, lambda_scale, gain_scale)
            self.health += drift + SIGMA_HEALTH * sigma_scale * z[Z_DIFF]

        # compound Poisson shocks (Bernoulli thinning, one uniform per slot)
        if scenario.shock_prob > 0.0 and self.node_type in scenario.shock_targets:
            if u[U_SHOCK] < scenario.shock_prob:
                self.health -= scenario.shock_magnitude * _unif(u[U_SHOCKMAG], 0.6, 1.2)

        self.health = float(np.clip(self.health, 0.0, 1.0))
        self.time_since_failure += 1

        if self.health < FAIL_HEALTH:
            self.status = STATUS_FAILED
            self.repair_timer = _randint(u[U_REPAIR], *REPAIR_STEPS)
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

        z, u = self._z, self._u
        self.health_hist.append(self.health)
        if self.status == STATUS_FAILED:
            f = u[U_FAILOBS0:U_FAILOBS0 + 12]
            obs = {
                "latency_ms": p.base_latency_ms * _unif(f[0], 6.0, 12.0),
                "jitter_ms": p.base_jitter_ms * _unif(f[1], 5.0, 10.0),
                "packet_loss": _unif(f[2], 0.55, 1.0),
                "throughput_mbps": p.capacity_mbps * _unif(f[3], 0.0, 0.05),
                "cpu_usage": _unif(f[4], 0.0, 12.0),
                "memory_usage": _unif(f[5], 0.0, 20.0),
                "signal_strength": (p.base_signal_dbm - GAMMA_SIGNAL) if p.wireless else 0.0,
                "availability": 0.0,
                "utilisation": 0.0,
                "queue_depth": _unif(f[6], 180.0, 320.0),
                "retx_rate": _unif(f[7], 0.5, 0.95),
                "energy_headroom": _unif(f[8], 0.0, 0.3) if p.battery_backed else 1.0,
            }
            self.ewma_availability = 0.90 * self.ewma_availability
            self.history.append(obs)
            return obs

        mis = self.obs_family == "misspecified"
        if mis:
            # Misspecified observation map: telemetry reflects the health of
            # OBS_LAG slots ago, degrades with different exponents and carries
            # twice the measurement noise.
            h = self.health_hist[-1 - OBS_LAG] if len(self.health_hist) > OBS_LAG else self.health_hist[0]
            deficit = 1.0 - h
        queue_term = min(QUEUE_TERM_CAP, rho / max(1e-3, 1.0 - rho))
        base_latency = self.latency_base_override or p.base_latency_ms
        base_jitter = self.jitter_base_override or p.base_jitter_ms
        e_lat, e_loss = (1.3, 2.0) if mis else (2.0, 3.0)
        latency = base_latency * (
            1.0 + ALPHA_LAT_HEALTH * deficit ** e_lat + ALPHA_LAT_LOAD * queue_term
        )
        jitter = base_jitter * (1.0 + 3.1 * deficit + 1.4 * queue_term)
        loss = (
            p.base_loss
            + BETA_LOSS_HEALTH * deficit ** e_loss
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
            latency *= _unif(u[U_HO_LAT], *HANDOVER_LATENCY_GAIN)
            jitter *= _unif(u[U_HO_JIT], *HANDOVER_JITTER_GAIN)
            loss = min(1.0, loss + _unif(u[U_HO_LOSS], *HANDOVER_LOSS_ADD))
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
        nscale = 2.0 if mis else 1.0
        for i, (key, value) in enumerate(raw.items()):
            zi = nscale * z[Z_OBS0 + i]
            if key in ("signal_strength",):
                noisy[key] = float(value + 1.8 * zi) if value != 0.0 else 0.0
            elif key in ("availability", "utilisation", "energy_headroom"):
                noisy[key] = float(np.clip(value + 0.012 * zi, 0.0, 1.0))
            elif key == "packet_loss":
                noisy[key] = float(np.clip(value * np.exp(OBS_NOISE * 1.6 * zi), 0.0, 1.0))
            elif key == "retx_rate":
                noisy[key] = float(np.clip(value * np.exp(OBS_NOISE * 1.4 * zi), 0.0, 1.0))
            else:
                noisy[key] = float(max(0.0, value * np.exp(OBS_NOISE * zi)))
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
        destroyed: Optional[tuple] = None,
        region_map: Optional[Dict[str, str]] = None,
        family: str = "jump_diffusion",
        obs_family: str = "nominal",
    ) -> None:
        self.scenario = SCENARIOS[scenario_name]
        self.region_map = region_map if region_map is not None else NODE_REGION
        self.family = family
        self.scenario_name = scenario_name
        # Independent, reproducible streams for every exogenous process. The
        # environment never shares a generator with the traffic model or the
        # controller, and every element owns its own stream from which a block
        # of fixed size is drawn in every slot (common random numbers).
        root = np.random.SeedSequence([int(seed), 0xE7])
        ss_init, ss_cc, ss_nodes = root.spawn(3)
        self.rng = rng if rng is not None else np.random.default_rng(ss_init)
        self._cc_rng = np.random.default_rng(ss_cc)
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
        for node in self.nodes.values():
            node.family = family
            node.obs_family = obs_family
        self._node_rng = {
            nid: np.random.default_rng(child)
            for nid, child in zip(sorted(self.nodes), ss_nodes.spawn(len(self.nodes)))
        }
        # Elements physically destroyed before the observation window starts.
        # They are unrecoverable within the episode (repair timer >> horizon).
        for nid in (self.scenario.destroyed_nodes if destroyed is None else destroyed):
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
            idio = 0.06 * node._z[Z_IDIO]
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
                if node._u[U_HOFAIL] < scen.handover_fail_prob:
                    node.health = float(
                        max(0.0, node.health - _unif(node._u[U_HODROP], *HANDOVER_FAIL_HEALTH_DROP))
                    )

    def _apply_common_cause(self) -> None:
        """S6: shared regional shock hitting every element of one region."""
        scen = self.scenario
        if scen.common_cause_rate <= 0.0 or not scen.common_cause_regions:
            return
        # a fixed triple of draws per slot keeps the shock sequence common to all policies
        u_occ, u_reg = self._cc_rng.random(2)
        g = float(self._cc_rng.gamma(2.0, scen.common_cause_magnitude / 2.0))
        if u_occ >= scen.common_cause_rate:
            return
        region = scen.common_cause_regions[min(len(scen.common_cause_regions) - 1,
                                               int(u_reg * len(scen.common_cause_regions)))]
        intensity = g
        self.last_common_cause = (self.t, region, intensity)
        for node in self.nodes.values():
            if self.region_map.get(node.node_id) != region or node.status == STATUS_FAILED:
                continue
            susceptibility = REGION_SUSCEPTIBILITY.get(node.node_type, 1.0)
            node.health = float(max(0.0, node.health - intensity * susceptibility))
            if node.health < FAIL_HEALTH:
                node.status = STATUS_FAILED
                node.repair_timer = _randint(node._u[U_CCREPAIR], *REPAIR_STEPS)

    # ------------------------------------------------------------------ step
    def step(self, load_map: Optional[Dict[str, float]] = None) -> Dict[str, Dict[str, float]]:
        """Advance one slot and return the telemetry snapshot for all nodes."""
        for nid, node in self.nodes.items():
            g = self._node_rng[nid]
            node._z = g.standard_normal(N_NORMAL)
            node._u = g.random(N_UNIFORM)
            if self.family == "gamma_wear":
                node._g = float(g.gamma(GAMMA_WEAR_SHAPE, 1.0 / GAMMA_WEAR_SHAPE))
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
