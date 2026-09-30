"""
Calibrated perfect-state risk (P5c, Reviewer 3.6).

The original perfect-state reference P5 mapped the latent health h onto the
controller's risk scale through an arbitrary monotone band,
r = clip((0.35 - h) / (0.35 - 0.18), 0, 1). That mapping is not a probability, so
a controller calibrated on probabilities cannot be expected to act on it
sensibly. P5c instead uses the exact quantity the predictor is trained to
estimate,

    r*(h; c, s) = P( exists u in (t, t+H]: h_n(u) < h_fail | h_n(t) = h, class c, scenario s ),

computed by Monte Carlo from the generative model itself (the jump-diffusion
of Eq. 1 with the scenario multiplier, the jump process, and the handover and
common-cause mechanisms averaged over their timing). Utilisation is fixed at the
nominal level; the load term of Eq. (2) is therefore neglected, which Section 7.8
shows to be immaterial at the nominal coupling.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Dict

import numpy as np

from ..simulator import failure_generator as fg
from ..simulator.config import (
    FAIL_HEALTH, NODE_PROFILES, PREDICTION_HORIZON, REGION_SUSCEPTIBILITY, SCENARIOS,
    HANDOVER_FAIL_HEALTH_DROP,
)

H_GRID = np.linspace(0.0, 1.0, 201)


@lru_cache(maxsize=None)
def risk_table(scenario: str, node_type: str, horizon: int = PREDICTION_HORIZON,
               n_paths: int = 6000, seed: int = 99, in_region: bool = True) -> np.ndarray:
    scen = SCENARIOS[scenario]
    prof = NODE_PROFILES[node_type]
    rng = np.random.default_rng(seed)
    nu = (fg.LAMBDA_BASE * prof.hazard_scale
          * scen.hazard_multiplier.get(node_type, 1.0) ** fg.HAZARD_MULT_EXPONENT)
    h = np.repeat(H_GRID[:, None], n_paths, axis=1)
    failed = h < FAIL_HEALTH
    shock_p = scen.shock_prob if node_type in scen.shock_targets else 0.0
    cc_p = (scen.common_cause_rate / max(1, len(scen.common_cause_regions))
            if scen.common_cause_rate > 0 and in_region else 0.0)
    susc = REGION_SUSCEPTIBILITY.get(node_type, 1.0)
    ho_p = (scen.handover_fail_prob / scen.handover_period
            if scen.handover_period > 0 and node_type == "LEO" else 0.0)
    for _ in range(horizon):
        h = h + fg.MU_RECOVERY * (1.0 - h) - nu + fg.SIGMA_HEALTH * rng.standard_normal(h.shape)
        if shock_p > 0:
            hit = rng.random(h.shape) < shock_p
            h = h - hit * scen.shock_magnitude * rng.uniform(0.6, 1.2, h.shape)
        if cc_p > 0:
            hit = rng.random(h.shape) < cc_p
            h = h - hit * susc * rng.gamma(2.0, scen.common_cause_magnitude / 2.0, h.shape)
        if ho_p > 0:
            hit = rng.random(h.shape) < ho_p
            h = h - hit * rng.uniform(*HANDOVER_FAIL_HEALTH_DROP, h.shape)
        h = np.clip(h, 0.0, 1.0)
        failed |= h < FAIL_HEALTH
    return failed.mean(axis=1)


def oracle_risk(scenario: str, node_type: str, health: float,
                horizon: int = PREDICTION_HORIZON, in_region: bool = True) -> float:
    table = risk_table(scenario, node_type, horizon, in_region=in_region)
    return float(np.interp(health, H_GRID, table))


def describe(scenario: str = "S4_disaster_collapse") -> Dict[str, Dict[float, float]]:
    out = {}
    for ntype in ("5G", "LEO", "UAV"):
        tab = risk_table(scenario, ntype)
        out[ntype] = {float(h): float(np.interp(h, H_GRID, tab)) for h in (0.2, 0.25, 0.3, 0.35, 0.4, 0.5)}
    return out


if __name__ == "__main__":  # pragma: no cover
    for k, v in describe().items():
        print(k, {h: round(p, 3) for h, p in v.items()})
