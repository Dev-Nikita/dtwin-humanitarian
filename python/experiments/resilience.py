"""
Reliability and resilience quantities.

The closed-loop experiments produce, for every run, a service-quality trajectory
Q(t) and a set of per-flow interruption records. This module converts them into
the system-safety quantities that the manuscript reports:

    reliability          R_c(t) = P(T_c > t), T_c = time to first interruption
    availability         fraction of flow-slots in the operational state
    resilience loss      L = integral of the quality deficit over the episode
    normalised resilience  Rn = 1 - L / (T * Q0)
    empirical hazard     h(t) = f(t) / S(t) of the interruption process
    anticipation lead time  slots between the first risk alarm and the failure
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Resilience of a quality trajectory
# ---------------------------------------------------------------------------
def resilience_loss(quality: Sequence[float], nominal: float = 1.0) -> float:
    """L = sum_t [Q0 - Q(t)]_+  (Eq. 20). Slot duration is one, so the sum is
    the discrete counterpart of the integral over the disruption window."""
    q = np.asarray(quality, dtype=float)
    return float(np.clip(nominal - q, 0.0, None).sum())


def normalised_resilience(quality: Sequence[float], nominal: float = 1.0) -> float:
    """Rn = 1 - L / (T Q0) in [0, 1]  (Eq. 21)."""
    q = np.asarray(quality, dtype=float)
    if q.size == 0 or nominal <= 0:
        return float("nan")
    return float(1.0 - resilience_loss(q, nominal) / (q.size * nominal))


def disruption_depth(quality: Sequence[float], nominal: float = 1.0) -> float:
    """Maximum instantaneous quality drop, i.e. the robustness component."""
    q = np.asarray(quality, dtype=float)
    return float(np.clip(nominal - q, 0.0, None).max()) if q.size else float("nan")


# ---------------------------------------------------------------------------
# Survival analysis of the service-interruption process
# ---------------------------------------------------------------------------
@dataclass
class SurvivalRecord:
    """Time to the first interruption of one flow, with right censoring."""

    duration: int      # slots observed before the event or before censoring
    event: int         # 1 = interruption observed, 0 = right censored


def kaplan_meier(records: Sequence[SurvivalRecord]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (times, survival, at_risk) for the Kaplan-Meier estimator."""
    if not records:
        return np.array([]), np.array([]), np.array([])
    durations = np.array([r.duration for r in records], dtype=float)
    events = np.array([r.event for r in records], dtype=int)
    order = np.argsort(durations)
    durations, events = durations[order], events[order]

    times = np.unique(durations)
    survival, at_risk_out = [], []
    s = 1.0
    n = len(durations)
    for t in times:
        at_risk = int((durations >= t).sum())
        d = int(((durations == t) & (events == 1)).sum())
        if at_risk > 0:
            s *= 1.0 - d / at_risk
        survival.append(s)
        at_risk_out.append(at_risk)
    return times, np.array(survival), np.array(at_risk_out, dtype=float)


def empirical_hazard(records: Sequence[SurvivalRecord], bin_width: int = 25) -> Tuple[np.ndarray, np.ndarray]:
    """Piecewise-constant hazard h(t) = events / (at risk * bin width)."""
    if not records:
        return np.array([]), np.array([])
    durations = np.array([r.duration for r in records], dtype=float)
    events = np.array([r.event for r in records], dtype=int)
    edges = np.arange(0, durations.max() + bin_width, bin_width)
    centres, hazard = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        at_risk = int((durations >= lo).sum())
        d = int(((durations >= lo) & (durations < hi) & (events == 1)).sum())
        if at_risk == 0:
            continue
        centres.append(0.5 * (lo + hi))
        hazard.append(d / (at_risk * bin_width))
    return np.array(centres), np.array(hazard)


def median_survival(times: np.ndarray, survival: np.ndarray) -> float:
    """First time at which the survival estimate drops to or below one half."""
    if times.size == 0:
        return float("nan")
    below = np.where(survival <= 0.5)[0]
    return float(times[below[0]]) if below.size else float("inf")


def restricted_mean_survival(times: np.ndarray, survival: np.ndarray, horizon: float) -> float:
    """RMST: area under the survival curve truncated at `horizon`."""
    if times.size == 0:
        return float("nan")
    t = np.concatenate([[0.0], times])
    s = np.concatenate([[1.0], survival])
    mask = t <= horizon
    t, s = t[mask], s[mask]
    if t[-1] < horizon:
        t = np.concatenate([t, [horizon]])
        s = np.concatenate([s, [s[-1]]])
    return float(np.sum(np.diff(t) * s[:-1]))


# ---------------------------------------------------------------------------
# Anticipation lead time
# ---------------------------------------------------------------------------
class LeadTimeTracker:
    """Horizon-consistent event matching for failure anticipation.

    A warning is credited only when its threshold crossing occurs in the same
    prediction horizon as the subsequent failure.  Earlier crossings are not
    retrospectively counted as successful predictions: at that time their
    ``failure within H`` label was negative.
    """

    def __init__(self, theta: float, horizon: int) -> None:
        self.theta = theta
        if horizon < 1:
            raise ValueError("horizon must be positive")
        self.horizon = horizon
        self.alarm_steps: Dict[str, deque[int]] = {}
        self.was_failed: Dict[str, bool] = {}
        self.lead_times: List[int] = []
        self.missed: int = 0
        self.alarms: int = 0

    def update(self, step: int, risk: Dict[str, float], failed: Dict[str, bool]) -> None:
        for nid, r in risk.items():
            previously_failed = self.was_failed.get(nid, False)
            now_failed = failed.get(nid, False)

            alarms = self.alarm_steps.setdefault(nid, deque())
            while alarms and alarms[0] < step - self.horizon:
                alarms.popleft()
            if r >= self.theta and not now_failed:
                alarms.append(step)
                self.alarms += 1

            if now_failed and not previously_failed:      # failure onset
                eligible = [t for t in alarms if 0 < step - t <= self.horizon]
                if eligible:
                    self.lead_times.append(step - min(eligible))
                else:
                    self.missed += 1
                alarms.clear()
            if previously_failed and not now_failed:      # repaired
                alarms.clear()
            self.was_failed[nid] = now_failed

    def summary(self) -> Dict[str, float]:
        lt = np.array(self.lead_times, dtype=float)
        total = len(lt) + self.missed
        out = {
            "anticipated_failures": int(len(lt)),
            "missed_failures": int(self.missed),
            "anticipation_rate": float(len(lt) / total) if total else float("nan"),
            "alarms_raised": int(self.alarms),
        }
        if lt.size:
            out.update({
                "lead_time_mean": float(lt.mean()),
                "lead_time_median": float(np.median(lt)),
                "lead_time_p25": float(np.percentile(lt, 25)),
                "lead_time_p75": float(np.percentile(lt, 75)),
                "lead_ge_1": float((lt >= 1).mean()),
                "lead_ge_3": float((lt >= 3).mean()),
                "lead_ge_5": float((lt >= 5).mean()),
                "lead_ge_10": float((lt >= 10).mean()),
            })
        else:
            out.update({k: float("nan") for k in
                        ("lead_time_mean", "lead_time_median", "lead_time_p25",
                         "lead_time_p75", "lead_ge_1", "lead_ge_3", "lead_ge_5", "lead_ge_10")})
        return out


# ---------------------------------------------------------------------------
# Decision-theoretic interpretation of the risk budget
# ---------------------------------------------------------------------------
def break_even_probability(migration_cost: float, failure_cost: float) -> float:
    """p* = C_m / C_f: the risk above which pre-emptive migration is rational
    under a linear cost model (Eq. 22)."""
    if failure_cost <= 0:
        return float("nan")
    return float(min(1.0, migration_cost / failure_cost))
