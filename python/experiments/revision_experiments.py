"""
Experiments added in the revision (JRESS-D-26-06464, round 1).

    python -m python.experiments.revision_experiments feedback      R5.3
    python -m python.experiments.revision_experiments misspec       R3.1, R1.3
    python -m python.experiments.revision_experiments large         R1.2
    python -m python.experiments.revision_experiments design        R1.6
    python -m python.experiments.revision_experiments endogeneity   R5.3 (direct test)

Every closed-loop experiment is paired: all policies of one configuration are run
on the same seeds, and the common-random-number environment guarantees that the
exogenous randomness (traffic, diffusion, jumps, repairs, observation noise) is
identical across them.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
from typing import Dict, List

import numpy as np
import pandas as pd

from ..orchestrator.policies import BUDGET_PROFILES
from ..predictor.features import episode_split, select_eligible
from ..simulator import failure_generator as fg
from ..simulator.dataset_generator import build_dataset
from .run_experiments import DATA_DEFAULT, prepare_context
from .runner import ExperimentRun

RESULTS = os.path.join("data", "results")
_CTX: Dict = {}


def _run(scen, policy, seed, steps=500, **kw):
    ctx = _CTX
    return ExperimentRun(scen, policy, seed, steps=steps, model=ctx["model"],
                         feature_columns=ctx["feature_columns"],
                         latency_medians=ctx["latency_medians"],
                         theta=float(ctx["meta"]["threshold"]), **kw)


def _pool_map(fn, tasks, workers=None):
    workers = workers or (os.cpu_count() or 1)
    if workers > 1:
        with mp.get_context("fork").Pool(workers) as pool:
            return pool.map(fn, tasks, chunksize=1)
    return [fn(t) for t in tasks]


# ---------------------------------------------------------------------------
# 1. feedback ablation at fixed traffic (R5.3)
# ---------------------------------------------------------------------------
COUPLINGS = {
    # name: (feedback gain scale, knee of the load term)
    "nominal": (1.0, fg.LOAD_HAZARD_KNEE),
    "strong": (4.0, 0.40),
}


def _feedback_task(args):
    coupling, scen, seed = args
    gain, knee = COUPLINGS[coupling]
    fg.LOAD_HAZARD_KNEE = knee
    rows = []
    ref = _run(scen, "reactive", seed, feedback_gain_scale=gain)
    s = ref.run()
    trace = ref.load_trace
    s.update(coupling=coupling, feedback="endogenous")
    rows.append(s)
    for policy in ("degradation_aware", "proactive_ai"):
        for mode in ("endogenous", "replayed_P1_load"):
            kw = dict(feedback_gain_scale=gain)
            if mode == "replayed_P1_load":
                kw["load_trace"] = trace
            r = _run(scen, policy, seed, **kw).run()
            r.update(coupling=coupling, feedback=mode)
            rows.append(r)
    fg.LOAD_HAZARD_KNEE = COUPLINGS["nominal"][1]
    return rows


def run_feedback(seeds: int = 24) -> pd.DataFrame:
    """P1 is run with the load it produces itself; its utilisation trajectory is
    recorded and replayed to the environment when P3 and P4 are evaluated with the
    feedback path cut. The offered load level, the traffic and all exogenous
    randomness are therefore identical in both arms; the only difference is
    whether the degradation process responds to the policy's own routing."""
    tasks = [(c, scen, 6000 + 29 * k) for c in COUPLINGS
             for scen in ("S3_leo_degradation", "S4_disaster_collapse", "S6_common_cause")
             for k in range(seeds)]
    out = pd.DataFrame([r for rows in _pool_map(_feedback_task, tasks) for r in rows])
    out.to_csv(os.path.join(RESULTS, "feedback_ablation_fixed.csv"), index=False)
    return out


# ---------------------------------------------------------------------------
# 2. direct endogeneity test: do policies change the failure process at all?
# ---------------------------------------------------------------------------
def _endo_task(args):
    coupling, scen, seed = args
    gain, knee = COUPLINGS[coupling]
    fg.LOAD_HAZARD_KNEE = knee
    rows = []
    for policy in ("static", "reactive", "proactive_ai"):
        r = _run(scen, policy, seed, feedback_gain_scale=gain)
        s = r.run()
        util = np.array([v for m in r.load_trace for v in m.values()])
        onsets = 0
        # failure onsets from the recorded timeline are not stored per node; count
        # node-failure slots and utilisation exceedance instead
        rows.append({"coupling": coupling, "scenario": scen, "policy": policy, "seed": seed,
                     "node_failure_slots": s["node_failure_slots"],
                     "util_above_knee": float((util > knee).mean()),
                     "util_p99": float(np.percentile(util, 99)),
                     "service_availability": s["service_availability"],
                     "resilience_loss": s["resilience_loss"], "onsets": onsets})
    fg.LOAD_HAZARD_KNEE = COUPLINGS["nominal"][1]
    return rows


def run_endogeneity(seeds: int = 12) -> pd.DataFrame:
    tasks = [(c, scen, 6000 + 29 * k) for c in COUPLINGS
             for scen in ("S2_bs_overload", "S4_disaster_collapse", "S6_common_cause")
             for k in range(seeds)]
    out = pd.DataFrame([r for rows in _pool_map(_endo_task, tasks) for r in rows])
    out.to_csv(os.path.join(RESULTS, "endogeneity_check.csv"), index=False)
    return out


# ---------------------------------------------------------------------------
# 3. model misspecification (R3.1, R1.3)
# ---------------------------------------------------------------------------
MISSPEC = dict(family="gamma_wear", obs_family="misspecified")


def _misspec_task(args):
    scen, seed = args
    rows = []
    for policy in ("reactive", "protection", "degradation_aware", "proactive_ai"):
        r = _run(scen, policy, seed, **MISSPEC).run()
        r.update(world="gamma_wear+misspecified_obs")
        rows.append(r)
    return rows


def run_misspec(seeds: int = 24, episodes: int = 6) -> pd.DataFrame:
    """Train on the jump-diffusion corpus; test on a structurally different
    world: monotone gamma-process wear-out without mean reversion, observed
    through a lagged (4 slots) map with different exponents and doubled noise."""
    from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
    from .robustness import fit_predict, local_threshold, score

    ctx = _CTX
    theta = float(ctx["meta"]["threshold"])
    offline = []
    tgt = select_eligible(build_dataset(episodes_per_scenario=episodes, steps=500,
                                        seed=424242, **MISSPEC))
    X = tgt[ctx["feature_columns"]]
    y = tgt["failure_within_H"].to_numpy()
    p = ctx["model"].predict_proba(X)[:, 1]
    frozen = score(y, p, theta)
    frozen.update(setting="frozen predictor (trained on jump-diffusion)")
    offline.append(frozen)
    th_local = local_threshold(y, p)
    local = score(y, p, th_local)
    local.update(setting="frozen predictor, threshold re-commissioned")
    offline.append(local)
    # reference: the same estimator trained on the target world itself
    tr, va, te = episode_split(tgt, seed=42)
    m = [tgt["episode"].isin(s).to_numpy() for s in (tr, va, te)]
    numeric = [c for c in ctx["feature_columns"] if c != "node_type"]
    yt, pt, tht = fit_predict(tgt, numeric, *m)
    ref = score(yt, pt, tht)
    ref.update(setting="reference: retrained on the misspecified world")
    offline.append(ref)
    pd.DataFrame(offline).to_csv(os.path.join(RESULTS, "misspec_offline.csv"), index=False)
    print(pd.DataFrame(offline)[["setting", "prevalence", "pr_auc", "lift", "recall",
                                 "precision"]].to_string(index=False), flush=True)

    tasks = [(scen, 8000 + 43 * k) for scen in ("S3_leo_degradation", "S4_disaster_collapse")
             for k in range(seeds)]
    out = pd.DataFrame([r for rows in _pool_map(_misspec_task, tasks) for r in rows])
    out.to_csv(os.path.join(RESULTS, "misspec_closed_loop.csv"), index=False)
    return out


# ---------------------------------------------------------------------------
# 4. scale: 98-element topology with three LEO planes (R1.2)
# ---------------------------------------------------------------------------
def _large_task(args):
    scen, seed = args
    rows = []
    for policy in ("reactive", "protection", "degradation_aware", "proactive_ai"):
        r = _run(scen, policy, seed, topology="large").run()
        rows.append(r)
    print(f"large {scen} {seed} done", flush=True)
    return rows


def run_large(seeds: int = 12) -> pd.DataFrame:
    tasks = [(scen, 5500 + 53 * k)
             for scen in ("S3_leo_degradation", "S4_disaster_collapse", "S6_common_cause")
             for k in range(seeds)]
    out = pd.DataFrame([r for rows in _pool_map(_large_task, tasks) for r in rows])
    out.to_csv(os.path.join(RESULTS, "large_topology.csv"), index=False)
    return out


# ---------------------------------------------------------------------------
# 5. controller design sensitivity: budget profile and Eq. (11) coefficients (R1.6)
# ---------------------------------------------------------------------------
DESIGN_SETTINGS = (
    [(f"b_c {name}", dict(budget_factors=f)) for name, f in BUDGET_PROFILES.items()]
    + [("loss coef 2", dict(score_coef={"loss": 2.0})),
       ("loss coef 8", dict(score_coef={"loss": 8.0})),
       ("risk base 1.5", dict(score_coef={"risk0": 1.5})),
       ("risk base 6", dict(score_coef={"risk0": 6.0})),
       ("criticality slope 0", dict(score_coef={"risk1": 0.0})),
       ("criticality slope 1.2", dict(score_coef={"risk1": 1.2}))]
)


def _design_task(args):
    name, seed = args
    kw = dict(DESIGN_SETTINGS)[name]
    r = _run("S4_disaster_collapse", "proactive_ai", seed, **kw).run()
    r["setting"] = name
    return [r]


def run_design(seeds: int = 24) -> pd.DataFrame:
    tasks = [(name, 7000 + 61 * k) for name, _ in DESIGN_SETTINGS for k in range(seeds)]
    rows = [r for rows in _pool_map(_design_task, tasks) for r in rows]
    ref = _pool_map(_ref_task, [7000 + 61 * k for k in range(seeds)])
    for r in ref:
        r["setting"] = "P1 reference"
    out = pd.DataFrame(rows + ref)
    out.to_csv(os.path.join(RESULTS, "design_sensitivity.csv"), index=False)
    return out


def _ref_task(seed):
    return _run("S4_disaster_collapse", "reactive", seed).run()


# ---------------------------------------------------------------------------
# 6. resilience / actuation frontier (R3.4, R5.2): compare the risk sources at
#    matched migration effort rather than at a single operating point
# ---------------------------------------------------------------------------
FRONTIER_KAPPA = (0.75, 1.0, 1.5, 2.0, 3.0)
FRONTIER_KPI = (1.1, 1.245, 1.5, 2.0, 3.0)
FRONTIER_SCEN = ("S3_leo_degradation", "S4_disaster_collapse", "S5_leo_handover", "S6_common_cause")


def frontier_tasks(seeds: int = 12):
    t = []
    for scen in FRONTIER_SCEN:
        for k in range(seeds):
            seed = 1000 + 37 * k
            for pol in ("degradation_aware", "proactive_ai", "oracle_calibrated"):
                for kap in FRONTIER_KAPPA:
                    t.append((scen, seed, pol, "kappa", kap))
            for ks in FRONTIER_KPI:
                t.append((scen, seed, "proactive_threshold", "kpi_scale", ks))
    return t


def _frontier_task(args):
    scen, seed, pol, knob, val = args
    kw = {"budget_scale": val} if knob == "kappa" else {"kpi_scale": val}
    r = _run(scen, pol, seed, **kw).run()
    r.update(knob=knob, knob_value=val)
    return [r]


def run_frontier(seeds: int = 12) -> pd.DataFrame:
    out = pd.DataFrame([r for rows in _pool_map(_frontier_task, frontier_tasks(seeds)) for r in rows])
    out.to_csv(os.path.join(RESULTS, "frontier.csv"), index=False)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["feedback", "misspec", "large", "design", "endogeneity",
                                        "frontier"])
    ap.add_argument("--seeds", type=int, default=None)
    args = ap.parse_args()
    global _CTX
    _CTX = prepare_context(DATA_DEFAULT)
    fn = {"feedback": run_feedback, "misspec": run_misspec, "large": run_large,
          "design": run_design, "endogeneity": run_endogeneity,
          "frontier": run_frontier}[args.command]
    out = fn(seeds=args.seeds) if args.seeds else fn()
    print(out.shape)


if __name__ == "__main__":
    main()
