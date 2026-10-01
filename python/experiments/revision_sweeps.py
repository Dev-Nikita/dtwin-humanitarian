"""
Closed-loop sweeps of the revision (JRESS-D-26-06464, round 1), all under the
corrected interruption accounting and with the predictive-protection family.

    python -m python.experiments.revision_sweeps count EXP
    python -m python.experiments.revision_sweeps run EXP LO HI [WORKERS]
    python -m python.experiments.revision_sweeps merge EXP

Every task is a single paired unit (one seed of one configuration); all policies
compared within a configuration share the seed and therefore the exogenous
randomness (common random numbers). Results are written in chunks to
data/results/sweeps/ and merged into data/results/<EXP>.csv.
"""

from __future__ import annotations

import os
import sys
from typing import Dict, List

import numpy as np
import pandas as pd

from ..orchestrator import digital_twin as dt
from ..orchestrator.policies import BUDGET_PROFILES
from ..simulator import failure_generator as fg
from . import runner as RR
from .run_experiments import DATA_DEFAULT, prepare_context

RESULTS = os.path.join("data", "results")
PARTS = os.path.join(RESULTS, "sweeps")
_CTX: Dict = {}

CORE = ("reactive", "protection", "degradation_aware", "proactive_ai",
        "protection_index", "protection_predictive")
KEYS = ("service_availability", "resilience_loss", "normalised_resilience", "sla_compliance",
        "mean_recovery_steps", "mean_recovery_failure", "mean_recovery_migration",
        "failure_interruptions", "migration_interruptions", "outage_slots_failure",
        "outage_slots_migration", "proactive_reroutes_total", "protection_switches",
        "backup_refreshes", "reactive_reroutes", "dead_path_migrations",
        "mig_trigger_already_failed", "mig_trigger_fails_within_H",
        "mig_trigger_no_failure_within_H", "node_failure_slots", "anticipation_rate",
        "lead_time_median", "lead_time_mean", "generated_flows", "success_command_control",
        "success_medical_telemetry", "avail_command_control", "mig_trigger_degraded_no_failure",
        "mig_trigger_false_alarm", "path_computations", "switch_infeasible", "protection_mode")


def _run(scen, policy, seed, **kw):
    ctx = _CTX
    return RR.ExperimentRun(scen, policy, seed, steps=500, model=ctx["model"],
                            feature_columns=ctx["feature_columns"],
                            latency_medians=ctx["latency_medians"],
                            theta=float(ctx["meta"]["threshold"]), **kw)


def _pack(summary, **labels):
    row = {k: summary.get(k) for k in KEYS}
    row.update({"scenario": summary["scenario"], "policy": summary["policy"],
                "seed": summary["seed"]})
    row.update(labels)
    return row


# ---------------------------------------------------------------- task lists
def tasks(exp: str) -> List[tuple]:
    if exp == "misspec_closed_loop":
        return [(s, 8000 + 43 * k) for s in ("S3_leo_degradation", "S4_disaster_collapse")
                for k in range(24)]
    if exp == "large_topology":
        return [(s, 5500 + 53 * k) for s in ("S3_leo_degradation", "S4_disaster_collapse",
                                             "S6_common_cause") for k in range(12)]
    if exp == "feedback_ablation_fixed":
        return [(c, s, 6000 + 29 * k) for c in ("nominal", "strong")
                for s in ("S3_leo_degradation", "S4_disaster_collapse", "S6_common_cause")
                for k in range(12)]
    if exp == "design_sensitivity":
        return [(name, 7000 + 61 * k) for name in DESIGN for k in range(24)]
    if exp == "frontier":
        return [(s, 1000 + 37 * k) for s in ("S3_leo_degradation", "S4_disaster_collapse",
                                             "S5_leo_handover", "S6_common_cause")
                for k in range(12)]
    if exp == "migration_cost_sweep":
        return [(c, s, 4000 + 41 * k) for c in (0, 1, 2, 3, 5, 8)
                for s in ("S3_leo_degradation", "S4_disaster_collapse") for k in range(12)]
    if exp == "topology_sweep":
        return [(t, 5000 + 47 * k) for t in ("sparse", "lean", "reference", "redundant",
                                             "satellite_heavy") for k in range(12)]
    if exp == "parameter_sensitivity":
        return [(name, 7000 + 61 * k) for name in PARAMS for k in range(12)]
    if exp == "controller_weight_sensitivity":
        return [(name, 7000 + 61 * k) for name in WEIGHTS for k in range(12)]
    if exp == "accounting_check":
        return [(f, sc, 1000 + 37 * k) for f in (True, False)
                for sc in ("S3_leo_degradation", "S4_disaster_collapse", "S6_common_cause")
                for k in range(12)]
    if exp == "endogeneity_check":
        return [(c, sc, 6000 + 29 * k) for c in ("nominal", "strong")
                for sc in ("S2_bs_overload", "S4_disaster_collapse", "S6_common_cause")
                for k in range(12)]
    if exp == "protection_modes":
        return [(m, s, 1000 + 37 * k) for m in ("feasible", "reserved", "optimistic")
                for s in ("S4_disaster_collapse", "S6_common_cause") for k in range(24)]
    if exp == "confirmation":
        return [(s, CONFIRM_SEED0 + CONFIRM_STEP * k) for s in CONFIRM_SCENARIOS
                for k in range(CONFIRM_N)]
    if exp == "p6_decomposition":
        # Post-confirmation exploratory ablation on the confirmation seeds.
        return [(s, CONFIRM_SEED0 + CONFIRM_STEP * k)
                for s in ("S4_disaster_collapse", "S6_common_cause") for k in range(CONFIRM_N)]
    if exp == "detection_delay":
        return [(d, s, 1000 + 37 * k) for d in (0, 1, 3, 5)
                for s in ("S4_disaster_collapse", "S6_common_cause") for k in range(12)]
    raise KeyError(exp)


DESIGN = {f"b_c {n}": dict(budget_factors=f) for n, f in BUDGET_PROFILES.items()}
DESIGN.update({
    "loss coef 2": dict(score_coef={"loss": 2.0}),
    "loss coef 8": dict(score_coef={"loss": 8.0}),
    "risk base 1.5": dict(score_coef={"risk0": 1.5}),
    "risk base 6": dict(score_coef={"risk0": 6.0}),
    "criticality slope 0": dict(score_coef={"risk1": 0.0}),
    "criticality slope 1.2": dict(score_coef={"risk1": 1.2}),
})
PARAMS = {
    "baseline": dict(),
    "lambda0 x0.5": dict(hazard_scale=0.5),
    "lambda0 x2.0": dict(hazard_scale=2.0),
    "sigma x0.5": dict(sigma_scale=0.5),
    "sigma x2.0": dict(sigma_scale=2.0),
    "feedback gain x0": dict(feedback_gain_scale=0.0),
    "feedback gain x2": dict(feedback_gain_scale=2.0),
}
WEIGHTS = {
    "W_RISK x0.5": ("W_RISK", 0.5), "W_RISK x2.0": ("W_RISK", 2.0),
    "W_LOSS x0.5": ("W_LOSS", 0.5), "W_LOSS x2.0": ("W_LOSS", 2.0),
    "W_LOAD x0.5": ("W_LOAD", 0.5), "W_LOAD x2.0": ("W_LOAD", 2.0),
}
COUPLINGS = {"nominal": (1.0, 0.65), "strong": (4.0, 0.40)}
# Frozen confirmation experiment (R1_CONFIRMATION_PROTOCOL.md). These seeds were
# never used for development or analysis before the protocol was committed.
CONFIRM_SEED0, CONFIRM_STEP, CONFIRM_N = 20000, 101, 64
CONFIRM_SCENARIOS = ("S3_leo_degradation", "S4_disaster_collapse", "S5_leo_handover",
                     "S6_common_cause")
CONFIRM_POLICIES = ("reactive", "protection", "degradation_aware", "proactive_ai",
                    "protection_index", "protection_predictive", "protection_oracle")
FRONTIER_KAPPA = (0.75, 1.0, 1.5, 2.0, 3.0)
FRONTIER_KPI = (1.1, 1.245, 1.5, 2.0, 3.0)


# ---------------------------------------------------------------- executors
def execute(exp: str, task: tuple) -> List[dict]:
    rows = []
    if exp == "misspec_closed_loop":
        scen, seed = task
        for pol in CORE:
            rows.append(_pack(_run(scen, pol, seed, family="gamma_wear",
                                   obs_family="misspecified").run(),
                              world="gamma_wear+misspecified_obs"))
    elif exp == "large_topology":
        scen, seed = task
        for pol in CORE:
            rows.append(_pack(_run(scen, pol, seed, topology="large").run()))
    elif exp == "feedback_ablation_fixed":
        coupling, scen, seed = task
        gain, knee = COUPLINGS[coupling]
        fg.LOAD_HAZARD_KNEE = knee
        try:
            ref = _run(scen, "reactive", seed, feedback_gain_scale=gain)
            rows.append(_pack(ref.run(), coupling=coupling, feedback="endogenous"))
            trace = ref.load_trace
            for pol in ("degradation_aware", "proactive_ai", "protection",
                        "protection_predictive"):
                for mode in ("endogenous", "replayed_P1_load"):
                    kw = dict(feedback_gain_scale=gain)
                    if mode == "replayed_P1_load":
                        kw["load_trace"] = trace
                    rows.append(_pack(_run(scen, pol, seed, **kw).run(),
                                      coupling=coupling, feedback=mode))
        finally:
            fg.LOAD_HAZARD_KNEE = COUPLINGS["nominal"][1]
    elif exp == "design_sensitivity":
        name, seed = task
        for pol in ("proactive_ai", "protection_predictive"):
            rows.append(_pack(_run("S4_disaster_collapse", pol, seed, **DESIGN[name]).run(),
                              setting=name))
        if name == "b_c nominal":
            for pol in ("reactive", "protection"):
                rows.append(_pack(_run("S4_disaster_collapse", pol, seed).run(),
                                  setting=f"{pol} reference"))
    elif exp == "frontier":
        scen, seed = task
        for pol in ("degradation_aware", "proactive_ai", "oracle_calibrated",
                    "protection_predictive"):
            for kap in FRONTIER_KAPPA:
                rows.append(_pack(_run(scen, pol, seed, budget_scale=kap).run(),
                                  knob="kappa", knob_value=kap))
        for ks in FRONTIER_KPI:
            rows.append(_pack(_run(scen, "proactive_threshold", seed, kpi_scale=ks).run(),
                              knob="kpi_scale", knob_value=ks))
        for pol in ("reactive", "protection"):
            rows.append(_pack(_run(scen, pol, seed).run(), knob="none", knob_value=0))
    elif exp == "migration_cost_sweep":
        cost, scen, seed = task
        for pol in ("proactive_ai", "protection", "protection_predictive"):
            rows.append(_pack(_run(scen, pol, seed, migration_cost=(cost, cost)).run(),
                              migration_cost=cost))
        if cost == 0:
            rows.append(_pack(_run(scen, "reactive", seed).run(), migration_cost=-1))
    elif exp == "topology_sweep":
        from ..simulator.topology_variants import build_variant, redundancy_descriptors
        topo, seed = task
        desc = redundancy_descriptors(build_variant(topo))
        for pol in ("reactive", "protection", "proactive_ai", "protection_predictive"):
            r = _pack(_run("S4_disaster_collapse", pol, seed, topology=topo).run(),
                      topology=topo)
            r.update({f"topo_{k}": v for k, v in desc.items()})
            rows.append(r)
    elif exp == "parameter_sensitivity":
        name, seed = task
        for pol in ("reactive", "protection", "proactive_ai", "protection_predictive"):
            rows.append(_pack(_run("S4_disaster_collapse", pol, seed, **PARAMS[name]).run(),
                              setting=name))
    elif exp == "controller_weight_sensitivity":
        name, seed = task
        key, factor = WEIGHTS[name]
        base = getattr(dt, key)
        setattr(dt, key, base * factor)
        try:
            for pol in ("reactive", "proactive_ai", "protection_predictive"):
                rows.append(_pack(_run("S4_disaster_collapse", pol, seed).run(), setting=name))
        finally:
            setattr(dt, key, base)
    elif exp == "accounting_check":
        flag, scen, seed = task
        base = RR.DEAD_PATH_MIGRATION_COST
        RR.DEAD_PATH_MIGRATION_COST = flag
        try:
            pols = ("reactive", "protection") if flag else ()
            pols += ("degradation_aware", "proactive_ai", "proactive_threshold_matched",
                     "oracle_calibrated")
            for pol in pols:
                rows.append(_pack(_run(scen, pol, seed).run(), flag=flag))
        finally:
            RR.DEAD_PATH_MIGRATION_COST = base
    elif exp == "endogeneity_check":
        coupling, scen, seed = task
        gain, knee = COUPLINGS[coupling]
        fg.LOAD_HAZARD_KNEE = knee
        try:
            for pol in ("static", "reactive", "proactive_ai", "protection_predictive"):
                r = _run(scen, pol, seed, feedback_gain_scale=gain)
                summ = r.run()
                util = np.array([v for m in r.load_trace for v in m.values()])
                row = _pack(summ, coupling=coupling)
                row["util_above_knee"] = float((util > knee).mean())
                rows.append(row)
        finally:
            fg.LOAD_HAZARD_KNEE = COUPLINGS["nominal"][1]
    elif exp == "protection_modes":
        mode, scen, seed = task
        for pol in ("reactive", "protection", "protection_index", "protection_predictive"):
            rows.append(_pack(_run(scen, pol, seed, protection_mode=mode).run(),
                              protection_mode_setting=mode))
    elif exp == "confirmation":
        scen, seed = task
        for pol in CONFIRM_POLICIES:
            rows.append(_pack(_run(scen, pol, seed).run()))
    elif exp == "p6_decomposition":
        scen, seed = task
        for pol in ("protection_predictive", "protection_predictive_route",
                    "protection_predictive_standby"):
            rows.append(_pack(_run(scen, pol, seed).run()))
    elif exp == "detection_delay":
        d, scen, seed = task
        base = RR.DETECTION_DELAY
        RR.DETECTION_DELAY = d
        try:
            for pol in ("reactive", "protection", "degradation_aware", "proactive_ai",
                        "oracle_calibrated", "protection_predictive"):
                rows.append(_pack(_run(scen, pol, seed).run(), detection_delay=d))
        finally:
            RR.DETECTION_DELAY = base
    return rows


def _worker(args):
    exp, task = args
    return execute(exp, task)


def main() -> None:
    cmd, exp = sys.argv[1], sys.argv[2]
    if cmd == "count":
        print(len(tasks(exp)))
        return
    if cmd == "merge":
        files = sorted(f for f in os.listdir(PARTS) if f.startswith(exp + "__"))
        out = pd.concat([pd.read_csv(os.path.join(PARTS, f)) for f in files], ignore_index=True)
        n_task = len(tasks(exp))
        covered = set()
        for f in files:
            lo, hi = f[len(exp) + 2:-4].split("_")
            covered |= set(range(int(lo), int(hi)))
        assert covered == set(range(n_task)), sorted(set(range(n_task)) - covered)
        out.to_csv(os.path.join(RESULTS, exp + ".csv"), index=False)
        print(exp, out.shape)
        return
    lo, hi = int(sys.argv[3]), int(sys.argv[4])
    workers = int(sys.argv[5]) if len(sys.argv) > 5 else (os.cpu_count() or 1)
    global _CTX
    _CTX = prepare_context(DATA_DEFAULT)
    import multiprocessing as mp
    t = [(exp, x) for x in tasks(exp)[lo:hi]]
    with mp.get_context("fork").Pool(workers) as pool:
        res = pool.map(_worker, t, chunksize=1)
    os.makedirs(PARTS, exist_ok=True)
    pd.DataFrame([r for rows in res for r in rows]).to_csv(
        os.path.join(PARTS, f"{exp}__{lo:03d}_{hi:03d}.csv"), index=False)
    print("ok", exp, lo, hi, flush=True)


if __name__ == "__main__":
    main()
