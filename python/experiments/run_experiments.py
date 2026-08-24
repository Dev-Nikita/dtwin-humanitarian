"""
Full experimental campaign.

    python -m python.experiments.run_experiments --seeds 10 --steps 500

Produces
    data/results/experiment_runs.csv        one row per (scenario, policy, seed)
    data/results/experiment_timeline.csv    per-step aggregates (seed 0 only)
    data/results/policy_comparison.csv      aggregated mean +/- std per policy
    data/results/significance_tests.csv     paired Wilcoxon / t-test vs reactive
    data/results/threshold_sweep.csv        sensitivity to the risk budget
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

from ..predictor.features import feature_columns, select_eligible
from ..simulator.config import EPISODE_STEPS, SCENARIO_NAMES
from .runner import ExperimentRun, load_predictor

RESULTS = os.path.join("data", "results")
POLICIES = ["static", "reactive", "proactive_threshold", "degradation_aware",
            "proactive_ai", "oracle"]
KEY_METRICS = [
    "service_availability",
    "resilience_loss",
    "normalised_resilience",
    "sla_compliance",
    "mean_latency_ms",
    "mean_packet_loss",
    "mean_recovery_steps",
    "avail_emergency_message",
    "success_emergency_message",
    "avail_medical_telemetry",
    "success_command_control",
]


def prepare_context(data_path: str) -> Dict:
    df = pd.read_parquet(data_path) if data_path.endswith(".parquet") else pd.read_csv(data_path)
    df = select_eligible(df)
    numeric, categorical = feature_columns(df)
    medians = df.groupby("node_type")["latency_ms"].median().to_dict()
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, "node_type_latency_median.json"), "w") as fh:
        json.dump({k: float(v) for k, v in medians.items()}, fh, indent=2)
    with open(os.path.join(RESULTS, "feature_columns.json"), "w") as fh:
        json.dump(numeric + categorical, fh, indent=2)
    model, meta = load_predictor()
    return {
        "model": model,
        "meta": meta,
        "feature_columns": numeric + categorical,
        "latency_medians": {k: float(v) for k, v in medians.items()},
    }


def run_campaign(
    ctx: Dict, seeds: int, steps: int, scenarios: Optional[List[str]] = None,
    seed_offset: int = 0, policies: Optional[List[str]] = None,
) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
    rows: List[Dict] = []
    timelines: List[pd.DataFrame] = []
    survivals: List[pd.DataFrame] = []
    for scenario in scenarios or SCENARIO_NAMES:
        for seed_idx in range(seed_offset, seed_offset + seeds):
            seed = 1000 + 37 * seed_idx
            for policy in (policies or POLICIES):
                run = ExperimentRun(
                    scenario=scenario,
                    policy=policy,
                    seed=seed,
                    steps=steps,
                    model=ctx["model"],
                    feature_columns=ctx["feature_columns"],
                    latency_medians=ctx["latency_medians"],
                    theta=float(ctx["meta"]["threshold"]),
                )
                summary = run.run()
                rows.append(summary)
                if seed_idx == seed_offset:
                    timelines.append(run.timeline_frame())
                if seed_idx - seed_offset < 3:
                    survivals.append(pd.DataFrame({
                        "scenario": scenario, "policy": policy, "seed": seed,
                        "duration": [r.duration for r in run.survival],
                        "event": [r.event for r in run.survival],
                        "duration_sustained": [r.duration for r in run.survival_sustained],
                        "event_sustained": [r.event for r in run.survival_sustained],
                    }))
            print(
                f"{scenario:22s} seed={seed} "
                + "  ".join(
                    f"{r['policy'][:9]}:A={r['service_availability']:.4f}"
                    for r in rows[-len(policies or POLICIES) :]
                )
            )
    tl = pd.concat(timelines, ignore_index=True) if timelines else pd.DataFrame()
    sv = pd.concat(survivals, ignore_index=True) if survivals else pd.DataFrame()
    return pd.DataFrame(rows), tl, sv


def aggregate(runs: pd.DataFrame) -> pd.DataFrame:
    metrics = [c for c in runs.columns if runs[c].dtype != object and c not in ("seed", "steps")]
    agg = runs.groupby(["scenario", "policy"])[metrics].agg(["mean", "std"])
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    return agg.reset_index()


PRIMARY_METRICS = ["service_availability", "resilience_loss",
                  "normalised_resilience", "mean_recovery_steps"]
PRIMARY_POLICY = "proactive_ai"
STRESSED = ["S2_bs_overload", "S3_leo_degradation", "S4_disaster_collapse",
            "S5_leo_handover", "S6_common_cause"]


def significance(runs: pd.DataFrame, reference: str = "reactive") -> pd.DataFrame:
    """Paired comparison of every policy against the reactive baseline.

    The primary inferential family contains only the primary
    resilience measures of the proposed policy in the five stressed scenarios
    (|F| = 20). Holm correction is applied to that family alone; every other
    comparison is reported as exploratory with uncorrected p-values, which is
    the standard way of avoiding a correction so conservative that it hides the
    effects it was meant to protect.
    """
    rows = []
    for scenario in list(runs["scenario"].unique()) + ["ALL"]:
        sub = runs if scenario == "ALL" else runs[runs["scenario"] == scenario]
        base = sub[sub["policy"] == reference].sort_values("seed")
        for policy in POLICIES:
            if policy == reference:
                continue
            cand = sub[sub["policy"] == policy].sort_values("seed")
            for metric in KEY_METRICS:
                if metric not in sub.columns:
                    continue
                a = base[metric].to_numpy(dtype=float)
                b = cand[metric].to_numpy(dtype=float)
                n = min(len(a), len(b))
                a, b = a[:n], b[:n]
                if n < 3 or np.allclose(a, b):
                    continue
                diff = b - a
                t_stat, t_p = stats.ttest_rel(b, a)
                try:
                    w_stat, w_p = stats.wilcoxon(b, a)
                except ValueError:
                    w_stat, w_p = np.nan, np.nan
                sd = diff.std(ddof=1)
                se = sd / np.sqrt(n) if n else np.nan
                half = stats.t.ppf(0.975, n - 1) * se if n > 1 else np.nan
                rows.append({
                    "scenario": scenario,
                    "policy": policy,
                    "reference": reference,
                    "metric": metric,
                    "family": ("confirmatory"
                               if (policy == PRIMARY_POLICY and metric in PRIMARY_METRICS
                                   and scenario in STRESSED)
                               else "exploratory"),
                    "mean_reference": a.mean(),
                    "mean_policy": b.mean(),
                    "mean_difference": diff.mean(),
                    "ci95_low": diff.mean() - half,
                    "ci95_high": diff.mean() + half,
                    "relative_change_pct": (100.0 * (b.mean() - a.mean()) / abs(a.mean())
                                            if a.mean() != 0 else np.nan),
                    "t_stat": t_stat,
                    "p_value_ttest": t_p,
                    "wilcoxon_stat": w_stat,
                    "p_value_wilcoxon": w_p,
                    "cohens_dz": diff.mean() / sd if sd > 0 else np.nan,
                    "n_pairs": n,
                })
    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df["p_holm"] = np.nan
    conf = df["family"] == "confirmatory"
    if conf.any():                       # Holm-Bonferroni on the confirmatory family
        idx = df[conf].sort_values("p_value_ttest").index
        m = len(idx)
        raw = df.loc[idx, "p_value_ttest"].to_numpy()
        adj = np.minimum.accumulate((raw * (m - np.arange(m)))[::-1])[::-1].clip(0, 1)
        df.loc[idx, "p_holm"] = adj
        df.loc[idx, "family_size"] = m
    df["significant_005"] = np.where(conf, df["p_holm"] < 0.05, df["p_value_ttest"] < 0.05)
    return df.sort_values(["family", "p_value_ttest"]).reset_index(drop=True)


def threshold_sweep(ctx: Dict, steps: int, seeds: int = 3,
                    scenarios: Optional[List[str]] = None) -> pd.DataFrame:
    """Sensitivity of the proposed policy to the risk-budget scale."""
    rows = []
    for scale in (0.4, 0.6, 0.8, 1.0, 1.5, 2.0, 3.0):
        for scenario in scenarios or SCENARIO_NAMES:
            for seed_idx in range(seeds):
                run = ExperimentRun(
                    scenario=scenario,
                    policy="proactive_ai",
                    seed=2000 + 53 * seed_idx,
                    steps=steps,
                    model=ctx["model"],
                    feature_columns=ctx["feature_columns"],
                    latency_medians=ctx["latency_medians"],
                    theta=float(ctx["meta"]["threshold"]),
                    budget_scale=scale,
                )
                s = run.run()
                s["budget_scale"] = scale
                rows.append(s)
        print(f"sweep scale={scale} done")
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--steps", type=int, default=EPISODE_STEPS)
    ap.add_argument("--data", default=os.path.join("data", "processed", "telemetry_dataset.parquet"))
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--scenarios", nargs="*", default=None)
    ap.add_argument("--append", action="store_true", default=True,
                    help="append to the existing run table (default)")
    ap.add_argument("--overwrite", action="store_true", help="start a fresh run table")
    ap.add_argument("--finalise", action="store_true", help="only recompute aggregates")
    ap.add_argument("--seed-offset", type=int, default=0)
    ap.add_argument("--policies", nargs="*", default=None)
    args = ap.parse_args()

    if args.overwrite:
        args.append = False
    runs_path = os.path.join(RESULTS, "experiment_runs.csv")
    timeline_path = os.path.join(RESULTS, "experiment_timeline.csv")

    if args.finalise:
        runs = pd.read_csv(runs_path)
        aggregate(runs).to_csv(os.path.join(RESULTS, "policy_comparison.csv"), index=False)
        significance(runs).to_csv(os.path.join(RESULTS, "significance_tests.csv"), index=False)
        print(runs.pivot_table(index="scenario", columns="policy",
                               values="service_availability").round(4))
        return

    ctx = prepare_context(args.data)

    if args.sweep:
        sweep_path = os.path.join(RESULTS, "threshold_sweep.csv")
        sw = threshold_sweep(ctx, args.steps, seeds=args.seeds, scenarios=args.scenarios)
        if args.append and os.path.exists(sweep_path):
            sw = pd.concat([pd.read_csv(sweep_path), sw], ignore_index=True)
        sw.to_csv(sweep_path, index=False)
        return

    runs, timeline, survival = run_campaign(
        ctx, args.seeds, args.steps, args.scenarios, args.seed_offset, args.policies)
    survival_path = os.path.join(RESULTS, "survival_records.csv")
    if args.append and os.path.exists(survival_path) and not survival.empty:
        survival = pd.concat([pd.read_csv(survival_path), survival], ignore_index=True)
    if not survival.empty:
        survival.to_csv(survival_path, index=False)
    if args.append and os.path.exists(runs_path):
        runs = pd.concat([pd.read_csv(runs_path), runs], ignore_index=True)
        if os.path.exists(timeline_path) and not timeline.empty:
            timeline = pd.concat([pd.read_csv(timeline_path), timeline], ignore_index=True)
    runs.to_csv(runs_path, index=False)
    if not timeline.empty:
        timeline.to_csv(timeline_path, index=False)
    aggregate(runs).to_csv(os.path.join(RESULTS, "policy_comparison.csv"), index=False)
    significance(runs).to_csv(os.path.join(RESULTS, "significance_tests.csv"), index=False)

    print("\n=== mean service availability ===")
    print(runs.pivot_table(index="scenario", columns="policy", values="service_availability").round(4))
    print("\n=== mean recovery (slots) ===")
    print(runs.pivot_table(index="scenario", columns="policy", values="mean_recovery_steps").round(2))


if __name__ == "__main__":
    main()
