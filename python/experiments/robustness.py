"""
Robustness experiments.

    python -m python.experiments.robustness ablation      feature ablation M0-M3
    python -m python.experiments.robustness ood           leave-one-scenario-out + held-out S5/S6
    python -m python.experiments.robustness horizon       prediction-horizon sweep
    python -m python.experiments.robustness migration     migration-cost sweep (closed loop)

Each sub-command writes one CSV into data/results and prints a summary.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ..predictor.features import episode_split, feature_columns, select_eligible
from ..simulator.config import FEATURE_WINDOW, TRAIN_SCENARIOS
from ..simulator.dataset_generator import add_labels
from .runner import ExperimentRun

RESULTS = os.path.join("data", "results")
DATA = os.path.join("data", "processed", "telemetry_dataset.parquet")
HELDOUT_DATA = os.path.join("data", "processed", "heldout_dataset.parquet")

RAW_KPIS = [
    "latency_ms", "jitter_ms", "packet_loss", "throughput_mbps", "cpu_usage",
    "memory_usage", "signal_strength", "availability", "utilisation",
    "queue_depth", "retx_rate", "energy_headroom",
]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def load(path: str = DATA) -> pd.DataFrame:
    df = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)
    return select_eligible(df)


def feature_subset(df: pd.DataFrame, variant: str) -> List[str]:
    """Column subsets defining the ablation ladder."""
    numeric, _ = feature_columns(df)
    w = FEATURE_WINDOW
    if variant == "M0_instantaneous":
        return [c for c in numeric if c in RAW_KPIS]
    if variant == "M1_rolling":
        return [c for c in numeric
                if c in RAW_KPIS or c.endswith(f"_mean{w}") or c.endswith(f"_std{w}")]
    return numeric  # M2 / M3: full temporal representation


def fit_predict(
    df: pd.DataFrame,
    cols: List[str],
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    test_mask: np.ndarray,
    calibrate: bool = True,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Fit logistic regression on the given columns; return (y_test, prob, theta)."""
    prep = ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                          ("scale", StandardScaler())]), cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), ["node_type"]),
    ])
    pipe = Pipeline([("prep", prep),
                     ("model", LogisticRegression(max_iter=2000, C=0.8,
                                                  class_weight="balanced", n_jobs=-1))])
    X = df[cols + ["node_type"]]
    y = df["failure_within_H"].to_numpy()
    pipe.fit(X[train_mask], y[train_mask])

    model = pipe
    if calibrate and val_mask.sum() > 0:
        model = CalibratedClassifierCV(pipe, method="isotonic", cv="prefit")
        model.fit(X[val_mask], y[val_mask])

    p_val = model.predict_proba(X[val_mask])[:, 1] if val_mask.sum() else None
    theta = 0.2
    if p_val is not None:
        grid = np.linspace(0.02, 0.95, 187)
        theta = float(grid[int(np.argmax(
            [f1_score(y[val_mask], (p_val >= t).astype(int), zero_division=0) for t in grid]))])
    p_test = model.predict_proba(X[test_mask])[:, 1]
    return y[test_mask], p_test, theta


def local_threshold(y: np.ndarray, p: np.ndarray, frac: float = 0.2,
                    seed: int = 11) -> float:
    """Operating point re-selected on a small commissioning sample drawn from the
    target regime. This emulates the field procedure of re-tuning the alarm level
    after deployment, without re-training the model."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))[: max(50, int(frac * len(y)))]
    grid = np.linspace(0.01, 0.95, 187)
    scores = [f1_score(y[idx], (p[idx] >= t).astype(int), zero_division=0) for t in grid]
    return float(grid[int(np.argmax(scores))])


def score(y: np.ndarray, p: np.ndarray, theta: float) -> Dict[str, float]:
    pred = (p >= theta).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "prevalence": float(y.mean()),
        "roc_auc": float(roc_auc_score(y, p)) if 0 < y.mean() < 1 else float("nan"),
        "pr_auc": float(average_precision_score(y, p)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "brier": float(brier_score_loss(y, p)),
        "false_alarm_rate": float(fp / max(1, fp + tn)),
        "lift": float(average_precision_score(y, p) / max(1e-9, y.mean())),
        "threshold": theta,
        "n_test": int(len(y)),
    }


# ---------------------------------------------------------------------------
# 1. feature ablation
# ---------------------------------------------------------------------------
def run_ablation() -> pd.DataFrame:
    df = load()
    tr, va, te = episode_split(df, seed=42)
    m_tr = df["episode"].isin(tr).to_numpy()
    m_va = df["episode"].isin(va).to_numpy()
    m_te = df["episode"].isin(te).to_numpy()

    variants = [
        ("M0_instantaneous", "instantaneous KPIs only", False),
        ("M1_rolling", "+ rolling mean and standard deviation", False),
        ("M2_trend", "+ slopes, first differences and ratios", False),
        ("M3_calibrated", "+ isotonic calibration", True),
    ]
    rows = []
    for name, description, calibrate in variants:
        cols = feature_subset(df, name)
        y, p, theta = fit_predict(df, cols, m_tr, m_va, m_te, calibrate=calibrate)
        row = {"variant": name, "description": description, "n_features": len(cols) + 1}
        row.update(score(y, p, theta))
        rows.append(row)
        print(f"{name:18s} feats={len(cols)+1:3d} PR-AUC={row['pr_auc']:.4f} "
              f"recall={row['recall']:.3f} Brier={row['brier']:.5f}")
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(RESULTS, "feature_ablation.csv"), index=False)
    return out


# ---------------------------------------------------------------------------
# 2. out-of-distribution evaluation
# ---------------------------------------------------------------------------
def run_ood() -> pd.DataFrame:
    df = load()
    numeric, _ = feature_columns(df)
    rows = []

    # (a) leave-one-scenario-out within the training family
    for held in TRAIN_SCENARIOS:
        m_te = (df["scenario"] == held).to_numpy()
        rest = df[~m_te]
        val_eps = set(rest["episode"].drop_duplicates().sample(frac=0.25, random_state=3))
        m_va = df["episode"].isin(val_eps).to_numpy() & ~m_te
        m_tr = ~m_te & ~m_va
        y, p, theta = fit_predict(df, numeric, m_tr, m_va, m_te)
        row = {"setting": "leave_one_scenario_out", "held_out": held}
        row.update(score(y, p, theta))
        if y.sum() > 0:
            th_local = local_threshold(y, p)
            local = score(y, p, th_local)
            row.update({"recall_recalibrated": local["recall"],
                        "f1_recalibrated": local["f1"],
                        "threshold_recalibrated": th_local})
        rows.append(row)
        print(f"LOSO {held:22s} PR-AUC={row['pr_auc']:.4f} recall={row['recall']:.3f} "
              f"lift={row['lift']:.1f}")

    # (b) unseen degradation mechanisms (S5, S6)
    if os.path.exists(HELDOUT_DATA):
        held_df = load(HELDOUT_DATA)
        tr, va, _ = episode_split(df, seed=42)
        m_tr = df["episode"].isin(tr).to_numpy()
        m_va = df["episode"].isin(va).to_numpy()
        combined = pd.concat([df, held_df], ignore_index=True)
        n_train = len(df)
        mask_tr = np.zeros(len(combined), dtype=bool)
        mask_va = np.zeros(len(combined), dtype=bool)
        mask_tr[:n_train] = m_tr
        mask_va[:n_train] = m_va
        for scen in sorted(held_df["scenario"].unique()):
            mask_te = np.zeros(len(combined), dtype=bool)
            mask_te[n_train:] = (held_df["scenario"] == scen).to_numpy()
            y, p, theta = fit_predict(combined, numeric, mask_tr, mask_va, mask_te)
            row = {"setting": "unseen_mechanism", "held_out": scen}
            row.update(score(y, p, theta))
            if y.sum() > 0:
                th_local = local_threshold(y, p)
                local = score(y, p, th_local)
                row.update({"recall_recalibrated": local["recall"],
                            "f1_recalibrated": local["f1"],
                            "threshold_recalibrated": th_local})
            rows.append(row)
            print(f"OOD  {scen:22s} PR-AUC={row['pr_auc']:.4f} recall={row['recall']:.3f} "
                  f"lift={row['lift']:.1f}")

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(RESULTS, "ood_evaluation.csv"), index=False)
    return out


# ---------------------------------------------------------------------------
# 3. prediction-horizon sweep
# ---------------------------------------------------------------------------
def run_horizon(horizons: Tuple[int, ...] = (3, 5, 10, 15, 20),
                closed_loop_seeds: int = 6, append: bool = False) -> pd.DataFrame:
    raw = pd.read_parquet(DATA)
    numeric, _ = feature_columns(select_eligible(raw))
    rows = []
    ctx_medians = json.load(open(os.path.join(RESULTS, "node_type_latency_median.json")))

    for H in horizons:
        relabelled = select_eligible(add_labels(raw.copy(), horizon=H))
        tr, va, te = episode_split(relabelled, seed=42)
        m_tr = relabelled["episode"].isin(tr).to_numpy()
        m_va = relabelled["episode"].isin(va).to_numpy()
        m_te = relabelled["episode"].isin(te).to_numpy()
        y, p, theta = fit_predict(relabelled, numeric, m_tr, m_va, m_te)
        row = {"horizon": H}
        row.update(score(y, p, theta))

        # closed-loop consequence of that horizon
        prep = ColumnTransformer([
            ("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                              ("scale", StandardScaler())]), numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore"), ["node_type"]),
        ])
        pipe = Pipeline([("prep", prep),
                         ("model", LogisticRegression(max_iter=2000, C=0.8,
                                                      class_weight="balanced", n_jobs=-1))])
        X = relabelled[numeric + ["node_type"]]
        yy = relabelled["failure_within_H"].to_numpy()
        pipe.fit(X[m_tr | m_va], yy[m_tr | m_va])
        model = CalibratedClassifierCV(pipe, method="isotonic", cv="prefit")
        model.fit(X[m_va], yy[m_va])

        avail, rec, lead = [], [], []
        for k in range(closed_loop_seeds):
            run = ExperimentRun("S4_disaster_collapse", "proactive_ai", 3000 + 31 * k,
                                steps=400, model=model, feature_columns=numeric + ["node_type"],
                                latency_medians=ctx_medians, theta=theta)
            s = run.run()
            avail.append(s["service_availability"])
            rec.append(s["mean_recovery_steps"])
            lead.append(s.get("lead_time_median", np.nan))
        row.update({
            "cl_availability": float(np.mean(avail)),
            "cl_recovery": float(np.mean(rec)),
            "cl_lead_time_median": float(np.nanmean(lead)),
        })
        rows.append(row)
        print(f"H={H:2d} PR-AUC={row['pr_auc']:.4f} recall={row['recall']:.3f} "
              f"avail={row['cl_availability']:.4f} rec={row['cl_recovery']:.2f}")

    out = pd.DataFrame(rows)
    path = os.path.join(RESULTS, "horizon_sweep.csv")
    if append and os.path.exists(path):
        out = pd.concat([pd.read_csv(path), out], ignore_index=True)
    out.to_csv(path, index=False)
    return out


# ---------------------------------------------------------------------------
# 4. migration-cost sweep
# ---------------------------------------------------------------------------
def run_migration(costs: Tuple[int, ...] = (0, 1, 2, 3, 5, 8),
                  scenarios: Optional[List[str]] = None,
                  seeds: int = 6, append: bool = False) -> pd.DataFrame:
    from .run_experiments import prepare_context

    ctx = prepare_context(DATA)
    theta = float(ctx["meta"]["threshold"])
    scenarios = scenarios or ["S3_leo_degradation", "S4_disaster_collapse"]
    rows = []
    for cost in costs:
        for scen in scenarios:
            for k in range(seeds):
                run = ExperimentRun(
                    scen, "proactive_ai", 4000 + 41 * k, steps=400,
                    model=ctx["model"], feature_columns=ctx["feature_columns"],
                    latency_medians=ctx["latency_medians"], theta=theta,
                    migration_cost=(cost, cost),
                )
                s = run.run()
                s["migration_cost"] = cost
                rows.append(s)
            # reactive reference under the same conditions (cost-independent)
            if cost == costs[0]:
                for k in range(seeds):
                    run = ExperimentRun(scen, "reactive", 4000 + 41 * k, steps=400,
                                        model=None, feature_columns=None,
                                        latency_medians=ctx["latency_medians"])
                    s = run.run()
                    s["migration_cost"] = -1
                    rows.append(s)
        print(f"migration cost {cost} done")
    out = pd.DataFrame(rows)
    path = os.path.join(RESULTS, "migration_cost_sweep.csv")
    if append and os.path.exists(path):
        out = pd.concat([pd.read_csv(path), out], ignore_index=True)
    out.to_csv(path, index=False)
    return out


# ---------------------------------------------------------------------------
# 5b. topology redundancy: where does anticipation pay off?
# ---------------------------------------------------------------------------
def run_topology(scenarios: Optional[List[str]] = None, seeds: int = 8,
                 steps: int = 400, append: bool = False,
                 topologies: Optional[List[str]] = None) -> pd.DataFrame:
    """Resilience gain of the proposed policy as a function of structural
    redundancy. Only the graph changes; elements, traffic and degradation are
    identical across classes."""
    from .run_experiments import prepare_context
    from ..simulator.topology_variants import (
        TOPOLOGY_CLASSES, build_variant, redundancy_descriptors,
    )

    ctx = prepare_context(DATA)
    theta = float(ctx["meta"]["threshold"])
    scenarios = scenarios or ["S4_disaster_collapse"]
    rows = []
    classes = topologies or TOPOLOGY_CLASSES
    for cls in classes:
        desc = redundancy_descriptors(build_variant(cls))
        for scen in scenarios:
            for policy in ("reactive", "proactive_ai"):
                for k in range(seeds):
                    run = ExperimentRun(
                        scen, policy, 5000 + 47 * k, steps=steps,
                        model=ctx["model"], feature_columns=ctx["feature_columns"],
                        latency_medians=ctx["latency_medians"], theta=theta,
                        topology=cls,
                    )
                    s = run.run()
                    s.update({f"topo_{k2}": v for k2, v in desc.items()})
                    rows.append(s)
        print(f"topology {cls} done", flush=True)
    out = pd.DataFrame(rows)
    path = os.path.join(RESULTS, "topology_sweep.csv")
    if append and os.path.exists(path):
        out = pd.concat([pd.read_csv(path), out], ignore_index=True)
    out.to_csv(path, index=False)
    return out


# ---------------------------------------------------------------------------
# 5. observation-noise sweep: when does temporal aggregation pay off?
# ---------------------------------------------------------------------------
def run_noise(levels: Tuple[float, ...] = (0.055, 0.15, 0.30, 0.50),
              episodes: int = 4, steps: int = 400) -> pd.DataFrame:
    """The observation map is nearly invertible at low noise, so a single sample
    already identifies the latent health; temporal aggregation can only pay off
    when the map is strongly corrupted. This sweep tests that prediction."""
    from ..simulator import failure_generator as fg
    from ..simulator.dataset_generator import build_dataset

    original = fg.OBS_NOISE
    rows = []
    for sigma in levels:
        fg.OBS_NOISE = sigma
        df = select_eligible(build_dataset(episodes_per_scenario=episodes, steps=steps))
        tr, va, te = episode_split(df, seed=42)
        m_tr = df["episode"].isin(tr).to_numpy()
        m_va = df["episode"].isin(va).to_numpy()
        m_te = df["episode"].isin(te).to_numpy()
        for variant in ("M0_instantaneous", "M2_trend"):
            cols = feature_subset(df, variant)
            y, p, theta = fit_predict(df, cols, m_tr, m_va, m_te, calibrate=False)
            row = {"obs_noise": sigma, "variant": variant, "n_features": len(cols) + 1}
            row.update(score(y, p, theta))
            rows.append(row)
            print(f"sigma={sigma:.3f} {variant:18s} PR-AUC={row['pr_auc']:.4f} "
                  f"recall={row['recall']:.3f}")
    fg.OBS_NOISE = original
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(RESULTS, "noise_sweep.csv"), index=False)
    return out



# ---------------------------------------------------------------------------
# 6. closed-loop feedback ablation: does the policy shape its own failure process?
# ---------------------------------------------------------------------------
def run_feedback(scenarios: Optional[List[str]] = None, seeds: int = 12,
                 steps: int = 500) -> pd.DataFrame:
    """Compare P1 and P3 with the utilisation-to-hazard path enabled and
    disabled. With the feedback disabled the environment follows the synthetic
    load profile instead of the load produced by the controller, so the failure
    process becomes exogenous and the comparison degenerates to a replay."""
    from .run_experiments import prepare_context

    ctx = prepare_context(DATA)
    theta = float(ctx["meta"]["threshold"])
    scenarios = scenarios or ["S3_leo_degradation", "S4_disaster_collapse", "S6_common_cause"]
    rows = []
    for feedback in (True, False):
        for scen in scenarios:
            for policy in ("reactive", "proactive_ai"):
                for k in range(seeds):
                    run = ExperimentRun(
                        scen, policy, 6000 + 29 * k, steps=steps,
                        model=ctx["model"], feature_columns=ctx["feature_columns"],
                        latency_medians=ctx["latency_medians"], theta=theta,
                        load_feedback=feedback,
                    )
                    rows.append(run.run())
        print(f"feedback={feedback} done", flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(RESULTS, "feedback_ablation.csv"), index=False)
    return out


# ---------------------------------------------------------------------------
# 7. sensitivity to the degradation-model parameters
# ---------------------------------------------------------------------------
def run_params(seeds: int = 8, steps: int = 400,
               scenarios: Optional[List[str]] = None,
               only: Optional[List[str]] = None,
               append: bool = False) -> pd.DataFrame:
    """One-at-a-time perturbation of the parameters that govern the generative
    model. The question is not whether the absolute numbers move - they must -
    but whether the sign and the approximate magnitude of the P3-P1 difference
    survive a factor-of-two change in the assumptions."""
    from .run_experiments import prepare_context

    ctx = prepare_context(DATA)
    theta = float(ctx["meta"]["threshold"])
    scenarios = scenarios or ["S4_disaster_collapse"]
    settings = [
        ("baseline", dict()),
        ("lambda0 x0.5", dict(hazard_scale=0.5)),
        ("lambda0 x2.0", dict(hazard_scale=2.0)),
        ("sigma x0.5", dict(sigma_scale=0.5)),
        ("sigma x2.0", dict(sigma_scale=2.0)),
        ("feedback gain x0", dict(feedback_gain_scale=0.0)),
        ("feedback gain x2", dict(feedback_gain_scale=2.0)),
    ]
    rows = []
    if only:
        settings = [(n, k) for n, k in settings if any(o in n for o in only)]
    for name, kwargs in settings:
        for scen in scenarios:
            for policy in ("reactive", "proactive_ai"):
                for k in range(seeds):
                    run = ExperimentRun(
                        scen, policy, 7000 + 61 * k, steps=steps,
                        model=ctx["model"], feature_columns=ctx["feature_columns"],
                        latency_medians=ctx["latency_medians"], theta=theta, **kwargs,
                    )
                    s = run.run()
                    s["setting"] = name
                    rows.append(s)
        print(f"{name} done", flush=True)
    out = pd.DataFrame(rows)
    path = os.path.join(RESULTS, "parameter_sensitivity.csv")
    if append and os.path.exists(path):
        out = pd.concat([pd.read_csv(path), out], ignore_index=True)
    out.to_csv(path, index=False)
    return out


def run_weights(seeds: int = 8, steps: int = 400,
                scenarios: Optional[List[str]] = None,
                only: Optional[List[str]] = None,
                append: bool = False) -> pd.DataFrame:
    """One-at-a-time perturbation of the controller cost coefficients themselves.
    The generative model is untouched; only the routing cost and the risk term
    that the controller optimises are rescaled. This separates 'the mechanism
    works' from 'the mechanism works at these hand-chosen coefficients'."""
    from .run_experiments import prepare_context
    from ..orchestrator import digital_twin as dt

    ctx = prepare_context(DATA)
    theta = float(ctx["meta"]["threshold"])
    scenarios = scenarios or ["S4_disaster_collapse"]
    base = dict(W_LOSS=dt.W_LOSS, W_RISK=dt.W_RISK, W_LOAD=dt.W_LOAD)
    settings = [
        ("W_RISK x0.5", dict(W_RISK=base["W_RISK"] * 0.5)),
        ("W_RISK x2.0", dict(W_RISK=base["W_RISK"] * 2.0)),
        ("W_LOSS x0.5", dict(W_LOSS=base["W_LOSS"] * 0.5)),
        ("W_LOSS x2.0", dict(W_LOSS=base["W_LOSS"] * 2.0)),
        ("W_LOAD x0.5", dict(W_LOAD=base["W_LOAD"] * 0.5)),
        ("W_LOAD x2.0", dict(W_LOAD=base["W_LOAD"] * 2.0)),
    ]
    if only:
        settings = [(n, k) for n, k in settings if any(o in n for o in only)]
    rows = []
    for name, overrides in settings:
        for key, value in base.items():
            setattr(dt, key, overrides.get(key, value))
        try:
            for scen in scenarios:
                for policy in ("reactive", "proactive_ai"):
                    for k in range(seeds):
                        run = ExperimentRun(
                            scen, policy, 7000 + 61 * k, steps=steps,
                            model=ctx["model"], feature_columns=ctx["feature_columns"],
                            latency_medians=ctx["latency_medians"], theta=theta,
                        )
                        r = run.run()
                        r["setting"] = name
                        rows.append(r)
        finally:
            for key, value in base.items():
                setattr(dt, key, value)
        print(f"{name} done", flush=True)
    out = pd.DataFrame(rows)
    path = os.path.join(RESULTS, "controller_weight_sensitivity.csv")
    if append and os.path.exists(path):
        out = pd.concat([pd.read_csv(path), out], ignore_index=True)
    out.to_csv(path, index=False)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["ablation", "ood", "horizon", "migration", "noise",
                                        "topology", "feedback", "params", "weights"])
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--scenarios", nargs="*", default=None)
    ap.add_argument("--horizons", nargs="*", type=int, default=None)
    ap.add_argument("--costs", nargs="*", type=int, default=None)
    ap.add_argument("--topologies", nargs="*", default=None)
    ap.add_argument("--settings", nargs="*", default=None)
    ap.add_argument("--append", action="store_true")
    args = ap.parse_args()
    os.makedirs(RESULTS, exist_ok=True)
    if args.command == "ablation":
        run_ablation()
    elif args.command == "ood":
        run_ood()
    elif args.command == "horizon":
        run_horizon(tuple(args.horizons) if args.horizons else (3, 5, 10, 15, 20),
                    closed_loop_seeds=args.seeds, append=args.append)
    elif args.command == "noise":
        run_noise()
    elif args.command == "feedback":
        run_feedback(scenarios=args.scenarios, seeds=args.seeds)
    elif args.command == "params":
        run_params(seeds=args.seeds, scenarios=args.scenarios,
                   only=args.settings, append=args.append)
    elif args.command == "weights":
        run_weights(seeds=args.seeds, scenarios=args.scenarios,
                    only=args.settings, append=args.append)
    elif args.command == "topology":
        run_topology(scenarios=args.scenarios, seeds=args.seeds, append=args.append,
                     topologies=args.topologies)
    else:
        run_migration(tuple(args.costs) if args.costs else (0, 1, 2, 3, 5, 8),
                      scenarios=args.scenarios, seeds=args.seeds, append=args.append)


if __name__ == "__main__":
    main()
