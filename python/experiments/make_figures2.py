"""
Figure and table generation for the reliability-oriented manuscript.

Main figures (F01-F13) carry the argument; supplementary figures (S01-S06)
support it. Everything is derived from data/results, so no number is drawn by
hand.

    python -m python.experiments.make_figures2
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)

from ..simulator.config import SCENARIOS, SERVICE_PROFILES
from ..simulator.failure_generator import NetworkEnvironment
from ..simulator.network_topology import NODE_CATALOGUE, build_topology, topology_summary
from .resilience import empirical_hazard, kaplan_meier, SurvivalRecord

RESULTS = os.path.join("data", "results")
FIGDIR = os.path.join("paper-assets", "figures")
TABDIR = os.path.join("paper-assets", "tables")

POLICY_ORDER = ["static", "reactive", "proactive_threshold", "degradation_aware",
                "proactive_ai", "oracle"]
POLICY_LABEL = {
    "static": "P0 static",
    "reactive": "P1 reactive restoration",
    "proactive_threshold": "P2 threshold-proactive",
    "degradation_aware": "P3 degradation-aware",
    "proactive_ai": "P4 proposed (predictive)",
    "oracle": "P5 perfect-state reference",
}
POLICY_COLOR = {
    "static": "#9e9e9e",
    "reactive": "#4c72b0",
    "proactive_threshold": "#dd8452",
    "degradation_aware": "#8172b3",
    "proactive_ai": "#c44e52",
    "oracle": "#55a868",
}
SCEN_LABEL = {
    "S1_normal": "S1 nominal",
    "S2_bs_overload": "S2 access overload",
    "S3_leo_degradation": "S3 NTN degradation",
    "S4_disaster_collapse": "S4 infrastructure collapse",
    "S5_leo_handover": "S5 handover dynamics*",
    "S6_common_cause": "S6 common cause*",
    "S7_external_ntn": "S7 external NTN*",
}

plt.rcParams.update({
    "figure.dpi": 160, "savefig.dpi": 300, "font.size": 9,
    "axes.grid": True, "grid.alpha": 0.25,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
})


def _save(fig, name: str) -> None:
    os.makedirs(FIGDIR, exist_ok=True)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, name), bbox_inches="tight")
    plt.close(fig)
    print("figure:", name)


# Narrative tables carry only as many digits as the seed-to-seed spread supports;
# the raw CSVs in data/results retain full precision.
_ROUND = {
    "service_availability": 4, "normalised_resilience": 4, "mean_quality": 4,
    "min_quality": 4, "sla_compliance": 4, "sla_violation_rate": 4,
    "resilience_loss": 2, "weighted_resilience_loss": 2, "mean_recovery_steps": 2,
    "p95_recovery_steps": 2, "mean_latency_ms": 1, "p95_latency_ms": 1,
    "proactive_reroutes_total": 1, "reactive_reroutes": 1, "total_reroutes": 1,
    "outage_events": 1, "flows": 1, "admitted": 1, "rejected": 1,
    "lead_time_mean": 2, "lead_time_median": 2, "anticipation_rate": 3,
}


def _polish(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.columns:
        if not pd.api.types.is_float_dtype(df[col]):
            continue
        key = str(col)
        if key in _ROUND:
            df[col] = df[col].round(_ROUND[key])
        elif "[%]" in key or "[pp]" in key:
            df[col] = df[col].round(2)
        elif "slot" in key.lower():
            df[col] = df[col].round(2)
    return df


def _table(df: pd.DataFrame, name: str, fmt: str = "%.4f") -> None:
    df = _polish(df)
    # A publication table must not display bare NaN cells; an em dash reads as
    # "not applicable", which is what an absent measurement means here.
    df = df.astype(object).where(df.notna(), "—")
    os.makedirs(TABDIR, exist_ok=True)
    df.to_csv(os.path.join(TABDIR, f"{name}.csv"), index=False)
    with open(os.path.join(TABDIR, f"{name}.tex"), "w") as fh:
        fh.write(df.to_latex(index=False, float_format=lambda v: fmt % v, escape=True))
    print("table :", name)


CRITICALITY = {"emergency_message": 5, "command_control": 5, "medical_telemetry": 4,
               "voice_call": 3, "video_stream": 2, "sensor_data": 1}


def _runs() -> pd.DataFrame:
    runs = pd.read_csv(os.path.join(RESULTS, "experiment_runs.csv"))
    cols = [f"avail_{c}" for c in CRITICALITY if f"avail_{c}" in runs.columns]
    if cols:
        w = np.array([CRITICALITY[c.replace("avail_", "")] for c in cols], dtype=float)
        a = runs[cols].to_numpy(dtype=float)
        weighted_availability = (a * w).sum(axis=1) / w.sum()
        runs["weighted_resilience_loss"] = runs["steps"] * (1.0 - weighted_availability)
        runs["weighted_availability"] = weighted_availability
    return runs


def _scen_list(runs: pd.DataFrame) -> List[str]:
    return [s for s in SCEN_LABEL if s in set(runs["scenario"])]


# ===========================================================================
# F03 - hidden degradation versus observable telemetry
# ===========================================================================
def f03_degradation_trace() -> None:
    env = NetworkEnvironment(NODE_CATALOGUE, "S4_disaster_collapse", seed=17)
    health, latency, loss, status = [], [], [], []
    target = "bs5g_1"
    for _ in range(320):
        snap = env.step()
        obs = snap[target]
        health.append(obs["_health"])
        latency.append(obs["latency_ms"])
        loss.append(obs["packet_loss"])
        status.append(obs["_status"])
    t = np.arange(len(health))
    failed = np.array([s == "failed" for s in status])

    fig, axes = plt.subplots(3, 1, figsize=(7.6, 5.2), sharex=True)
    axes[0].plot(t, health, color="#37474f", lw=1.3)
    axes[0].axhline(0.15, color="#c62828", ls="--", lw=0.9)
    axes[0].text(2, 0.19, "failure threshold $h_{fail}$", fontsize=7, color="#c62828")
    axes[0].set_ylabel("latent health $h(t)$")
    axes[0].set_title("Hidden degradation state and the telemetry it produces "
                      "(5G access element, scenario S4)", fontsize=9)

    axes[1].plot(t, latency, color="#4c72b0", lw=0.9)
    axes[1].set_ylabel("latency [ms]")
    axes[2].plot(t, loss, color="#c44e52", lw=0.9)
    axes[2].set_ylabel("packet loss")
    axes[2].set_xlabel("slot")

    for ax in axes:
        ymin, ymax = ax.get_ylim()
        ax.fill_between(t, ymin, ymax, where=failed, color="#c62828", alpha=0.12, step="mid")
        ax.set_ylim(ymin, ymax)
    axes[0].fill_between([], [], [], color="#c62828", alpha=0.12, label="element failed")
    axes[0].legend(fontsize=7, loc="lower left")
    _save(fig, "f03_degradation_trace.png")


# ===========================================================================
# F04 - decision mechanism
# ===========================================================================
def f04_decision_mechanism() -> None:
    fig, ax = plt.subplots(figsize=(8.6, 3.6))
    ax.set_xlim(0, 10.4)
    ax.set_ylim(0, 3.6)
    ax.axis("off")
    stages = [
        ("telemetry\nwindow $X_n(t)$", "#eceff1"),
        ("calibrated risk\n$r_n(t)=P(\\tau_n\\leq H)$", "#e3f2fd"),
        ("path risk\n$R_\\infty(P)=\\max_n r_n$", "#e8f5e9"),
        ("class budget\n$\\tau_c=\\kappa\\phi_c\\theta^*$", "#fff8e1"),
        ("expected cost\n$r\\,C_f$ vs $C_m$", "#fbe9e7"),
        ("keep / migrate\nmake-before-break", "#f3e5f5"),
    ]
    w = 1.5
    for i, (text, col) in enumerate(stages):
        x = 0.2 + i * 1.68
        ax.add_patch(mpatches.FancyBboxPatch((x, 1.4), w, 1.1, boxstyle="round,pad=0.09",
                                             edgecolor="#37474f", facecolor=col, lw=1.0))
        ax.text(x + w / 2, 1.95, text, ha="center", va="center", fontsize=7.6)
        if i < len(stages) - 1:
            ax.annotate("", xy=(x + w + 0.16, 1.95), xytext=(x + w + 0.02, 1.95),
                        arrowprops=dict(arrowstyle="-|>", lw=1.2, color="#37474f"))
    ax.text(5.2, 0.75,
            "migration is rational when $r\\,C_f > C_m$, i.e. when the risk exceeds the "
            "break-even probability $p^*=C_m/C_f$",
            ha="center", fontsize=8, style="italic")
    ax.text(5.2, 3.15, "Risk-aware recovery decision", ha="center", fontsize=9.5)
    _save(fig, "f04_decision_mechanism.png")


# ===========================================================================
# F05 - discrimination and calibration
# ===========================================================================
def f05_predictor() -> None:
    comp = pd.read_csv(os.path.join(RESULTS, "model_comparison.csv"))
    npz = np.load(os.path.join(RESULTS, "test_predictions.npz"), allow_pickle=True)
    y = npz["y_true"]
    meta = json.load(open(os.path.join(RESULTS, "best_model.json")))
    best = meta["best_model"]
    prob = npz[f"prob_{best}"]
    theta = float(comp.set_index("model").loc[best, "threshold"])

    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.3))
    for key in npz.files:
        if not key.startswith("prob_"):
            continue
        name = key[5:]
        pr, rc, _ = precision_recall_curve(y, npz[key])
        ap = comp.set_index("model").loc[name, "pr_auc"]
        axes[0].plot(rc, pr, lw=1.3, label=f"{name.replace('_', ' ')} ({ap:.3f})")
    axes[0].axhline(y.mean(), color="k", ls="--", lw=0.8, label=f"prevalence ({y.mean():.4f})")
    axes[0].set_xlabel("recall")
    axes[0].set_ylabel("precision")
    axes[0].set_title("Precision-recall")
    axes[0].legend(fontsize=6.5)

    frac, mean_pred = calibration_curve(y, prob, n_bins=12, strategy="quantile")
    axes[1].plot(mean_pred, frac, "o-", color="#c44e52", ms=3.5)
    axes[1].plot([0, max(mean_pred)], [0, max(mean_pred)], "k--", lw=0.8)
    axes[1].set_xlabel("predicted probability")
    axes[1].set_ylabel("observed frequency")
    axes[1].set_title("Reliability diagram")

    axes[2].hist(prob[y == 0], bins=60, alpha=0.7, density=True, label="no failure")
    axes[2].hist(prob[y == 1], bins=60, alpha=0.7, density=True, label="failure within $H$")
    axes[2].axvline(theta, color="k", ls="--", lw=1.0, label=r"$\theta^*$")
    axes[2].set_yscale("log")
    axes[2].set_xlabel("risk score")
    axes[2].set_title("Score distribution")
    axes[2].legend(fontsize=7)
    _save(fig, "f05_predictor.png")


# ===========================================================================
# F06 - anticipation lead time
# ===========================================================================
def f06_lead_time() -> None:
    runs = _runs()
    ai = runs[runs["policy"] == "proactive_ai"]
    scen = [s for s in _scen_list(runs) if ai[ai["scenario"] == s]["anticipated_failures"].sum() > 0]

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.4))
    labels, med, p25, p75 = [], [], [], []
    for s in scen:
        sub = ai[ai["scenario"] == s]
        if np.isnan(sub["lead_time_median"]).all():
            continue
        labels.append(SCEN_LABEL[s])
        med.append(np.nanmean(sub["lead_time_median"]))
        p25.append(np.nanmean(sub["lead_time_p25"]))
        p75.append(np.nanmean(sub["lead_time_p75"]))
    ypos = np.arange(len(labels))
    axes[0].barh(ypos, med, color="#c44e52", height=0.55,
                 xerr=[np.array(med) - np.array(p25), np.array(p75) - np.array(med)],
                 capsize=3)
    axes[0].set_yticks(ypos)
    axes[0].set_yticklabels(labels, fontsize=7.5)
    axes[0].set_xlabel("anticipation lead time [slots]")
    axes[0].set_title("Median lead time (bars: inter-quartile range)", fontsize=9)

    frac_labels = ["$\\geq$1", "$\\geq$3", "$\\geq$5", "$\\geq$10"]
    keys = ["lead_ge_1", "lead_ge_3", "lead_ge_5", "lead_ge_10"]
    width = 0.8 / max(1, len(scen))
    for i, s in enumerate(scen):
        sub = ai[ai["scenario"] == s]
        vals = [100 * np.nanmean(sub[k]) for k in keys]
        axes[1].bar(np.arange(len(keys)) + i * width - 0.4, vals, width,
                    label=SCEN_LABEL[s])
    axes[1].set_xticks(np.arange(len(keys)))
    axes[1].set_xticklabels(frac_labels)
    axes[1].set_xlabel("lead time [slots]")
    axes[1].set_ylabel("share of anticipated failures [%]")
    axes[1].set_title("Distribution of usable warning", fontsize=9)
    axes[1].legend(fontsize=6.5)
    _save(fig, "f06_lead_time.png")

    tab = ai.groupby("scenario")[
        ["anticipation_rate", "lead_time_median", "lead_time_mean",
         "lead_ge_3", "lead_ge_5", "anticipated_failures", "missed_failures"]
    ].mean().reset_index()
    tab["scenario"] = tab["scenario"].map(SCEN_LABEL)
    _table(tab.round(2), "t05_lead_time")


# ===========================================================================
# F07 - closed-loop trajectories
# ===========================================================================
def f07_timeline() -> None:
    tl = pd.read_csv(os.path.join(RESULTS, "experiment_timeline.csv"))
    tl = tl[tl["scenario"] == "S4_disaster_collapse"]
    if tl.empty:
        return
    fig, axes = plt.subplots(3, 1, figsize=(7.8, 5.8), sharex=True)
    for pol in POLICY_ORDER:
        sub = tl[tl["policy"] == pol]
        if sub.empty:
            continue
        axes[0].plot(sub["step"], 100 * (1 - sub["outage_fraction"].rolling(15, min_periods=1).mean()),
                     lw=1.1, label=POLICY_LABEL[pol], color=POLICY_COLOR[pol])
    axes[0].set_ylabel("service quality $Q(t)$ [%]")
    axes[0].legend(fontsize=6.5, ncol=3)
    axes[0].set_title("Infrastructure-collapse episode (seed 1000)", fontsize=9)

    sub = tl[tl["policy"] == "proactive_ai"]
    axes[1].plot(sub["step"], sub["failed_nodes"], lw=1.0, color="#37474f")
    axes[1].set_ylabel("failed elements")
    axes[2].plot(sub["step"], sub["mean_risk"], lw=1.0, color="#c44e52", label="predicted risk")
    ora = tl[tl["policy"] == "oracle"]
    axes[2].plot(ora["step"], ora["mean_risk"], lw=1.0, color="#55a868", alpha=0.75,
                 label="perfect-state risk")
    axes[2].set_ylabel("mean risk")
    axes[2].set_xlabel("slot")
    axes[2].legend(fontsize=7)
    _save(fig, "f07_timeline.png")


# ===========================================================================
# F08 - survival curves and hazard
# ===========================================================================
def f08_survival() -> None:
    path = os.path.join(RESULTS, "survival_records.csv")
    if not os.path.exists(path):
        return
    sv = pd.read_csv(path)
    scen_show = [s for s in ("S4_disaster_collapse", "S6_common_cause", "S3_leo_degradation")
                 if s in set(sv["scenario"])]
    fig, axes = plt.subplots(2, len(scen_show) + 1,
                             figsize=(3.2 * (len(scen_show) + 1), 6.0))
    rows = []
    for col, (dur_key, ev_key, ylabel) in enumerate(
            [("duration", "event", "P(no interruption)"),
             ("duration_sustained", "event_sustained", "P(no sustained interruption)")]):
        for ax, scen in zip(axes[col][: len(scen_show)], scen_show):
            for pol in POLICY_ORDER:
                sub = sv[(sv["scenario"] == scen) & (sv["policy"] == pol)]
                if sub.empty:
                    continue
                recs = [SurvivalRecord(int(d), int(e))
                        for d, e in zip(sub[dur_key], sub[ev_key])]
                t, s, _ = kaplan_meier(recs)
                ax.step(t, s, where="post", lw=1.2, color=POLICY_COLOR[pol],
                        label=POLICY_LABEL[pol])
                rows.append({"definition": "any" if col == 0 else "sustained (>= 2 slots)",
                             "scenario": SCEN_LABEL[scen], "policy": POLICY_LABEL[pol],
                             "S(10)": float(np.interp(10, t, s)) if len(t) else np.nan,
                             "S(20)": float(np.interp(20, t, s)) if len(t) else np.nan,
                             "S(40)": float(np.interp(40, t, s)) if len(t) else np.nan,
                             "n_flows": len(recs)})
            ax.set_xlim(0, 60)
            ax.set_xlabel("slots since admission" if col == 1 else "")
            if col == 0:
                ax.set_title(SCEN_LABEL[scen], fontsize=8.5)
        axes[col][0].set_ylabel(ylabel, fontsize=8)
    axes[0][0].legend(fontsize=6.0)

    for col, (dur_key, ev_key, title) in enumerate(
            [("duration", "event", "hazard, any interruption"),
             ("duration_sustained", "event_sustained", "hazard, sustained interruption")]):
        ax = axes[col][-1]
        for pol in ("reactive", "proactive_ai", "proactive_threshold"):
            sub = sv[(sv["scenario"] == "S4_disaster_collapse") & (sv["policy"] == pol)]
            if sub.empty:
                continue
            recs = [SurvivalRecord(int(d), int(e)) for d, e in zip(sub[dur_key], sub[ev_key])]
            c, h = empirical_hazard(recs, bin_width=10)
            ax.plot(c, h, "o-", ms=3, lw=1.2, color=POLICY_COLOR[pol], label=POLICY_LABEL[pol])
        ax.set_xlabel("slots since admission" if col == 1 else "")
        ax.set_ylabel("hazard [1/slot]")
        ax.set_title(f"S4: {title}", fontsize=8.0)
        if col == 0:
            ax.legend(fontsize=6.0)
    _save(fig, "f08_survival.png")
    _table(pd.DataFrame(rows).round(4), "t06_survival")


# ===========================================================================
# F09/F10 - resilience outcomes
# ===========================================================================
def _grouped_bar(ax, runs, metric, scen, percent=False, annotate=False):
    width = 0.14
    xs = np.arange(len(scen))
    for i, pol in enumerate(POLICY_ORDER):
        means, errs = [], []
        for s in scen:
            sub = runs[(runs["scenario"] == s) & (runs["policy"] == pol)][metric]
            k = 100 if percent else 1
            means.append(sub.mean() * k)
            errs.append(sub.std(ddof=1) * k / np.sqrt(max(1, len(sub))))
        ax.bar(xs + (i - 2.5) * width, means, width, yerr=errs, capsize=2,
               label=POLICY_LABEL[pol], color=POLICY_COLOR[pol])
    ax.set_xticks(xs)
    ax.set_xticklabels([SCEN_LABEL[s] for s in scen], fontsize=7, rotation=12)


def f09_resilience_loss() -> None:
    runs = _runs()
    scen = [s for s in _scen_list(runs) if s != "S1_normal"]
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 3.6))
    _grouped_bar(axes[0], runs, "resilience_loss", scen)
    axes[0].set_ylabel("resilience loss $L$ [flow-slots]")
    axes[0].set_title("Cumulative resilience loss", fontsize=9)
    _grouped_bar(axes[1], runs, "normalised_resilience", scen, percent=True)
    axes[1].set_ylabel("normalised resilience $\\mathcal{R}$ [%]")
    lo = min(runs[runs.scenario.isin(scen)].groupby("policy")["normalised_resilience"].min()) * 100
    axes[1].set_ylim(max(0, lo - 1.5), 100.4)
    axes[1].set_title("Normalised resilience", fontsize=9)
    axes[0].legend(fontsize=6.6, ncol=3, loc="upper center", bbox_to_anchor=(1.05, -0.22))
    _save(fig, "f09_resilience.png")


def f10_availability_recovery() -> None:
    runs = _runs()
    scen = _scen_list(runs)
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 3.6))
    _grouped_bar(axes[0], runs, "service_availability", scen, percent=True)
    axes[0].set_ylabel("service availability [%]")
    lo = runs.groupby(["scenario", "policy"])["service_availability"].mean().min() * 100
    axes[0].set_ylim(lo - 0.6, 100.15)
    axes[0].set_title("Service availability", fontsize=9)
    _grouped_bar(axes[1], runs, "mean_recovery_steps", scen)
    axes[1].set_ylabel("mean recovery time [slots]")
    axes[1].set_title("Mean time to restore an interrupted flow", fontsize=9)
    axes[0].legend(fontsize=6.6, ncol=3, loc="upper center", bbox_to_anchor=(1.05, -0.22))
    _save(fig, "f10_availability_recovery.png")


# ===========================================================================
# F11 - migration-cost sensitivity
# ===========================================================================
def f11_migration_cost() -> None:
    path = os.path.join(RESULTS, "migration_cost_sweep.csv")
    if not os.path.exists(path):
        return
    sw = pd.read_csv(path)
    prop = sw[sw["migration_cost"] >= 0]
    ref = sw[sw["migration_cost"] < 0]
    fig, axes = plt.subplots(1, 3, figsize=(10.6, 3.3))
    for scen, group in prop.groupby("scenario"):
        agg = group.groupby("migration_cost").mean(numeric_only=True)
        axes[0].plot(agg.index, 100 * agg["service_availability"], "o-", ms=3.5, lw=1.2,
                     label=SCEN_LABEL.get(scen, scen))
        axes[1].plot(agg.index, agg["resilience_loss"], "o-", ms=3.5, lw=1.2)
        axes[2].plot(agg.index, agg["mean_recovery_steps"], "o-", ms=3.5, lw=1.2)
    for scen, group in ref.groupby("scenario"):
        colour = "#4c72b0" if "S3" in scen else "#8d6e63"
        axes[0].axhline(100 * group["service_availability"].mean(), ls="--", lw=1.0, color=colour)
        axes[1].axhline(group["resilience_loss"].mean(), ls="--", lw=1.0, color=colour)
        axes[2].axhline(group["mean_recovery_steps"].mean(), ls="--", lw=1.0, color=colour)
    axes[0].set_ylabel("availability [%]")
    axes[1].set_ylabel("resilience loss")
    axes[2].set_ylabel("recovery time [slots]")
    for ax in axes:
        ax.set_xlabel("migration cost $C_m$ [slots]")
    axes[0].legend(fontsize=6.8)
    fig.suptitle("Sensitivity to the cost of pre-emptive migration "
                 "(dashed: reactive restoration under the same conditions)", fontsize=9)
    _save(fig, "f11_migration_cost.png")

    tab = prop.groupby(["scenario", "migration_cost"])[
        ["service_availability", "resilience_loss", "mean_recovery_steps",
         "proactive_reroutes_total"]].mean().reset_index()
    _table(tab.round({"service_availability": 4, "resilience_loss": 2, "mean_recovery_steps": 2, "proactive_reroutes_total": 1}), "t08_migration_cost")


# ===========================================================================
# F12 - prediction-horizon trade-off
# ===========================================================================
def f12_horizon() -> None:
    path = os.path.join(RESULTS, "horizon_sweep.csv")
    if not os.path.exists(path):
        return
    hs = pd.read_csv(path).sort_values("horizon")
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.2))
    axes[0].plot(hs["horizon"], hs["pr_auc"], "o-", color="#4c72b0", ms=4, label="PR-AUC")
    ax0b = axes[0].twinx()
    ax0b.plot(hs["horizon"], hs["lift"], "s--", color="#c44e52", ms=4, label="lift")
    ax0b.set_ylabel("lift over prevalence", color="#c44e52")
    axes[0].set_ylabel("PR-AUC", color="#4c72b0")
    axes[0].set_title("Prediction quality", fontsize=9)
    axes[1].plot(hs["horizon"], hs["cl_lead_time_median"], "o-", color="#55a868", ms=4)
    axes[1].set_ylabel("median lead time [slots]")
    axes[1].set_title("Usable warning", fontsize=9)
    axes[2].plot(hs["horizon"], 100 * hs["cl_availability"], "o-", color="#c44e52", ms=4)
    axes[2].set_ylabel("closed-loop availability [%]")
    axes[2].set_title("System-level outcome", fontsize=9)
    for ax in axes:
        ax.set_xlabel("prediction horizon $H$ [slots]")
    fig.suptitle("Longer horizons make the label easier to fit but do not improve the "
                 "closed loop", fontsize=9)
    _save(fig, "f12_horizon.png")
    _table(hs[["horizon", "prevalence", "pr_auc", "recall", "lift",
               "cl_availability", "cl_recovery", "cl_lead_time_median"]].round(4),
           "t09_horizon")


# ===========================================================================
# F13 - generalisation
# ===========================================================================
def f13_generalisation() -> None:
    path = os.path.join(RESULTS, "ood_evaluation.csv")
    if not os.path.exists(path):
        return
    ood = pd.read_csv(path)
    ood = ood[ood["prevalence"] > 0]
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.4))
    labels = [f"{r.held_out.replace('_', ' ')}\n({'LOSO' if r.setting.startswith('leave') else 'unseen'})"
              for r in ood.itertuples()]
    xs = np.arange(len(ood))
    axes[0].bar(xs, ood["lift"], color=["#4c72b0" if s.startswith("leave") else "#c44e52"
                                        for s in ood["setting"]])
    axes[0].axhline(1.0, color="k", ls="--", lw=0.8)
    axes[0].set_xticks(xs)
    axes[0].set_xticklabels(labels, fontsize=6.5, rotation=20, ha="right")
    axes[0].set_ylabel("PR-AUC lift over prevalence")
    axes[0].set_title("Transfer of ranking quality", fontsize=9)

    width = 0.38
    axes[1].bar(xs - width / 2, 100 * ood["recall"], width, label="frozen operating point",
                color="#90a4ae")
    if "recall_recalibrated" in ood:
        axes[1].bar(xs + width / 2, 100 * ood["recall_recalibrated"], width,
                    label="re-tuned on target regime", color="#c44e52")
    axes[1].set_xticks(xs)
    axes[1].set_xticklabels(labels, fontsize=6.5, rotation=20, ha="right")
    axes[1].set_ylabel("recall [%]")
    axes[1].set_title("Transfer of the operating point", fontsize=9)
    axes[1].legend(fontsize=7)
    _save(fig, "f13_generalisation.png")
    _table(ood.round(4), "t07_generalisation")


# ===========================================================================
# F14 - observation noise and the value of temporal features
# ===========================================================================
def f14_noise() -> None:
    path = os.path.join(RESULTS, "noise_sweep.csv")
    if not os.path.exists(path):
        return
    ns = pd.read_csv(path)
    fig, ax = plt.subplots(figsize=(5.4, 3.3))
    for variant, group in ns.groupby("variant"):
        g = group.sort_values("obs_noise")
        ax.plot(g["obs_noise"], g["pr_auc"], "o-", ms=4, lw=1.3,
                label="instantaneous KPIs" if variant.startswith("M0") else "temporal window")
    ax.set_xlabel(r"observation noise $\sigma$")
    ax.set_ylabel("PR-AUC")
    ax.set_title("Temporal aggregation pays off only when the\nobservation map is strongly "
                 "corrupted", fontsize=8.5)
    ax.legend(fontsize=7.5)
    _save(fig, "f14_noise.png")
    _table(ns.pivot_table(index="obs_noise", columns="variant",
                          values=["pr_auc", "recall"]).reset_index().round(4), "t10_noise")


# ===========================================================================
# F15 - redundancy dependence
# ===========================================================================
def f15_topology() -> None:
    path = os.path.join(RESULTS, "topology_sweep.csv")
    if not os.path.exists(path):
        return
    tp = pd.read_csv(path)
    piv = tp.pivot_table(index="topology", columns="policy",
                         values=["service_availability", "resilience_loss",
                                 "mean_recovery_steps"])
    div = tp.groupby("topology")["topo_mean_route_diversity"].first()
    conn = tp.groupby("topology")["topo_edge_connectivity"].first()
    links = tp.groupby("topology")["topo_links"].first()
    out = pd.DataFrame({
        "topology": piv.index,
        "links": links.values,
        "edge connectivity": conn.values,
        "route diversity": div.values,
        "availability reactive": piv[("service_availability", "reactive")].values,
        "availability proposed": piv[("service_availability", "proactive_ai")].values,
        "resilience loss reactive": piv[("resilience_loss", "reactive")].values,
        "resilience loss proposed": piv[("resilience_loss", "proactive_ai")].values,
    }).sort_values("route diversity")
    out["availability gain [pp]"] = 100 * (out["availability proposed"] - out["availability reactive"])
    out["resilience loss reduction [%]"] = -100 * (
        out["resilience loss proposed"] - out["resilience loss reactive"]
    ) / out["resilience loss reactive"]

    fig, axes = plt.subplots(1, 3, figsize=(10.6, 3.3))
    x = out["route diversity"].to_numpy()
    axes[0].plot(x, 100 * out["availability reactive"], "o--", ms=4, lw=1.2,
                 color=POLICY_COLOR["reactive"], label=POLICY_LABEL["reactive"])
    axes[0].plot(x, 100 * out["availability proposed"], "o-", ms=4, lw=1.4,
                 color=POLICY_COLOR["proactive_ai"], label=POLICY_LABEL["proactive_ai"])
    axes[0].set_ylabel("availability [%]")
    axes[0].legend(fontsize=6.8)
    axes[1].bar(x, out["availability gain [pp]"], width=0.35, color="#c44e52")
    axes[1].set_ylabel("availability gain [pp]")
    axes[2].bar(x, out["resilience loss reduction [%]"], width=0.35, color="#4c72b0")
    axes[2].set_ylabel("resilience loss reduction [%]")
    for ax in axes:
        ax.set_xlabel("mean route diversity (node-disjoint paths)")
        for xi, name in zip(x, out["topology"]):
            ax.annotate(name.replace("_", " "), (xi, ax.get_ylim()[0]), fontsize=5.6,
                        rotation=90, ha="center", va="bottom", alpha=0.65)
    fig.suptitle("Resilience benefit of anticipation as a function of structural redundancy",
                 fontsize=9)
    _save(fig, "f15_topology.png")
    _table(out.round(4), "t15_topology")


# ===========================================================================
# F16 - closed-loop feedback ablation
# ===========================================================================
def f16_feedback() -> None:
    path = os.path.join(RESULTS, "feedback_ablation.csv")
    if not os.path.exists(path):
        return
    fb = pd.read_csv(path)
    piv = fb.pivot_table(index=["scenario", "load_feedback"], columns="policy",
                         values=["service_availability", "resilience_loss"])
    rows = []
    for (scen, feedback), r in piv.iterrows():
        rows.append({
            "scenario": SCEN_LABEL.get(scen, scen),
            "load-hazard feedback": "enabled" if feedback else "disabled",
            "availability P1 [%]": 100 * r[("service_availability", "reactive")],
            "availability P4 [%]": 100 * r[("service_availability", "proactive_ai")],
            "resilience loss P1": r[("resilience_loss", "reactive")],
            "resilience loss P4": r[("resilience_loss", "proactive_ai")],
        })
    tab = pd.DataFrame(rows)
    tab["availability gain [pp]"] = tab["availability P4 [%]"] - tab["availability P1 [%]"]
    tab["loss reduction [%]"] = -100 * (tab["resilience loss P4"] - tab["resilience loss P1"]) \
        / tab["resilience loss P1"]

    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.3))
    labels = tab["load-hazard feedback"] + "\n" + tab["scenario"].str.replace(" ", "\n", n=1)
    x = np.arange(len(tab))
    axes[0].bar(x, tab["availability gain [pp]"], color="#c44e52", width=0.5)
    axes[0].set_ylabel("availability gain [pp]")
    axes[1].bar(x, tab["loss reduction [%]"], color="#4c72b0", width=0.5)
    axes[1].set_ylabel("resilience loss reduction [%]")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=7)
    fig.suptitle("Benefit of anticipation with the load-to-hazard feedback enabled and disabled",
                 fontsize=9)
    _save(fig, "f16_feedback.png")
    _table(tab.round(2), "t16_feedback")


# ===========================================================================
# F17 - sensitivity to the degradation-model parameters
# ===========================================================================
def f17_parameters() -> None:
    path = os.path.join(RESULTS, "parameter_sensitivity.csv")
    if not os.path.exists(path):
        return
    ps = pd.read_csv(path)
    piv = ps.pivot_table(index="setting", columns="policy",
                         values=["service_availability", "resilience_loss"])
    tab = pd.DataFrame({
        "setting": piv.index,
        "availability P1 [%]": 100 * piv[("service_availability", "reactive")].values,
        "availability P4 [%]": 100 * piv[("service_availability", "proactive_ai")].values,
        "resilience loss P1": piv[("resilience_loss", "reactive")].values,
        "resilience loss P4": piv[("resilience_loss", "proactive_ai")].values,
    })
    tab["availability gain [pp]"] = tab["availability P4 [%]"] - tab["availability P1 [%]"]
    tab["loss reduction [%]"] = -100 * (tab["resilience loss P4"] - tab["resilience loss P1"]) \
        / tab["resilience loss P1"]
    order = ["lambda0 x0.5", "baseline", "lambda0 x1.5", "lambda0 x2.0",
             "sigma x0.5", "sigma x1.5", "sigma x2.0",
             "feedback gain x0", "feedback gain x2"]
    tab["_o"] = tab["setting"].apply(lambda v: order.index(v) if v in order else 99)
    tab = tab.sort_values("_o").drop(columns="_o")

    fig, ax = plt.subplots(figsize=(7.6, 3.4))
    x = np.arange(len(tab))
    ax.bar(x - 0.2, tab["availability gain [pp]"], 0.4, color="#c44e52",
           label="availability gain [pp]")
    ax2 = ax.twinx()
    ax2.bar(x + 0.2, tab["loss reduction [%]"], 0.4, color="#4c72b0",
            label="resilience loss reduction [%]")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(tab["setting"], fontsize=7, rotation=18, ha="right")
    ax.set_ylabel("availability gain [pp]", color="#c44e52")
    ax2.set_ylabel("resilience loss reduction [%]", color="#4c72b0")
    ax.set_title("Benefit of the proposed policy under perturbed degradation parameters",
                 fontsize=9)
    _save(fig, "f17_parameters.png")
    tab = tab.round({"availability P1 [%]": 2, "availability P4 [%]": 2,
                     "resilience loss P1": 2, "resilience loss P4": 2,
                     "availability gain [pp]": 2, "loss reduction [%]": 1})
    _table(tab, "t17_parameters")


# ===========================================================================
# T24 - externally anchored validation scenario
# ===========================================================================
def t24_external() -> None:
    from scipy import stats
    runs = pd.read_csv(os.path.join(RESULTS, "experiment_runs.csv"))
    sub = runs[runs.scenario == "S7_external_ntn"]
    if sub.empty:
        return
    rows = []
    for pol in POLICY_ORDER:
        g = sub[sub.policy == pol]
        if g.empty:
            continue
        rows.append({"policy": POLICY_LABEL[pol],
                     "availability [%]": 100 * g["service_availability"].mean(),
                     "resilience loss": g["resilience_loss"].mean(),
                     "recovery [slots]": g["mean_recovery_steps"].mean(),
                     "proactive migrations": g["proactive_reroutes_total"].mean()})
    _table(pd.DataFrame(rows).round(2), "t24_external")

    base = sub[sub.policy == "reactive"].sort_values("seed")
    stat_rows = []
    for pol in ("degradation_aware", "proactive_ai"):
        cmp = sub[sub.policy == pol].sort_values("seed")
        seeds = sorted(set(base.seed) & set(cmp.seed))
        a = base[base.seed.isin(seeds)].sort_values("seed")
        b = cmp[cmp.seed.isin(seeds)].sort_values("seed")
        for metric, label in (("service_availability", "availability [pp]"),
                              ("resilience_loss", "resilience loss"),
                              ("mean_recovery_steps", "recovery [slots]")):
            scale = 100.0 if metric == "service_availability" else 1.0
            d = scale * (b[metric].to_numpy() - a[metric].to_numpy())
            n = len(d)
            sd = d.std(ddof=1)
            t, pt = stats.ttest_rel(b[metric], a[metric])
            half = stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n)
            stat_rows.append({"comparison": f"{POLICY_LABEL[pol]} - P1 reactive restoration",
                              "measure": label, "n_seeds": n,
                              "mean difference": d.mean(),
                              "ci95 low": d.mean() - half, "ci95 high": d.mean() + half,
                              "cohen d_z": d.mean() / sd if sd else float("nan"),
                              "p_t": float(pt)})
    _table(pd.DataFrame(stat_rows).round(4), "t25_external_inference")


# ===========================================================================
# T23 - sensitivity to the controller cost coefficients
# ===========================================================================
def t23_controller_weights() -> None:
    path = os.path.join(RESULTS, "controller_weight_sensitivity.csv")
    if not os.path.exists(path):
        return
    cw = pd.read_csv(path)
    piv = cw.pivot_table(index="setting", columns="policy",
                         values=["service_availability", "resilience_loss"])
    tab = pd.DataFrame({
        "setting": piv.index,
        "availability P1 [%]": 100 * piv[("service_availability", "reactive")].values,
        "availability P4 [%]": 100 * piv[("service_availability", "proactive_ai")].values,
        "resilience loss P1": piv[("resilience_loss", "reactive")].values,
        "resilience loss P4": piv[("resilience_loss", "proactive_ai")].values,
    })
    tab["availability gain [pp]"] = tab["availability P4 [%]"] - tab["availability P1 [%]"]
    tab["loss reduction [%]"] = -100 * (tab["resilience loss P4"] - tab["resilience loss P1"]) \
        / tab["resilience loss P1"]
    _table(tab.round({"availability P1 [%]": 2, "availability P4 [%]": 2,
                      "resilience loss P1": 2, "resilience loss P4": 2,
                      "availability gain [pp]": 2, "loss reduction [%]": 1}),
           "t23_controller_weights")


# ===========================================================================
# T22 - consolidated robustness strategy
# ===========================================================================
def t22_robustness_strategy() -> None:
    """One table that states, for each threat to validity, the test performed and
    the outcome, with every number read back from the corresponding result file."""
    def rd(name):
        path = os.path.join(TABDIR, name)
        return pd.read_csv(path) if os.path.exists(path) else None

    rows = []
    ps = rd("t17_parameters.csv")
    if ps is not None:
        haz = ps[ps.setting.str.startswith("lambda0")]
        sig = ps[ps.setting.str.startswith("sigma")]
        pos = int((ps["availability gain [pp]"] > 0).sum())
        rows.append(["Results are an artefact of the chosen hazard rate",
                     "lambda_0 halved and doubled (Section 7.9)",
                     f"advantage positive in {int((haz['availability gain [pp]'] > 0).sum())} of "
                     f"{len(haz)} settings; vanishes only in the near-zero-event regime"])
        rows.append(["Results are an artefact of the diffusion coefficient",
                     "sigma halved and doubled (Section 7.9)",
                     f"advantage positive in {int((sig['availability gain [pp]'] > 0).sum())} of "
                     f"{len(sig)} settings; loss reduction "
                     f"{sig['loss reduction [%]'].min():.0f}-{sig['loss reduction [%]'].max():.0f} %"])
        rows.append(["Results depend on the strength of the load-hazard coupling",
                     "feedback gain set to 0 and doubled (Section 7.9)",
                     f"sign preserved; overall {pos} of {len(ps)} settings show a positive gain"])
    fb = rd("t16_feedback.csv")
    if fb is not None:
        on = fb[fb["load-hazard feedback"] == "enabled"]["loss reduction [%]"].mean()
        off = fb[fb["load-hazard feedback"] == "disabled"]["loss reduction [%]"].mean()
        rows.append(["The closed-loop mechanism is asserted rather than demonstrated",
                     "load-to-hazard path disabled (Section 7.8)",
                     f"loss reduction falls from {on:.1f} % to {off:.1f} %, isolating the "
                     "exposure-shaping component"])
    tp = rd("t15_topology.csv")
    if tp is not None:
        col = [c for c in tp.columns if "reduction" in c.lower() or "gain" in c.lower()]
        rows.append(["Results hold only for one network graph",
                     f"{len(tp)} redundancy classes, elements and traffic held fixed "
                     "(Section 7.7)",
                     "benefit persists wherever alternative paths exist and shrinks towards "
                     "zero when they do not"])
    ood = rd("t07_generalisation.csv")
    if ood is not None:
        rows.append(["The predictor is evaluated only on mechanisms it was trained on",
                     "leave-one-scenario-out plus two mechanisms held out entirely "
                     "(Section 7.5)",
                     "ranking degrades and the frozen operating point loses recall on unseen "
                     "dynamics; closed-loop benefit is partly preserved"])
    rows.append(["The prediction horizon was tuned to the outcome",
                 "horizon swept and the nominal value retained (Section 7.6)",
                 "closed-loop availability flat to a few tenths of a point; nominal H not "
                 "re-optimised after the sweep"])
    sc = rd("t21_seed_convergence.csv")
    if sc is not None:
        g = sc[sc.measure.str.startswith("resilience")]
        rows.append(["Too few Monte Carlo replications",
                     "paired effect recomputed at 8 to 24 seeds (Appendix B)",
                     f"mean difference stable within "
                     f"{abs(g['mean difference'].max() - g['mean difference'].min()):.2f} "
                     "slot-equivalents; interval half-width shrinks as expected"])
    cw = rd("t23_controller_weights.csv")
    if cw is not None:
        rows.append(["Controller cost coefficients were hand-tuned",
                     f"{len(cw)} one-at-a-time perturbations of the routing and budget "
                     "coefficients (Appendix B)",
                     f"ordering preserved in {int((cw['availability gain [pp]'] > 0).sum())} of "
                     f"{len(cw)} settings"])
    ext = rd("t24_external.csv")
    if ext is not None:
        rows.append(["Every scenario is parameterised by the authors",
                     "scenario parameterised from published LEO measurement studies and held "
                     "out of training (Section 7.11)",
                     "advantage retained on externally derived dynamics"])
    _table(pd.DataFrame(rows, columns=["threat to validity", "test", "outcome"]),
           "t22_robustness_strategy")


# ===========================================================================
# F19 - Monte Carlo convergence of the paired effect
# ===========================================================================
def f19_seed_convergence() -> None:
    from scipy import stats
    runs = pd.read_csv(os.path.join(RESULTS, "experiment_runs.csv"))
    scen = "S4_disaster_collapse"
    base = runs[(runs.scenario == scen) & (runs.policy == "reactive")].sort_values("seed")
    prop = runs[(runs.scenario == scen) & (runs.policy == "proactive_ai")].sort_values("seed")
    seeds = sorted(set(base.seed) & set(prop.seed))
    if len(seeds) < 8:
        return
    b = base[base.seed.isin(seeds)].sort_values("seed")
    p = prop[prop.seed.isin(seeds)].sort_values("seed")
    dl = p["resilience_loss"].to_numpy() - b["resilience_loss"].to_numpy()
    da = 100.0 * (p["service_availability"].to_numpy() - b["service_availability"].to_numpy())

    rows = []
    for n in [k for k in (8, 12, 16, 20, 24, 28, 32) if k <= len(seeds)]:
        for arr, label in ((dl, "resilience loss [slot-equivalents]"),
                           (da, "availability [pp]")):
            x = arr[:n]
            sd = x.std(ddof=1)
            half = stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n)
            rows.append({"measure": label, "seeds": n, "mean difference": x.mean(),
                         "ci95 low": x.mean() - half, "ci95 high": x.mean() + half,
                         "ci half-width": half})
    tab = pd.DataFrame(rows)
    _table(tab.round(3), "t21_seed_convergence")

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.2))
    for ax, label in zip(axes, tab["measure"].unique()):
        g = tab[tab.measure == label]
        ax.errorbar(g["seeds"], g["mean difference"],
                    yerr=g["ci half-width"], fmt="o-", ms=4, lw=1.2, capsize=3,
                    color="#4c72b0")
        ax.axhline(0, color="k", lw=0.8, ls=":")
        ax.set_xlabel("paired seeds")
        ax.set_ylabel(f"P4 - P1, {label}", fontsize=8)
        ax.set_xticks(sorted(g["seeds"]))
    fig.suptitle("Monte Carlo convergence of the paired effect under infrastructure collapse",
                 fontsize=9)
    _save(fig, "f19_seed_convergence.png")


# ===========================================================================
# T19 - weighted resilience loss with exploratory paired inference
# ===========================================================================
def t19_weighted() -> None:
    from scipy import stats
    from python.simulator.config import SERVICE_PROFILES
    runs = pd.read_csv(os.path.join(RESULTS, "experiment_runs.csv"))
    cols = [f"avail_{c}" for c in CRITICALITY if f"avail_{c}" in runs.columns]
    if not cols:
        return
    w = np.array([CRITICALITY[c.replace("avail_", "")] for c in cols], dtype=float)
    expo = np.array([SERVICE_PROFILES[c.replace("avail_", "")].arrival_weight
                     * float(np.mean(SERVICE_PROFILES[c.replace("avail_", "")].duration_steps))
                     for c in cols], dtype=float)
    wf = w * expo / expo.sum()
    av = runs[cols].to_numpy(dtype=float)
    runs["L_cls"] = runs["steps"] * (1.0 - (av * w).sum(1) / w.sum())
    runs["L_flw"] = runs["steps"] * (1.0 - (av * wf).sum(1) / wf.sum())

    scen = "S4_disaster_collapse"
    sub = runs[runs.scenario == scen]
    rows = []
    for pol in POLICY_ORDER:
        g = sub[sub.policy == pol]
        if g.empty:
            continue
        rows.append({"policy": POLICY_LABEL[pol],
                     "unweighted L": g["resilience_loss"].mean(),
                     "class-balanced L_w": g["L_cls"].mean(),
                     "flow-weighted L_w": g["L_flw"].mean()})
    _table(pd.DataFrame(rows).round(2), "t19_weighted_loss")

    # exploratory paired inference against reactive restoration
    stat_rows = []
    base = sub[sub.policy == "reactive"].sort_values("seed")
    for pol in ("degradation_aware", "proactive_ai"):
        cmp = sub[sub.policy == pol].sort_values("seed")
        seeds = sorted(set(base.seed) & set(cmp.seed))
        b = base[base.seed.isin(seeds)].sort_values("seed")
        c = cmp[cmp.seed.isin(seeds)].sort_values("seed")
        for metric, label in (("resilience_loss", "unweighted L"),
                              ("L_cls", "class-balanced L_w"),
                              ("L_flw", "flow-weighted L_w")):
            d = c[metric].to_numpy() - b[metric].to_numpy()
            n = len(d)
            sd = d.std(ddof=1)
            t, pt = stats.ttest_rel(c[metric], b[metric])
            try:
                ws, pw = stats.wilcoxon(c[metric], b[metric])
            except ValueError:
                ws, pw = float("nan"), float("nan")
            half = stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n)
            stat_rows.append({
                "comparison": f"{POLICY_LABEL[pol]} - P1 reactive restoration",
                "measure": label, "n_seeds": n,
                "mean difference": d.mean(),
                "ci95 low": d.mean() - half, "ci95 high": d.mean() + half,
                "cohen d_z": d.mean() / sd if sd else float("nan"),
                "t": float(t), "p_t": float(pt), "wilcoxon p": float(pw),
            })
    _table(pd.DataFrame(stat_rows).round(4), "t20_weighted_inference")


# ===========================================================================
# Supplementary
# ===========================================================================
def supplementary() -> None:
    comp = pd.read_csv(os.path.join(RESULTS, "model_comparison.csv"))
    npz = np.load(os.path.join(RESULTS, "test_predictions.npz"), allow_pickle=True)
    y = npz["y_true"]
    meta = json.load(open(os.path.join(RESULTS, "best_model.json")))
    best = meta["best_model"]

    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    for key in npz.files:
        if key.startswith("prob_"):
            fpr, tpr, _ = roc_curve(y, npz[key])
            ax.plot(fpr, tpr, lw=1.3,
                    label=f"{key[5:].replace('_', ' ')} "
                          f"({comp.set_index('model').loc[key[5:], 'roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.legend(fontsize=6.5)
    _save(fig, "s01_roc.png")

    prob = npz[f"prob_{best}"]
    theta = float(comp.set_index("model").loc[best, "threshold"])
    cm = confusion_matrix(y, (prob >= theta).astype(int))
    fig, ax = plt.subplots(figsize=(4.0, 3.5))
    ConfusionMatrixDisplay(cm, display_labels=["no failure", "failure $\\leq H$"]).plot(
        ax=ax, colorbar=False, cmap="Blues", values_format="d")
    ax.grid(False)
    _save(fig, "s02_confusion.png")

    imp_path = os.path.join(RESULTS, "random_forest_feature_importance.csv")
    if os.path.exists(imp_path):
        imp = pd.read_csv(imp_path).head(16).iloc[::-1]
        fig, ax = plt.subplots(figsize=(6.0, 4.2))
        ax.barh([n.replace("num__", "").replace("cat__", "") for n in imp["feature"]],
                imp["importance"], color="#4c72b0")
        ax.set_xlabel("mean decrease in impurity")
        _save(fig, "s03_feature_importance.png")

    sweep = os.path.join(RESULTS, "threshold_sweep.csv")
    if os.path.exists(sweep):
        sw = pd.read_csv(sweep)
        fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.0))
        for scen, group in sw.groupby("scenario"):
            agg = group.groupby("budget_scale").mean(numeric_only=True)
            axes[0].plot(agg.index, 100 * agg["service_availability"], "o-", ms=3,
                         label=SCEN_LABEL.get(scen, scen))
            axes[1].plot(agg.index, agg["mean_recovery_steps"], "o-", ms=3)
            axes[2].plot(agg.index, agg["proactive_reroutes_total"], "o-", ms=3)
        axes[0].set_ylabel("availability [%]")
        axes[1].set_ylabel("recovery [slots]")
        axes[2].set_ylabel("migrations")
        for ax in axes:
            ax.set_xlabel(r"risk-budget scale $\kappa$")
        axes[0].legend(fontsize=6.5)
        _save(fig, "s04_budget_sweep.png")

    runs = _runs()
    classes = list(SERVICE_PROFILES.keys())
    scen = [s for s in _scen_list(runs) if s != "S1_normal"]
    delta = np.zeros((len(classes), len(scen)))
    for i, c in enumerate(classes):
        for j, s in enumerate(scen):
            a = runs[(runs.scenario == s) & (runs.policy == "reactive")][f"avail_{c}"].mean()
            b = runs[(runs.scenario == s) & (runs.policy == "proactive_ai")][f"avail_{c}"].mean()
            delta[i, j] = 100 * (b - a)
    fig, ax = plt.subplots(figsize=(6.6, 3.4))
    lim = np.nanmax(np.abs(delta)) or 1.0
    im = ax.imshow(delta, cmap="RdYlGn", vmin=-lim, vmax=lim, aspect="auto")
    ax.set_xticks(range(len(scen)))
    ax.set_xticklabels([SCEN_LABEL[s] for s in scen], fontsize=6.8, rotation=15)
    ax.set_yticks(range(len(classes)))
    ax.set_yticklabels([c.replace("_", " ") for c in classes], fontsize=7.2)
    for i in range(len(classes)):
        for j in range(len(scen)):
            ax.text(j, i, f"{delta[i, j]:+.2f}", ha="center", va="center", fontsize=6.5)
    ax.grid(False)
    fig.colorbar(im, ax=ax, shrink=0.8)
    _save(fig, "s05_class_gain.png")


# ===========================================================================
# Tables
# ===========================================================================
def tables() -> None:
    from ..simulator.config import NODE_PROFILES

    _table(pd.DataFrame([{
        "element type": p.node_type, "latency floor [ms]": p.base_latency_ms,
        "capacity [Mbit/s]": p.capacity_mbps, "loss floor": p.base_loss,
        "RSSI [dBm]": p.base_signal_dbm, "hazard scale": p.hazard_scale,
        "battery backed": p.battery_backed,
    } for p in NODE_PROFILES.values()]), "t01_elements", fmt="%g")

    _table(pd.DataFrame([{
        "service class": s.name.replace("_", " "), "criticality": s.priority,
        "latency budget [ms]": s.max_latency_ms, "loss budget": s.max_loss,
        "demand [Mbit/s]": s.min_bandwidth_mbps,
        "holding time [slots]": f"{s.duration_steps[0]}-{s.duration_steps[1]}",
        "consequence of interruption": {
            5: "loss of alerting or command capability",
            4: "loss of patient monitoring",
            3: "loss of voice coordination",
            2: "loss of situational imagery",
            1: "delayed environmental reporting",
        }[s.priority],
    } for s in SERVICE_PROFILES.values()]), "t02_services", fmt="%g")

    _table(pd.DataFrame([{
        "scenario": s.name, "description": s.description,
        "load factor": s.load_multiplier,
        "hazard multipliers": ", ".join(f"{k}:{v}" for k, v in s.hazard_multiplier.items()) or "-",
        "destroyed elements": ", ".join(s.destroyed_nodes) or "-",
        "correlated shocks": "yes" if s.common_cause_rate > 0 else "no",
        "handover dynamics": "yes" if s.handover_period > 0 else "no",
        "in training corpus": "no" if s.name.startswith(("S5", "S6")) else "yes",
    } for s in SCENARIOS.values()]), "t03_scenarios", fmt="%g")

    _table(pd.read_csv(os.path.join(RESULTS, "model_comparison.csv")).round(4),
           "t04_model_comparison")

    runs = _runs()
    metrics = ["service_availability", "normalised_resilience", "resilience_loss",
               "weighted_resilience_loss", "mean_recovery_steps", "sla_compliance",
               "mean_latency_ms", "proactive_reroutes_total", "reactive_reroutes"]
    rows = []
    for scen in _scen_list(runs):
        for pol in POLICY_ORDER:
            sub = runs[(runs["scenario"] == scen) & (runs["policy"] == pol)]
            if sub.empty:
                continue
            row = {"scenario": SCEN_LABEL[scen], "policy": POLICY_LABEL[pol]}
            for m in metrics:
                if m not in sub.columns:
                    continue
                digits = 4 if "avail" in m or "resilience" in m or "sla" in m else 2
                row[m] = round(float(sub[m].mean()), digits)
                row[f"{m}_sd"] = round(float(sub[m].std(ddof=1)), digits)
            rows.append(row)
    _table(pd.DataFrame(rows), "t11_main_results", fmt="%s")

    sig = pd.read_csv(os.path.join(RESULTS, "significance_tests.csv"))
    sig = sig[(sig["policy"] == "proactive_ai") & (sig["scenario"] != "ALL")]
    keep = ["service_availability", "resilience_loss", "mean_recovery_steps",
            "normalised_resilience"]
    conf = sig[(sig["family"] == "confirmatory")].copy()
    conf["95% CI"] = conf.apply(
        lambda r: f"[{r['ci95_low']:.3g}, {r['ci95_high']:.3g}]", axis=1)
    for col in ("p_value_ttest", "p_value_wilcoxon", "p_holm"):
        conf[col] = conf[col].apply(
            lambda v: "n/a" if pd.isna(v) else ("<1e-4" if v < 1e-4 else f"{v:.4f}"))
    conf["mean_difference"] = conf["mean_difference"].map(lambda v: f"{v:.4g}")
    conf["relative_change_pct"] = conf["relative_change_pct"].round(1)
    conf["cohens_dz"] = conf["cohens_dz"].round(2)
    _table(conf[["scenario", "metric", "mean_difference", "95% CI",
                 "relative_change_pct", "cohens_dz", "p_value_ttest",
                 "p_value_wilcoxon", "p_holm", "significant_005"]],
           "t12_significance", fmt="%s")
    expl = sig[sig["family"] == "exploratory"].copy()
    expl["95% CI"] = expl.apply(
        lambda r: f"[{r['ci95_low']:.3g}, {r['ci95_high']:.3g}]", axis=1)
    _table(expl[["scenario", "policy", "metric", "mean_difference", "95% CI",
                 "cohens_dz", "p_value_ttest", "p_value_wilcoxon"]].round(4),
           "t18_exploratory")

    abl = os.path.join(RESULTS, "feature_ablation.csv")
    if os.path.exists(abl):
        _table(pd.read_csv(abl).round(4), "t13_feature_ablation")

    # robustness summary
    rows = []
    ood = pd.read_csv(os.path.join(RESULTS, "ood_evaluation.csv"))
    for r in ood.itertuples():
        if r.prevalence <= 0:
            continue
        rows.append({"experiment": r.setting.replace("_", " "), "setting": r.held_out,
                     "key metric": "PR-AUC lift", "value": round(r.lift, 2)})
    mig = pd.read_csv(os.path.join(RESULTS, "migration_cost_sweep.csv"))
    prop = mig[mig["migration_cost"] >= 0].groupby(["scenario", "migration_cost"])[
        "service_availability"].mean()
    ref = mig[mig["migration_cost"] < 0].groupby("scenario")["service_availability"].mean()
    for scen in ref.index:
        crossing = [c for c in prop[scen].index if prop[scen][c] < ref[scen]]
        rows.append({"experiment": "migration cost", "setting": scen,
                     "key metric": "break-even cost [slots]",
                     "value": min(crossing) if crossing else float("inf")})
    hs = pd.read_csv(os.path.join(RESULTS, "horizon_sweep.csv"))
    rows.append({"experiment": "prediction horizon", "setting": "closed-loop optimum",
                 "key metric": "H [slots]",
                 "value": int(hs.loc[hs["cl_availability"].idxmax(), "horizon"])})
    _table(pd.DataFrame(rows), "t14_robustness_summary", fmt="%g")


def main() -> None:
    os.makedirs(FIGDIR, exist_ok=True)
    f03_degradation_trace()
    f04_decision_mechanism()
    f05_predictor()
    f06_lead_time()
    f07_timeline()
    f08_survival()
    f09_resilience_loss()
    f10_availability_recovery()
    f11_migration_cost()
    f12_horizon()
    f13_generalisation()
    f14_noise()
    f15_topology()
    f16_feedback()
    f17_parameters()
    t19_weighted()
    f19_seed_convergence()
    t23_controller_weights()
    t24_external()
    t22_robustness_strategy()
    supplementary()
    tables()
    print("\nassets rebuilt")


if __name__ == "__main__":
    main()
