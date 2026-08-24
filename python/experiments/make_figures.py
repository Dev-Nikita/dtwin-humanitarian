"""
Publication asset generation.

Produces every figure and every table used in the paper from the CSV/NPZ
artefacts written by the training and experiment stages. Nothing is drawn by
hand: if a number changes in `data/results`, the figure and the table change
with it.

    python -m python.experiments.make_figures
"""

from __future__ import annotations

import json
import os
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)

from ..simulator.config import SCENARIOS, SERVICE_PROFILES
from ..simulator.network_topology import build_topology, topology_summary

RESULTS = os.path.join("data", "results")
FIGDIR = os.path.join("paper-assets", "figures")
TABDIR = os.path.join("paper-assets", "tables")

POLICY_LABEL = {
    "static": "Static",
    "reactive": "Reactive restoration",
    "proactive_ai": "Proposed (AI DT)",
    "oracle": "Perfect-state reference",
}
POLICY_ORDER = ["static", "reactive", "proactive_ai", "oracle"]
POLICY_COLOR = {
    "static": "#9e9e9e",
    "reactive": "#4c72b0",
    "proactive_ai": "#c44e52",
    "oracle": "#55a868",
}
SCEN_LABEL = {
    "S1_normal": "S1 nominal",
    "S2_bs_overload": "S2 BS overload",
    "S3_leo_degradation": "S3 NTN degradation",
    "S4_disaster_collapse": "S4 disaster collapse",
}

plt.rcParams.update(
    {
        "figure.dpi": 160,
        "savefig.dpi": 300,
        "font.size": 9,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
    }
)


def _save(fig, name: str) -> str:
    os.makedirs(FIGDIR, exist_ok=True)
    path = os.path.join(FIGDIR, name)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print("figure:", path)
    return path


def _save_table(df: pd.DataFrame, name: str, float_fmt: str = "%.4f") -> None:
    os.makedirs(TABDIR, exist_ok=True)
    df.to_csv(os.path.join(TABDIR, f"{name}.csv"), index=False)
    with open(os.path.join(TABDIR, f"{name}.tex"), "w") as fh:
        fh.write(df.to_latex(index=False, float_format=lambda v: float_fmt % v, escape=True))
    print("table :", name)


# ---------------------------------------------------------------------------
# Fig. 1 - framework architecture
# ---------------------------------------------------------------------------
def fig_architecture() -> None:
    fig, ax = plt.subplots(figsize=(7.4, 8.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12.4)
    ax.axis("off")

    blocks = [
        (0.4, 10.8, 9.2, 1.3, "Humanitarian application layer\n"
         "emergency messaging | medical telemetry | command & control | voice | video | sensors",
         "#e8eef7"),
        (0.4, 9.0, 9.2, 1.3, "Service orchestration layer (Contribution 3)\n"
         "risk-aware constrained routing | service-differentiated risk budgets | "
         "make-before-break migration", "#fbe9e7"),
        (0.4, 7.2, 4.4, 1.3, "AI failure predictor (Contribution 2)\n"
         "window features -> calibrated P(failure <= H)", "#e8f5e9"),
        (5.2, 7.2, 4.4, 1.3, "Resilience assessment (Contribution 4)\n"
         "availability | SLA | recovery | class-level KPIs", "#e8f5e9"),
        (0.4, 5.4, 9.2, 1.3, "Digital twin core (Contribution 1)\n"
         "node/link state | observed-status estimation | time-varying cost graph | "
         "capacity ledger", "#fff8e1"),
        (0.4, 3.6, 9.2, 1.3, "Telemetry & synchronisation layer\n"
         "KPI collection | detection delay model | feature ring buffers", "#f3e5f5"),
        (0.4, 1.4, 9.2, 1.8, "Physical hybrid network\n"
         "6G / 5G / LTE access | UAV relays | LEO constellation + feeder links | "
         "edge nodes | HQ core", "#eceff1"),
    ]
    for x, y, w, h, text, color in blocks:
        ax.add_patch(mpatches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.12", linewidth=1.1,
            edgecolor="#37474f", facecolor=color))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8.4)

    arrows = [
        (5.0, 3.2, 5.0, 3.55, "telemetry"),
        (3.0, 5.0, 3.0, 5.35, ""),
        (2.6, 6.7, 2.6, 7.15, "state window"),
        (7.4, 6.7, 7.4, 7.15, "ground truth"),
        (2.6, 8.5, 2.6, 8.95, "risk r_n(t)"),
        (5.0, 10.3, 5.0, 10.75, "path install"),
        (8.6, 8.95, 8.6, 8.55, "decisions"),
    ]
    for x0, y0, x1, y1, label in arrows:
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", lw=1.3, color="#37474f"))
        if label:
            ax.text(x1 + 0.15, (y0 + y1) / 2, label, fontsize=7.4, va="center")

    ax.annotate("", xy=(0.35, 2.3), xytext=(0.05, 2.3),
                arrowprops=dict(arrowstyle="-|>", lw=1.2, color="#c62828"))
    ax.plot([0.05, 0.05], [2.3, 9.65], color="#c62828", lw=1.2)
    ax.annotate("", xy=(0.05, 9.65), xytext=(0.05, 9.3),
                arrowprops=dict(arrowstyle="-", lw=1.2, color="#c62828"))
    ax.annotate("", xy=(0.38, 9.65), xytext=(0.05, 9.65),
                arrowprops=dict(arrowstyle="<|-", lw=1.2, color="#c62828"))
    ax.text(0.15, 6.0, "closed control loop\n(load feeds back into the hazard process)",
            rotation=90, fontsize=7.2, color="#c62828", va="center")
    _save(fig, "fig01_architecture.png")


# ---------------------------------------------------------------------------
# Fig. 2 - reference topology
# ---------------------------------------------------------------------------
def fig_topology() -> None:
    """Reference topology drawn as a layered infrastructure schematic rather than
    a force-directed graph export: deterministic positions, banded segments and
    explicit labelling of the redundancy that the recovery policies exploit."""
    g = build_topology()
    layer_y = {"user": 0.0, "terrestrial_access": 1.6, "aerial": 2.9,
               "edge": 2.9, "non_terrestrial": 4.2, "core": 5.5}
    order: Dict[str, List[str]] = {}
    for n, d in g.nodes(data=True):
        order.setdefault(d["layer"], []).append(n)
    pos = {}
    for layer, nodes in order.items():
        nodes = sorted(nodes)
        for i, n in enumerate(nodes):
            x = (i - (len(nodes) - 1) / 2) * 2.3
            if layer == "edge":
                x += 6.4
            pos[n] = (x, layer_y[layer])

    color = {"MOBILE": "#90a4ae", "4G": "#8d6e63", "5G": "#4c72b0", "6G": "#1a237e",
             "UAV": "#f9a825", "LEO": "#c44e52", "EDGE": "#55a868", "HQ": "#6a1b9a"}
    fig, ax = plt.subplots(figsize=(10.0, 6.2))

    # segment bands, drawn first so that everything else sits on top of them
    bands = [
        (-0.55, 0.55, "User equipment", "#eceff1"),
        (1.05, 2.15, "Terrestrial access (4G / 5G / 6G)", "#e8eaf6"),
        (2.35, 3.45, "Aerial relays and edge compute", "#e8f5e9"),
        (3.65, 4.75, "Non-terrestrial segment (LEO)", "#fdecea"),
        (4.95, 6.05, "Core and command", "#f3e5f5"),
    ]
    # reserve a clear left margin so that the band captions never collide with nodes
    node_xmin = min(x for x, _ in pos.values())
    xmin = node_xmin - 9.2
    xmax = max(x for x, _ in pos.values()) + 1.6
    for lo, hi, label, col in bands:
        ax.axhspan(lo, hi, xmin=0, xmax=1, facecolor=col, zorder=0)
        ax.text(node_xmin - 1.5, (lo + hi) / 2, label, fontsize=8.0, va="center",
                ha="right", color="#37474f", style="italic", zorder=1)

    nx.draw_networkx_edges(g, pos, ax=ax, alpha=0.30, width=0.9,
                           edge_color="#455a64")
    for ntype, col in color.items():
        nodes = [n for n, d in g.nodes(data=True) if d["node_type"] == ntype]
        if not nodes:
            continue
        nx.draw_networkx_nodes(g, pos, nodelist=nodes, node_color=col, node_size=980,
                               ax=ax, edgecolors="white", linewidths=1.4, label=ntype)
    nx.draw_networkx_labels(g, pos, font_size=6.4, font_color="white", ax=ax,
                            font_weight="bold")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.01), fontsize=8.0,
              ncol=8, frameon=False, handletextpad=1.4, columnspacing=1.6,
              borderpad=0.6, scatterpoints=1)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(-0.75, 6.25)
    ax.set_axis_off()
    summary = topology_summary(g)
    ax.set_title(
        "Reference topology. Every user group reaches the core through at least two "
        "disjoint segments;\n"
        f"|V| = {summary['nodes']}, |E| = {summary['links']}, diameter = {summary['diameter']}, "
        f"edge connectivity = {summary['edge_connectivity']}, "
        f"algebraic connectivity = {summary['algebraic_connectivity']:.3f}",
        fontsize=8.8, linespacing=1.5)
    fig.tight_layout()
    _save(fig, "fig02_topology.png")
    _save_table(pd.DataFrame([summary]), "table01_topology_summary")


# ---------------------------------------------------------------------------
# Fig. 3 - data and control pipeline
# ---------------------------------------------------------------------------
def fig_pipeline() -> None:
    fig, ax = plt.subplots(figsize=(9.6, 3.0))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3)
    ax.axis("off")
    stages = [
        ("Jump-diffusion\nhealth process", "#eceff1"),
        ("Observation map\n+ log-normal noise", "#f3e5f5"),
        ("Ring buffer\nW = 12 slots", "#e3f2fd"),
        ("57 features\nmean/std/slope/delta", "#e8f5e9"),
        ("Calibrated\nclassifier", "#fff8e1"),
        ("Risk-aware\norchestration", "#fbe9e7"),
        ("Resilience\nKPIs", "#e8eef7"),
    ]
    w = 1.28
    for i, (text, col) in enumerate(stages):
        x = 0.15 + i * 1.4
        ax.add_patch(mpatches.FancyBboxPatch((x, 1.0), w, 1.0,
                                             boxstyle="round,pad=0.08",
                                             edgecolor="#37474f", facecolor=col, lw=1.0))
        ax.text(x + w / 2, 1.5, text, ha="center", va="center", fontsize=7.4)
        if i < len(stages) - 1:
            ax.annotate("", xy=(x + w + 0.11, 1.5), xytext=(x + w + 0.01, 1.5),
                        arrowprops=dict(arrowstyle="-|>", lw=1.2, color="#37474f"))
    ax.text(5.0, 0.45, "label: node enters FAILED within H = 10 slots, "
                       "evaluated only on slots where the node is still operational",
            ha="center", fontsize=7.6, style="italic")
    _save(fig, "fig03_pipeline.png")


# ---------------------------------------------------------------------------
# Fig. 4-8 - predictor evaluation
# ---------------------------------------------------------------------------
def fig_model_results() -> None:
    comp = pd.read_csv(os.path.join(RESULTS, "model_comparison.csv"))
    npz = np.load(os.path.join(RESULTS, "test_predictions.npz"), allow_pickle=True)
    y = npz["y_true"]
    with open(os.path.join(RESULTS, "best_model.json")) as fh:
        meta = json.load(fh)
    best = meta["best_model"]

    # Fig. 4 bar comparison
    metrics = ["f1", "roc_auc", "pr_auc", "recall", "precision"]
    labels = ["F1", "ROC-AUC", "PR-AUC", "Recall", "Precision"]
    fig, ax = plt.subplots(figsize=(7.6, 3.4))
    width = 0.15
    xs = np.arange(len(metrics))
    for i, (_, row) in enumerate(comp.iterrows()):
        ax.bar(xs + (i - len(comp) / 2) * width + width / 2,
               [row[m] for m in metrics], width, label=row["model"].replace("_", " "))
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("score")
    ax.legend(fontsize=7.5, ncol=3)
    ax.set_title("Failure-prediction performance on episode-disjoint test data")
    _save(fig, "fig04_model_comparison.png")

    # Fig. 5 ROC
    fig, ax = plt.subplots(figsize=(4.4, 4.0))
    for key in npz.files:
        if not key.startswith("prob_"):
            continue
        name = key[5:]
        fpr, tpr, _ = roc_curve(y, npz[key])
        auc = comp.set_index("model").loc[name, "roc_auc"]
        ax.plot(fpr, tpr, lw=1.4, label=f"{name.replace('_', ' ')} ({auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.set_title("ROC curves")
    ax.legend(fontsize=7)
    _save(fig, "fig05_roc.png")

    # Fig. 6 PR
    fig, ax = plt.subplots(figsize=(4.4, 4.0))
    for key in npz.files:
        if not key.startswith("prob_"):
            continue
        name = key[5:]
        pr, rc, _ = precision_recall_curve(y, npz[key])
        ap = comp.set_index("model").loc[name, "pr_auc"]
        ax.plot(rc, pr, lw=1.4, label=f"{name.replace('_', ' ')} ({ap:.3f})")
    ax.axhline(y.mean(), color="k", ls="--", lw=0.8, label=f"prevalence ({y.mean():.4f})")
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.set_title("Precision-recall curves")
    ax.legend(fontsize=7)
    _save(fig, "fig06_precision_recall.png")

    # Fig. 7 confusion matrix
    prob = npz[f"prob_{best}"]
    threshold = float(comp.set_index("model").loc[best, "threshold"])
    cm = confusion_matrix(y, (prob >= threshold).astype(int))
    fig, ax = plt.subplots(figsize=(4.0, 3.6))
    ConfusionMatrixDisplay(cm, display_labels=["no failure", "failure <= H"]).plot(
        ax=ax, colorbar=False, cmap="Blues", values_format="d")
    ax.set_title(f"{best.replace('_', ' ')} at operating point {threshold:.3f}")
    ax.grid(False)
    _save(fig, "fig07_confusion_matrix.png")

    # Fig. 8 calibration + risk histogram
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.4))
    frac, mean_pred = calibration_curve(y, prob, n_bins=12, strategy="quantile")
    axes[0].plot(mean_pred, frac, "o-", color="#c44e52")
    axes[0].plot([0, max(mean_pred)], [0, max(mean_pred)], "k--", lw=0.8)
    axes[0].set_xlabel("predicted probability")
    axes[0].set_ylabel("observed frequency")
    axes[0].set_title("Reliability diagram")
    axes[1].hist(prob[y == 0], bins=60, alpha=0.7, label="negative", density=True)
    axes[1].hist(prob[y == 1], bins=60, alpha=0.7, label="positive", density=True)
    axes[1].axvline(threshold, color="k", ls="--", lw=1.0, label="operating point")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("risk score")
    axes[1].set_title("Risk-score distribution")
    axes[1].legend(fontsize=7.5)
    _save(fig, "fig08_calibration_risk.png")

    _save_table(comp.round(4), "table03_model_comparison")


def fig_feature_importance() -> None:
    path = os.path.join(RESULTS, "random_forest_feature_importance.csv")
    if not os.path.exists(path):
        return
    imp = pd.read_csv(path).head(18).iloc[::-1]
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    names = [n.replace("num__", "").replace("cat__", "") for n in imp["feature"]]
    ax.barh(names, imp["importance"], color="#4c72b0")
    ax.set_xlabel("mean decrease in impurity")
    ax.set_title("Most informative telemetry features (random forest)")
    _save(fig, "fig09_feature_importance.png")
    _save_table(pd.read_csv(path).head(18).round(5), "table04_feature_importance")


# ---------------------------------------------------------------------------
# Fig. 10-13 - orchestration results
# ---------------------------------------------------------------------------
def _bar_by_scenario(runs: pd.DataFrame, metric: str, ylabel: str, title: str,
                     name: str, percent: bool = False, logy: bool = False,
                     zoom: bool = False, annotate: bool = True) -> None:
    scen = [s for s in SCEN_LABEL if s in runs["scenario"].unique()]
    fig, ax = plt.subplots(figsize=(7.8, 3.8))
    width = 0.2
    xs = np.arange(len(scen))
    lo, hi = np.inf, -np.inf
    for i, pol in enumerate(POLICY_ORDER):
        means, errs = [], []
        for s in scen:
            sub = runs[(runs["scenario"] == s) & (runs["policy"] == pol)][metric]
            means.append(sub.mean() * (100 if percent else 1))
            errs.append(sub.std(ddof=1) * (100 if percent else 1) / np.sqrt(max(1, len(sub))))
        bars = ax.bar(xs + (i - 1.5) * width, means, width, yerr=errs, capsize=2.5,
                      label=POLICY_LABEL[pol], color=POLICY_COLOR[pol])
        lo = min(lo, min(np.array(means) - np.array(errs)))
        hi = max(hi, max(np.array(means) + np.array(errs)))
        if annotate:
            for b, m in zip(bars, means):
                ax.annotate(f"{m:.2f}" if not percent else f"{m:.2f}",
                            (b.get_x() + b.get_width() / 2, b.get_height()),
                            ha="center", va="bottom", fontsize=5.6, rotation=90,
                            xytext=(0, 6), textcoords="offset points")
    ax.set_xticks(xs)
    ax.set_xticklabels([SCEN_LABEL[s] for s in scen], fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=9)
    if logy:
        ax.set_yscale("symlog", linthresh=1.0)
    elif zoom:
        span = max(1e-9, hi - lo)
        ax.set_ylim(max(0.0, lo - 0.35 * span), hi + 0.45 * span)
    ax.legend(fontsize=7.5, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.14))
    _save(fig, name)


def fig_orchestration() -> None:
    runs = pd.read_csv(os.path.join(RESULTS, "experiment_runs.csv"))
    _bar_by_scenario(runs, "service_availability", "service availability [%]",
                     f"Service availability (mean +/- s.e., {int(runs.groupby([chr(39)+chr(39)]).size().max()) if False else runs.groupby(['scenario','policy']).size().max()} paired seeds)",
                     "fig10_availability.png", percent=True, zoom=True)
    _bar_by_scenario(runs, "mean_recovery_steps", "mean recovery time [slots]",
                     "Mean service recovery time after an outage",
                     "fig11_recovery_time.png")
    _bar_by_scenario(runs, "sla_compliance", "SLA compliance [%]",
                     "Fraction of flow-slots meeting the full service SLA",
                     "fig12_sla_compliance.png", percent=True, zoom=True)

    # per-class availability heat map for the proposed policy vs reactive
    classes = list(SERVICE_PROFILES.keys())
    scen = [s for s in SCEN_LABEL if s in runs["scenario"].unique()]
    delta = np.zeros((len(classes), len(scen)))
    for i, c in enumerate(classes):
        col = f"avail_{c}"
        for j, s in enumerate(scen):
            a = runs[(runs.scenario == s) & (runs.policy == "reactive")][col].mean()
            b = runs[(runs.scenario == s) & (runs.policy == "proactive_ai")][col].mean()
            delta[i, j] = 100.0 * (b - a)
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    lim = np.nanmax(np.abs(delta)) or 1.0
    im = ax.imshow(delta, cmap="RdYlGn", vmin=-lim, vmax=lim, aspect="auto")
    ax.set_xticks(range(len(scen)))
    ax.set_xticklabels([SCEN_LABEL[s] for s in scen], fontsize=7.5)
    ax.set_yticks(range(len(classes)))
    ax.set_yticklabels([c.replace("_", " ") for c in classes], fontsize=7.5)
    for i in range(len(classes)):
        for j in range(len(scen)):
            ax.text(j, i, f"{delta[i, j]:+.2f}", ha="center", va="center", fontsize=7)
    ax.set_title("Availability gain of the proposed policy over reactive restoration [pp]")
    ax.grid(False)
    fig.colorbar(im, ax=ax, shrink=0.8)
    _save(fig, "fig13_class_gain.png")


def fig_timeline() -> None:
    tl = pd.read_csv(os.path.join(RESULTS, "experiment_timeline.csv"))
    tl = tl[tl["scenario"] == "S4_disaster_collapse"]
    if tl.empty:
        return
    fig, axes = plt.subplots(3, 1, figsize=(7.8, 6.0), sharex=True)
    for pol in POLICY_ORDER:
        sub = tl[tl["policy"] == pol]
        axes[0].plot(sub["step"], sub["outage_fraction"].rolling(15, min_periods=1).mean(),
                     lw=1.2, label=POLICY_LABEL[pol], color=POLICY_COLOR[pol])
    axes[0].set_ylabel("flows in outage")
    axes[0].legend(fontsize=7, ncol=4)
    axes[0].set_title("Disaster-collapse scenario, seed 1000")

    sub = tl[tl["policy"] == "proactive_ai"]
    axes[1].plot(sub["step"], sub["failed_nodes"], lw=1.0, color="#37474f")
    axes[1].set_ylabel("failed nodes")

    axes[2].plot(sub["step"], sub["mean_risk"], lw=1.0, color="#c44e52",
                 label="mean predicted risk")
    ora = tl[tl["policy"] == "oracle"]
    axes[2].plot(ora["step"], ora["mean_risk"], lw=1.0, color="#55a868", alpha=0.7,
                 label="perfect-state risk")
    axes[2].set_ylabel("risk")
    axes[2].set_xlabel("slot")
    axes[2].legend(fontsize=7)
    _save(fig, "fig14_timeline.png")


def fig_sweep() -> None:
    path = os.path.join(RESULTS, "threshold_sweep.csv")
    if not os.path.exists(path):
        return
    sw = pd.read_csv(path)
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.2))
    for scen, group in sw.groupby("scenario"):
        agg = group.groupby("budget_scale").mean(numeric_only=True)
        axes[0].plot(agg.index, 100 * agg["service_availability"], "o-",
                     lw=1.2, ms=3.5, label=SCEN_LABEL.get(scen, scen))
        axes[1].plot(agg.index, agg["mean_recovery_steps"], "o-", lw=1.2, ms=3.5)
        axes[2].plot(agg.index, agg["proactive_reroutes_total"], "o-", lw=1.2, ms=3.5)
    axes[0].set_ylabel("availability [%]")
    axes[1].set_ylabel("recovery [slots]")
    axes[2].set_ylabel("pre-emptive migrations")
    for ax in axes:
        ax.set_xlabel(r"risk-budget scale $\kappa$  ($\tau_c=\kappa\,\phi_c\,\theta^*$)")
    axes[0].legend(fontsize=7)
    fig.suptitle("Sensitivity of the proposed policy to the risk budget", fontsize=9)
    _save(fig, "fig15_budget_sweep.png")
    _save_table(
        sw.groupby(["scenario", "budget_scale"])[
            ["service_availability", "sla_compliance", "mean_recovery_steps",
             "proactive_reroutes_total"]
        ].mean().reset_index().round(4),
        "table07_budget_sweep",
    )


def fig_deployment() -> None:
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")
    ax.add_patch(mpatches.FancyBboxPatch((0.2, 0.3), 9.6, 5.4, boxstyle="round,pad=0.1",
                                         edgecolor="#1565c0", facecolor="#f5f9ff", lw=1.4))
    ax.text(5.0, 5.45, "Kubernetes namespace  dtwin", ha="center", fontsize=9, color="#1565c0")
    pods = [
        (0.7, 3.6, "telemetry-service\n(Go, DaemonSet)\n:8083"),
        (3.1, 3.6, "digital-twin-service\n(Go, 2 replicas)\n:8080"),
        (5.5, 3.6, "predictor-service\n(FastAPI, 3 replicas)\n:8081"),
        (7.9, 3.6, "orchestrator-service\n(Go, 2 replicas)\n:8082"),
        (1.9, 1.2, "Prometheus\n+ ServiceMonitor"),
        (4.3, 1.2, "ConfigMap\nmodel + thresholds"),
        (6.7, 1.2, "Ingress\n/api/v1"),
    ]
    for x, y, text in pods:
        ax.add_patch(mpatches.FancyBboxPatch((x, y), 1.9, 1.2, boxstyle="round,pad=0.08",
                                             edgecolor="#37474f", facecolor="#ffffff", lw=1.0))
        ax.text(x + 0.95, y + 0.6, text, ha="center", va="center", fontsize=7.4)
    for x0, x1 in ((2.6, 3.1), (5.0, 5.5), (7.4, 7.9)):
        ax.annotate("", xy=(x1, 4.2), xytext=(x0, 4.2),
                    arrowprops=dict(arrowstyle="-|>", lw=1.2, color="#37474f"))
    ax.annotate("", xy=(4.05, 3.55), xytext=(4.05, 2.45),
                arrowprops=dict(arrowstyle="<|-|>", lw=1.0, color="#78909c"))
    ax.annotate("", xy=(8.85, 3.55), xytext=(7.65, 2.45),
                arrowprops=dict(arrowstyle="<|-", lw=1.0, color="#78909c"))
    ax.text(5.0, 0.55, "REST/JSON between services; Prometheus scrapes /metrics on every pod",
            ha="center", fontsize=7.6, style="italic")
    _save(fig, "fig16_deployment.png")


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------
def tables() -> None:
    from ..simulator.config import NODE_PROFILES

    node_tab = pd.DataFrame([
        {
            "node type": p.node_type,
            "base latency [ms]": p.base_latency_ms,
            "base jitter [ms]": p.base_jitter_ms,
            "capacity [Mbit/s]": p.capacity_mbps,
            "loss floor": p.base_loss,
            "RSSI [dBm]": p.base_signal_dbm,
            "hazard scale": p.hazard_scale,
            "battery backed": p.battery_backed,
        }
        for p in NODE_PROFILES.values()
    ])
    _save_table(node_tab, "table02_node_profiles")

    svc_tab = pd.DataFrame([
        {
            "service class": s.name.replace("_", " "),
            "priority": s.priority,
            "latency budget [ms]": s.max_latency_ms,
            "loss budget": s.max_loss,
            "bandwidth [Mbit/s]": s.min_bandwidth_mbps,
            "arrival weight": s.arrival_weight,
            "holding time [slots]": f"{s.duration_steps[0]}-{s.duration_steps[1]}",
        }
        for s in SERVICE_PROFILES.values()
    ])
    _save_table(svc_tab, "table05_service_classes")

    scen_tab = pd.DataFrame([
        {
            "scenario": s.name,
            "description": s.description,
            "load multiplier": s.load_multiplier,
            "hazard multipliers": ", ".join(f"{k}:{v}" for k, v in s.hazard_multiplier.items()) or "-",
            "shock rate": s.shock_prob,
            "destroyed elements": ", ".join(s.destroyed_nodes) or "-",
        }
        for s in SCENARIOS.values()
    ])
    _save_table(scen_tab, "table06_scenarios")

    runs = pd.read_csv(os.path.join(RESULTS, "experiment_runs.csv"))
    metrics = ["service_availability", "sla_compliance", "mean_latency_ms",
               "mean_packet_loss", "mean_recovery_steps", "avail_emergency_message",
               "avail_medical_telemetry", "proactive_reroutes_total", "reactive_reroutes"]
    rows = []
    for scen, group in runs.groupby("scenario"):
        for pol in POLICY_ORDER:
            sub = group[group["policy"] == pol]
            row = {"scenario": scen, "policy": POLICY_LABEL[pol]}
            for m in metrics:
                row[m] = sub[m].mean()
                row[f"{m}_sd"] = sub[m].std(ddof=1)
            rows.append(row)
    _save_table(pd.DataFrame(rows).round(4), "table08_policy_results")

    sig = pd.read_csv(os.path.join(RESULTS, "significance_tests.csv"))
    sig = sig[(sig["policy"] == "proactive_ai")]
    _save_table(
        sig[["scenario", "metric", "mean_reference", "mean_policy", "relative_change_pct",
             "cohens_dz", "p_value_ttest", "p_holm", "significant_005"]].round(5),
        "table09_significance",
    )

    # research-gap matrix (qualitative, Section 2.5)
    gap = pd.DataFrame(
        [
            ["Digital twin for RAN/NTN", "yes", "no", "partial", "no", "no"],
            ["Learned failure prediction", "no", "yes", "yes", "partial", "no"],
            ["LEO / NTN segment", "partial", "no", "yes", "yes", "no"],
            ["Closed-loop orchestration", "partial", "no", "no", "yes", "partial"],
            ["Humanitarian service classes", "no", "no", "no", "no", "yes"],
            ["Reproducible open pipeline", "no", "partial", "no", "no", "no"],
        ],
        columns=["capability", "DT surveys", "ML-NMS works", "NTN works",
                 "SDN restoration", "disaster-comm works"],
    )
    gap["this work"] = "yes"
    _save_table(gap, "table10_research_gap", float_fmt="%s")


def main() -> None:
    os.makedirs(FIGDIR, exist_ok=True)
    os.makedirs(TABDIR, exist_ok=True)
    fig_architecture()
    fig_topology()
    fig_pipeline()
    fig_model_results()
    fig_feature_importance()
    fig_orchestration()
    fig_timeline()
    fig_sweep()
    fig_deployment()
    tables()
    print("\nall assets written to paper-assets/")


if __name__ == "__main__":
    main()
