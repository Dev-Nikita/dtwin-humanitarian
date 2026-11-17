"""
Figures and tables of the revised manuscript (JRESS-D-26-06464, round 1).

    python -m python.experiments.make_figures_r1

Reuses the generic helpers of make_figures2 and regenerates every figure and
table whose content depends on the revised policy set, the corrected
interruption accounting or the revision experiments.
"""

from __future__ import annotations

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from . import make_figures2 as MF
from .resilience import SurvivalRecord, empirical_hazard, kaplan_meier

RESULTS, FIGDIR, TABDIR = MF.RESULTS, MF.FIGDIR, MF.TABDIR
_save, _table = MF._save, MF._table

LABEL = {
    "static": "P0 static",
    "reactive": "P1 reactive restoration",
    "protection": "P1b 1:1 protection",
    "proactive_threshold": "P2 threshold-proactive",
    "proactive_threshold_matched": "P2m threshold, matched rate",
    "degradation_aware_uniform": "P3u degradation-aware, uniform",
    "degradation_aware": "P3 degradation-aware migration",
    "proactive_ai_uniform": "P4u predictive, uniform",
    "proactive_ai": "P4 predictive migration",
    "oracle": "P5 perfect state, band map",
    "oracle_calibrated": "P5c perfect state, calibrated",
    "protection_index": "P6i protection + index",
    "protection_predictive": "P6 predictive protection",
    "protection_oracle": "P6o protection + perfect state",
    "protection_predictive_mig": "P6m P6 + migration",
    "protection_predictive_route": "P6-route risk-aware primary only",
    "protection_predictive_standby": "P6-standby risk-aware standby only",
    "protection_repair": "P1b-r protection + risk-free standby repair",
    "protection_predictive_slow": "P6-slow P6, refresh rate-limited x5",
}
SHORT = {k: v.split(" ")[0] for k, v in LABEL.items()}
# categorical slots of the validated default palette, fixed order (dataviz skill)
COLOR = {
    "reactive": "#2a78d6", "protection": "#eb6834", "degradation_aware": "#1baf7a",
    "proactive_ai": "#eda100", "oracle_calibrated": "#e87ba4",
    "protection_predictive": "#008300", "protection_oracle": "#4a3aa7",
    "proactive_threshold": "#8a8a8a", "proactive_threshold_matched": "#8a8a8a",
    "protection_index": "#4a3aa7", "static": "#8a8a8a",
}
HATCH = {"protection": "//", "protection_predictive": "//", "protection_oracle": "//"}
CORE = ["reactive", "protection", "degradation_aware", "proactive_ai", "oracle_calibrated",
        "protection_predictive", "protection_oracle"]
SCEN = MF.SCEN_LABEL
STRESSED = ["S2_bs_overload", "S3_leo_degradation", "S4_disaster_collapse", "S5_leo_handover",
            "S6_common_cause", "S7_external_ntn"]


def _runs():
    return MF._runs()


def _ci(x):
    x = np.asarray(x, float)
    n = len(x)
    return x.mean(), stats.t.ppf(0.975, n - 1) * x.std(ddof=1) / np.sqrt(n)


def _paired(df, a, b, metric, key="seed"):
    x = df[df.policy == a].sort_values(key)[metric].to_numpy(float)
    y = df[df.policy == b].sort_values(key)[metric].to_numpy(float)
    d = x - y
    m, h = _ci(d)
    sd = d.std(ddof=1)
    try:
        wp = stats.wilcoxon(x, y).pvalue if not np.allclose(d, 0) else 1.0
    except ValueError:
        wp = np.nan
    return {"mean_a": x.mean(), "mean_b": y.mean(), "diff": m, "lo": m - h, "hi": m + h,
            "rel": 100 * m / abs(y.mean()) if y.mean() else np.nan,
            "dz": m / sd if sd > 0 else np.nan, "p": wp, "n": len(d)}


# ===========================================================================
def r_decomposition():
    runs = _runs()
    scen = ["S3_leo_degradation", "S4_disaster_collapse", "S6_common_cause", "S7_external_ntn"]
    pols = [p for p in CORE if p != "reactive"]
    fig, axes = plt.subplots(1, len(scen), figsize=(12.0, 3.4), sharey=True)
    for ax, s in zip(axes, scen):
        sub = runs[runs.scenario == s]
        base = sub[sub.policy == "reactive"].sort_values("seed").resilience_loss.to_numpy()
        for i, pol in enumerate(pols):
            x = sub[sub.policy == pol].sort_values("seed").resilience_loss.to_numpy()
            m, h = _ci(100 * (x - base) / base.mean())
            ax.barh(i, m, 0.7, xerr=h, capsize=2.5, color=COLOR[pol], edgecolor="white",
                    linewidth=1.5, hatch=HATCH.get(pol), label=LABEL[pol])
            ax.text(m - h - 1.0 if m < 0 else m + h + 1.0, i, f"{m:+.1f}%",
                    va="center", ha="right" if m < 0 else "left", fontsize=6.8, color="#333333")
        ax.axvline(0, color="#2a78d6", lw=1.0)
        ax.set_yticks(range(len(pols)))
        ax.set_yticklabels([LABEL[p] for p in pols], fontsize=7)
        ax.invert_yaxis()
        ax.set_xlim(-38, 18)
        ax.set_title(SCEN[s], fontsize=8.5)
        ax.set_xlabel("change in resilience loss vs P1 [%]", fontsize=7.5)
    fig.suptitle("Paired change in resilience loss relative to reactive restoration P1 "
                 "(mean and 95% CI over 24 paired seeds; hatched = 1:1 protection family)",
                 fontsize=9)
    _save(fig, "r01_decomposition.png")


def pooled_recovery(sub, kind="failure"):
    """Mean recovery per interruption, pooled over all runs (runs without an
    interruption of that cause contribute nothing, not a zero)."""
    if kind == "failure":
        n, o = sub.failure_interruptions.sum(), sub.outage_slots_failure.sum()
    elif kind == "migration":
        n, o = sub.migration_interruptions.sum(), sub.outage_slots_migration.sum()
    else:
        n = sub.failure_interruptions.sum() + sub.migration_interruptions.sum()
        o = sub.outage_slots_failure.sum() + sub.outage_slots_migration.sum()
    return float(o / n) if n else float("nan")


def r_main_tables():
    runs = _runs()
    order = ["static", "reactive", "protection", "proactive_threshold",
             "proactive_threshold_matched", "degradation_aware_uniform", "degradation_aware",
             "proactive_ai_uniform", "proactive_ai", "oracle", "oracle_calibrated",
             "protection_index", "protection_predictive", "protection_oracle",
             "protection_predictive_mig"]
    # compact: resilience loss per stressed scenario for the core policies
    rows = []
    for pol in ["reactive", "protection", "proactive_threshold_matched", "degradation_aware",
                "proactive_ai", "oracle_calibrated", "protection_index",
                "protection_predictive", "protection_oracle"]:
        row = {"policy": LABEL[pol]}
        for s in STRESSED:
            row[SCEN[s].split(" ")[0]] = round(
                runs[(runs.scenario == s) & (runs.policy == pol)].resilience_loss.mean(), 2)
        rows.append(row)
    _table(pd.DataFrame(rows), "r_t_compact", fmt="%s")

    # full appendix table
    metrics = ["service_availability", "normalised_resilience", "resilience_loss",
               "mean_recovery_steps", "mean_recovery_failure", "sla_compliance",
               "proactive_reroutes_total", "protection_switches"]
    rows = []
    for s in MF._scen_list(runs):
        for pol in order:
            sub = runs[(runs.scenario == s) & (runs.policy == pol)]
            if sub.empty:
                continue
            row = {"scenario": SCEN[s], "policy": SHORT[pol]}
            for m in metrics:
                if m == "mean_recovery_steps":
                    row[m] = round(pooled_recovery(sub, "all"), 2)
                elif m == "mean_recovery_failure":
                    row[m] = round(pooled_recovery(sub, "failure"), 2)
                else:
                    row[m] = round(float(sub[m].mean()), 4 if ("avail" in m or "normal" in m
                                                              or "sla" in m) else 2)
            rows.append(row)
    df = pd.DataFrame(rows).rename(columns={
        "service_availability": "availability", "normalised_resilience": "Rn",
        "resilience_loss": "loss L", "mean_recovery_steps": "recovery, all",
        "mean_recovery_failure": "recovery, failure-induced", "sla_compliance": "SLA",
        "proactive_reroutes_total": "migrations", "protection_switches": "switch-overs"})
    df.fillna("–").to_csv(os.path.join(TABDIR, "r_t_full.csv"), index=False)

    # confirmatory family (pre-specified): P4 vs P1
    sig = pd.read_csv(os.path.join(RESULTS, "significance_tests.csv"))
    conf = sig[sig.family == "confirmatory"].copy()
    conf["scenario"] = conf["scenario"].map(lambda v: SCEN.get(v, v))
    conf["95% CI"] = conf.apply(lambda r: f"[{r['ci95_low']:.3g}, {r['ci95_high']:.3g}]", axis=1)
    conf["relative change [%]"] = conf["relative_change_pct"].round(1)
    conf["Cohen d_z"] = conf["cohens_dz"].round(2)
    conf["Holm p"] = conf["p_holm"].map(lambda v: "<0.0001" if v < 1e-4 else f"{v:.4f}")
    conf["metric"] = conf["metric"].map(lambda v: MF._METRIC_LABEL.get(v, v))
    conf["mean difference"] = conf["mean_difference"].map(lambda v: f"{v:.4g}")
    order_m = {"availability": 0, "resilience loss": 1, "normalised resilience": 2,
               "mean recovery [slots]": 3}
    conf = conf.assign(_o=conf.metric.map(order_m)).sort_values(["scenario", "_o"])
    conf[["scenario", "metric", "mean difference", "95% CI", "relative change [%]",
          "Cohen d_z", "Holm p"]].to_csv(os.path.join(TABDIR, "r_t_confirmatory.csv"),
                                         index=False)

    # exploratory decomposition contrasts on resilience loss
    pairs = [("protection", "reactive"), ("proactive_ai", "reactive"),
             ("degradation_aware", "reactive"), ("proactive_ai", "degradation_aware"),
             ("proactive_ai", "protection"), ("protection_predictive", "protection"),
             ("protection_predictive", "protection_index"),
             ("protection_oracle", "protection_predictive"),
             ("oracle_calibrated", "proactive_ai"),
             ("protection_predictive_mig", "protection_predictive"),
             ("proactive_ai_uniform", "degradation_aware_uniform"),
             ("degradation_aware", "degradation_aware_uniform"),
             ("proactive_ai", "proactive_ai_uniform")]
    rows = []
    for s in STRESSED[1:]:
        sub = runs[runs.scenario == s]
        for a, b in pairs:
            r = _paired(sub, a, b, "resilience_loss")
            rows.append({"scenario": SCEN[s], "contrast": f"{SHORT[a]} - {SHORT[b]}",
                         "mean difference": round(r["diff"], 3),
                         "95% CI": f"[{r['lo']:.2f}, {r['hi']:.2f}]",
                         "relative [%]": round(r["rel"], 1), "Cohen d_z": round(r["dz"], 2),
                         "Wilcoxon p": ("<0.0001" if r["p"] < 1e-4 else f"{r['p']:.4f}")})
    pd.DataFrame(rows).to_csv(os.path.join(TABDIR, "r_t_contrasts.csv"), index=False)

    # recovery decomposition: failure-induced vs migration-induced (R5.4)
    rows = []
    for s in ["S3_leo_degradation", "S4_disaster_collapse", "S6_common_cause"]:
        for pol in ["reactive", "protection", "degradation_aware", "proactive_ai",
                    "protection_predictive"]:
            sub = runs[(runs.scenario == s) & (runs.policy == pol)]
            rows.append({"scenario": SCEN[s], "policy": SHORT[pol],
                         "recovery, all [slots]": round(pooled_recovery(sub, "all"), 2),
                         "failure-induced interruptions": round(sub.failure_interruptions.mean(), 1),
                         "recovery, failure-induced [slots]": round(pooled_recovery(sub, "failure"), 2),
                         "migration-induced interruptions": round(sub.migration_interruptions.mean(), 1),
                         "recovery, migration-induced [slots]": round(pooled_recovery(sub, "migration"), 2)})
    pd.DataFrame(rows).fillna("–").to_csv(os.path.join(TABDIR, "r_t_recovery_split.csv"), index=False)

    # per-class SLA-compliant availability (R5.4)
    classes = ["command_control", "emergency_message", "medical_telemetry", "voice_call",
               "video_stream", "sensor_data"]
    rows = []
    s = "S4_disaster_collapse"
    for pol in ["reactive", "protection", "proactive_ai", "protection_predictive"]:
        sub = runs[(runs.scenario == s) & (runs.policy == pol)]
        row = {"policy": SHORT[pol]}
        for c in classes:
            row[f"{c.replace('_', ' ')} A"] = round(100 * sub[f"avail_{c}"].mean(), 2)
            row[f"{c.replace('_', ' ')} SLA-A"] = round(100 * sub[f"success_{c}"].mean(), 2)
        rows.append(row)
    pd.DataFrame(rows).to_csv(os.path.join(TABDIR, "r_t_class_sla.csv"), index=False)


def r_detection_delay():
    d = pd.read_csv(os.path.join(RESULTS, "detection_delay.csv"))
    pols = ["reactive", "protection", "degradation_aware", "proactive_ai", "oracle_calibrated",
            "protection_predictive"]
    scen = ["S4_disaster_collapse", "S6_common_cause"]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.4))
    rows = []
    for ax, s in zip(axes, scen):
        for pol in pols:
            g = d[(d.scenario == s) & (d.policy == pol)].groupby("detection_delay")
            m = g.resilience_loss.mean()
            h = g.resilience_loss.apply(lambda x: _ci(x)[1])
            ax.errorbar(m.index, m.values, yerr=h.values, marker="o", ms=4.5, lw=1.6, capsize=2,
                        color=COLOR[pol], label=LABEL[pol],
                        ls="--" if pol in HATCH else "-")
            for D, v in m.items():
                rows.append({"scenario": SCEN[s], "policy": SHORT[pol],
                             "detection delay D": int(D), "resilience loss": round(v, 2)})
        ax.set_xlabel("detection delay D [slots]")
        ax.set_xticks(sorted(d.detection_delay.unique()))
        ax.set_title(SCEN[s], fontsize=8.5)
    axes[0].set_ylabel("resilience loss [slot-equivalents]")
    axes[1].legend(fontsize=6.5, loc="upper left")
    fig.suptitle("The advantage of migration-based policies scales with the detection delay "
                 "(12 paired seeds)", fontsize=9)
    _save(fig, "r02_detection_delay.png")
    t = pd.DataFrame(rows).pivot_table(index=["scenario", "policy"], columns="detection delay D",
                                       values="resilience loss").reset_index()
    t.columns = [str(c) if not isinstance(c, (int, np.integer)) else f"D = {c}" for c in t.columns]
    t.to_csv(os.path.join(TABDIR, "r_t_detection.csv"), index=False)


def r_frontier():
    fr = pd.read_csv(os.path.join(RESULTS, "frontier.csv"))
    scen = ["S3_leo_degradation", "S4_disaster_collapse", "S5_leo_handover", "S6_common_cause"]
    fig, axes = plt.subplots(1, 4, figsize=(12.4, 3.3))
    for ax, s in zip(axes, scen):
        sub = fr[fr.scenario == s]
        for pol, knob in (("proactive_threshold", "kpi_scale"), ("degradation_aware", "kappa"),
                          ("proactive_ai", "kappa"), ("oracle_calibrated", "kappa"),
                          ("protection_predictive", "kappa")):
            g = sub[(sub.policy == pol) & (sub.knob == knob)].groupby("knob_value")[
                ["proactive_reroutes_total", "resilience_loss"]].mean()
            ax.plot(g.proactive_reroutes_total, g.resilience_loss, "o-", ms=4, lw=1.4,
                    color=COLOR[pol], label=LABEL[pol] if pol != "proactive_threshold"
                    else "P2 threshold-proactive")
        for pol in ("reactive", "protection"):
            v = sub[sub.policy == pol].resilience_loss.mean()
            ax.axhline(v, color=COLOR[pol], lw=1.0, ls=":", label=LABEL[pol])
        ax.set_xscale("symlog", linthresh=1)
        ax.set_xlim(left=-0.3)
        ax.set_xlabel("proactive migrations per run")
        ax.set_title(SCEN[s], fontsize=8.5)
    axes[0].set_ylabel("resilience loss")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, ncol=4, fontsize=7, loc="lower center", bbox_to_anchor=(0.5, -0.14))
    fig.suptitle("Resilience against actuation effort: operating points swept by the budget "
                 "scale kappa (P3-P6) and the alarm scale (P2), 12 paired seeds", fontsize=9)
    _save(fig, "r03_frontier.png")
    rows = []
    for s in scen:
        sub = fr[fr.scenario == s]
        for pol in ("proactive_threshold", "degradation_aware", "proactive_ai",
                    "oracle_calibrated", "protection_predictive"):
            g = sub[sub.policy == pol].groupby("knob_value")[
                ["resilience_loss", "proactive_reroutes_total"]].mean()
            best = g.resilience_loss.idxmin()
            rows.append({"scenario": SCEN[s], "policy": SHORT[pol],
                         "best setting": best, "loss at best": round(g.loc[best, "resilience_loss"], 2),
                         "migrations at best": round(g.loc[best, "proactive_reroutes_total"], 1)})
        for pol in ("reactive", "protection"):
            rows.append({"scenario": SCEN[s], "policy": SHORT[pol], "best setting": "-",
                         "loss at best": round(sub[sub.policy == pol].resilience_loss.mean(), 2),
                         "migrations at best": 0.0})
    pd.DataFrame(rows).to_csv(os.path.join(TABDIR, "r_t_frontier.csv"), index=False)


def r_migration_cost():
    mg = pd.read_csv(os.path.join(RESULTS, "migration_cost_sweep.csv"))
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.3))
    rows = []
    for ax, s in zip(axes, ["S3_leo_degradation", "S4_disaster_collapse"]):
        sub = mg[mg.scenario == s]
        ref = sub[sub.policy == "reactive"].resilience_loss.mean()
        ax.axhline(ref, color=COLOR["reactive"], ls=":", lw=1.0, label=LABEL["reactive"])
        for pol in ("proactive_ai", "protection", "protection_predictive"):
            g = sub[sub.policy == pol].groupby("migration_cost").resilience_loss.mean()
            ax.plot(g.index, g.values, "o-", ms=4, lw=1.5, color=COLOR[pol], label=LABEL[pol])
            for c, v in g.items():
                rows.append({"scenario": SCEN[s], "policy": SHORT[pol], "C_m": int(c),
                             "resilience loss": round(v, 2)})
        rows.append({"scenario": SCEN[s], "policy": "P1", "C_m": "-", "resilience loss": round(ref, 2)})
        ax.set_xlabel("switch-over / migration cost $C_m$ [slots]")
        ax.set_title(SCEN[s], fontsize=8.5)
    axes[0].set_ylabel("resilience loss")
    axes[1].legend(fontsize=6.8)
    _save(fig, "f11_migration_cost.png")
    t = pd.DataFrame(rows)
    t = t[t.policy != "P1"].pivot_table(index=["scenario", "policy"], columns="C_m",
                                        values="resilience loss").reset_index()
    t.columns = [f"C_m = {c}" if isinstance(c, (int, np.integer)) else c for c in t.columns]
    ref = mg[mg.policy == "reactive"].groupby("scenario").resilience_loss.mean()
    t["P1 reference"] = t["scenario"].map({SCEN[k]: round(v, 2) for k, v in ref.items()})
    t.round(2).to_csv(os.path.join(TABDIR, "r_t_migration.csv"), index=False)


def _sweep_table(path, key, name, pols=("reactive", "protection", "proactive_ai",
                                         "protection_predictive")):
    df = pd.read_csv(os.path.join(RESULTS, path))
    rows = []
    for setting, g in df.groupby(key, sort=False):
        row = {key: setting}
        ref = g[g.policy == "reactive"].resilience_loss.mean() if "reactive" in set(g.policy) else None
        for pol in pols:
            if pol not in set(g.policy):
                continue
            v = g[g.policy == pol].resilience_loss.mean()
            row[f"{SHORT[pol]} loss"] = round(v, 2)
        if ref:
            for pol in pols[1:]:
                if pol in set(g.policy):
                    v = g[g.policy == pol].resilience_loss.mean()
                    row[f"{SHORT[pol]} vs P1 [%]"] = round(100 * (v - ref) / ref, 1)
        rows.append(row)
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(TABDIR, name), index=False)
    return t


def r_sweeps():
    t = _sweep_table("topology_sweep.csv", "topology", "r_t_topology.csv")
    tp = pd.read_csv(os.path.join(RESULTS, "topology_sweep.csv"))
    if "topo_route_diversity" in tp:
        div = tp.groupby("topology").topo_route_diversity.first()
        t.insert(1, "route diversity", t.topology.map(div).round(2))
        t = t.sort_values("route diversity")
        t.to_csv(os.path.join(TABDIR, "r_t_topology.csv"), index=False)
    _sweep_table("parameter_sensitivity.csv", "setting", "r_t_parameters.csv")
    _sweep_table("controller_weight_sensitivity.csv", "setting", "r_t_weights.csv",
                 pols=("reactive", "proactive_ai", "protection_predictive"))
    for path, name, key in (("misspec_closed_loop.csv", "r_t_misspec.csv", "scenario"),
                            ("large_topology.csv", "r_t_large.csv", "scenario")):
        if not os.path.exists(os.path.join(RESULTS, path)):
            continue
        df = pd.read_csv(os.path.join(RESULTS, path))
        rows = []
        for s, g in df.groupby("scenario"):
            ref = g[g.policy == "reactive"].resilience_loss.mean()
            for pol in ["reactive", "protection", "degradation_aware", "proactive_ai",
                        "protection_index", "protection_predictive"]:
                gg = g[g.policy == pol]
                r = _paired(g, pol, "reactive", "resilience_loss") if pol != "reactive" else None
                rows.append({"scenario": SCEN[s], "policy": SHORT[pol],
                             "availability [%]": round(100 * gg.service_availability.mean(), 2),
                             "resilience loss": round(gg.resilience_loss.mean(), 2),
                             "vs P1 [%]": "-" if r is None else round(r["rel"], 1),
                             "95% CI of difference": "-" if r is None else f"[{r['lo']:.2f}, {r['hi']:.2f}]",
                             "migrations": round(gg.proactive_reroutes_total.mean(), 1)})
        pd.DataFrame(rows).to_csv(os.path.join(TABDIR, name), index=False)

    # design sensitivity (b_c profiles, Eq. 11 coefficients)
    ds = pd.read_csv(os.path.join(RESULTS, "design_sensitivity.csv"))
    ref = ds[ds.setting == "reactive reference"].resilience_loss.mean()
    prot = ds[ds.setting == "protection reference"].resilience_loss.mean()
    rows = []
    for setting, g in ds[~ds.setting.str.endswith("reference")].groupby("setting", sort=False):
        row = {"setting": setting}
        for pol in ("proactive_ai", "protection_predictive"):
            v = g[g.policy == pol].resilience_loss.mean()
            row[f"{SHORT[pol]} loss"] = round(v, 2)
            row[f"{SHORT[pol]} vs P1 [%]"] = round(100 * (v - ref) / ref, 1)
        row["P6 vs P1b [%]"] = round(100 * (g[g.policy == "protection_predictive"].resilience_loss.mean()
                                            - prot) / prot, 1)
        rows.append(row)
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(TABDIR, "r_t_design.csv"), index=False)

    # feedback ablation at fixed traffic
    fb = pd.read_csv(os.path.join(RESULTS, "feedback_ablation_fixed.csv"))
    rows = []
    for (c, s), g in fb.groupby(["coupling", "scenario"]):
        ref = g[g.policy == "reactive"].resilience_loss.mean()
        for pol in ("degradation_aware", "proactive_ai", "protection", "protection_predictive"):
            e = g[(g.policy == pol) & (g.feedback == "endogenous")].sort_values("seed")
            r = g[(g.policy == pol) & (g.feedback == "replayed_P1_load")].sort_values("seed")
            dd = e.resilience_loss.to_numpy() - r.resilience_loss.to_numpy()
            m, h = _ci(dd) if not np.allclose(dd, 0) else (0.0, 0.0)
            rows.append({"coupling": c, "scenario": SCEN[s], "policy": SHORT[pol],
                         "loss, endogenous load": round(e.resilience_loss.mean(), 2),
                         "loss, replayed P1 load": round(r.resilience_loss.mean(), 2),
                         "difference 95% CI": f"[{m - h:.2f}, {m + h:.2f}]"})
    pd.DataFrame(rows).to_csv(os.path.join(TABDIR, "r_t_feedback.csv"), index=False)


def f12_horizon_r1():
    path = os.path.join(RESULTS, "horizon_sweep.csv")
    hs = pd.read_csv(path).sort_values("horizon")
    ref_path = os.path.join(RESULTS, "horizon_reactive_reference.csv")
    ref = pd.read_csv(ref_path).resilience_loss.mean() if os.path.exists(ref_path) else None
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.2))
    axes[0].plot(hs.horizon, hs.pr_auc, "o-", color="#2a78d6", ms=4, label="PR-AUC")
    axes[0].plot(hs.horizon, hs.lift / hs.lift.max(), "s--", color="#eb6834", ms=4,
                 label="lift (scaled to max)")
    axes[0].set_title("Prediction quality", fontsize=9)
    axes[0].legend(fontsize=7)
    axes[1].plot(hs.horizon, hs.cl_lead_time_median, "o-", color="#eda100", ms=4,
                 label="median")
    axes[1].plot(hs.horizon, hs.horizon, ":", color="#8a8a8a", lw=1, label="bound T_lead = H")
    axes[1].set_ylabel("closed-loop lead time [slots]")
    axes[1].set_title("Usable warning (P4)", fontsize=9)
    axes[1].legend(fontsize=7)
    axes[2].plot(hs.horizon, hs.cl_resilience_loss, "o-", color=COLOR["proactive_ai"], ms=4,
                 label=LABEL["proactive_ai"])
    if "p6_resilience_loss" in hs:
        axes[2].plot(hs.horizon, hs.p6_resilience_loss, "o-", color=COLOR["protection_predictive"],
                     ms=4, label=LABEL["protection_predictive"])
    if ref is not None:
        axes[2].axhline(ref, color=COLOR["reactive"], ls=":", lw=1, label=LABEL["reactive"])
    axes[2].set_ylabel("resilience loss, S4")
    axes[2].set_title("System-level outcome", fontsize=9)
    axes[2].legend(fontsize=6.5)
    for ax in axes:
        ax.set_xlabel("prediction horizon $H$ [slots]")
    _save(fig, "f12_horizon.png")
    cols = ["horizon", "prevalence", "pr_auc", "recall", "lift", "cl_lead_time_median",
            "cl_lead_time_mean", "cl_anticipation_rate", "cl_resilience_loss",
            "cl_migrations", "p6_resilience_loss"]
    t = hs[[c for c in cols if c in hs]].copy()
    t = t.rename(columns={"prevalence": "prevalence", "pr_auc": "PR-AUC", "lift": "lift",
                          "cl_lead_time_median": "median lead [slots]",
                          "cl_lead_time_mean": "mean lead [slots]",
                          "cl_anticipation_rate": "anticipation rate",
                          "cl_resilience_loss": "P4 loss", "cl_migrations": "P4 migrations",
                          "p6_resilience_loss": "P6 loss"})
    t.round(4).to_csv(os.path.join(TABDIR, "r_t_horizon.csv"), index=False)


def f08_survival_r1():
    sv = pd.read_csv(os.path.join(RESULTS, "survival_records.csv"))
    pols = ["reactive", "protection", "proactive_ai", "protection_predictive"]
    scen = ["S3_leo_degradation", "S4_disaster_collapse", "S6_common_cause"]
    fig, axes = plt.subplots(2, 3, figsize=(10.2, 5.8))
    rows = []
    for r_, (dk, ek, yl) in enumerate([("duration", "event", "P(no interruption)"),
                                       ("duration_sustained", "event_sustained",
                                        "P(no sustained interruption)")]):
        for ax, s in zip(axes[r_], scen):
            for pol in pols:
                sub = sv[(sv.scenario == s) & (sv.policy == pol)]
                recs = [SurvivalRecord(int(d), int(e)) for d, e in zip(sub[dk], sub[ek])]
                t, S, _ = kaplan_meier(recs)
                ax.step(t, S, where="post", lw=1.4, color=COLOR[pol], label=LABEL[pol],
                        ls="--" if pol in HATCH else "-")
                rows.append({"definition": "any" if r_ == 0 else "sustained (>= 2 slots)",
                             "scenario": SCEN[s], "policy": SHORT[pol],
                             "S(10)": round(float(np.interp(10, t, S)), 4),
                             "S(20)": round(float(np.interp(20, t, S)), 4),
                             "S(40)": round(float(np.interp(40, t, S)), 4),
                             "flows": len(recs)})
            ax.set_xlim(0, 60)
            if r_ == 0:
                ax.set_title(SCEN[s], fontsize=8.5)
            else:
                ax.set_xlabel("slots since admission")
        axes[r_][0].set_ylabel(yl, fontsize=8)
    axes[0][0].legend(fontsize=6.5)
    _save(fig, "f08_survival.png")
    pd.DataFrame(rows).to_csv(os.path.join(TABDIR, "r_t_survival.csv"), index=False)


def f07_timeline_r1():
    tl = pd.read_csv(os.path.join(RESULTS, "experiment_timeline.csv"))
    tl = tl[tl.scenario == "S4_disaster_collapse"]
    fig, axes = plt.subplots(3, 1, figsize=(7.8, 5.8), sharex=True)
    for pol in ["reactive", "protection", "proactive_ai", "protection_predictive"]:
        sub = tl[tl.policy == pol]
        axes[0].plot(sub.step, 100 * (1 - sub.outage_fraction.rolling(15, min_periods=1).mean()),
                     lw=1.2, label=LABEL[pol], color=COLOR[pol])
    axes[0].set_ylabel("service quality $Q(t)$ [%]")
    axes[0].legend(fontsize=6.5, ncol=2)
    axes[0].set_title("Infrastructure-collapse episode (seed 1000, 15-slot moving average)",
                      fontsize=9)
    sub = tl[tl.policy == "proactive_ai"]
    axes[1].plot(sub.step, sub.failed_nodes, lw=1.0, color="#37474f")
    axes[1].set_ylabel("failed elements")
    axes[2].plot(sub.step, sub.mean_risk, lw=1.0, color=COLOR["proactive_ai"], label="predicted risk")
    ora = tl[tl.policy == "oracle_calibrated"]
    axes[2].plot(ora.step, ora.mean_risk, lw=1.0, color=COLOR["oracle_calibrated"],
                 label="calibrated perfect-state risk")
    axes[2].set_ylabel("mean risk")
    axes[2].set_xlabel("slot")
    axes[2].legend(fontsize=7)
    _save(fig, "f07_timeline.png")


def f04_decision_r1():
    fig, ax = plt.subplots(figsize=(9.0, 3.9))
    ax.set_xlim(0, 10.6)
    ax.set_ylim(0, 4.0)
    ax.axis("off")
    stages = [("telemetry\nwindow $X_n(t)$", "#eceff1"),
              ("calibrated risk\n$r_n(t)=P(\\tau_n\\leq H)$", "#e3f2fd"),
              ("path risk\n$R_\\infty(P)=\\max_n r_n$", "#e8f5e9"),
              ("class budget\n$\\tau_c=\\kappa b_c\\theta^*$", "#fff8e1")]
    w = 1.7
    for i, (text, col) in enumerate(stages):
        x = 0.2 + i * 1.95
        ax.add_patch(mpatches.FancyBboxPatch((x, 1.6), w, 1.0, boxstyle="round,pad=0.09",
                                             edgecolor="#37474f", facecolor=col, lw=1.0))
        ax.text(x + w / 2, 2.1, text, ha="center", va="center", fontsize=7.6)
        if i < len(stages) - 1:
            ax.annotate("", xy=(x + w + 0.2, 2.1), xytext=(x + w + 0.03, 2.1),
                        arrowprops=dict(arrowstyle="-|>", lw=1.2, color="#37474f"))
    x0 = 0.2 + 4 * 1.95
    for y, text, col in ((2.85, "P4: migrate live flow\nmake-before-break, cost $C_m$", "#fbe9e7"),
                         (1.0, "P6: re-select standby path\nno interruption", "#e8f5e9")):
        ax.add_patch(mpatches.FancyBboxPatch((x0, y - 0.45), 2.35, 0.9, boxstyle="round,pad=0.09",
                                             edgecolor="#37474f", facecolor=col, lw=1.0))
        ax.text(x0 + 1.17, y, text, ha="center", va="center", fontsize=7.4)
        ax.annotate("", xy=(x0 - 0.02, y), xytext=(x0 - 0.2 + 0.02, 2.1),
                    arrowprops=dict(arrowstyle="-|>", lw=1.1, color="#37474f"))
    ax.text(5.3, 0.15, "ex-post reading of the budget: migrating pays when $r\\,C_f > C_m$, "
            "i.e. above $p^*=C_m/C_f$ (heuristic, not derived)", ha="center", fontsize=8,
            style="italic")
    ax.text(5.3, 3.75, "Where the risk estimate is spent", ha="center", fontsize=9.5)
    _save(fig, "f04_decision_mechanism.png")


def f19_seed_convergence_r1():
    runs = pd.read_csv(os.path.join(RESULTS, "experiment_runs.csv"))
    s = "S4_disaster_collapse"
    rows = []
    for a, b in (("proactive_ai", "reactive"), ("protection_predictive", "protection")):
        x = runs[(runs.scenario == s) & (runs.policy == a)].sort_values("seed").resilience_loss.to_numpy()
        y = runs[(runs.scenario == s) & (runs.policy == b)].sort_values("seed").resilience_loss.to_numpy()
        d = x - y
        for n in (8, 12, 16, 20, 24):
            m, h = _ci(d[:n])
            rows.append({"contrast": f"{SHORT[a]} - {SHORT[b]}", "seeds": n,
                         "mean difference": round(m, 3), "ci half-width": round(h, 3)})
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(TABDIR, "r_t_seeds.csv"), index=False)
    fig, ax = plt.subplots(figsize=(5.8, 3.2))
    for (c, g), col in zip(t.groupby("contrast", sort=False), ("#eda100", "#008300")):
        ax.errorbar(g.seeds, g["mean difference"], yerr=g["ci half-width"], fmt="o-", ms=4,
                    capsize=3, color=col, label=c)
    ax.axhline(0, color="k", lw=0.8, ls=":")
    ax.set_xlabel("paired seeds")
    ax.set_ylabel("paired difference in resilience loss, S4")
    ax.legend(fontsize=7)
    _save(fig, "f19_seed_convergence.png")


def _pfmt(v):
    return "<0.0001" if v < 1e-4 else f"{v:.4f}"


def r_confirmation():
    """Tables and figure of the frozen confirmation experiment (R1_CONFIRMATION_PROTOCOL.md)."""
    path = os.path.join(RESULTS, "confirmation_tests.csv")
    if not os.path.exists(path):
        print("confirmation results missing")
        return
    t = pd.read_csv(path)
    eq = pd.read_csv(os.path.join(RESULTS, "confirmation_equivalence.csv"))
    conf = pd.read_csv(os.path.join(RESULTS, "confirmation.csv"))

    def lab(c):
        a, b = [x.strip() for x in c.split(" - ")]
        return f"{SHORT[a]} - {SHORT[b]}"

    rows = []
    for _, r in t[t.family != "exploratory"].iterrows():
        rows.append({"family": r.family.split(" ")[0], "scenario": SCEN[r.scenario],
                     "contrast": lab(r.contrast), "mean difference": round(r.mean_diff, 3),
                     "95% CI": f"[{r.ci_low:.2f}, {r.ci_high:.2f}]",
                     "relative [%]": round(r.rel_pct, 1), "Cohen d_z": round(r.dz, 2),
                     "Wilcoxon p": _pfmt(r.p_wilcoxon), "Holm p": _pfmt(r.p_holm)})
    pd.DataFrame(rows).to_csv(os.path.join(TABDIR, "r_t_conf_tests.csv"), index=False)

    rows = []
    for _, r in eq.iterrows():
        rows.append({"scenario": SCEN[r.scenario], "contrast": lab(r.contrast),
                     "margin": f"±{r.margin:.2f}", "mean difference": round(r.mean_diff, 3),
                     "90% CI": f"[{r.ci90_low:.3f}, {r.ci90_high:.3f}]",
                     "TOST p": _pfmt(r.p_tost),
                     "equivalent": "yes" if r.equivalent else "not established"})
    pd.DataFrame(rows).to_csv(os.path.join(TABDIR, "r_t_conf_equiv.csv"), index=False)

    rows = []
    for _, r in t[(t.family == "exploratory") & t.contrast.str.startswith("protection_oracle")].iterrows():
        rows.append({"scenario": SCEN[r.scenario], "contrast": lab(r.contrast),
                     "mean difference": round(r.mean_diff, 3),
                     "95% CI": f"[{r.ci_low:.3f}, {r.ci_high:.3f}]",
                     "relative [%]": round(r.rel_pct, 2)})
    pd.DataFrame(rows).to_csv(os.path.join(TABDIR, "r_t_conf_headroom.csv"), index=False)

    pols = ["reactive", "protection", "degradation_aware", "proactive_ai", "protection_index",
            "protection_predictive", "protection_oracle"]
    rows = []
    for pol in pols:
        row = {"policy": LABEL[pol]}
        for s in sorted(conf.scenario.unique()):
            m, h = _ci(conf[(conf.scenario == s) & (conf.policy == pol)].resilience_loss)
            row[SCEN[s].split(" ")[0]] = f"{m:.2f} ± {h:.2f}"
        rows.append(row)
    pd.DataFrame(rows).to_csv(os.path.join(TABDIR, "r_t_conf_means.csv"), index=False)

    # figure: paired contrasts of B1 and B2 with Holm-adjusted significance
    order = ["protection - reactive", "proactive_ai - reactive", "proactive_ai - protection",
             "protection_predictive - proactive_ai", "protection_predictive - protection"]
    scen = sorted(conf.scenario.unique())
    fig, axes = plt.subplots(1, len(scen), figsize=(12.0, 3.2), sharey=True)
    cols = ["#eb6834", "#eda100", "#8a8a8a", "#4a3aa7", "#008300"]
    for ax, s in zip(axes, scen):
        for i, c in enumerate(order):
            r = t[(t.scenario == s) & (t.contrast == c) & (t.family != "exploratory")]
            if r.empty:
                ax.text(0, i, "not tested", fontsize=6.5, va="center", ha="center",
                        color="#8a8a8a")
                continue
            r = r.iloc[0]
            m = 100 * r.mean_diff / r.mean_b
            lo, hi = 100 * r.ci_low / r.mean_b, 100 * r.ci_high / r.mean_b
            ax.barh(i, m, 0.65, color=cols[i], edgecolor="white", linewidth=1.2,
                    alpha=1.0 if r.p_holm < 0.05 else 0.35)
            ax.errorbar(m, i, xerr=[[m - lo], [hi - m]], color="#333333", capsize=2.5, lw=1)
            ax.text(hi + 0.8 if m >= 0 else lo - 0.8, i, f"{m:+.1f}%",
                    ha="left" if m >= 0 else "right", va="center", fontsize=6.6)
        ax.axvline(0, color="k", lw=0.8)
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels([lab(c) for c in order], fontsize=7.5)
        ax.invert_yaxis()
        ax.set_xlim(-33, 16)
        ax.set_title(SCEN[s], fontsize=8.5)
        ax.set_xlabel("paired difference [% of second policy]", fontsize=7.5)
    fig.suptitle("Frozen confirmation on 64 new seeds: paired contrasts in resilience loss "
                 "(95% CI; faded = not significant after Holm)", fontsize=9)
    _save(fig, "r04_confirmation.png")


def r_migration_classes():
    """Ground-truth classification of migration triggers (development campaign + confirmation)."""
    rows = []
    keys = ["mig_trigger_already_failed", "mig_trigger_fails_within_H",
            "mig_trigger_degraded_no_failure", "mig_trigger_false_alarm"]
    names = ["already failed (detection-delay bypass)", "fails within H",
             "degraded, no failure within H", "false alarm"]
    for src, df in (("development", _runs()),
                    ("confirmation", pd.read_csv(os.path.join(RESULTS, "confirmation.csv")))):
        for s in ["S3_leo_degradation", "S4_disaster_collapse", "S5_leo_handover",
                  "S6_common_cause"]:
            for pol in ["degradation_aware", "proactive_ai", "proactive_threshold_matched",
                        "oracle_calibrated"]:
                sub = df[(df.scenario == s) & (df.policy == pol)]
                if sub.empty or keys[0] not in sub:
                    continue
                tot = sub[keys].sum(axis=1).mean()
                row = {"seeds": src, "scenario": SCEN[s], "policy": SHORT[pol],
                       "triggers per run": round(tot, 1)}
                for k, n in zip(keys, names):
                    row[n + " [%]"] = round(100 * sub[k].mean() / tot, 1) if tot else 0.0
                rows.append(row)
    pd.DataFrame(rows).to_csv(os.path.join(TABDIR, "r_t_migration_classes.csv"), index=False)


def r_protection_modes():
    path = os.path.join(RESULTS, "protection_modes.csv")
    if not os.path.exists(path):
        print("protection_modes missing")
        return
    d = pd.read_csv(path)
    mcol = "protection_mode_setting"
    rows = []
    for s in ["S4_disaster_collapse", "S6_common_cause"]:
        for mode in ("feasible", "reserved", "optimistic"):
            sub = d[(d.scenario == s) & (d[mcol] == mode)]
            row = {"scenario": SCEN[s], "standby model": mode}
            for pol in ("reactive", "protection", "protection_index", "protection_predictive"):
                row[SHORT[pol]] = round(sub[sub.policy == pol].resilience_loss.mean(), 2)
            r = _paired(sub, "protection_predictive", "protection", "resilience_loss")
            row["P6 - P1b"] = f"{r['diff']:+.2f} [{r['lo']:.2f}, {r['hi']:.2f}]"
            row["infeasible switches / run (P1b)"] = round(
                sub[sub.policy == "protection"].switch_infeasible.mean(), 2)
            rows.append(row)
    pd.DataFrame(rows).to_csv(os.path.join(TABDIR, "r_t_protection_modes.csv"), index=False)


def r_accounting_check():
    path = os.path.join(RESULTS, "accounting_check.csv")
    if not os.path.exists(path):
        return
    d = pd.read_csv(path)
    rows = []
    for s in ["S3_leo_degradation", "S4_disaster_collapse", "S6_common_cause"]:
        sub = d[d.scenario == s]
        p1 = sub[(sub.policy == "reactive")].resilience_loss.mean()
        for pol in ("degradation_aware", "proactive_ai", "proactive_threshold_matched",
                    "oracle_calibrated"):
            row = {"scenario": SCEN[s], "policy": SHORT[pol]}
            for flag, name in ((False, "dead-path migration charged C_m"),
                               (True, "dead-path migration charged C_r")):
                v = sub[(sub.policy == pol) & (sub.flag == flag)].resilience_loss.mean()
                row[name] = round(v, 2)
                row[name + " vs P1 [%]"] = round(100 * (v - p1) / p1, 1)
            rows.append(row)
    pd.DataFrame(rows).to_csv(os.path.join(TABDIR, "r_t_accounting.csv"), index=False)


def r_p6_decomposition():
    """Exploratory post-confirmation ablation of P6 on the 64 confirmation seeds."""
    path = os.path.join(RESULTS, "p6_decomposition.csv")
    if not os.path.exists(path):
        print("p6_decomposition missing")
        return
    d = pd.read_csv(path)
    c = pd.read_csv(os.path.join(RESULTS, "confirmation.csv"))
    c = c[(c.policy == "protection") & c.scenario.isin(d.scenario.unique())]
    d = pd.concat([d, c], ignore_index=True)
    pairs = [("protection_predictive_route", "protection"),
             ("protection_predictive_standby", "protection"),
             ("protection_predictive", "protection"),
             ("protection_predictive", "protection_predictive_route"),
             ("protection_predictive", "protection_predictive_standby")]
    rows = []
    for s in ["S4_disaster_collapse", "S6_common_cause"]:
        sub = d[d.scenario == s]
        for a, b in pairs:
            r = _paired(sub, a, b, "resilience_loss")
            rows.append({"scenario": SCEN[s], "contrast": f"{SHORT[a]} - {SHORT[b]}",
                         "loss of first": round(r["mean_a"], 2),
                         "paired difference [95% CI]": f"{r['diff']:+.3f} [{r['lo']:.3f}, {r['hi']:.3f}]",
                         "relative [%]": round(r["rel"], 2), "n": r["n"]})
    pd.DataFrame(rows).to_csv(os.path.join(TABDIR, "r_t_p6_decomposition.csv"), index=False)


def r_refresh_control():
    """Exploratory post-confirmation controls for the P6 - P1b contrast on the 64
    confirmation seeds: risk-free standby repair (P1b-r) and a rate-limited P6."""
    path = os.path.join(RESULTS, "refresh_control.csv")
    if not os.path.exists(path):
        print("refresh_control missing")
        return
    d = pd.read_csv(path)
    pairs = [("protection_repair", "protection"),
             ("protection_predictive", "protection_repair"),
             ("protection_predictive", "protection"),
             ("protection_predictive_slow", "protection_predictive")]
    rows, ops = [], []
    for s in ["S4_disaster_collapse", "S6_common_cause"]:
        sub = d[d.scenario == s]
        for a, b in pairs:
            r = _paired(sub, a, b, "resilience_loss")
            rows.append({"scenario": SCEN[s], "contrast": f"{SHORT[a]} - {SHORT[b]}",
                         "loss of first": f"{r['mean_a']:.2f}",
                         "paired difference [95% CI]": f"{r['diff']:+.3f} [{r['lo']:.3f}, {r['hi']:.3f}]",
                         "relative [%]": f"{r['rel']:+.2f}", "Wilcoxon p": round(r["p"], 4),
                         "n": r["n"]})
        for pol in ("protection", "protection_repair", "protection_predictive",
                    "protection_predictive_slow"):
            q = sub[sub.policy == pol]
            ops.append({"scenario": SCEN[s], "policy": SHORT[pol],
                        "loss": f"{q.resilience_loss.mean():.2f}",
                        "standby refreshes": f"{q.backup_refreshes.mean():.1f}",
                        "path computations": f"{q.path_computations.mean():.0f}",
                        "protection switches": round(q.protection_switches.mean(), 1),
                        "switch-overs with no feasible standby": round(q.switch_infeasible.mean(), 1),
                        "reactive restorations": round(q.reactive_reroutes.mean(), 1)})
    pd.DataFrame(rows).to_csv(os.path.join(TABDIR, "r_t_refresh_control.csv"), index=False)
    pd.DataFrame(ops).to_csv(os.path.join(TABDIR, "r_t_control_ops.csv"), index=False)


def main():
    MF.f03_degradation_trace()
    f04_decision_r1()
    MF.f05_predictor()
    MF.f06_lead_time()
    f07_timeline_r1()
    f08_survival_r1()
    r_decomposition()
    r_main_tables()
    r_detection_delay()
    r_frontier()
    r_migration_cost()
    f12_horizon_r1()
    MF.f13_generalisation()
    MF.f14_noise()
    r_sweeps()
    f19_seed_convergence_r1()
    for fn in (r_confirmation, r_migration_classes, r_protection_modes, r_accounting_check,
               r_p6_decomposition, r_refresh_control):
        fn()
    try:
        MF.tables()
    except Exception as exc:  # legacy summary tables are optional
        print("legacy tables skipped:", exc)


if __name__ == "__main__":
    main()
