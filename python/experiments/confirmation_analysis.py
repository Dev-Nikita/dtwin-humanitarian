"""
Analysis of the frozen revision-confirmatory experiment (R1_CONFIRMATION_PROTOCOL.md).
Written and committed before any confirmation seed was run.

    python -m python.experiments.confirmation_analysis
"""

from __future__ import annotations

import json
import os
import subprocess

import numpy as np
import pandas as pd
from scipy import stats

RES = os.path.join("data", "results")
S3, S4, S5, S6 = ("S3_leo_degradation", "S4_disaster_collapse", "S5_leo_handover",
                  "S6_common_cause")
MARGIN = {S4: 0.40, S6: 0.81}


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    m = len(p)
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def paired(df, scen, a, b):
    sub = df[df.scenario == scen]
    x = sub[sub.policy == a].sort_values("seed").resilience_loss.to_numpy(float)
    y = sub[sub.policy == b].sort_values("seed").resilience_loss.to_numpy(float)
    assert len(x) == len(y) and len(x) > 2
    d = x - y
    n = len(d)
    sd = d.std(ddof=1)
    half = stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n)
    t_p = stats.ttest_rel(x, y).pvalue
    try:
        w_p = stats.wilcoxon(x, y).pvalue
    except ValueError:
        w_p = np.nan
    return {"scenario": scen, "contrast": f"{a} - {b}", "n": n, "mean_a": x.mean(),
            "mean_b": y.mean(), "mean_diff": d.mean(), "ci_low": d.mean() - half,
            "ci_high": d.mean() + half, "rel_pct": 100 * d.mean() / y.mean(),
            "dz": d.mean() / sd if sd > 0 else np.nan, "p_t": t_p, "p_wilcoxon": w_p}


def tost(df, scen, a, b, margin):
    sub = df[df.scenario == scen]
    x = sub[sub.policy == a].sort_values("seed").resilience_loss.to_numpy(float)
    y = sub[sub.policy == b].sort_values("seed").resilience_loss.to_numpy(float)
    d = x - y
    n = len(d)
    se = d.std(ddof=1) / np.sqrt(n)
    t_low = (d.mean() + margin) / se
    t_high = (d.mean() - margin) / se
    p_low = 1 - stats.t.cdf(t_low, n - 1)
    p_high = stats.t.cdf(t_high, n - 1)
    ci90 = stats.t.ppf(0.95, n - 1) * se
    return {"scenario": scen, "contrast": f"{a} - {b}", "margin": margin,
            "mean_diff": d.mean(), "ci90_low": d.mean() - ci90, "ci90_high": d.mean() + ci90,
            "p_tost": max(p_low, p_high), "equivalent": bool(max(p_low, p_high) < 0.05)}


def main():
    df = pd.read_csv(os.path.join(RES, "confirmation.csv"))
    rows = []
    b1 = [paired(df, s, "protection_predictive", "protection") for s in (S4, S6)]
    for r, p in zip(b1, holm([r["p_t"] for r in b1])):
        r.update(family="B1 primary", p_holm=p,
                 supported=bool(p < 0.05 and r["mean_diff"] < 0))
    rows += b1
    b2 = []
    for s in (S3, S4, S5, S6):
        for a, b in (("protection", "reactive"), ("proactive_ai", "reactive"),
                     ("proactive_ai", "protection"), ("protection_predictive", "proactive_ai")):
            b2.append(paired(df, s, a, b))
    for r, p in zip(b2, holm([r["p_t"] for r in b2])):
        r.update(family="B2 secondary", p_holm=p)
    rows += b2
    for s in (S3, S4, S5, S6):
        for a, b in (("protection_oracle", "protection_predictive"),
                     ("protection_oracle", "protection_index"),
                     ("protection_index", "protection"),
                     ("proactive_ai", "degradation_aware"),
                     ("protection_predictive", "reactive"),
                     ("degradation_aware", "reactive")):
            r = paired(df, s, a, b)
            r.update(family="exploratory")
            rows.append(r)
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(RES, "confirmation_tests.csv"), index=False)
    eq = pd.DataFrame([tost(df, s, "protection_predictive", "protection_index", MARGIN[s])
                       for s in (S4, S6)])
    eq.to_csv(os.path.join(RES, "confirmation_equivalence.csv"), index=False)
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                              text=True).stdout.strip()
    except Exception:
        head = "unknown"
    manifest = {"protocol": "R1_CONFIRMATION_PROTOCOL.md", "analysis_commit": head,
                "n_runs": int(len(df)), "seeds": sorted(int(x) for x in df.seed.unique()),
                "scenarios": sorted(df.scenario.unique().tolist()),
                "policies": sorted(df.policy.unique().tolist())}
    json.dump(manifest, open(os.path.join(RES, "confirmation_manifest.json"), "w"), indent=1)
    pd.set_option("display.width", 220)
    print(out[out.family != "exploratory"][["family", "scenario", "contrast", "mean_diff",
                                             "ci_low", "ci_high", "rel_pct", "p_holm"]].round(4))
    print(eq.round(4))


if __name__ == "__main__":
    main()
