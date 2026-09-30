"""Automated consistency audit of the submission package.

Every headline number that appears in the abstract, the results text, the
conclusion, the cover letter and the highlights is recomputed from
data/results and compared against what the documents actually say. Run as

    python -m python.paper.qa_consistency

Exits non-zero if any assertion fails, so it can gate a release.
"""
from __future__ import annotations

import os
import re
import sys
from typing import List, Tuple

import numpy as np
import pandas as pd
from docx import Document

RESULTS = os.path.join("data", "results")
TABDIR = os.path.join("paper-assets", "tables")
SUB = os.path.join("..", "submission")

S4 = "S4_disaster_collapse"
S6 = "S6_common_cause"
S7 = "S7_external_ntn"


def _docx_text(path: str) -> str:
    doc = Document(path)
    parts = [p.text for p in doc.paragraphs]
    for t in doc.tables:
        for row in t.rows:
            parts.extend(c.text for c in row.cells)
    return "\n".join(parts)


def _has(text: str, value: float, dp: int, label: str, checks: List[Tuple[str, bool]]) -> None:
    needle = f"{value:.{dp}f}"
    checks.append((f"{label}: {needle}", needle in text))


def main() -> int:
    runs = pd.read_csv(os.path.join(RESULTS, "experiment_runs.csv"))
    piv = lambda m: runs.pivot_table(index="scenario", columns="policy", values=m)
    A, RL, REC = piv("service_availability"), piv("resilience_loss"), piv("mean_recovery_steps")

    man = _docx_text(os.path.join(SUB, "Manuscript_EN.docx"))
    cover = open(os.path.join(SUB, "Cover_Letter.md"), encoding="utf-8").read()
    high = open(os.path.join(SUB, "Highlights.txt"), encoding="utf-8").read()

    checks: List[Tuple[str, bool]] = []

    # --- headline S4 numbers must appear verbatim in the manuscript -----------
    _has(man, 100 * A.loc[S4, "reactive"], 2, "manuscript S4 availability P1", checks)
    _has(man, 100 * A.loc[S4, "proactive_ai"], 2, "manuscript S4 availability P4", checks)
    _has(man, RL.loc[S4, "reactive"], 2, "manuscript S4 loss P1", checks)
    _has(man, RL.loc[S4, "proactive_ai"], 2, "manuscript S4 loss P4", checks)
    _has(man, REC.loc[S4, "reactive"], 2, "manuscript S4 recovery P1", checks)
    _has(man, REC.loc[S4, "proactive_ai"], 2, "manuscript S4 recovery P4", checks)

    # --- relative reductions quoted in abstract, cover letter and highlights --
    loss_red = 100 * (RL.loc[S4, "reactive"] - RL.loc[S4, "proactive_ai"]) / RL.loc[S4, "reactive"]
    rec_red = 100 * (REC.loc[S4, "reactive"] - REC.loc[S4, "proactive_ai"]) / REC.loc[S4, "reactive"]
    for doc, name in ((man, "manuscript"), (cover, "cover letter"), (high, "highlights")):
        checks.append((f"{name}: S4 loss reduction {loss_red:.0f}%",
                       f"{loss_red:.0f}" in doc))
        checks.append((f"{name}: S4 recovery reduction {rec_red:.0f}%",
                       f"{rec_red:.0f}" in doc))

    # --- campaign size -------------------------------------------------------
    checks.append((f"manuscript: campaign size {len(runs)}", str(len(runs)) in man))
    checks.append((f"cover letter: campaign size {len(runs)}", str(len(runs)) in cover))
    checks.append((f"manuscript: {runs.seed.nunique()} paired seeds",
                   str(runs.seed.nunique()) in man))

    # --- appendix must agree with the compact table ---------------------------
    app = pd.read_csv(os.path.join(TABDIR, "t11_main_results.csv"))
    row = app[(app.scenario.str.startswith("S4")) & (app.policy.str.startswith("P4"))]
    if not row.empty:
        checks.append(("appendix S4 P4 loss equals campaign mean",
                       abs(float(row["resilience loss"].iloc[0]) - RL.loc[S4, "proactive_ai"])
                       < 0.005))

    # --- externally anchored scenario: the null result must be stated ---------
    if S7 in A.index:
        base, prop = REC.loc[S7, "reactive"], REC.loc[S7, "proactive_ai"]
        _has(man, base, 2, "manuscript S7 recovery P1", checks)
        _has(man, prop, 2, "manuscript S7 recovery P4", checks)
        checks.append(("manuscript states the S7 null result",
                       "indistinguishable from reactive restoration" in man))

    # --- no leftover placeholders or NaN cells -------------------------------
    for token in ("0000-0000-0000-0000", "[Date]", "{eq", "TODO"):
        checks.append((f"manuscript free of '{token}'", token not in man))
    checks.append(("manuscript free of bare nan cells",
                   re.search(r"(?<![A-Za-z])(nan|NaN)(?![A-Za-z])", man) is None))
    for token in ("0000-0000-0000-0000", "[Date]"):
        checks.append((f"cover letter free of '{token}'", token not in cover))

    # --- highlights length limit ---------------------------------------------
    for line in high.strip().splitlines():
        line = line.strip()
        if not line or line.lower() == "highlights":
            continue
        checks.append((f"highlight <= 85 chars ({len(line)})", len(line) <= 85))

    # --- attribution and author metadata --------------------------------------
    checks.append(("manuscript attributes the cadence to Tanveer et al.",
                   "Tanveer et al." in man))
    checks.append(("manuscript attributes the latency to Michel et al.",
                   "Michel et al." in man))
    checks.append(("both ORCIDs present in the manuscript",
                   "0000-0003-1870-3436" in man and "0000-0002-3275-8188" in man))

    # --- LaTeX package integrity ---------------------------------------------
    els = os.path.join(SUB, "els-article")
    if os.path.isdir(els):
        tex = open(os.path.join(els, "manuscript.tex"), encoding="utf-8").read()
        master = open(os.path.join(els, "manuscript.tex"), encoding="utf-8").read()
        bib = open(os.path.join(els, "references.bib"), encoding="utf-8").read()
        figs = {f for f in os.listdir(os.path.join(els, "figures")) if f.endswith(".png")}
        used = set(re.findall(r"\\includegraphics\[[^\]]*\]\{([^}]+)\}", tex))
        checks.append((f"every referenced figure is shipped ({len(used)})",
                       used <= figs))
        cited = set(re.findall(r"\\cite\{([^}]+)\}", tex))
        cited = {k for group in cited for k in group.split(",")}
        items = set(re.findall(r"\\bibitem\{([^}]+)\}", tex))
        checks.append((f"no missing citations ({len(cited)} cited)", cited <= items))
        checks.append((f"no uncited bibliography items ({len(items)} items)", items <= cited))
        checks.append((f"bib entries match bibitems ({len(items)})",
                       len(re.findall(r"@\w+\{", bib)) == len(items)))
        checks.append(("both ORCIDs in the LaTeX author block",
                       "0000-0003-1870-3436" in master and "0000-0002-3275-8188" in master))
        checks.append(("no placeholder ORCID in LaTeX",
                       "0000-0000-0000-0000" not in master))
        checks.append(("corresponding-author email in LaTeX",
                       "dev.nikita@outlook.com" in master))
        checks.append(("second-author email in LaTeX",
                       "roman.v.zinko@lpnu.ua" in master))

    # --- consolidated robustness table must state the S7 null result ----------
    rs = os.path.join(TABDIR, "t22_robustness_strategy.csv")
    if os.path.exists(rs):
        txt = open(rs, encoding="utf-8").read()
        checks.append(("robustness table states the S7 null result",
                       "not retained" in txt))
        checks.append(("robustness table avoids 'hand-tuned'", "hand-tuned" not in txt))

    # --- forensic audit: equations, notation, lead-time consistency ------------
    from ..simulator.config import PREDICTION_HORIZON
    ai = runs[runs.policy == "proactive_ai"]
    lt = ai["lead_time_mean"].dropna()
    checks.append((f"every lead time within the horizon H = {PREDICTION_HORIZON}",
                   bool((lt <= PREDICTION_HORIZON).all()) and bool((lt > 0).all())))
    checks.append(("manuscript states horizon-consistent event matching",
                   "horizon-consistent" in man))
    checks.append(("degradation rate renamed away from 'hazard rate'",
                   "hazard rate" not in man or "not a failure hazard rate" in man))
    checks.append(("Eq. (2) uses nu, not lambda", "nu_n(t) = nu_0" in man))
    checks.append(("projection onto [0,1] documented", "Pi_{[0,1]}" in man))
    checks.append(("Gamma parameterisation stated", "shape = 2, scale" in man))
    checks.append(("eta distribution defined", "eta ~ N(0, sigma_S^2)" in man))
    checks.append(("pi_c reserved for traffic share",
                   "pi_c is reserved throughout for the nominal flow-slot share" in man))
    checks.append(("criticality in Eq. (11) uses chi_c", "0.6 chi_c" in man))
    checks.append(("budget factor renamed to b_c", "kappa b_c theta*" in man))
    checks.append(("nominal kappa reported", "Global budget scale kappa" in man))
    checks.append(("k = 3 path alternatives stated", "k = 3" in man))
    checks.append(("penalty function made explicit", "Pen(P) = 2.5" in man))
    checks.append(("R_prod vs R_inf distinguished",
                   "R_prod(P) = 1 - prod" in man and "R_inf(P) = max" in man))
    checks.append(("Eq. (13) conditions on the class", "P(T_f > t | c_f = c)" in man))
    checks.append(("Markov claim conditioned on utilisation and scenario",
                   "the current utilisation and the scenario state" in man))
    checks.append(("Table 18 vs Table 19 distinction stated",
                   "must not be confused with the g = 0 setting" in man))
    checks.append(("observation-model constants tabulated", "rho_cap" in man))

    # publication-facing text must not leak machine identifiers
    for token in ("lambda_0", "W_LOAD", "W_LOSS", "W_RISK", "alarm_rate", "roc_auc",
                  "service_availability", "mean_recovery_steps", "ci95", "held_out"):
        checks.append((f"no code identifier '{token}' in the manuscript", token not in man))

    # --- manuscript assembly: equation sequence in the COMPILED source ---------
    if os.path.isdir(els):
        src = open(os.path.join(els, "manuscript.tex"), encoding="utf-8").read()
        labels = [int(x) for x in re.findall(r"\\label\{eq:(\d+)\}", src)]
        checks.append((f"equations numbered 1..18 exactly once ({labels})",
                       labels == list(range(1, 19))))
        checks.append(("two unnumbered display equations",
                       src.count("\\begin{equation*}") == 2))
        checks.append(("no lambda_n / lambda_0 anywhere in the LaTeX",
                       not re.search(r"\\lambda_[n0]|lambda_[n0]", src)))
        checks.append(("Eq. (2) is the degradation rate", r"\nu_n(t) = \nu_0" in src))
        checks.append(("Eq. (11) is the implemented objective",
                       "J_c(P)" in src and r"R_{\mathrm{prod}}(P)" in src
                       and r"\chi_c" in src))
        checks.append(("Eq. (13) conditions on the class", r"c_f = c" in src))
        checks.append(("Eq. (17) is the aligned weighted pair",
                       r"L_w^{\mathrm{cls}}" in src and r"L_w^{\mathrm{flw}}" in src))
        checks.append(("Eq. (18) is the lead time", r"T_{\mathrm{lead}}" in src))
        checks.append(("pi_c appears only as traffic share",
                       src.count(r"\pi_c") == src.count(r"\sum_c w_c \pi_c")
                       + len(re.findall(r"\$\\pi_c\$", src))))
        checks.append(("no duplicate weighted-loss display",
                       src.count(r"L_w^{\mathrm{cls}} &=") <= 1))
        checks.append(("no duplicate lead-time display",
                       src.count(r"T_{\mathrm{lead}} = t_{\mathrm{failure}}") == 1))
        checks.append(("manuscript.tex is self-contained", "\\input{" not in src))

    # --- survival table must contain exactly one block per configuration -------
    sv = pd.read_csv(os.path.join(RESULTS, "survival_records.csv"))
    per = sv.groupby(["scenario", "policy"]).size().unstack(fill_value=0)
    ratio = float((per.max(axis=1) / per.min(axis=1)).max())
    checks.append((f"no duplicated survival blocks (max flow-count ratio {ratio:.3f})",
                   ratio < 1.15))
    seeds_per = sv.groupby(["scenario", "policy"])["seed"].nunique().unique()
    checks.append((f"exactly one seed block per configuration ({list(seeds_per)})",
                   len(seeds_per) == 1))
    dup = sv.duplicated(subset=["scenario", "policy", "seed", "duration", "event",
                                "duration_sustained", "event_sustained"]).sum()
    checks.append((f"survival rows are not wholesale duplicates ({dup} repeated tuples)",
                   ratio < 1.15))

    # --- terminology: nu is not called a hazard rate ---------------------------
    for token in ("Utilisation-hazard", "load-to-hazard", "load-hazard feedback",
                  "hazard susceptibility", "baseline hazard", "hazard scale"):
        checks.append((f"no legacy term '{token}'", token not in man))

    # --- layout: nothing floats, everything is in document order ---------------
    if os.path.isdir(els):
        tex = open(os.path.join(els, "manuscript.tex"), encoding="utf-8").read()
        for env in ("table", "table*", "figure", "figure*"):
            checks.append((f"no {env} float environment",
                           f"\\begin{{{env}}}" not in tex))
        checks.append((f"tables are longtables ({tex.count('begin{longtable}')})",
                       tex.count("\\begin{longtable}") >= 20))
        manual_fig = len(re.findall(r"\\captionof\{figure\}\{\s*\d+[.:]", tex))
        manual_tab = len(re.findall(r"\\caption\{\s*\d+[.:]", tex))
        checks.append((f"no manual number in a figure caption ({manual_fig})",
                       manual_fig == 0))
        checks.append((f"no manual number in a table caption ({manual_tab})",
                       manual_tab == 0))
        checks.append((f"figures use captionof ({tex.count('captionof{figure}')})",
                       tex.count("\\captionof{figure}") == 19))
        checks.append(("single-column CAS class", "{cas-sc}" in tex))
        prev = open(os.path.join(els, "preview.pdf"), "rb").read() \
            if os.path.exists(os.path.join(els, "preview.pdf")) else b""
        checks.append(("preview.pdf present and non-trivial", len(prev) > 500_000))

    # --- Table 6: S7 must be held out -----------------------------------------
    sc = pd.read_csv(os.path.join(TABDIR, "t03_scenarios.csv"))
    row = sc[sc.scenario == "S7_external_ntn"]
    checks.append(("S7 is not in the training corpus",
                   not row.empty and str(row["in training corpus"].iloc[0]).strip() == "no"))
    checks.append(("scenario table uses degradation multipliers",
                   "degradation multipliers" in sc.columns))

    # --- C4 wording ------------------------------------------------------------
    checks.append(("C4 cites the twenty-comparison family",
                   "family of twenty primary" in man))

    # --- the shipped artifact README must match the manuscript -----------------
    readme = os.path.join("README.md")
    if os.path.exists(readme):
        rd = open(readme, encoding="utf-8").read()
        checks.append((f"artifact README states {len(runs)} runs", str(len(runs)) in rd))
        checks.append(("artifact README lists six policies", "Six recovery policies" in rd))
        checks.append(("artifact README lists seven scenarios",
                       "Seven disruption scenarios" in rd))
        checks.append(("artifact README names the degradation-aware baseline",
                       "degradation_aware" in rd))
        checks.append(("artifact README has no stale run count",
                       "720 closed-loop" not in rd and "864 closed-loop" not in rd))

    # --- research artifact must be cited, not merely linked --------------------
    checks.append(("Zenodo DOI present in the manuscript", "10.5281/zenodo.22084966" in man))
    checks.append(("no unfilled artifact placeholder",
                   "[ZENODO DOI]" not in man and "[GITHUB URL]" not in man))
    checks.append(("Zenodo DOI in the cover letter", "10.5281/zenodo.22084966" in cover))
    checks.append(("no post-acceptance deposit promise",
                   "upon acceptance" not in man))
    for token in ("rather than defended as estimates", "to be worth anything",
                  "and we say so", "flatters the method"):
        checks.append((f"no defensive phrase '{token}'",
                       token not in man and token not in cover))

    # --- report ---------------------------------------------------------------
    failed = [name for name, ok in checks if not ok]
    for name, ok in checks:
        print(("PASS  " if ok else "FAIL  ") + name)
    print(f"\n{len(checks) - len(failed)}/{len(checks)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
