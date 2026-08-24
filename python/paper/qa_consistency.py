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
                       abs(float(row["resilience_loss"].iloc[0]) - RL.loc[S4, "proactive_ai"])
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

    # --- report ---------------------------------------------------------------
    failed = [name for name, ok in checks if not ok]
    for name, ok in checks:
        print(("PASS  " if ok else "FAIL  ") + name)
    print(f"\n{len(checks) - len(failed)}/{len(checks)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
