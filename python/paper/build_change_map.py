"""Revision change map for JRESS-D-26-06464 (R1): every reviewer comment mapped to the change
made and to its place in the revised manuscript.

Table and figure numbers are resolved from the manuscript's caption list, so they always match
the manuscript built by build_paper_r1.py. Output: Revision_Change_Map_R1.docx (A4 landscape)
and Revision_Change_Map_R1.md.

    python -m python.paper.build_change_map --captions <manuscript>.docx.captions.json --out <dir>
"""
from __future__ import annotations

import argparse
import json
import os
import re

from docx import Document
from docx.enum.section import WD_ORIENT
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

TITLE = ("Where Does Predictive Resilience Come From? A Simulation-Based Digital-Twin "
         "Assessment of Failure Anticipation, Protection and Risk-Aware Recovery in Hybrid "
         "5G/6G-Class and LEO Satellite Networks")

# Caption keys -> resolved to "Table n" / "Fig. n".
KEYS = {
    "T5": "Model and control parameters", "T7": "Anticipation performance on episode-disjoint",
    "T9": "Cumulative resilience loss L", "T10": "Family A, retained",
    "T11": "Exploratory paired contrasts", "T12": "Frozen confirmation experiment",
    "T13": "Confirmatory families B1", "T14": "Family B3",
    "T15": "Exploratory headroom of perfect-state", "T16": "Ground-truth classification of migration",
    "T17": "Recovery time separated by the cause", "T18": "Availability (A) and SLA-compliant",
    "T20": "Best operating point of each risk source",
    "T21": "Closed-loop outcome in an alternative degradation world",
    "T22": "Results on a 98-element topology", "T23": "Sensitivity of the protection results",
    "T24": "Exploratory decomposition of predictive protection",
    "T25": "Exploratory controls for the P6", "T26": "Control-plane operations per run",
    "T27": "Prediction-horizon sweep", "T28": "Feedback ablation at fixed traffic",
    "T29": "Sensitivity to the controller design choices", "T32": "Sensitivity to the cost weights",
    "T33": "All fifteen policies", "T34": "Calibration quality",
    "F8": "Frozen confirmation on 64 new seeds", "F13": "Resilience loss against the number of proactive",
    "F14": "Generalisation of the anticipation model", "F15": "Effect of the prediction horizon H",
}

GLOBAL = [
    ("New title: the paper asks where the resilience gain comes from instead of claiming a gain "
     "for one predictive policy.", "Title page; cover letter", "R1.5, R3.10"),
    ("Abstract rewritten (under 200 words, six keywords). The original headline figures "
     "(−33% resilience loss, −74% recovery time) are withdrawn.", "Abstract", "R3.2, R3.3, R3.10"),
    ("'Simulation-based digital twin' throughout, with what the twin adds over a network "
     "simulator; no physical synchronisation is claimed.", "Abstract; §1; §4.1; §8.4", "R4.1, R1.1"),
    ("Implementation audit with six corrections, each covered by a unit test: paired design "
     "(keyed random streams), horizon-bounded lead-time tracker, restoration cost for migration "
     "away from a failed path, keyed recovery durations, interruption of exactly C slots, "
     "capacity-checked protection switch-over.", "§3.4; §6.4; §7.4, {T16}; §7.10", "R3.2, R3.3, R3.8, R5.1"),
    ("Policy set expanded from 6 to 15; where the design allows, one mechanism varies at a time.",
     "§6.3; {T33}", "R1.4, R1.5, R5.2, R3.4–R3.6"),
    ("Frozen confirmatory evaluation on 64 independent seeds (1,792 runs): primary family B1, "
     "key-secondary family B2, equivalence family B3; protocol committed before the runs.",
     "§6.5; §7.3; {T12}–{T14}; {F8}", "R3.10, R3.11"),
    ("Exploratory controls after the confirmation, on the same seeds: decomposition of P6, "
     "risk-free standby repair (P1b-r), rate-limited refresh (P6-slow) and control-plane "
     "operations. Under collapse repair alone reproduces 91% of the P6 − P1b gain; under "
     "common-cause disruption the gain comes from degradation-informed standby selection.",
     "§7.9; {T24}–{T26}", "R1.4, R1.5, R3.10"),
    ("Related work on protection in terrestrial and satellite networks, and three recent studies "
     "in this journal [86]–[88].", "§2.1; §2.2; §2.4", "R1.4, R3.10"),
    ("Methodological novelty stated explicitly, with the limits of the separation of mechanisms.",
     "§1.2", "R3.10"),
    ("Limitations rewritten (nine items), including protection assumptions, a single alternative "
     "topology and the uncharged control-plane cost of standby refresh.", "§9", "R1.2, R1.4, R3.1, R3.12"),
    ("Reproducibility: code, frozen protocol and result files archived on Zenodo (release v1.4-R1, "
     "doi:10.5281/zenodo.24316819; raw campaign results of v1.1-R1 unchanged); all numbers in text "
     "and tables generated from the result files; 20 unit tests.",
     "Data and code availability; ref. [78]", "R3.3, R3.9"),
    ("Use of AI tools stated in Methods (coding assistance for the revision code) and in the "
     "declaration.", "§6.1; Declaration", "Journal policy"),
]

# (comment, concern, change made, where in R1)
POINTS = {
    "Reviewer 4": [
        ("R4.1", "Clarify 'digital twin'", "Called simulation-based throughout; §8.4 states what the "
         "twin adds over a simulator; no validation on a real system is claimed.", "Abstract; §4.1; §8.4"),
        ("R4.2", "Slot vs real time", "One slot = 1 s; all delays and costs in seconds; C_m drawn "
         "from {0, 1}; interruptions counted in whole slots.", "§3.4; {T5}"),
        ("R4.3", "Model-selection criterion", "Highest validation PR-AUC (tolerance 0.01), then "
         "validation Brier, then inference cost; tie-break not invoked; test labels not used; "
         "optimism of the reused validation fold stated.", "§4.3; {T7}; App. B, {T34}"),
        ("R4.4", "S7 is exploratory", "S7 labelled an exploratory, externally parameterised stress "
         "test throughout.", "§6.2; §7.8"),
    ],
    "Reviewer 1": [
        ("R1.1", "Twin vs simulator", "Separate physical-state, telemetry and assessment layers "
         "explained; contribution of the twin beyond a simulator stated.", "§4.1; §8.4"),
        ("R1.2", "19 nodes insufficient", "Second topology: 98 elements, 233 links (3 LEO planes × 8 "
         "satellites, 2 core/feeder sites, 8 terrestrial regions); manifest unit test; "
         "single-structure limitation.", "§6.2; §7.8; {T22}; §9"),
        ("R1.3", "Predictor learns the authors' patterns", "Alternative degradation world "
         "(gamma-process wear-out, changed observation map) in closed loop; framework validation "
         "distinguished from predictor validation.", "§7.8; {T21}; §9"),
        ("R1.4", "Protection baselines", "1:1 protection P1b with three capacity models; risk-free "
         "standby repair P1b-r and rate-limited P6-slow; control-plane operations; unadjusted CIs "
         "noted.", "§2.4; §6.3; §7.3; §7.9; {T13}; {T23}–{T26}"),
        ("R1.5", "Share due to anticipation", "Gain decomposed into protection, standby repair, "
         "degradation information, detection-delay bypass and anticipation.",
         "§7.2–§7.9; {T14}–{T16}; {T25}; §8.1–§8.2"),
        ("R1.6", "Budget factors and weights", "Rationale for the budget factors; sensitivity to ten "
         "controller-design settings and to the cost weights.", "§4.4; §7.10; {T29}; {T32}"),
        ("R1.7", "S7: advantage disappears", "Explained; with pooled recovery accounting P4 recovers "
         "more slowly than P1 under S7; the recovery-time claim is withdrawn.", "§7.8"),
    ],
    "Reviewer 5": [
        ("R5.1", "Lead times exceed H", "Lead-time tracker corrected and horizon sweep rerun; lead "
         "time ≤ H holds and is plotted.", "§7.10; {T27}; {F15}"),
        ("R5.2", "Prediction vs differentiation", "Uniform variants P3u, P4u; direct paired P3–P4 "
         "contrasts; risk sources compared at matched actuation effort.",
         "§6.3; §7.7; {T11}; {T20}; {F13}"),
        ("R5.3", "Confounded feedback ablation", "Fixed-traffic replay; the earlier claim is removed.",
         "§7.10; {T28}"),
        ("R5.4", "Availability vs SLA", "Per-class SLA-compliant availability; recovery split into "
         "failure- and migration-induced interruptions.", "§7.4; {T17}; {T18}"),
    ],
    "Reviewer 3": [
        ("R3.1", "Same generative family", "Misspecified degradation world in closed loop; offline "
         "transfer; limitation stated.", "§7.8; {T21}; {F14}; §9"),
        ("R3.2", "Lead-time inconsistency", "Computation error corrected (Eqs. (17)–(18)); horizon "
         "sweep rerun.", "§5.4; §7.10; {T27}; {F15}"),
        ("R3.3", "Flow counts; common random numbers", "Keyed per-process random streams; identical "
         "flow populations across policies, verified by a unit test.", "§6.4; {T9}; {T33}"),
        ("R3.4", "Specify P2; matched comparison", "P2 fully specified; P2m with matched alarm rate; "
         "actuation frontier; the earlier claim is removed.", "§6.3; §7.6; {F13}; {T20}"),
        ("R3.5", "Data for P3 matching", "Matched on dedicated validation runs with disjoint seeds, "
         "frozen before evaluation.", "§6.3"),
        ("R3.6", "P5 is not an oracle", "P5 map defined; P5c is a Monte-Carlo-calibrated "
         "privileged-state reference, not an upper bound; over-actuation claim restricted to P5.",
         "§6.3; §7.3; {T15}"),
        ("R3.7", "Eq. (12) too strong", "Recast as an ex-post economic reading; controller described "
         "as a heuristic.", "§4.5"),
        ("R3.8", "Slot meaning; fractional C_m", "One slot = 1 s; half-open interruption accounting; "
         "C_m drawn from {0, 1}.", "§3.4"),
        ("R3.9", "S1 'indistinguishable'", "Sentence removed; Results and Conclusion rewritten; "
         "numbers generated from result files; automated claim check.", "§7; §11"),
        ("R3.10", "Narrower claim", "Paper framed as a decomposition; confirmatory and exploratory "
         "evidence separated; risk-free repair control added.", "Title; Abstract; §1.2; §7.9; §8; §11"),
        ("R3.11", "Weighting is exploratory", "Criticality-weighted loss labelled exploratory; "
         "per-class SLA reported.", "§5.2; {T18}"),
        ("R3.12", "S7 is a stress test", "Called an externally parameterised stress test, not a "
         "validation.", "§6.2; §7.8; §8.3; §9"),
    ],
}

WITHDRAWN = [
    ("Failure prediction drives a large resilience gain (−33% loss, −74% recovery time under "
     "collapse).", "Withdrawn. Plain protection is the most consistent source of gain; P4 is not "
     "significantly better than protection after Holm correction.", "§7.3; {T12}; {T13}"),
    ("Predictive protection gains through risk information.", "Qualified. Confirmed against static "
     "protection; under collapse 91% of the gain is reproduced by risk-free standby repair; under "
     "common-cause disruption it comes from degradation-informed standby selection.",
     "§7.9; {T25}"),
    ("Perfect state knowledge over-actuates.", "Restricted to the uncalibrated P5 map.", "§6.3; {T15}"),
    ("Lead times 9.67 and 10.67 for H = 3 and 5.", "Corrected: a computation error; now bounded by H.",
     "§7.10; {T27}"),
    ("S7: only the recovery-time benefit survives.", "Reversed: with pooled recovery accounting, P4 "
     "recovers more slowly than P1 in S7.", "§7.8"),
    ("Criticality-weighted loss as main evidence.", "Labelled exploratory; per-class SLA reported.",
     "§5.2; {T18}"),
]


def resolver(captions):
    def ref(key):
        frag = KEYS[key]
        for c in captions:
            m = re.match(r"^(Table|Fig\.) (\d+)\. ", c)
            if m and frag in c:
                return f"{m.group(1)} {m.group(2)}"
        raise KeyError(f"no caption for {key}: {frag!r}")

    def fill(s):
        out = re.sub(r"\{(\w+)\}", lambda m: ref(m.group(1)), s)
        out = re.sub(r"Table (\d+)–Table (\d+)", r"Tables \1–\2", out)
        return out
    return fill


def _shade(cell, hex_fill):
    tc = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), hex_fill)
    tc.append(shd)


def _row_props(row, header=False):
    trPr = row._tr.get_or_add_trPr()
    cant = OxmlElement("w:cantSplit")
    trPr.append(cant)
    if header:
        h = OxmlElement("w:tblHeader")
        trPr.append(h)


def _set_cell(cell, text, bold=False, size=9, color=None):
    cell.text = ""
    p = cell.paragraphs[0]
    p.paragraph_format.space_after = Pt(0)
    run = p.add_run(text)
    run.bold = bold
    run.font.size = Pt(size)
    if color:
        run.font.color.rgb = RGBColor.from_string(color)


def _table(doc, headers, widths, rows, groups=None):
    t = doc.add_table(rows=1, cols=len(headers))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False
    hdr = t.rows[0]
    _row_props(hdr, header=True)
    for i, h in enumerate(headers):
        _set_cell(hdr.cells[i], h, bold=True, size=9, color="FFFFFF")
        _shade(hdr.cells[i], "1F3A5F")
    items = groups if groups else [(None, rows)]
    for gname, grows in items:
        if gname:
            r = t.add_row()
            _row_props(r)
            merged = r.cells[0].merge(r.cells[-1])
            _set_cell(merged, gname, bold=True, size=9.5)
            _shade(merged, "D9E2EC")
        for k, vals in enumerate(grows):
            r = t.add_row()
            _row_props(r)
            for i, v in enumerate(vals):
                _set_cell(r.cells[i], v, bold=(i == 0 and len(headers) == 4), size=9)
            if k % 2 == 1:
                for c in r.cells:
                    _shade(c, "F5F7FA")
    tblPr = t._tbl.tblPr
    layout = OxmlElement("w:tblLayout")
    layout.set(qn("w:type"), "fixed")
    tblPr.append(layout)
    for i, gc in enumerate(t._tbl.tblGrid.findall(qn("w:gridCol"))):
        gc.set(qn("w:w"), str(int(widths[i] * 567)))
    for i, col in enumerate(t.columns):
        col.width = Cm(widths[i])
    for row in t.rows:
        for i, w in enumerate(widths):
            row.cells[i].width = Cm(w)
    return t


def build(captions_path, out_dir):
    fill = resolver(json.load(open(captions_path)))
    doc = Document()
    sec = doc.sections[0]
    sec.orientation = WD_ORIENT.LANDSCAPE
    sec.page_width, sec.page_height = Cm(29.7), Cm(21.0)
    sec.left_margin = sec.right_margin = Cm(1.8)
    sec.top_margin = sec.bottom_margin = Cm(1.6)
    st = doc.styles["Normal"]
    st.font.name, st.font.size = "Calibri", Pt(10)
    st.paragraph_format.space_after = Pt(4)

    fp = sec.footer.paragraphs[0]
    fp.text = "JRESS-D-26-06464 · Revision change map (R1)"
    fp.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    fp.runs[0].font.size = Pt(8)

    h = doc.add_paragraph()
    r = h.add_run("Revision change map")
    r.bold, r.font.size = True, Pt(16)
    p = doc.add_paragraph()
    p.add_run("Manuscript JRESS-D-26-06464, first revision. ").bold = True
    p.add_run(TITLE + ". Nikita Tarasov and Roman Zinko.")
    doc.add_paragraph(
        "This map links every reviewer comment to the change made and to its place in the revised "
        "manuscript; it replaces a word-level comparison, which would mark most of Sections 1 and "
        "6–11. Section (§), table and figure numbers refer to the revised manuscript. The full "
        "answers are in the reply to each reviewer. Results marked exploratory were obtained "
        "outside the frozen confirmation protocol.")

    doc.add_heading("1. Changes that affect the whole paper", level=2)
    _table(doc, ["Change", "Where in the revised manuscript", "Prompted by"], [17.0, 5.6, 3.5],
           [(c, fill(w), by) for c, w, by in GLOBAL])

    doc.add_heading("2. Point-by-point map of the 27 comments", level=2)
    groups = [(g, [(cid, concern, change, fill(where)) for cid, concern, change, where in rows])
              for g, rows in POINTS.items()]
    _table(doc, ["Comment", "Concern", "What changed", "Where in the revised manuscript"],
           [2.1, 4.3, 12.9, 6.8], None, groups=groups)

    doc.add_heading("3. Results withdrawn, qualified or reversed", level=2)
    _table(doc, ["Original claim", "Status in the revision", "Where"], [8.0, 13.2, 4.9],
           [(a, b, fill(c)) for a, b, c in WITHDRAWN])

    os.makedirs(out_dir, exist_ok=True)
    doc.core_properties.author = "Nikita Tarasov"
    doc.core_properties.last_modified_by = "Nikita Tarasov"
    doc.core_properties.title = "Revision change map — JRESS-D-26-06464 (R1)"
    doc.save(os.path.join(out_dir, "Revision_Change_Map_R1.docx"))

    md = ["# Revision change map — JRESS-D-26-06464 (R1)", "",
          "## 1. Changes that affect the whole paper", "",
          "| Change | Where | Prompted by |", "|---|---|---|"]
    md += [f"| {c} | {fill(w)} | {by} |" for c, w, by in GLOBAL]
    md += ["", "## 2. Point-by-point map", "",
           "| Comment | Concern | What changed | Where |", "|---|---|---|---|"]
    for g, rows in POINTS.items():
        md.append(f"| **{g}** | | | |")
        md += [f"| {a} | {b} | {c} | {fill(d)} |" for a, b, c, d in rows]
    md += ["", "## 3. Results withdrawn, qualified or reversed", "",
           "| Original claim | Status | Where |", "|---|---|---|"]
    md += [f"| {a} | {b} | {fill(c)} |" for a, b, c in WITHDRAWN]
    open(os.path.join(out_dir, "Revision_Change_Map_R1.md"), "w").write("\n".join(md) + "\n")
    assert sum(len(v) for v in POINTS.values()) == 27


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--captions", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    build(a.captions, a.out)


if __name__ == "__main__":
    main()
