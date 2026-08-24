"""
Generate the Elsevier CAS (els-cas-templates) LaTeX package from the manuscript.

The LaTeX source is derived from the same .docx that the Word version produces,
so the two cannot diverge. Equations are substituted from a hand-written LaTeX
dictionary because the Word version stores them as plain text.

    python -m python.paper.build_latex --docx "Predictive_Resilience.docx" --out submission/latex
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
from typing import Dict, List, Optional, Tuple

from docx import Document
from docx.table import Table
from docx.text.paragraph import Paragraph

# ---------------------------------------------------------------------------
# Equations, in document order. The Word version renders them as plain text;
# here they are typeset properly.
# ---------------------------------------------------------------------------
EQUATIONS: Dict[int, str] = {
    1: r"\mathrm{d}h_n = \mu\,(1-h_n)\,\mathrm{d}t - \lambda_n(t)\,\mathrm{d}t "
       r"+ \sigma\,\mathrm{d}W_n - J\,\mathrm{d}N_n ,",
    2: r"\lambda_n(t) = \lambda_0\,k_n\,m_s(\mathrm{class}_n)"
       r"\left[1 + g\,\max\!\left(0,\rho_n(t)-\rho^{*}\right)^{3/2}\right] ,",
    3: r"h_i(t^{+}) = h_i(t) - \alpha_i Z_t ,\qquad Z_t \sim \Gamma(2,\beta/2) ,",
    4: r"L_n(t) = L_n^{0}\left[1 + a_1 d^{2} + a_2 q\right]e^{\varepsilon} ,\qquad "
       r"\varepsilon \sim \mathcal{N}(0,\sigma_o^{2}) ,",
    5: r"p_n(t) = p_n^{0} + b\,d^{3} + c\,\max\!\left(0,\rho_n-0.8\right)^{2} ,",
    6: r"S_n(t) = S_n^{0} - \gamma\,d^{1.2} + \eta ,",
    7: r"\phi_n(t) = w_L\frac{L_n(t)}{L_{\mathrm{ref}}} + w_P\,p_n(t) "
       r"+ w_U\max\!\left(0,\rho_n-0.75\right)^{2} + w_R\,r_n(t)^{2} ,",
    8: r"c_{uv}(t) = w_L\frac{d_{uv}}{L_{\mathrm{ref}}} "
       r"+ \tfrac{1}{2}\left[\phi_u(t)+\phi_v(t)\right] ,",
    9: r"y_n(t) = \mathbb{1}\left\{\exists\,u \in (t,\,t+H] : "
       r"\mathrm{state}_n(u)=\mathrm{FAILED}\right\} ,",
    10: r"\tau_c = \min\!\left(0.95,\ \kappa\,\phi_c\,\theta^{*}\right) ,",
    11: r"J(P) = \frac{L(P)}{L_c^{\max}} + 4\,\frac{\ell(P)}{\ell_c^{\max}} "
        r"+ \left(3+0.6\,\pi_c\right)R_{\infty}(P) + \mathrm{penalties} ,",
    12: r"r > p^{*} = \frac{C_m}{C_f} .",
    13: r"R_c(t) = \Pr\!\left(T_c > t\right) ,",
    14: r"A = 1 - \frac{\sum_f o_f}{\sum_f a_f} ,",
    15: r"L = \sum_{t=1}^{T}\left[Q_0 - Q(t)\right]_{+} ,",
    16: r"\mathcal{R} = 1 - \frac{L}{T\,Q_0} .",
    17: r"L_w^{\mathrm{cls}} = T\left[1-\frac{\sum_c w_c A_c}{\sum_c w_c}\right] ,\qquad L_w^{\mathrm{flw}} = T\left[1-\frac{\sum_c w_c \pi_c A_c}{\sum_c w_c \pi_c}\right] ,",
    18: r"T_{\mathrm{lead}} = t_{\mathrm{failure}} - t_{\mathrm{alarm}} ,",
}

# Figure files in the order in which they appear in the manuscript.
FIGURE_ORDER: List[str] = [
    "fig02_topology.png",
    "f03_degradation_trace.png",
    "fig01_architecture.png",
    "f04_decision_mechanism.png",
    "f05_predictor.png",
    "f06_lead_time.png",
    "f10_availability_recovery.png",
    "f09_resilience.png",
    "f08_survival.png",
    "f07_timeline.png",
    "f11_migration_cost.png",
    "f13_generalisation.png",
    "f12_horizon.png",
    "f14_noise.png",
    "s04_budget_sweep.png",
    "f15_topology.png",
    "f16_feedback.png",
    "f17_parameters.png",
    "f19_seed_convergence.png",
]

SPECIAL = {
    "—": "---", "–": "--", "“": "``", "”": "''", "‘": "`", "’": "'",
    "≤": r"$\leq$", "≥": r"$\geq$", "×": r"$\times$", "±": r"$\pm$",
    "μ": r"$\mu$", "σ": r"$\sigma$", "λ": r"$\lambda$", "θ": r"$\theta$",
    "κ": r"$\kappa$", "τ": r"$\tau$", "φ": r"$\phi$", "ρ": r"$\rho$",
    "Δ": r"$\Delta$", "α": r"$\alpha$", "β": r"$\beta$", "γ": r"$\gamma$",
    "…": r"\ldots", "·": r"$\cdot$", "\u00a0": "~",
}


def cite(text: str) -> str:
    """Turn the numeric citations of the Word version into \cite commands."""
    text = re.sub(r"\[(\d+)\](?:,\s*\[(\d+)\])*", _cite_group, text)
    return text


def _cite_group(m: re.Match) -> str:
    nums = re.findall(r"\d+", m.group(0))
    return "\\cite{" + ",".join(f"ref{n}" for n in nums) + "}"


MATH_TOKEN = "\u0001MATH%d\u0001"


def esc(text: str) -> str:
    """Escape a plain-text paragraph for LaTeX, protecting injected maths."""
    stash: List[str] = []
    for src, dst in SPECIAL.items():
        if dst.startswith("$"):
            def _keep(_m, _dst=dst):
                stash.append(_dst)
                return MATH_TOKEN % (len(stash) - 1)
            text = re.sub(re.escape(src), _keep, text)
        else:
            text = text.replace(src, dst)
    out = []
    for ch in text:
        if ch in "&%#_{}":
            out.append("\\" + ch)
        elif ch == "$":
            out.append(r"\$")
        elif ch == "^":
            out.append(r"\textasciicircum{}")
        elif ch == "~":
            out.append("~")
        elif ch == "\\":
            out.append(r"\textbackslash{}")
        else:
            out.append(ch)
    text = "".join(out)
    for i, math in enumerate(stash):
        text = text.replace(MATH_TOKEN % i, math)
    return text


def is_caption(text: str) -> Optional[Tuple[str, int]]:
    m = re.match(r"^(Fig\.|Table)\s+(\d+)\.\s", text)
    if not m:
        return None
    return ("figure" if m.group(1).startswith("Fig") else "table"), int(m.group(2))


def eq_number(text: str) -> Optional[int]:
    m = re.match(r"^(.{3,}?)\s*\((\d+)\)$", text.strip())
    if not m:
        return None
    n = int(m.group(2))
    return n if n in EQUATIONS else None


def table_to_latex(table, caption: str, label: str) -> str:
    rows = []
    for r in table.rows:
        cells = [esc(c.text.replace("\n", " ").strip()) for c in r.cells]
        rows.append(" & ".join(cells) + r" \\")
    ncol = len(table.rows[0].cells)
    colspec = "l" + "r" * (ncol - 1)
    body = "\n".join(rows[1:])
    return (
        "\\begin{table*}[!ht]\n\\centering\n\\scriptsize\n"
        f"\\caption{{{caption}}}\n\\label{{{label}}}\n"
        f"\\begin{{tabular}}{{{colspec}}}\n\\toprule\n"
        f"{rows[0]}\n\\midrule\n{body}\n\\bottomrule\n"
        "\\end{tabular}\n\\end{table*}\n"
    )


PREAMBLE = r"""%% Elsevier CAS template (els-cas-templates), double-column article class.
%% Place cas-dc.cls and cas-common.sty from the official els-cas-templates
%% bundle in this directory before compiling.
\documentclass[a4paper,fleqn]{cas-dc}

\usepackage[numbers,sort&compress]{natbib}
\usepackage{amsmath,amssymb}
\usepackage{booktabs}
\usepackage{graphicx}
\usepackage{multirow}
\usepackage{url}

\graphicspath{{figures/}}

\begin{document}

\let\WriteBookmarks\relax
\def\floatpagepagefraction{1}
\def\textpagefraction{.001}

\shorttitle{Predictive resilience of hybrid 5G/6G-class and LEO satellite networks}
\shortauthors{N. Tarasov and R. Zinko}

\title[mode=title]{Predictive Resilience of Hybrid 5G/6G-Class and LEO Satellite Networks:
A Digital-Twin Framework for Failure Anticipation and Risk-Aware Recovery}

%% Add [orcid=XXXX-XXXX-XXXX-XXXX] to each author before submission.
\author[1]{Nikita Tarasov}
\ead{dev.nikita@outlook.com}
\cormark[1]
\credit{Conceptualisation, Methodology, Software, Validation, Formal analysis,
Investigation, Data curation, Writing -- original draft, Visualisation}

\author[1]{Roman Zinko}
\credit{Conceptualisation, Methodology, Writing -- review \& editing, Supervision}

\affiliation[1]{organization={Lviv Polytechnic National University},
    addressline={S. Bandery St. 12},
    city={Lviv},
    postcode={79013},
    country={Ukraine}}

\cortext[1]{Corresponding author}
"""

POSTAMBLE = r"""
\end{document}
"""


def _close_itemize(body: List[str]) -> None:
    if body and body[-1].startswith("\\begin{itemize}") \
            and not body[-1].rstrip().endswith("\\end{itemize}"):
        body[-1] += "\\end{itemize}\n"


def build(docx_path: str, out_dir: str, figdir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "figures"), exist_ok=True)
    doc = Document(docx_path)

    body: List[str] = []
    abstract: List[str] = []
    keywords = ""
    references: List[str] = []
    fig_idx = 0
    pending_table_caption: Optional[str] = None
    tables = list(doc.tables)
    table_idx = 0
    section = "front"

    def iter_blocks(document):
        parent = document.element.body
        for child in parent.iterchildren():
            if child.tag.endswith("}p"):
                yield Paragraph(child, document)
            elif child.tag.endswith("}tbl"):
                yield Table(child, document)

    for block in iter_blocks(doc):
        if isinstance(block, Table):
            if pending_table_caption is not None:
                body.append(table_to_latex(block, pending_table_caption,
                                           f"tab:{table_idx + 1}"))
                table_idx += 1
                pending_table_caption = None
            continue
        par = block
        text = par.text.strip()
        style = par.style.name
        if not text:
            continue

        _close_itemize(body)
        if style == "Heading 1":
            head = text
            if head.lower() == "abstract":
                section = "abstract"
                continue
            if head.lower() == "references":
                section = "references"
                continue
            section = "body"
            m = re.match(r"^(\d+)\.\s+(.*)$", head)
            if m:
                body.append(f"\n\\section{{{esc(m.group(2))}}}\n")
            else:
                body.append(f"\n\\section*{{{esc(head)}}}\n")
            continue

        _close_itemize(body)
        if style == "Heading 2":
            m = re.match(r"^\d+\.\d+\s+(.*)$", text)
            body.append(f"\n\\subsection{{{esc(m.group(1) if m else text)}}}\n")
            continue

        if section == "abstract":
            if text.startswith("Keywords:"):
                keywords = text[len("Keywords:"):].strip().rstrip(".")
                section = "body"
            else:
                abstract.append(text)
            continue

        if section == "references":
            m = re.match(r"^\[(\d+)\]\s+(.*)$", text)
            if m:
                references.append(esc(m.group(2)))
            continue

        if section == "front":
            continue

        _close_itemize(body)
        cap = is_caption(text)
        if cap:
            kind, number = cap
            caption = esc(text.split(". ", 1)[1] if ". " in text else text)
            if kind == "figure":
                name = FIGURE_ORDER[fig_idx] if fig_idx < len(FIGURE_ORDER) else None
                fig_idx += 1
                if name:
                    src = os.path.join(figdir, name)
                    if os.path.exists(src):
                        shutil.copy(src, os.path.join(out_dir, "figures", name))
                    body.append(
                        "\n\\begin{figure*}[!ht]\n\\centering\n"
                        f"\\includegraphics[width=0.86\\textwidth]{{{name}}}\n"
                        f"\\caption{{{caption}}}\n\\label{{fig:{number}}}\n"
                        "\\end{figure*}\n"
                    )
            else:
                pending_table_caption = caption
            continue

        _close_itemize(body)
        n = eq_number(text)
        if n:
            body.append("\n\\begin{equation}\n" + EQUATIONS[n] + f"\n\\label{{eq:{n}}}\n"
                        "\\end{equation}\n")
            continue

        if style == "List Bullet":
            if not (body and body[-1].startswith("\\begin{itemize}")
                    and not body[-1].rstrip().endswith("\\end{itemize}")):
                body.append("\\begin{itemize}\n")
            body[-1] += f"\\item {cite(esc(text))}\n"
            continue
        _close_itemize(body)

        body.append(cite(esc(text)) + "\n")

    _close_itemize(body)

    bib = ["\n\\begin{thebibliography}{99}"]
    for i, ref in enumerate(references, start=1):
        bib.append(f"\\bibitem{{ref{i}}} {ref}")
    bib.append("\\end{thebibliography}\n")

    tex = [PREAMBLE]
    tex.append("\n\\begin{abstract}\n" + esc(" ".join(abstract)) + "\n\\end{abstract}\n")
    kw = " \\sep ".join(esc(k.strip()) for k in keywords.split(";") if k.strip())
    tex.append("\\begin{keywords}\n" + kw + "\n\\end{keywords}\n")
    tex.append("\\maketitle\n")
    tex.extend(body)
    tex.extend(bib)
    tex.append(POSTAMBLE)

    path = os.path.join(out_dir, "manuscript.tex")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(tex))
    print("latex:", path)
    print(f"figures copied: {fig_idx}, tables converted: {table_idx}, "
          f"references: {len(references)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--docx", required=True)
    ap.add_argument("--out", default=os.path.join("submission", "latex"))
    ap.add_argument("--figdir", default=os.path.join("paper-assets", "figures"))
    args = ap.parse_args()
    build(args.docx, args.out, args.figdir)


if __name__ == "__main__":
    main()
