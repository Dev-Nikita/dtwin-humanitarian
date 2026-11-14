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
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.table import Table
from docx.text.paragraph import Paragraph

# ---------------------------------------------------------------------------
# Equations, in document order. The Word version renders them as plain text;
# here they are typeset properly.
# ---------------------------------------------------------------------------
def _norm(text: str) -> str:
    """Whitespace-insensitive key for matching a display equation in the .docx
    against its LaTeX rendering."""
    return re.sub(r"\s+", " ", text).strip()


# Every display equation emitted by the manuscript generator, keyed by its exact
# source text and carrying a flag for whether it is numbered. Keying by content
# rather than by position is deliberate: if the generator changes an equation and
# this table is not updated, the build fails loudly instead of silently shifting
# the numbering or dropping an equation.
LATEX_EQUATIONS: Dict[str, Tuple[str, bool]] = {
    _norm("dh_n = mu (1 - h_n) dt - nu_n(t) dt + sigma dW_n - J_n dN_n ,"): (
        r"\mathrm{d}h_n = \mu\,(1-h_n)\,\mathrm{d}t - \nu_n(t)\,\mathrm{d}t "
        r"+ \sigma\,\mathrm{d}W_n - J_n\,\mathrm{d}N_n ,", True),
    _norm("nu_n(t) = nu_0 k_n m_s(c_n) [1 + g [rho_n(t) - rho^*]_+^{3/2}] ,"): (
        r"\nu_n(t) = \nu_0\,k_n\,m_s(c_n)\left[1 + g\,"
        r"\bigl[\rho_n(t)-\rho^{\star}\bigr]_{+}^{3/2}\right] ,", True),
    _norm("h_i(t^+) = max(0, h_i(t) - alpha_i Z_t) ,  "
          "Z_t ~ Gamma(shape = 2, scale = beta / 2) ,"): (
        r"h_i(t^{+}) = \max\!\bigl(0,\;h_i(t) - \alpha_i Z_t\bigr) ,\qquad "
        r"Z_t \sim \Gamma(\text{shape}=2,\ \text{scale}=\beta/2) ,", True),
    _norm("h_n(t+1) <- Pi_{[0,1]}( h_n(t) + mu (1 - h_n(t)) - nu_n(t) + sigma eps_t "
          "- J_n dN_n ) ,   Pi_{[0,1]}(x) = min(1, max(0, x)) ,"): (
        r"h_n(t+1) \;\leftarrow\; \Pi_{[0,1]}\!\Bigl(h_n(t) + \mu\bigl(1-h_n(t)\bigr) "
        r"- \nu_n(t) + \sigma\,\varepsilon_t - J_n\,\mathrm{d}N_n\Bigr) ,\qquad "
        r"\Pi_{[0,1]}(x) = \min\bigl(1,\max(0,x)\bigr) ,", False),
    _norm("L_n(t) = L_n^0 [1 + a_1 d_n(t)^2 + a_2 q_n(t)] e^{eps},  "
          "eps ~ N(0, sigma_o^2) ,"): (
        r"L_n(t) = L_n^{0}\left[1 + a_1 d_n(t)^{2} + a_2 q_n(t)\right]e^{\varepsilon} ,"
        r"\qquad \varepsilon \sim \mathcal{N}(0,\sigma_o^{2}) ,", True),
    _norm("p_n(t) = Pi_{[0,1]}( [p_n^0 + b d_n(t)^3 + c [rho~_n(t) - 0.8]_+^2] e^{eps_p} ) ,  "
          "eps_p ~ N(0, (1.6 sigma_o)^2) ,"): (
        r"p_n(t) = \Pi_{[0,1]}\!\Bigl(\bigl[p_n^{0} + b\,d_n(t)^{3} "
        r"+ c\,[\tilde{\rho}_n(t)-0.8]_{+}^{2}\bigr]e^{\varepsilon_p}\Bigr) ,\qquad "
        r"\varepsilon_p \sim \mathcal{N}\bigl(0,(1.6\,\sigma_o)^{2}\bigr) ,", True),
    _norm("S_n(t) = S_n^0 - gamma d_n(t)^{1.2} + eta ,  eta ~ N(0, sigma_S^2) ,"): (
        r"S_n(t) = S_n^{0} - \gamma\,d_n(t)^{1.2} + \eta ,\qquad "
        r"\eta \sim \mathcal{N}(0,\sigma_S^{2}) ,", True),
    _norm("phi_n(t) = w_L L_n(t)/L_ref + w_P p_n(t) + w_U max(0, rho_n - 0.75)^2 "
          "+ w_R r_n(t)^2 ,"): (
        r"\phi_n(t) = w_L\frac{L_n(t)}{L_{\mathrm{ref}}} + w_P\,p_n(t) "
        r"+ w_U\bigl[\rho_n(t)-0.75\bigr]_{+}^{2} + w_R\,r_n(t)^{2} ,", True),
    _norm("c_uv(t) = w_L d_uv / L_ref + [phi_u(t) + phi_v(t)] / 2 ,"): (
        r"c_{uv}(t) = w_L\frac{d_{uv}}{L_{\mathrm{ref}}} "
        r"+ \tfrac{1}{2}\left[\phi_u(t)+\phi_v(t)\right] ,", True),
    _norm("y_n(t) = 1{ exists u in (t, t+H] : state_n(u) = FAILED } ,"): (
        r"y_n(t) = \mathbf{1}\left\{\exists\,u \in (t,\,t+H] : "
        r"\mathrm{state}_n(u)=\mathrm{FAILED}\right\} ,", True),
    _norm("tau_c = min(0.95, kappa b_c theta*) ,"): (
        r"\tau_c = \min\!\left(0.95,\ \kappa\,b_c\,\theta^{\star}\right) ,", True),
    _norm("J(P) = D(P)/D_c^max + 4 p_loss(P)/p_c^max + (3 + 0.6 chi_c) R_prod(P) + Pen(P) ,"): (
        r"J_c(P) = \frac{D(P)}{D_c^{\max}} + 4\,\frac{p_{\mathrm{loss}}(P)}{p_c^{\max}} "
        r"+ \left(3+0.6\,\chi_c\right)R_{\mathrm{prod}}(P) + \mathrm{Pen}(P) ,", True),
    _norm("Pen(P) = 2.5 * 1{residual capacity < demand} + 1.5 * 1{D(P) > D_c^max} "
          "+ 1.5 * 1{p_loss(P) > p_c^max} ,"): (
        r"\mathrm{Pen}(P) = 2.5\,\mathbf{1}\{\text{residual capacity} < \text{demand}\} "
        r"+ 1.5\,\mathbf{1}\{D(P) > D_c^{\max}\} "
        r"+ 1.5\,\mathbf{1}\{p_{\mathrm{loss}}(P) > p_c^{\max}\} ,", False),
    _norm("r > p* = C_m / C_f ."): (
        r"r > p^{\star} = \frac{C_m}{C_f} .", True),
    _norm("R_c(t) = P(T_f > t | c_f = c) ,"): (
        r"R_c(t) = \Pr\!\left(T_f > t \,\middle|\, c_f = c\right) ,", True),
    _norm("A = 1 - sum_f o_f / sum_f a_f ,"): (
        r"A = 1 - \frac{\sum_f o_f}{\sum_f a_f} ,", True),
    _norm("L = sum_{t=1..T} [Q_0 - Q(t)]_+ ,"): (
        r"L = \sum_{t=1}^{T}\left[Q_0 - Q(t)\right]_{+} ,", True),
    _norm("Rn = 1 - L / (T Q_0) ."): (
        r"\mathcal{R} = 1 - \frac{L}{T\,Q_0} .", True),
    _norm("L_w^{cls} = T [1 - sum_c w_c A_c / sum_c w_c] ,   "
          "L_w^{flw} = T [1 - sum_c w_c pi_c A_c / sum_c w_c pi_c] ,"): (
        r"\begin{aligned}"
        r"L_w^{\mathrm{cls}} &= T\left[1-\frac{\sum_c w_c A_c}{\sum_c w_c}\right] ,\\[2pt]"
        r"L_w^{\mathrm{flw}} &= T\left[1-\frac{\sum_c w_c \pi_c A_c}{\sum_c w_c \pi_c}\right] ,"
        r"\end{aligned}", True),
    _norm("T_lead = t_failure - t_alarm ,"): (
        r"T_{\mathrm{lead}} = t_{\mathrm{failure}} - t_{\mathrm{alarm}} ,", True),
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
    "ν": r"$\nu$", "χ": r"$\chi$", "π": r"$\pi$", "η": r"$\eta$",
    "Π": r"$\Pi$", "ε": r"$\varepsilon$", "ω": r"$\omega$", "Σ": r"$\Sigma$",
    "₀": r"$_0$", "⁻": r"$^{-}$", "√": r"$\sqrt{\,}$", "≈": r"$\approx$",
    "∈": r"$\in$", "→": r"$\rightarrow$", "−": "-",
}


def cite(text: str) -> str:
    """Turn the numeric citations of the Word version into \\cite commands."""
    text = re.sub(r"\[(\d+)\](?:,\s*\[(\d+)\])*", _cite_group, text)
    return text


def _cite_group(m: re.Match) -> str:
    nums = re.findall(r"\d+", m.group(0))
    return "\\cite{" + ",".join(f"ref{n}" for n in nums) + "}"


MATH_TOKEN = "\u0001MATH%d\u0001"


GREEK = {
    "alpha": r"\alpha", "beta": r"\beta", "gamma": r"\gamma", "delta": r"\delta",
    "epsilon": r"\varepsilon", "eps": r"\varepsilon", "eta": r"\eta",
    "theta": r"\theta", "kappa": r"\kappa", "lambda": r"\lambda", "mu": r"\mu",
    "nu": r"\nu", "pi": r"\pi", "rho": r"\rho", "sigma": r"\sigma", "tau": r"\tau",
    "phi": r"\phi", "chi": r"\chi", "omega": r"\omega", "Pi": r"\Pi",
    "Gamma": r"\Gamma", "Delta": r"\Delta",
}

# Identifiers that look like maths but are ordinary prose or file names.
_MATH_BLOCKLIST = {"P0", "P1", "P2", "P3", "P4", "P5", "S1", "S2", "S3", "S4", "S5",
                   "S6", "S7", "M0", "M1", "M2", "M3", "C4", "RQ1", "RQ2", "RQ3", "RQ4"}


def _mathify_token(tok: str) -> str:
    """Render one ASCII maths token, e.g. nu_n(t) or L_c^max, as LaTeX."""
    body = tok
    # function argument, if any
    arg = ""
    m = re.match(r"^([^()]+)(\([^()]*\))$", body)
    if m:
        body, arg = m.group(1), m.group(2)
        arg = re.sub(r"\b(" + "|".join(GREEK) + r")\b",
                     lambda g: GREEK[g.group(1)], arg)
    parts = re.split(r"([_^])", body)
    out = []
    for part in parts:
        if part in ("_", "^"):
            out.append(part)
        elif part in GREEK:
            out.append(GREEK[part])
        elif part in ("max", "min", "ref", "fail", "cap", "lead", "alarm", "failure",
                      "loss", "prod", "inf", "cls", "flw", "nom", "o", "S", "p"):
            out.append(r"\mathrm{%s}" % part if len(part) > 1 else part)
        else:
            out.append(part)
    # brace multi-character sub/superscripts
    joined = "".join(out)
    joined = re.sub(r"([_^])(\\mathrm\{[^}]*\}|[A-Za-z0-9\\]{2,})(?![A-Za-z0-9])",
                    lambda mm: mm.group(1) + "{" + mm.group(2) + "}", joined)
    for word in ("cls", "flw", "max", "min", "ref", "fail", "nom", "lead", "prod", "inf"):
        joined = re.sub(r"(?<!\\mathrm)\{%s\}" % word, r"{\\mathrm{%s}}" % word, joined)
    joined = joined.replace(r"\mathrm{\mathrm{", r"\mathrm{")
    joined = re.sub(r"\\mathrm\{(\w+)\}\}\}", r"\\mathrm{\1}}", joined)
    joined = joined.replace("*", r"^{\star}")
    return "$" + joined + arg.replace("*", r"^{\star}") + "$"


_MATH_TOKEN_RE = re.compile(
    r"(?<![\w$\\])"
    r"((?:[A-Za-z]{1,8})(?:[_^](?:\{[^}]{1,12}\}|[A-Za-z0-9*]{1,8}))+(?:\([^()]{0,10}\))?)"
    r"(?![\w])")
_GREEK_WORD_RE = re.compile(r"(?<![\w$\\])(" + "|".join(sorted(GREEK, key=len, reverse=True))
                            + r")(\*?)(?![\w])")
_STARRED_RE = re.compile(r"(?<![\w$\\])(p|r|q)\*(?![\w])")


_TILDE_RE = re.compile(r"(?<![\w$\\])rho_tilde_n\(t\)(?![\w])")


def mathify(text: str) -> str:
    """Convert the ASCII maths used in the manuscript prose into inline LaTeX.

    The manuscript generator writes prose in plain ASCII so that the Word version
    stays readable; this pass renders the same tokens properly for LaTeX. Only
    tokens carrying a subscript or superscript, or a bare Greek letter name, are
    converted, which keeps ordinary words untouched.
    """
    def _tok(m):
        t = m.group(1)
        if t in _MATH_BLOCKLIST or t.split("_")[0] in _MATH_BLOCKLIST:
            return t
        # snake_case English identifiers (leave_one_scenario_out, common_cause, ...)
        # are prose, not mathematics: a maths token has at most one subscript level
        # and its parts are short.
        head = t.split("(")[0]
        subs = head.split("_")
        if len(subs) > 2 and all(q.isalpha() and q.islower() and len(q) >= 2
                                 for q in subs):
            return t          # snake_case prose; the escaper handles it
        return _mathify_token(t)
    text = _TILDE_RE.sub(r"$\\tilde{\\rho}_n(t)$", text)
    text = _MATH_TOKEN_RE.sub(_tok, text)
    text = _GREEK_WORD_RE.sub(
        lambda m: "$" + GREEK[m.group(1)] + (r"^{\star}" if m.group(2) else "") + "$", text)
    text = _STARRED_RE.sub(lambda m: "$" + m.group(1) + r"^{\star}$", text)
    return text


def esc(text: str) -> str:
    """Escape a plain-text paragraph for LaTeX, protecting injected maths."""
    text = mathify(text)
    stash: List[str] = []
    # Protect the inline maths that mathify() just injected from the escaper below.
    def _stash_math(m):
        stash.append(m.group(0))
        return MATH_TOKEN % (len(stash) - 1)
    text = re.sub(r"\$[^$]+\$", _stash_math, text)
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


def _looks_like_equation(par) -> bool:
    """A centred, italic, non-empty paragraph is a display equation by construction."""
    if par.alignment != WD_ALIGN_PARAGRAPH.CENTER:
        return False
    runs = [r for r in par.runs if r.text.strip()]
    if not runs or not runs[0].italic:
        return False
    text = par.text.strip()
    # the centred author block is italic too; equations always contain a relation
    return bool(re.search(r"[=<>]|\\leftarrow|<-", text)) and "@" not in text


def equation_source(par) -> Optional[str]:
    """Return the source text of a display equation paragraph, or None.

    Display equations are the centred paragraphs whose first non-empty run is
    italic, which is exactly how the manuscript generator emits them. The
    trailing tabbed number, when present, is stripped: numbering is re-derived
    here so that the .docx and the .tex can never disagree about it.
    """
    if par.alignment != WD_ALIGN_PARAGRAPH.CENTER:
        return None
    runs = [r for r in par.runs if r.text.strip()]
    if not runs or not runs[0].italic:
        return None
    body = "".join(r.text for r in runs)
    m = re.match(r"^(.*?)\s*\(\d+\)\s*$", body, re.S)
    body = m.group(1) if m else body
    key = _norm(body)
    return key if key in LATEX_EQUATIONS else None


def table_to_latex(table, caption: str, label: str) -> str:
    """Render a table exactly where it occurs, never as a float.

    longtable places the table at the point of call and breaks it across pages
    if it is tall, which reproduces the flow of the Word version: no table ever
    migrates to the top of a later page and no paragraph is interrupted by one.
    Column widths are allocated from the measured content so that wide tables
    fill the text block instead of drifting past the right margin.
    """
    raw = [[c.text.replace("\n", " ").strip() for c in r.cells] for r in table.rows]
    ncol = len(raw[0])
    rows = [" & ".join(esc(c) for c in r[:ncol]) + r" \\" for r in raw]

    # Two measurements per column: the longest cell, which drives how much space
    # the column would like, and the longest unbreakable word, which is the width
    # below which the column will overflow into its neighbour.
    def longest_word(cell: str) -> int:
        return max((len(w) for w in re.split(r"[\s/]+", cell) if w), default=0)

    widths = [max(len(r[i]) if i < len(r) else 0 for r in raw) for i in range(ncol)]
    hard = [max(longest_word(r[i]) if i < len(r) else 0 for r in raw)
            for i in range(ncol)]
    total = sum(widths)
    numeric = [
        all(re.fullmatch(r"[-+]?[\d.,%\s\u2014eE\[\]<>]*", r[i] or "")
            for r in raw[1:] if i < len(r))
        for i in range(ncol)
    ]

    if total <= 95 and ncol <= 8:
        # narrow enough to set naturally
        colspec = "@{}l" + "".join("r" if numeric[i] else "l"
                                   for i in range(1, ncol)) + "@{}"
        size = r"\small" if total <= 70 else r"\footnotesize"
    else:
        # allocate the text block proportionally; a floor keeps narrow numeric
        # columns readable and stops the last column from being pushed out
        # A p-column table occupies sum(widths) + 2*tabcolsep*(ncol-1) once the
        # outer separators are removed with @{}. The inter-column space must
        # therefore be reserved exactly, not estimated, or every wide table
        # overflows the text block by precisely that shortfall.
        TEXTWIDTH_PT = 468.33                 # cas-sc \textwidth on A4
        colsep_pt = 4.0 if ncol <= 8 else (3.0 if ncol <= 10 else 2.0)
        reserve = 2.0 * colsep_pt * (ncol - 1) / TEXTWIDTH_PT
        usable = max(0.30, 1.0 - reserve - 0.010)      # 1 % safety margin
        # every column gets at least the room its longest word needs
        chars_per_line = 108.0                # measured for \footnotesize on A4
        floor = [max(0.045, (h + 1.2) / chars_per_line) for h in hard]
        free = max(0.0, usable - sum(floor))
        want = [w / total for w in widths]
        wsum = sum(want) or 1.0
        share = [floor[i] + free * want[i] / wsum for i in range(ncol)]
        over = sum(share) / usable
        if over > 1.0:                        # longest words alone overflow: shrink type
            share = [x / over for x in share]
        cols = []
        for i, x in enumerate(share):
            # RaggedRight/RaggedLeft keep hyphenation active, so a long word is
            # hyphenated instead of being split at an arbitrary point.
            align = r">{\RaggedLeft\arraybackslash}" if numeric[i] \
                else r">{\RaggedRight\arraybackslash}"
            cols.append(f"{align}p{{{x:.4f}\\linewidth}}")
        colspec = "@{}" + "".join(cols) + "@{}"
        size = r"\scriptsize" if (total > 160 or sum(hard) > 90) else r"\footnotesize"

    head = rows[0]
    body = "\n".join(rows[1:])
    sep = 4.0 if (total <= 95 and ncol <= 8) else (
        4.0 if ncol <= 8 else (3.0 if ncol <= 10 else 2.0))
    return (
        "\n\\begingroup\n" + size + f"\n\\setlength{{\\tabcolsep}}{{{sep:g}pt}}\n"
        "\\renewcommand{\\arraystretch}{1.12}\n"
        f"\\begin{{longtable}}{{{colspec}}}\n"
        f"\\caption{{{caption}}}\\label{{{label}}}\\\\\n"
        f"\\toprule\n{head}\n\\midrule\n\\endfirsthead\n"
        f"\\multicolumn{{{ncol}}}{{@{{}}l}}{{\\textit{{Table~\\ref{{{label}}} continued}}}}\\\\\n"
        f"\\toprule\n{head}\n\\midrule\n\\endhead\n"
        f"\\midrule\n\\multicolumn{{{ncol}}}{{r@{{}}}}{{\\textit{{continued on the next page}}}}\\\\\n"
        "\\endfoot\n\\bottomrule\n\\endlastfoot\n"
        f"{body}\n"
        "\\end{longtable}\n\\endgroup\n"
    )


def figure_to_latex(name: str, caption: str, label: str, width: float = 0.92) -> str:
    """Place a figure inline, in document order, without a float environment."""
    return (
        "\n\\begin{center}\n"
        f"\\includegraphics[width={width}\\linewidth,"
        "height=0.42\\textheight,keepaspectratio]{" + name + "}\n"
        f"\\captionof{{figure}}{{{caption}}}\\label{{{label}}}\n"
        "\\end{center}\n"
    )


PREAMBLE = r"""%% Elsevier CAS template (els-cas-templates), double-column article class.
%% Place cas-sc.cls and cas-common.sty from the official els-cas-templates
%% bundle in this directory before compiling.
%% Single-column CAS layout. Elsevier accepts either cas-sc or cas-dc; the
%% single-column setting is used deliberately because every table and figure in
%% this manuscript is placed inline, in document order, and the wide result
%% tables need the full text block.
\documentclass[a4paper,fleqn]{cas-sc}

% cas-sc loads STIX fonts when they are installed (as on Editorial Manager and full
% TeX Live) and Computer Modern otherwise. Latin Modern and amssymb are loaded only in
% the second case, because mixing them with STIX remaps math symbols (\pm prints as a comma).
\makeatletter
\@ifpackageloaded{stix}{%
  \renewcommand{\sfdefault}{lmss}\renewcommand{\ttdefault}{lmtt}%
}{\usepackage{lmodern}}
\makeatother
\usepackage[numbers,sort&compress]{natbib}
\usepackage{amsmath}
\makeatletter\@ifpackageloaded{stix}{}{\usepackage{amssymb}}\makeatother
\usepackage{booktabs}
\usepackage{array}
\usepackage{longtable}
\usepackage{graphicx}
\usepackage{multirow}
\usepackage{url}
\usepackage{capt-of}
\usepackage{ragged2e}

%% Tables and figures are set inline rather than as floats, so nothing migrates
%% away from the text that introduces it and no paragraph is interrupted.
\setlength{\LTcapwidth}{\linewidth}
\setlength{\LTpre}{6pt}
\setlength{\LTpost}{6pt}
\emergencystretch=2em
\hbadness=10000
\vbadness=10000
\raggedbottom
\clubpenalty=300
\widowpenalty=300
\displaywidowpenalty=300
\interlinepenalty=0
\LTchunksize=40

\graphicspath{{figures/}}

\begin{document}

\let\WriteBookmarks\relax
\def\floatpagepagefraction{1}
\def\textpagefraction{.001}

\shorttitle{Where does predictive resilience come from?}
\shortauthors{N. Tarasov and R. Zinko}

\title[mode=title]{Where Does Predictive Resilience Come From? A Simulation-Based Digital-Twin
Assessment of Failure Anticipation, Protection and Risk-Aware Recovery in Hybrid
5G/6G-Class and LEO Satellite Networks}

%% First author
\author[1]{Nikita Tarasov}[orcid=0000-0003-1870-3436]
\cormark[1]
\ead{dev.nikita@outlook.com}
\credit{Conceptualization, Methodology, Software, Investigation, Writing -- original draft}

%% Second author
\author[1]{Roman Zinko}[orcid=0000-0002-3275-8188]
\ead{roman.v.zinko@lpnu.ua}
\credit{Validation, Writing -- review \& editing}

\affiliation[1]{organization={Lviv Polytechnic National University,
                Department of Computer-Aided Design Systems},
            addressline={S.\,Bandery Street 12},
            city={Lviv},
            postcode={79013},
            country={Ukraine}}

\cortext[cor1]{Corresponding author}
"""


PREVIEW_PREAMBLE = r"""%% Dependency-free preview build.
%%
%% This file exists so that the manuscript can be compiled and proof-read with a
%% stock TeX Live installation, before the two Elsevier class files are added.
%% It shares body.tex with manuscript.tex, so the text, figures, tables and
%% bibliography are identical; only the page layout differs. Do NOT submit the
%% PDF produced from this file - submit the one produced from manuscript.tex.
\documentclass[11pt]{article}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{graphicx,amsmath,amssymb,booktabs,longtable,array,geometry,hyperref}
\geometry{margin=2.2cm}
\usepackage{capt-of}
\usepackage{ragged2e}
\setlength{\LTcapwidth}{\linewidth}
\setlength{\LTpre}{6pt}
\setlength{\LTpost}{6pt}
\emergencystretch=2em
\hbadness=10000
\vbadness=10000
\raggedbottom
\clubpenalty=300
\widowpenalty=300
\displaywidowpenalty=300
\interlinepenalty=0
\LTchunksize=40
\usepackage{caption}
\captionsetup{font=small,labelfont=bf}
\graphicspath{{figures/}{./}}

%% Minimal stand-ins for the CAS commands used in body.tex and the author block.
\providecommand{\cormark}[1][]{$^{*}$}
\providecommand{\fnmark}[1][]{}
\providecommand{\cortext}[2][]{}
\providecommand{\fntext}[2][]{}
\providecommand{\ead}[2][]{}
\providecommand{\credit}[1]{}
\providecommand{\printcredits}{}
\providecommand{\tnotetext}[2][]{}
\providecommand{\tnotemark}[1][]{}
\providecommand{\shorttitle}[1]{}
\providecommand{\shortauthors}[1]{}
\newcommand{\affiliation}[2][]{}
\providecommand{\sep}{\textbullet\ }
\newenvironment{keywords}{\par\medskip\noindent\textbf{Keywords:}\ }{\par\medskip}
\let\WriteBookmarks\relax

\title{Where Does Predictive Resilience Come From? A Simulation-Based Digital-Twin
Assessment of Failure Anticipation, Protection and Risk-Aware Recovery in Hybrid
5G/6G-Class and LEO Satellite Networks}
\author{Nikita Tarasov\thanks{Corresponding author: dev.nikita@outlook.com.
ORCID 0000-0003-1870-3436.} \and Roman Zinko\thanks{ORCID 0000-0002-3275-8188.}}
\date{Department of Computer-Aided Design Systems, Lviv Polytechnic National
University, S.\,Bandery Street 12, 79013 Lviv, Ukraine}

\begin{document}
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
    eq_counter = [0]
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
            # Strip the whole label, not just up to the first ". ": splitting on
            # ". " alone consumes "Fig." and leaves the number behind, which the
            # LaTeX class then numbers a second time ("Figure 1: 1. ...").
            caption = esc(re.sub(r"^(?:Fig\.|Figure|Table)\s+\d+[.:]\s*", "", text))
            if kind == "figure":
                order = FIGURE_ORDER
                _fj = docx_path + ".figures.json"
                if os.path.exists(_fj):
                    import json as _json
                    order = _json.load(open(_fj))
                name = order[fig_idx] if fig_idx < len(order) else None
                fig_idx += 1
                if name:
                    src = os.path.join(figdir, name)
                    if os.path.exists(src):
                        shutil.copy(src, os.path.join(out_dir, "figures", name))
                    body.append(figure_to_latex(name, caption, f"fig:{number}"))
            else:
                pending_table_caption = caption
            continue

        _close_itemize(body)
        key = equation_source(par)
        if key is not None:
            latex, numbered = LATEX_EQUATIONS[key]
            if numbered:
                eq_counter[0] += 1
                k = eq_counter[0]
                body.append("\n\\begin{equation}\n" + latex
                            + f"\n\\label{{eq:{k}}}\n\\end{{equation}}\n")
            else:
                body.append("\n\\begin{equation*}\n" + latex + "\n\\end{equation*}\n")
            continue
        if _looks_like_equation(par):
            raise RuntimeError(
                "Unmapped display equation in the manuscript; add it to "
                "LATEX_EQUATIONS in build_latex.py:\n  " + _norm(par.text))

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

    front = []
    front.append("\n\\begin{abstract}\n" + esc(" ".join(abstract)) + "\n\\end{abstract}\n")
    # One keyword per line: the CAS classes set the keyword block in a box, and a
    # single long line overflows the text width by exactly its excess.
    kw = " \\sep\n".join(esc(k.strip()) for k in keywords.split(";") if k.strip())
    front.append("\\begin{keywords}\n" + kw + "\n\\end{keywords}\n")
    front.append("\\maketitle\n")

    # Both master files are self-contained: the manuscript that is submitted needs
    # no auxiliary .tex include, and the preview is written from the same in-memory
    # content in the same build call, so the two cannot diverge.
    content = "\n".join(front + body + bib)

    path = os.path.join(out_dir, "manuscript.tex")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(PREAMBLE + "\n" + content + "\n" + POSTAMBLE)

    with open(os.path.join(out_dir, "preview.tex"), "w", encoding="utf-8") as fh:
        fh.write(PREVIEW_PREAMBLE + "\n" + content + "\n" + POSTAMBLE)
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
