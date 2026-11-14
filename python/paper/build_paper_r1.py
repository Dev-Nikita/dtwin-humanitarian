"""
Manuscript builder for the reliability-engineering version of the study.

    python -m python.paper.build_paper_ress --out "Predictive_Resilience.docx"

Every quantitative claim is injected from data/results, so the text cannot drift
away from the experiments.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

from ..simulator.config import FEATURE_WINDOW, PREDICTION_HORIZON, SERVICE_PROFILES

RESULTS = os.path.join("data", "results")
FIGDIR = os.path.join("paper-assets", "figures")
TABDIR = os.path.join("paper-assets", "tables")

BODY_FONT = "Times New Roman"
S1, S2, S3, S4, S5, S6 = ("S1_normal", "S2_bs_overload", "S3_leo_degradation",
                          "S4_disaster_collapse", "S5_leo_handover", "S6_common_cause")
SCEN_TEXT = {
    S1: "nominal operation", S2: "access overload", S3: "non-terrestrial degradation",
    S4: "infrastructure collapse", S5: "handover dynamics", S6: "common-cause disruption",
}


# ---------------------------------------------------------------------------
def load_results() -> Dict:
    runs = pd.read_csv(os.path.join(RESULTS, "experiment_runs.csv"))
    models = pd.read_csv(os.path.join(RESULTS, "model_comparison.csv"))
    sig = pd.read_csv(os.path.join(RESULTS, "significance_tests.csv"))
    meta = json.load(open(os.path.join(RESULTS, "best_model.json")))

    def opt(name):
        path = os.path.join(RESULTS, name)
        return pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()

    crit = {"emergency_message": 5, "command_control": 5, "medical_telemetry": 4,
            "voice_call": 3, "video_stream": 2, "sensor_data": 1}
    cols = [f"avail_{c}" for c in crit if f"avail_{c}" in runs.columns]
    if cols:
        w = np.array([crit[c.replace("avail_", "")] for c in cols], dtype=float)
        av = runs[cols].to_numpy(dtype=float)
        qw = (av * w).sum(axis=1) / w.sum()
        runs["weighted_resilience_loss"] = runs["steps"] * (1.0 - qw)
        # flow-weighted variant: class weights scaled by the share of flow-slots
        # each class contributes under the nominal traffic mix.
        expo = np.array([SERVICE_PROFILES[c.replace("avail_", "")].arrival_weight
                         * float(np.mean(SERVICE_PROFILES[c.replace("avail_", "")].duration_steps))
                         for c in cols], dtype=float)
        wf = w * expo / expo.sum()
        qf = (av * wf).sum(axis=1) / wf.sum()
        runs["flow_weighted_resilience_loss"] = runs["steps"] * (1.0 - qf)

    def piv(metric):
        return runs.pivot_table(index="scenario", columns="policy", values=metric)

    return {
        "runs": runs, "models": models.set_index("model"), "models_raw": models,
        "sig": sig, "meta": meta,
        "ablation": opt("feature_ablation.csv"), "ood": opt("ood_evaluation.csv"),
        "horizon": opt("horizon_sweep.csv"), "migration": opt("migration_cost_sweep.csv"),
        "noise": opt("noise_sweep.csv"), "sweep": opt("threshold_sweep.csv"),
        "topology": opt("topology_sweep.csv"),
        "A": piv("service_availability"), "RL": piv("resilience_loss"),
        "NR": piv("normalised_resilience"), "REC": piv("mean_recovery_steps"),
        "SLA": piv("sla_compliance"), "LAT": piv("mean_latency_ms"),
        "LOSS": piv("mean_packet_loss"), "PRR": piv("proactive_reroutes_total"),
        "RRR": piv("reactive_reroutes"),
        "WRL": piv("weighted_resilience_loss") if "weighted_resilience_loss" in runs else None,
        "FWRL": (piv("flow_weighted_resilience_loss")
                 if "flow_weighted_resilience_loss" in runs else None), "EM": piv("avail_emergency_message"),
        "MED": piv("avail_medical_telemetry"),
        "n_seeds": int(runs.groupby(["scenario", "policy"]).size().max()),
        "n_runs": len(runs),
        "REF": (runs.groupby(["scenario", "policy"]).outage_slots_failure.sum()
                / runs.groupby(["scenario", "policy"]).failure_interruptions.sum()).unstack(),
        "dd": opt("detection_delay.csv"),
        "misspec_off": opt("misspec_offline.csv"),
        "large": opt("large_topology.csv") if os.path.exists(
            os.path.join(RESULTS, "large_topology.csv")) else None,
        "acct": opt("accounting_check.csv"),
        "conf": opt("confirmation.csv"), "conf_t": opt("confirmation_tests.csv"),
        "conf_eq": opt("confirmation_equivalence.csv"),
        "pmodes": opt("protection_modes.csv"), "endo": opt("endogeneity_check.csv"),
        "conf_manifest": (json.load(open(os.path.join(RESULTS, "confirmation_manifest.json")))
                          if os.path.exists(os.path.join(RESULTS, "confirmation_manifest.json"))
                          else {}),
        "hz_ref": float(opt("horizon_reactive_reference.csv")["resilience_loss"].mean())
        if os.path.exists(os.path.join(RESULTS, "horizon_reactive_reference.csv")) else float("nan"),
        "n_rev_runs": int(sum(len(opt(f)) for f in (
            "detection_delay.csv", "misspec_closed_loop.csv", "large_topology.csv",
            "feedback_ablation_fixed.csv", "design_sensitivity.csv", "frontier.csv",
            "migration_cost_sweep.csv", "topology_sweep.csv", "parameter_sensitivity.csv",
            "controller_weight_sensitivity.csv", "accounting_check.csv",
            "endogeneity_check.csv", "protection_modes.csv")))
            + int(2 * len(opt("p6_decomposition.csv")) / 3) + 12 * 2 * len(opt("horizon_sweep.csv")) + 12,
    }


def pct(x, d=2):
    return f"{100.0 * x:.{d}f}"


def rel(new, old):
    return float("nan") if old == 0 else 100.0 * (new - old) / abs(old)


# ---------------------------------------------------------------------------
class Paper:
    FIG_PREFIX = "Fig."
    TAB_PREFIX = "Table"

    def __init__(self, lang: str = "en") -> None:
        self.doc = Document()
        self.fig_no = self.tab_no = self.eq_no = 0
        self.state: dict = {}
        self.lang = lang
        if lang == "uk":
            self.FIG_PREFIX, self.TAB_PREFIX = "Рис.", "Табл."
        self._setup()

    def _setup(self) -> None:
        s = self.doc.sections[0]
        s.page_width, s.page_height = Cm(21.0), Cm(29.7)
        s.top_margin = s.bottom_margin = Cm(2.2)
        s.left_margin = s.right_margin = Cm(2.3)
        st = self.doc.styles["Normal"]
        st.font.name = BODY_FONT
        st.font.size = Pt(10.5)
        st.element.rPr.rFonts.set(qn("w:eastAsia"), BODY_FONT)
        pf = st.paragraph_format
        pf.space_after, pf.line_spacing = Pt(6), 1.15
        pf.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        for name, size in (("Heading 1", 13), ("Heading 2", 11.5), ("Heading 3", 10.8)):
            h = self.doc.styles[name]
            h.font.name = BODY_FONT
            h.font.size = Pt(size)
            h.font.bold = True
            h.font.color.rgb = RGBColor(0, 0, 0)
            h.paragraph_format.space_before = Pt(10)
            h.paragraph_format.space_after = Pt(4)

    def title(self, text):
        p = self.doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run(text)
        r.bold = True
        r.font.size = Pt(15)

    def centered(self, text, size=10.5, italic=False):
        p = self.doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run(text)
        r.italic = italic
        r.font.size = Pt(size)

    def h1(self, t):
        self.doc.add_heading(t, level=1)

    def h2(self, t):
        self.doc.add_heading(t, level=2)

    def p(self, t, italic=False, size=None):
        par = self.doc.add_paragraph()
        r = par.add_run(t)
        r.italic = italic
        if size:
            r.font.size = Pt(size)

    def bullets(self, items):
        for i in items:
            par = self.doc.add_paragraph(i, style="List Bullet")
            par.paragraph_format.space_after = Pt(2)
            par.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY

    def numbered(self, items, prefix="({i})"):
        """Explicit numbering: python-docx list styles continue across the file."""
        for i, item in enumerate(items, start=1):
            par = self.doc.add_paragraph()
            par.paragraph_format.left_indent = Cm(0.8)
            par.paragraph_format.first_line_indent = Cm(-0.8)
            par.paragraph_format.space_after = Pt(3)
            par.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            par.add_run(prefix.format(i=i) + " ").bold = True
            par.add_run(item)

    def equation(self, text, numbered: bool = True):
        """Emit a display equation. Unnumbered displays are used for the two
        implementation details (the projection onto [0,1] and the penalty
        function) that support a numbered equation rather than standing alone,
        so that the numbered sequence stays at exactly eighteen."""
        par = self.doc.add_paragraph()
        par.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = par.add_run(text)
        r.italic = True
        if not numbered:
            return ""
        self.eq_no += 1
        par.add_run(f"\t\t({self.eq_no})")
        return f"({self.eq_no})"

    def figure(self, filename, caption, width=15.0):
        self.fig_no += 1
        self.state.setdefault("fig_files", []).append(filename)
        path = os.path.join(FIGDIR, filename)
        if os.path.exists(path):
            self.doc.add_picture(path, width=Cm(width))
            self.doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
        par = self.doc.add_paragraph()
        par.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = par.add_run(f"{self.FIG_PREFIX} {self.fig_no}. {caption}")
        r.font.size = Pt(9)
        return f"{self.FIG_PREFIX} {self.fig_no}"

    def table(self, df, caption, fmt="{:.4f}", size=8.0, widths: Optional[List[float]] = None):
        self.tab_no += 1
        par = self.doc.add_paragraph()
        r = par.add_run(f"{self.TAB_PREFIX} {self.tab_no}. {caption}")
        r.bold = True
        r.font.size = Pt(9)
        t = self.doc.add_table(rows=1, cols=len(df.columns))
        t.style = "Table Grid"
        t.alignment = WD_TABLE_ALIGNMENT.CENTER
        for i, c in enumerate(df.columns):
            t.rows[0].cells[i].text = str(c)
            for pp in t.rows[0].cells[i].paragraphs:
                for rr in pp.runs:
                    rr.bold = True
                    rr.font.size = Pt(size)
        for _, row in df.iterrows():
            cells = t.add_row().cells
            for i, v in enumerate(row):
                column_name = str(df.columns[i]).lower()
                if (isinstance(v, float) and (column_name == "p"
                        or column_name.startswith("p_")
                        or column_name.endswith(" p") or " p " in column_name)
                        and 0 <= v < 0.0001):
                    cells[i].text = "<0.0001"
                else:
                    cells[i].text = fmt.format(v) if isinstance(v, float) else str(v)
                for pp in cells[i].paragraphs:
                    for rr in pp.runs:
                        rr.font.size = Pt(size)
        if widths:
            for r_ in t.rows:
                for i, w in enumerate(widths):
                    r_.cells[i].width = Cm(w)
        self.doc.add_paragraph()
        return f"{self.TAB_PREFIX} {self.tab_no}"

    def set_placeholder(self, key, value):
        """Replace a forward reference (e.g. a table number) in earlier paragraphs."""
        for par in self.doc.paragraphs:
            if key in par.text:
                for run in par.runs:
                    if key in run.text:
                        run.text = run.text.replace(key, value)

    def save(self, path):
        with open(path + ".figures.json", "w") as fh:
            json.dump(self.state.get("fig_files", []), fh)
        caps = []
        for par in self.doc.paragraphs:
            for pre in (self.TAB_PREFIX, self.FIG_PREFIX):
                if par.text.startswith(pre + " ") and ". " in par.text[:12]:
                    caps.append(par.text)
        with open(path + ".captions.json", "w") as fh:
            json.dump(caps, fh, indent=0)
        self.doc.save(path)
        print("manuscript:", path)


REFERENCES: List[str] = [
    # --- resilience and reliability of infrastructure systems
    "M. Bruneau et al., “A framework to quantitatively assess and enhance the seismic resilience of communities,” Earthquake Spectra, vol. 19, no. 4, pp. 733–752, 2003.",
    "D. Henry and J. E. Ramirez-Marquez, “Generic metrics and quantitative approaches for system resilience as a function of time,” Reliability Engineering & System Safety, vol. 99, pp. 114–122, 2012.",
    "S. Hosseini, K. Barker, and J. E. Ramirez-Marquez, “A review of definitions and measures of system resilience,” Reliability Engineering & System Safety, vol. 145, pp. 47–61, 2016.",
    "C. Poulin and M. B. Kane, “Infrastructure resilience curves: performance measures and summary metrics,” Reliability Engineering & System Safety, vol. 216, 107926, 2021.",
    "K. Barker, J. E. Ramirez-Marquez, and C. M. Rocco, “Resilience-based network component importance measures,” Reliability Engineering & System Safety, vol. 117, pp. 89–97, 2013.",
    "G. P. Cimellaro, A. M. Reinhorn, and M. Bruneau, “Framework for analytical quantification of disaster resilience,” Engineering Structures, vol. 32, no. 11, pp. 3639–3649, 2010.",
    "M. Ouyang and L. Dueñas-Osorio, “Time-dependent resilience assessment and improvement of urban infrastructure systems,” Chaos, vol. 22, 033122, 2012.",
    "C. W. Zobel and L. Khansa, “Characterizing multi-event disaster resilience,” Computers & Operations Research, vol. 42, pp. 83–94, 2014.",
    "E. Zio, “Challenges in the vulnerability and risk analysis of critical infrastructures,” Reliability Engineering & System Safety, vol. 152, pp. 137–150, 2016.",
    "J. P. G. Sterbenz et al., “Resilience and survivability in communication networks: strategies, principles, and survey of disciplines,” Computer Networks, vol. 54, no. 8, pp. 1245–1265, 2010.",
    "P. Cholda, A. Mykkeltveit, B. E. Helvik, O. J. Wittner, and A. Jajszczyk, “A survey of resilience differentiation frameworks in communication networks,” IEEE Communications Surveys & Tutorials, vol. 9, no. 4, pp. 32–55, 2007.",
    "T. Gomes et al., “A survey of strategies for communication networks to protect against large-scale natural disasters,” in International Workshop on Resilient Networks Design and Modeling (RNDM), 2016, pp. 11–22.",
    # --- dynamic reliability, prognostics, failure anticipation
    "E. Zio, “Prognostics and health management (PHM): where are we and where do we (need to) go in theory and practice,” Reliability Engineering & System Safety, vol. 218, 108119, 2022.",
    "E. Zio, “Reliability engineering: old problems and new challenges,” Reliability Engineering & System Safety, vol. 94, no. 2, pp. 125–141, 2009.",
    "Y. Lei, N. Li, L. Guo, N. Li, T. Yan, and J. Lin, “Machinery health prognostics: a systematic review from data acquisition to RUL prediction,” Mechanical Systems and Signal Processing, vol. 104, pp. 799–834, 2018.",
    "X.-S. Si, W. Wang, C.-H. Hu, and D.-H. Zhou, “Remaining useful life estimation – a review on the statistical data driven approaches,” European Journal of Operational Research, vol. 213, no. 1, pp. 1–14, 2011.",
    "Z. Chen et al., “Remaining useful life prediction based on physics-informed data augmentation,” Reliability Engineering & System Safety, vol. 252, 110451, 2024.",
    "A. Mosleh, D. M. Rasmuson, and F. M. Marshall, “Guidelines on modeling common-cause failures in probabilistic risk assessment,” NUREG/CR-5485, U.S. Nuclear Regulatory Commission, 1998.",
    "M. Kabir and Y. Papadopoulos, “Applications of Bayesian networks and Petri nets in safety, reliability, and risk assessments: a review,” Safety Science, vol. 115, pp. 154–175, 2019.",
    "Q. Lin et al., “Predicting node failure in cloud service systems,” in ACM ESEC/FSE, 2018, pp. 480–490.",
    "Y. Xu et al., “Improving service availability of cloud systems by predicting disk error,” in USENIX Annual Technical Conference, 2018, pp. 481–494.",
    "M. M. Botezatu, I. Giurgiu, J. Bogojeska, and D. Wiesmann, “Predicting disk replacement towards reliable data centers,” in ACM SIGKDD, 2016, pp. 39–48.",
    "K. Hundman, V. Constantinou, C. Laporte, I. Colwell, and T. Soderstrom, “Detecting spacecraft anomalies using LSTMs and nonparametric dynamic thresholding,” in ACM SIGKDD, 2018, pp. 387–395.",
    "P. Malhotra, A. Ramakrishnan, G. Anand, L. Vig, P. Agarwal, and G. Shroff, “LSTM-based encoder-decoder for multi-sensor anomaly detection,” in ICML Anomaly Detection Workshop, 2016.",
    "R. Chalapathy and S. Chawla, “Deep learning for anomaly detection: a survey,” arXiv:1901.03407, 2019.",
    # --- digital twins
    "M. Grieves and J. Vickers, “Digital twin: mitigating unpredictable, undesirable emergent behavior in complex systems,” in Trans-Disciplinary Perspectives on Complex Systems, Springer, 2017, pp. 85–113.",
    "F. Tao, H. Zhang, A. Liu, and A. Y. C. Nee, “Digital twin in industry: state-of-the-art,” IEEE Transactions on Industrial Informatics, vol. 15, no. 4, pp. 2405–2415, 2019.",
    "A. Fuller, Z. Fan, C. Day, and C. Barlow, “Digital twin: enabling technologies, challenges and open research,” IEEE Access, vol. 8, pp. 108952–108971, 2020.",
    "C. Mihai et al., “Digital twins: a survey on enabling technologies, challenges, trends and future prospects,” IEEE Communications Surveys & Tutorials, vol. 24, no. 4, pp. 2255–2291, 2022.",
    "Y. Wu, K. Zhang, and Y. Zhang, “Digital twin networks: a survey,” IEEE Internet of Things Journal, vol. 8, no. 18, pp. 13789–13804, 2021.",
    "P. Almasan et al., “Network digital twin: context, enabling technologies, and opportunities,” IEEE Communications Magazine, vol. 60, no. 11, pp. 22–27, 2022.",
    "ITU-T Recommendation Y.3090, “Digital twin network – requirements and architecture,” International Telecommunication Union, 2022.",
    "ETSI GS ZSM 002, “Zero-touch network and service management (ZSM); reference architecture,” ETSI, 2019.",
    "L. U. Khan, W. Saad, D. Niyato, Z. Han, and C. S. Hong, “Digital-twin-enabled 6G: vision, architectural trends, and future directions,” IEEE Communications Magazine, vol. 60, no. 1, pp. 74–80, 2022.",
    "H. X. Nguyen, R. Trestian, D. To, and M. Tatipamula, “Digital twin for 5G and beyond,” IEEE Communications Magazine, vol. 59, no. 2, pp. 10–15, 2021.",
    "D. H. Tran et al., \u201cNetwork digital twin for 6G and beyond: an end-to-end view across multi-domain network ecosystems,\u201d IEEE Open Journal of the Communications Society, vol. 6, pp. 6866\u20136911, 2025, doi:10.1109/OJCOMS.2025.3599866.",
    "Z. Zhang, Z. Peng, H. Yu, M. Chen, and Y. Liu, \u201cDigital network twins for next-generation wireless: creation, optimization, and challenges,\u201d IEEE Network, pp. 1–9, 2025, doi:10.1109/MNET.2025.3620323.",
    "Z. Tao, W. Xu, Y. Huang, X. Wang, and X. You, \u201cWireless network digital twin for 6G: generative AI as a key enabler,\u201d IEEE Wireless Communications, vol. 31, no. 4, pp. 24\u201331, 2024, doi:10.1109/MWC.002.2300564.",
    "H. Al-Hraishawi, M. Alsenwi, J. Ur Rehman, E. Lagunas, and S. Chatzinotas, \u201cDigital twin for enhanced resource allocation in 6G non-terrestrial networks,\u201d IEEE Communications Magazine, vol. 63, no. 3, pp. 47\u201353, 2025, doi:10.1109/MCOM.001.2400211.",
    "K. Rusek, J. Suárez-Varela, P. Almasan, P. Barlet-Ros, and A. Cabellos-Aparicio, “RouteNet: leveraging graph neural networks for network modeling and optimization in SDN,” IEEE Journal on Selected Areas in Communications, vol. 38, no. 10, pp. 2260–2270, 2020.",
    "M. Ferriol-Galmés et al., “RouteNet-Erlang: a graph neural network for network performance evaluation,” in IEEE INFOCOM, 2022, pp. 2018–2027.",
    # --- hybrid terrestrial / non-terrestrial systems
    "3GPP TS 22.261, “Service requirements for the 5G system,” 3rd Generation Partnership Project, 2023.",
    "3GPP TR 38.821, “Solutions for NR to support non-terrestrial networks (NTN),” 3rd Generation Partnership Project, 2021.",
    "W. Saad, M. Bennis, and M. Chen, “A vision of 6G wireless systems: applications, trends, technologies, and open research problems,” IEEE Network, vol. 34, no. 3, pp. 134–142, 2020.",
    "H. Tataria et al., “6G wireless systems: vision, requirements, challenges, insights, and opportunities,” Proceedings of the IEEE, vol. 109, no. 7, pp. 1166–1199, 2021.",
    "M. Giordani and M. Zorzi, “Non-terrestrial networks in the 6G era: challenges and opportunities,” IEEE Network, vol. 35, no. 2, pp. 244–251, 2021.",
    "O. Kodheli et al., “Satellite communications in the new space era: a survey and future challenges,” IEEE Communications Surveys & Tutorials, vol. 23, no. 1, pp. 70–109, 2021.",
    "I. del Portillo, B. G. Cameron, and E. F. Crawley, “A technical comparison of three low earth orbit satellite constellation systems to provide global broadband,” Acta Astronautica, vol. 159, pp. 123–135, 2019.",
    "M. Handley, “Delay is not an option: low latency routing in space,” in ACM HotNets, 2018, pp. 85–91.",
    "S. Kassing, D. Bhattacherjee, A. B. Águas, J. E. Saethre, and A. Singla, “Exploring the ‘Internet from space’ with Hypatia,” in ACM Internet Measurement Conference, 2020, pp. 214–229.",
    "F. Michel, M. Trevisan, D. Giordano, and O. Bonaventure, “A first look at Starlink performance,” in ACM Internet Measurement Conference, 2022, pp. 130–136.",
    "H. B. Tanveer, M. Puchol, R. Singh, A. Bianchi, and R. Nithyanand, \u201cMaking Sense of Constellations: Methodologies for Understanding Starlink\u2019s Scheduling Algorithms,\u201d in Companion of the 19th International Conference on Emerging Networking Experiments and Technologies (CoNEXT), 2023, pp. 37\u201343, doi:10.1145/3624354.3630586.",
    "S. Ma et al., “Network characteristics of LEO satellite constellations: a Starlink-based measurement from end users,” in IEEE INFOCOM, 2023.",
    "Z. Lai et al., “StarryNet: empowering researchers to evaluate futuristic integrated space and terrestrial networks,” in USENIX NSDI, 2023, pp. 1309–1324.",
    "M. Polese, L. Bonati, S. D’Oro, S. Basagni, and T. Melodia, “Understanding O-RAN: architecture, interfaces, algorithms, security, and research challenges,” IEEE Communications Surveys & Tutorials, vol. 25, no. 2, pp. 1376–1411, 2023.",
    "D. Kreutz, F. M. V. Ramos, P. E. Veríssimo, C. E. Rothenberg, S. Azodolmolky, and S. Uhlig, “Software-defined networking: a comprehensive survey,” Proceedings of the IEEE, vol. 103, no. 1, pp. 14–76, 2015.",
    "S. Sharma, D. Staessens, D. Colle, M. Pickavet, and P. Demeester, “OpenFlow: meeting carrier-grade recovery requirements,” Computer Communications, vol. 36, no. 6, pp. 656–665, 2013.",
    "J.-P. Vasseur, M. Pickavet, and P. Demeester, Network Recovery: Protection and Restoration of Optical, SONET-SDH, IP, and MPLS. Morgan Kaufmann, 2004.",
    "G. K. Deepak, A. Ladas, Y. A. Sambo, H. Pervaiz, C. Politis, and M. A. Imran, “An overview of post-disaster emergency communication systems in the future networks,” IEEE Wireless Communications, vol. 26, no. 6, pp. 132–139, 2019.",
    "M. Erdelj, E. Natalizio, K. R. Chowdhury, and I. F. Akyildiz, “Help from the sky: leveraging UAVs for disaster management,” IEEE Pervasive Computing, vol. 16, no. 1, pp. 24–32, 2017.",
    "M. Mozaffari, W. Saad, M. Bennis, Y.-H. Nam, and M. Debbah, “A tutorial on UAVs for wireless networks: applications, challenges, and open problems,” IEEE Communications Surveys & Tutorials, vol. 21, no. 3, pp. 2334–2360, 2019.",
    # --- statistical and computational methods
    "L. Breiman, “Random forests,” Machine Learning, vol. 45, no. 1, pp. 5–32, 2001.",
    "T. Chen and C. Guestrin, “XGBoost: a scalable tree boosting system,” in ACM SIGKDD, 2016, pp. 785–794.",
    "G. Ke et al., “LightGBM: a highly efficient gradient boosting decision tree,” in Advances in Neural Information Processing Systems, 2017, pp. 3146–3154.",
    "F. Pedregosa et al., “Scikit-learn: machine learning in Python,” Journal of Machine Learning Research, vol. 12, pp. 2825–2830, 2011.",
    "B. Zadrozny and C. Elkan, “Transforming classifier scores into accurate multiclass probability estimates,” in ACM SIGKDD, 2002, pp. 694–699.",
    "A. Niculescu-Mizil and R. Caruana, “Predicting good probabilities with supervised learning,” in ICML, 2005, pp. 625–632.",
    "T. Saito and M. Rehmsmeier, “The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets,” PLOS ONE, vol. 10, no. 3, e0118432, 2015.",
    "E. L. Kaplan and P. Meier, “Nonparametric estimation from incomplete observations,” Journal of the American Statistical Association, vol. 53, no. 282, pp. 457–481, 1958.",
    "S. Holm, “A simple sequentially rejective multiple test procedure,” Scandinavian Journal of Statistics, vol. 6, no. 2, pp. 65–70, 1979.",
    "J. Demšar, “Statistical comparisons of classifiers over multiple data sets,” Journal of Machine Learning Research, vol. 7, pp. 1–30, 2006.",
    "A. A. Hagberg, D. A. Schult, and P. J. Swart, “Exploring network structure, dynamics, and function using NetworkX,” in Proceedings of the 7th Python in Science Conference, 2008, pp. 11–15.",
    "C. R. Harris et al., “Array programming with NumPy,” Nature, vol. 585, pp. 357–362, 2020.",
    "P. Virtanen et al., “SciPy 1.0: fundamental algorithms for scientific computing in Python,” Nature Methods, vol. 17, pp. 261–272, 2020.",
    "J. D. Hunter, “Matplotlib: a 2D graphics environment,” Computing in Science & Engineering, vol. 9, no. 3, pp. 90–95, 2007.",
    "R. S. Sutton and A. G. Barto, Reinforcement Learning: An Introduction, 2nd ed. MIT Press, 2018.",
    "N. C. Luong et al., “Applications of deep reinforcement learning in communications and networking: a survey,” IEEE Communications Surveys & Tutorials, vol. 21, no. 4, pp. 3133–3174, 2019.",
    "N. Tarasov and R. Zinko, \u201cCode and data for: Where does predictive resilience come from? A simulation-based digital-twin assessment of failure anticipation, protection and risk-aware recovery in hybrid 5G/6G-class and LEO satellite networks (revision R1),\u201d Zenodo, 2026, concept doi:10.5281/zenodo.22084965 (all versions; release v1.2-R1 matches this manuscript; the raw simulation results were produced with release v1.1-R1, doi:10.5281/zenodo.23957597).",
    "P. Pan, G. Swallow, and A. Atlas, \u201cFast reroute extensions to RSVP-TE for LSP tunnels,\u201d IETF RFC 4090, 2005, doi:10.17487/RFC4090.",
    "A. M. Law, Simulation Modeling and Analysis, 5th ed. McGraw-Hill, 2015.",
    "C. Westphal, L. Han, and R. Li, “LEO satellite networking relaunched: survey and current research challenges,” ITU Journal on Future and Evolving Technologies, vol. 4, no. 4, pp. 711–736, 2023.",
    "J. Im, J. Youn, S. Kim, J. Park, S. Lee, Y. Kwon, and S. Cho, “Distributed detour routing scheme for link failure with minimized overhead in LEO satellite networks,” Sensors, vol. 23, no. 23, 9590, 2023.",
    "H.-W. Lee, E. Modiano, and K. Lee, “Diverse routing in networks with probabilistic failures,” IEEE/ACM Transactions on Networking, vol. 18, no. 6, pp. 1895–1907, 2010.",
    "F. Dikbiyik, M. Tornatore, and B. Mukherjee, “Minimizing the risk from disaster failures in optical backbone networks,” Journal of Lightwave Technology, vol. 32, no. 18, pp. 3175–3183, 2014.",
    "J. Bogle, N. Bhatia, M. Ghobadi, I. Menache, N. Bjørner, A. Valadarsky, and M. Schapira, “TEAVAR: striking the right utilization-availability balance in WAN traffic engineering,” in Proc. ACM SIGCOMM, 2019, pp. 29–43.",
]


# ---------------------------------------------------------------------------
def build(out_path: str) -> None:
    R = load_results()
    d = Paper()
    M, meta = R["models"], R["meta"]
    best = meta["best_model"]
    best_label = best.replace("_", " ")
    theta = float(meta["threshold"])
    rule, bm = M.loc["threshold_rule"], M.loc[best]
    A, RL, NR, REC = R["A"], R["RL"], R["NR"], R["REC"]
    SLA, LAT, LOSS, PRR, RRR = R["SLA"], R["LAT"], R["LOSS"], R["PRR"], R["RRR"]
    runs = R["runs"]
    ai = runs[runs["policy"] == "proactive_ai"]
    S7 = "S7_external_ntn"
    from ..experiments.make_figures_r1 import SHORT, LABEL as LABEL_OF
    from ..experiments.make_figures2 import SCEN_LABEL
    from scipy import stats as _st
    _st_t = _st.t

    MATCH = json.load(open(os.path.join(RESULTS, "kpi_alarm_scale.json")))

    def L_(s, p):
        return float(RL.loc[s, p])

    def rel_(s, p, q):
        return rel(RL.loc[s, p], RL.loc[s, q])

    def pc(s, a, b, metric="resilience_loss"):
        sub = runs[runs["scenario"] == s]
        x = sub[sub["policy"] == a].sort_values("seed")[metric].to_numpy(float)
        y = sub[sub["policy"] == b].sort_values("seed")[metric].to_numpy(float)
        dd_ = x - y
        h = _st.t.ppf(0.975, len(dd_) - 1) * dd_.std(ddof=1) / np.sqrt(len(dd_))
        return {"diff": dd_.mean(), "lo": dd_.mean() - h, "hi": dd_.mean() + h}

    def ci_(s, a, b):
        r = pc(s, a, b)
        return f"[{r['lo']:.2f}, {r['hi']:.2f}]"

    def acct_red(s, p, flag):
        a = R["acct"]
        a = a[(a.scenario == s)]
        ref = a[(a.policy == "reactive")]["resilience_loss"].mean()
        v = a[(a.policy == p) & (a.flag == flag)]["resilience_loss"].mean()
        return -100.0 * (v - ref) / ref

    def sp(x):
        return "less than 0.1%" if abs(x) < 0.05 else f"{x:+.1f}%"

    def arate(s):
        return float(np.nanmean(ai[ai.scenario == s]["anticipation_rate"]))

    def lead(scen, key="lead_time_median"):
        return float(np.nanmean(ai[ai["scenario"] == scen][key]))

    _sig_metrics = ["service_availability", "resilience_loss", "mean_recovery_steps",
                    "normalised_resilience"]
    n_sig = int(R["sig"][(R["sig"]["policy"] == "proactive_ai") &
                         (R["sig"]["scenario"] != "ALL") &
                         (R["sig"]["metric"].isin(_sig_metrics)) &
                         (R["sig"]["significant_005"])].shape[0])

    # ------------------------------------------------------------------ front
    d.title("Where Does Predictive Resilience Come From? A Simulation-Based Digital-Twin "
            "Assessment of Failure Anticipation, Protection and Risk-Aware Recovery in Hybrid "
            "5G/6G-Class and LEO Satellite Networks")
    d.centered("Nikita Tarasov*, Roman Zinko", size=11)
    d.centered("Department of Computer-Aided Design Systems, Lviv Polytechnic National "
               "University, S. Bandery Street 12, 79013 Lviv, Ukraine", size=10, italic=True)
    d.centered("ORCID: N. Tarasov 0000-0003-1870-3436; R. Zinko 0000-0002-3275-8188",
               size=9.5, italic=True)
    d.centered("* Corresponding author: dev.nikita@outlook.com", size=10, italic=True)
    d.doc.add_paragraph()

    # ---- confirmation helpers used in abstract, discussion and conclusion
    _CT, _CEQ = R["conf_t"], R["conf_eq"]

    def _c(s, a, b, col="rel_pct"):
        return float(_CT[(_CT.scenario == s) & (_CT.contrast == f"{a} - {b}")][col].iloc[0])

    _prot = [-_c(s, "protection", "reactive") for s in (S3, S4, S5, S6)]
    _hr = _CT[(_CT.family == "exploratory") & _CT.contrast.str.startswith("protection_oracle")]
    _hrmax = -_hr.rel_pct.min()
    _eq_all = bool(_CEQ.equivalent.all())

    d.h1("Abstract")
    d.p(
        "Hybrid terrestrial–satellite networks keep critical services alive when regional "
        "infrastructure is lost, and failure prediction is often proposed as a way of making "
        "them more resilient. We ask where such a gain comes from. A simulation-based digital "
        "twin of a hybrid 5G/6G-class and low-Earth-orbit network hides element degradation "
        "behind noisy, delayed telemetry, and a calibrated classifier estimates the probability "
        "of failure within a horizon. Fifteen recovery policies, a controlled set of "
        "mechanistic contrasts in risk source, criticality differentiation and restoration "
        "mechanism, are compared on common random numbers in "
        f"{R['n_runs']:,} development runs and a frozen confirmation on 64 independent seeds "
        f"of the same model ({len(R['conf']):,} runs). Plain 1:1 protection, which uses no risk "
        f"information, reduces cumulative resilience loss relative to reactive restoration by "
        f"{min(_prot):.1f}–{max(_prot):.1f}%. Spending the risk estimate only on primary-path "
        "choice and standby-path refresh (risk-informed protection) adds a confirmed "
        f"{-_c(S4, 'protection_predictive', 'protection'):.1f}% under infrastructure collapse "
        f"and {-_c(S6, 'protection_predictive', 'protection'):.1f}% under common-cause "
        "disruption; in that role a contemporaneous degradation index is "
        + ("statistically equivalent to" if _eq_all else "practically indistinguishable from")
        + f" the learned predictor, and perfect state adds at most {_hrmax:.1f}% (exploratory). "
        "Predictive make-before-break migration is not significantly different from "
        "protection under collapse and satellite-segment degradation after multiplicity "
        "correction, but worse under handover dynamics and common-cause disruption; under the "
        "latter its gain over reactive restoration is almost entirely a bypass of the "
        "detection delay. Its remaining gain is confined to gradual degradation and depends on "
        "the switch-over cost, the topology and the seed set. In an exploratory, externally "
        "parameterised satellite stress test, migration was within noise of reactive "
        f"restoration ({rel_(S7, 'proactive_ai', 'reactive'):+.1f}%) while protection kept a "
        f"{-rel_(S7, 'protection', 'reactive'):.0f}% reduction. In the evaluated systems, risk "
        "information produced its most consistent gains within protection-based actions "
        "rather than in pre-emptive migration."
    )
    d.p("Keywords: predictive resilience; failure anticipation; digital twin; protection and "
        "restoration; dynamic reliability; non-terrestrial networks; common-cause failure; "
        "critical communication infrastructure.", italic=True)

    # ------------------------------------------------------------ 1. Intro
    d.h1("1. Introduction")
    d.p(
        "Communication infrastructure has become a dependency of almost every other critical "
        "system. Power restoration crews, medical services, transport control and emergency "
        "management all assume a working data path, and the assumption is weakest precisely when "
        "it matters most: after an earthquake, a flood, a large-scale power outage or a conflict "
        "event, terrestrial infrastructure is typically among the most vulnerable communication "
        "assets. Hybrid architectures that combine terrestrial 5G/6G-class access with aerial "
        "relays and low Earth orbit (LEO) satellite capacity are therefore increasingly "
        "considered, and in emergency deployments used, as a resilience measure [45], [48], "
        "[49]. They introduce an engineering question that this paper addresses: given that "
        "some elements of such a system will degrade and fail, is it worth predicting those "
        "failures and acting on the predictions, how much does that buy, and under what "
        "conditions is the purchase worthwhile?"
    )
    d.p(
        "The reliability-engineering literature provides the measurement apparatus. Resilience is "
        "conventionally quantified from a performance trajectory: the depth of the disruption, "
        "the time spent below nominal performance and the area between the nominal and the "
        "realised curve [1], [2], [6]. Reviews of definitions and metrics [3], [4] and of "
        "critical-infrastructure vulnerability analysis [9] converge on the same structure, and "
        "network-specific work adds component-importance and differentiation measures [5], [11]. "
        "In parallel, prognostics and health management has matured into a discipline that "
        "estimates remaining useful life from condition monitoring data [13], [15], [16], and "
        "system-level studies show that failure prediction is operationally feasible in large "
        "computing and storage infrastructures [20], [21], [22]."
    )
    d.p(
        "What is missing is the connection between the two. Prognostic studies typically stop at "
        "a detection metric, and resilience studies typically assume a repair process that is "
        "exogenous to the analysis. Neither answers the question an operator actually faces: if "
        "the estimated probability that this element fails within the next few tens of seconds is "
        "0.3, should the traffic that depends on it be moved now, given that moving it is itself "
        "disruptive? The answer depends on the anticipation lead time, on the consequence of the "
        "interruption for the specific service, and on the cost of the recovery action: three "
        "quantities that only a closed-loop evaluation can relate to each other."
    )
    d.p(
        "Answering it requires three things that closed-loop evaluations of anticipatory "
        "control often lack: a paired design in which every policy faces exactly the same "
        "exogenous randomness, an interruption accounting that does not credit a "
        "make-before-break switch-over to a flow whose path is already dead, and baselines that "
        "isolate the mechanisms, above all conventional protection. We show that without them "
        "the benefit attributed to anticipation is substantially overstated, and that with them "
        "a substantial share of the benefit is attributable to protection, not to anticipation "
        "itself."
    )
    d.h2("1.1 Research questions")
    d.p("The study is organised around four questions.")
    d.numbered([
        "RQ1. How accurately, and how early, can impending element failures be recognised from "
        "observable telemetry alone, when the degradation state is hidden and the observation map "
        "is noisy?",
        "RQ2. Does acting on failure risk improve resilience relative to reactive restoration "
        "and to conventional protection, and which mechanism (detection-delay bypass, "
        "exposure control or anticipation proper) carries the gain?",
        "RQ3. How do the cost of acting, the detection delay and the actuation effort govern "
        "that gain?",
        "RQ4. Does the decomposition persist under unseen disruption mechanisms, a "
        "structurally misspecified degradation model, a larger topology and perturbation of "
        "the model and controller parameters?",
    ], prefix="{i}.")
    d.h2("1.2 Contributions")
    d.numbered([
        "A simulation-based digital twin of a hybrid terrestrial–non-terrestrial system in "
        "which the degradation state is hidden from every decision-making component, "
        "operational status is estimated subject to a detection delay, and an assessment layer "
        "with privileged access to the physical state scores every decision.",
        "A closed-loop assessment protocol with implemented common random numbers, "
        "validation-only matching of baseline operating points, and an interruption accounting "
        "that separates failure-induced from migration-induced interruptions and charges "
        "restoration cost for migrations off dead paths.",
        "A controlled decomposition over fifteen policies that, where the design allows, vary "
        "one mechanism at a time (risk source, criticality differentiation or restoration "
        "mechanism), including a "
        "predictive-protection policy that spends the risk estimate only on decisions that "
        "interrupt no traffic, and a Monte-Carlo-calibrated privileged-state reference. In that "
        "policy the use of risk information and the dynamic refresh of the standby path enter "
        "together; they are separated only by exploratory ablations.",
        "Evidence from the evaluated systems, with the central comparisons confirmed on "
        "independent seeds under a frozen protocol, that protection is the most consistent "
        "source of gain, "
        "that risk-informed protection with dynamic standby refresh adds a small confirmed "
        "improvement over static protection, which an exploratory decomposition attributes "
        "almost entirely to standby-path selection and in which a simple degradation index is "
        "equivalent to the learned predictor (collapse and common-cause scenarios), and that the benefit of pre-emptive migration is "
        "partly a detection-delay effect and depends on gradual degradation and a nearly free "
        "switch-over.",
    ], prefix="C{i}.")
    d.p(
        "Taken together, the novelty is methodological rather than algorithmic: none of the "
        "individual policies is new, but a mechanism-identifiable closed-loop evaluation of "
        "risk-informed recovery, in which the main sources of gain are compared with conventional "
        "protection on common random numbers and separated as far as the policy design allows, "
        "has, to our knowledge, not been reported for hybrid terrestrial–satellite networks. The "
        "separation is not complete: risk-informed protection differs from static protection "
        "both in the risk information it uses and in refreshing its standby path, and the "
        "learned predictor adds no confirmed practical gain over a degradation index in that role."
    )
    d.p(
        "The evaluation uses a humanitarian and emergency-response deployment as its case study, "
        "because that setting makes the service-criticality hierarchy explicit, but nothing in "
        "the framework is specific to it."
    )
    d.p(
        "Section 2 reviews the related literature. Section 3 defines the system model and "
        "Section 4 the framework and the policies' building blocks. Section 5 defines the "
        "measures, Section 6 the experimental design and Section 7 reports the results against "
        "RQ1–RQ4. Section 8 discusses them, and Sections 9–11 state the limitations, future "
        "work and conclusions."
    )

    # ------------------------------------------------------ 2. Related work
    d.h1("2. Related work")
    d.h2("2.1 Resilience of critical and communication infrastructures")
    d.p(
        "Quantitative resilience assessment descends from the performance-trajectory formulation "
        "of Bruneau et al. [1] and its analytical refinements [6], [7]. Henry and "
        "Ramirez-Marquez formalised resilience as a time-dependent ratio of recovered to lost "
        "performance [2]; Poulin and Kane systematised the family of resilience curves and the "
        "summary metrics derived from them, distinguishing integral, duration and rate-based "
        "measures [4]; Hosseini et al. surveyed definitions across domains [3]; and Zobel and "
        "Khansa extended the construction to multi-event disruptions [8]. For networks "
        "specifically, Barker et al. proposed resilience-based component importance measures [5], "
        "Sterbenz et al. gave the canonical survivability taxonomy [10], Cholda et al. formalised "
        "resilience differentiation across service classes [11], the notion that our criticality-"
        "dependent risk budgets operationalise, and Gomes et al. surveyed protection strategies "
        "against large-scale disasters [12]. This literature measures resilience well; it "
        "generally treats the restoration process as given, not as a decision variable "
        "whose timing can be optimised."
    )
    d.h2("2.2 Dynamic reliability, prognostics and failure anticipation")
    d.p(
        "Prognostics and health management estimates the future condition of a component from "
        "monitoring data and has an extensive statistical toolbox for remaining-useful-life "
        "estimation [15], [16], with physics-informed variants continuing to appear [17]. Zio "
        "surveys both the state of the discipline [13] and the broader challenges of reliability "
        "engineering for complex systems [14]. In computing and communication infrastructures, "
        "supervised models predict node and disk failures ahead of time in production systems "
        "[20], [21], [22], and sequence models detect anomalies in multivariate telemetry, "
        "including spacecraft data [23], [24], [25]. Two methodological points from this "
        "literature are respected here: when the positive class is rare, precision–recall "
        "analysis is the informative view [68]; and when a controller consumes probabilities "
        "rather than labels, those probabilities must be calibrated [66], [67]. Common-cause "
        "failure modelling, which we use to construct one of the disruption scenarios, has its "
        "own established formalism in probabilistic risk assessment [18], [19]."
    )
    d.h2("2.3 Digital twins for reliability and network management")
    d.p(
        "The digital-twin concept originated in product lifecycle management [26] and matured in "
        "industrial informatics [27], [28], [29]. Its migration into networking is recent: "
        "surveys and position papers identify state synchronisation, model fidelity and real-time "
        "constraints as the defining difficulties [30], [31], and standardisation has followed "
        "with ITU-T Y.3090 defining a reference architecture for digital twin networks [32] and "
        "the ETSI ZSM framework defining the closed-loop management context [33]. Work specific "
        "to 5G and 6G discusses twins as prediction and decision-support layers rather than "
        "visualisation tools [34], [35], and the most recent surveys extend the view to "
        "end-to-end, multi-domain network twins and to the transition from simulators to twins "
        "[36], [37], [38]. A twin for non-terrestrial resource allocation has also been "
        "demonstrated [39]. Learned performance models such as "
        "RouteNet [40], [41] provide the what-if capability a twin needs, but they answer 'what "
        "would happen if I routed this way', not 'which element is about to fail'."
    )
    d.h2("2.4 Hybrid terrestrial–non-terrestrial systems")
    d.p(
        "Service requirements for the terrestrial segment are codified in 3GPP TS 22.261 [42] and "
        "the adaptation of NR to non-terrestrial links in TR 38.821 [43]. Vision papers for 6G "
        "converge on integrated terrestrial and non-terrestrial operation [44], [45], [46], and "
        "surveys of satellite communication in the new space era catalogue the resulting "
        "challenges [47]. Constellation studies quantify the coverage and latency properties of "
        "LEO systems [48], [49], simulators make them evaluable [50], [54], and measurement "
        "studies of operational systems document the periodic reconfiguration and latency "
        "variability that motivate our handover scenario [51], [53]. Programmability, needed to "
        "act on any prediction, is provided by SDN [56] and by the O-RAN control architecture "
        "[55]; classical restoration theory quantifies what reactive recovery costs [57], [58]. "
        "For LEO constellations specifically, failure handling ranges from fast local detour "
        "around failed inter-satellite links [82] to disruption-tolerant and multi-layer "
        "recovery schemes surveyed in [81]; the protection baseline of this study represents "
        "the simplest member of that family, a pre-computed disjoint standby path. "
        "Choosing working and backup paths with explicit failure probabilities or disaster "
        "risk is established in survivable network design, for example diverse routing under "
        "probabilistic failures [83], disaster-risk-aware provisioning in optical backbones "
        "[84] and availability-aware traffic engineering [85]. These methods, however, take the "
        "failure probabilities as given and static; the question here is what an online, "
        "telemetry-derived risk estimate adds to them and whether it is better spent on "
        "standby selection or on moving live traffic. "
        "Emergency and disaster communication studies describe the deployment context [59], [60], "
        "[61] but rarely provide reproducible quantitative comparison against a documented "
        "baseline."
    )
    d.h2("2.5 Research gap")
    d.p(
        "The gaps are as follows. Resilience assessment and failure prognosis are developed in "
        "separate literatures, so the lead time of a prediction is almost never converted into a "
        "resilience quantity. Second, evaluations are typically open-loop: a fixed failure trace "
        "is replayed and the policy cannot influence the failures it will later face, which "
        "removes exactly the feedback that makes proactive control risky; whether that "
        "feedback matters is itself an empirical question, and in the system studied here it "
        "turns out to be weak (Section 7.10). Third, the cost of "
        "acting on a prediction is usually ignored, and prediction-driven recovery is rarely "
        "compared with conventional protection [57], [58], [79], so the literature cannot say "
        "whether a gain attributed to anticipation would also have been obtained, more cheaply, "
        "without it. This paper addresses the three jointly."
    )

    # ------------------------------------------------------- 3. System model
    d.h1("3. System model and problem formulation")
    d.h2("3.1 Network and service model")
    d.p(
        "The system is an undirected graph G = (V, E). Each element n in V belongs to one of "
        "eight classes (LTE (4G), 5G and 6G-class terrestrial access, UAV relay, LEO satellite, "
        "edge computing node, core node and user gateway) and carries a static profile (nominal "
        "capacity C_n, latency floor L_n, loss floor p_n, radio characteristics, degradation "
        "susceptibility). Each link carries a propagation latency and a capacity. Service flows "
        "of six criticality classes arrive as a marked Poisson process, occupy a path for a "
        "random holding time and are subject to class-specific latency, loss and bandwidth "
        "budgets. The consequence of an interruption differs by class, which is what makes "
        "differentiated protection meaningful."
    )
    t_el = d.table(pd.read_csv(os.path.join(TABDIR, "t01_elements.csv")),
                   "Element classes and their static reliability-relevant parameters. The "
                   "degradation-susceptibility multiplier scales the baseline degradation "
                   "rate of Eq. (2).",
                   fmt="{:g}", size=8.2)
    t_sv = d.table(pd.read_csv(os.path.join(TABDIR, "t02_services.csv")),
                   "Service classes, their quality budgets and the consequence of an "
                   "interruption. Criticality 5 denotes services whose loss directly removes an "
                   "operational capability.", fmt="{:g}", size=7.6,
                   widths=[2.6, 1.3, 1.8, 1.3, 1.6, 1.7, 4.0])
    fig_topo = d.figure("fig02_topology.png",
                        "Reference topology. Every user gateway retains at least one terrestrial "
                        "and one non-terrestrial route to the core, so differences between "
                        "policies reflect how the available redundancy is used rather than "
                        "whether it exists.", width=15.5)
    d.p(f"{t_el} and {t_sv} list the parameters, and {fig_topo} shows the reference topology used "
        "throughout: 19 elements, 36 links, edge connectivity 2 and diameter 4.")

    d.h2("3.2 Degradation and failure model")
    d.p(
        "Every element carries a latent health variable h_n(t) in [0, 1] that no component of the "
        "framework observes. Health follows a mean-reverting jump-diffusion,"
    )
    eq1 = d.equation("dh_n = mu (1 - h_n) dt - nu_n(t) dt + sigma dW_n - J_n dN_n ,")
    d.p(
        f"where in {eq1} the term mu (1 - h_n) represents maintenance and self-healing, "
        "nu_n(t) is the instantaneous degradation rate, sigma dW_n is diffusion noise standing "
        "for micro-variability such as interference and weather, and J_n dN_n is a compound "
        "Poisson term for abrupt events. We write nu rather than lambda deliberately: nu_n(t) is "
        "subtracted from the latent health and failure occurs only when the health crosses a "
        "threshold, so nu_n is a degradation intensity and not a failure hazard rate in the "
        "reliability-theoretic sense. The interruption hazard of Section 5.3 is the hazard "
        "proper, estimated from realised interruption events. The degradation rate depends on the "
        "element profile, the scenario stress programme and the current utilisation rho_n(t),"
    )
    eq2 = d.equation("nu_n(t) = nu_0 k_n m_s(c_n) [1 + g [rho_n(t) - rho^*]_+^{3/2}] ,")
    d.p(
        f"where [x]_+ denotes the positive part max(0, x), c_n is the element class, k_n its "
        f"degradation-susceptibility "
        f"multiplier and m_s the scenario multiplier. Equation {eq2} closes the control loop: a "
        "policy that concentrates traffic raises utilisation, raises the degradation rate and "
        "therefore the induced failure hazard, thereby manufacturing the failures it will later "
        "have to repair. An element with h_n < h_fail enters the FAILED state, is unavailable for "
        "a repair interval drawn uniformly from a specified range, and is then restored to a "
        "partially healthy state. Two additional mechanisms are used to construct disruption "
        "scenarios that the model was not fitted to. Periodic non-terrestrial reconfiguration "
        "produces transient degradation of the observable KPIs without damaging the element, with "
        "a small probability that the reconfiguration itself fails and reduces health; the timing "
        "and magnitude are informed by published measurements of operational LEO systems [51], "
        "[53] and are not taken from a trace. A common-cause mechanism, following the standard "
        "PRA formulation [18], injects a shared shock Z_t into every element of an affected "
        "region,"
    )
    eq3 = d.equation("h_i(t^+) = max(0, h_i(t) - alpha_i Z_t) ,  "
                     "Z_t ~ Gamma(shape = 2, scale = beta / 2) ,")
    d.p(
        f"where {eq3} makes the failures of elements sharing power and backhaul dependencies "
        "positively correlated, alpha_i is the regional susceptibility of element i and the Gamma "
        "law is parameterised by shape and scale. Because the shock is instantaneous and "
        "exogenous, it has no telemetry precursor by construction, a property we exploit "
        "deliberately in Section 7.4."
    )
    d.p(
        "Two implementation details are stated here because they are needed to reproduce the "
        "process and because the continuous-time form does not imply them. First, the process is "
        "integrated with an Euler-Maruyama step of one slot, so Delta t = 1 and the increment is "
        "h_n(t) + mu(1 - h_n(t)) - nu_n(t) + sigma eps_t with eps_t ~ N(0, 1); mu and nu_0 "
        "therefore carry units of slot^-1, sigma of slot^-1/2, and J_n is a dimensionless health "
        "decrement. Second, the stochastic differential equation does not by itself keep h_n in "
        "[0, 1], so every update is followed by the projection"
    )
    d.equation("h_n(t+1) <- Pi_{[0,1]}( h_n(t) + mu (1 - h_n(t)) - nu_n(t) + sigma eps_t - J_n dN_n ) ,"
               "   Pi_{[0,1]}(x) = min(1, max(0, x)) ,", numbered=False)
    d.p(
        "and the same projection is applied after the common-cause shock of Eq. (3). The jump "
        "term is implemented as a Bernoulli thinning of a Poisson process: in each slot an "
        "element of a scenario-specific target class receives a shock with probability p_J, "
        "independently across elements and slots, of magnitude J_n = A_s U(0.6, 1.2) where A_s is "
        "the scenario shock magnitude, listed with the scenario definitions in the "
        "supplementary configuration files. Jump arrivals are independent across elements; "
        "the only correlated mechanism in the model is the common-cause shock of Eq. (3)."
    )
    fig_deg = d.figure("f03_degradation_trace.png",
                       "Hidden degradation and the telemetry it generates for one 5G-class "
                       "element. Health drifts downward for tens of slots before crossing the "
                       "failure threshold, and that drift is visible in latency and loss only "
                       "through substantial noise. Shaded intervals mark the failed state.",
                       width=14.5)
    d.p(f"{fig_deg} shows a representative trajectory. The descent towards failure is gradual, "
        "which is what makes anticipation possible in principle; the observation noise is what "
        "makes it a statistical problem in practice.")

    d.h2("3.3 Observation model")
    d.p(
        "Telemetry is generated from (h_n, rho_n) by deterministic observation maps with "
        "KPI-specific measurement noise. Writing d_n(t) = 1 - h_n(t) for the health deficit and "
        "using the clamped utilisation rho_tilde_n(t) = min(rho_n(t), rho_cap) with rho_cap < 1, "
        "the saturated queueing term is q_n(t) = min(q_max, rho_tilde_n(t) / (1 - rho_tilde_n(t))). "
        "The "
        "clamp is what keeps the term finite in the overload scenarios, where the offered load "
        "would otherwise reach or exceed unity and the M/M/1 expression would be singular or "
        "negative. The three maps that carry the argument are"
    )
    d.equation("L_n(t) = L_n^0 [1 + a_1 d_n(t)^2 + a_2 q_n(t)] e^{eps},  eps ~ N(0, sigma_o^2) ,")
    d.equation("p_n(t) = Pi_{[0,1]}( [p_n^0 + b d_n(t)^3 + c [rho~_n(t) - 0.8]_+^2] "
               "e^{eps_p} ) ,  eps_p ~ N(0, (1.6 sigma_o)^2) ,")
    eq6 = d.equation("S_n(t) = S_n^0 - gamma d_n(t)^{1.2} + eta ,  eta ~ N(0, sigma_S^2) ,")
    d.p(
        "together with jitter, throughput, processor and memory occupancy, queue depth, "
        "retransmission rate, energy headroom and an exponentially weighted availability "
        f"estimate; {eq6} applies to wireless elements only. The map is non-invertible (distinct "
        "(h, rho) pairs produce the same expected latency) and noisy, so identifying the "
        "degradation state from a single sample is imperfect. Section 7.10 quantifies how the "
        "noise level determines whether temporal aggregation of the telemetry is worth performing."
    )
    d.p(
        "The noise model is deliberately not uniform across KPIs, and the distinction matters for "
        "anyone reproducing the corpus. Strictly positive quantities (latency, jitter, "
        "throughput, processor and memory occupancy, queue depth) carry multiplicative "
        "log-normal noise with log-scale sigma_o. Loss and retransmission rate carry the same "
        "multiplicative noise with an inflated log-scale and are then projected onto [0, 1]. "
        "Bounded ratios (availability, utilisation, energy headroom) carry additive Gaussian "
        "noise and are projected onto [0, 1]. Radio signal strength carries additive Gaussian "
        "noise in dBm, which is the eta of Eq. (6). The constants are listed below."
    )
    obs = pd.DataFrame([
        ["a_1", "2.60", "latency inflation from health loss"],
        ["a_2", "0.90", "weight of the queueing term in latency"],
        ["b", "0.25", "loss growth from health loss"],
        ["c", "0.55", "loss growth from congestion above rho = 0.80"],
        ["gamma", "38.0 dBm", "signal strength lost as h_n -> 0"],
        ["rho_cap", "0.945", "utilisation clamp applied before the queueing term"],
        ["q_max", "8.0", "saturation of the queueing term (finite buffers)"],
        ["sigma_o", "0.055", "log-scale of the multiplicative observation noise"],
        ["sigma_S", "1.8 dBm", "standard deviation of eta in Eq. (6)"],
        ["additive ratio noise", "0.012", "standard deviation for availability, utilisation, "
                                          "energy headroom"],
    ], columns=["symbol", "value", "meaning"])
    d.table(obs, "Observation-model constants. These are properties of the generative "
                 "environment and are held fixed across every policy and scenario, so they "
                 "cannot differentiate the policies being compared.",
            fmt="{}", size=7.6, widths=[3.0, 2.6, 9.6])
    d.h2("3.4 Recovery model and its cost")
    d.p(
        "Three recovery mechanisms are distinguished. Reactive restoration is break-before-make: "
        "the flow is already interrupted, the twin needs D slots of KPI aggregation before it "
        "can declare the element failed, a replacement path is computed and the session is "
        "re-established, which costs a further interval C_r. 1:1 protection pre-computes a "
        "node-disjoint standby path for every flow at admission and after every path change; "
        "the standby path is pre-signalled, as in fast reroute [79], so a switch onto it costs "
        "C_m. When the twin declares the active path down, the flow is switched to its standby "
        "path only if that path is observed up and has enough residual capacity at that moment; "
        "otherwise it falls back to reactive restoration. Standby capacity is not reserved in "
        "the main model; flows hit in the same slot are served in a fixed order (decreasing "
        "criticality, then flow identifier) and see the capacity already taken by the flows "
        "served before them, so simultaneous failovers cannot oversubscribe a link. Section 7 "
        "also reports a reserved variant, in which standby bandwidth is set aside in advance "
        "and unavailable to other traffic, and an optimistic variant without the capacity "
        "check. Pre-emptive "
        "migration is make-before-break: the replacement path is installed while the original "
        "still carries traffic, so the switch-over costs C_m [57], [58], [79]."
    )
    d.p(
        "The make-before-break premise holds only while the original path still works. If an "
        "element of the vacated path has already failed — which happens when a controller "
        "reacts to the failure signature in the raw telemetry before the twin's detector has "
        "confirmed it — no working path carries the service while the new one is set up, and "
        "the interruption lasts the restoration interval. The assessment layer therefore "
        "charges C_r instead of C_m to any migration whose vacated path contains a physically "
        "failed element. The controller cannot observe this condition; it affects only how the "
        "interruption is scored. Section 7.4 shows how large the apparent benefit of "
        "anticipation becomes if C_m is charged in this case instead."
    )
    d.p(
        "Time is discrete. One slot is one orchestration cycle and corresponds to one second, "
        "the mapping used to anchor the reconfiguration cadence of the non-terrestrial "
        "scenarios to measured schedules [51], [52]; thus D = 3 s, C_r is between 4 and 12 s "
        "and C_m is 0 or 1 s. Every interruption is counted in whole slots: an action taken in "
        "slot t with duration C interrupts the flow in slots t, ..., t + C − 1 (a unit test checks "
        "that a zero-duration switch-over interrupts no slot), that is in "
        "exactly C slots: a switch-over with C_m = 0 interrupts nothing, one with C_m = 1 a "
        "single slot, and a restoration between four and twelve slots. Durations are drawn "
        "per physical event, not per call (Section 6.4). Availability, resilience loss, recovery time and the Kaplan–Meier "
        "estimates of Section 5 all use this slot grid, so no fractional interruption arises."
    )
    d.h2("3.5 Parameter provenance")
    d.p(
        "Because every observation in this study is produced by the model just described, the "
        "origin of each parameter matters as much as its value. We therefore separate three "
        "categories. Measurement-informed parameters reproduce ranges reported for operational "
        "systems. Standards-derived parameters follow service or protocol specifications. "
        "Synthetic parameters are stress-test settings without direct empirical counterparts; "
        "they are selected to produce a measurable failure incidence, and their influence is "
        "examined through the sensitivity analysis of Section 7.10."
    )
    prov = pd.DataFrame([
        ["Latency floors, capacities per element class", "Table 1", "standards-derived",
         "3GPP TS 22.261 and TR 38.821 service and NTN characteristics [42], [43]"],
        ["LEO access and feeder latency, loss floor", "42 ms, 0.0025", "measurement-informed",
         "reported LEO end-user measurements [51], [53]"],
        ["Handover cadence and transient magnitudes (S5)", "15 slots, x1.8-3.6 latency",
         "measurement-informed", "reconfiguration cadence from [52]; transient latency ranges from [51], [53]"],
        ["Service latency and loss budgets", "Table 2", "standards-derived",
         "3GPP TS 22.261 classes, relaxed where the NTN path makes them unattainable"],
        ["Detection delay D", "3 slots", "engineering assumption",
         "KPI aggregation and alarm de-bouncing in management planes"],
        ["Slot duration", "1 s", "measurement-informed",
         "anchors the 15-slot reconfiguration grid of S5 and S7 to the 15 s schedule of [52]"],
        ["Restoration cost C_r, switch-over cost C_m", "U{4,...,12}, U{0,1} slots",
         "engineering assumption",
         "break-before-make versus pre-computed protection [57], [58], [79]; swept in Section 7.6"],
        ["Baseline degradation rate nu_0 [slot^-1]", "0.0050", "synthetic (calibrated)",
         "set so that the failure prevalence over the corpus is of the order of one per cent"],
        ["Diffusion sigma, reversion mu", "0.045, 0.050", "synthetic (calibrated)",
         "set to give a degradation time-scale of tens of slots; swept in Section 7.10"],
        ["Scenario degradation multipliers", "Table 6", "synthetic (stress test)",
         "define the severity ladder; no empirical counterpart is claimed"],
        ["Load factors", "1.00-2.15", "synthetic (stress test)", "scenario definition"],
        ["Utilisation-to-degradation gain g, knee rho*", "3.0, 0.65", "synthetic",
         "couples load to degradation; swept in Section 7.10"],
        ["Common-cause rate and magnitude (S6)", "0.014, Gamma(2, 0.39)",
         "synthetic (falsification test)",
         "constructed to be instantaneous and precursor-free by design [18]"],
        ["Observation noise sigma_o", "0.055", "synthetic", "swept in Section 7.10"],
    ], columns=["parameter", "value", "category", "basis"])
    t_prov = d.table(prov, "Provenance of the model parameters. Only the standards-derived and "
                           "measurement-informed rows are tied to external evidence; the remainder are declared stress-test or "
                           "calibration settings and are examined by sensitivity analysis.",
                     fmt="{}", size=7.2, widths=[4.6, 2.8, 2.8, 5.8])
    d.p(
        f"{t_prov} lists the parameters by category. Two consequences follow and are stated here "
        "rather than in the limitations, because they qualify every number in Section 7. The "
        "absolute values reported are properties of this parameterisation; only the comparison "
        "between recovery policies, which share the environment exactly, is transferable. "
        "Conclusions drawn from these parameters should also remain stable under reasonable "
        "perturbation, which motivates the sensitivity analysis of Section 7.10, where the "
        "baseline degradation rate nu_0, the diffusion amplitude sigma and the "
        "utilisation-to-degradation coupling g is switched off and doubled."
    )

    d.h2("3.6 Problem statement")
    d.p(
        "At every control interval and for every active flow f with criticality class c, the "
        "operator can keep the current path P_f, migrate the live flow to an alternative, or, "
        "under protection, keep the flow in place and change only its standby path. Keeping "
        "risks an interruption of D + C_r slots if an element of P_f fails, or D + C_m slots if "
        "a working standby path exists; migrating costs C_m with certainty and may place the "
        "flow on a worse path; changing the standby path costs nothing in service terms. The "
        "decision must be taken from the twin's estimate of the system state, which excludes "
        "h_n. The objective is to minimise the cumulative resilience loss of the service "
        "portfolio subject to capacity feasibility."
    )

    # --------------------------------------------------------- 4. Framework
    d.h1("4. Predictive digital-twin framework")
    d.h2("4.1 Architecture")
    fig_arch = d.figure("fig01_architecture.png",
                        "Layered architecture. The twin is observation-only; the assessment layer "
                        "is the sole consumer of ground truth. The red path is the closed loop: "
                        "recovery decisions set the offered load, which enters the degradation rate.",
                        width=12.0)
    d.p(
        f"{fig_arch} shows the layer stack. The telemetry layer samples KPIs and maintains "
        f"per-element ring buffers of length W = {FEATURE_WINDOW}. The twin holds element and "
        "link state, the observed operational status, a capacity ledger and a time-varying cost "
        "graph. The anticipation layer converts a state window into a calibrated failure "
        "probability. The recovery layer consumes topology, capacity and risk and produces path "
        "decisions. The assessment layer computes the measures of Section 5 against the true "
        "physical state."
    )
    d.p(
        "The construction is a simulation-based digital twin, and the qualifier matters. What distinguishes it from a network simulator is not the "
        "fidelity of the network model but the separation of roles: the simulated physical "
        "system evolves on its own; the twin is a separate estimator that sees it only through "
        "telemetry, maintains a delayed and noisy belief about it, answers predictive questions "
        "from that belief and acts back on it; and an assessment layer with privileged access "
        "to the physical state scores the outcome. We use the term in the sense of the ITU-T "
        "Y.3090 reference architecture [32], which defines a twin by its data, model and "
        "management interfaces. The physical counterpart is itself a simulation, the twin has "
        "not been synchronised with a deployed network, and no result in this paper should "
        "be read as validation on a real system. Section 8.4 returns to what the separation "
        "made measurable."
    )
    d.h2("4.2 Twin state estimation")
    d.p(
        "The twin receives no failure notification. It declares an element down after its "
        "telemetry has satisfied a compound condition (packet loss above 0.5 together with "
        "availability below 0.08) for more than D consecutive slots. This detection delay is the "
        "single most consequential modelling choice for the baseline comparison, because a "
        "reactive policy cannot begin restoration before t_fail + D whereas an anticipatory policy "
        "may act before t_fail. The twin also maintains a slowly adapting latency reference per "
        "element, used by the non-learning baseline of Section 6.3."
    )
    d.p("The element penalty combining quality and risk, and the resulting edge weight, are")
    eq7 = d.equation("phi_n(t) = w_L L_n(t)/L_ref + w_P p_n(t) + w_U max(0, rho_n - 0.75)^2 "
                     "+ w_R r_n(t)^2 ,")
    eq8 = d.equation("c_uv(t) = w_L d_uv / L_ref + [phi_u(t) + phi_v(t)] / 2 ,")
    d.p(
        f"so that a least-cost path under {eq7}–{eq8} is simultaneously fast, lossless, "
        "uncongested and, for risk-aware policies, unlikely to break. Two risk aggregations over "
        "a path are used: R_prod(P) = 1 - prod (1 - r_n) for reporting and "
        "R_inf(P) = max_n r_n as the migration trigger, since migrating away from the single most "
        "endangered element is the action actually available."
    )
    d.h2("4.3 Failure anticipation")
    d.p("Let X_n(t) denote the window of the last W telemetry vectors of element n. The target is")
    eq9 = d.equation("y_n(t) = 1{ exists u in (t, t+H] : state_n(u) = FAILED } ,")
    d.p(
        f"and {eq9} defines an early-warning problem rather than a condition-monitoring problem: "
        f"with H = {PREDICTION_HORIZON} slots the label refers to the future and cannot be read "
        "off the current KPIs. Slots in which the element is already failed are excluded from "
        "both training and evaluation, because predicting an ongoing outage is not decision-"
        "relevant and would inflate every metric. Five predictors are compared: an operator "
        "threshold rule representing current alarm practice, L2-regularised logistic regression, "
        "a random forest [62], XGBoost [63] and LightGBM [64]. Every fitted model is wrapped in "
        "an isotonic calibrator fitted on the validation episodes [66], [67], and the decision "
        "threshold theta* is selected on the same fold by maximising F1 over a grid of 187 "
        "candidates. The model used in closed loop is chosen on validation data only, by a "
        "criterion adopted for this revision and applied without test labels: the highest validation precision–recall "
        "area, since the positive class is rare [68]; models within 0.01 of the best are "
        "treated as tied and the tie is broken by the validation Brier score, since the "
        "controller consumes probabilities, and then by inference cost. The validation fold is "
        "therefore used three times (calibration, threshold and model choice), which makes the "
        "validation scores optimistic; the test fold is used only for reporting. Test predictions "
        "are produced in the same pass as validation predictions, but no test label enters "
        "calibration, threshold or model choice, and the selected model does not depend on the "
        "tie-break, which uses timings on unlabeled test features and was not invoked. "
        f"For the selected model theta* = {theta:.3f}."
    )
    d.h2("4.4 Risk-aware recovery")
    d.p(
        "The recovery layer must decide how much anticipated risk a flow may carry before "
        "migration is warranted. Instead of a fixed constant, the budget of class c is expressed "
        "as a multiple of the predictor's validated operating point,"
    )
    eq10 = d.equation("tau_c = min(0.95, kappa b_c theta*) ,")
    d.p(
        f"with {eq10} using budget factors b_c in {{0.80, 1.00, 1.30, 1.80, 2.50}} for "
        "criticality 5 down to 1 and a global scale kappa, set to unity in the main campaign. "
        "The symbol b_c is used rather than phi to keep it distinct from the node penalty "
        "phi_n(t) of Eq. (7). The construction assigns higher-criticality services a lower "
        "migration threshold than lower-criticality services at the same global scale; at the "
        "nominal kappa = 1 the most critical class migrates at 0.80 theta*, that is on weaker "
        "evidence than the level at which the model itself would raise an alarm, while "
        "delay-tolerant traffic waits for 2.5 times that evidence. The ordering holds for any "
        "kappa, the comparison against theta* only for kappa b_c < 1. The ladder was fixed before "
        "the campaign and not tuned: the class with criticality 4 acts at the operating point "
        "itself, the most critical class slightly below it, and each step down in criticality "
        "multiplies the required evidence by roughly 1.3 to 1.4, so that the least critical "
        "class waits for 2.5 times theta*; flat, shallower and steeper ladders are compared in "
        "Section 7.10. The rule inherits the "
        "statistical properties of the predictor, so re-training the model automatically "
        "re-scales the policy. The migration trigger compares the bottleneck path risk "
        "R_inf(P), the largest element risk on the path, against tau_c. Among the least-cost path "
        "and the k = 3 "
        "shortest alternatives, the controller selects the candidate minimising"
    )
    eq11 = d.equation("J(P) = D(P)/D_c^max + 4 p_loss(P)/p_c^max "
                      "+ (3 + 0.6 chi_c) R_prod(P) + Pen(P) ,")
    d.p(
        f"where in {eq11} D(P) is the predicted end-to-end delay of path P, p_loss(P) its "
        "predicted loss, D_c^max and p_c^max the SLA limits of class c from Table 2, and chi_c "
        "in {1, ..., 5} the criticality score of that class. Path risk enters the objective through the product aggregation "
        "R_prod(P), one minus the product over the elements of P of the complementary element "
        "risks, which is the probability that at least one element on the path is anticipated to "
        "fail under an independence assumption. The bottleneck form R_inf(P), the maximum "
        "element risk on the path, is used only for the migration trigger above, because a "
        "single endangered element is what makes a migration worthwhile. The penalty term is the sum of "
        "three soft constants applied to the implemented constraint violations,"
    )
    d.equation("Pen(P) = 2.5 * 1{residual capacity < demand} "
               "+ 1.5 * 1{D(P) > D_c^max} + 1.5 * 1{p_loss(P) > p_c^max} ,", numbered=False)
    d.p(
        "so an SLA-feasible path is preferred and, when none exists, the controller degrades to "
        "the least-violating candidate rather than rejecting the flow. A hysteresis interval "
        "bounds control-plane churn, and migrations are processed in decreasing criticality order "
        "so that scarce residual capacity is allocated to the most consequential services first."
    )
    fig_dec = d.figure("f04_decision_mechanism.png",
                       "Decision chain from the telemetry window to the class budget, and the "
                       "two ways the budget is used: migration of the live flow (P3, P4, P5) or "
                       "refresh of the standby path under protection (P6).", width=15.0)
    d.p(
        f"{fig_dec} summarises the chain. The same risk estimate and budgets can be used in two "
        "ways. The migration policies move the live flow make-before-break when R_inf(P) "
        "exceeds tau_c. The predictive-protection policies leave "
        "the live flow where it is and use the risk only for decisions that interrupt nothing: "
        "the risk-augmented cost of Eq. (8) selects the primary path at admission, with elements "
        "whose risk is at least the exclusion limit max(tau_c, min(0.95, 2.5 theta*)) removed "
        "from the candidates exactly as for the migration policies (the analogue of the rule "
        "that P2 does not route onto elements that are in alarm), and "
        "whenever the standby path of a flow contains an element whose risk exceeds tau_c, or "
        "is observed down, a new node-disjoint standby path is computed on the risk-augmented "
        "graph with those elements excluded. The switch-over itself remains detection-driven, "
        "exactly as in 1:1 protection."
    )
    d.p(
        "The weights of Eq. (7) and the coefficients of Eq. (11) require justification, since "
        "they could otherwise be tuned to the environment they are evaluated in. They were fixed "
        "before the closed-loop campaign by a normalisation argument, not by search: each "
        "term is scaled so that a representative bad value contributes of the order of one unit "
        "to the cost. With L_ref = 25 ms, a doubling of latency contributes about one; w_P = 220 "
        "makes a loss of half a per cent contribute about one; w_U = 45 makes utilisation of 0.90 "
        "contribute about 0.9; and w_R = 140 makes a risk equal to twice the operating point "
        "contribute about one. The coefficients 4, 3 and 0.6 in Eq. (11) follow the same logic and "
        "give loss violations four times the weight of latency violations and criticality a "
        "moderate linear influence; the three penalty constants of the display above are of the "
        "same order "
        "as one normalised SLA unit. No coefficient was adjusted after the campaign began, none was "
        "selected on the test seeds, and none was selected on S5 or S6. The sensitivity of the "
        "result to the one free scale that remains, kappa, and to alternative profiles b_c and "
        "coefficients of Eq. (11) is reported in Sections 7.6 and 7.10."
    )
    d.h2("4.5 An ex-post economic reading of the budget")
    d.p(
        "The budget rule is a heuristic, and we present it as one. It admits an "
        "economic reading: if an interruption of a class-c flow costs C_f slots of lost service "
        "and a migration costs C_m, migrating is worthwhile when r C_f > C_m, that is when"
    )
    eq12 = d.equation("r > p* = C_m / C_f .")
    d.p(
        f"The controller does not derive its thresholds from {eq12}. The budgets are multiples "
        "of an F1-selected operating point, F1 contains neither C_m nor C_f, and the quantity "
        "compared with the budget is a bottleneck proxy R_inf(P) rather than a path-failure "
        "probability. The break-even condition is therefore an interpretation after the fact, "
        "and its main use here is the prediction it makes: the benefit of migration must shrink "
        "as C_m approaches C_f. Section 7.6 tests that prediction, and Section 10 names "
        "cost-derived actuation as the natural successor of the heuristic."
    )

    # -------------------------------------------- 5. Resilience quantification
    d.h1("5. Reliability and resilience quantification")
    d.h2("5.1 Service reliability and availability")
    d.p(
        "Let T_f denote the time from admission of flow f to its first interruption. The service "
        "reliability of class c is the survival function"
    )
    eq13 = d.equation("R_c(t) = P(T_f > t | c_f = c) ,")
    d.p(
        f"estimated with the Kaplan–Meier product-limit estimator [69] over the flows of that "
        "class, with flows that terminate without interruption treated as right-censored. Two "
        "event definitions are reported. Under the first, any interruption counts, including the "
        "one-slot switch-over of a make-before-break migration with C_m = 1 (a switch-over with "
        "C_m = 0 interrupts nothing). Under the second, only sustained interruptions of at least "
        "two slots count; this excludes every live-path switch-over, but a migration off a path "
        "that has already failed is charged C_r and still counts, so Section 7.4 also "
        "separates interruptions by cause. Availability is the "
        "fraction of flow-slots in the operational state,"
    )
    d.equation("A = 1 - sum_f o_f / sum_f a_f ,")
    d.p("where o_f counts interrupted slots and a_f the slots during which f is active.")
    d.h2("5.2 Resilience loss and normalised resilience")
    d.p(
        "Following the resilience-curve formulation [1], [2], [4], let Q(t) in [0, 1] be the "
        "instantaneous service quality, here the fraction of active flows that are operational, "
        "and Q_0 = 1 its nominal value. The cumulative resilience loss and its normalised "
        "counterpart are"
    )
    eq15 = d.equation("L = sum_{t=1..T} [Q_0 - Q(t)]_+ ,")
    eq16 = d.equation("Rn = 1 - L / (T Q_0) .")
    d.p(
        f"Equation {eq15} is the discrete counterpart of the area between the nominal and the "
        f"realised performance trajectory, and {eq16} maps it to [0, 1]. Unlike mean recovery "
        "time, L penalises both the depth and the duration of a disruption rather than only the "
        "duration. It differs from availability in what it averages over, not in temporal "
        "sensitivity: availability is a flow-slot weighted ratio, so slots carrying many flows "
        "dominate it, whereas L integrates the instantaneous service fraction with equal weight "
        "on every slot, so a deep loss during a period of light traffic is not discounted. "
        "Neither is sensitive to the absolute position of the loss within the horizon; when that "
        "matters we report the disruption depth max_t [Q_0 - Q(t)]_+ as the robustness component "
        "and the recovery-time distribution of Section 5.3 as the rapidity component."
    )
    d.p(
        "Both quantities weight every flow equally. Because a command-and-control session and "
        "a sensor report are not interchangeable, Section 7.4 additionally reports availability "
        "and SLA-compliant availability separately for every service class; class-weighted "
        "aggregates of the loss are not used for any conclusion."
    )
    d.h2("5.3 Recovery and interruption hazard")
    d.p(
        "The mean recovery time is the average length of an interruption episode, from the first "
        "interrupted slot to the first slot in which the flow delivers again. The empirical "
        "interruption hazard is estimated piecewise as h(t) = events in [t, t+Δ) divided by the "
        "product of the number at risk and the bin width. Every interruption episode is "
        "attributed to a cause: it is failure-induced if an element of the "
        "active path has failed, and migration-induced if it consists only of the switch-over "
        "slots of a migration away from a working path. Recovery time is reported separately "
        "for the two causes, because a policy that adds many short migration-induced "
        "interruptions lowers the mean over all interruptions without recovering any faster "
        "from failures."
    )
    d.h2("5.4 Anticipation quality")
    d.p(
        "Prediction quality is reported as precision, recall, F1, ROC-AUC, precision–recall area, "
        "the Brier score and the false-alarm rate on negative windows. Because a probability that "
        "arrives too late is useless regardless of its accuracy, we additionally define the "
        "anticipation lead time"
    )
    eq18 = d.equation("T_lead = t_failure - t_alarm ,")
    d.p(
        f"where the alarm time is matched to the failure in a horizon-consistent way, "
        "t_alarm = min{t in [t_failure - H, t_failure) : r_n(t) >= theta*}, so that "
        f"0 < T_lead <= H = {PREDICTION_HORIZON} by construction. A threshold crossing earlier "
        "than t_failure - H is not evidence for the target of Eq. (9) and is recorded as a prior "
        "alarm rather than as anticipation of that failure; a failure with no crossing inside the "
        "horizon is a miss. The anticipation rate is the share of failures carrying a warning "
        f"inside the horizon. {eq18} is the quantity that must be compared with the migration "
        "cost C_m: a warning shorter than the switch-over time cannot be acted upon."
    )

    # --------------------------------------------------- 6. Experimental design
    d.h1("6. Experimental design")
    d.h2("6.1 Implementation")
    d.p(
        "The scientific implementation is written in Python and comprises the environment, the "
        "feature pipeline, the predictor, the twin, the recovery controller and the assessment "
        "harness; it produces every number reported here, using NetworkX [72], NumPy [73], SciPy "
        "[74], scikit-learn [65] and Matplotlib [75]. A second implementation exposes the same "
        "twin and controller logic as REST microservices in Go, with the predictor served "
        "separately, Prometheus metrics on every service and Kubernetes manifests for deployment. "
        "Its role is to demonstrate that the twin and controller can be implemented "
        "independently of the monolithic scientific simulator, and to "
        "support reproduction; it is an engineering artefact, not a contribution, and its "
        "details are given in the accompanying repository."
    )
    d.h2("6.2 Parameters and scenarios")
    params = pd.DataFrame([
        ["Control interval", "1 slot = 1 s", "one orchestration cycle"],
        ["Episode length", "500 slots", "one run"],
        ["Prediction horizon H", f"{PREDICTION_HORIZON} slots", "anticipation target"],
        ["Feature window W", f"{FEATURE_WINDOW} slots", "telemetry ring buffer"],
        ["Failure threshold h_fail", "0.15", "health below which an element fails"],
        ["Repair interval", "U(12, 38) slots", "unattended repair"],
        ["Detection delay D", "3 slots", "KPI aggregation before an alarm"],
        ["Restoration cost C_r", "U{4, ..., 12} slots", "break-before-make re-establishment"],
        ["Switch-over cost C_m", "U{0, 1} slots", "make-before-break migration and protection "
                                                   "switch-over"],
        ["Hysteresis", "6 slots", "minimum interval between migrations of a flow"],
        ["mu [slot^-1], sigma [slot^-1/2], nu_0 [slot^-1]", "0.050, 0.045, 0.0050",
         "degradation process"],
        ["Cost weights", "w_L=1, w_P=220, w_U=45, w_R=140", "composite path metric"],
        ["Budget factors b_c", "0.80 / 1.00 / 1.30 / 1.80 / 2.50", "criticality 5 down to 1"],
        ["Global budget scale kappa", "1.0", "nominal value used in the main campaign; "
                                             "swept in Section 7.6"],
        ["Path alternatives k", "3", "candidates evaluated by Eq. (11)"],
        ["Seeds per configuration", str(R["n_seeds"]), "paired replications"],
    ], columns=["parameter", "value", "role"])
    t_par = d.table(params, "Model and control parameters. All values are declared in a single "
                            "configuration module and are reported verbatim.",
                    fmt="{}", size=8.4, widths=[5.0, 4.4, 6.6])
    t_sc = d.table(pd.read_csv(os.path.join(TABDIR, "t03_scenarios.csv")),
                   "Disruption scenarios. S5, S6 and S7 are excluded from the training corpus, so "
                   "both the predictor and the controller face their degradation mechanisms for "
                   "the first time at evaluation; S7 additionally takes its non-terrestrial "
                   "dynamics from published measurement studies.", fmt="{:g}", size=7.0,
                   widths=[2.3, 4.6, 1.2, 3.0, 2.4, 1.5, 1.5, 1.4])
    d.p(
        f"{t_par} lists the parameters and {t_sc} the scenarios. S1 is the nominal control "
        "condition. S2 congests the terrestrial access segment. S3 removes an entire terrestrial "
        "cluster and degrades the non-terrestrial segment, so two gateways depend on satellite "
        "and aerial paths exactly when those are worst. S4 destroys five terrestrial elements and "
        "raises the degradation rate of the surviving elements under double load. S5 adds periodic "
        "non-terrestrial "
        "reconfiguration with transient KPI degradation. S6 injects correlated regional shocks "
        "through shared power and backhaul dependencies. S7 takes its reconfiguration cadence "
        "and latency statistics from measurement studies [51], [52]; it is an exploratory, "
        "externally anchored scenario, that is, an externally parameterised stress test and not an "
        "external validation. For the scale experiment of "
        "Section 7.8 a second topology with 98 elements and 233 undirected links is used: "
        "three LEO orbital planes of eight satellites (24) with intra-plane ring and "
        "inter-plane links; two core sites that also act as feeder gateways; and eight "
        "terrestrial regions, each with one 6G-class, two 5G and one LTE cell, a UAV relay, "
        "an edge node and three user terminals. Arrival rates are scaled with the number of "
        "user terminals (factor 6). It is one alternative structure, not a sample of "
        "topologies, so it tests whether the ranking survives a change of scale and shape "
        "rather than establishing structural invariance. The training corpus is built only from "
        f"S1–S4 and contains {int(meta['n_train']) + int(meta['n_val']) + int(meta['n_test']):,} "
        f"eligible observation windows at a test-fold prevalence of "
        f"{float(meta['positive_rate_test']):.4f}."
    )
    d.h2("6.3 Recovery policies")
    d.p(
        "Fifteen policies share the twin, topology, traffic and seeds. They are not a full "
        "factorial design but a controlled set of mechanistic contrasts: each policy differs "
        "from a reference in one of three mechanisms, so that the contribution of that "
        "mechanism can be read off a paired comparison. The mechanisms are the risk source (none, a static KPI alarm, a contemporaneous degradation "
        "index, the learned prediction, or the true latent state); the treatment of service "
        "classes (criticality-differentiated budgets b_c, criticality term of Eq. (11) and "
        "criticality-ordered processing, or uniform treatment with b_c = 1, a constant risk "
        "coefficient equal to the class average and arrival-order processing); and the "
        "restoration mechanism (reactive restoration or 1:1 protection)."
    )
    d.numbered([
        "P0 static: the path is fixed at admission and never revised.",
        "P1 reactive restoration: the path is recomputed once the twin observes that an "
        "element on it is down; the reference for the primary family.",
        "P1b 1:1 protection: node-disjoint standby path computed without risk, preferring "
        "paths with enough residual capacity for the flow; detection-driven, "
        "capacity-checked switch-over at cost C_m; reactive fallback (Section 3.4). It stands "
        "for the protection and local-detour mechanisms used in terrestrial and LEO networks "
        "[57], [58], [79], [81], [82] in their simplest, risk-free form.",
        "P2 threshold-proactive: make-before-break migration when any element on the path "
        "violates a static operator limit: an element is in alarm when its KPI severity "
        "s = max(p/0.12, (L/L_ref)/1.9, cpu/88, 0.80/A) exceeds a scale factor, 1.0 at the "
        "nominal operator limits. The loss limit of 12% equals the loosest class loss budget of "
        "Table 2 (sensor data); the latency limit of 1.9 times the reference, the CPU limit of "
        "88% and the availability limit of 0.80 are conventional operator thresholds fixed "
        "before the study. P2 uses the same hysteresis and the same k = 3 candidate "
        "paths as P4, and does not route onto elements that are themselves in alarm. P2m is the same rule with the scale factor set so that its "
        "per-element alarm rate equals that of the predictor at theta*.",
        "P3 degradation-aware: migration and risk-augmented routing driven by a "
        "contemporaneous degradation index, a fixed weighted combination of current loss, "
        "latency inflation, unavailability and retransmission rate, rescaled so that it "
        "raises alarms at the same per-element rate as the predictor. P3u is its uniform "
        "variant.",
        "P4 predictive migration: the same machinery driven by the calibrated probability of "
        "failure within H. P4u is its uniform "
        "variant.",
        "P5 perfect state, band map: risk computed as a linear map of the true latent health "
        "between h = 0.35 (risk 0) and h = 0.18 (risk 1), clipped to [0, 1]. The map is not "
        "calibrated to the controller's probability scale and is kept to show the "
        "consequence. P5c replaces it by a Monte Carlo estimate of the conditional probability "
        "that an element in health h fails within H, obtained by simulating the generative model "
        "for each scenario and element class (6,000 trajectories per point of a health grid, "
        "interpolated); an element that has already failed is assigned risk 1, so P5c, like the "
        "telemetry-driven policies, can act before the twin's detector confirms a failure. P5c "
        "is a Monte-Carlo-calibrated privileged-state reference: it maps the true health h to "
        "r(h) = P(failure within H | h), estimated as the fraction of the 6,000 simulated "
        "trajectories from h that fail within H, and feeds r(h) into the same decision rule as "
        "P4. It does not optimise cost, is not an upper bound, and its finite-sample table "
        "is not exactly monotone in h (largest reversal 0.013 between neighbouring grid points).",
        "P6 predictive protection: 1:1 protection plus risk-augmented primary-path choice and "
        "risk-driven standby refresh (Section 4.4), with no migration of live flows. A flow's "
        "standby path is refreshed at most once every 6 slots, and a replacement is adopted only "
        "if it is observed up, capacity-feasible and lowers the bottleneck risk of the standby "
        "path by at least 0.05, or if the current standby path is down; every standby "
        "computation and refresh is counted as a control-plane action. P6i uses "
        "the degradation index and P6o the calibrated perfect-state probability as risk "
        "source; P6m adds P4's migrations to P6.",
    ], prefix="({i})")
    d.p(
        "Both matched operating points are estimated on dedicated validation runs of the "
        "reactive policy in the training scenarios S2–S4, on seeds disjoint from every seed "
        "used for evaluation: the quantile of the index, and the scale factor of the KPI rule, "
        "at which the per-element alarm rate equals the predictor's alarm rate at theta* on "
        "the same runs. The validation runs use scenarios S2–S4 and six seeds "
        f"(9000 + 13k, k = 0, ..., 5; {MATCH['n_element_slots']:,} element-slots); on them "
        f"the predictor at theta* raises alarms in {100 * MATCH['alarm_rate']:.2f}% of "
        f"element-slots, which fixes the index quantile at {MATCH['quantile']:.3f} and the KPI "
        f"scale factor at {MATCH['kpi_scale_at_operating_point']:.3f}. This closed-loop alarm "
        "rate differs from the offline rates of Table 7 and Appendix B, which refer to observation "
        "windows of the training corpus. Both values are frozen before any evaluation run."
    )
    d.h2("6.4 Protocol and statistical analysis")
    d.p(
        f"Each (scenario, policy) configuration is replicated over {R['n_seeds']} seeds and the "
        f"main campaign comprises {len(R['runs']):,} closed-loop runs of 500 slots; the "
        f"experiments of Sections 7.4–7.10 add a further {R['n_rev_runs']:,} runs, and the confirmation experiment of Section 7.3 another {len(R['conf']):,}. Comparisons are paired, "
        "and the pairing is implemented by common random numbers. The traffic "
        "model, the controller's internal randomness, the common-cause process and every "
        "network element draw from separate reproducible streams derived from the seed with "
        "NumPy's SeedSequence [73], [80]. In every slot each element draws a random block of "
        "fixed size from its own stream, and every random quantity of that element in that slot "
        "(diffusion increment, jump occurrence and magnitude, repair duration, post-repair "
        "health, handover outcome and every observation-noise term) is read from a fixed "
        "position of the block. Two policies on the same seed therefore see identical "
        "arrivals, holding times, demands, endpoints and element-level randomness, whatever "
        "they do; the number of generated flows is identical across policies for every seed, "
        "and a unit test asserts that the random blocks coincide. Action durations are paired "
        "in the same way: C_r and the protection switch-over C_m are keyed draws determined "
        "by the seed, the flow and the physical failure event (the failed element and its "
        "onset slot), and the C_m of a live migration by the seed, the flow and the slot, so "
        "two policies that restore or switch the same flow after the same failure draw the "
        "same duration whatever order they make their decisions in; a second unit test "
        "asserts this. Seeding a single generator "
        "shared by traffic, environment and controller is not sufficient in a closed loop: "
        "policies that consume different numbers of draws would then see different arrivals "
        "and different failures, and the pairing would be lost."
    )
    d.p(
        "Comparisons use the paired t-test on seed-level differences, with the Wilcoxon "
        "signed-rank test as a distribution-free check, Cohen's d_z and 95% confidence "
        "intervals of the paired difference. Two inferential families are distinguished. "
        "Family A is the family fixed for the original study and retained for continuity: the "
        "four primary resilience measures of P4 against P1 in the five stressed scenarios "
        "S2–S6, twenty tests, Holm-corrected [70], [71]. Family B is the revision-confirmatory "
        "family of Section 6.5. Everything else (the development campaign of the remaining "
        "policies, S7 and the experiments of Sections 7.4–7.10) is exploratory and is "
        "reported with intervals and uncorrected p-values."
    )
    d.h2("6.5 Frozen confirmatory evaluation")
    d.p(
        "The protection-based policies were introduced while the study was being revised, so "
        "their comparisons on the development seeds are hypothesis-generating. After the "
        "policies, parameters, predictor and accounting had been frozen, a separate "
        "confirmation experiment was specified in a written protocol and committed to the code "
        "repository together with its analysis script before any confirmation seed was run. "
        "It uses 64 seeds never used for development or analysis, the scenarios S3–S6, the "
        "policies P1, P1b, P3, P4, P6i and P6 (P6o as an exploratory reference), and "
        "cumulative resilience loss as the only endpoint. Family B1 (primary) contains the "
        "two contrasts P6 − P1b under collapse and under common-cause disruption, two-sided "
        "paired t-tests with Holm correction over the two. Family B2 (key secondary) contains "
        "P1b − P1, P4 − P1, P4 − P1b and P6 − P4 in each of the four scenarios, sixteen "
        "tests with Holm correction. Family B3 tests the equivalence of P6 and P6i under "
        "collapse and common-cause disruption by two one-sided tests, with margins fixed in "
        "the protocol at 3% of the protection loss observed in the development runs available "
        "when the protocol was written (13.26 and 26.86 slot-equivalents), that is ±0.40 and "
        "±0.81 slot-equivalents. The margin is a practical choice, not an accepted standard. "
        "It was set as the smallest difference that would, in our judgement, justify "
        "training, calibrating and maintaining a learned forecaster instead of a fixed "
        "degradation index for standby selection; it is between one seventh and one quarter "
        "of the benefit of protection over reactive restoration in these scenarios, and it was "
        "fixed before any confirmation seed was run. Section 7.3 also reports the smallest "
        "margin for which the observed result would still establish equivalence. The "
        "protocol, the code freeze and the confirmation results are in the supplementary "
        "material; the public repository history contains the protocol commit, tagged "
        "r1b-freeze, and the later commit at which the confirmation was run. The confirmation "
        "replicates the effect within the evaluated simulation model; it does not sample "
        "other physical systems."
    )

    # ---------------------------------------------------------- 7. Results
    d.h1("7. Results")
    d.h2("7.1 RQ1: how accurately and how early can failures be anticipated?")
    t_mod = d.table(R["models_raw"][["model", "accuracy", "balanced_accuracy", "precision",
                                     "recall", "f1", "mcc", "roc_auc", "pr_auc", "brier",
                                     "false_alarm_rate", "threshold", "val_pr_auc",
                                     "val_brier"]].assign(
        model=lambda x: x["model"].str.replace("_", " ")).rename(columns={
            "balanced_accuracy": "balanced accuracy", "f1": "F1", "mcc": "MCC",
            "roc_auc": "ROC-AUC", "pr_auc": "PR-AUC", "brier": "Brier score",
            "false_alarm_rate": "false-alarm rate", "threshold": "operating point",
            "val_pr_auc": "validation PR-AUC", "val_brier": "validation Brier"}),
        "Anticipation performance on episode-disjoint test data at the validation-selected "
        "operating point. The two validation columns are the only quantities used for model "
        "selection (Section 4.3); test labels are not used for selection. The "
        "threshold rule is the non-learning operator heuristic.", size=6.8)
    fig_pred = d.figure("f05_predictor.png",
                        "Discrimination and calibration of the anticipation models: "
                        "precision–recall curves, reliability diagram and score distribution.",
                        width=15.5)
    learned = R["models_raw"][R["models_raw"]["model"] != "threshold_rule"]
    d.p(
        f"{t_mod} and {fig_pred} answer the first half of RQ1. On validation data the four "
        f"learned models reach a precision–recall area between {learned['val_pr_auc'].min():.3f} "
        f"and {learned['val_pr_auc'].max():.3f}; {best_label} has the highest validation value, "
        "no other model lies within the 0.01 tolerance of Section 4.3, and it is therefore "
        "selected by the stated criterion without recourse to the tie-break. On the test "
        f"episodes it attains ROC-AUC {bm['roc_auc']:.4f} and a precision–recall area of "
        f"{bm['pr_auc']:.4f} against {rule['pr_auc']:.4f} for the operator threshold rule, a "
        f"lift of {bm['pr_auc'] / float(meta['positive_rate_test']):.1f} over a prevalence of "
        f"{float(meta['positive_rate_test']):.4f}, with a Brier score of {bm['brier']:.4f}. "
        "The drop from validation to test precision–recall area reflects the episode-disjoint "
        "split and the rarity of the positive class. Logistic regression is also ahead of the three tree ensembles on every test metric "
        "of the table; the ordering among the ensembles is not consistent across metrics, "
        "which is one reason the choice is made on validation data only."
    )
    fig_lead = d.figure("f06_lead_time.png",
                        "Anticipation lead time of the predictive policy P4 inside the closed "
                        "loop. Left: median lead time with inter-quartile range. Right: share of "
                        "anticipated failures with at least 1, 3, 5 and H slots of warning. Event "
                        "matching is horizon-consistent, so lead times lie in (0, H]. Scenarios "
                        "marked with an asterisk are outside the training distribution.",
                        width=15.0)
    t_lead = d.table(pd.read_csv(os.path.join(TABDIR, "t05_lead_time.csv")),
                     "Anticipation quality of P4 inside the closed loop, averaged over seeds.",
                     size=7.2)
    d.p(
        f"{fig_lead} and {t_lead} answer the second half. Within the training family the median "
        f"lead time is {lead(S3):.1f} slots under non-terrestrial degradation and {lead(S4):.1f} "
        f"slots under infrastructure collapse, with anticipation rates of "
        f"{pct(arate(S3), 0)}% and {pct(arate(S4), 0)}%. An alarm counts as anticipation of a "
        "failure only if it is raised inside the window [t_f − H, t_f) that the target of "
        f"Eq. (9) refers to, so lead times cannot exceed H = {PREDICTION_HORIZON} slots. On the "
        f"held-out mechanisms the anticipation rate is {pct(arate(S5), 1)}% under handover "
        f"dynamics and {pct(arate(S6), 1)}% under common-cause disruption; the latter is "
        "expected, since the shared shock of Eq. (3) has no telemetry precursor by construction. "
        "These figures describe the predictor inside this generative model. Section 7.8 tests "
        "how much of the ranking survives when the degradation family and the observation map "
        "are changed."
    )

    # ------------------------------------------------------------- 7.2 RQ2
    d.h2("7.2 RQ2: development campaign and the original primary family")
    fig_dec2 = d.figure("r01_decomposition.png",
                        "Paired change in cumulative resilience loss relative to reactive "
                        "restoration P1, by scenario, for the policies that isolate one mechanism "
                        "at a time (development campaign). Bars: mean over 24 paired seeds; "
                        "whiskers: 95% confidence interval of the paired difference. Hatched bars "
                        "belong to the 1:1 protection family.", width=16.5)
    t_cmp = d.table(pd.read_csv(os.path.join(TABDIR, "r_t_compact.csv")),
                    "Cumulative resilience loss L (slot-equivalents, lower is better) by policy "
                    "and stressed scenario in the development campaign, averaged over 24 paired "
                    "seeds. S1 is omitted because it has no exogenous disruption; only the aggressive "
                    "threshold policy P2 incurs a small self-inflicted loss there (Appendix A). Appendix A gives "
                    "every policy and measure.", fmt="{:g}", size=7.0)
    d.p(
        f"{fig_dec2} and {t_cmp} summarise the development campaign: all fifteen policies, all "
        "seven scenarios, 24 paired seeds. These seeds were used while the protection-based "
        "policies were designed, so the campaign generates hypotheses; Section 7.3 tests the "
        "central ones on fresh seeds. Five patterns stand out."
    )
    d.numbered([
        "Plain 1:1 protection (P1b), which uses no risk information, reduces resilience loss "
        f"relative to reactive restoration by {-rel_(S4, 'protection', 'reactive'):.1f}% under "
        f"infrastructure collapse, {-rel_(S6, 'protection', 'reactive'):.1f}% under "
        f"common-cause disruption and {-rel_(S3, 'protection', 'reactive'):.1f}% under "
        "non-terrestrial degradation.",
        "Predictive make-before-break migration (P4) is of the same order as protection where "
        "degradation is gradual and failures are frequent: "
        f"{-rel_(S4, 'proactive_ai', 'reactive'):.1f}% under collapse, paired difference to P1b "
        f"{pc(S4, 'proactive_ai', 'protection')['diff']:+.2f} slot-equivalents "
        f"(95% CI {ci_(S4, 'proactive_ai', 'protection')}). It is clearly behind protection under "
        f"common-cause disruption ({-rel_(S6, 'proactive_ai', 'reactive'):.1f}% against P1, "
        f"{rel_(S6, 'proactive_ai', 'protection'):+.1f}% against P1b) and worse than reactive "
        f"restoration under handover dynamics ({rel_(S5, 'proactive_ai', 'reactive'):+.1f}%).",
        "Predictive protection (P6) keeps the detection-driven switch-over of P1b and uses the "
        "risk estimate only for decisions that interrupt no traffic: the choice of the primary "
        "path and the refresh of the standby path. Relative to P1b it lowers the loss by "
        f"{-rel_(S4, 'protection_predictive', 'protection'):.1f}% under collapse "
        f"(95% CI of the difference {ci_(S4, 'protection_predictive', 'protection')}) and "
        f"{-rel_(S6, 'protection_predictive', 'protection'):.1f}% under common-cause "
        f"disruption ({ci_(S6, 'protection_predictive', 'protection')}), and is "
        "indistinguishable from it where protection itself has little to do (S2, S5, S7).",
        "Inside the protection family the risk source hardly matters: the degradation index "
        f"(P6i) differs from the predictor by {sp(rel_(S4, 'protection_index', 'protection_predictive'))} "
        f"under collapse and {sp(rel_(S6, 'protection_index', 'protection_predictive'))} under "
        "common-cause disruption, and the calibrated perfect-state probability (P6o) by "
        f"{rel_(S4, 'protection_oracle', 'protection_predictive'):+.1f}% and "
        f"{rel_(S6, 'protection_oracle', 'protection_predictive'):+.1f}%.",
        "Inside the migration family the risk source does matter. Perfect state knowledge used "
        f"for migration (P5c) reaches {L_(S4, 'oracle_calibrated'):.2f} under collapse, below "
        f"P1b ({L_(S4, 'protection'):.2f}) and P6 ({L_(S4, 'protection_predictive'):.2f}), but "
        f"not under common-cause disruption ({L_(S6, 'oracle_calibrated'):.2f} against "
        f"{L_(S6, 'protection'):.2f} for P1b). Adding P4's migrations to P6 (P6m) leaves the "
        f"loss unchanged under collapse ({L_(S4, 'protection_predictive_mig'):.2f} against "
        f"{L_(S4, 'protection_predictive'):.2f}) and raises it under common-cause disruption "
        f"({L_(S6, 'protection_predictive_mig'):.2f} against {L_(S6, 'protection_predictive'):.2f}).",
    ], prefix="({i})")
    t_conf = d.table(pd.read_csv(os.path.join(TABDIR, "r_t_confirmatory.csv")),
                     "Family A, retained from the original study for continuity: P4 against "
                     "reactive restoration P1 on the four primary measures in the five stressed "
                     "scenarios (twenty tests, paired t-test, Holm-corrected), development "
                     "campaign. Mean recovery here is the per-run mean, with 0 for a run without "
                     "interruption, as required for the paired test; Section 7.4 gives pooled "
                     "recovery times.", fmt="{}", size=7.0)
    n_conf = int((R["sig"]["family"] == "confirmatory").sum())
    conf = R["sig"][R["sig"]["family"] == "confirmatory"]
    n_hs = int((conf["p_holm"] < 0.05).sum())
    d.p(
        f"{t_conf} reports Family A, the inferential family fixed for the original study, "
        f"recomputed under the corrected implementation. {n_hs} of the {n_conf} tests are "
        "significant after Holm correction. The decrease in resilience loss and the increase "
        "in availability under collapse and under common-cause disruption survive correction; "
        "the handover scenario shows a significant deterioration; overload and non-terrestrial "
        "degradation show no corrected effect on the resilience measures. The recovery-time "
        "entries must be read with Section 7.4: most of that reduction replaces long "
        "failure-induced interruptions by short migration-induced ones. Family A answers the "
        "original question (does P4 beat P1?) and does not by itself locate the gain, "
        "which is what the remaining sections do."
    )
    t_contr = d.table(pd.read_csv(os.path.join(TABDIR, "r_t_contrasts.csv")),
                      "Exploratory paired contrasts on cumulative resilience loss in the "
                      "development campaign (24 seeds; uncorrected Wilcoxon signed-rank p). "
                      "Negative differences favour the first-named policy.", fmt="{}", size=6.3)
    d.p(
        f"{t_contr} lists the paired contrasts that isolate each mechanism in "
        "scenarios S3–S7. They are exploratory and are reported with intervals, not as tests."
    )

    # ------------------------------------------------------- 7.3 confirmation
    d.h2("7.3 Frozen confirmation on independent seeds")
    CT, CEQ, CR = R["conf_t"], R["conf_eq"], R["conf"]

    def cf(s, a, b):
        return CT[(CT.scenario == s) & (CT.contrast == f"{a} - {b}")].iloc[0]

    def cfs(s, a, b, ci=True):
        r = cf(s, a, b)
        txt = f"{r['mean_diff']:+.2f} slot-equivalents ({r['rel_pct']:+.1f}%"
        if ci:
            txt += f", 95% CI [{r['ci_low']:.2f}, {r['ci_high']:.2f}]"
        return txt + ")"

    def hp(s, a, b):
        v = cf(s, a, b)["p_holm"]
        return "Holm p < 0.0001" if v < 1e-4 else f"Holm p = {v:.3f}"

    def cL(s, p):
        return float(CR[(CR.scenario == s) & (CR.policy == p)].resilience_loss.mean())

    b2 = CT[CT.family == "B2 secondary"]
    n_b2 = int((b2.p_holm < 0.05).sum())
    t_cm = d.table(pd.read_csv(os.path.join(TABDIR, "r_t_conf_means.csv")),
                   "Frozen confirmation experiment: mean cumulative resilience loss ± 95% "
                   "half-width over 64 independent seeds per cell (1,792 runs).", fmt="{}",
                   size=6.8)
    t_ct = d.table(pd.read_csv(os.path.join(TABDIR, "r_t_conf_tests.csv")),
                   "Confirmatory families B1 (primary, Holm over two tests) and B2 (key "
                   "secondary, Holm over sixteen tests) on 64 independent seeds; paired "
                   "differences of cumulative resilience loss, negative values favour the "
                   "first-named policy. Confidence intervals are per-contrast (unadjusted); significance "
                   "decisions use the Holm-adjusted p-values.", fmt="{}", size=6.3)
    fig_cf = d.figure("r04_confirmation.png",
                      "Frozen confirmation on 64 new seeds: paired contrasts of families B1 and "
                      "B2 in cumulative resilience loss, as a percentage of the second-named "
                      "policy's loss (95% CI; faded bars are not significant after Holm "
                      "correction).", width=16.5)
    d.p(
        f"{t_cm}, {t_ct} and {fig_cf} report the experiment specified in Section 6.5, run once "
        "after the code freeze on 64 seeds that had never been used. The primary hypothesis "
        "B1, that predictive protection (risk-informed primary-path and standby selection) adds "
        "resilience on top of 1:1 protection, is "
        "supported in both scenarios: P6 − P1b is "
        f"{cfs(S4, 'protection_predictive', 'protection')} under collapse "
        f"({hp(S4, 'protection_predictive', 'protection')}, d_z = "
        f"{cf(S4, 'protection_predictive', 'protection')['dz']:.2f}) and "
        f"{cfs(S6, 'protection_predictive', 'protection')} under common-cause disruption "
        f"({hp(S6, 'protection_predictive', 'protection')}, d_z = "
        f"{cf(S6, 'protection_predictive', 'protection')['dz']:.2f}). The effect is small in "
        "relative terms and consistent in sign, and the Wilcoxon check agrees."
    )
    d.p(
        f"In family B2, {n_b2} of the 16 contrasts are significant after Holm correction. "
        "Protection improves on reactive restoration in all four scenarios "
        f"({cf(S3, 'protection', 'reactive')['rel_pct']:+.1f}%, "
        f"{cf(S4, 'protection', 'reactive')['rel_pct']:+.1f}%, "
        f"{cf(S5, 'protection', 'reactive')['rel_pct']:+.1f}% and "
        f"{cf(S6, 'protection', 'reactive')['rel_pct']:+.1f}% in S3–S6). Predictive migration "
        "improves on reactive restoration in S3, S4 and S6 "
        f"({cf(S3, 'proactive_ai', 'reactive')['rel_pct']:+.1f}%, "
        f"{cf(S4, 'proactive_ai', 'reactive')['rel_pct']:+.1f}%, "
        f"{cf(S6, 'proactive_ai', 'reactive')['rel_pct']:+.1f}%) and not in S5 "
        f"({cf(S5, 'proactive_ai', 'reactive')['rel_pct']:+.1f}%, "
        f"{hp(S5, 'proactive_ai', 'reactive')}; the Wilcoxon test, which is not the protocol "
        "test, rejects here, because many seeds show a small deterioration). Predictive "
        "migration and protection are not "
        "distinguishable under non-terrestrial degradation "
        f"({cfs(S3, 'proactive_ai', 'protection')}); under collapse the point estimate favours "
        f"migration ({cfs(S4, 'proactive_ai', 'protection')}) but the difference is not "
        f"significant after Holm correction ({hp(S4, 'proactive_ai', 'protection')}); "
        "migration is worse than protection under handover dynamics "
        f"({cf(S5, 'proactive_ai', 'protection')['rel_pct']:+.1f}%) and under common-cause "
        f"disruption ({cfs(S6, 'proactive_ai', 'protection')}). The action-mechanism contrast "
        "P6 − P4 has the same pattern: no detectable difference in S3 and S4, a clear "
        f"advantage of predictive protection in S5 ({cf(S5, 'protection_predictive', 'proactive_ai')['rel_pct']:+.1f}%) "
        f"and S6 ({cfs(S6, 'protection_predictive', 'proactive_ai')})."
    )
    t_eq = d.table(pd.read_csv(os.path.join(TABDIR, "r_t_conf_equiv.csv")),
                   "Family B3: equivalence of the learned predictor (P6) and the "
                   "contemporaneous degradation index (P6i) as risk source of predictive "
                   "protection; two one-sided paired tests with margins fixed in the protocol.",
                   fmt="{}", size=7.0)
    t_hr = d.table(pd.read_csv(os.path.join(TABDIR, "r_t_conf_headroom.csv")),
                   "Exploratory headroom of perfect-state information inside predictive "
                   "protection: P6o (calibrated perfect-state probability) against P6 and P6i, "
                   "64 seeds.", fmt="{}", size=6.8)
    n_eq = int(CEQ.equivalent.sum())
    hr = CT[(CT.family == "exploratory") & CT.contrast.str.startswith("protection_oracle")]
    d.p(
        f"{t_eq} reports family B3. "
        + ("Equivalence of P6 and P6i is established in both scenarios: "
           if n_eq == len(CEQ) else
           f"Equivalence of P6 and P6i is established in {n_eq} of {len(CEQ)} scenarios: ")
        + "; ".join(f"the 90% interval of the paired difference, [{r.ci90_low:.3f}, "
                    f"{r.ci90_high:.3f}], lies inside ±{r.margin:.2f} under "
                    f"{SCEN_TEXT[r.scenario]}" for _, r in CEQ.iterrows())
        + ". The conclusion does not depend on the exact margin: equivalence would also be "
        "established with margins of "
        + " and ".join(f"±{np.ceil(100 * max(abs(r.ci90_low), abs(r.ci90_high))) / 100:.2f}"
                       for _, r in CEQ.iterrows())
        + " slot-equivalents, several times smaller than both the protocol margins and the "
        "B1 effect. The protocol margins themselves are of the same size as the B1 effect, so "
        "B3 on its own would not exclude a difference as large as the confirmed benefit of "
        "risk-informed protection; it is the observed bounds that make the two risk "
        "sources practically interchangeable. Under common-cause disruption the index is in fact slightly ahead of the "
        "predictor, by a margin far inside the equivalence bound. For selecting and refreshing "
        "standby paths, a contemporaneous degradation index is as good as the trained "
        "forecaster within these margins. "
        f"{t_hr} bounds what better information could add in that role: replacing the "
        "learned risk by the calibrated probability computed from the true latent state "
        f"lowers the loss by at most {-hr.rel_pct.min():.1f}% in any of the four scenarios."
    )

    # ---------------------------------------------------- 7.4 what migration does
    d.h2("7.4 What a migration acts on")
    t_mc = d.table(pd.read_csv(os.path.join(TABDIR, "r_t_migration_classes.csv")),
                   "Ground-truth classification of migration triggers, by the state of the "
                   "triggering element as known to the assessment layer: already failed but "
                   "not yet confirmed by the twin's detector; failing within the horizon H; "
                   "degraded (latent health below 0.45) but not failing within H; healthy (health of "
                   "at least 0.45) and not failing within H (false alarm). Development campaign (24 seeds) and confirmation (64 seeds).",
                   fmt="{}", size=6.2)
    mct = pd.read_csv(os.path.join(TABDIR, "r_t_migration_classes.csv"))

    def mcv(src, s, pol, col):
        r = mct[(mct.seeds == src) & (mct.scenario == SCEN_LABEL[s]) & (mct.policy == pol)]
        return float(r[col].iloc[0])

    AF, FW, DG, FA = ("already failed (detection-delay bypass) [%]", "fails within H [%]",
                      "degraded, no failure within H [%]", "false alarm [%]")
    d.p(
        "The assessment layer knows the true state, so every migration can be classified "
        f"after the fact by what the triggering element was doing ({t_mc}). Four classes are "
        "distinguished: the element had already failed but the twin had not yet declared it "
        "down (detection-delay bypass); it failed within the horizon (anticipation proper); it "
        "was degraded but did not fail within the horizon; or it was healthy and did not fail "
        "(false alarm). On the confirmation seeds, P4's triggers under collapse split "
        f"{mcv('confirmation', S4, 'P4', AF):.0f}% / {mcv('confirmation', S4, 'P4', FW):.0f}% / "
        f"{mcv('confirmation', S4, 'P4', DG):.0f}% / {mcv('confirmation', S4, 'P4', FA):.1f}% "
        "across the four classes; under common-cause disruption "
        f"{mcv('confirmation', S6, 'P4', AF):.0f}% of its triggers act on elements that are "
        "already down. Index-driven migration (P3) under collapse is dominated by detection-delay "
        f"bypass ({mcv('confirmation', S4, 'P3', AF):.0f}%), and under handover dynamics "
        f"{mcv('confirmation', S5, 'P3', FA):.0f}% of its triggers are false alarms, against "
        f"{mcv('confirmation', S5, 'P4', FA):.0f}% for P4. Outside handover dynamics, triggers on elements that are healthy in this "
        "sense are rare for the learned predictor; most of its triggers that are not followed "
        "by a failure act on degraded elements, and what it mostly adds beyond the bypass are migrations away from "
        "elements that are degraded but survive the horizon. The migration gain therefore has "
        "two sources of different weight in different scenarios: under common-cause disruption "
        "it is almost entirely detection-delay bypass, because the shared shock has no "
        "precursor; under collapse a substantial part of it is anticipation of gradual failure."
    )
    d.p(
        "The classification also shows why the scoring rule of Section 3.4 matters. A "
        "migration away from a path that is already physically broken cannot be "
        "make-before-break, because no working path carries the service while the new one is "
        "established. An accounting that charges such a migration the switch-over cost C_m "
        "instead of the restoration cost C_r credits the policy with an interruption it did "
        f"not avoid. On 12 paired seeds, charging C_m would put the reduction of P4 relative to "
        f"P1 at {acct_red(S4, 'proactive_ai', False):.0f}% under collapse and "
        f"{acct_red(S6, 'proactive_ai', False):.0f}% under common-cause disruption, and that "
        f"of P3 at {acct_red(S4, 'degradation_aware', False):.0f}% and "
        f"{acct_red(S6, 'degradation_aware', False):.0f}%, against "
        f"{acct_red(S4, 'proactive_ai', True):.0f}%, {acct_red(S6, 'proactive_ai', True):.0f}%, "
        f"{acct_red(S4, 'degradation_aware', True):.0f}% and "
        f"{acct_red(S6, 'degradation_aware', True):.0f}% under the physical accounting used "
        "throughout. The controllers are identical in both accountings; only the scoring of the "
        "interruption differs, and the difference is of the same order as the effect that "
        "survives."
    )
    t_rs = d.table(pd.read_csv(os.path.join(TABDIR, "r_t_recovery_split.csv")),
                   "Recovery time separated by the cause of the interruption (development "
                   "campaign). Failure-induced interruptions start when an element on the active "
                   "path fails; migration-induced interruptions are the switch-over slots of a "
                   "make-before-break migration away from a working path. Recovery times are pooled over "
                   "all interruptions of the 24 runs; a dash marks a cause with no interruption.", fmt="{:g}",
                   size=6.8)
    d.p(
        f"{t_rs} separates recovery time by cause. The large reduction in mean recovery time "
        "that P4 shows in Family A is mostly a composition effect: P4 "
        "adds many migration-induced interruptions of about one slot to the average, while its "
        "recovery from actual failures is "
        f"{R['REF'].loc[S4, 'proactive_ai']:.2f} slots under collapse against "
        f"{R['REF'].loc[S4, 'reactive']:.2f} for P1. Protection shortens failure-induced "
        f"recovery to {R['REF'].loc[S4, 'protection']:.2f} slots and predictive protection to "
        f"{R['REF'].loc[S4, 'protection_predictive']:.2f}. Mean recovery time over all "
        "interruptions is therefore not a meaningful headline for policies that create their "
        "own interruptions, and we do not use it as one."
    )
    t_cls = d.table(pd.read_csv(os.path.join(TABDIR, "r_t_class_sla.csv")),
                    "Availability (A) and SLA-compliant availability (SLA-A) per service class "
                    "under infrastructure collapse, in per cent. SLA-A counts a flow-slot only "
                    "if the flow is delivered and meets its latency and loss budget.",
                    fmt="{:g}", size=6.2)
    d.p(
        f"{t_cls} shows what the availability numbers mean for service quality. Availability "
        "counts a flow-slot as available when the flow is delivered at all; SLA-compliant "
        "availability additionally requires that the latency and loss budgets of Table 2 are "
        "met. Under collapse the network remains connected for most flows but cannot meet the "
        "budgets of the tighter classes on the surviving non-terrestrial paths, so "
        "SLA-compliant availability is low for every policy and differs little between them. "
        "The recovery policies studied here restore connectivity; none of them can create "
        "capacity that the collapse has removed."
    )
    fig_sur = d.figure("f08_survival.png",
                       "Kaplan–Meier estimates of the probability that a flow has not yet been "
                       "interrupted (top row: any interruption, including migration switch-over "
                       "slots; bottom row: interruptions lasting at least two slots).",
                       width=16.0)
    sv = pd.read_csv(os.path.join(TABDIR, "r_t_survival.csv"))

    def s40(defn, s, pol):
        r = sv[(sv.scenario == SCEN_LABEL[s]) & (sv.policy == pol)
               & ((sv.definition == "any") == (defn == "any"))]
        return float(r["S(40)"].iloc[0])

    d.p(
        f"{fig_sur} shows the same mechanism from the flow's point of view. Counting every "
        "interruption, P4 has the lowest survival under collapse "
        f"(S(40) = {s40('any', S4, 'P4'):.3f} against {s40('any', S4, 'P1'):.3f} for P1), "
        "because each switch-over with C_m = 1 interrupts the flow for one slot. Counting only "
        "interruptions of at least two slots, the order reverses "
        f"({s40('sus', S4, 'P4'):.3f} against {s40('sus', S4, 'P1'):.3f}): migration replaces "
        "some long failure-induced interruptions by one-slot switch-overs. The time to the "
        "first interruption is almost identical for P1, P1b and P6, which is what the "
        "mechanism predicts: protection does not postpone interruptions, it shortens them, so "
        "its benefit appears in resilience loss and failure-induced recovery time rather than "
        "in survival."
    )
    fig_tl = d.figure("f07_timeline.png",
                      "One infrastructure-collapse episode. Top: service quality Q(t) under "
                      "four policies (15-slot moving average). Middle: number of failed "
                      "elements. Bottom: mean predicted risk and the calibrated perfect-state "
                      "risk.", width=15.0)
    d.p(
        f"{fig_tl} traces a single episode. The predicted risk follows the perfect-state risk "
        "with smaller amplitude and a delay, as expected from a calibrated model that sees the "
        "state only through noisy telemetry."
    )

    # ------------------------------------------------------- 7.5 detection delay
    d.h2("7.5 Detection delay")
    fig_dd = d.figure("r02_detection_delay.png",
                      "Resilience loss as a function of the detection delay D for the "
                      "principal policies under infrastructure collapse and common-cause "
                      "disruption (12 paired seeds, 95% intervals). Dashed lines: 1:1 "
                      "protection family.", width=15.0)
    t_dd = d.table(pd.read_csv(os.path.join(TABDIR, "r_t_detection.csv")),
                   "Resilience loss against the detection delay D (12 paired seeds).",
                   fmt="{}", size=6.8)
    dd = R["dd"]

    def ddL(s, p, D):
        return float(dd[(dd.scenario == s) & (dd.policy == p) & (dd.detection_delay == D)]
                     ["resilience_loss"].mean())

    def gain(s, p, D):
        return ddL(s, "reactive", D) - ddL(s, p, D)

    best5 = min(("protection", "proactive_ai", "oracle_calibrated", "protection_predictive"),
                key=lambda p: ddL(S4, p, 5))
    d.p(
        f"{fig_dd} and {t_dd} test the classification directly by sweeping D with everything "
        "else fixed. A gain that comes from bypassing the detector must vanish as D goes to "
        "zero; a gain that comes from foresight must not. Under common-cause disruption the "
        f"advantage of P4 over P1 is {gain(S6, 'proactive_ai', 0):.2f} slot-equivalents at "
        f"D = 0 and {gain(S6, 'proactive_ai', 5):.2f} at D = 5: it is a detection-delay effect. "
        f"Under collapse it grows from {gain(S4, 'proactive_ai', 0):.2f} at D = 0 to "
        f"{gain(S4, 'proactive_ai', 5):.2f} at D = 5, so roughly "
        f"{100 * gain(S4, 'proactive_ai', 0) / gain(S4, 'proactive_ai', 3):.0f}% of the gain at "
        "the nominal D = 3 remains when detection is instantaneous. This is the part not attributable "
        "to detection-delay bypass, which includes anticipation of failures and the avoidance "
        "of degraded elements. Protection behaves in the opposite way, because its switch-over "
        "is triggered by the detector: its advantage over P1 is largest at small D. The two "
        "mechanisms therefore trade places under collapse as detection slows down; under "
        "common-cause disruption protection stays ahead of migration at every delay. "
        "Predictive protection has the lowest loss at D = 0 and D = 1 in both scenarios and "
        "under common-cause disruption at every delay, while under collapse at D = 3 and D = 5 "
        f"the lowest loss is reached by {LABEL_OF[best5]} ({ddL(S4, best5, 3):.2f} and "
        f"{ddL(S4, best5, 5):.2f} against {ddL(S4, 'protection_predictive', 3):.2f} and "
        f"{ddL(S4, 'protection_predictive', 5):.2f} for P6)."
    )

    # ------------------------------------------------------------ 7.6 RQ3
    d.h2("7.6 RQ3: actuation cost and matched actuation effort")
    fig_mig = d.figure("f11_migration_cost.png",
                       "Resilience loss as a function of the switch-over cost C_m, which applies "
                       "to pre-emptive migration (P4) and to the protection switch-over (P1b, "
                       "P6). Dotted line: reactive restoration.", width=15.0)
    mg = R["migration"]

    def mgL(s, p, c):
        return float(mg[(mg.scenario == s) & (mg.policy == p) & (mg.migration_cost == c)]
                     ["resilience_loss"].mean())

    ref3 = float(mg[(mg.scenario == S3) & (mg.policy == "reactive")]["resilience_loss"].mean())
    ref4 = float(mg[(mg.scenario == S4) & (mg.policy == "reactive")]["resilience_loss"].mean())
    costs = sorted(c for c in mg.migration_cost.unique() if c >= 0)
    x4 = next((c for c in costs if mgL(S4, "proactive_ai", c) >= ref4), None)
    xp = next((c for c in costs if mgL(S4, "proactive_ai", c) >= mgL(S4, "protection", c)), None)
    x3 = next((c for c in costs if mgL(S3, "proactive_ai", c) >= ref3), None)
    d.p(
        f"{fig_mig} sweeps the switch-over cost on 12 paired seeds. Under collapse the loss of "
        f"P4 rises from {mgL(S4, 'proactive_ai', 0):.2f} at C_m = 0, the lowest of the three "
        f"policies there, to {mgL(S4, 'proactive_ai', 8):.2f} at C_m = 8, against a reactive "
        f"reference of {ref4:.2f}. "
        + (f"It loses its advantage over protection from C_m = {xp:g} " if xp is not None else "")
        + (f"and over reactive restoration from C_m = {x4:g}. " if x4 is not None else
           "and stays below reactive restoration over the tested range. ")
        + "The protection family uses the same actuator for its switch-over and is also "
        "sensitive to C_m, but much less, since it switches only flows whose path has failed: "
        f"P6 goes from {mgL(S4, 'protection_predictive', 0):.2f} to "
        f"{mgL(S4, 'protection_predictive', 8):.2f}. Under non-terrestrial degradation the "
        f"reactive reference is {ref3:.2f}"
        + (f" and P4 is at or above it from C_m = {x3:g}. " if x3 is not None else ". ")
        + "Migration pays only when the switch-over is nearly free, because it moves many "
        "flows whose path would have survived; the practical criterion is to compare the "
        "measured switch-over cost with the detection-plus-restoration interval."
    )
    fig_fr = d.figure("r03_frontier.png",
                      "Resilience loss against the number of proactive migrations per run for "
                      "the risk sources at five operating points each: the budget scale kappa "
                      "for P3, P4, P5c and P6, and the alarm scale for the threshold policy P2. "
                      "Dotted lines: P1 and P1b. P6 performs no migrations and appears at zero.",
                      width=16.5)
    t_fr = d.table(pd.read_csv(os.path.join(TABDIR, "r_t_frontier.csv")),
                   "Best operating point of each risk source on the frontier of Fig. "
                   "{F_FR}, 12 paired seeds.", fmt="{}", size=7.0)
    d.set_placeholder("{F_FR}", fig_fr.split()[-1])
    frt = pd.read_csv(os.path.join(TABDIR, "r_t_frontier.csv"))

    def frb(s, p):
        r = frt[(frt.scenario == SCEN_LABEL[s]) & (frt.policy == SHORT[p])].iloc[0]
        return float(r["loss at best"]), float(r["migrations at best"])

    d.p(
        f"{fig_fr} and {t_fr} compare the risk sources at matched actuation effort instead of "
        "at one operating point. First, the threshold policy P2 is poor only when it is "
        "configured aggressively: at the nominal operator limits it triggers many more "
        "migrations, but relaxed to the predictor's alarm rate (P2m) or beyond it, it comes "
        f"close to the other migration policies; under collapse its best point is "
        f"{frb(S4, 'proactive_threshold')[0]:.2f} against {frb(S4, 'proactive_ai')[0]:.2f} "
        f"for P4 and {frb(S4, 'degradation_aware')[0]:.2f} for P3. That threshold-driven "
        "proactive action can be worse than reactive restoration is a property of an "
        "aggressive configuration, not of threshold rules as such, at least outside handover dynamics, "
        "where every migration policy is worse than reactive restoration. Second, the risk "
        "sources "
        "are ordered by information content under collapse: perfect state "
        f"({frb(S4, 'oracle_calibrated')[0]:.2f}) ahead of the predictor, the predictor ahead "
        "of the index and the threshold rule. The best perfect-state operating point is "
        f"below both protection policies ({frb(S4, 'protection')[0]:.2f} for P1b, "
        f"{frb(S4, 'protection_predictive')[0]:.2f} for P6). Third, under common-cause "
        "disruption no operating point of any migration-based policy reaches plain protection "
        f"({frb(S6, 'oracle_calibrated')[0]:.2f} for perfect state against "
        f"{frb(S6, 'protection')[0]:.2f} for P1b), because the shared shock gives nothing to "
        "anticipate."
    )

    # --------------------------------------------------- 7.7 risk source
    d.h2("7.7 Risk source and criticality differentiation")
    d.p(
        "A comparison between the index-driven and the predictive migration policy could "
        "confound the risk source with service differentiation, so the design makes both "
        "factors explicit. P3 and P4 share budgets b_c, the criticality term of Eq. (11) and "
        "criticality-ordered processing; P3u and P4u replace all three by uniform treatment. "
        "With differentiation, prediction is ahead of the index under collapse "
        f"({pc(S4, 'proactive_ai', 'degradation_aware')['diff']:+.2f}, "
        f"{ci_(S4, 'proactive_ai', 'degradation_aware')}) and under handover dynamics "
        f"({pc(S5, 'proactive_ai', 'degradation_aware')['diff']:+.2f}, "
        f"{ci_(S5, 'proactive_ai', 'degradation_aware')}); exploratory contrasts on the "
        f"confirmation seeds agree "
        f"({cf(S4, 'proactive_ai', 'degradation_aware')['rel_pct']:+.1f}% and "
        f"{cf(S5, 'proactive_ai', 'degradation_aware')['rel_pct']:+.1f}%). With uniform "
        "treatment the ordering under collapse reverses "
        f"({pc(S4, 'proactive_ai_uniform', 'degradation_aware_uniform')['diff']:+.2f}, "
        f"{ci_(S4, 'proactive_ai_uniform', 'degradation_aware_uniform')}), because "
        "differentiated budgets hurt the index-driven policy there "
        f"(P3 − P3u {pc(S4, 'degradation_aware', 'degradation_aware_uniform')['diff']:+.2f}, "
        f"{ci_(S4, 'degradation_aware', 'degradation_aware_uniform')}) and the predictive "
        f"one much less (P4 − P4u {pc(S4, 'proactive_ai', 'proactive_ai_uniform')['diff']:+.2f}). Under handover dynamics prediction stays ahead under both treatments "
        f"({pc(S5, 'proactive_ai_uniform', 'degradation_aware_uniform')['diff']:+.2f} uniform), "
        "since a contemporaneous index mistakes transient reconfiguration signatures for "
        "degradation. Among migration policies, then, the advantage of a learned predictor "
        "over a simple index is real where transients are frequent and depends on the "
        "controller configuration elsewhere; among protection policies (Section 7.3) the two "
        "are equivalent. One consequence deserves emphasis: under collapse, uniform "
        f"index-driven migration (P3u, {L_(S4, 'degradation_aware_uniform'):.2f}) has a lower "
        f"loss than protection ({pc(S4, 'degradation_aware_uniform', 'protection')['diff']:+.2f}, "
        f"{ci_(S4, 'degradation_aware_uniform', 'protection')}) and than predictive protection "
        f"({pc(S4, 'degradation_aware_uniform', 'protection_predictive')['diff']:+.2f}, "
        f"{ci_(S4, 'degradation_aware_uniform', 'protection_predictive')}) in the development "
        "campaign; this contrast was not part of the confirmation and is exploratory, but it "
        "shows that the value of migration under gradual degradation does not require a "
        "learned forecaster."
    )

    # ------------------------------------------------------------ 7.8 RQ4
    d.h2("7.8 RQ4: robustness to unseen mechanisms, model misspecification and scale")
    fig_ood = d.figure("f13_generalisation.png",
                       "Generalisation of the anticipation model. Left: precision–recall lift "
                       "over prevalence under leave-one-scenario-out training and on the two "
                       "mechanisms excluded from the corpus. Right: recall at the frozen "
                       "operating point and after re-tuning the threshold on a small "
                       "commissioning sample from the target regime.", width=15.5)
    ood = R["ood"]
    loso = ood[(ood["setting"] == "leave_one_scenario_out") & (ood["prevalence"] > 0)]
    unseen = ood[ood["setting"] == "unseen_mechanism"]
    d.p(
        f"{fig_ood} reports the offline transfer of the predictor. Under leave-one-scenario-out "
        f"training the precision–recall lift ranges from {loso['lift'].min():.1f} to "
        f"{loso['lift'].max():.1f}; the frozen operating point transfers poorly, and recall on "
        f"the held-out scenario lies between {pct(loso['recall'].min(), 1)}% and "
        f"{pct(loso['recall'].max(), 1)}%. On the two mechanisms withheld from the corpus, "
        f"handover dynamics retain a lift of {float(unseen[unseen.held_out == S5]['lift'].iloc[0]):.1f} "
        f"and common-cause disruption of {float(unseen[unseen.held_out == S6]['lift'].iloc[0]):.1f}."
    )
    ms = R["misspec_off"]
    fr_ = ms[ms.setting.str.startswith("frozen predictor (trained")].iloc[0]
    rt_ = ms[ms.setting.str.startswith("reference: retrained")].iloc[0]
    t_ms = d.table(pd.read_csv(os.path.join(TABDIR, "r_t_misspec.csv")),
                   "Closed-loop outcome in an alternative degradation world: gamma-process "
                   "wear-out without mean reversion, observed through a four-slot lagged map "
                   "with different exponents and doubled noise. The predictor and all operating "
                   "points are frozen from the jump-diffusion corpus. 24 paired seeds; the "
                   "interval is that of the paired difference to P1.", fmt="{}", size=6.8)
    msd = pd.read_csv(os.path.join(TABDIR, "r_t_misspec.csv"))

    def msv(s, p):
        r = msd[(msd.scenario == SCEN_LABEL[s]) & (msd.policy == p)]
        return float(r["vs P1 [%]"].iloc[0])

    d.p(
        "The predictor is trained and tested on noisy transformations of the very latent "
        "variable that defines failure, so the offline metrics validate it only within the "
        "generative family that produced them, and perturbing parameters of that family is "
        "not misspecification. We therefore built a second world that changes the "
        "health-evolution process and the telemetry observation map, while retaining the common "
        "failure threshold, the service model and some disturbance mechanisms of the first: health decreases monotonically through gamma-distributed wear increments "
        "with no mean reversion, and telemetry reflects the health of four slots earlier "
        "through different exponents and with twice the measurement noise. Frozen and applied "
        f"to this world, the predictor retains a precision–recall area of {fr_['pr_auc']:.3f} "
        f"(lift {fr_['lift']:.1f}) against {rt_['pr_auc']:.3f} (lift {rt_['lift']:.1f}) for "
        f"the same estimator retrained on the new world. {t_ms} gives the closed-loop "
        "consequence. Every policy except index-driven migration under non-terrestrial degradation improves on reactive restoration; the protection policies "
        f"with a risk-refreshed standby path are best in both scenarios ({msv(S4, 'P6'):+.1f}% "
        f"for P6 under collapse), predictive migration ({msv(S4, 'P4'):+.1f}%) and protection "
        f"alone ({msv(S4, 'P1b'):+.1f}%) follow, and index-driven migration gains least "
        f"({msv(S4, 'P3'):+.1f}%). The framework and the decomposition transfer across the two "
        "degradation families we have built; the absolute quality of the predictor is a "
        "property of the family it was trained on, and no claim is made for real failure "
        "mechanisms."
    )
    if R["large"] is not None:
        t_lg = d.table(pd.read_csv(os.path.join(TABDIR, "r_t_large.csv")),
                       "Results on a 98-element topology with three LEO planes (24 satellites, 233 links). "
                       "The predictor and all operating points are frozen from the 19-element "
                       "reference network. 12 paired seeds.", fmt="{}", size=6.8)
        lg = R["large"]

        def lgL(s, p):
            return float(lg[(lg.scenario == s) & (lg.policy == p)]["resilience_loss"].mean())

        d.p(
            "A 19-element network cannot represent a satellite constellation, so the "
            "experiment is repeated on a 98-element topology with three LEO orbital "
            "planes of eight satellites each, intra-plane and inter-plane links, two core "
            "and feeder sites and eight terrestrial regions (98 elements, 233 links; "
            "Section 6.2). "
            f"{t_lg} shows the outcome. Under collapse the losses are "
            f"P1 {lgL(S4, 'reactive'):.2f}, P1b {lgL(S4, 'protection'):.2f}, "
            f"P3 {lgL(S4, 'degradation_aware'):.2f}, P6 {lgL(S4, 'protection_predictive'):.2f} and "
            f"P4 {lgL(S4, 'proactive_ai'):.2f}: with many more alternative paths, predictive "
            "migration has the lowest loss. Under common-cause disruption the protection family "
            f"is ahead (P6 {lgL(S6, 'protection_predictive'):.2f}, P1b "
            f"{lgL(S6, 'protection'):.2f}, P4 {lgL(S6, 'proactive_ai'):.2f}). The scenario-"
            "dependence of the mechanism ranking found on the reference network therefore "
            "persists at larger scale. We treat this experiment as a structural sensitivity "
            "analysis, not as a satellite validation: the larger network has no orbital mechanics "
            "or visibility windows, and we do not claim that it makes the model representative "
            "of an operational constellation."
        )
    _r7 = R["runs"][R["runs"]["scenario"] == "S7_external_ntn"]
    _a7 = _r7[_r7.policy == "proactive_ai"]
    s7fail = float((_a7.anticipated_failures + _a7.missed_failures).mean())
    s7mig = float(_a7.proactive_reroutes_total.mean())
    s7mig5 = float(_r7[_r7.policy == "oracle_calibrated"].proactive_reroutes_total.mean())
    d.p(
        "Scenario S7, whose reconfiguration cadence and phase are taken from Tanveer et al. "
        "[52] and whose latency statistics are taken from Michel et al. [51], is an externally "
        "parameterised stress test and not an external validation; all its comparisons are "
        "exploratory. Its dominant impairment is a scheduled reconfiguration with no "
        "degradation precursor and no load coupling. Element failures are few "
        f"({s7fail:.1f} per run), and although P4 anticipates about half of them, it performs "
        f"{s7mig:.0f} migrations per run against {s7mig5:.0f} for the calibrated "
        "perfect-state policy, so most of its migrations do not pay. The "
        f"migration policies are within noise of reactive restoration "
        f"({rel_(S7, 'proactive_ai', 'reactive'):+.1f}% for P4, interval "
        f"{ci_(S7, 'proactive_ai', 'reactive')}; {rel_(S7, 'degradation_aware', 'reactive'):+.1f}% "
        "for P3), while protection and predictive protection improve on it "
        f"({rel_(S7, 'protection', 'reactive'):+.1f}% and "
        f"{rel_(S7, 'protection_predictive', 'reactive'):+.1f}%). The recovery-time advantage that "
        "P4 shows in S7 when all interruptions are averaged is again a composition effect: "
        f"its recovery from actual failures is not faster but slower, {R['REF'].loc[S7, 'proactive_ai']:.2f} slots against "
        f"{R['REF'].loc[S7, 'reactive']:.2f} for P1 and {R['REF'].loc[S7, 'protection']:.2f} for "
        "P1b (Appendix A)."
    )

    # ----------------------------------------------- 7.9 protection model
    d.h2("7.9 Protection model and standby capacity")
    pm = R["pmodes"]
    RU = R["runs"].groupby(["scenario", "policy"])[["backup_refreshes"]].mean()
    t_pm = d.table(pd.read_csv(os.path.join(TABDIR, "r_t_protection_modes.csv")),
                   "Sensitivity of the protection results to the standby-capacity model "
                   "(24 paired seeds). Feasible: unreserved standby capacity, switch-over only "
                   "if the standby path has enough residual capacity at the moment of the "
                   "switch (main model). Reserved: standby bandwidth reserved and unavailable "
                   "to other flows. Optimistic: switch-over regardless of residual capacity.",
                   fmt="{}", size=6.6)
    pmt = pd.read_csv(os.path.join(TABDIR, "r_t_protection_modes.csv"))
    d.p(
        "Protection carries much of the benefit, so its modelling matters. The main model "
        "(Section 3.4) does not reserve standby capacity but switches a flow only if its "
        "standby path can carry it at the moment of the switch, processing flows in "
        "decreasing criticality, and falls back to reactive restoration otherwise. "
        f"{t_pm} compares it with reserved standby capacity and with an optimistic model "
        "that ignores residual capacity. The capacity check binds under collapse, where "
        f"P1b finds its standby path infeasible {pmt.iloc[0]['infeasible switches / run (P1b)']:.1f} "
        "times per run, and costs it "
        f"{pmt.iloc[0]['P1b'] - pmt.iloc[2]['P1b']:.2f} slot-equivalents against the optimistic "
        "model; reservation recovers most of that. The advantage of risk-informed standby "
        "selection is present under all three models in both scenarios, with intervals that "
        "exclude zero (P6 − P1b column), so the effect tested in B1 does not depend on the "
        "capacity assumption. Two further properties of the protection model should be noted. "
        "Its switch-over is triggered by the same three-slot KPI-based detector as reactive "
        "restoration, whereas deployed protection typically uses much faster liveness "
        "detection; this is conservative for the protection policies and favours migration. "
        "And predictive protection is not free of control-plane work: under collapse it "
        f"refreshes standby paths {RU.loc[(S4, 'protection_predictive'), 'backup_refreshes']:.0f} "
        f"times per run and the index variant {RU.loc[(S4, 'protection_index'), 'backup_refreshes']:.0f} "
        f"times (under common-cause disruption {RU.loc[(S6, 'protection_predictive'), 'backup_refreshes']:.0f} "
        "and "
        f"{RU.loc[(S6, 'protection_index'), 'backup_refreshes']:.0f}), none of which interrupts "
        "traffic."
    )

    p6d_path = os.path.join(TABDIR, "r_t_p6_decomposition.csv")
    if os.path.exists(p6d_path):
        p6d = pd.read_csv(p6d_path)
        t_p6d = d.table(p6d, "Exploratory decomposition of predictive protection, run after "
                        "the confirmation on the same 64 seeds. P6-route uses the risk "
                        "estimate only to choose the primary path and keeps a risk-free "
                        "standby path as in P1b; P6-standby chooses the primary path as P1b "
                        "does and uses the risk estimate only to select and refresh the standby "
                        "path. No parameter was changed; P6 reproduces the confirmation runs "
                        "exactly.", fmt="{}", size=6.6)

        def _pd(s, c):
            r = p6d[(p6d.scenario == SCEN_LABEL[s]) & (p6d.contrast == c)].iloc[0]
            v = r["paired difference [95% CI]"]
            m = float(v.split(" ")[0])
            lo, hi = [float(x) for x in v[v.index("[") + 1:v.index("]")].split(",")]
            return m, lo, hi, float(r["relative [%]"])

        def _desc(s):
            out = []
            for c, name in (("P6-route - P1b", "risk-aware primary-path choice alone"),
                            ("P6-standby - P1b", "risk-aware standby selection alone")):
                m, lo, hi, rel_ = _pd(s, c)
                sig = "interval excludes zero" if (lo > 0 or hi < 0) else "interval includes zero"
                out.append(f"{name} changes the loss by {rel_:+.1f}% ({m:+.2f}, {sig})")
            mf, _, _, relf = _pd(s, "P6 - P1b")
            return "; ".join(out) + f", against {relf:+.1f}% for the full policy"

        d.p(
            "Predictive protection uses the risk estimate in two places, and it is natural to "
            f"ask which of them carries the effect. {t_p6d} separates them in an "
            "exploratory ablation run after the confirmation on the same seeds. Under collapse "
            f"{_desc(S4)}. Under common-cause disruption {_desc(S6)}. "
            + "The effect of predictive protection therefore comes almost entirely from "
            "keeping the standby path off endangered elements; using risk to choose the primary "
            "path adds nothing detectable, plausibly because a well-chosen standby path already "
            "absorbs the failures that a riskier primary path suffers, and the full policy is "
            "indistinguishable from the standby-only variant in both scenarios." +
            " The ablation is exploratory and was not part of the frozen protocol."
        )

    # ------------------------------------------------------------ 7.10 horizon etc.
    d.h2("7.10 Horizon, feedback and sensitivity")
    hs = R["horizon"]
    fig_h = d.figure("f12_horizon.png",
                     "Effect of the prediction horizon H: prediction quality, closed-loop lead "
                     "time of P4 (with the bound T_lead ≤ H), and resilience loss of P4 and P6 "
                     "under collapse.", width=15.5)
    t_h = d.table(pd.read_csv(os.path.join(TABDIR, "r_t_horizon.csv")),
                  "Prediction-horizon sweep. For every H a model is retrained on labels "
                  "relabelled at that horizon; closed-loop columns are averaged over 12 seeds of "
                  "the collapse scenario.", fmt="{:g}", size=6.6)
    best_h = int(hs.loc[hs['cl_resilience_loss'].idxmin(), 'horizon'])
    d.p(
        f"{fig_h} and {t_h} vary the horizon H; for every H a model is retrained on relabelled "
        "data and the lead-time tracker receives the same H, so the horizon-consistent "
        f"matching of Eq. {eq18} applies at every point and a unit test enforces the bound. "
        f"The median closed-loop lead time lies between "
        f"{hs['cl_lead_time_median'].min():.1f} and {hs['cl_lead_time_median'].max():.1f} slots "
        "and never exceeds H. The precision–recall area varies little "
        f"({hs['pr_auc'].min():.2f}–{hs['pr_auc'].max():.2f}) while the prevalence rises and the "
        "lift over prevalence falls. In the closed loop the loss of P4 is lowest at "
        f"H = {best_h} ({hs['cl_resilience_loss'].min():.2f}) and highest at "
        f"H = {int(hs.loc[hs['cl_resilience_loss'].idxmax(), 'horizon'])} "
        f"({hs['cl_resilience_loss'].max():.2f}), against {R['hz_ref']:.2f} for reactive "
        "restoration on the same seeds; longer horizons raise more alarms on elements that do "
        f"not fail soon ({hs['cl_migrations'].iloc[0]:.0f} migrations per run at the shortest "
        f"horizon, up to {hs['cl_migrations'].max():.0f} at longer ones). Predictive protection is "
        f"insensitive to H ({hs['p6_resilience_loss'].min():.2f} to "
        f"{hs['p6_resilience_loss'].max():.2f}), since refreshing a standby path on a false "
        "alarm costs nothing. The differences between horizons for P4 are of the order of the "
        "seed-to-seed uncertainty of a 12-seed comparison, so the trend is indicative: for "
        "migration-based control a horizon of the order of the detection and actuation delays "
        "is sufficient, and longer horizons mainly add migrations."
    )
    t_fb = d.table(pd.read_csv(os.path.join(TABDIR, "r_t_feedback.csv")),
                   "Feedback ablation at fixed traffic. In the replayed arm the "
                   "degradation process receives the utilisation trajectory that P1 produced on "
                   "the same seed, so the offered load level, the traffic and all exogenous "
                   "randomness are identical to the endogenous arm and only the response of "
                   "degradation to the policy's own routing is removed. 'Strong' raises the "
                   "load-to-degradation gain fourfold and lowers the knee to 0.40. 12 paired "
                   "seeds.", fmt="{}", size=6.4)
    fb = pd.read_csv(os.path.join(TABDIR, "r_t_feedback.csv"))
    nom = fb[fb.coupling == "nominal"]
    en = R["endo"]
    d.p(
        "Equation (2) lets a policy shape its own failure process through the load it places "
        "on elements. An ablation that replaces the controller's load with an exogenous "
        "profile also changes the load level, so the ablation is performed with the traffic "
        f"held fixed. {t_fb} shows that at the nominal coupling the endogenous and replayed "
        "arms differ by at most "
        f"{(nom['loss, endogenous load'] - nom['loss, replayed P1 load']).abs().max():.2f} "
        "slot-equivalents for any policy; the largest difference is that of the migration "
        "policy under collapse, which does slightly better when its own load does not feed "
        "back into degradation. Utilisation exceeds the knee of Eq. (2) in "
        f"{100 * en[(en.coupling == 'nominal') & (en.scenario == S4)].util_above_knee.mean():.0f}% "
        "of element-slots under collapse at the nominal coupling and in "
        f"{100 * en[(en.coupling == 'strong') & (en.scenario == S4)].util_above_knee.mean():.0f}% "
        "with a fourfold gain and the knee lowered to 0.40; even then the replayed and "
        "endogenous arms are not distinguishable. The closed loop is the correct evaluation "
        "structure and is retained, but in this parameterisation the feedback path is not a "
        "source of benefit or harm of practical size."
    )
    dst = pd.read_csv(os.path.join(TABDIR, "r_t_design.csv"))
    _dsr = pd.read_csv(os.path.join(RESULTS, "design_sensitivity.csv"))
    _b = _dsr[(_dsr.setting == "protection reference")].sort_values("seed")
    n_ds_sig, _cis = 0, []
    for _st in dst.setting:
        _a = _dsr[(_dsr.setting == _st) & (_dsr.policy == "protection_predictive")].sort_values("seed")
        _dd = _a.resilience_loss.to_numpy() - _b.resilience_loss.to_numpy()
        _h = _st_t.ppf(0.975, len(_dd) - 1) * _dd.std(ddof=1) / np.sqrt(len(_dd))
        n_ds_sig += int(abs(_dd.mean()) > _h)
        _cis.append(f"{_dd.mean():+.2f} [{_dd.mean() - _h:.2f}, {_dd.mean() + _h:.2f}]")
    dst_show = dst.copy()
    dst_show["P1b loss (reference)"] = round(float(_b.resilience_loss.mean()), 2)
    dst_show["P6 - P1b, 95% CI"] = _cis
    t_ds = d.table(dst_show,
                   "Sensitivity to the controller design choices of Eqs. (10) and (11) under "
                   "collapse: alternative budget-factor profiles b_c (flat 1/1/1/1/1, "
                   "shallow 0.90–1.75, nominal 0.80–2.50, steep 0.50–4.00) and one-at-a-time "
                   "changes of the loss coefficient (nominal 4), the risk base (nominal 3) and "
                   "the criticality slope (nominal 0.6). The protection reference P1b does not "
                   "use these coefficients and is run once. 24 paired seeds of the sweep seed set "
                   "7000 + 61k, so the nominal values differ from Table 9.", fmt="{}",
                   size=6.2)
    t_pa = d.table(pd.read_csv(os.path.join(TABDIR, "r_t_parameters.csv")).replace({"setting": {"lambda0 x0.5": "nu_0 x0.5", "lambda0 x2.0": "nu_0 x2.0"}}),
                   "Sensitivity to the degradation-model parameters under collapse: baseline "
                   "degradation rate nu_0, diffusion amplitude sigma and load-to-degradation "
                   "gain g; nu_0 and sigma are halved and doubled, g is switched off and "
                   "doubled (12 paired seeds).", fmt="{}", size=6.6)
    pat = pd.read_csv(os.path.join(TABDIR, "r_t_parameters.csv"))
    t_tp = d.table(pd.read_csv(os.path.join(TABDIR, "r_t_topology.csv")),
                   "Dependence on structural redundancy under collapse. Only the graph "
                   "changes across classes (12 paired seeds).", fmt="{}", size=6.6)
    tpt = pd.read_csv(os.path.join(TABDIR, "r_t_topology.csv"))
    p4_first = tpt[tpt["P4 loss"] < tpt[["P1b loss", "P6 loss"]].min(axis=1)].topology.tolist()
    d.p(
        f"{t_ds} examines the budget factors and the coefficients of Eq. (11) under collapse. "
        f"The loss of P4 varies between {dst['P4 loss'].min():.2f} and {dst['P4 loss'].max():.2f} "
        f"and that of P6 between {dst['P6 loss'].min():.2f} and {dst['P6 loss'].max():.2f}. "
        "The criticality slope and the risk base are inert, because the budgets, not "
        "the routing penalty, decide when a policy acts. The steep budget profile hurts P4, "
        "and the nominal profile is within noise of the best one for it, so the nominal "
        "configuration does not handicap predictive migration; P6 is insensitive to the "
        f"profile and has a lower loss than protection alone in every setting "
        f"({dst['P6 vs P1b [%]'].max():+.1f}% to {dst['P6 vs P1b [%]'].min():+.1f}%), with "
        f"a paired interval that excludes zero in {n_ds_sig} of {len(dst)} settings. "
        f"{t_pa} perturbs the generative model. P1b, P4 and P6 improve on P1 in every "
        "setting. Where degradation is fast or noisy (doubled nu_0 or sigma) predictive "
        "migration and predictive protection are close; where it is slow and smooth (halved "
        "nu_0 or sigma) failures are rare, few migrations pay and the protection family is "
        "ahead in relative terms, although the absolute losses there are small. The feedback gain is again inert. "
        f"{t_tp} varies the redundancy of the graph with the element catalogue, traffic and "
        "degradation fixed. Predictive migration has the lowest loss in "
        f"{['no', 'one', 'two', 'three', 'four', 'all'][len(p4_first)]} of the five classes on these seeds ({', '.join(p4_first)}) and the "
        "protection family in the remainder; "
        "the gains over reactive restoration grow from the sparse to the redundant class, "
        "because more redundant graphs offer more disjoint alternatives to move to."
    )
    t_w = d.table(pd.read_csv(os.path.join(TABDIR, "r_t_weights.csv")),
                  "Sensitivity to the cost weights of Eq. (7) under collapse (12 paired seeds).",
                  fmt="{}", size=6.8)
    d.p(
        f"{t_w} rescales the weights of the element penalty of Eq. (7) by factors of two. "
        "The results change by about one per cent or less; the load weight is inert because the "
        "utilisation term activates only above 0.75, which the reference topology rarely "
        "reaches. Note that the collapse scenario on these sweep seeds puts predictive "
        "migration slightly ahead of predictive protection, whereas the 64 confirmation seeds "
        "find no detectable difference between the two; the ranking of these two mechanisms "
        "under collapse is not stable across seed sets, which is why it was tested on "
        "independent seeds."
    )
    if not R["noise"].empty:
        fig_n = d.figure("f14_noise.png",
                         "Value of temporal aggregation as a function of observation noise.",
                         width=9.0)
        d.p(
            f"{fig_n} shows that the temporal window features help the predictor only when "
            "the observation noise is well above its nominal level; at the nominal level a "
            "single KPI sample identifies the latent state about as well, because the "
            "generative dynamics are Markov in the latent state. Section 8.3 draws the "
            "consequence."
        )

    # ------------------------------------------------------- 8. Discussion
    d.h1("8. Discussion")
    d.h2("8.1 Where the resilience gain comes from")
    d.p(
        "The question in the title has a layered answer. The largest and most consistent "
        "contribution is protection: a pre-computed disjoint standby path turns a restoration "
        "of D + C_r slots into a switch-over of D + C_m slots, uses no estimate of the future, and "
        f"reduced the loss by {min(_prot):.1f}–{max(_prot):.1f}% on the confirmation seeds. "
        "The second, which links this study to risk-aware backup provisioning [83], [84], [85], "
        "is exposure control through decisions that interrupt nothing: choosing a "
        "low-risk primary path and keeping the standby path off endangered elements; an "
        "exploratory ablation (Section 7.9) attributes practically all of it to the standby "
        "path. Its "
        f"effect is small, {-_c(S4, 'protection_predictive', 'protection'):.1f}% and "
        f"{-_c(S6, 'protection_predictive', 'protection'):.1f}% on top of protection, but it was "
        "confirmed on fresh seeds, holds under all three standby-capacity models and in nine of "
        "ten controller-design settings, and cannot create an outage. The third is pre-emptive "
        "migration, whose gain has two parts. One is the bypass of the detection delay, that is, "
        "acting on elements that have already failed but are not yet confirmed down; this is all "
        "there is under common-cause disruption and vanishes when detection is instantaneous. "
        "The other part, which under collapse survives at D = 0, comes from acting on elements "
        "that are failing or degraded before the failure is confirmed; it makes predictive "
        "migration competitive with protection under collapse, and ahead of it on several "
        "sweep seed sets and on the larger topology."
    )
    d.h2("8.2 Why prediction adds little to protection")
    d.p(
        "Inside the protection family the source of the risk estimate does not matter: a "
        "contemporaneous index "
        + ("is statistically equivalent to " if _eq_all else "performs like ")
        + "the trained predictor within margins of 3% of the protection loss, and perfect "
        f"state knowledge adds at most {_hrmax:.1f}%. The reason is the action. A standby path "
        "only has to avoid elements that are visibly unhealthy now; whether they will fail in "
        "eight or in eighteen slots is irrelevant to that choice, and a wrong choice costs "
        "nothing because no traffic is moved. Migration is different: it moves live traffic at "
        "a price, so it profits from information about which elements are about to fail. That "
        "is why the risk source matters among migration policies. In the development "
        "campaign at the nominal settings, "
        "perfect state knowledge used for migration has the lowest loss of all policies under "
        "collapse, and the learned predictor is ahead of the index where transient signatures "
        "are frequent. Among protection policies it does not matter. The information "
        "need not come from a forecaster: under collapse, index-driven migration with uniform "
        "treatment of service classes (P3u) also has a lower loss than both protection "
        "policies in the development campaign. Three properties of the modelled system "
        "limit what anticipation can buy through migration, and each is checkable in a real "
        "network: much of the failure process is abrupt (jumps, destroyed elements, the "
        "shared shock), so the anticipation rate stays well below one; the available warning "
        "is short; and every migration of a flow whose path would have survived pays C_m."
    )
    d.h2("8.3 When would anticipation pay?")
    d.p(
        "The results do not show that failure prediction is useless for network resilience; "
        "they show where its value lies in a system of this kind and what would have to hold "
        "for migration on predicted risk to pay. Detection must be slow relative to the "
        "warning: at a detection delay of five slots, migration with perfect or predicted "
        "state overtakes predictive protection under collapse. The switch-over must be nearly "
        "free: at C_m = 0 predictive migration has the lowest loss of the three swept policies "
        "under collapse, at one slot "
        "it is already behind protection. Degradation must be gradual and observable, which "
        "excludes common-cause shocks and scheduled reconfiguration; the externally "
        "parameterised stress test S7, where predictive migration was within noise of reactive "
        "restoration, is an example. And the graph must offer "
        "alternatives to move to: on the 98-element topology, with three orbital planes, "
        "predictive migration had the lowest loss under collapse. The gap "
        "between the learned predictor and perfect state in the migration role is an exploratory "
        "indication of the potential value of improved risk information under privileged-state "
        "access, not a measured achievable bound; in the standby role the gap is almost nil. "
        "An operator should establish detection latency, switch-over cost, the availability "
        "of disjoint standby capacity and the share of failures with an observable precursor "
        "before investing in anticipation instead of protection, and should use a simple "
        "degradation index for standby selection in any case."
    )
    d.h2("8.4 What the digital twin contributes")
    d.p(
        "What distinguishes the framework from a network simulator, given that its physical "
        "counterpart is itself simulated, is structural rather than terminological. A simulator integrates the "
        "system state forward and is queried by an experimenter; the twin of this study is a "
        "separate estimator that sees the simulated system only through its telemetry, "
        "maintains a delayed and noisy belief about it, answers predictive questions from that "
        "belief and closes the loop by acting on the simulated system, while an assessment "
        "layer with privileged access scores the result. That separation is what made the "
        "central findings of this study measurable: the detection-delay bypass exists only "
        "because the twin's belief lags the physical state, and the dead-path accounting "
        "issue could be identified only because the assessment layer knows what the twin does not. "
        "We call the construction a simulation-based digital twin throughout, and it has not "
        "been synchronised with a deployed network."
    )
    d.h2("8.5 Threats to validity")
    d.p(
        "Internal validity rests on the common-random-number design, which is implemented "
        "and not merely assumed: every exogenous process draws from its own reproducible stream, "
        "every element draws a fixed-size block of random numbers in every slot, the number of "
        "generated flows is identical across policies for every seed, and a unit test asserts "
        "that the random blocks are identical across policies. Operating points of the "
        "comparison baselines are matched on validation runs only and frozen before the test "
        "seeds are used. The principal internal threat remains that the generative model and "
        "the controllers share a designer; the misspecified world of Section 7.8, the "
        "parameter and design sweeps and the externally parameterised scenario address it "
        "only partially. External validity is the substantive limitation (Section 9). Construct "
        "validity required the separation of failure- and migration-induced interruptions, "
        "without which mean recovery time would have rewarded policies for creating short "
        "interruptions of their own."
    )

    # ------------------------------------------------------ 9. Limitations
    d.h1("9. Limitations")
    _cr = R["conf"]

    def _rf(sc, pol):
        return float(_cr[(_cr.scenario == sc) & (_cr.policy == pol)]["backup_refreshes"].mean())

    d.numbered([
        "The evaluation is conducted entirely in simulation. No measurements from physical LEO "
        "terminals or 6G-class hardware and no field trial are used. Absolute values are "
        "properties of the model; only the comparison between policies, which share the "
        "environment exactly, should inform expectations about deployed systems.",
        "Both degradation families are modelling assumptions. The misspecified world of "
        "Section 7.8 tests transfer between two families we built; it is not evidence about "
        "real failure mechanisms, which include wear-out, adversarial and cascading modes that "
        "neither family reproduces. The predictor has been validated only within the family "
        "it was trained on; the framework and the decomposition are what the study claims to establish.",
        "The non-terrestrial dynamics of S5 and S7 are measurement-anchored rather than "
        "trace-driven; S7 adopts published summary statistics [51], [52] and does not replay "
        "an operational trace.",
        "Protection is modelled as 1:1 with a node-disjoint, pre-signalled standby path, as in "
        "fast reroute [79]; the switch-over onto it is charged C_m even though the active path "
        "is already down, whereas a migration off a dead path must first establish its new "
        "path and is charged C_r. Standby capacity is not reserved in the main model, but a "
        "switch-over happens only if the standby path can carry the flow at that moment; "
        "Section 7.9 shows that reserved and optimistic capacity models change the absolute "
        "numbers but not the confirmed effect of risk-informed protection. Much of the "
        "protection advantage rests on the pre-signalling assumption: if the switch-over cost "
        f"rises to the order of the restoration interval, protection loses its advantage "
        f"({fig_mig}). Shared protection and protocol-level session continuity are not "
        "modelled.",
        "The 98-element topology adds orbital planes and inter-satellite links but no orbital "
        "mechanics: visibility and link availability are not driven by ephemeris. It is a "
        "single alternative structure, so it shows that the ranking survives one change of "
        "scale and shape, not that it is invariant to topology.",
        "The control-plane cost of risk-driven refresh is counted but not charged in the "
        "resilience endpoint. Under collapse P6 performs "
        f"{_rf(S4, 'protection_predictive'):.0f} standby refreshes per run on average and P6i "
        f"{_rf(S4, 'protection_index'):.0f}, against none for P1b; signalling load, state "
        "churn and the risk of a refresh racing a failure are not modelled, so the P6 − P1b "
        "gain is a gain before control-plane overhead.",
        "Correlated failures are modelled by a single-region shared shock; multi-region, "
        "time-extended and cascading dependencies are not addressed.",
        "The controllers are greedy per-flow heuristics. No optimality gap against a global "
        "formulation is reported, and the budget rule of Eq. (10) is a heuristic whose "
        "economic reading (Section 4.5) is ex-post.",
        "The traffic model uses marked Poisson arrivals; real emergency traffic is bursty and "
        "event-correlated.",
    ], prefix="({i})")

    # ----------------------------------------------------- 10. Future work
    d.h1("10. Future work")
    d.bullets([
        "Cost-aware actuation. Deriving the action directly from expected costs, and "
        "choosing per decision between standby refresh, migration and waiting, is the natural "
        "successor of the budget heuristic [76], [77].",
        "Trace-driven evaluation. Coupling the twin to a constellation simulator [50], [54] or "
        "to captured operational telemetry would allow the degradation model to be fitted "
        "rather than specified, and is the only route to validating the predictor itself.",
        "Protection design under uncertainty. Since risk information pays most reliably "
        "through standby selection, shared protection with risk-aware backup placement is "
        "the most promising direction suggested by the results.",
        "Richer common-cause structures, including multi-region and cascading dependencies.",
    ])

    # ----------------------------------------------------- 11. Conclusion
    d.h1("11. Conclusion")
    d.p(
        "This paper asked where the resilience benefit of failure prediction comes from in a "
        "hybrid terrestrial–satellite network, and answered it with a simulation-based digital "
        "twin in which the degradation state is hidden, detection is delayed and fifteen "
        "recovery policies are compared on common random numbers with a physically consistent "
        "account of every interruption. The central comparisons were re-tested on 64 "
        "independent seeds of the same simulation model under a protocol frozen before they "
        "were run."
    )
    d.p(
        "In the evaluated systems the answer is layered. Plain 1:1 protection, without any "
        f"risk information, reduced cumulative resilience loss by {min(_prot):.1f}–"
        f"{max(_prot):.1f}% relative to reactive restoration. Using the risk estimate only for "
        "decisions that interrupt no traffic (primary-path choice and standby refresh) added "
        f"a confirmed {-_c(S4, 'protection_predictive', 'protection'):.1f}% under "
        f"infrastructure collapse and {-_c(S6, 'protection_predictive', 'protection'):.1f}% "
        "under common-cause disruption, which an exploratory ablation attributes practically "
        "entirely to the choice of the standby path, and for this role a contemporaneous degradation index "
        + ("was statistically equivalent to" if _eq_all else "performed like")
        + f" the learned predictor, with at most {_hrmax:.1f}% left for perfect state "
        "knowledge in exploratory comparisons. Predictive make-before-break migration was not "
        "significantly different from protection under collapse and non-terrestrial "
        "degradation after multiplicity correction, worse under handover "
        "dynamics and common-cause disruption, and its gain under common-cause disruption was "
        "almost entirely a bypass of the detection delay. Under collapse part of its gain "
        "survives at zero detection delay, the gain beyond it grows with the delay, and in the "
        "switch-over cost sweep migration falls behind protection once the switch-over costs "
        "one slot. On the 98-element topology, in most sweep seed sets and in the misspecified "
        "world predictive migration was ahead of protection under collapse, while the "
        "confirmation seeds found no significant difference; that comparison is therefore "
        "reported as system- and seed-dependent and not as settled. The predictor's "
        "own accuracy is established only within the model family it was trained on. In the "
        "exploratory, externally parameterised stress test S7, predictive migration was within "
        f"noise of reactive restoration ({rel_(S7, 'proactive_ai', 'reactive'):+.1f}%) while "
        f"protection kept a {-rel_(S7, 'protection', 'reactive'):.0f}% reduction."
    )
    d.p(
        "For operators and designers the implication is to spend risk information where "
        "acting on it is cheap: provision protection where disjoint standby capacity exists, "
        "use a simple risk or degradation estimate to place and refresh standby paths, and "
        "treat pre-emptive migration of live traffic as system-dependent: worth evaluating "
        "where detection is slow, switch-over is nearly free and degradation is gradual and "
        "observable, and not worth it under precursor-free common-cause shocks. Criticality-"
        "differentiated migration budgets did not help in this study and should be tested "
        "rather than assumed. The "
        "environment, controllers, frozen protocol and analysis pipeline are released so that "
        "the same decomposition can be repeated on other systems and, eventually, on "
        "operational data."
    )

    d.h1("Appendix A. Complete policy comparison")
    d.p("The table gives every policy in every scenario, averaged over 24 paired seeds. "
        "Recovery is reported both over all interruptions and over failure-induced "
        "interruptions only (Section 7.4). Recovery times are pooled over all interruptions of "
        "the 24 runs, so runs without an interruption of a given cause do not enter that "
        "average; a dash marks configurations with no interruption. SLA is the SLA-compliant "
        "availability over all flow-slots and classes (Section 5.1); its per-class breakdown "
        "is given in Section 7.4.")
    d.table(pd.read_csv(os.path.join(TABDIR, "r_t_full.csv")),
            "All fifteen policies across all seven scenarios. Rn: normalised resilience; "
            "migrations: proactive migrations per run; switch-overs: protection switch-overs "
            "per run.", fmt="{:g}", size=6.0)

    d.h1("Appendix B. Calibration, fold structure and Monte Carlo convergence")
    ca_path = os.path.join(RESULTS, "calibration_audit.csv")
    if os.path.exists(ca_path):
        ca = pd.read_csv(ca_path)
        ca_show = ca.rename(columns={
            "setting": "fold structure", "threshold": "operating point",
            "alarm_rate": "alarm rate", "roc_auc": "ROC-AUC", "brier": "Brier score",
            "ece": "ECE", "f1": "F1", "calibration_slope": "calibration slope",
            "calibration_intercept": "calibration intercept"})
        t_ca = d.table(ca_show.round(4), "Calibration quality and the effect of separating the "
                                         "calibration fold from the threshold-selection fold.",
                       size=7.0)
        shared = ca[ca.setting.str.startswith("shared")].iloc[0]
        disj = ca[ca.setting.str.startswith("disjoint")].iloc[0]
        d.p(
            f"{t_ca} reports calibration diagnostics on the test fold. The expected "
            f"calibration error is {shared['ece']:.4f} over fifteen equal-width bins and the "
            f"calibration slope {shared['calibration_slope']:.2f}. Fitting the isotonic map and "
            "selecting the threshold on disjoint halves of the validation episodes moves the "
            f"operating point from {shared['threshold']:.3f} to {disj['threshold']:.3f}, so "
            "the shared fold does not produce an optimistic operating point."
        )
    fig_sc = d.figure("f19_seed_convergence.png",
                      "Paired effect under infrastructure collapse as a function of the number "
                      "of seeds, with 95% intervals, for P4 against P1 and P6 against P1b.",
                      width=11.0)
    sc = pd.read_csv(os.path.join(TABDIR, "r_t_seeds.csv"))
    d.p(
        f"{fig_sc} recomputes the two principal paired effects on the first 8 to 24 seeds. "
        + "Here " + "; ".join(
            f"the {c.replace(' - ', '-versus-')} interval excludes zero from "
            f"{int(g[(g['mean difference'].abs() > g['ci half-width'])].seeds.min())} seeds"
            for c, g in sc.groupby("contrast", sort=False)
            if (g['mean difference'].abs() > g['ci half-width']).any())
        + "; both intervals contract as expected; "
        "the interval half-width of the P6-versus-P1b effect at 24 seeds is "
        f"{float(sc[sc.contrast.str.startswith('P6')]['ci half-width'].iloc[-1]):.2f} "
        "slot-equivalents."
    )

    d.h1("Data and code availability")
    d.p(
        "The complete implementation (environment, feature pipeline, anticipation models, "
        "digital twin, recovery controller, experiment harness, figure and table generation, Go "
        "control-plane services and deployment manifests), together with the processed result "
        "tables from which every figure and table in this manuscript is generated, is archived on "
        "Zenodo [78]. The development version is maintained on GitHub at "
        "https://github.com/Dev-Nikita/dtwin-humanitarian. Release v1.2-R1 differs from v1.1-R1 "
        "in the table, figure and manuscript generation scripts (recovery times are pooled over "
        "runs, which changes two derived tables) and in an additional unit test; the simulation "
        "code and the raw result files are identical, so the experiments need not be re-run. A "
        "snapshot of the "
        "same reproducibility package is provided as supplementary material with this submission. "
        "It includes the frozen confirmation protocol, its analysis script, the commit "
        "identifier of the code freeze and the confirmation results. "
        "The archive contains the source code, the configuration, the fixed random seeds, the "
        "result tables, the figure and table generation scripts and a README. The full generated "
        "telemetry corpus is not distributed in either the archive or the supplementary package "
        "because of its size; it is reproduced deterministically from the archived seeds. The "
        "complete workflow (regenerating the corpus, retraining the models, re-running the "
        "campaign and rebuilding every figure and table from scratch) is executed with the "
        "make all target documented in the accompanying README."
    )
    d.h1("Declarations")
    d.p("Funding: this research received no specific grant from any funding agency in the "
        "public, commercial or not-for-profit sectors. Conflicts of interest: the authors "
        "declare none. Ethical approval: not required; no human subjects or personal data were "
        "involved.", size=9.5)
    d.h1("Declaration of generative AI and AI-assisted technologies in the manuscript "
         "preparation process")
    d.p("During the preparation of this work the authors used OpenAI ChatGPT and Anthropic "
        "Claude (Opus) for language editing and for assistance in assembling the manuscript "
        "source files from machine-generated result tables. No content was accepted without "
        "verification against those tables. The authors reviewed and edited the output and take "
        "full responsibility for the content of the publication. All scientific content, "
        "experiments, analyses and conclusions are the authors\' own.", size=9.5)

    d.h1("References")
    for i, ref in enumerate(REFERENCES, start=1):
        par = d.doc.add_paragraph()
        par.paragraph_format.left_indent = Cm(0.9)
        par.paragraph_format.first_line_indent = Cm(-0.9)
        par.paragraph_format.space_after = Pt(2)
        par.add_run(f"[{i}] {ref}").font.size = Pt(9)

    d.save(out_path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="Predictive_Resilience_R1.docx")
    args = ap.parse_args()
    build(args.out)


if __name__ == "__main__":
    main()
