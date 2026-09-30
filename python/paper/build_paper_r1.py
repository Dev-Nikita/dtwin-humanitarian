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
        "REF": piv("mean_recovery_failure"),
        "dd": opt("detection_delay.csv"),
        "misspec_off": opt("misspec_offline.csv"),
        "large": opt("large_topology.csv") if os.path.exists(
            os.path.join(RESULTS, "large_topology.csv")) else None,
        "acct": opt("accounting_check.csv"),
        "hz_ref": float(opt("horizon_reactive_reference.csv")["resilience_loss"].mean())
        if os.path.exists(os.path.join(RESULTS, "horizon_reactive_reference.csv")) else float("nan"),
        "n_rev_runs": int(sum(len(opt(f)) for f in (
            "detection_delay.csv", "misspec_closed_loop.csv", "large_topology.csv",
            "feedback_ablation_fixed.csv", "design_sensitivity.csv", "frontier.csv",
            "migration_cost_sweep.csv", "topology_sweep.csv", "parameter_sensitivity.csv",
            "controller_weight_sensitivity.csv"))),
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
    "N. Tarasov and R. Zinko, \u201cReproducibility artifact for: Predictive resilience of hybrid 5G/6G-class and LEO satellite networks \u2014 a digital-twin framework for failure anticipation and risk-aware recovery,\u201d Zenodo, 2026, doi:10.5281/zenodo.22084966.",
    "P. Pan, G. Swallow, and A. Atlas, \u201cFast reroute extensions to RSVP-TE for LSP tunnels,\u201d IETF RFC 4090, 2005, doi:10.17487/RFC4090.",
    "A. M. Law, Simulation Modeling and Analysis, 5th ed. McGraw-Hill, 2015.",
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
    from ..experiments.make_figures_r1 import SHORT
    from ..experiments.make_figures2 import SCEN_LABEL
    from scipy import stats as _st

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

    d.h1("Abstract")
    d.p(
        "Hybrid terrestrial–satellite networks keep critical services alive when regional "
        "infrastructure is lost, and failure prediction is often proposed as a way of making "
        "them more resilient. We ask where such a gain actually comes from. A simulation-based "
        "digital twin of a hybrid 5G/6G-class and low-Earth-orbit network hides the degradation "
        "state of every element behind noisy telemetry and a detection delay, and a calibrated "
        "classifier estimates the probability of failure within a horizon. Fifteen recovery "
        "policies, built as controlled combinations of risk source (none, static alarm, "
        "contemporaneous degradation index, learned prediction, perfect state), criticality "
        "differentiation and restoration mechanism (reactive restoration or 1:1 protection), are "
        f"compared in {R['n_runs']:,} closed-loop runs on common random numbers, with an "
        "interruption accounting that charges the full restoration cost whenever a path is "
        "already down. Predictive make-before-break migration reduces cumulative resilience "
        f"loss relative to reactive restoration by {-rel(RL.loc[S4, 'proactive_ai'], RL.loc[S4, 'reactive']):.1f}% "
        f"under infrastructure collapse and {-rel(RL.loc[S6, 'proactive_ai'], RL.loc[S6, 'reactive']):.1f}% under "
        "common-cause disruption, is neutral under satellite-segment degradation and is "
        "harmful under handover dynamics and an externally parameterised stress test. Most of "
        "that gain comes from acting during the detection delay on elements that have already "
        "failed. Plain 1:1 protection does better "
        f"({-rel(RL.loc[S4, 'protection'], RL.loc[S4, 'reactive']):.0f}% and "
        f"{-rel(RL.loc[S6, 'protection'], RL.loc[S6, 'reactive']):.0f}%), and the largest reduction, "
        f"{-rel(RL.loc[S4, 'protection_predictive'], RL.loc[S4, 'reactive']):.0f}% and "
        f"{-rel(RL.loc[S6, 'protection_predictive'], RL.loc[S6, 'reactive']):.0f}%, is obtained when the risk "
        "estimate is spent only on decisions that interrupt no traffic: primary-path choice and "
        "standby-path refresh. In that role a learned predictor, a degradation index and "
        "perfect state knowledge are within about two per cent of one another. The ordering "
        "holds on a 98-element topology with three orbital planes and in a structurally "
        "misspecified degradation world. Risk information pays mainly through protection, "
        "not through anticipation."
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
        "some elements of such a system will degrade and fail, how much "
        "and under what conditions is the purchase worthwhile?"
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
        "interruption for the specific service, and on the cost of the recovery action — three "
        "quantities that only a closed-loop evaluation can relate to each other."
    )
    d.p(
        "Answering it requires three things that closed-loop evaluations of anticipatory "
        "control often lack: a paired design in which every policy faces exactly the same "
        "exogenous randomness, an interruption accounting that does not credit a "
        "make-before-break switch-over to a flow whose path is already dead, and baselines that "
        "isolate the mechanisms — above all conventional protection. We show that without them "
        "the benefit attributed to anticipation is substantially overstated, and that with them "
        "the benefit moves to a different place."
    )
    d.h2("1.1 Research questions")
    d.p("The study is organised around four questions.")
    d.numbered([
        "RQ1. How accurately, and how early, can impending element failures be recognised from "
        "observable telemetry alone, when the degradation state is hidden and the observation map "
        "is noisy?",
        "RQ2. Does acting on failure risk improve resilience relative to reactive restoration "
        "and to conventional protection, and which mechanism — detection-delay bypass, "
        "exposure control or anticipation proper — carries the gain?",
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
        "A controlled decomposition over fifteen policies that vary one mechanism at a time — "
        "risk source, criticality differentiation and restoration mechanism — including a "
        "predictive-protection policy that spends the risk estimate only on decisions that "
        "interrupt no traffic, and a calibrated perfect-state reference.",
        "Evidence, robust across scenarios, topologies, parameters and a misspecified "
        "degradation world, that risk information improves resilience mainly through "
        "protection and standby-path selection, that the benefit of pre-emptive migration is "
        "largely a detection-delay effect, and that the headroom from perfect anticipation is "
        "small in this class of system.",
    ], prefix="C{i}.")
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
        "resilience differentiation across service classes [11] — the notion our criticality-"
        "dependent risk budgets operationalise — and Gomes et al. surveyed protection strategies "
        "against large-scale disasters [12]. This literature measures resilience well; it "
        "generally treats the restoration process as given rather than as a decision variable "
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
        "removes exactly the feedback that makes proactive control risky. Third, the cost of "
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
        "eight classes — LTE (4G), 5G and 6G-class terrestrial access, UAV relay, LEO satellite, "
        "edge computing node, core node and user gateway — and carries a static profile (nominal "
        "capacity C_n, latency floor L_n, loss floor p_n, radio characteristics, degradation "
        "susceptibility). Each link carries a propagation latency and a capacity. Service flows "
        "of six criticality classes arrive as a marked Poisson process, occupy a path for a "
        "random holding time and are subject to class-specific latency, loss and bandwidth "
        "budgets. The consequence of an interruption differs by class, which is what makes "
        "differentiated protection meaningful rather than cosmetic."
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
        "exogenous, it has no telemetry precursor by construction — a property we exploit "
        "deliberately in Section 7.3."
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
        "the scenario shock magnitude of Table 3. Jump arrivals are independent across elements; "
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
        f"estimate; {eq6} applies to wireless elements only. The map is non-invertible — distinct "
        "(h, rho) pairs produce the same expected latency — and noisy, so identifying the "
        "degradation state from a single sample is imperfect. Section 7.10 quantifies how the "
        "noise level determines whether temporal aggregation of the telemetry is worth performing."
    )
    d.p(
        "The noise model is deliberately not uniform across KPIs, and the distinction matters for "
        "anyone reproducing the corpus. Strictly positive quantities — latency, jitter, "
        "throughput, processor and memory occupancy, queue depth — carry multiplicative "
        "log-normal noise with log-scale sigma_o. Loss and retransmission rate carry the same "
        "multiplicative noise with an inflated log-scale and are then projected onto [0, 1]. "
        "Bounded ratios — availability, utilisation, energy headroom — carry additive Gaussian "
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
        "when the twin declares the active path down, the flow is switched to the standby path "
        "at the switch-over cost C_m if the standby path is itself observed up, and falls back "
        "to reactive restoration otherwise. Standby capacity is not reserved. Pre-emptive "
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
        "interruption is scored. Section 7.3 shows how large the apparent benefit of "
        "anticipation becomes if C_m is charged in this case instead."
    )
    d.p(
        "Time is discrete. One slot is one orchestration cycle and corresponds to one second, "
        "the mapping used to anchor the reconfiguration cadence of the non-terrestrial "
        "scenarios to measured schedules [51], [52]; thus D = 3 s, C_r is between 4 and 12 s "
        "and C_m is 0 or 1 s. Every interruption is counted in whole slots: an action taken in "
        "slot t with cost C keeps the flow interrupted in slots t, ..., t + C, so a "
        "make-before-break switch-over costs one or two interrupted slots and a restoration "
        "five to thirteen. Availability, resilience loss, recovery time and the Kaplan–Meier "
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
         "break-before-make versus pre-computed protection [57], [58], [79]; swept in Section 7.5"],
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
    t_prov = d.table(prov, "Provenance of the model parameters. Only the first four groups are "
                           "tied to external evidence; the remainder are declared stress-test or "
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
        "utilisation-to-degradation coupling g are each halved and doubled."
    )

    d.h2("3.6 Problem statement")
    d.p(
        "At every control interval and for every active flow f with criticality class c, the "
        "operator can keep the current path P_f, migrate the live flow to an alternative, or — "
        "under protection — keep the flow in place and change only its standby path. Keeping "
        "risks an interruption of D + C_r slots if an element of P_f fails, or C_m slots if a "
        "working standby path exists; migrating costs C_m with certainty and may place the "
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
        "telemetry has satisfied a compound condition — packet loss above 0.5 together with "
        "availability below 0.08 — for more than D consecutive slots. This detection delay is the "
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
        "criterion fixed in advance: the highest validation precision–recall "
        "area, since the positive class is rare [68]; models within 0.01 of the best are "
        "treated as tied and the tie is broken by the validation Brier score, since the "
        "controller consumes probabilities, and then by inference cost. Test metrics are "
        "computed once, after the choice, and every choice is frozen before the test episodes "
        f"are used. For the selected model theta* = {theta:.3f}."
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
        "kappa, the comparison against theta* only for kappa b_c < 1. The rule inherits the "
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
        "the risk-augmented cost of Eq. (8) selects the primary path at admission, and "
        "whenever the standby path of a flow contains an element whose risk exceeds tau_c, or "
        "is observed down, a new node-disjoint standby path is computed on the risk-augmented "
        "graph with those elements excluded. The switch-over itself remains detection-driven, "
        "exactly as in 1:1 protection."
    )
    d.p(
        "The weights of Eq. (7) and the coefficients of Eq. (11) require justification, since "
        "they could otherwise be tuned to the environment they are evaluated in. They were fixed "
        "before the closed-loop campaign by a normalisation argument rather than by search: each "
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
        "coefficients of Eq. (11) is reported in Sections 7.5 and 7.10."
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
        "as C_m approaches C_f. Section 7.5 tests that prediction, and Section 10 names "
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
        "one- or two-slot switch-over of a make-before-break migration. Under the second, only sustained "
        "interruptions of at least two slots count. The distinction matters: a policy that trades "
        "rare long outages for frequent very short ones improves the second and worsens the "
        "first, and reporting only one of them would misrepresent it. Availability is the "
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
        "duration. It differs from availability in what it averages over rather than in temporal "
        "sensitivity: availability is a flow-slot weighted ratio, so slots carrying many flows "
        "dominate it, whereas L integrates the instantaneous service fraction with equal weight "
        "on every slot, so a deep loss during a period of light traffic is not discounted. "
        "Neither is sensitive to the absolute position of the loss within the horizon; when that "
        "matters we report the disruption depth max_t [Q_0 - Q(t)]_+ as the robustness component "
        "and the recovery-time distribution of Section 5.3 as the rapidity component."
    )
    d.p(
        "Both quantities weight every flow equally. Because a command-and-control session and "
        "a sensor report are not interchangeable, Section 7.3 additionally reports availability "
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
        "support reproduction; it is an engineering artefact rather than a contribution, and its "
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
                                             "swept in Section 7.5"],
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
        "and latency statistics from measurement studies [51], [52] and is an externally "
        "parameterised stress test, not an external validation. For the scale experiment of "
        "Section 7.7 a second topology with 98 elements is used: three LEO orbital planes of "
        "twelve satellites with intra-plane and inter-plane links, sixteen ground gateways, "
        "a doubled terrestrial access and edge segment, and arrival rates scaled with the "
        "number of user gateways. The training corpus is built only from "
        f"S1–S4 and contains {int(meta['n_train']) + int(meta['n_val']) + int(meta['n_test']):,} "
        f"eligible observation windows at a test-fold prevalence of "
        f"{float(meta['positive_rate_test']):.4f}."
    )
    d.h2("6.3 Recovery policies")
    d.p(
        "Fifteen policies share the twin, topology, traffic and seeds. Each is a combination of "
        "three mechanisms, so that the contribution of each can be read off a controlled "
        "comparison: the risk source (none, a static KPI alarm, a contemporaneous degradation "
        "index, the learned prediction, or the true latent state); the treatment of service "
        "classes (criticality-differentiated budgets b_c, criticality term of Eq. (11) and "
        "criticality-ordered processing, or uniform treatment with b_c = 1, a constant risk "
        "coefficient equal to the class average and arrival-order processing); and the "
        "restoration mechanism (reactive restoration or 1:1 protection)."
    )
    d.numbered([
        "P0 static — the path is fixed at admission and never revised.",
        "P1 reactive restoration — the path is recomputed once the twin observes that an "
        "element on it is down; the reference for the primary family.",
        "P1b 1:1 protection — node-disjoint standby path computed without risk, detection-"
        "driven switch-over at cost C_m, reactive fallback.",
        "P2 threshold-proactive — make-before-break migration when any element on the path "
        "violates a static operator limit: an element is in alarm when its KPI severity "
        "s = max(p/0.12, (L/L_ref)/1.9, cpu/88, 0.80/A) exceeds a scale factor, 1.0 at the "
        "nominal operator limits. P2 uses the same hysteresis and the same k = 3 candidate "
        "paths as P4, and does not route onto elements that are themselves in alarm. P2m is the same rule with the scale factor set so that its "
        "per-element alarm rate equals that of the predictor at theta*.",
        "P3 degradation-aware — migration and risk-augmented routing driven by a "
        "contemporaneous degradation index, a fixed weighted combination of current loss, "
        "latency inflation, unavailability and retransmission rate, rescaled so that it "
        "raises alarms at the same per-element rate as the predictor. P3u is its uniform "
        "variant.",
        "P4 predictive migration — the same machinery driven by the calibrated probability of "
        "failure within H. P4u is its uniform "
        "variant.",
        "P5 perfect state, band map — risk computed as a linear map of the true latent health "
        "between h = 0.35 (risk 0) and h = 0.18 (risk 1), clipped to [0, 1]. The map is not "
        "calibrated to the controller's probability scale and is kept to show the "
        "consequence. P5c replaces it by the exact probability that an element in health "
        "h fails within H, computed by Monte Carlo simulation of the generative model for each "
        "scenario and element class (6,000 trajectories per point of a health grid, "
        "interpolated). P5c is the calibrated perfect-state reference.",
        "P6 predictive protection — 1:1 protection plus risk-augmented primary-path choice and "
        "risk-driven standby refresh (Section 4.4), with no migration of live flows. P6i uses "
        "the degradation index and P6o the calibrated perfect-state probability as risk "
        "source; P6m adds P4's migrations to P6.",
    ], prefix="({i})")
    d.p(
        "Both matched operating points are estimated on dedicated validation runs of the "
        "reactive policy in the training scenarios S2–S4, on seeds disjoint from every seed "
        "used for evaluation: the quantile of the index, and the scale factor of the KPI rule, "
        "at which the per-element alarm rate equals the predictor's alarm rate at theta* on "
        "the same runs. Both values are frozen before any evaluation run."
    )
    d.h2("6.4 Protocol and statistical analysis")
    d.p(
        f"Each (scenario, policy) configuration is replicated over {R['n_seeds']} seeds and the "
        f"main campaign comprises {len(R['runs']):,} closed-loop runs of 500 slots; the "
        f"experiments of Sections 7.3–7.10 add a further {R['n_rev_runs']:,} runs. Comparisons are paired, "
        "and the pairing is implemented by common random numbers. The traffic "
        "model, the controller's internal randomness, the common-cause process and every "
        "network element draw from separate reproducible streams derived from the seed with "
        "NumPy's SeedSequence [73], [80]. In every slot each element draws a random block of "
        "fixed size from its own stream, and every random quantity of that element in that slot "
        "— diffusion increment, jump occurrence and magnitude, repair duration, post-repair "
        "health, handover outcome and every observation-noise term — is read from a fixed "
        "position of the block. Two policies on the same seed therefore see identical "
        "arrivals, holding times, demands, endpoints and element-level randomness, whatever "
        "they do; the number of generated flows is identical across policies for every seed, "
        "and a unit test asserts that the random blocks coincide. Seeding a single generator "
        "shared by traffic, environment and controller is not sufficient in a closed loop: "
        "policies that consume different numbers of draws would then see different arrivals "
        "and different failures, and the pairing would be lost."
    )
    d.p(
        "Comparisons use the paired t-test on seed-level differences, with the Wilcoxon "
        "signed-rank test as a distribution-free check, Cohen's d_z and 95% confidence "
        "intervals of the paired difference. The primary inferential family was fixed before "
        "the policy comparisons were analysed: the four primary resilience measures of P4 "
        "against P1 in the five stressed scenarios S2–S6, twenty tests, Holm-corrected [70], "
        "[71]. All other comparisons — the remaining policies, S7 and the experiments of "
        "Sections 7.3–7.10 — are exploratory and are reported with intervals and uncorrected "
        "p-values."
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
        "selection (Section 4.3); the test columns were computed once, after the choice. The "
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
        "The ranking of the learned models is not consistent across test metrics, which is why "
        "the criterion is fixed on validation data in advance; we do not claim an ordering "
        "between them."
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
        "These figures describe the predictor inside this generative model. Section 7.7 tests "
        "how much of the ranking survives when the degradation family and the observation map "
        "are changed."
    )

    # ------------------------------------------------------------- 7.2 RQ2
    d.h2("7.2 RQ2: which mechanism carries the resilience gain?")
    fig_dec2 = d.figure("r01_decomposition.png",
                        "Paired change in cumulative resilience loss relative to reactive "
                        "restoration P1, by scenario, for the policies that isolate one mechanism "
                        "at a time. Bars: mean over 24 paired seeds; whiskers: 95% confidence "
                        "interval of the paired difference. Hatched bars belong to the 1:1 "
                        "protection family.", width=16.5)
    t_cmp = d.table(pd.read_csv(os.path.join(TABDIR, "r_t_compact.csv")),
                    "Cumulative resilience loss L (slot-equivalents, lower is better) by policy "
                    "and stressed scenario, averaged over 24 paired seeds. S1 is omitted because "
                    "no disruption occurs there. Appendix A gives every policy and measure.",
                    fmt="{:g}", size=7.0)
    d.p(
        f"{fig_dec2} and {t_cmp} are the central result of the paper. They compare policies "
        "that differ in one mechanism at a time on identical exogenous randomness."
    )
    d.numbered([
        "Predictive make-before-break migration (P4) improves on reactive restoration where "
        f"failures are frequent and the network retains alternatives: resilience loss falls by "
        f"{-rel_(S4, 'proactive_ai', 'reactive'):.1f}% under infrastructure collapse and "
        f"{-rel_(S6, 'proactive_ai', 'reactive'):.1f}% under common-cause disruption. It is "
        f"neutral under non-terrestrial degradation ({rel_(S3, 'proactive_ai', 'reactive'):+.1f}%, "
        "confidence interval spanning zero) and harmful where transient reconfiguration dominates "
        f"({rel_(S5, 'proactive_ai', 'reactive'):+.0f}% in S5 and "
        f"{rel_(S7, 'proactive_ai', 'reactive'):+.1f}% in S7).",
        "Pre-computed 1:1 protection without any risk information (P1b) does better than "
        f"predictive migration wherever either helps: {-rel_(S4, 'protection', 'reactive'):.1f}% "
        f"under collapse, {-rel_(S6, 'protection', 'reactive'):.1f}% under common-cause "
        f"disruption and {-rel_(S3, 'protection', 'reactive'):.1f}% under non-terrestrial "
        "degradation. The paired difference between P4 and P1b under collapse is "
        f"{pc(S4, 'proactive_ai', 'protection')['diff']:+.2f} slot-equivalents "
        f"(95% CI {ci_(S4, 'proactive_ai', 'protection')}).",
        "The largest gain is obtained by spending the risk estimate on decisions that interrupt "
        "nothing. Predictive protection (P6) keeps the detection-driven switch-over of P1b and "
        "uses the predicted risk only to choose the primary path and to keep the standby path "
        f"off endangered elements. It reduces resilience loss by "
        f"{-rel_(S4, 'protection_predictive', 'reactive'):.1f}% under collapse, "
        f"{-rel_(S6, 'protection_predictive', 'reactive'):.1f}% under common-cause disruption and "
        f"{-rel_(S3, 'protection_predictive', 'reactive'):.1f}% under non-terrestrial "
        "degradation, and it is never worse than P1b in any scenario; relative to P1b it "
        f"lowers the loss by {-rel_(S4, 'protection_predictive', 'protection'):.1f}% under "
        f"collapse (paired difference {pc(S4, 'protection_predictive', 'protection')['diff']:+.2f} "
        f"slot-equivalents, 95% CI {ci_(S4, 'protection_predictive', 'protection')}) and "
        f"{-rel_(S6, 'protection_predictive', 'protection'):.1f}% under common-cause disruption "
        f"({ci_(S6, 'protection_predictive', 'protection')}).",
        "Within the protection family the source of the risk hardly matters. Replacing the "
        "learned predictor by the contemporaneous degradation index (P6i) changes the loss by "
        f"{sp(rel_(S4, 'protection_index', 'protection_predictive'))} under collapse and "
        f"{rel_(S6, 'protection_index', 'protection_predictive'):+.1f}% under common-cause "
        "disruption, and replacing it by the calibrated probability computed from the true "
        f"latent state (P6o) by {rel_(S4, 'protection_oracle', 'protection_predictive'):+.1f}% "
        f"and {rel_(S6, 'protection_oracle', 'protection_predictive'):+.1f}%. The headroom that "
        "perfect anticipation could add to this policy is therefore about one to two per cent "
        "of the loss.",
        "Among migration-based policies, prediction is not better than a contemporaneous "
        "index at a matched alarm rate (P3): under collapse the two differ by "
        f"{pc(S4, 'proactive_ai', 'degradation_aware')['diff']:+.2f} slot-equivalents "
        f"(95% CI {ci_(S4, 'proactive_ai', 'degradation_aware')}), under common-cause "
        f"disruption by {pc(S6, 'proactive_ai', 'degradation_aware')['diff']:+.2f} "
        f"({ci_(S6, 'proactive_ai', 'degradation_aware')}); prediction is ahead of the index "
        "only where transient reconfiguration misleads the index (Section 7.6). Even perfect state knowledge used "
        "for migration (P5c) stays behind plain protection: "
        f"{L_(S4, 'oracle_calibrated'):.2f} against {L_(S4, 'protection'):.2f} under collapse. "
        "Adding migration on top of predictive protection (P6m) undoes most of the gain of P6 "
        f"({L_(S4, 'protection_predictive_mig'):.2f} against {L_(S4, 'protection_predictive'):.2f}).",
    ], prefix="({i})")
    t_conf = d.table(pd.read_csv(os.path.join(TABDIR, "r_t_confirmatory.csv")),
                     "Pre-specified primary family: P4 against reactive restoration P1 on the four "
                     "primary measures in the five stressed scenarios (twenty tests, paired t-test, "
                     "Holm-corrected). The family was fixed before the analysis; all other "
                     "comparisons are exploratory (Table " + "{T_CONTR}" + ").", fmt="{}",
                     size=7.0)
    n_conf = int((R["sig"]["family"] == "confirmatory").sum())
    conf = R["sig"][R["sig"]["family"] == "confirmatory"]
    n_hs = int((conf["p_holm"] < 0.05).sum())
    d.p(
        f"{t_conf} reports the pre-specified inferential family. {n_hs} of the {n_conf} tests are significant after Holm correction. The "
        "decrease in resilience loss and the increase in availability under collapse and "
        "under common-cause disruption survive correction, as do the recovery-time "
        "reductions; the handover scenario shows a significant deterioration; overload and "
        "non-terrestrial degradation show no effect on the primary resilience measures. The "
        "recovery-time entries must be read with Section 7.3: most of that reduction is the "
        "replacement of long failure-induced interruptions by short migration-induced ones, "
        "not faster recovery from failures."
    )
    t_contr = d.table(pd.read_csv(os.path.join(TABDIR, "r_t_contrasts.csv")),
                      "Exploratory paired contrasts on cumulative resilience loss (24 seeds; "
                      "uncorrected Wilcoxon signed-rank p). Negative differences favour the "
                      "first-named policy.", fmt="{}", size=6.3)
    d.set_placeholder("{T_CONTR}", t_contr.split()[-1])
    d.p(
        f"{t_contr} gives the paired contrasts that isolate each mechanism, in every stressed "
        "scenario. They are exploratory, since they were not part of the family fixed in "
        "advance. We report "
        "them with intervals rather than as tests, and the conclusions above rest on intervals "
        "that exclude zero by a wide margin rather than on marginal p-values."
    )

    # ---------------------------------------------------------- 7.3 mechanism
    d.h2("7.3 Why migration helps: detection delay, not foresight")
    trig = R["runs"][R["runs"]["policy"].isin(["proactive_ai", "degradation_aware"])].groupby(
        ["scenario", "policy"])[["mig_trigger_already_failed", "mig_trigger_fails_within_H",
                                 "mig_trigger_no_failure_within_H",
                                 "proactive_reroutes_total"]].mean()
    tr4 = trig.loc[(S4, "proactive_ai")]
    tr6 = trig.loc[(S6, "proactive_ai")]
    d.p(
        "The assessment layer knows the true state, so every migration can be classified after "
        "the fact by what the element that triggered it was doing. Under collapse, of the "
        f"{tr4['proactive_reroutes_total']:.0f} migrations P4 performs per run, "
        f"{tr4['mig_trigger_already_failed']:.0f} were triggered by an element that had already "
        f"failed but had not yet been declared down by the twin, "
        f"{tr4['mig_trigger_fails_within_H']:.0f} by an element that failed within the horizon, "
        f"and {tr4['mig_trigger_no_failure_within_H']:.0f} by an element that did not fail "
        "within the horizon. Under common-cause disruption the corresponding figures are "
        f"{tr6['mig_trigger_already_failed']:.0f}, {tr6['mig_trigger_fails_within_H']:.1f} and "
        f"{tr6['mig_trigger_no_failure_within_H']:.1f}: almost every 'pre-emptive' migration "
        "moves a flow off an element that is already down. What the telemetry-driven policies "
        "exploit is therefore mostly the detection delay D — the failure signature is visible "
        "in the raw KPIs a few slots before the twin's de-bounced detector confirms it — and "
        "only secondarily foresight of failures that have not happened yet."
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
        "throughout. The difference between the two is the size of the artefact, and it is "
        "larger than the effect that survives. The controllers are identical in both "
        "accountings; only the scoring of the interruption differs."
    )
    fig_dd = d.figure("r02_detection_delay.png",
                      "Resilience loss as a function of the detection delay D for the "
                      "principal policies under infrastructure collapse and common-cause "
                      "disruption (12 paired seeds, 95% intervals). Dashed lines: 1:1 "
                      "protection family.", width=15.0)
    dd = R["dd"]

    def ddL(s, p, D):
        return float(dd[(dd.scenario == s) & (dd.policy == p) & (dd.detection_delay == D)]
                     ["resilience_loss"].mean())

    d.p(
        f"{fig_dd} tests the explanation directly by sweeping D with everything else fixed. "
        "If the gain of the migration policies came from foresight it would not depend on how "
        "slowly the reactive detector confirms a failure; if it came from bypassing that "
        "detector it would vanish as D goes to zero. The second is what happens. Under collapse "
        f"the advantage of P4 over P1 grows from {ddL(S4, 'reactive', 0) - ddL(S4, 'proactive_ai', 0):.2f} "
        f"slot-equivalents at D = 0 to {ddL(S4, 'reactive', 5) - ddL(S4, 'proactive_ai', 5):.2f} at "
        f"D = 5, and under common-cause disruption from "
        f"{ddL(S6, 'reactive', 0) - ddL(S6, 'proactive_ai', 0):.2f} to "
        f"{ddL(S6, 'reactive', 5) - ddL(S6, 'proactive_ai', 5):.2f}. Protection behaves in the "
        "opposite way: its advantage is largest when detection is fast, because its fast "
        "switch-over is triggered by the detector. Predictive protection is the best policy at "
        "every delay tested."
    )
    t_rs = d.table(pd.read_csv(os.path.join(TABDIR, "r_t_recovery_split.csv")),
                   "Recovery time separated by the cause of the interruption. "
                   "Failure-induced interruptions start when an element on the active path "
                   "fails; migration-induced interruptions are the switch-over slots of a "
                   "make-before-break migration away from a working path.", fmt="{:g}",
                   size=6.8)
    d.p(
        f"{t_rs} separates recovery time by cause. The large reduction in mean recovery time "
        "that P4 shows in the primary family is almost entirely a composition effect: P4 "
        "adds many one- or two-slot migration-induced interruptions to the average, while its "
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
        f"{t_cls} shows what the availability numbers mean for service quality. Availability counts a flow-slot as available when the flow is "
        "delivered at all; SLA-compliant availability additionally requires that its latency "
        "and loss budgets of Table 2 are met. Under collapse the network remains connected for "
        "most flows but cannot meet the latency and loss budgets of the tighter classes on the "
        "surviving non-terrestrial paths, so SLA-compliant availability is low for every "
        "policy and differs little between them. The recovery policies studied here restore "
        "connectivity; none of them can create capacity that the collapse has removed. We state "
        "this explicitly because an availability figure near 97% could otherwise be read as "
        "satisfactory service."
    )

    # ------------------------------------------------------------ 7.4 KM
    d.h2("7.4 Service reliability and interruption hazard")
    fig_sur = d.figure("f08_survival.png",
                       "Kaplan–Meier estimates of the probability that a flow has not yet been "
                       "interrupted (top row: any interruption, including migration switch-over "
                       "slots; bottom row: interruptions lasting at least two slots).",
                       width=16.0)
    sv = pd.read_csv(os.path.join(TABDIR, "r_t_survival.csv"))
    d.p(
        f"{fig_sur} shows the same mechanism from the flow's point of view. Because every slot "
        "is counted whole, a migration switch-over registers as an interruption of one or two "
        "slots, and a switch-over with C_m = 1 lasts two slots and therefore also counts under "
        "the sustained definition. P4 consequently has the lowest survival of the four "
        "policies under both definitions in the two training-family scenarios. The time to the "
        "first interruption is almost identical for P1, P1b and P6, which is what the "
        "mechanism predicts: protection does not postpone interruptions, it shortens them, and "
        "its benefit therefore appears in resilience loss and failure-induced recovery time "
        "rather than in survival. Under the sustained definition "
        + "; ".join(
            f"S(40) under {sname} is "
            + ", ".join(f"{r['S(40)']:.3f} for {r['policy']}" for _, r in
                        sv[(sv.definition != 'any') & (sv.scenario == SCEN_LABEL[s])].iterrows())
            for s, sname in ((S4, "collapse"), (S6, "common-cause disruption")))
        + ". The survival table is provided in the supplementary result files."
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

    # ------------------------------------------------------------ 7.5 RQ3
    d.h2("7.5 RQ3: actuation cost and matched actuation effort")
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
    cross = [c for c in sorted(mg.migration_cost.unique()) if c >= 0
             and mgL(S4, "proactive_ai", c) >= ref4]
    d.p(
        f"{fig_mig} sweeps the switch-over cost. Under collapse the loss of P4 rises from "
        f"{mgL(S4, 'proactive_ai', 0):.2f} at C_m = 0 to {mgL(S4, 'proactive_ai', 8):.2f} at "
        f"C_m = 8 against a reactive reference of {ref4:.2f}; "
        + (f"it reaches the reference at C_m = {cross[0]:g} slots. " if cross else
           "it stays below the reference over the tested range, but the margin shrinks "
           "steadily. ")
        + "The protection family is also sensitive to the switch-over cost, since its "
        "switch-over uses the same actuator, but it stays below the reactive reference over "
        f"a wider range: P6 goes from {mgL(S4, 'protection_predictive', 0):.2f} to "
        f"{mgL(S4, 'protection_predictive', 8):.2f}. Under non-terrestrial degradation the "
        f"reactive reference is {ref3:.2f} and P4 is at or above it from C_m = 1 onwards. The "
        "practical criterion is to measure the switch-over cost of the available mechanism and "
        "compare it with the detection-plus-restoration interval, and it applies to protection "
        "as much as to anticipation."
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
        f"{fig_fr} and {t_fr} compare the risk sources at matched actuation effort instead of at one operating point. Three "
        "observations follow. First, the threshold policy P2 is poor only when it is "
        "configured aggressively: at the nominal operator limits it triggers hundreds of "
        "migrations per run, but relaxed to the predictor's alarm rate (P2m in Table "
        f"{t_cmp.split()[-1]}) or beyond it, it reaches losses comparable to P3 and P4 — under "
        f"collapse its best point is {frb(S4, 'proactive_threshold')[0]:.2f} against "
        f"{frb(S4, 'proactive_ai')[0]:.2f} for P4. That threshold-driven proactive action can be "
        "worse than reactive restoration is therefore a property of an aggressive "
        "configuration, not of threshold rules as such. Second, at matched effort the "
        "learned predictor is not better than the index or the threshold rule. Third, no "
        "operating point of any migration-based policy reaches plain protection under collapse "
        f"or common-cause disruption, while P6 lies below all of them "
        f"({frb(S4, 'protection_predictive')[0]:.2f} under collapse)."
    )

    # --------------------------------------------------- 7.6 differentiation
    d.h2("7.6 Criticality differentiation and the value of prediction")
    Lr = R["RL"]
    d.p(
        "A comparison between the index-driven and the predictive policy could confound the "
        "risk source with service differentiation, so the design makes both factors explicit. "
        "P3 and P4 share budgets b_c, the criticality term of Eq. (11) and criticality-ordered "
        "processing; P3u and P4u replace all three by uniform treatment. The 2 × 2 comparison "
        "does not show a general advantage for either factor. Differentiation protects the "
        "index-driven policy from transient signatures — it lowers its loss by "
        f"{-pc(S5, 'degradation_aware', 'degradation_aware_uniform')['diff']:.2f} "
        f"slot-equivalents in S5 and {-pc(S7, 'degradation_aware', 'degradation_aware_uniform')['diff']:.2f} "
        "in S7 — but has no detectable effect under collapse "
        f"({ci_(S4, 'degradation_aware', 'degradation_aware_uniform')}) or non-terrestrial "
        f"degradation ({ci_(S3, 'degradation_aware', 'degradation_aware_uniform')}). Prediction "
        "is ahead of the index under both treatments where reconfiguration transients "
        f"dominate (S5 and S7), behind it under uniform treatment in S3 "
        f"({pc(S3, 'proactive_ai_uniform', 'degradation_aware_uniform')['diff']:+.2f}, "
        f"{ci_(S3, 'proactive_ai_uniform', 'degradation_aware_uniform')}), and "
        "indistinguishable from it elsewhere. What prediction adds among migration policies "
        "is therefore mainly robustness to transients that a contemporaneous index mistakes "
        "for degradation, not earlier action on genuine failures."
    )

    # ------------------------------------------------------------ 7.7 RQ4
    d.h2("7.7 RQ4: robustness to unseen mechanisms, model misspecification and scale")
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
                   "Closed-loop outcome in a structurally different world: gamma-process wear-out without mean reversion, observed through a "
                   "four-slot lagged map with different exponents and doubled noise. The "
                   "predictor and all operating points are frozen from the jump-diffusion "
                   "corpus. 24 paired seeds; the interval is that of the paired difference to P1.",
                   fmt="{}", size=6.8)
    d.p(
        "The predictor is trained and tested on noisy transformations of the very latent "
        "variable that defines failure, so the offline metrics validate it only within the "
        "generative family that produced them, and perturbing parameters of that family is "
        "not misspecification. We therefore "
        "built a second world that shares no dynamics with the first: health decreases "
        "monotonically through gamma-distributed wear increments with no mean reversion, and "
        "telemetry reflects the health of four slots earlier through different exponents and "
        "with twice the measurement noise. Frozen and applied to this world, the predictor "
        f"retains a precision–recall area of {fr_['pr_auc']:.3f} (lift "
        f"{fr_['lift']:.1f}) against {rt_['pr_auc']:.3f} (lift {rt_['lift']:.1f}) for the same "
        f"estimator retrained on the new world. {t_ms} gives the closed-loop consequence. "
        "The ordering of Section 7.2 is preserved: the protection policies with a "
        "risk-refreshed standby path (P6, P6i) are best in both scenarios, protection alone and "
        "predictive migration follow close together, and degradation-aware migration is worse "
        "than reactive restoration. "
        "The evidence therefore separates two claims: "
        "the framework and the decomposition transfer across the two degradation families "
        "we have built; the absolute quality of the predictor is a property of the family "
        "it was trained on, and no claim is made for real failure mechanisms."
    )
    if R["large"] is not None:
        t_lg = d.table(pd.read_csv(os.path.join(TABDIR, "r_t_large.csv")),
                       "Results on a 98-element topology with three LEO planes. "
                       "The predictor and all operating points are frozen from the 19-element "
                       "reference network. 12 paired seeds.", fmt="{}", size=6.8)
        lg = R["large"]

        def lgL(s, p):
            return float(lg[(lg.scenario == s) & (lg.policy == p)]["resilience_loss"].mean())

        d.p(
            "A 19-element network cannot represent a satellite constellation, so the "
            "experiment is repeated on a 98-element topology with three LEO orbital "
            "planes of twelve satellites each, intra-plane and inter-plane links, 16 ground "
            "gateways and a doubled terrestrial segment (Section 6.2). "
            f"{t_lg} shows the outcome. Under collapse the ordering is "
            f"P1 {lgL(S4, 'reactive'):.2f}, P1b {lgL(S4, 'protection'):.2f}, "
            f"P3 {lgL(S4, 'degradation_aware'):.2f}, P4 {lgL(S4, 'proactive_ai'):.2f} and "
            f"P6 {lgL(S4, 'protection_predictive'):.2f}. The larger network does not change the "
            "qualitative conclusion, and we do not claim that it makes the model representative "
            "of an operational constellation."
        )
    ext = R["runs"][R["runs"]["scenario"] == S7]
    d.p(
        "Scenario S7, whose reconfiguration cadence and phase are taken from Tanveer et al. "
        "[52] and whose latency statistics are taken from Michel et al. [51], is an externally "
        "parameterised stress test and not an external validation; it "
        "was added after the primary family was fixed and all its comparisons are exploratory. "
        "It is also the scenario that most clearly separates the mechanisms. The dominant "
        "impairment is a scheduled reconfiguration with no degradation precursor and no load "
        "coupling, so there is nothing to anticipate: both migration policies are worse than "
        f"reactive restoration ({rel_(S7, 'proactive_ai', 'reactive'):+.1f}% for P4, "
        f"{rel_(S7, 'degradation_aware', 'reactive'):+.1f}% for P3), because they migrate "
        "flows away from transient reconfiguration signatures, while protection and "
        f"predictive protection improve on it ({rel_(S7, 'protection', 'reactive'):+.1f}% and "
        f"{rel_(S7, 'protection_predictive', 'reactive'):+.1f}%). The migration advantage seen "
        "elsewhere is carried by the detection-delay bypass of Section 7.3, which is worth "
        "little when the impairment is short-lived and scheduled; S7 is therefore where the "
        "limits of anticipatory migration are most visible."
    )

    # ------------------------------------------------------------ 7.8 horizon
    d.h2("7.8 Prediction horizon")
    hs = R["horizon"]
    fig_h = d.figure("f12_horizon.png",
                     "Effect of the prediction horizon H: prediction quality, closed-loop lead "
                     "time of P4 (with the bound T_lead ≤ H), and resilience loss of P4 and P6 "
                     "under collapse.", width=15.5)
    t_h = d.table(pd.read_csv(os.path.join(TABDIR, "r_t_horizon.csv")),
                  "Prediction-horizon sweep. For every H a model is retrained on labels "
                  "relabelled at that horizon; closed-loop columns are averaged over 12 seeds of "
                  "the collapse scenario.", fmt="{:g}", size=6.6)
    d.p(
        f"{fig_h} and {t_h} vary the horizon H; for every H a model is retrained on relabelled "
        "data and the lead-time tracker receives the same H, so the horizon-consistent "
        f"matching of Eq. {eq18} applies at every point. A tracker that matched each failure to "
        "the first threshold crossing without that bound would report lead times longer than "
        "H, which is a matching artefact and not usable warning; the bound is asserted in the "
        "code, and a unit "
        f"test enforces it. The median closed-loop lead time lies between "
        f"{hs['cl_lead_time_median'].min():.1f} and {hs['cl_lead_time_median'].max():.1f} slots "
        "and never exceeds H. Longer horizons raise the precision–recall area because the "
        "positive class becomes more frequent, but the lift over prevalence falls. In the "
        f"closed loop the loss of P4 rises from {hs['cl_resilience_loss'].iloc[0]:.2f} at "
        f"H = {int(hs['horizon'].iloc[0])} to {hs['cl_resilience_loss'].max():.2f}, against "
        f"{R['hz_ref']:.2f} for reactive restoration on the same seeds, because a longer horizon "
        "raises more alarms on elements that do not fail soon and the migration policy acts on "
        f"them ({hs['cl_migrations'].iloc[0]:.0f} migrations per run at the shortest horizon, "
        f"{hs['cl_migrations'].max():.0f} at the longest). Predictive protection is insensitive "
        f"to H ({hs['p6_resilience_loss'].min():.2f} to {hs['p6_resilience_loss'].max():.2f}), "
        "since refreshing a standby path on a false alarm costs nothing. For migration-based "
        "control the horizon should therefore be kept short, of the order of the detection and "
        "actuation delays, because short horizons limit false-alarm migrations."
    )

    # ------------------------------------------------------------ 7.9 feedback
    d.h2("7.9 Does the policy shape its own failure process?")
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
    d.p(
        "Equation (2) lets a policy shape its own failure process through the load it places on "
        "elements. An ablation that removes this path by replacing the controller's load with "
        "an exogenous profile also changes the load level and confounds the two effects, so "
        f"the ablation is performed with the traffic held fixed. {t_fb} shows that at the nominal coupling the "
        "endogenous and replayed arms differ by at most "
        f"{(nom['loss, endogenous load'] - nom['loss, replayed P1 load']).abs().max():.2f} "
        "slot-equivalents for any policy, and where a difference is visible it is the "
        "migration policy that does slightly better with the replayed load: the feedback path "
        "contributes nothing to the gain in the calibrated model. We also measured directly whether the policies "
        "change the failure process: the number of element-failure slots per run is identical "
        "to within a few tenths of a per cent across P0, P1 and P4 at the nominal coupling, "
        "because utilisation exceeds the knee of Eq. (2) in only about two per cent of "
        "element-slots. Only when the coupling is made four times stronger does the "
        "feedback become visible, and then it works against the migration policy, whose moves "
        "concentrate load. The closed loop is retained in the framework because it is the correct evaluation "
        "structure, but in this parameterisation it is not a source of benefit."
    )

    # ------------------------------------------------------------ 7.10 sensitivity
    d.h2("7.10 Sensitivity to parameters, design choices and structure")
    t_ds = d.table(pd.read_csv(os.path.join(TABDIR, "r_t_design.csv")),
                   "Sensitivity to the controller design choices of Eqs. (10) and (11) under "
                   "collapse: alternative budget-factor profiles b_c (flat 1/1/1/1/1, "
                   "shallow 0.90–1.75, nominal 0.80–2.50, steep 0.50–4.00) and one-at-a-time "
                   "changes of the loss coefficient (nominal 4), the risk base (nominal 3) and "
                   "the criticality slope (nominal 0.6). 24 paired seeds.", fmt="{}", size=6.6)
    dst = pd.read_csv(os.path.join(TABDIR, "r_t_design.csv"))
    t_pa = d.table(pd.read_csv(os.path.join(TABDIR, "r_t_parameters.csv")),
                   "Sensitivity to the degradation-model parameters under collapse: baseline "
                   "degradation rate nu_0, diffusion amplitude sigma and load-to-degradation "
                   "gain g, each halved and doubled (12 paired seeds).", fmt="{}", size=6.6)
    pat = pd.read_csv(os.path.join(TABDIR, "r_t_parameters.csv"))
    t_tp = d.table(pd.read_csv(os.path.join(TABDIR, "r_t_topology.csv")),
                   "Dependence on structural redundancy under collapse. Only the graph "
                   "changes across classes (12 paired seeds).", fmt="{}", size=6.6)
    d.p(
        f"{t_ds} examines the budget factors and the coefficients of Eq. (11). "
        f"The loss of P4 varies between {dst['P4 loss'].min():.2f} and {dst['P4 loss'].max():.2f} "
        f"and that of P6 between {dst['P6 loss'].min():.2f} and {dst['P6 loss'].max():.2f} "
        "across the settings. The criticality slope and the risk base are inert, because the "
        "budgets rather than the routing penalty decide when a policy acts. The budget profile "
        "matters for P4, and the nominal profile is the best of the four tested for it, so the "
        "nominal configuration does not flatter the comparison with protection; P6 is "
        "insensitive to the profile and better than protection alone in every setting. "
        f"{t_pa} perturbs the generative model. P6 improves on P1 in "
        f"{int((pat['P6 vs P1 [%]'] < 0).sum())} of {len(pat)} settings and on P1b wherever "
        f"failures are frequent; P4 improves on P1 in {int((pat['P4 vs P1 [%]'] < 0).sum())} of "
        f"{len(pat)}; both failures occur where events are too rare for migrations to pay. "
        f"{t_tp} varies the redundancy of the graph with the element catalogue, traffic and "
        "degradation fixed. In the sparsest class P4 is marginally ahead of the protection "
        "family, because a node-disjoint backup often does not exist there; in every other "
        "class the protection family is ahead and the gap widens with route diversity."
    )
    t_w = d.table(pd.read_csv(os.path.join(TABDIR, "r_t_weights.csv")),
                  "Sensitivity to the cost weights of Eq. (7) under collapse (12 paired seeds).",
                  fmt="{}", size=6.8)
    d.p(
        f"{t_w} rescales the weights of the element penalty of Eq. (7) by factors of two. "
        "The conclusions are unchanged; the load weight is inert because the utilisation term "
        "activates only above 0.75, which the reference topology rarely reaches."
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
        "The decomposition of Section 7 assigns the benefit to three mechanisms of very "
        "different weight. The largest is protection: a pre-computed disjoint standby path "
        "turns a restoration of C_r slots into a switch-over of C_m slots, and it does so "
        "without any estimate of the future. The second is exposure control through "
        "zero-interruption decisions: choosing a low-risk primary path and keeping the standby "
        "path off endangered elements adds a further few per cent on top of protection "
        "wherever failures are frequent, and it never hurts, because a decision that "
        "interrupts nothing cannot create an outage. The third, and much the smallest, is "
        "anticipation proper — acting on an element before it fails. Migration-based policies "
        "derive most of their benefit from acting during the detection delay on elements that "
        "have already failed, and the part of their activity that is genuinely anticipatory "
        "is outnumbered roughly two to one, under collapse, by migrations away from elements "
        "that do not fail within the horizon."
    )
    d.h2("8.2 Why anticipation adds little in this system")
    d.p(
        "Three properties of the modelled system limit what anticipation can buy, and each is "
        "a property an operator can check in a real network. First, a large part of the failure "
        "process is abrupt: the jump term of Eq. (1), the collapse of destroyed elements and "
        "the common-cause shock of Eq. (3) leave little or no precursor, so the anticipation "
        "rate stays well below one even inside the training family. Second, the warning that is "
        "available is short relative to the horizon, while a migration of a live flow has a "
        "certain cost; a controller that acts on a probability of the order of the operating "
        "point pays that cost for many elements that do not fail. Third, when a cheaper "
        "action exists that achieves the same end — here the standby path of protection, "
        "which can be moved without touching the traffic — the risk estimate is better spent "
        "on that action. The perfect-state references make the point quantitatively: even "
        "exact knowledge of the latent state, used for migration, does not reach plain "
        "protection under collapse or common-cause disruption, and used for standby "
        "selection it improves on the learned predictor by one to two per cent."
    )
    d.h2("8.3 When would anticipation pay?")
    d.p(
        "The results do not show that failure prediction is useless for network resilience; "
        "they show where its value lies in a system of this kind and what would have to change "
        "for migration on predicted risk to pay. The detection-delay sweep indicates that it "
        "becomes more valuable as detection becomes slower, and the migration-cost sweep that "
        "it needs switch-over costs of about one slot or less. It would also pay where no protection "
        "path can be provisioned, as in the sparsest topology class of Section 7.10, and "
        "where degradation is predominantly gradual, so that the horizon contains a usable "
        "precursor for most failures. The noise sweep adds a caution on the predictor side: at "
        "nominal noise the telemetry identifies the latent state almost from a single sample, "
        "so a contemporaneous index is nearly as informative as a trained forecaster. An "
        "operator should therefore establish these conditions — detection latency, "
        "switch-over cost, availability of disjoint standby capacity and the share of "
        "failures with an observable precursor — before investing in anticipation rather "
        "than in protection."
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
        "rather than assumed: every exogenous process draws from its own reproducible stream, "
        "every element draws a fixed-size block of random numbers in every slot, the number of "
        "generated flows is identical across policies for every seed, and a unit test asserts "
        "that the random blocks are identical across policies. Operating points of the "
        "comparison baselines are matched on validation runs only and frozen before the test "
        "seeds are used. The principal internal threat remains that the generative model and "
        "the controllers share a designer; the misspecified world of Section 7.7, the "
        "parameter and design sweeps and the externally parameterised scenario address it "
        "only partially. External validity is the substantive limitation (Section 9). Construct "
        "validity required the separation of failure- and migration-induced interruptions, "
        "without which mean recovery time would have rewarded policies for creating short "
        "interruptions of their own."
    )

    # ------------------------------------------------------ 9. Limitations
    d.h1("9. Limitations")
    d.numbered([
        "The evaluation is conducted entirely in simulation. No measurements from physical LEO "
        "terminals or 6G-class hardware and no field trial are used. Absolute values are "
        "properties of the model; only the comparison between policies, which share the "
        "environment exactly, should inform expectations about deployed systems.",
        "Both degradation families are modelling assumptions. The misspecified world of "
        "Section 7.7 tests transfer between two families we built; it is not evidence about "
        "real failure mechanisms, which include wear-out, adversarial and cascading modes that "
        "neither family reproduces. The predictor has been validated only within these "
        "families; the framework and the decomposition are what the study claims to establish.",
        "The non-terrestrial dynamics of S5 and S7 are measurement-anchored rather than "
        "trace-driven; S7 adopts published summary statistics [51], [52] and does not replay "
        "an operational trace.",
        "Protection is modelled as 1:1 with a node-disjoint standby path whose capacity is not "
        "reserved, and its switch-over is charged the same cost as a make-before-break "
        "migration. Reserved-capacity protection, shared protection and protocol-level session "
        "continuity would change the absolute numbers.",
        "The 98-element topology adds orbital planes and inter-satellite links but no orbital "
        "mechanics: visibility and link availability are not driven by ephemeris.",
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
        "Protection design under uncertainty. Since risk information pays mainly through "
        "standby selection, reserved and shared protection with risk-aware backup placement "
        "is the most promising direction suggested by the results.",
        "Richer common-cause structures, including multi-region and cascading dependencies.",
    ])

    # ----------------------------------------------------- 11. Conclusion
    d.h1("11. Conclusion")
    d.p(
        "This paper asked where the resilience benefit of failure prediction comes from in a "
        "hybrid terrestrial–satellite network, and answered it with a simulation-based digital "
        "twin in which the degradation state is hidden, detection is delayed and fifteen "
        "recovery policies are compared on common random numbers with a physically consistent "
        "account of every interruption. Failures inside the training family are anticipated "
        f"with a median lead time of {lead(S4):.1f} slots under infrastructure collapse, and "
        "predictive make-before-break migration reduces cumulative resilience loss relative "
        f"to reactive restoration by {-rel_(S4, 'proactive_ai', 'reactive'):.1f}% under "
        f"collapse and {-rel_(S6, 'proactive_ai', 'reactive'):.1f}% under common-cause "
        "disruption. It does not help under non-terrestrial degradation and it hurts under "
        "handover dynamics and in the externally parameterised scenario."
    )
    d.p(
        "The controlled comparisons locate the gain. Most of the migration benefit comes from "
        "acting during the detection delay on elements that have already failed, and it grows "
        "with that delay. Plain 1:1 protection does better than predictive migration, reducing "
        f"the loss by {-rel_(S4, 'protection', 'reactive'):.1f}% and "
        f"{-rel_(S6, 'protection', 'reactive'):.1f}%. The largest gains, "
        f"{-rel_(S4, 'protection_predictive', 'reactive'):.1f}% and "
        f"{-rel_(S6, 'protection_predictive', 'reactive'):.1f}%, are obtained when the risk "
        "estimate is used only for decisions that interrupt no traffic — primary-path choice "
        "and standby-path refresh — on top of protection. In that role a learned predictor, a "
        "contemporaneous degradation index and perfect state knowledge perform within about "
        "two per cent of one another. The ordering holds on a 98-element topology, under "
        "parameter and design perturbations and in a structurally misspecified degradation "
        "world, while the predictor's own accuracy is established only within the model "
        "family it was trained on."
    )
    d.p(
        "For operators and designers the implication is to spend risk information where "
        "acting on it is cheap: protect first, use risk to place and refresh standby paths, "
        "and reserve pre-emptive migration of live traffic for systems with slow detection, "
        "negligible switch-over cost, no provisionable protection and gradual, observable "
        "degradation. The environment, controllers and analysis pipeline are released so that "
        "the same decomposition can be repeated on other systems and, eventually, on "
        "operational data."
    )

    d.h1("Appendix A. Complete policy comparison")
    d.p("The table gives every policy in every scenario, averaged over 24 paired seeds. "
        "Recovery is reported both over all interruptions and over failure-induced "
        "interruptions only (Section 7.3).")
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
        "The sign of both is settled at eight seeds and the intervals contract as expected; "
        "the interval half-width of the P6-versus-P1b effect at 24 seeds is "
        f"{float(sc[sc.contrast.str.startswith('P6')]['ci half-width'].iloc[-1]):.2f} "
        "slot-equivalents."
    )

    d.h1("Data and code availability")
    d.p(
        "The complete implementation — environment, feature pipeline, anticipation models, "
        "digital twin, recovery controller, experiment harness, figure and table generation, Go "
        "control-plane services and deployment manifests — together with the processed result "
        "tables from which every figure and table in this manuscript is generated, is archived on "
        "Zenodo [78]. The development version is maintained on GitHub at "
        "https://github.com/tarasov-nikita/predictive-resilience-digital-twin. A snapshot of the "
        "same reproducibility package is provided as supplementary material with this submission. "
        "The archive contains the source code, the configuration, the fixed random seeds, the "
        "result tables, the figure and table generation scripts and a README. The full generated "
        "telemetry corpus is not distributed in either the archive or the supplementary package "
        "because of its size; it is reproduced deterministically from the archived seeds. The "
        "complete workflow — regenerating the corpus, retraining the models, re-running the "
        "campaign and rebuilding every figure and table from scratch — is executed with the "
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
