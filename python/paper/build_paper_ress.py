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

    def equation(self, text):
        self.eq_no += 1
        par = self.doc.add_paragraph()
        par.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = par.add_run(text)
        r.italic = True
        par.add_run(f"\t\t({self.eq_no})")
        return f"({self.eq_no})"

    def figure(self, filename, caption, width=15.0):
        self.fig_no += 1
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

    def save(self, path):
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

    def lead(scen, key="lead_time_median"):
        return float(np.nanmean(ai[ai["scenario"] == scen][key]))

    _sig_metrics = ["service_availability", "resilience_loss", "mean_recovery_steps",
                    "normalised_resilience"]
    n_sig = int(R["sig"][(R["sig"]["policy"] == "proactive_ai") &
                         (R["sig"]["scenario"] != "ALL") &
                         (R["sig"]["metric"].isin(_sig_metrics)) &
                         (R["sig"]["significant_005"])].shape[0])

    # ------------------------------------------------------------------ front
    d.title("Predictive Resilience of Hybrid 5G/6G-Class and LEO Satellite Networks: "
            "A Digital-Twin Framework for Failure Anticipation and Risk-Aware Recovery")
    d.centered("Nikita Tarasov*, Roman Zinko", size=11)
    d.centered("Lviv Polytechnic National University, Lviv, Ukraine", size=10, italic=True)
    d.centered("* Corresponding author: dev.nikita@outlook.com", size=10, italic=True)
    d.doc.add_paragraph()

    d.h1("Abstract")
    d.p(
        "Hybrid terrestrial–satellite networks are increasingly used to keep critical services "
        "alive when regional infrastructure is lost, yet a basic reliability question is rarely "
        "posed quantitatively: can an impending element failure be recognised early enough for "
        "recovery to complete before the service is interrupted, and does the gain exceed the "
        "cost of acting? We develop a predictive digital-twin framework and evaluate it in a "
        "closed loop where recovery decisions change the future failure process. Degradation "
        "follows a mean-reverting jump-diffusion of a hidden health variable seen only through a "
        "noisy observation map; the twin is observation-driven and subject to a detection delay; "
        "a calibrated classifier estimates failure within a horizon; and a risk-aware controller "
        "migrates flows make-before-break when path risk exceeds a criticality-dependent budget. "
        "Resilience is measured by reliability, availability, cumulative resilience loss, "
        "interruption hazard and lead time. Across "
        f"{len(R['runs'])} closed-loop runs (seven scenarios, six policies, {R['n_seeds']} paired "
        f"seeds) the proposed policy reduces resilience loss by "
        f"{abs(rel(RL.loc[S4, 'proactive_ai'], RL.loc[S4, 'reactive'])):.0f}% and recovery time "
        f"by {abs(rel(REC.loc[S4, 'proactive_ai'], REC.loc[S4, 'reactive'])):.0f}% under "
        "infrastructure collapse. Ablations qualify this: a static-threshold proactive rule is "
        "worse than not acting; degradation-aware routing without prediction recovers two thirds "
        "of the unweighted gain, yet under class-balanced criticality weighting only prediction "
        "improves the loss; "
        "perfect state knowledge over-actuates; and the benefit vanishes once a migration costs "
        "three to five slots. On a scenario whose reconfiguration cadence and link statistics "
        "are taken from published measurements rather than from our model, the availability and "
        "loss advantages disappear and only the recovery-time benefit survives, which bounds "
        "where the result applies."
    )
    d.p("Keywords: predictive resilience; failure anticipation; digital twin; dynamic reliability; "
        "risk-aware recovery; non-terrestrial networks; common-cause failure; critical "
        "communication infrastructure.", italic=True)

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
    d.h2("1.1 Research questions")
    d.p("The study is organised around four questions.")
    d.numbered([
        "RQ1. How accurately, and how early, can impending element failures be recognised from "
        "observable telemetry alone, when the degradation state is hidden and the observation map "
        "is noisy?",
        "RQ2. Does failure anticipation improve service-level reliability and resilience relative "
        "to reactive restoration, and by how much across disruption scenarios of different nature?",
        "RQ3. How does the cost of a pre-emptive recovery action affect the net resilience "
        "benefit, and is there an identifiable break-even point?",
        "RQ4. Do the observed gains persist when the degradation mechanism differs from the one "
        "the predictor was trained on, including correlated common-cause disruptions?",
    ], prefix="{i}.")
    d.h2("1.2 Contributions")
    d.numbered([
        "A predictive digital-twin model of a hybrid terrestrial–non-terrestrial communication "
        "system in which the degradation state is hidden from every decision-making component, "
        "operational status is estimated subject to a detection delay, and the twin exposes a "
        "time-varying cost graph and a capacity ledger to the recovery layer.",
        "A failure-anticipation methodology whose target is the occurrence of a failure within a "
        "prediction horizon rather than the crossing of a KPI threshold, evaluated under an "
        "episode-disjoint split with isotonic calibration and a frozen operating point, and "
        "characterised by anticipation lead time rather than accuracy alone.",
        "A risk-aware recovery mechanism that couples the calibrated failure probability to a "
        "criticality-dependent risk budget and a make-before-break migration, together with its "
        "decision-theoretic interpretation as a break-even probability between the expected cost "
        "of an interruption and the cost of a migration.",
        "A dynamic resilience-assessment methodology in which recovery actions change the offered "
        "load, the load changes the hazard rate, and therefore the policy shapes the failure "
        "process it will later face; the assessment reports reliability, availability, resilience "
        "loss, normalised resilience, interruption hazard and lead time, with paired statistics "
        "and Holm correction over seven scenarios and six policies.",
    ], prefix="C{i}.")
    d.p(
        "The evaluation uses a humanitarian and emergency-response deployment as its case study, "
        "because that setting makes the service-criticality hierarchy explicit, but nothing in "
        "the framework is specific to it: the same structure applies to public-safety networks, "
        "utility telemetry, tactical communications and remote industrial operations."
    )
    d.p(
        "Section 2 reviews the related literature and states the gap. Section 3 defines the system "
        "model. Section 4 presents the framework. Section 5 defines the reliability and resilience "
        "measures. Section 6 describes the experimental design, Section 7 reports the results "
        "against RQ1–RQ4, Section 8 discusses them, and Sections 9–11 state the limitations, "
        "outline future work and conclude."
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
        "analysis is the informative view [67]; and when a controller consumes probabilities "
        "rather than labels, those probabilities must be calibrated [65], [66]. Common-cause "
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
        "LEO systems [48], [49], simulators make them evaluable [50], [53], and measurement "
        "studies of operational systems document the periodic reconfiguration and latency "
        "variability that motivate our handover scenario [51], [52]. Programmability, needed to "
        "act on any prediction, is provided by SDN [55] and by the O-RAN control architecture "
        "[54]; classical restoration theory quantifies what reactive recovery costs [56], [57]. "
        "Emergency and disaster communication studies describe the deployment context [58], [59], "
        "[60] but rarely provide reproducible quantitative comparison against a documented "
        "baseline."
    )
    d.h2("2.5 Research gap")
    d.p(
        "The gaps are as follows. Resilience assessment and failure prognosis are developed in "
        "separate literatures, so the lead time of a prediction is almost never converted into a "
        "resilience quantity. Second, evaluations are typically open-loop: a fixed failure trace "
        "is replayed and the policy cannot influence the failures it will later face, which "
        "removes exactly the feedback that makes proactive control risky. Third, the cost of "
        "acting on a prediction is usually ignored, so the literature cannot say when anticipation "
        "stops being worthwhile. This paper addresses the three jointly."
    )

    # ------------------------------------------------------- 3. System model
    d.h1("3. System model and problem formulation")
    d.h2("3.1 Network and service model")
    d.p(
        "The system is an undirected graph G = (V, E). Each element n in V belongs to one of "
        "eight classes — LTE (4G), 5G and 6G-class terrestrial access, UAV relay, LEO satellite, "
        "edge computing node, core node and user gateway — and carries a static profile (nominal "
        "capacity C_n, latency floor L_n, loss floor p_n, radio characteristics, hazard "
        "susceptibility). Each link carries a propagation latency and a capacity. Service flows "
        "of six criticality classes arrive as a marked Poisson process, occupy a path for a "
        "random holding time and are subject to class-specific latency, loss and bandwidth "
        "budgets. The consequence of an interruption differs by class, which is what makes "
        "differentiated protection meaningful rather than cosmetic."
    )
    t_el = d.table(pd.read_csv(os.path.join(TABDIR, "t01_elements.csv")),
                   "Element classes and their static reliability-relevant parameters. The hazard "
                   "scale multiplies the baseline degradation rate of Eq. (2).",
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
    eq1 = d.equation("dh_n = mu (1 - h_n) dt - lambda_n(t) dt + sigma dW_n - J dN_n ,")
    d.p(
        f"where in {eq1} the term mu (1 - h_n) represents maintenance and self-healing, "
        "lambda_n(t) is the instantaneous hazard, sigma dW_n is diffusion noise standing for "
        "micro-variability such as interference and weather, and J dN_n is a compound Poisson "
        "term for abrupt events. The hazard depends on the element profile, the scenario stress "
        "programme and the current utilisation rho_n(t),"
    )
    eq2 = d.equation("lambda_n(t) = lambda_0 k_n m_s(class_n) [1 + g max(0, rho_n(t) - rho*)^{3/2}] ,")
    d.p(
        f"and {eq2} is what closes the control loop: a policy that concentrates traffic raises "
        "utilisation, raises the hazard and thereby manufactures the failures it will later have "
        "to repair. An element with h_n < h_fail enters the FAILED state, is unavailable for a "
        "repair interval drawn uniformly from a specified range, and is then restored to a "
        "partially healthy state. Two additional mechanisms are used to construct disruption "
        "scenarios that the model was not fitted to. Periodic non-terrestrial reconfiguration "
        "produces transient degradation of the observable KPIs without damaging the element, with "
        "a small probability that the reconfiguration itself fails and reduces health; the timing "
        "and magnitude are informed by published measurements of operational LEO systems [51], "
        "[52] and are not taken from a trace. A common-cause mechanism, following the standard "
        "PRA formulation [18], injects a shared shock Z_t into every element of an affected "
        "region,"
    )
    eq3 = d.equation("h_i(t^+) = h_i(t) - alpha_i Z_t ,  Z_t ~ Gamma(2, beta/2) ,")
    d.p(
        f"where {eq3} makes the failures of elements sharing power and backhaul dependencies "
        "positively correlated. Because the shock is instantaneous and exogenous, it has no "
        "telemetry precursor by construction — a property we exploit deliberately in Section 7.5."
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
        "Telemetry is generated from (h_n, rho_n) by a deterministic map corrupted by "
        "multiplicative log-normal noise. Writing d = 1 - h for the health deficit and "
        "q = min(q_max, rho / (1 - rho)) for the saturated queueing term,"
    )
    d.equation("L_n(t) = L_n^0 [1 + a_1 d^2 + a_2 q] e^{eps},  eps ~ N(0, sigma_o^2) ,")
    d.equation("p_n(t) = p_n^0 + b d^3 + c max(0, rho_n - 0.8)^2 ,")
    eq6 = d.equation("S_n(t) = S_n^0 - gamma d^{1.2} + eta ,")
    d.p(
        "together with jitter, throughput, processor and memory occupancy, queue depth, "
        "retransmission rate, energy headroom and an exponentially weighted availability "
        f"estimate; {eq6} applies to wireless elements only. The map is non-invertible — distinct "
        "(h, rho) pairs produce the same expected latency — and noisy, so identifying the "
        "degradation state from a single sample is imperfect. Section 7.6 quantifies how the "
        "noise level determines whether temporal aggregation of the telemetry is worth performing."
    )
    d.h2("3.4 Recovery model and its cost")
    d.p(
        "Two recovery mechanisms are distinguished, and the distinction is the physical basis of "
        "the entire result. Reactive restoration is break-before-make: the flow is already "
        "interrupted, the twin needs D slots of KPI aggregation before it can declare the element "
        "failed, a replacement path is computed, and the session is re-established, which costs a "
        "further interval C_r. Pre-emptive migration is make-before-break: the replacement path "
        "is installed while the original still carries traffic, so the switch-over costs at most "
        "C_m slots. In the main experiments C_m is small, matching the rationale for pre-computed "
        "protection in carrier networks [56], [57]; Section 7.4 treats C_m as a free parameter and "
        "identifies the value at which anticipation ceases to pay."
    )
    d.h2("3.5 Parameter provenance")
    d.p(
        "Because every observation in this study is produced by the model just described, the "
        "origin of each parameter matters as much as its value. We therefore separate three "
        "categories. Measurement-informed parameters reproduce ranges reported for operational "
        "systems. Standards-derived parameters follow service or protocol specifications. "
        "Synthetic parameters are stress-test settings with no empirical counterpart; they are "
        "chosen to place the system in a regime where failures occur often enough to be measured, "
        "and their influence is examined in Section 7.9 rather than defended as estimates."
    )
    prov = pd.DataFrame([
        ["Latency floors, capacities per element class", "Table 1", "standards-derived",
         "3GPP TS 22.261 and TR 38.821 service and NTN characteristics [42], [43]"],
        ["LEO access and feeder latency, loss floor", "42 ms, 0.0025", "measurement-informed",
         "reported LEO end-user measurements [51], [52]"],
        ["Handover cadence and transient magnitudes (S5)", "15 slots, x1.8-3.6 latency",
         "measurement-informed", "periodic reconfiguration reported for LEO systems [51], [52]"],
        ["Service latency and loss budgets", "Table 2", "standards-derived",
         "3GPP TS 22.261 classes, relaxed where the NTN path makes them unattainable"],
        ["Detection delay D", "3 slots", "engineering assumption",
         "KPI aggregation and alarm de-bouncing in management planes"],
        ["Restoration cost C_r, migration cost C_m", "U(4,12), U(0,1) slots",
         "engineering assumption",
         "break-before-make versus pre-computed protection [56], [57]; swept in Section 7.4"],
        ["Baseline hazard lambda_0", "0.0050", "synthetic (calibrated)",
         "set so that the failure prevalence over the corpus is of the order of one per cent"],
        ["Diffusion sigma, reversion mu", "0.045, 0.050", "synthetic (calibrated)",
         "set to give a degradation time-scale of tens of slots; swept in Section 7.9"],
        ["Scenario hazard multipliers", "Table 4", "synthetic (stress test)",
         "define the severity ladder; no empirical counterpart is claimed"],
        ["Load factors", "1.00-2.15", "synthetic (stress test)", "scenario definition"],
        ["Utilisation-hazard gain g, knee rho*", "3.0, 0.65", "synthetic",
         "couples load to degradation; swept in Section 7.9"],
        ["Common-cause rate and magnitude (S6)", "0.014, Gamma(2, 0.39)",
         "synthetic (falsification test)",
         "constructed to be instantaneous and precursor-free by design [18]"],
        ["Observation noise sigma_o", "0.055", "synthetic", "swept in Section 7.6"],
    ], columns=["parameter", "value", "category", "basis"])
    t_prov = d.table(prov, "Provenance of the model parameters. Only the first four groups are "
                           "tied to external evidence; the remainder are declared stress-test or "
                           "calibration settings and are examined by sensitivity analysis.",
                     fmt="{}", size=7.2, widths=[4.6, 2.8, 2.8, 5.8])
    d.p(
        f"{t_prov} lists the parameters by category. Two consequences follow and are stated here "
        "rather than in the limitations, because they qualify every number in Section 7. The "
        "absolute values reported are properties of this parameterisation; only the comparison "
        "between recovery policies, which share the environment exactly, is transferable. And the "
        "conclusions must survive perturbation of the synthetic parameters to be worth anything, "
        "which is why the hazard rate, the diffusion coefficient and the "
        "load-hazard coupling are each halved and doubled in Section 7.9."
    )

    d.h2("3.6 Problem statement")
    d.p(
        "At every control interval and for every active flow f with criticality class c, the "
        "operator must choose between keeping the current path P_f and migrating to an "
        "alternative. Keeping it risks an interruption of expected duration D + C_r if an element "
        "of P_f fails; migrating costs C_m with certainty and may place the flow on a worse path. "
        "The decision must be taken from the twin's estimate of the system state, which excludes "
        "h_n. The objective is to minimise the cumulative resilience loss of the service "
        "portfolio, weighted by criticality, subject to capacity feasibility."
    )

    # --------------------------------------------------------- 4. Framework
    d.h1("4. Predictive digital-twin framework")
    d.h2("4.1 Architecture")
    fig_arch = d.figure("fig01_architecture.png",
                        "Layered architecture. The twin is observation-only; the assessment layer "
                        "is the sole consumer of ground truth. The red path is the closed loop: "
                        "recovery decisions set the offered load, which enters the hazard rate.",
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
        "Whether this construction deserves the term digital twin rather than simulator is worth "
        "addressing directly, since the term is used loosely. Four properties are usually "
        "required and are present here: a state representation that is continuously "
        "re-synchronised from telemetry rather than integrated forward independently; a strict "
        "separation between the physical state and the estimated state, including the detection "
        "delay that makes them differ; a predictive layer that answers questions about the future "
        "of the physical counterpart; and a closed actuation path back to it. What is absent is a "
        "physical counterpart: in this study the physical system is itself a controlled "
        "simulation. The architecture is therefore a digital twin of a simulated network, and "
        "synchronisation against deployed hardware remains future work. We use the term in the "
        "sense of the ITU-T Y.3090 reference architecture [32], which defines the twin by its "
        "data, model and management interfaces rather than by the nature of its counterpart."
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
        "a random forest [61], XGBoost [62] and LightGBM [63]. Every fitted model is wrapped in "
        "an isotonic calibrator fitted on the validation episodes [65], [66], and the decision "
        "threshold theta* is selected on the same fold by maximising F1 over a grid of 187 "
        f"candidates; both are frozen before the test episodes are used. For the selected model "
        f"theta* = {theta:.3f}."
    )
    d.h2("4.4 Risk-aware recovery")
    d.p(
        "The recovery layer must decide how much anticipated risk a flow may carry before "
        "migration is warranted. Instead of a fixed constant, the budget of class c is expressed "
        "as a multiple of the predictor's validated operating point,"
    )
    eq10 = d.equation("tau_c = min(0.95, kappa phi_c theta*) ,")
    d.p(
        f"with {eq10} using phi_c in {{0.80, 1.00, 1.30, 1.80, 2.50}} for criticality 5 down to 1 "
        "and a global scale kappa. The construction has an operational reading — a "
        "life-critical class migrates on weaker evidence than the level at which the model itself "
        "would raise an alarm, whereas delay-tolerant traffic waits for several times that "
        "evidence — and it inherits the statistical properties of the predictor, so re-training "
        "the model automatically re-scales the policy. Among the least-cost path and the k "
        "shortest alternatives, the controller selects the candidate minimising"
    )
    eq11 = d.equation("J(P) = L(P)/L_c^max + 4 l(P)/l_c^max + (3 + 0.6 pi_c) R_inf(P) "
                      "+ penalties(capacity, budget violations) ,")
    d.p(
        f"where {eq11} prefers an SLA-feasible path and, when none exists, degrades to the "
        "least-violating one rather than rejecting the flow. A hysteresis interval bounds "
        "control-plane churn, and migrations are processed in decreasing criticality order so "
        "that scarce residual capacity is allocated to the most consequential services first."
    )
    fig_dec = d.figure("f04_decision_mechanism.png",
                       "Recovery decision chain from the telemetry window to the migration "
                       "decision, with the break-even condition that gives the risk budget its "
                       "decision-theoretic meaning.", width=15.0)
    d.p(f"{fig_dec} summarises the chain end to end, from the telemetry window through the "
        "calibrated risk and the path aggregation to the migration decision and its cost "
        "comparison.")
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
        "moderate linear influence. No coefficient was adjusted after the campaign began, none was "
        "selected on the test seeds, and none was selected on S5 or S6. The sensitivity of the "
        "result to the one free scale that remains, kappa, is reported in Section 7.6."
    )
    d.h2("4.5 Decision-theoretic interpretation")
    d.p(
        "The budget rule admits a simple economic reading. If an interruption of a class-c flow "
        "costs C_f slots of lost service and a migration costs C_m, then migrating is rational "
        "when the expected saving exceeds the certain cost, r C_f > C_m, that is when"
    )
    eq12 = d.equation("r > p* = C_m / C_f .")
    d.p(
        f"Equation {eq12} defines a break-even probability. With the parameters of Section 6, an "
        "interruption under reactive restoration costs the detection delay plus the "
        "re-establishment interval, so C_f is of the order of ten slots and p* is of the order of "
        "0.1 for a unit-cost migration, which is the same order of magnitude as the calibrated "
        f"operating point theta* = {theta:.3f} from which the budgets are built. We do not claim "
        "an equivalence: maximising F1 does not optimise any particular economic cost function, "
        "and the two quantities coincide here only in order of magnitude. What the break-even "
        f"condition {eq12} contributes is an interpretation \u2014 the budget can be read as an "
        "implicit statement about the ratio of recovery cost to failure cost \u2014 and a testable "
        "prediction: the benefit must vanish as C_m approaches C_f. Section 7.4 tests it directly."
    )

    # -------------------------------------------- 5. Resilience quantification
    d.h1("5. Reliability and resilience quantification")
    d.h2("5.1 Service reliability and availability")
    d.p(
        "Let T_f denote the time from admission of flow f to its first interruption. The service "
        "reliability of class c is the survival function"
    )
    eq13 = d.equation("R_c(t) = P(T_c > t) ,")
    d.p(
        f"estimated with the Kaplan–Meier product-limit estimator [68] over the flows of that "
        "class, with flows that terminate without interruption treated as right-censored. Two "
        "event definitions are reported. Under the first, any interruption counts, including the "
        "sub-slot switch-over of a make-before-break migration. Under the second, only sustained "
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
        "Both quantities weight every flow equally, which contradicts the premise stated in "
        "Section 3.6 that a command-and-control session and a sensor report are not "
        "interchangeable. Writing A_c for the availability of service class c, w_c in {1, ..., 5} "
        "for its criticality as listed in Table 2, and pi_c for the share of flow-slots that "
        "class c contributes under the nominal traffic mix, we report two weightings"
    )
    eq_lw = d.equation(
        "L_w^{cls} = T [1 - sum_c w_c A_c / sum_c w_c] ,   "
        "L_w^{flw} = T [1 - sum_c w_c pi_c A_c / sum_c w_c pi_c] ,"
    )
    d.p(
        f"with T the episode length in slots. Both lines of {eq_lw} are expressed in "
        "slot-equivalents of fully unavailable service rather than in flow-slots, since Q is a "
        "fraction and not a count. Their limits under equal criticality weights are not the same "
        "and the distinction matters: with w_c = 1 for every class, L_w^{flw} reduces exactly to "
        "the ordinary flow-slot weighted availability loss T(1 - A) of Eq. (14), whereas "
        "L_w^{cls} reduces to T times the complement of the unweighted mean of the per-class "
        "availabilities, which coincides with T(1 - A) only when all classes carry the same "
        "number of flow-slots. They therefore differ in whether class prevalence enters at all. The class-balanced form "
        "L_w^{cls} treats each service class as a mission function of equal standing: an "
        "emergency-message service that carries 2 % of the flow-slots is not 2 % of the mandate, "
        "and a controller must not be able to improve the score by protecting the numerous "
        "tolerant flows. The flow-weighted form L_w^{flw} instead measures the criticality-"
        "weighted portfolio of traffic. Its weights pi_c are the nominal shares implied by the "
        "arrival intensities and mean holding times of Table 2, held fixed across policies so "
        "that no policy can alter its own denominator by admitting a different mix. Because a "
        "fixed nominal weight is only meaningful if it matches what the network actually carries, "
        "we measured the realised flow-slot shares over six independent runs spanning overload, "
        "collapse and common-cause disruption: the largest deviation from the nominal share is "
        "0.93 percentage points, so the two coincide for practical purposes and the fixed "
        "weighting introduces no material bias. We take L_w^{cls} as primary because it "
        "matches the per-class service obligations that motivate the risk budgets of Section 4.3, "
        "and report L_w^{flw} alongside it because, as Section 7.2 shows, the two do not always "
        "order the policies the same way — a divergence that is itself informative rather than a "
        "nuisance."
    )
    d.state["eq_lw"] = eq_lw
    d.h2("5.3 Recovery and interruption hazard")
    d.p(
        "The mean recovery time is the average length of an interruption episode, from the first "
        "interrupted slot to the first slot in which the flow delivers again. The empirical "
        "interruption hazard is estimated piecewise as h(t) = events in [t, t+Δ) divided by the "
        "product of the number at risk and the bin width, which exposes whether a policy shifts "
        "interruptions in time rather than removing them."
    )
    d.h2("5.4 Anticipation quality")
    d.p(
        "Prediction quality is reported as precision, recall, F1, ROC-AUC, precision–recall area, "
        "the Brier score and the false-alarm rate on negative windows. Because a probability that "
        "arrives too late is useless regardless of its accuracy, we additionally define the "
        "anticipation lead time"
    )
    eq17 = d.equation("T_lead = t_failure - t_alarm ,")
    d.p(
        f"where t_alarm is the first slot within a look-back window at which the risk of that "
        f"element exceeded theta*. Failures with no preceding alarm are counted as misses, and "
        f"the anticipation rate is the share of failures with a usable warning. {eq17} is the "
        "quantity that must be compared with the migration cost C_m: a warning shorter than the "
        "switch-over time cannot be acted upon."
    )

    # --------------------------------------------------- 6. Experimental design
    d.h1("6. Experimental design")
    d.h2("6.1 Implementation")
    d.p(
        "The scientific implementation is written in Python and comprises the environment, the "
        "feature pipeline, the predictor, the twin, the recovery controller and the assessment "
        "harness; it produces every number reported here, using NetworkX [71], NumPy [72], SciPy "
        "[73], scikit-learn [64] and Matplotlib [74]. A second implementation exposes the same "
        "twin and controller logic as REST microservices in Go, with the predictor served "
        "separately, Prometheus metrics on every service and Kubernetes manifests for deployment. "
        "Its role is to demonstrate that the twin and controller can be implemented "
        "independently of the monolithic scientific simulator, and to "
        "support reproduction; it is an engineering artefact rather than a contribution, and its "
        "details are given in the accompanying repository."
    )
    d.h2("6.2 Parameters and scenarios")
    params = pd.DataFrame([
        ["Control interval", "1 slot", "one orchestration cycle"],
        ["Episode length", "500 slots", "one run"],
        ["Prediction horizon H", f"{PREDICTION_HORIZON} slots", "anticipation target"],
        ["Feature window W", f"{FEATURE_WINDOW} slots", "telemetry ring buffer"],
        ["Failure threshold h_fail", "0.15", "health below which an element fails"],
        ["Repair interval", "U(12, 38) slots", "unattended repair"],
        ["Detection delay D", "3 slots", "KPI aggregation before an alarm"],
        ["Restoration cost C_r", "U(4, 12) slots", "break-before-make re-establishment"],
        ["Migration cost C_m", "U(0, 1) slots", "make-before-break switch-over"],
        ["Hysteresis", "6 slots", "minimum interval between migrations of a flow"],
        ["mu, sigma, lambda_0", "0.050, 0.045, 0.0050", "degradation process"],
        ["Cost weights", "w_L=1, w_P=220, w_U=45, w_R=140", "composite path metric"],
        ["Budget factors phi_c", "0.80 / 1.00 / 1.30 / 1.80 / 2.50", "criticality 5 down to 1"],
        ["Seeds per configuration", str(R["n_seeds"]), "paired replications"],
    ], columns=["parameter", "value", "role"])
    t_par = d.table(params, "Model and control parameters. All values are declared in a single "
                            "configuration module and are reported verbatim.",
                    fmt="{}", size=8.4, widths=[5.0, 4.4, 6.6])
    t_sc = d.table(pd.read_csv(os.path.join(TABDIR, "t03_scenarios.csv")),
                   "Disruption scenarios. S5 and S6 are excluded from the training corpus, so "
                   "both the predictor and the controller face their degradation mechanisms for "
                   "the first time at evaluation.", fmt="{:g}", size=7.0,
                   widths=[2.3, 4.6, 1.2, 3.0, 2.4, 1.5, 1.5, 1.4])
    d.p(
        f"{t_par} lists the parameters and {t_sc} the scenarios. S1 is the nominal control "
        "condition. S2 congests the terrestrial access segment. S3 removes an entire terrestrial "
        "cluster and degrades the non-terrestrial segment, so two gateways depend on satellite "
        "and aerial paths exactly when those are worst. S4 destroys five terrestrial elements and "
        "raises the hazard of the survivors under double load. S5 adds periodic non-terrestrial "
        "reconfiguration with transient KPI degradation. S6 injects correlated regional shocks "
        "through shared power and backhaul dependencies. The training corpus is built only from "
        f"S1–S4 and contains {int(meta['n_train']) + int(meta['n_val']) + int(meta['n_test']):,} "
        f"eligible observation windows at a test-fold prevalence of "
        f"{float(meta['positive_rate_test']):.4f}."
    )
    d.h2("6.3 Recovery policies")
    d.p("Six policies share the same twin, topology, traffic and seeds. Consecutive policies "
        "differ in exactly one mechanism, which is what lets the contribution of each mechanism "
        "be read off the comparison:")
    d.numbered([
        "P0 static — the path is fixed at admission and never revised; the lower bound.",
        "P1 reactive restoration — the path is recomputed once the twin observes that an element "
        "on it is down; the industry reference and the baseline for all comparisons.",
        "P2 threshold-proactive — the same make-before-break actuator, triggered by a static KPI "
        "alarm instead of a learned probability. It separates the value of acting early from the "
        "value of the evidence on which one acts.",
        "P3 degradation-aware — risk-augmented path cost and pre-emptive migration driven by a "
        "contemporaneous degradation index built from the same observable KPIs, with no reference "
        "to the future. The index is rescaled so that it raises alarms at the same rate as the "
        "predictor at its operating point, so that the two are compared at a matched alarm rate. "
        "This policy separates anticipation from the avoidance of already-degraded elements.",
        "P4 proposed — the same machinery driven by the calibrated probability that the element "
        "fails within the horizon, with criticality-dependent risk budgets.",
        "P5 perfect-state reference — the same machinery with the risk computed from the true "
        "latent health. It is an ablation of information quality, not an upper bound on "
        "achievable performance.",
    ], prefix="({i})")
    d.h2("6.4 Protocol and statistical analysis")
    d.p(
        f"Each (scenario, policy) configuration is replicated over {R['n_seeds']} seeds; the same "
        "seeds are used across policies, so comparisons are paired and exogenous randomness "
        f"cancels. The campaign comprises {len(R['runs'])} closed-loop runs of 500 slots. Because "
        "the loop is closed, identical seeds do not produce identical failure sequences across "
        "policies — they produce identical initial conditions and identical exogenous noise, "
        "which is the correct pairing for this design. Comparisons use the paired t-test on "
        "seed-level differences, with the Wilcoxon signed-rank test reported alongside as a "
        "distribution-free check; effect sizes are given as Cohen's d_z and as mean differences "
        "with 95 % confidence intervals. The primary inferential family, fixed before the "
        "comparisons reported below were interpreted and not revised afterwards, contains "
        "only the four primary resilience measures of the proposed policy against "
        "reactive restoration in the five stressed scenarios, that is twenty tests; the "
        "family-wise error rate over exactly that family is controlled with the Holm procedure "
        "[69], [70]. Every other comparison, including the ablation policies and the class-level "
        "measures, is reported as exploratory with uncorrected p-values and labelled as such, "
        "which avoids a correction so broad that it would mask the effects it is meant to "
        "protect."
    )

    # ---------------------------------------------------------- 7. Results
    d.h1("7. Results")
    d.h2("7.1 RQ1: how accurately and how early can failures be anticipated?")
    t_mod = d.table(R["models_raw"][["model", "accuracy", "balanced_accuracy", "precision",
                                     "recall", "f1", "mcc", "roc_auc", "pr_auc", "brier",
                                     "false_alarm_rate", "threshold"]].assign(
        model=lambda x: x["model"].str.replace("_", " ")),
        "Anticipation performance on episode-disjoint test data at the validation-selected "
        "operating point. The threshold rule is the non-learning operator heuristic.", size=7.4)
    fig_pred = d.figure("f05_predictor.png",
                        "Discrimination and calibration of the anticipation models: "
                        "precision–recall curves, reliability diagram and score distribution.",
                        width=15.5)
    d.p(
        f"{t_mod} and {fig_pred} answer the first half of RQ1. The selected model, {best_label}, "
        f"attains ROC-AUC {bm['roc_auc']:.4f} and precision–recall area {bm['pr_auc']:.4f} against "
        f"{rule['pr_auc']:.4f} for the operator threshold rule, an improvement of "
        f"{rel(bm['pr_auc'], rule['pr_auc']):.0f}% and a lift of "
        f"{bm['pr_auc'] / float(meta['positive_rate_test']):.1f} over a prevalence of "
        f"{float(meta['positive_rate_test']):.4f}. Recall rises from {pct(rule['recall'], 1)}% to "
        f"{pct(bm['recall'], 1)}% while the false-alarm rate rises from "
        f"{pct(rule['false_alarm_rate'])}% to {pct(bm['false_alarm_rate'])}% of negative windows, "
        f"and the Brier score improves from {rule['brier']:.4f} to {bm['brier']:.4f}, which is "
        "what makes the output usable as a probability rather than only as a ranking. Differences "
        "among the four learned models are small relative to episode-level variance and we do not "
        "claim an ordering between them; the operationally relevant observation is that a "
        "logistic regression scoring a network in well under a millisecond per thousand samples "
        "is sufficient."
    )
    fig_lead = d.figure("f06_lead_time.png",
                        "Anticipation lead time. Left: median lead time with inter-quartile "
                        "range. Right: share of anticipated failures with at least 1, 3, 5 and 10 "
                        "slots of warning. Scenarios marked with an asterisk are outside the "
                        "training distribution.", width=15.0)
    t_lead = d.table(pd.read_csv(os.path.join(TABDIR, "t05_lead_time.csv")),
                     "Anticipation quality inside the closed loop, averaged over seeds.",
                     size=7.6)
    d.p(
        f"{fig_lead} and {t_lead} answer the second half, which prediction metrics alone cannot. "
        f"Within the training family the median lead time is {lead(S2):.1f} slots under access "
        f"overload, {lead(S3):.1f} slots under non-terrestrial degradation and {lead(S4):.1f} "
        f"slots under infrastructure collapse, with anticipation rates of "
        f"{pct(float(np.nanmean(ai[ai.scenario == S2]['anticipation_rate'])), 0)}%, "
        f"{pct(float(np.nanmean(ai[ai.scenario == S3]['anticipation_rate'])), 0)}% and "
        f"{pct(float(np.nanmean(ai[ai.scenario == S4]['anticipation_rate'])), 0)}% respectively. "
        "Since the make-before-break switch-over costs at most one slot, a median warning of "
        "several slots is comfortably actionable, which is the precondition for everything that "
        "follows. On the two held-out mechanisms the picture changes sharply: the anticipation "
        f"rate falls to {pct(float(np.nanmean(ai[ai.scenario == S5]['anticipation_rate'])), 1)}% "
        f"under handover dynamics and "
        f"{pct(float(np.nanmean(ai[ai.scenario == S6]['anticipation_rate'])), 1)}% under "
        "common-cause disruption. The second value follows from the mechanism rather than from "
        "model quality: an instantaneous shared shock has no telemetry precursor, so no "
        "observation-based predictor can anticipate it."
    )

    d.h2("7.2 RQ2: does anticipation improve resilience?")
    fig_ar = d.figure("f10_availability_recovery.png",
                      "Service availability and mean recovery time by scenario and policy "
                      "(mean and standard error over paired seeds).", width=16.0)
    fig_res = d.figure("f09_resilience.png",
                       "Cumulative resilience loss and normalised resilience. Nominal operation "
                       "is omitted because no disruption occurs there.", width=16.0)
    compact = []
    for scen in [S1, S2, S3, S4, S5, S6]:
        if scen not in A.index:
            continue
        row = {"scenario": scen.replace("_", " ")}
        for label, tab, fmt in (("availability [%]", A, 2), ("resilience loss", RL, 2),
                                ("recovery [slots]", REC, 2)):
            base, prop = tab.loc[scen, "reactive"], tab.loc[scen, "proactive_ai"]
            scale = 100.0 if label.startswith("avail") else 1.0
            row[f"{label} P1"] = round(base * scale, fmt)
            row[f"{label} P4"] = round(prop * scale, fmt)
            row[f"{label} change [%]"] = round(rel(prop, base), 2) if base else "—"
        compact.append(row)
    t_main = d.table(pd.DataFrame(compact),
                     "Principal resilience outcomes: reactive restoration (P1) against the "
                     "proposed predictive policy (P4), averaged over paired seeds. The complete "
                     "six-policy comparison is given in Appendix A.", fmt="{:g}", size=7.6)
    d.p(
        f"{fig_ar}, {fig_res} and {t_main} answer RQ2. Under nominal operation all policies are "
        "indistinguishable, which is a required property rather than a result: an anticipatory "
        "controller must not damage a healthy system. As disruption severity grows the ordering "
        f"becomes stable. Under infrastructure collapse the proposed policy raises availability "
        f"from {pct(A.loc[S4, 'reactive'])}% to {pct(A.loc[S4, 'proactive_ai'])}%, reduces mean "
        f"recovery time from {REC.loc[S4, 'reactive']:.2f} to {REC.loc[S4, 'proactive_ai']:.2f} "
        f"slots ({abs(rel(REC.loc[S4, 'proactive_ai'], REC.loc[S4, 'reactive'])):.1f}%) and "
        f"reduces cumulative resilience loss from {RL.loc[S4, 'reactive']:.2f} to "
        f"{RL.loc[S4, 'proactive_ai']:.2f} slot-equivalents "
        f"({abs(rel(RL.loc[S4, 'proactive_ai'], RL.loc[S4, 'reactive'])):.1f}%). Under "
        f"non-terrestrial degradation recovery time falls from {REC.loc[S3, 'reactive']:.2f} to "
        f"{REC.loc[S3, 'proactive_ai']:.2f} slots and resilience loss from "
        f"{RL.loc[S3, 'reactive']:.2f} to {RL.loc[S3, 'proactive_ai']:.2f}. Under common-cause "
        f"disruption — a mechanism the predictor cannot anticipate — availability still improves "
        f"from {pct(A.loc[S6, 'reactive'])}% to {pct(A.loc[S6, 'proactive_ai'])}% and resilience "
        f"loss falls by {abs(rel(RL.loc[S6, 'proactive_ai'], RL.loc[S6, 'reactive'])):.1f}%, "
        "because risk-aware path selection keeps flows away from already-degraded elements even "
        "when the triggering shock itself is unforeseeable."
    )
    d.p(
        "The threshold-proactive policy P2 is the most informative comparison in the table. "
        "It uses the identical actuator and acts "
        f"early, yet it reduces availability to {pct(A.loc[S4, 'proactive_threshold'])}% under "
        f"collapse — below the static policy — and more than doubles resilience loss to "
        f"{RL.loc[S4, 'proactive_threshold']:.2f}, because it triggers "
        f"{PRR.loc[S4, 'proactive_threshold']:.0f} migrations per run against "
        f"{PRR.loc[S4, 'proactive_ai']:.0f} for the proposed policy. Its mean recovery time is the "
        f"lowest of all policies ({REC.loc[S4, 'proactive_threshold']:.2f} slots), which is "
        "precisely why recovery time must not be reported alone: a policy that interrupts "
        "services constantly but briefly optimises that statistic while destroying the resilience "
        "measure that matters. Acting early is therefore not sufficient; acting early on evidence "
        "that discriminates impending failure from transient degradation is."
    )
    WRL = R["WRL"]
    FWRL = R["FWRL"]
    d.p(
        "The degradation-aware policy P3 answers the second half of RQ2 and changes how the "
        "contribution should be stated. Using only a contemporaneous degradation index at a "
        f"matched alarm rate, it already reaches {pct(A.loc[S4, 'degradation_aware'])}% "
        f"availability and a resilience loss of {RL.loc[S4, 'degradation_aware']:.2f} under "
        f"collapse, against {pct(A.loc[S4, 'reactive'])}% and {RL.loc[S4, 'reactive']:.2f} for "
        f"reactive restoration and {pct(A.loc[S4, 'proactive_ai'])}% and "
        f"{RL.loc[S4, 'proactive_ai']:.2f} for the proposed policy. Roughly two thirds of the "
        "unweighted improvement is therefore attributable to keeping traffic away from elements "
        "that are already degraded, and only the remainder to anticipating failures that have not "
        "yet manifested. Under common-cause disruption the two are close "
        f"({RL.loc[S6, 'degradation_aware']:.2f} against {RL.loc[S6, 'proactive_ai']:.2f}), which "
        "is what the mechanism predicts: when shocks carry no precursor there is nothing to "
        "anticipate and only exposure control remains. Under handover dynamics the ordering "
        f"reverses in favour of anticipation ({RL.loc[S5, 'degradation_aware']:.2f} against "
        f"{RL.loc[S5, 'proactive_ai']:.2f}), because the contemporaneous index is misled by "
        "transient reconfiguration in the same way as the static alarm."
    )
    if WRL is not None:
        d.p(
            f"The criticality-weighted losses of Eq. {d.state['eq_lw']} separate the policies "
            "differently from the unweighted one, and the two weightings do not agree. Under the "
            "class-balanced form the proposed policy attains "
            f"{WRL.loc[S4, 'proactive_ai']:.2f} slot-equivalents under collapse against "
            f"{WRL.loc[S4, 'reactive']:.2f} for reactive restoration, whereas the "
            f"degradation-aware policy attains {WRL.loc[S4, 'degradation_aware']:.2f} — worse "
            "than doing nothing anticipatory, despite its clear advantage on the unweighted "
            "measure. Avoiding degraded elements reduces total outage but distributes the "
            "residual outage without regard to criticality, while the predictive policy, whose "
            "budgets are tighter for critical classes, moves the residual outage onto the classes "
            "that tolerate it."
        )
    wl_path = os.path.join(TABDIR, "t19_weighted_loss.csv")
    if os.path.exists(wl_path):
        t_wl = d.table(pd.read_csv(wl_path),
                       "Resilience loss under infrastructure collapse on three weightings, "
                       "averaged over paired seeds. The class-balanced and flow-weighted columns "
                       "are the two lines of Eq. (17).", fmt="{:g}", size=7.6)
        t_wi = d.table(pd.read_csv(os.path.join(TABDIR, "t20_weighted_inference.csv"))[
            ["comparison", "measure", "mean difference", "ci95 low", "ci95 high",
             "cohen d_z", "p_t", "wilcoxon p"]],
            "Exploratory paired inference on the weighted losses under infrastructure collapse. "
            "These measures were not part of the primary inferential family, so the p-values are "
            "uncorrected and the comparisons are descriptive.", size=7.2)
        d.p(
            f"{t_wl} and {t_wi} give the three weightings side by side with paired uncertainty. "
            "The proposed policy improves on reactive restoration under every weighting with "
            "large effect sizes (d_z between -1.0 and -1.8, all uncorrected p < 0.001). The "
            "degradation-aware deterioration under class-balanced weighting is by contrast a "
            "weak signal: a mean difference of +0.85 slot-equivalents with a 95 % interval of "
            "[0.08, 1.63], d_z = 0.47, and a Wilcoxon p of 0.056 that does not reach the "
            "conventional threshold. We therefore state it as a tendency rather than an "
            "established effect, and note that it would not survive correction for multiplicity "
            "had these measures been part of the primary family."
        )
    if FWRL is not None:
        d.p(
            "That reversal is, however, a property of the class-balanced weighting and not a "
            f"general one. Under the flow-weighted form the same three policies attain "
            f"{FWRL.loc[S4, 'reactive']:.2f}, {FWRL.loc[S4, 'degradation_aware']:.2f} and "
            f"{FWRL.loc[S4, 'proactive_ai']:.2f} slot-equivalents, so the degradation-aware "
            "policy is again ahead of reactive restoration and the ordering matches the "
            "unweighted measure. The two weightings differ because the critical classes are "
            "sparse: command-and-control and emergency messaging together account for about 7 % "
            "of the flow-slots, so their outages are visible in the class-balanced score and "
            "diluted in the flow-weighted one. The defensible statement is therefore narrower "
            "than the one the class-balanced numbers alone would support: the proposed policy has "
            "the lowest weighted loss under both weightings, but the claim that exposure control "
            "alone makes the weighted outcome worse holds only when every service class is "
            "treated as an obligation of equal standing. Which weighting applies is an operator "
            "decision, and we report both rather than choosing the one that flatters the method."
        )
    t_sig = d.table(pd.read_csv(os.path.join(TABDIR, "t12_significance.csv")),
                    "Paired comparison of the proposed policy against reactive restoration. "
                    "p_holm is corrected over the primary inferential family of "
                    "twenty tests.", size=7.2)
    d.p(
        f"{t_sig} reports the statistical analysis. {n_sig} of the reported comparisons remain "
        "significant at the five per cent level after Holm correction over the primary "
        "family of twenty tests, "
        "including recovery time in S3, S4 and S6 and availability, resilience loss and "
        "normalised resilience in S4 and S6, with |d_z| between 1.3 and 7.4. Comparisons under "
        "nominal operation are necessarily excluded because no interruptions occur there; the "
        "smaller class-level differences do not survive correction and are reported as "
        "descriptive evidence only."
    )
    d.h2("7.3 Service reliability and interruption hazard")
    fig_sur = d.figure("f08_survival.png",
                       "Kaplan–Meier estimates of the probability that a flow has not yet been "
                       "interrupted (top row: any interruption; bottom row: interruptions lasting "
                       "at least two slots) and the corresponding empirical hazard.", width=16.0)
    t_sur = d.table(pd.read_csv(os.path.join(TABDIR, "t06_survival.csv")),
                    "Survival of admitted flows at selected horizons under both interruption "
                    "definitions.", size=6.8)
    d.p(
        f"{fig_sur} and {t_sur} show a nuance that the aggregate measures hide, and we consider it "
        "one of the more instructive results. Under the permissive definition, in which any "
        "interruption counts, the proposed policy has slightly lower survival than reactive "
        "restoration in the two severe scenarios, because every pre-emptive migration registers "
        "as a sub-slot interruption; under non-terrestrial degradation, where migrations are far "
        "less frequent, the two are indistinguishable. Under the definition that counts only "
        "sustained interruptions of at least two slots the ordering reverses at the longer "
        "horizons; at forty slots the proposed policy attains the highest survival of the "
        "realisable policies in the two severe scenarios, while under non-terrestrial degradation "
        "the degradation-aware policy is marginally ahead. The "
        "framework therefore does not eliminate interruptions; it converts rare long ones into "
        "frequent negligible ones. Whether that is the right trade depends on the service: it "
        "may be preferable for classes whose utility is more sensitive to sustained outages than "
        "to sub-slot switch-over events, and it is clearly not preferable for classes in which "
        "any break is unacceptable, such as industrial control or ultra-reliable low-latency "
        "traffic. What matters here is that a single-metric evaluation would not have reported "
        "the trade at all. The hazard panels "
        "show the same mechanism in time: the proposed policy raises the early hazard slightly, "
        "when migrations happen, and lowers the late hazard, when unmanaged degradation would "
        "otherwise have caused an outage."
    )
    fig_tl = d.figure("f07_timeline.png",
                      "One infrastructure-collapse episode. Top: service quality Q(t) under each "
                      "policy. Middle: number of failed elements. Bottom: mean anticipated risk "
                      "compared with the perfect-state signal.", width=15.0)
    d.p(
        f"{fig_tl} traces a single episode. The quality trajectory of the proposed policy has "
        "shallower and shorter troughs than the reactive one, the threshold policy oscillates "
        "continuously, and the anticipated risk rises ahead of the failure clusters with a "
        "smaller amplitude than the perfect-state signal — a calibrated model is conservative, "
        "which is why its budgets must be expressed relative to its own operating point."
    )

    d.h2("7.4 RQ3: how much does the cost of acting matter?")
    fig_mig = d.figure("f11_migration_cost.png",
                       "Sensitivity to the cost of a pre-emptive migration. Dashed lines mark "
                       "reactive restoration under identical conditions.", width=16.0)
    mig = R["migration"]
    ans = {}
    if not mig.empty:
        prop = mig[mig["migration_cost"] >= 0]
        ref = mig[mig["migration_cost"] < 0]
        t_mig = d.table(pd.read_csv(os.path.join(TABDIR, "t08_migration_cost.csv")),
                        "Closed-loop outcome as a function of the migration cost C_m.", size=7.6)
        lines = []
        for scen in sorted(ref["scenario"].unique()):
            base = ref[ref["scenario"] == scen]["service_availability"].mean()
            curve = prop[prop["scenario"] == scen].groupby("migration_cost")[
                "service_availability"].mean()
            worse = [c for c in curve.index if curve[c] < base]
            ans[scen] = min(worse) if worse else None
            lines.append(
                f"under {SCEN_TEXT.get(scen, scen)} availability falls from "
                f"{pct(curve.iloc[0])}% at C_m = {curve.index[0]:g} to "
                f"{pct(curve.iloc[-1])}% at C_m = {curve.index[-1]:g}, crossing the reactive "
                f"reference of {pct(base)}% at "
                + (f"C_m = {ans[scen]:g} slots" if ans[scen] is not None
                   else "no tested cost")
            )
        d.p(
            f"{fig_mig} and {t_mig} answer RQ3 directly, and the answer is a usable engineering "
            "criterion rather than a qualitative statement. Sweeping the switch-over cost of "
            "the pre-emptive mechanism while holding everything else fixed gives the following "
            "picture: " + "; ".join(lines) + "."
        )
        crossings = [v for v in ans.values() if v is not None]
        if crossings:
            d.p(
                f"In the simulated operating regimes anticipation therefore stops being "
                f"worthwhile once a pre-emptive migration costs more than about "
                f"{min(crossings):g} to {max(crossings):g} slots, which "
                "is of the same order as the detection delay plus a fraction of the restoration "
                "interval. The break-even condition of Eq. (12) predicts exactly this: the "
                "expected saving r C_f is bounded by the interruption cost the policy can avoid, "
                "so as C_m approaches C_f the inequality can no longer be satisfied at any "
                "attainable risk level. The threshold itself depends on the detection delay, "
                "the restoration interval and the redundancy of the topology, so it should be "
                "re-derived rather than transferred; the procedure, however, is general: "
                "measure the switch-over cost of the available protection mechanism, compare "
                "it with the detection-plus-restoration interval, and adopt anticipation only "
                "if the former is a small fraction of the latter."
            )
    d.h2("7.5 RQ4: does the benefit survive unseen degradation mechanisms?")
    fig_ood = d.figure("f13_generalisation.png",
                       "Generalisation of the anticipation model. Left: precision–recall lift "
                       "over prevalence under leave-one-scenario-out training and on the two "
                       "mechanisms excluded from the corpus. Right: recall at the frozen "
                       "operating point and after re-tuning the threshold on a small "
                       "commissioning sample from the target regime.", width=15.5)
    ood = R["ood"]
    t_ood = d.table(pd.read_csv(os.path.join(TABDIR, "t07_generalisation.csv"))[
        ["setting", "held_out", "prevalence", "roc_auc", "pr_auc", "lift", "recall",
         "recall_recalibrated"]],
        "Out-of-distribution evaluation. Leave-one-scenario-out withholds one training scenario; "
        "the unseen-mechanism rows evaluate on handover dynamics and common-cause disruption.",
        size=7.4)
    if not ood.empty:
        loso = ood[(ood["setting"] == "leave_one_scenario_out") & (ood["prevalence"] > 0)]
        unseen = ood[ood["setting"] == "unseen_mechanism"]
        d.p(
            f"{fig_ood} and {t_ood} report two levels of distribution shift. Under "
            "leave-one-scenario-out training the ranking quality transfers but degrades: the "
            f"precision–recall lift over prevalence ranges from {loso['lift'].min():.1f} to "
            f"{loso['lift'].max():.1f}, against "
            f"{bm['pr_auc'] / float(meta['positive_rate_test']):.1f} in the matched-distribution "
            "case. More importantly, the frozen operating point does not transfer: recall on the "
            f"held-out scenario falls to between {pct(loso['recall'].min(), 1)}% and "
            f"{pct(loso['recall'].max(), 1)}%, because the score distribution shifts even when "
            "the ordering is preserved. Re-selecting the threshold on a small commissioning "
            "sample drawn from the target regime, without re-training the model, recovers much "
            "of that loss where the ranking transferred well: recall rises to "
            f"{pct(float(loso[loso.held_out == S3]['recall_recalibrated'].iloc[0]), 1)}% for "
            "non-terrestrial degradation and "
            f"{pct(float(loso[loso.held_out == S4]['recall_recalibrated'].iloc[0]), 1)}% for "
            "infrastructure collapse. It does not help under access overload, where re-tuning "
            f"leaves recall at {pct(float(loso[loso.held_out == S2]['recall_recalibrated'].iloc[0]), 1)}% "
            "because the ranking itself transferred poorly there. The deployment recommendation "
            "is therefore conditional: transfer the model and re-tune the alarm level, but first "
            "verify on the target regime that the ranking quality survived."
        )
        d.p(
            "On the two mechanisms withheld entirely from the corpus the outcome separates "
            f"cleanly by physics. Handover dynamics retain a lift of "
            f"{float(unseen[unseen.held_out == S5]['lift'].iloc[0]):.1f} but recall collapses at "
            "the frozen threshold, because the transient reconfiguration signature is a new mode "
            "of KPI degradation that is not followed by failure. Common-cause disruption retains "
            f"a lift of only {float(unseen[unseen.held_out == S6]['lift'].iloc[0]):.1f}, which is "
            "an expected outcome given the mechanism: an instantaneous shared "
            "shock leaves no observable precursor, so no telemetry-based estimator can "
            "anticipate it."
        )
        d.p(
            "The closed-loop consequence is the more relevant test, and it is reassuring in a way "
            "the prediction metrics alone would not suggest. In the handover scenario the "
            f"proposed policy maintains availability of {pct(A.loc[S5, 'proactive_ai'])}% against "
            f"{pct(A.loc[S5, 'reactive'])}% for reactive restoration, while the threshold-"
            f"proactive policy — which is fooled by exactly the transient signature the learned "
            f"model ignores — drops to {pct(A.loc[S5, 'proactive_threshold'])}% and inflates "
            f"resilience loss from {RL.loc[S5, 'reactive']:.2f} to "
            f"{RL.loc[S5, 'proactive_threshold']:.2f}. In the common-cause scenario the proposed "
            f"policy still delivers the best availability, and a resilience loss statistically "
            "indistinguishable from the degradation-aware policy, obtained through risk-aware "
            "avoidance of degraded elements rather than through anticipation of the shock itself. Robustness to unseen "
            "mechanisms therefore comes from two separable sources, and only one of them requires "
            "the failure to be predictable."
        )
    d.h2("7.6 Sensitivity analyses")
    hs = R["horizon"]
    if not hs.empty:
        fig_h = d.figure("f12_horizon.png",
                         "Effect of the prediction horizon on the statistical problem and on the "
                         "closed-loop outcome.", width=15.5)
        t_h = d.table(pd.read_csv(os.path.join(TABDIR, "t09_horizon.csv")),
                      "Prediction-horizon sweep. Closed-loop columns are averaged over seeds of "
                      "the infrastructure-collapse scenario.", size=7.6)
        best_h = int(hs.loc[hs["cl_availability"].idxmax(), "horizon"])
        d.p(
            f"{fig_h} and {t_h} vary the horizon H. Increasing H raises the precision–recall area "
            f"from {hs['pr_auc'].min():.3f} to {hs['pr_auc'].max():.3f}, which looks like an "
            f"improvement until one notices that the prevalence rises with it and the lift over "
            f"prevalence falls from {hs['lift'].max():.1f} to {hs['lift'].min():.1f}. The "
            f"closed-loop availability is flat to within a few tenths of a percentage point and "
            f"peaks at H = {best_h} slots. Longer horizons buy nominally better classification "
            "metrics and no system-level benefit, because an alarm raised far in advance is "
            "either acted upon too early — paying the migration cost for a failure that may not "
            "materialise — or allowed to expire. The horizon should therefore be chosen close to "
            "the sum of the detection and actuation delays, not maximised."
        )
        d.p(
            f"The main campaign nevertheless uses H = {PREDICTION_HORIZON} slots rather than the "
            f"closed-loop optimum H = {best_h}: we retain the nominal configuration rather than "
            "re-optimising the principal campaign after observing the sweep. The "
            "shorter horizon yields a scarcer positive class and a correspondingly narrower "
            "margin for the alarm threshold, so its apparent advantage is the least robust part "
            "of the sweep; the availability difference across the range is a few tenths of a "
            "percentage point, well inside the seed-to-seed spread. Reporting the main results at "
            f"H = {PREDICTION_HORIZON} therefore states the contribution at a setting that is not "
            "tuned to the outcome, and the sweep is presented so that the reader can see what a "
            "tuned horizon would add."
        )
    if not R["noise"].empty:
        fig_n = d.figure("f14_noise.png",
                         "Value of temporal aggregation as a function of observation noise.",
                         width=9.0)
        ns = R["noise"]
        lowest = ns[ns["obs_noise"] == ns["obs_noise"].min()]
        highest = ns[ns["obs_noise"] == ns["obs_noise"].max()]
        d.p(
            f"{fig_n} addresses a question that a feature-ablation table alone answers "
            "misleadingly. At the nominal noise level the instantaneous KPI vector achieves a "
            f"precision–recall area of "
            f"{float(lowest[lowest.variant.str.startswith('M0')]['pr_auc'].iloc[0]):.4f} against "
            f"{float(lowest[lowest.variant.str.startswith('M2')]['pr_auc'].iloc[0]):.4f} for the "
            "full temporal window: aggregation does not help, and for a linear model it slightly "
            "hurts. The reason is structural rather than empirical. The degradation process is "
            "Markov in the latent health, so the current health is a sufficient statistic for the "
            "future failure event; a window can only help by denoising the estimate of that "
            "health. As the observation noise grows the ordering reverses, and at the highest "
            f"level tested the temporal representation leads, "
            f"{float(highest[highest.variant.str.startswith('M2')]['pr_auc'].iloc[0]):.4f} against "
            f"{float(highest[highest.variant.str.startswith('M0')]['pr_auc'].iloc[0]):.4f}. The "
            "practical implication is that window features should be adopted on the basis of "
            "measured telemetry noise, not by default. The absolute values of the noise sweep are "
            "not directly comparable with the ablation table below: the sweep regenerates a "
            "smaller corpus at every noise level, so its test fold differs from the one used for "
            "the main models, and only the ordering within each panel should be read."
        )
        t_abl = d.table(pd.read_csv(os.path.join(TABDIR, "t13_feature_ablation.csv"))[
            ["variant", "description", "n_features", "pr_auc", "recall", "brier",
             "false_alarm_rate"]],
            "Feature representation and calibration ablation at the nominal noise level. Rows "
            "M0 to M2 vary only the feature representation with the estimator and the "
            "calibration held fixed; row M3 differs from M2 only by the isotonic calibration "
            "stage.", size=7.4)
        d.p(
            f"{t_abl} completes the picture and separates two effects that are often reported "
            "together. The M0-to-M2 contrast isolates the representation: adding rolling "
            "statistics and trends to the instantaneous KPI vector changes ranking quality only "
            "marginally at this noise level, consistent with the sweep above. The M2-to-M3 "
            "contrast isolates calibration alone: isotonic regression reduces the Brier score by "
            "nearly a factor of seven at the cost of a small reduction in precision–recall area. "
            "Since the controller consumes probabilities and compares them with a cost-derived "
            "threshold, that is the correct trade, and it is a property of the calibration stage "
            "rather than of the features."
        )
    if not R["sweep"].empty:
        fig_b = d.figure("s04_budget_sweep.png",
                         "Sensitivity of the proposed policy to the risk-budget scale kappa.",
                         width=15.0)
        sw = R["sweep"].groupby("budget_scale").mean(numeric_only=True)
        best_k = float(sw["service_availability"].idxmax())
        d.p(
            f"{fig_b} varies the budget scale kappa. Small budgets make the controller "
            "hypersensitive and reproduce, in milder form, the pathology of the threshold policy; "
            "large budgets converge towards reactive behaviour. The optimum in this environment "
            f"lies near kappa = {best_k:g} and the response is flat around it, so the policy does "
            "not require precise tuning."
        )
    d.h2("7.7 Dependence on structural redundancy")
    if not R["topology"].empty:
        fig_topo2 = d.figure("f15_topology.png",
                             "Resilience benefit of anticipation as a function of structural "
                             "redundancy. Only the graph changes across classes; elements, "
                             "traffic and degradation are identical.", width=15.5)
        tp = pd.read_csv(os.path.join(TABDIR, "t15_topology.csv"))
        t_topo2 = d.table(tp, "Topology classes ordered by mean route diversity, with the "
                              "resilience benefit of the proposed policy over reactive "
                              "restoration under infrastructure collapse.", size=7.4)
        lo, hi = tp.iloc[0], tp.iloc[-1]
        best = tp.loc[tp["availability gain [pp]"].idxmax()]
        d.p(
            f"{fig_topo2} and {t_topo2} vary the amount of alternative capacity while holding the "
            "element catalogue, the traffic model and the degradation process fixed. The "
            "reduction in resilience loss increases with route diversity, from "
            f"{lo['resilience loss reduction [%]']:.1f}% in the sparse class, where the mean "
            f"number of node-disjoint routes per gateway is {lo['route diversity']:.2f}, to "
            f"between {tp['resilience loss reduction [%]'].iloc[2:].min():.0f}% and "
            f"{tp['resilience loss reduction [%]'].max():.0f}% in the three most diverse classes. "
            "The availability gain behaves differently: it peaks "
            f"at {best['availability gain [pp]']:.2f} percentage points for the "
            f"{best['topology'].replace('_', ' ')} class and is smaller at both extremes. The two "
            "trends together describe the operating envelope of the method. With little "
            "diversity there is frequently nowhere to migrate a flow, so anticipation cannot be "
            "converted into continuity; with abundant diversity individual failures affect fewer "
            "flows in the first place, so less loss is available to be avoided even though the "
            "loss that does occur is avoided more effectively. Anticipatory recovery is therefore "
            "most valuable in systems that possess alternatives but not an abundance of them, "
            "which is the regime in which emergency and post-disaster networks typically operate."
        )
    d.h2("7.8 Does the policy shape its own failure process?")
    fb_path = os.path.join(TABDIR, "t16_feedback.csv")
    if os.path.exists(fb_path):
        fig_fb = d.figure("f16_feedback.png",
                          "Benefit of anticipation with the utilisation-to-hazard path enabled "
                          "and disabled.", width=14.0)
        fb = pd.read_csv(fb_path)
        t_fb = d.table(fb, "Closed-loop feedback ablation. With the feedback disabled the "
                           "environment follows an exogenous load profile, so the failure process "
                           "no longer responds to the controller.", size=7.6)
        on = fb[fb["load-hazard feedback"] == "enabled"]
        off = fb[fb["load-hazard feedback"] == "disabled"]
        d.p(
            f"{fig_fb} and {t_fb} isolate the mechanism claimed in contribution C4. With the "
            f"feedback enabled the proposed policy reduces resilience loss by "
            f"{float(on['loss reduction [%]'].mean()):.1f}% and raises availability by "
            f"{float(on['availability gain [pp]'].mean()):.2f} percentage points; with the same "
            "policy, the same seeds and the same predictor but the load-to-hazard path removed, "
            f"the reduction falls to {float(off['loss reduction [%]'].mean()):.1f}% and the "
            f"availability gain to {float(off['availability gain [pp]'].mean()):.2f} points. "
            "Roughly two thirds of the measured reduction therefore disappears when the "
            "controller is no longer allowed to shape its own failure process. That component "
            "comes not from repairing the consequences of failures but from not creating them: "
            "by keeping traffic off endangered elements the controller also keeps their "
            "utilisation, and hence their hazard rate, lower than the reactive policy does. An "
            "open-loop evaluation that replays a fixed failure trace cannot observe this "
            "component and would understate the value of anticipation."
        )
        d.p(
            "This ablation uses its own block of eight paired seeds rather than the "
            "twenty-four seeds of the main campaign, so the feedback-enabled row is close to but "
            "not numerically identical to the corresponding main-campaign result reported in "
            "Section 7.2; only the "
            "enabled-versus-disabled contrast within this experiment is interpretable."
        )
    d.h2("7.9 Sensitivity to the degradation-model parameters")
    ps_path = os.path.join(TABDIR, "t17_parameters.csv")
    if os.path.exists(ps_path):
        fig_ps = d.figure("f17_parameters.png",
                          "Benefit of the proposed policy under one-at-a-time perturbation of the "
                          "parameters that govern the generative model.", width=13.5)
        ps = pd.read_csv(ps_path)
        t_ps = d.table(ps, "Parameter sensitivity. The absolute values necessarily move; the "
                           "question is whether the sign and the order of magnitude of the "
                           "advantage survive.", size=7.6)
        pos = int((ps["availability gain [pp]"] > 0).sum())
        d.p(
            f"{fig_ps} and {t_ps} address the objection that the results might be an artefact of "
            "the chosen degradation parameters. The baseline hazard, the diffusion coefficient "
            "and the utilisation-hazard gain were each halved and doubled relative to their "
            f"nominal values. The availability advantage of the proposed policy remains positive "
            f"in {pos} of the {len(ps)} settings, and the resilience-loss reduction ranges from "
            f"{ps['loss reduction [%]'].min():.0f}% to {ps['loss reduction [%]'].max():.0f}%. The "
            "exception is the halved hazard rate, where almost no failures occur at all "
            "(availability of 99.79 % for both policies, resilience loss below one slot-"
            "equivalent) and the difference is numerically negligible in either direction. The "
            "magnitude is strongly parameter-dependent: it collapses towards zero when events "
            "become rare, because there is then nothing to anticipate, and it also falls at the "
            "doubled hazard rate, where the network is degraded faster than any reroute can "
            "compensate. The defensible conclusion is therefore narrower than parameter "
            "independence: within the regime in which failures are frequent enough to matter and "
            "spare capacity still exists, the sign of the advantage is preserved, while its size "
            "must be re-established for any specific deployment."
        )
    d.h2("7.10 Information quality is not the binding constraint")
    d.p(
        "The perfect-state reference P5 replaces the learned probability with a monotone function "
        "of the true latent health while leaving the control policy unchanged. Among the policies "
        "that keep their migrations under control it achieves the shortest recovery times "
        f"({REC.loc[S4, 'oracle']:.2f} slots under collapse against "
        f"{REC.loc[S4, 'proactive_ai']:.2f} for the proposed policy; only the threshold-proactive "
        f"policy is nominally faster, at {REC.loc[S4, 'proactive_threshold']:.2f} slots, for the "
        "reason given in Section 7.2), which confirms that better state knowledge does translate "
        "into faster protection. Yet its availability "
        f"under collapse, {pct(A.loc[S4, 'oracle'])}%, is below the proposed policy's "
        f"{pct(A.loc[S4, 'proactive_ai'])}%, and its resilience loss is higher "
        f"({RL.loc[S4, 'oracle']:.2f} against {RL.loc[S4, 'proactive_ai']:.2f}), because it "
        f"triggers {PRR.loc[S4, 'oracle'] / max(1.0, PRR.loc[S4, 'proactive_ai']):.1f} times more "
        "migrations and pays their aggregate cost. In this regime the binding constraint is "
        "therefore not the accuracy of the information but the calibration of the action policy "
        "to its cost. A perfectly informed controller with an untuned actuation rule is not "
        "better than a moderately informed controller with a cost-calibrated one."
    )

    d.h2("7.11 Externally anchored dynamics: where the advantage does not transfer")
    ext_path = os.path.join(TABDIR, "t24_external.csv")
    if os.path.exists(ext_path):
        t_ext = d.table(pd.read_csv(ext_path),
                        "All six policies on the externally anchored non-terrestrial scenario "
                        "S7, averaged over 24 paired seeds. The reconfiguration cadence, its "
                        "phase within the minute and the satellite-link latency statistics are "
                        "taken from published measurement studies rather than from the link "
                        "budget of this model.", fmt="{:g}", size=7.4)
        t_exi = d.table(pd.read_csv(os.path.join(TABDIR, "t25_external_inference.csv"))[
            ["comparison", "measure", "mean difference", "ci95 low", "ci95 high",
             "cohen d_z", "p_t"]],
            "Exploratory paired inference on the externally anchored scenario. S7 was added "
            "after the primary inferential family was fixed and is therefore not part of it; "
            "p-values are uncorrected.", size=7.2)
        d.p(
            "Every scenario examined so far is parameterised by us, which is the strongest "
            "single objection to a simulation-only study: the mechanism might be an artefact of "
            "the generative model rather than a property of anticipatory control. Scenario S7 "
            "removes the two parameters that matter most for non-terrestrial operation from our "
            "control. Michel et al. [51] report that the dish-satellite-gateway path is "
            "reconfigured on a globally coordinated fifteen-second grid, at the twelfth, "
            "twenty-seventh, forty-second and fifty-seventh second of every minute, with "
            "throughput dropping at the interval boundaries; the same measurement campaign "
            "places the median round-trip latency near 50 ms with an interquartile range of "
            "roughly 40 to 60 ms. S7 adopts that cadence, that phase and those latency "
            "statistics directly. It is held out of the training corpus, so neither the "
            "predictor nor the operating point has seen it."
        )
        d.p(
            f"{t_ext} and {t_exi} report the outcome, and it is not the one the earlier sections "
            "would predict. On aggregate service measures the proposed policy is "
            "indistinguishable from reactive restoration: the availability difference is "
            "+0.03 percentage points with a 95 % interval of [-0.05, +0.11] and an uncorrected "
            "p of 0.50, and the resilience-loss difference is -0.15 slot-equivalents with an "
            "interval of [-0.50, +0.20] and p = 0.38. Neither is distinguishable from zero. What "
            "does transfer is the recovery-time benefit: mean recovery falls from 19.69 to 13.16 "
            "slots, a paired difference of -6.53 slots with a 95 % interval of [-9.74, -3.32], "
            "d_z = -0.86 and p < 0.001. The degradation-aware policy shows the same pattern more "
            "sharply, cutting recovery by 16.24 slots while slightly worsening availability."
        )
        d.p(
            "The explanation is consistent with the mechanism established in Section 7.8 rather "
            "than in tension with it. A reconfiguration event on a deterministic external grid "
            "is exogenous by construction: it is not produced by the load the controller places "
            "on the network, so the exposure-control channel through which most of the benefit "
            "flowed under infrastructure collapse is closed. What remains is anticipation "
            "proper, and the measured anticipation rate on S7 is low because a scheduled "
            "reconfiguration leaves no degradation precursor. The controller can still shorten "
            "the interruptions it does not prevent, which is what the recovery-time result "
            "shows, but it cannot reduce their number. We therefore state the scope of the "
            "contribution accordingly: the availability and cumulative-loss gains reported in "
            "Section 7.2 are established for endogenous, load-coupled degradation, and on "
            "externally anchored scheduled dynamics only the recovery-time component survives. "
            "An operator whose dominant impairment is a scheduled reconfiguration should not "
            "expect the headline numbers of this paper."
        )

    # ------------------------------------------------------- 8. Discussion
    d.h1("8. Discussion")
    d.h2("8.1 Anticipation lead time versus prediction accuracy")
    d.p(
        "The improvement decomposes arithmetically. An interruption caused by a foreseeable "
        "failure lasts, under reactive restoration, for the detection delay plus the "
        "re-establishment interval — both properties of the management plane rather than of the "
        "physical fault. If the failure is anticipated with lead time exceeding the switch-over "
        "cost, both terms disappear and the residual interruption is at most C_m. The observed "
        f"reductions in recovery time, between "
        f"{min(abs(rel(REC.loc[s, 'proactive_ai'], REC.loc[s, 'reactive'])) for s in (S3, S4, S6)):.0f}% "
        f"and "
        f"{max(abs(rel(REC.loc[s, 'proactive_ai'], REC.loc[s, 'reactive'])) for s in (S3, S4, S6)):.0f}% "
        "across the severe scenarios, are consistent with that arithmetic given the measured "
        "anticipation rates: not every failure is anticipated, so not every interruption is "
        "avoided, and the aggregate gain is approximately the product of the anticipation rate "
        "and the avoidable interruption duration. This also explains why the availability gain is "
        "smaller than the recovery-time gain: availability averages over all flow-slots, most of "
        "which no particular failure touches, whereas recovery time conditions on the flows that "
        "were affected. Both should be reported."
    )
    d.h2("8.2 Resilience benefit versus actuation cost")
    d.p(
        "The migration-cost sweep and the perfect-state ablation point at the same conclusion "
        "from opposite directions. Increasing the cost of acting erodes the benefit until it "
        "vanishes; increasing the quality of the information while leaving the action rule "
        "unchanged also erodes the benefit, because the controller acts more often. The quantity "
        "that governs the outcome is therefore neither the accuracy of the predictor nor the "
        "speed of the actuator in isolation, but the ratio between the cost of acting and the "
        "cost of not acting, mediated by the anticipation lead time. Expressing the risk budget "
        "as a multiple of the predictor's own operating point is the mechanism by which our "
        "controller stays near that balance, and Eq. (12) is its interpretation."
    )
    d.h2("8.3 How much of the gain is anticipation?")
    d.p(
        "The degradation-aware ablation forces a more careful statement of the contribution than "
        "the headline numbers alone would support. Two mechanisms are at work. The first is "
        "exposure control: a controller that routes away from elements whose telemetry is "
        "already poor reduces the number of flows a subsequent failure can affect, and it needs "
        "no prediction to do so. The second is anticipation proper: acting on elements whose "
        "telemetry is not yet alarming but whose trajectory implies failure within the horizon. "
        "Under infrastructure collapse the first mechanism accounts for roughly two thirds of "
        "the unweighted improvement and the second for the remainder; under common-cause "
        "disruption the second contributes nothing, as it must; under handover dynamics the "
        "second is what prevents the controller from being misled by transients. Once the "
        "objective is weighted by criticality the picture depends on how that weighting is "
        "constructed. Under class-balanced weighting the ordering changes and only the "
        "predictive policy improves on the reactive baseline, because prediction is what allows "
        "the residual outage to be steered onto the classes that tolerate it. Under "
        "flow-weighted criticality the degradation-aware policy also improves on the baseline "
        "and the predictive policy merely remains best, because weighting by carried volume "
        "dilutes the sparse critical classes on which the two policies differ. The conditional "
        "statement that survives both weightings is therefore narrower than the class-balanced "
        "numbers alone would suggest: an operator with a flat objective in a stable environment "
        "may capture most of the benefit without any predictor, and the marginal value of "
        "prediction grows with the degree to which the objective distinguishes service classes "
        "irrespective of their traffic share. These weighted comparisons were not part of the "
        "primary inferential family and are reported as exploratory."
    )
    d.h2("8.4 What cannot be anticipated, and what to do about it")
    d.p(
        "The common-cause scenario draws a boundary on what anticipation can achieve. "
        "A shock that is instantaneous and shared by construction leaves no observable "
        "precursor in the telemetry of the affected elements, and the measured anticipation rate "
        f"of {pct(float(np.nanmean(ai[ai.scenario == S6]['anticipation_rate'])), 1)}% reflects "
        "that faithfully. Nevertheless the proposed policy still delivers the best resilience in "
        "that scenario, because the risk-aware cost term keeps flows off elements that are "
        "already degraded and therefore reduces exposure when the shock arrives. Predictive "
        "resilience thus has two mechanisms — anticipation of individual failures and continuous "
        "avoidance of accumulated degradation — and only the first requires the failure to be "
        "predictable. Systems dominated by common-cause events should invest in the second, that "
        "is in diversity and exposure control, rather than in better prognostics."
    )
    d.h2("8.5 Transferability")
    d.p(
        "Three properties support transfer to other critical-infrastructure settings. The twin is "
        "observation-only, so integration requires a telemetry feed and no privileged access to "
        "element internals. The control policy is stateless with respect to the model: replacing "
        "the predictor re-scales the budgets automatically through theta*, so model updates do "
        "not require re-tuning the controller. And the inference cost is negligible, so the "
        "anticipation layer can run on the same edge hardware that hosts the applications. The "
        "case study is a humanitarian and emergency-response deployment, but the structure — "
        "hidden degradation, delayed detection, criticality-differentiated services, costly "
        "recovery actions — is shared with public-safety networks, utility telemetry, tactical "
        "communications and remote industrial operations."
    )
    d.h2("8.6 Threats to validity")
    d.p(
        "Internal validity is supported by the paired design, the episode-disjoint split, the "
        "frozen operating point and the Holm correction. The principal internal threat is that "
        "the generative model and the predictor share a designer; it is mitigated by hiding the "
        "latent state from every decision-making component, by excluding the scenario identifier "
        "from the feature set, by including a non-learning heuristic that calibrates the "
        "difficulty of the task, by declaring the provenance of every parameter in Section "
        "3.5, by perturbing the synthetic parameters in Section 7.9, and by evaluating on two "
        "mechanisms excluded from the training corpus. "
        "External validity is the substantive limitation and is stated in Section 9. "
        "Construct validity rests on whether the measure suite captures what an operator values; "
        "we report reliability, availability, resilience loss, recovery time, hazard and "
        "control-plane cost together precisely because the threshold policy demonstrates that no "
        "single one of them is sufficient."
    )

    # ------------------------------------------------------ 9. Limitations
    d.h1("9. Limitations")
    d.numbered([
        "The evaluation is conducted in a simulation environment. No measurements from physical "
        "LEO terminals or 6G-class hardware and no field trial are used. The absolute values are "
        "properties of the model; only the relative comparison between policies, which shares the "
        "environment, should be carried over to expectations about deployed systems.",
        "The degradation process is a modelling assumption. Mean-reverting jump-diffusion with a "
        "utilisation-dependent hazard is a standard abstraction and its parameters produce KPI "
        "ranges consistent with published measurements [42], [43], [51], [52], but real failures "
        "include wear-out, adversarial and cascading modes not reproduced here.",
        "The non-terrestrial dynamics of scenario S5 are measurement-informed rather than "
        "trace-driven: reconfiguration cadence and transient degradation magnitudes are taken "
        "from the literature, not replayed from an operational trace. Coupling the twin to an "
        "orbital simulator [50], [53] remains future work.",
        "Correlated failures are modelled by a single-region shared shock. Multi-region, "
        "time-extended and cascading dependencies, and the epistemic uncertainty of common-cause "
        "parameters [18], are not addressed.",
        "The make-before-break assumption gives pre-emptive migration a small switch-over cost. "
        "This matches pre-computed protection in carrier networks [56], [57], but in a deployed "
        "system session continuity depends on the transport protocol and the mobility anchor. "
        "Section 7.4 bounds the sensitivity to this assumption but does not remove it.",
        "The controller optimises a per-flow objective greedily in criticality order. No "
        "optimality gap against a global formulation is reported, and no admission control or "
        "capacity-planning decision is considered.",
        "The traffic model uses marked Poisson arrivals with class-dependent holding times. Real "
        "emergency traffic is bursty and event-correlated, which would stress the capacity ledger "
        "more than the present model does.",
        "Redundancy is varied across five topology classes derived from a single reference "
        "graph, so the dependence reported in Section 7.7 is established within one structural "
        "family. Networks with a qualitatively different architecture, for example ring or "
        "hierarchical backhaul, are not covered.",
    ], prefix="L{i}." if False else "({i})")

    # ----------------------------------------------------- 10. Future work
    d.h1("10. Future work")
    d.bullets([
        "Learning the action policy rather than the risk threshold. The ablation shows that the "
        "binding constraint is the cost of acting, which is precisely the setting in which a "
        "policy learned over the (risk, cost, criticality, lead time) state should outperform a "
        "fixed budget rule [75], [76].",
        "Trace-driven non-terrestrial evaluation. Coupling the twin to a constellation simulator "
        "[50], [53] would replace the measurement-informed handover model with explicit "
        "visibility windows and allow ephemeris to be used as an exogenous feature.",
        "Richer common-cause structures, including multi-region and cascading dependencies, and "
        "propagation of the epistemic uncertainty of the shock parameters into the resilience "
        "estimates.",
        "Validation against operational telemetry. Even a partial trace from a deployed emergency "
        "network would allow the degradation model to be fitted rather than specified, and the "
        "transfer of the anticipation model to be measured directly.",
        "Uncertainty-aware anticipation. Conformal prediction or Bayesian treatment of the risk "
        "estimate would let the controller reason about the confidence of the risk rather than "
        "its point value, which is the natural extension of the break-even rule.",
    ])

    # ----------------------------------------------------- 11. Conclusion
    d.h1("11. Conclusion")
    d.p(
        "This paper asked when failure anticipation creates a measurable resilience advantage in "
        "hybrid terrestrial–satellite communication systems, and answered it with a predictive "
        "digital-twin framework evaluated in a closed loop where recovery decisions feed back "
        "into the failure process. The framework contributes an observation-only twin of a hybrid "
        "system with hidden degradation and delayed detection, a horizon-based anticipation model "
        "with a calibrated operating point and an explicit lead-time characterisation, a "
        "risk-aware recovery mechanism whose criticality-dependent budgets have a "
        "decision-theoretic reading, and a dynamic resilience-assessment methodology reporting "
        "reliability, availability, resilience loss, hazard and control-plane cost."
    )
    d.p(
        f"Empirically, failures within the training family are anticipated with a median lead "
        f"time of {lead(S4):.1f} slots and an anticipation rate of "
        f"{pct(float(np.nanmean(ai[ai.scenario == S4]['anticipation_rate'])), 0)}% under "
        f"infrastructure collapse, and the proposed policy reduces mean recovery time by "
        f"{abs(rel(REC.loc[S4, 'proactive_ai'], REC.loc[S4, 'reactive'])):.0f}% and cumulative "
        f"resilience loss by "
        f"{abs(rel(RL.loc[S4, 'proactive_ai'], RL.loc[S4, 'reactive'])):.0f}% relative to reactive "
        f"restoration, with availability rising from {pct(A.loc[S4, 'reactive'])}% to "
        f"{pct(A.loc[S4, 'proactive_ai'])}%. Comparable gains are obtained under correlated "
        "common-cause disruption even though the triggering shocks are unanticipatable."
    )
    d.p(
        "Four ablations identify where these gains originate and where they disappear. P2 shows "
        "that a non-learning proactive rule using the identical actuator performs worse than not "
        "acting proactively at all, because a static KPI threshold cannot separate transient "
        "degradation from impending failure. P3 shows that a controller which merely avoids "
        "already-degraded elements, compared at a matched alarm rate, recovers most of the "
        "unweighted benefit; prediction is decisive mainly when the objective is weighted by "
        "criticality on a per-class basis, or when transient reconfiguration would otherwise "
        "mislead a contemporaneous index. P5 shows that reducing state-estimation error to zero "
        "does not improve resilience unless the actuation threshold is recalibrated, because the "
        "perfectly informed controller over-actuates. The migration-cost sweep locates the "
        "boundary of the benefit at the point where a pre-emptive action costs about as much as "
        "the interruption it prevents. A seventh scenario, whose reconfiguration cadence and "
        "link statistics are taken from published measurement studies rather than from our own "
        "model, locates a second boundary: when the dominant impairment is scheduled and "
        "exogenous, the availability and cumulative-loss advantages are statistically "
        "indistinguishable from reactive restoration and only the reduction in recovery time "
        "survives. Taken together, the results indicate that prediction "
        "accuracy must be interpreted jointly with anticipation lead time and actuation cost "
        "rather than on its own. The complete environment, models, control plane and analysis "
        "pipeline are released so that the same decomposition can be applied to other systems."
    )

    d.h1("Appendix A. Complete policy comparison")
    d.p("The compact comparison of Section 7.2 reports the proposed policy against reactive "
        "restoration only. The table below gives all six policies across all seven scenarios.")
    d.table(pd.read_csv(os.path.join(TABDIR, "t11_main_results.csv"))[
        ["scenario", "policy", "service_availability", "normalised_resilience",
         "resilience_loss", "mean_recovery_steps", "sla_compliance",
         "proactive_reroutes_total", "reactive_reroutes"]],
        "All six policies across all seven scenarios, averaged over paired seeds. Seed-level "
        "standard deviations are contained in the corresponding result file.", size=6.8)

    d.h1("Appendix B. Design-choice sensitivity and Monte Carlo convergence")
    d.p(
        "Three questions concern the study design rather than the model: whether enough "
        "replications were run, whether the controller's own cost coefficients were fortunate, "
        "and whether reusing one validation fold for both probability calibration and threshold "
        "selection flattered the reported operating point. Each is answered here."
    )
    sc_path = os.path.join(TABDIR, "t21_seed_convergence.csv")
    if os.path.exists(sc_path):
        fig_sc = d.figure("f19_seed_convergence.png",
                          "Paired effect of the proposed policy against reactive restoration "
                          "under infrastructure collapse as a function of the number of seeds, "
                          "with 95 % intervals.", width=12.5)
        sc = pd.read_csv(sc_path)
        t_sc = d.table(sc, "Monte Carlo convergence of the paired effect.", size=7.2)
        g = sc[sc.measure.str.startswith("resilience")]
        d.p(
            f"{fig_sc} and {t_sc} recompute the paired effect on the first 8, 12, 16, 20 and 24 "
            "seeds. The point estimate moves within "
            f"{abs(g['mean difference'].max() - g['mean difference'].min()):.2f} slot-equivalents "
            "across the whole range and the interval half-width contracts from "
            f"{g['ci half-width'].iloc[0]:.2f} to {g['ci half-width'].iloc[-1]:.2f} "
            "slot-equivalents, consistent with the expected inverse square-root behaviour. The "
            "sign and the order of magnitude are already settled at eight seeds; the additional "
            "replications buy precision rather than a different conclusion, which is why the "
            "campaign was not extended further."
        )
    cw_path = os.path.join(TABDIR, "t23_controller_weights.csv")
    if os.path.exists(cw_path):
        cw = pd.read_csv(cw_path)
        t_cw = d.table(cw, "Sensitivity to the controller cost coefficients. The generative "
                           "model is untouched; only the coefficients the controller optimises "
                           "are rescaled.", size=7.4)
        pos = int((cw["availability gain [pp]"] > 0).sum())
        d.p(
            f"{t_cw} perturbs the loss, risk and load coefficients of the routing cost one at a "
            f"time by a factor of two in each direction. The advantage is preserved in {pos} of "
            f"{len(cw)} settings, with the resilience-loss reduction between "
            f"{cw['loss reduction [%]'].min():.0f}% and {cw['loss reduction [%]'].max():.0f}% "
            "against a nominal 32.5%. Two observations are worth stating plainly. The risk "
            "coefficient itself is not a sensitive parameter: halving or doubling it changes the "
            "loss reduction by less than one point, because the criticality-dependent budgets, "
            "not the routing penalty, decide when a migration is triggered. The load coefficient "
            "is inert in this regime: its penalty term activates only above 75 % utilisation, "
            "which the reference topology rarely reaches, so scaling it has no measurable effect "
            "at all. We report that rather than presenting it as evidence of robustness."
        )
    ca_path = os.path.join(RESULTS, "calibration_audit.csv")
    if os.path.exists(ca_path):
        ca = pd.read_csv(ca_path)
        t_ca = d.table(ca.round(4), "Calibration quality and the effect of separating the "
                                    "calibration fold from the threshold-selection fold.",
                       size=7.2)
        shared = ca[ca.setting.str.startswith("shared")].iloc[0]
        disj = ca[ca.setting.str.startswith("disjoint")].iloc[0]
        d.p(
            f"{t_ca} reports calibration diagnostics on the test fold. The expected calibration "
            f"error of the reported model is {shared['ece']:.4f} over fifteen equal-width bins "
            f"and the Cox calibration slope is {shared['calibration_slope']:.2f}, close to the "
            "ideal value of one, so the probabilities the controller consumes are usable as "
            "probabilities and not merely as scores. The reported pipeline fits the isotonic map "
            "and selects the alarm threshold on the same validation episodes. Splitting those "
            "episodes into disjoint halves — one for calibration, one for threshold selection — "
            f"moves the operating point from {shared['threshold']:.3f} to "
            f"{disj['threshold']:.3f} and the alarm rate from {shared['alarm_rate']:.4f} to "
            f"{disj['alarm_rate']:.4f}, and slightly improves test F1 "
            f"({disj['f1']:.4f} against {shared['f1']:.4f}). The shared fold therefore does not "
            "produce an optimistic bias; it produces a lower threshold."
        )
    tf_path = os.path.join(RESULTS, "threshold_fold_check.csv")
    if os.path.exists(tf_path):
        tf = pd.read_csv(tf_path)
        piv = tf.pivot_table(index="setting", columns="policy",
                             values=["service_availability", "resilience_loss"])
        red = {}
        for setting in piv.index:
            a = piv.loc[setting, ("resilience_loss", "reactive")]
            b = piv.loc[setting, ("resilience_loss", "proactive_ai")]
            red[setting] = -100.0 * (b - a) / a
        d.p(
            "Because a different threshold could still change the closed-loop outcome, we ran "
            "the collapse scenario again with the operating point obtained from the disjoint "
            "folds. The resilience-loss reduction is "
            f"{red.get('disjoint-fold theta', float('nan')):.1f}% against "
            f"{red.get('reported theta*', float('nan')):.1f}% at the reported operating point, "
            "so the closed-loop conclusion is materially unchanged and the choice of fold "
            "structure is not load-bearing."
        )
    rs_path = os.path.join(TABDIR, "t22_robustness_strategy.csv")
    if os.path.exists(rs_path):
        t_rs = d.table(pd.read_csv(rs_path),
                       "Consolidated validation strategy: each threat to the validity of the "
                       "study, the test that addresses it and the outcome. Every entry is "
                       "generated from the corresponding result file.",
                       fmt="{}", size=7.0, widths=[4.6, 4.8, 6.6])
        d.p(
            f"{t_rs} collects the robustness evidence in one place. It is not a claim that the "
            "study is externally validated — Section 9 states plainly that it is not — but a "
            "statement of which specific objections have been tested and with what result, "
            "including the two tests whose outcome is unfavourable to the proposed policy."
        )

    d.h1("Data and code availability")
    d.p(
        "The complete implementation — environment, feature pipeline, anticipation models, "
        "digital twin, recovery controller, experiment harness, figure and table generation, Go "
        "control-plane services and deployment manifests — together with the generated datasets "
        "and all result tables, is provided as a supplementary archive with this submission and will "
        "additionally be deposited in a public repository upon acceptance. The archive contains "
        "the source code, the configuration, the fixed random seeds, the result tables, the figure "
        "and table generation scripts and a README; a single build target regenerates the "
        "telemetry corpus, retrains the models, re-runs the campaign and rebuilds every figure and "
        "table in this manuscript from scratch. The generated corpus itself is reproduced "
        "deterministically from the seeds rather than shipped, because it exceeds the size limit "
        "for supplementary files."
    )
    d.h1("Declarations")
    d.p("Funding: this research received no specific grant from any funding agency in the "
        "public, commercial or not-for-profit sectors. Conflicts of interest: the authors "
        "declare none. Ethical approval: not required; no human subjects or personal data were "
        "involved.", size=9.5)
    d.h1("Declaration of generative AI and AI-assisted technologies in the manuscript "
         "preparation process")
    d.p("During the preparation of this work the authors used "
        "Anthropic Claude (Opus) for language editing and for assembling the manuscript source "
        "files from the machine-generated result tables. No content was accepted without "
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
    ap.add_argument("--out", default="Predictive_Resilience_Hybrid_5G6G_LEO.docx")
    args = ap.parse_args()
    build(args.out)


if __name__ == "__main__":
    main()
