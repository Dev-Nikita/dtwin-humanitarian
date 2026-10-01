"""
Response to reviewers, cover letter and highlights for revision R1 of
JRESS-D-26-06464. Every number is read from data/results.

    python -m python.paper.build_response_r1 --out ../submission_R1
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
from docx import Document
from docx.shared import Pt
from scipy import stats

RES = os.path.join("data", "results")
CAPTIONS = None  # path to <manuscript>.captions.json, set in main()
S3, S4, S5, S6, S7 = ("S3_leo_degradation", "S4_disaster_collapse", "S5_leo_handover",
                      "S6_common_cause", "S7_external_ntn")


def load():
    r = pd.read_csv(os.path.join(RES, "experiment_runs.csv"))
    RL = r.pivot_table(index="scenario", columns="policy", values="resilience_loss")

    def rel(s, p, q="reactive"):
        return 100 * (RL.loc[s, p] - RL.loc[s, q]) / RL.loc[s, q]

    def pc(s, a, b, m="resilience_loss"):
        sub = r[r.scenario == s]
        x = sub[sub.policy == a].sort_values("seed")[m].to_numpy(float)
        y = sub[sub.policy == b].sort_values("seed")[m].to_numpy(float)
        d = x - y
        h = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
        return d.mean(), d.mean() - h, d.mean() + h

    flows_equal = bool((r.groupby(["scenario", "seed"]).generated_flows.nunique() == 1).all())
    return r, RL, rel, pc, flows_equal


def build_texts():
    r, RL, rel, pc, flows_equal = load()
    PR = r.groupby(["scenario", "policy"]).proactive_reroutes_total.mean()
    REF = r.pivot_table(index="scenario", columns="policy", values="mean_recovery_failure")
    REC = r.pivot_table(index="scenario", columns="policy", values="mean_recovery_steps")
    sig = pd.read_csv(os.path.join(RES, "significance_tests.csv"))
    conf = sig[sig.family == "confirmatory"]
    n_hs = int((conf.p_holm < 0.05).sum())
    hs = pd.read_csv(os.path.join(RES, "horizon_sweep.csv")).sort_values("horizon")
    hz_ref = pd.read_csv(os.path.join(RES, "horizon_reactive_reference.csv")).resilience_loss.mean()
    kpi = json.load(open(os.path.join(RES, "kpi_alarm_scale.json")))
    deg = json.load(open(os.path.join(RES, "degradation_index_scale.json")))
    best = json.load(open(os.path.join(RES, "best_model.json")))
    mc = pd.read_csv(os.path.join(RES, "model_comparison.csv")).set_index("model")
    ms_off = pd.read_csv(os.path.join(RES, "misspec_offline.csv"))
    ms = pd.read_csv(os.path.join(RES, "misspec_closed_loop.csv"))
    lg = pd.read_csv(os.path.join(RES, "large_topology.csv"))
    dd = pd.read_csv(os.path.join(RES, "detection_delay.csv"))
    acct = pd.read_csv(os.path.join(RES, "accounting_check.csv"))
    fb = pd.read_csv(os.path.join(RES, "feedback_ablation_fixed.csv"))
    ds = pd.read_csv(os.path.join("paper-assets", "tables", "r_t_design.csv"))
    pa = pd.read_csv(os.path.join("paper-assets", "tables", "r_t_parameters.csv"))
    fr = pd.read_csv(os.path.join("paper-assets", "tables", "r_t_frontier.csv"))
    cls = pd.read_csv(os.path.join("paper-assets", "tables", "r_t_class_sla.csv"))

    def msL(s, p):
        return ms[(ms.scenario == s) & (ms.policy == p)].resilience_loss.mean()

    def lgL(s, p):
        return lg[(lg.scenario == s) & (lg.policy == p)].resilience_loss.mean()

    def ddL(s, p, D):
        return dd[(dd.scenario == s) & (dd.policy == p) & (dd.detection_delay == D)].resilience_loss.mean()

    def acred(s, p, flag):
        a = acct[acct.scenario == s]
        ref = a[a.policy == "reactive"].resilience_loss.mean()
        return -100 * (a[(a.policy == p) & (a.flag == flag)].resilience_loss.mean() - ref) / ref

    def frb(s, p):
        x = fr[(fr.scenario.str.startswith(s[:2])) & (fr.policy == p)].iloc[0]
        return x["loss at best"]

    f_fr = ms_off[ms_off.setting.str.startswith("frozen predictor (trained")].iloc[0]
    f_rt = ms_off[ms_off.setting.str.startswith("reference: retrained")].iloc[0]
    nomfb = fb[fb.coupling == "nominal"]
    fbmax = 0.0
    for (s, p), g in nomfb[nomfb.policy != "reactive"].groupby(["scenario", "policy"]):
        e = g[g.feedback == "endogenous"].resilience_loss.mean()
        q = g[g.feedback == "replayed_P1_load"].resilience_loss.mean()
        fbmax = max(fbmax, abs(e - q))

    # ---------------------------------------------------- confirmation and refs
    CT = pd.read_csv(os.path.join(RES, "confirmation_tests.csv"))
    CEQ = pd.read_csv(os.path.join(RES, "confirmation_equivalence.csv"))
    CR = pd.read_csv(os.path.join(RES, "confirmation.csv"))
    MAN = json.load(open(os.path.join(RES, "confirmation_manifest.json")))
    MC = pd.read_csv(os.path.join("paper-assets", "tables", "r_t_migration_classes.csv"))
    PM = pd.read_csv(os.path.join("paper-assets", "tables", "r_t_protection_modes.csv"))
    caps = json.load(open(CAPTIONS)) if CAPTIONS and os.path.exists(CAPTIONS) else []

    import re as _re

    def ref(key):
        for c in caps:
            if key in c:
                m = _re.match(r"^(Table|Fig\.) (\d+)", c)
                return f"{m.group(1)} {m.group(2)}"
        raise KeyError(f"no caption containing {key!r}")

    def cf(s, a, b, col="rel_pct"):
        return float(CT[(CT.scenario == s) & (CT.contrast == f"{a} - {b}")][col].iloc[0])

    def cfs(s, a, b):
        r_ = CT[(CT.scenario == s) & (CT.contrast == f"{a} - {b}")].iloc[0]
        return (f"{r_.mean_diff:+.2f} slot-equivalents, {r_.rel_pct:+.1f}%, 95% CI "
                f"[{r_.ci_low:.2f}, {r_.ci_high:.2f}]")

    def mcv(s, pol, col):
        x = MC[(MC.seeds == "confirmation") & (MC.scenario.str.startswith(s[:2]))
               & (MC.policy == pol)]
        return float(x[col].iloc[0])

    prot = [-cf(s, "protection", "reactive") for s in (S3, S4, S5, S6)]
    hr = CT[(CT.family == "exploratory") & CT.contrast.str.startswith("protection_oracle")]
    hrmax = -hr.rel_pct.min()
    eq_all = bool(CEQ.equivalent.all())
    b2 = CT[CT.family == "B2 secondary"]
    AF = "already failed (detection-delay bypass) [%]"
    FA = "false alarm [%]"
    gain0 = ddL(S4, "reactive", 0) - ddL(S4, "proactive_ai", 0)
    gain3 = ddL(S4, "reactive", 3) - ddL(S4, "proactive_ai", 3)
    gain5 = ddL(S4, "reactive", 5) - ddL(S4, "proactive_ai", 5)
    g6_0 = ddL(S6, "reactive", 0) - ddL(S6, "proactive_ai", 0)
    g6_5 = ddL(S6, "reactive", 5) - ddL(S6, "proactive_ai", 5)
    d43 = pc(S4, "proactive_ai", "degradation_aware")
    d43u = pc(S4, "proactive_ai_uniform", "degradation_aware_uniform")

    summary = [
        "We thank the editor and the four reviewers for an unusually careful set of reports.",
        "The reviewers' comments prompted an end-to-end implementation audit. The audit "
        "identified three issues that materially affected the original quantitative "
        "interpretation: (i) traffic, environment and controller shared one random-number "
        "generator, so policies that consumed different numbers of draws saw different "
        "arrivals and failures and the paired design was not actually paired (R3.3); (ii) the "
        "horizon-sweep table had been produced by an older lead-time tracker without the "
        "horizon bound (R3.2, R5.1); and (iii) a migration away from a path whose element had "
        "already failed — which the telemetry-driven policies frequently perform during the "
        "three-slot detection delay — was charged the make-before-break switch-over cost C_m "
        "instead of the restoration cost C_r, although no working path existed during the "
        "switch. A second pass over the corrected code found three further issues of smaller "
        "effect, also corrected: the durations of restoration and switch-over were drawn from "
        "the controller's random stream and were therefore not paired across policies (they "
        "are now keyed to the seed, the flow and the physical failure event); an action of "
        "duration C was counted as interrupting C + 1 slots (it now interrupts exactly C); and "
        "the standby path of the protection baseline could absorb traffic without capacity "
        "limits (switch-over now requires residual capacity, with reserved and optimistic "
        "variants as sensitivity checks). Unit tests now cover each of these points.",
        "With the corrections and the baselines the reviewers asked for, the central claim of "
        "the original submission — that failure anticipation drives a large resilience gain — "
        "is not supported, and we have rewritten the paper around what the evidence does "
        "support. The protection-based policies were introduced during the revision, so their "
        "comparisons on the development seeds were hypothesis-generating. We therefore froze "
        f"the code (commit recorded in the supplementary material), wrote a confirmation "
        "protocol with fixed hypotheses, families, corrections and equivalence margins, "
        "committed it together with its analysis script, and only then ran "
        f"{len(MAN.get('seeds', []))} seeds that had never been used ({len(CR):,} runs). The "
        "results reported below and in the abstract come from that experiment unless stated "
        "otherwise.",
        "In the evaluated systems the corrected evidence gives a layered answer to the "
        "question in the new title. Plain 1:1 protection, which uses no risk information, "
        f"reduces cumulative resilience loss relative to reactive restoration by "
        f"{min(prot):.1f}–{max(prot):.1f}%. Using the risk estimate only for decisions that "
        "interrupt no traffic — primary-path choice and standby refresh (new policy P6) — adds "
        f"a confirmed {-cf(S4, 'protection_predictive', 'protection'):.1f}% under "
        f"infrastructure collapse and {-cf(S6, 'protection_predictive', 'protection'):.1f}% "
        "under common-cause disruption (primary family B1, both Holm p < 0.0001). For that "
        "role a contemporaneous degradation index is "
        + ("statistically equivalent to " if eq_all else "practically indistinguishable from ")
        + f"the learned predictor (TOST, family B3), and in exploratory comparisons perfect state "
        f"adds at most {hrmax:.1f}%. Predictive migration (P4) is not significantly different "
        "from protection under collapse and non-terrestrial degradation after multiplicity "
        "correction (under collapse the point estimate favours migration), is worse under handover dynamics and "
        "common-cause disruption, and its gain under common-cause disruption is almost "
        "entirely a bypass of the detection delay; the rest of its gain exists where "
        "degradation is gradual and is sensitive to the switch-over cost, the topology and the "
        "seed set.",
        "We recognise that the revised paper makes a more modest and different claim than the "
        "original. The title keeps the subject and states the question: 'Where Does Predictive "
        "Resilience Come From? A Simulation-Based Digital-Twin Assessment of Failure "
        "Anticipation, Protection and Risk-Aware Recovery in Hybrid 5G/6G-Class and LEO "
        "Satellite Networks'. The abstract, research questions, contributions, results, "
        "discussion and conclusions are new; the system model, the framework and the related "
        "work are retained with the clarifications requested.",
        f"Summary of new work: the development campaign was rerun (15 policies × 7 scenarios × "
        f"24 paired seeds, {len(r):,} runs); the frozen confirmation experiment was added "
        f"({len(CR):,} runs); and further closed-loop experiments were added (detection-delay "
        "sweep, actuation-effort frontier, migration-cost sweep, misspecified degradation "
        "world, 98-element topology, fixed-traffic feedback ablation, standby-capacity models, "
        "accounting check, controller design sensitivity, parameter, weight and topology "
        "sensitivity, horizon sweep, and an exploratory decomposition of predictive protection "
        "run after the confirmation, which attributes its effect almost entirely to the "
        "risk-aware standby path). Results are organised by mechanism. The inferential "
        "family of the original study (P4 against P1, twenty tests) is retained as Family A "
        "for continuity; Family B is the frozen confirmation. Reviewer comments are answered "
        "point by point below; section, table and figure numbers refer to the revised "
        "manuscript.",
    ]

    R = {}
    R["R4.1"] = ("The use of the term digital twin should be clarified; add 'simulation-based'.",
        "Agreed. The framework is called a simulation-based digital twin in the title, the "
        "abstract and at first introduction (Section 4.1), where we also state that the physical "
        "counterpart is itself a simulation, that the twin has not been synchronised with a "
        "deployed network and that no result should be read as validation on a real system. "
        "Section 8.4 explains what distinguishes the construction from a simulator (see also R1.1).")
    R["R4.2"] = ("Relationship between slot and real time.",
        "Added in Section 3.4 and the parameter tables: one slot corresponds to one second, the "
        "mapping that anchors the 15-slot reconfiguration grid of S5 and S7 to the 15 s Starlink "
        "schedule [52]. Hence D = 3 s, C_r = 4–12 s, C_m = 0–1 s and H = 10 s. We also state how "
        "interruptions are counted: an action taken in slot t with duration C interrupts the "
        "flow in slots t, …, t + C − 1, exactly C whole slots, so a switch-over costs zero or one "
        "slot and no fractional interruption arises (see R3.8). The original text described "
        "C_m as U(0,1); the implementation draws an integer from {0, 1}, and the text now says so.")
    R["R4.3"] = ("Criterion for selecting the prediction model.",
        "Section 4.3 now states the criterion, fixed in advance and applied on validation data "
        "only: highest validation PR-AUC (the positive class is rare); models within 0.01 are "
        "tied and the tie is broken by validation Brier score (the controller consumes "
        f"probabilities) and then by inference cost. {ref('Anticipation performance on episode-disjoint')} "
        "now shows the validation columns. Logistic regression has the highest validation "
        f"PR-AUC ({mc.loc['logistic_regression', 'val_pr_auc']:.3f}); no other model is within "
        "the tolerance. The code was changed accordingly — the original version selected by "
        "test PR-AUC, which we should not have done.")
    R["R4.4"] = ("Distinguish the exploratory nature of S7.",
        "Done. S7 is described as an externally parameterised stress test, not an external "
        "validation, in Sections 6.2 and 7.8, and Section 6.2 uses your wording 'exploratory, "
        "externally anchored scenario'; it is outside every inferential family and all "
        "its comparisons are labelled exploratory. The revised abstract and conclusions do not "
        "use S7 as validation.")

    R["R1.1"] = ("Essential difference between the framework and a conventional network simulator.",
        "Sections 4.1 and 8.4 now answer this directly. The distinction is structural: the "
        "simulated physical system evolves on its own; the twin is a separate estimator that "
        "sees it only through telemetry, holds a delayed and noisy belief, answers predictive "
        "questions and acts back; an assessment layer with privileged access scores the "
        "outcome. This separation is what made the main findings measurable — the "
        "detection-delay bypass exists only because the twin's belief lags the physical state, "
        "and the dead-path accounting issue could be found only because the assessment layer "
        "knows what the twin does not. We use 'simulation-based digital twin' throughout and do "
        "not claim synchronisation with a real network.")
    R["R1.2"] = ("19 nodes are insufficient for satellite networks.",
        "We added a 98-element topology with three LEO orbital planes of twelve satellites, "
        "intra- and inter-plane links, sixteen ground gateways and a doubled terrestrial "
        f"segment (Sections 6.2 and 7.8, {ref('98-element topology with three LEO planes')}). "
        "With the predictor and all operating points frozen from the reference network, the "
        f"losses under collapse are P1 {lgL(S4, 'reactive'):.2f}, P1b {lgL(S4, 'protection'):.2f}, "
        f"P6 {lgL(S4, 'protection_predictive'):.2f} and P4 {lgL(S4, 'proactive_ai'):.2f}, and "
        f"under common-cause disruption P1 {lgL(S6, 'reactive'):.2f}, P4 "
        f"{lgL(S6, 'proactive_ai'):.2f}, P1b {lgL(S6, 'protection'):.2f} and P6 "
        f"{lgL(S6, 'protection_predictive'):.2f}. On the larger graph predictive migration does "
        "relatively better under collapse than on the reference network, and the protection "
        "family remains ahead under common-cause disruption; the paper reports the mechanism "
        "ranking under collapse as system-dependent for this reason. We state as a limitation "
        "that the model has no orbital mechanics and is not representative of an operational "
        "constellation.")
    R["R1.3"] = ("The predictor mainly learns the degradation patterns predefined by the authors.",
        "We agree, and the revision treats this as a central limitation rather than a detail. "
        "We built a second, structurally different world — monotone gamma-process wear-out "
        "without mean reversion, observed through a four-slot lagged map with different "
        "exponents and doubled noise — and applied the frozen predictor and controllers to it "
        f"(Section 7.8, {ref('structurally different world')}). The predictor's PR-AUC falls to "
        f"{f_fr['pr_auc']:.3f} (lift {f_fr['lift']:.1f}) against {f_rt['pr_auc']:.3f} for the "
        "same estimator retrained there; the closed-loop ordering is preserved (collapse: "
        f"P1 {msL(S4, 'reactive'):.2f}, P1b {msL(S4, 'protection'):.2f}, "
        f"P4 {msL(S4, 'proactive_ai'):.2f}, P6 {msL(S4, 'protection_predictive'):.2f}). The "
        "paper now separates the two claims explicitly: the framework and the decomposition "
        "transfer across the two families we built; the predictor is validated only within the "
        "family it was trained on, and no claim is made for real failure mechanisms "
        "(Limitation 2).")
    R["R1.4"] = ("Additional baselines, e.g. protection mechanisms.",
        "Added: P1b, 1:1 protection with a pre-computed node-disjoint standby path and "
        "detection-driven, capacity-checked switch-over at cost C_m [57], [58], [79]. Section "
        "2.4 now also discusses protection and failure handling for LEO constellations "
        "specifically — local detour routing around failed inter-satellite links [82] and the "
        "recovery schemes surveyed in [81] — and Section 6.3 states that P1b represents this "
        "class in its simplest, risk-free form. This "
        "baseline changed the conclusions. On the 64 confirmation seeds it reduces the loss "
        f"relative to P1 by {min(prot):.1f}–{max(prot):.1f}% in S3–S6, and predictive migration "
        f"is not significantly different from it under collapse after Holm correction (P4 − P1b: {cfs(S4, 'proactive_ai', 'protection')}) "
        f"and worse under common-cause disruption ({cfs(S6, 'proactive_ai', 'protection')}). It "
        "also led to the new policy P6, which adds risk-driven primary-path choice and standby "
        "refresh to protection; P6 − P1b is the primary confirmatory hypothesis (Section 6.5) "
        f"and is supported under collapse ({cfs(S4, 'protection_predictive', 'protection')}) "
        f"and common-cause disruption ({cfs(S6, 'protection_predictive', 'protection')}). "
        f"Because protection carries much of the benefit, {ref('standby-capacity model')} checks "
        "that the result holds when standby capacity is reserved and when capacity limits are "
        "ignored; the P6 − P1b interval excludes zero under all three capacity models. An "
        f"exploratory ablation on the confirmation seeds ({ref('Exploratory decomposition of predictive protection')}) "
        "shows that the effect comes almost entirely from the risk-aware choice of the "
        "standby path, not from the choice of the primary path.")
    R["R1.5"] = ("How much of the improvement is truly contributed by failure anticipation?",
        "This is now the question the paper is built around (RQ2, Sections 7.2–7.7). The "
        "answer depends on the action. (a) In protection, very little: a contemporaneous index "
        + ("is statistically equivalent to " if eq_all else "performs like ")
        + f"the predictor ({ref('Family B3')}), and perfect state adds at most {hrmax:.1f}% "
        f"({ref('headroom of perfect-state')}). (b) In migration, part of the gain is not "
        "anticipation at all. Classifying every trigger by ground truth "
        f"({ref('Ground-truth classification of migration')}) shows that under common-cause "
        f"disruption {mcv(S6, 'P4', AF):.0f}% of P4's migrations leave elements that have "
        "already failed but are not yet confirmed down; sweeping the detection delay shows "
        f"that its advantage over P1 there is {g6_0:.2f} slot-equivalents at D = 0 and "
        f"{g6_5:.2f} at D = 5 — a pure detection-delay effect. Under collapse "
        f"{mcv(S4, 'P4', AF):.0f}% of triggers are of that kind, and about "
        f"{100 * gain0 / gain3:.0f}% of the gain at the nominal D = 3 survives at D = 0 "
        f"({gain0:.2f} of {gain3:.2f}): that part is not attributable to the bypass and "
        "includes anticipation of failures and the avoidance of degraded elements. (c) Among migration "
        "policies the risk source matters: with differentiated budgets prediction is ahead of "
        f"the index under collapse ({d43[0]:+.2f} [{d43[1]:.2f}, {d43[2]:.2f}]), but with "
        f"uniform treatment the order reverses ({d43u[0]:+.2f} [{d43u[1]:.2f}, {d43u[2]:.2f}]), "
        "and perfect state used for migration has the lowest loss of all policies under "
        "collapse in the development campaign at the nominal settings (an exploratory result); uniform index-driven migration (P3u) is also ahead "
        "of protection there. Anticipation therefore contributes through migration where "
        "degradation is gradual and the switch-over is cheap, and hardly at all through "
        "protection.")
    R["R1.6"] = ("Budget factors and weights 4, 3, 0.6: why these values? Sensitivity analysis.",
        "Section 4.4 explains the normalisation argument used to fix them before the campaign, "
        f"and Section 7.10 ({ref('controller design choices')}) now reports a design "
        "sensitivity on 24 paired seeds, with the P1b reference and the paired P6 − P1b "
        "interval for every setting: four budget-factor profiles (flat, shallow, nominal, "
        "steep) and one-at-a-time changes of the loss coefficient (2, 8), risk base (1.5, 6) "
        f"and criticality slope (0, 1.2). P4 ranges {ds['P4 loss'].min():.2f}–"
        f"{ds['P4 loss'].max():.2f} and P6 {ds['P6 loss'].min():.2f}–{ds['P6 loss'].max():.2f}; "
        "the steep profile hurts P4 and the nominal one is within noise of the best for it, so "
        "the nominal configuration does not handicap predictive migration; P6 has a lower "
        "loss than protection alone in every setting, with an interval excluding zero in all "
        "but one. "
        f"{ref('cost weights of Eq. (7)')} adds the Eq. (7) weights.")
    R["R1.7"] = ("In S7 the advantage disappears except recovery time; explain.",
        f"With the corrections, P4 is within noise of P1 in S7 ({rel(S7, 'proactive_ai'):+.1f}%, "
        f"interval [{pc(S7, 'proactive_ai', 'reactive')[1]:.2f}, "
        f"{pc(S7, 'proactive_ai', 'reactive')[2]:.2f}]), and its apparent recovery-time "
        "advantage was a composition effect of short migration-induced interruptions (see "
        "R5.4). The explanation (Section 7.8) follows from the decomposition: the dominant "
        "impairment in S7 is a short, scheduled reconfiguration without precursor; element "
        "failures are few, and although P4 anticipates about half of them it performs many "
        "more migrations than the calibrated perfect-state policy, most of which do not pay. "
        "Its failure-induced recovery in S7 is "
        f"{REF.loc[S7, 'proactive_ai']:.2f} slots against {REF.loc[S7, 'reactive']:.2f} for P1, "
        "so the all-interruption advantage is again a composition effect (Section 7.8). "
        "Protection and "
        f"P6 still improve on P1 in S7 ({rel(S7, 'protection'):+.1f}% and "
        f"{rel(S7, 'protection_predictive'):+.1f}%). We agree that the result limits the "
        "applicability of anticipatory migration, and the revised paper says so.")

    R["R5.1"] = ("Inconsistent lead times in Table 16 (9.67, 10.67 for H = 3, 5).",
        "You are right; this was an error. The horizon sweep had been produced by an older "
        "lead-time tracker that matched each failure to the first threshold crossing without "
        "the horizon bound of the lead-time definition, and the table was not regenerated when "
        "the tracker was replaced. The sweep now passes H to the tracker, the bound is asserted "
        "in the code and enforced by a unit test. Corrected median lead times range "
        f"{hs.cl_lead_time_median.min():.1f}–{hs.cl_lead_time_median.max():.1f} slots, never "
        f"above H ({ref('Prediction-horizon sweep')}, {ref('Effect of the prediction horizon')}). "
        "The conclusions that relied on this column were re-derived (see R3.2).")
    R["R5.2"] = ("Isolate predictive information from service differentiation; direct P3–P4 comparisons.",
        "The revised design makes both factors explicit (Section 6.3): P3 and P4 share "
        "budgets, the criticality term and criticality-ordered processing; uniform variants P3u "
        "and P4u remove all three. Direct paired P3–P4 contrasts are in "
        f"{ref('Exploratory paired contrasts')} for every scenario and the 2 × 2 analysis is in "
        "Section 7.7; the uniform-variant contrasts (P4u − P3u, P3 − P3u, P4 − P4u) are also "
        "rows of that table. The comparison confirms your concern: under collapse prediction is ahead "
        f"of the index with differentiation ({d43[0]:+.2f} [{d43[1]:.2f}, {d43[2]:.2f}]) but "
        f"behind it with uniform treatment ({d43u[0]:+.2f} [{d43u[1]:.2f}, {d43u[2]:.2f}]), "
        "because differentiated budgets hurt the index-driven policy there. Under handover "
        "dynamics prediction is ahead under both treatments, since the index mistakes "
        "reconfiguration transients for degradation. We also compare all risk sources at "
        f"matched actuation effort ({ref('number of proactive migrations per run')}, "
        f"{ref('Best operating point of each risk source')}).")
    R["R5.3"] = ("Confounded feedback ablation; Table 20 retains a benefit at g = 0.",
        "You are right, and the claim that two thirds of the gain came from feedback is "
        "withdrawn. The ablation is repeated with traffic held fixed: P1 is run with its "
        "endogenous load, its utilisation trajectory is recorded, and P3, P4, P1b and P6 are "
        "evaluated with the degradation process driven by that replayed trajectory on the same "
        f"seeds (Section 7.10, {ref('Feedback ablation at fixed traffic')}). At nominal coupling "
        f"the two arms differ by at most {fbmax:.2f} slot-equivalents for any policy, and the "
        "parameter sweep with the feedback gain switched off or doubled changes no conclusion. "
        "The closed loop is kept as the correct evaluation structure, not as a source of "
        "benefit.")
    R["R5.4"] = ("97.66% availability vs 41% SLA compliance; per-class SLA-compliant availability; "
                 "separate failure- and migration-induced recovery.",
        f"Both added. {ref('SLA-compliant availability (SLA-A) per service class')} reports "
        "availability and SLA-compliant availability per service class under collapse: "
        "connectivity is largely maintained but the surviving non-terrestrial paths cannot "
        "meet the latency/loss budgets of the tighter classes, so SLA compliance is low for "
        "every policy — recovery policies restore connectivity, they cannot restore removed "
        f"capacity (Section 7.4). {ref('Recovery time separated by the cause')} separates "
        "recovery time by cause. Your concern was justified: P4's all-interruption mean "
        f"recovery ({REC.loc[S4, 'proactive_ai']:.2f} vs {REC.loc[S4, 'reactive']:.2f} slots "
        "for P1 under collapse) is a composition effect; its failure-induced recovery is "
        f"{REF.loc[S4, 'proactive_ai']:.2f} vs {REF.loc[S4, 'reactive']:.2f}. Protection and P6 "
        f"shorten failure-induced recovery to {REF.loc[S4, 'protection']:.2f} and "
        f"{REF.loc[S4, 'protection_predictive']:.2f}. Mean recovery over all interruptions is no "
        "longer used as a headline.")

    R["R3.1"] = ("Predictive performance derived from the same synthetic latent-health model; "
                 "distinguish framework and predictor validation; independently generated degradation.",
        "See R1.3. We implemented an independently structured degradation world (gamma "
        "wear-out, lagged misspecified observation map) and report both offline transfer and "
        "the closed-loop consequence (Section 7.8). The manuscript now distinguishes explicitly "
        "between validation of the framework and decomposition (which transfer between the two "
        "families) and of the predictor (valid only within its training family), in Section "
        "7.8, Limitation 2 and the conclusion. A trace-driven evaluation remains future work; we "
        "did not have access to operational failure traces.")
    R["R3.2"] = ("Table 16 inconsistency with Eq. (18).",
        "See R5.1 — corrected, with the cause explained. The re-derived sweep shows that the "
        f"loss of P4 is lowest, on 12 seeds and without a significant difference, at a short horizon (H = {int(hs.loc[hs.cl_resilience_loss.idxmin(), 'horizon'])}, "
        f"{hs.cl_resilience_loss.min():.2f}) and rises to {hs.cl_resilience_loss.max():.2f} at "
        f"longer ones, against {hz_ref:.2f} for P1, because longer horizons trigger more "
        "migrations on elements that do not fail soon; P6 is insensitive to H. The "
        "recommendation to keep the horizon short is therefore retained for migration-based "
        "control, with the corrected reason (Section 7.10).")
    R["R3.3"] = ("Different flow counts between policies; common random numbers.",
        "Thank you — this identified a genuine defect. Traffic, environment and controller "
        "shared one RNG, so policies consuming different numbers of draws saw different "
        "arrivals and failures. The revision implements common random numbers (Section 6.4): "
        "separate SeedSequence streams for traffic, controller, common-cause process and every "
        "element; each element draws a fixed-size random block per slot and every random "
        "quantity (diffusion, jump occurrence and size, repair time, post-repair health, "
        "handover outcome, observation noise) is read from a fixed position. Restoration and "
        "switch-over durations are keyed draws determined by the seed, the flow and the "
        "physical failure event, so two policies that restore the same flow after the same "
        "failure draw the same duration whatever order they decide in. The number of generated "
        "flows is now identical across policies for every (scenario, seed) "
        f"({'verified for all' if flows_equal else 'NOT verified for all'} 168 scenario-seed "
        "pairs), and unit tests assert that the random blocks coincide across policies and "
        "that keyed durations do not depend on the order of decisions. All results were "
        "recomputed.")
    R["R3.4"] = ("Specify P2 fully; compare P2 and P4 at matched alarm or migration rate.",
        "P2 is now fully specified (Section 6.3): the KPI severity rule, the nominal operator "
        "limits, the same hysteresis and k = 3 candidates as P4, and exclusion of alarmed "
        "elements from routing; the loss limit equals the loosest class loss budget of Table 2 "
        "and the other limits are conventional operator thresholds fixed before the study. We added P2m with the alarm scale matched on validation runs "
        f"to the predictor's per-element alarm rate (scale {kpi['kpi_scale_at_operating_point']:.3f}), "
        "and a frontier sweeping the operating point of every risk source "
        f"({ref('number of proactive migrations per run')}). Your suspicion was correct: "
        "relaxed to a matched or lower alarm rate, P2 comes close to the other migration "
        f"policies (collapse, best points {frb(S4, 'P2'):.2f} for P2, {frb(S4, 'P3'):.2f} for P3, "
        f"{frb(S4, 'P4'):.2f} for P4). The claim that proactive action on conventional "
        "thresholds is worse than reactive restoration has been removed; it was a property of "
        "one aggressive configuration.")
    R["R3.5"] = ("On which data was the P3 alarm-rate matching performed?",
        "It was not documented, and we re-did it under an explicit protocol (Section 6.3): the "
        "index quantile and the KPI scale are estimated on dedicated validation runs of P1 in "
        "S2–S4 on six seeds (9000 + 13k, k = 0…5) disjoint from every evaluation seed, at the "
        f"predictor's closed-loop per-element alarm rate on those runs ({100 * deg['alarm_rate']:.2f}% "
        f"of {deg['n_element_slots']:,} element-slots), and frozen before any evaluation run. "
        "Section 6.3 now states these values and explains why this closed-loop alarm rate "
        "differs from the offline rates in the predictor tables. The source of each scale is "
        "recorded in the result files.")
    R["R3.6"] = ("P5 mapping undefined; perfect state with a mis-calibrated map is not an oracle.",
        "Agreed. The original map is now defined (linear in health between 0.35 and 0.18) and "
        "retained as P5 to show its consequence. We added P5c, the calibrated perfect-state "
        "reference: the exact probability that an element at health h fails within H, computed "
        "by Monte Carlo simulation of the generative model per scenario and element class. "
        f"Under collapse P5c performs {PR.loc[(S4, 'oracle_calibrated')]:.0f} migrations per run "
        f"against {PR.loc[(S4, 'oracle')]:.0f} for P5 and reaches a loss of "
        f"{RL.loc[S4, 'oracle_calibrated']:.2f} against {RL.loc[S4, 'oracle']:.2f} for P5, "
        f"{RL.loc[S4, 'proactive_ai']:.2f} for P4 and {RL.loc[S4, 'protection']:.2f} for P1b — "
        "the lowest of all policies there in the development campaign at the nominal settings (exploratory); under common-cause disruption it stays behind "
        f"protection ({RL.loc[S6, 'oracle_calibrated']:.2f} against {RL.loc[S6, 'protection']:.2f}). "
        "The gap between P4 and P5c is the headroom a better forecaster could exploit in the "
        "migration role. The statement that perfect state knowledge leads to excessive "
        "actuation has been removed; it applied only to the uncalibrated map.")
    R["R3.7"] = ("Eq. (12) presented too strongly.",
        "Section 4.5 is rewritten as 'An ex-post economic reading of the budget': the rule is "
        "stated to be a heuristic, the thresholds are not derived from Eq. (12), F1 contains no "
        "costs and R_inf is a proxy. Cost-derived actuation is named as future work.")
    R["R3.8"] = ("Physical meaning of a slot; fractional C_m.",
        "See R4.2. One slot = 1 s; C_m is an integer in {0, 1} (the original text said U(0,1), "
        "which was wrong); an action of duration C interrupts exactly C whole slots, on the "
        "same grid for availability, recovery and Kaplan–Meier, so no fractional interruption "
        "arises and a zero-cost switch-over interrupts nothing (Section 3.4; a unit test checks "
        "this).")
    R["R3.9"] = ("'All policies indistinguishable' under S1 is inconsistent with P2.",
        "Corrected; the sentence was removed and the whole Results and Conclusion text was "
        "rewritten. Every number in the text is now generated from the result files by the "
        "same script that produces the tables, an automated check scans the manuscript, "
        "highlights and letters for claims of the original version, and the qualitative "
        "statements were checked against the result files in an independent pass before "
        "resubmission.")
    R["R3.10"] = ("State the incremental value of prediction more narrowly.",
        "Fully agreed. The paper is now framed as a decomposition: protection carries most of "
        "the gain; risk-informed standby selection adds a small confirmed improvement for "
        "which a simple index suffices; and pre-emptive migration on predicted risk pays only "
        "where degradation is gradual, detection is slow and the switch-over is nearly free. "
        "The abstract and conclusion state these findings with the qualifier 'in the "
        "evaluated systems'.")
    R["R3.11"] = ("Criticality-weighted result is exploratory.",
        "The class-weighted aggregates are no longer used for any conclusion (Section 5.2); "
        "per-class availability and SLA-compliant availability are reported instead "
        f"({ref('SLA-compliant availability (SLA-A) per service class')}). The corrected design "
        "did not reproduce the weighted-loss advantage, and the claim has been removed from "
        "the abstract and conclusions.")
    R["R3.12"] = ("S7 is externally parameterised stress testing, not validation.",
        "Agreed; this wording is used throughout (Sections 6.2, 7.8). S7 is outside every "
        "inferential family and is discussed as the scenario where there is least to "
        "anticipate.")

    return summary, R, dict(n_hs=n_hs)


REVIEW_ORDER = ["R4.1", "R4.2", "R4.3", "R4.4", "R1.1", "R1.2", "R1.3", "R1.4", "R1.5", "R1.6",
                "R1.7", "R5.1", "R5.2", "R5.3", "R5.4"] + [f"R3.{i}" for i in range(1, 13)]
REVIEWER_TITLE = {"R4": "Reviewer 4", "R1": "Reviewer 1", "R5": "Reviewer 5", "R3": "Reviewer 3"}


def write_response(out_dir):
    summary, R, _ = build_texts()
    doc = Document()
    st = doc.styles["Normal"]
    st.font.name = "Times New Roman"
    st.font.size = Pt(11)
    doc.add_heading("Response to the reviewers — JRESS-D-26-06464 (revision 1)", level=1)
    md = ["# Response to the reviewers — JRESS-D-26-06464 (revision 1)\n"]
    doc.add_heading("Summary of the revision", level=2)
    md.append("## Summary of the revision\n")
    for p in summary:
        doc.add_paragraph(p)
        md.append(p + "\n")
    cur = None
    for key in REVIEW_ORDER:
        rv = key.split(".")[0]
        if rv != cur:
            cur = rv
            doc.add_heading(REVIEWER_TITLE[rv], level=2)
            md.append(f"## {REVIEWER_TITLE[rv]}\n")
        q, a = R[key]
        p = doc.add_paragraph()
        p.add_run(f"Comment {key.split('.')[1]}. ").bold = True
        p.add_run(q).italic = True
        doc.add_paragraph("Response. " + a)
        md.append(f"**Comment {key.split('.')[1]}.** *{q}*\n\nResponse. {a}\n")
    doc.save(os.path.join(out_dir, "Response_to_Reviewers.docx"))
    open(os.path.join(out_dir, "Response_to_Reviewers.md"), "w").write("\n".join(md))


def _conf():
    CT = pd.read_csv(os.path.join(RES, "confirmation_tests.csv"))
    CEQ = pd.read_csv(os.path.join(RES, "confirmation_equivalence.csv"))

    def cf(s, a, b):
        return float(CT[(CT.scenario == s) & (CT.contrast == f"{a} - {b}")].rel_pct.iloc[0])

    prot = [-cf(s, "protection", "reactive") for s in (S3, S4, S5, S6)]
    hr = CT[(CT.family == "exploratory") & CT.contrast.str.startswith("protection_oracle")]
    return cf, prot, -hr.rel_pct.min(), bool(CEQ.equivalent.all())


def headline_numbers():
    """Numbers that must appear verbatim in the manuscript and the response letter."""
    cf, prot, hrmax, _ = _conf()
    p6s4 = f"{-cf(S4, 'protection_predictive', 'protection'):.1f}%"
    p6s6 = f"{-cf(S6, 'protection_predictive', 'protection'):.1f}%"
    rng = f"{min(prot):.1f}–{max(prot):.1f}%"
    hr = f"{hrmax:.1f}%"
    out = {}
    for doc in ("manuscript", "response"):
        out[f"{doc}:B1 S4"] = p6s4
        out[f"{doc}:B1 S6"] = p6s6
        out[f"{doc}:protection range"] = rng
        out[f"{doc}:oracle headroom"] = hr
    return out


def write_cover(out_dir):
    r, RL, rel, pc, _ = load()
    cf, prot, hrmax, eq_all = _conf()
    text = [
        "1 October 2026",
        "Editor, Reliability Engineering & System Safety",
        "Re: Revised manuscript JRESS-D-26-06464",
        "Dear Editor,",
        "We submit the revised version of our manuscript, now titled 'Where Does Predictive "
        "Resilience Come From? A Simulation-Based Digital-Twin Assessment of Failure "
        "Anticipation, Protection and Risk-Aware Recovery in Hybrid 5G/6G-Class and LEO "
        "Satellite Networks'.",
        "The reviewers' comments prompted an end-to-end implementation audit. The audit "
        "identified three issues that materially affected the original quantitative "
        "interpretation — the paired design was not correctly paired, one table used an "
        "outdated lead-time computation, and migrations away from already-failed paths were "
        "scored with the switch-over instead of the restoration cost — and a second pass "
        "found three further issues of smaller effect. All are corrected and covered by unit "
        "tests. With the corrections and the baselines the reviewers requested, above all "
        "1:1 protection, the corrected results materially changed the quantitative "
        "interpretation, and we therefore revised the paper around the evidence supported by "
        "the corrected implementation.",
        "Because the protection-based policies were introduced during the revision, we froze "
        "the code and wrote a confirmation protocol with fixed hypotheses, multiplicity "
        "corrections and equivalence margins before running 64 previously unused seeds. In the "
        f"evaluated systems, plain protection reduces resilience loss by {min(prot):.1f}–"
        f"{max(prot):.1f}% relative to reactive restoration; risk-informed standby selection "
        f"adds a confirmed {-cf(S4, 'protection_predictive', 'protection'):.1f}% under "
        f"infrastructure collapse and {-cf(S6, 'protection_predictive', 'protection'):.1f}% "
        "under common-cause disruption, with a simple degradation index "
        + ("statistically equivalent to" if eq_all else "performing like")
        + " the learned predictor in that role; and pre-emptive migration on predicted risk "
        "is competitive with protection only where degradation is gradual, while under "
        "common-cause disruption its gain is almost entirely a bypass of the detection delay. We believe this controlled "
        "decomposition, with a confirmation protocol committed before any confirmation run, is a more useful and more reliable contribution for RESS "
        "readers, and the response letter documents every change point by point.",
        "The revised manuscript has not been submitted elsewhere. The updated code, the frozen "
        "protocol and the result files are provided as supplementary material and will be "
        "archived as a new version of the Zenodo record.",
        "Yours sincerely,",
        "Nikita Tarasov (corresponding author) and Roman Zinko",
        "Lviv Polytechnic National University",
    ]
    doc = Document()
    for t in text:
        doc.add_paragraph(t)
    doc.save(os.path.join(out_dir, "Cover_Letter_R1.docx"))
    open(os.path.join(out_dir, "Cover_Letter_R1.md"), "w").write("\n\n".join(text))

    hl = [
        "Simulation-based digital twin compares 15 recovery policies on common random numbers",
        f"Plain 1:1 protection cuts resilience loss by {min(prot):.1f}-{max(prot):.1f}% vs reactive restoration",
        f"Risk-informed standby selection adds a confirmed {-cf(S4, 'protection_predictive', 'protection'):.1f}% and {-cf(S6, 'protection_predictive', 'protection'):.1f}% on new seeds",
        "For standby selection a degradation index is equivalent to the learned predictor",
        "Migration gain under common-cause shocks is a bypass of the detection delay",
    ]
    for h in hl:
        assert len(h) <= 85, (len(h), h)
    doc = Document()
    doc.add_heading("Highlights", level=1)
    for h in hl:
        doc.add_paragraph(h, style="List Bullet")
    doc.save(os.path.join(out_dir, "Highlights_R1.docx"))
    open(os.path.join(out_dir, "Highlights_R1.txt"), "w").write("\n".join(hl) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join("..", "submission_R1"))
    ap.add_argument("--captions", required=True, help="<manuscript>.docx.captions.json")
    a = ap.parse_args()
    global CAPTIONS
    CAPTIONS = a.captions
    os.makedirs(a.out, exist_ok=True)
    write_response(a.out)
    write_cover(a.out)
    print("written to", a.out)


if __name__ == "__main__":
    main()
