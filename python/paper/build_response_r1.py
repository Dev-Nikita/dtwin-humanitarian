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

    d46 = pc(S4, "proactive_ai", "protection")
    d6b = pc(S4, "protection_predictive", "protection")
    d6b6 = pc(S6, "protection_predictive", "protection")
    d43 = pc(S4, "proactive_ai", "degradation_aware")

    summary = [
        "We thank the editor and the four reviewers for an unusually careful set of reports. "
        "Addressing them properly changed the paper substantially, and we want to state at the "
        "outset what changed and why, because it affects every response below.",
        "Three of the points raised — the pairing of the simulation design (R3.3), the lead-time "
        "inconsistency of Table 16 (R3.2, R5.1) and the missing protection baseline (R1.4) — "
        "led us to audit the implementation end to end. The audit found three defects, all now "
        "corrected: (i) traffic, environment and controller shared one random-number generator, "
        "so policies that consumed different numbers of draws saw different arrivals and failures "
        "and the paired design was not actually paired; (ii) the horizon-sweep table had been "
        "produced by an older lead-time tracker without the horizon bound; and (iii) a migration "
        "away from a path whose element had already failed — which the telemetry-driven policies "
        "frequently perform during the three-slot detection delay — was charged the "
        "make-before-break switch-over cost C_m instead of the restoration cost C_r, although no "
        "working path existed during the switch. The third defect was the most consequential.",
        f"After the corrections, and with the additional baselines the reviewers asked for, the "
        f"central claim of the first submission does not hold. Predictive migration (P4) reduces "
        f"resilience loss relative to reactive restoration by {-rel(S4, 'proactive_ai'):.1f}% under "
        f"infrastructure collapse (previously reported: 33%) and {-rel(S6, 'proactive_ai'):.1f}% under "
        f"common-cause disruption; it is neutral in S3 and harmful in S5 and S7. Plain 1:1 protection "
        f"does better ({-rel(S4, 'protection'):.1f}% and {-rel(S6, 'protection'):.1f}%). Most of the "
        "migration benefit comes from acting during the detection delay on elements that have "
        "already failed rather than from foresight. The largest gain is obtained by a policy we "
        "added in response to R1.4 and R5.2 — predictive protection (P6), which uses the risk "
        "estimate only for decisions that interrupt no traffic (primary-path choice and "
        f"standby-path refresh): {-rel(S4, 'protection_predictive'):.1f}% and "
        f"{-rel(S6, 'protection_predictive'):.1f}%. Within that policy a learned predictor, a "
        "contemporaneous index and perfect state knowledge are within about two per cent of one "
        "another.",
        "We therefore rewrote the paper around what the evidence supports. The title, abstract, "
        "research questions, contributions, results, discussion and conclusions are new; the "
        "system model, the framework and the related work are retained with the clarifications "
        "requested. The new title is 'Where Does Predictive Resilience Come From? A "
        "Simulation-Based Digital-Twin Assessment of Failure Anticipation, Protection and "
        "Risk-Aware Recovery in Hybrid 5G/6G-Class and LEO Satellite Networks'. We recognise "
        "that the revised paper makes a more modest and different claim than the original. We "
        "believe it is the more useful contribution for the readership of RESS — a controlled "
        "decomposition showing where the resilience benefit of risk information comes from, with "
        "a methodology that exposed the artefacts that would otherwise have inflated it — and we "
        "hope the editor and reviewers agree.",
        f"Summary of new work: the main campaign was rerun (15 policies × 7 scenarios × 24 paired "
        f"seeds, {len(r):,} runs) and nine further closed-loop experiments were added (detection-delay "
        "sweep, actuation-effort frontier, migration-cost sweep, misspecified degradation world, "
        "98-element topology, fixed-traffic feedback ablation, controller design sensitivity, "
        "parameter and weight sensitivity, horizon sweep). Reviewer comments are answered point "
        "by point below; section, table and figure numbers refer to the revised manuscript.",
    ]

    R = {}
    R["R4.1"] = ("The use of the term digital twin should be clarified; add 'simulation-based'.",
        "Agreed. The framework is called a simulation-based digital twin in the title, the "
        "abstract and at first introduction (Section 4.1), where we also state that the physical "
        "counterpart is itself a simulation, that the twin has not been synchronised with a "
        "deployed network and that no result should be read as validation on a real system. "
        "Section 8.4 explains what distinguishes the construction from a simulator (see also R1.1).")
    R["R4.2"] = ("Relationship between slot and real time.",
        "Added in Section 3.4, Table 4 and Table 5: one slot corresponds to one second, the mapping "
        "that anchors the 15-slot reconfiguration grid of S5 and S7 to the 15 s Starlink schedule "
        "[52]. Hence D = 3 s, C_r = 4–12 s and C_m = 0–1 s, and H = 10 s. We also state how "
        "interruptions are counted: an action taken in slot t with cost C interrupts the flow in "
        "slots t…t+C, so a switch-over costs one or two whole slots and no fractional "
        "interruption arises (see R3.8). The original text described C_m as U(0,1); the "
        "implementation draws an integer from {0, 1}, and the text now says so.")
    R["R4.3"] = ("Criterion for selecting the prediction model.",
        "Section 4.3 now states the criterion, fixed in advance and applied on validation data "
        "only: highest validation PR-AUC (the positive class is rare); models within 0.01 are "
        "tied and the tie is broken by validation Brier score (the controller consumes "
        f"probabilities) and then by inference cost. Table 7 now shows the validation columns. "
        f"Logistic regression has the highest validation PR-AUC "
        f"({mc.loc['logistic_regression', 'val_pr_auc']:.3f}); no other model is within the tolerance. "
        "The code was changed accordingly — the first version selected by test PR-AUC, which we "
        "should not have done.")
    R["R4.4"] = ("Distinguish the exploratory nature of S7.",
        "Done. S7 is described as an externally parameterised stress test, not an external "
        "validation, in Sections 6.2 and 7.7; it is outside the primary inferential family and all "
        "its comparisons are labelled exploratory. The revised abstract and conclusions do not use "
        "S7 as validation.")

    R["R1.1"] = ("Essential difference between the framework and a conventional network simulator.",
        "Sections 4.1 and 8.4 now answer this directly. The distinction is structural: the "
        "simulated physical system evolves on its own; the twin is a separate estimator that sees "
        "it only through telemetry, holds a delayed and noisy belief, answers predictive questions "
        "and acts back; an assessment layer with privileged access scores the outcome. This "
        "separation is what made the main findings measurable — the detection-delay bypass exists "
        "only because the twin's belief lags the physical state, and the accounting defect was "
        "found only because the assessment layer knows what the twin does not. We use "
        "'simulation-based digital twin' throughout and do not claim synchronisation with a real "
        "network.")
    R["R1.2"] = ("19 nodes are insufficient for satellite networks.",
        "We added a 98-element topology with three LEO orbital planes of twelve satellites, "
        "intra- and inter-plane links, sixteen ground gateways and a doubled terrestrial segment "
        "(Sections 6.2 and 7.7, Table 16). With the predictor and all operating points frozen from "
        f"the reference network, the ordering under collapse is P1 {lgL(S4, 'reactive'):.2f}, P1b "
        f"{lgL(S4, 'protection'):.2f}, P4 {lgL(S4, 'proactive_ai'):.2f}, P6 "
        f"{lgL(S4, 'protection_predictive'):.2f}; the conclusions are unchanged. We state as a "
        "limitation that the model has no orbital mechanics and is not representative of an "
        "operational constellation.")
    R["R1.3"] = ("The predictor mainly learns the degradation patterns predefined by the authors.",
        "We agree, and the revision treats this as a central limitation rather than a detail. "
        "We built a second, structurally different world — monotone gamma-process wear-out "
        "without mean reversion, observed through a four-slot lagged map with different exponents "
        "and doubled noise — and applied the frozen predictor and controllers to it (Section 7.7, "
        f"Table 15). The predictor's PR-AUC falls to {f_fr['pr_auc']:.3f} (lift {f_fr['lift']:.1f}) "
        f"against {f_rt['pr_auc']:.3f} for the same estimator retrained there; the closed-loop "
        f"ordering is preserved (collapse: P1 {msL(S4, 'reactive'):.2f}, P1b {msL(S4, 'protection'):.2f}, "
        f"P4 {msL(S4, 'proactive_ai'):.2f}, P6 {msL(S4, 'protection_predictive'):.2f}). The paper now "
        "separates the two claims explicitly: the framework and the decomposition transfer across "
        "the two families we built; the predictor is validated only within the family it was "
        "trained on, and no claim is made for real failure mechanisms (Limitation 2).")
    R["R1.4"] = ("Additional baselines, e.g. protection mechanisms.",
        "Added: P1b, 1:1 protection with a pre-computed node-disjoint standby path and "
        "detection-driven switch-over at cost C_m [57], [58], [79]. This baseline changed the "
        f"conclusions: it beats predictive migration (collapse: {-rel(S4, 'protection'):.1f}% vs "
        f"{-rel(S4, 'proactive_ai'):.1f}% loss reduction relative to P1; paired P4 − P1b = "
        f"{d46[0]:+.2f} [{d46[1]:.2f}, {d46[2]:.2f}]). It also led to the new policy P6, which adds "
        "risk-driven primary-path choice and standby refresh to protection. It is the best or "
        "tied-best non-oracle policy in every stressed scenario and significantly better than "
        f"P1b under collapse and common-cause disruption ({-rel(S4, 'protection_predictive'):.1f}% "
        "vs P1 under collapse); in S5 and S7 it is within noise of P1b.")
    R["R1.5"] = ("How much of the improvement is truly contributed by failure anticipation?",
        "This is now the question the paper is built around (RQ2, Sections 7.2–7.6). The answer "
        "is: little, in this system. (a) Migration triggers classified by ground truth show that "
        f"under collapse {PR.loc[(S4, 'proactive_ai')]:.0f} P4 migrations per run include "
        f"{r[(r.scenario == S4) & (r.policy == 'proactive_ai')].mig_trigger_already_failed.mean():.0f} "
        "off already-failed elements and "
        f"{r[(r.scenario == S4) & (r.policy == 'proactive_ai')].mig_trigger_no_failure_within_H.mean():.0f} "
        "off elements that do not fail within H; under common-cause disruption almost all are "
        "off already-failed elements. (b) Sweeping the detection delay shows the P4 advantage "
        f"over P1 rising from {ddL(S4, 'reactive', 0) - ddL(S4, 'proactive_ai', 0):.2f} at D = 0 "
        f"to {ddL(S4, 'reactive', 5) - ddL(S4, 'proactive_ai', 5):.2f} at D = 5 (collapse): it is a "
        "detection-delay effect. (c) At matched alarm rate, prediction is not better than a "
        f"contemporaneous index (collapse: {d43[0]:+.2f} [{d43[1]:.2f}, {d43[2]:.2f}]). (d) Within "
        "predictive protection, index and perfect state are within 1–2% of the predictor.")
    R["R1.6"] = ("Budget factors and weights 4, 3, 0.6: why these values? Sensitivity analysis.",
        "Section 4.4 explains the normalisation argument used to fix them before the campaign, and "
        "Section 7.10 (Table 19) now reports a design sensitivity on 24 paired seeds: four "
        "budget-factor profiles (flat, shallow, nominal, steep) and one-at-a-time changes of the "
        "loss coefficient (2, 8), risk base (1.5, 6) and criticality slope (0, 1.2). P4 ranges "
        f"{ds['P4 loss'].min():.2f}–{ds['P4 loss'].max():.2f} and P6 "
        f"{ds['P6 loss'].min():.2f}–{ds['P6 loss'].max():.2f}; the nominal profile is the best of "
        "the four for P4, so it does not flatter the comparison with protection; P6 beats "
        "protection in every setting. Table 22 adds the Eq. (7) weights.")
    R["R1.7"] = ("In S7 the advantage disappears except recovery time; explain.",
        f"With the corrections, P4 is worse than P1 in S7 ({rel(S7, 'proactive_ai'):+.1f}%), and "
        "its apparent recovery-time advantage was a composition effect of short migration-induced "
        "interruptions (see R5.4). The explanation (Section 7.7) follows from the decomposition: "
        "the migration benefit elsewhere is carried by the detection-delay bypass, which is worth "
        "little when the impairment is a short, scheduled reconfiguration without precursor, while "
        "migrations triggered by reconfiguration transients cost interruptions. Protection and P6 "
        f"still improve on P1 in S7 ({rel(S7, 'protection'):+.1f}% and "
        f"{rel(S7, 'protection_predictive'):+.1f}%). We agree the result limits the applicability of "
        "anticipatory migration and the revised paper says so.")

    R["R5.1"] = ("Inconsistent lead times in Table 16 (9.67, 10.67 for H = 3, 5).",
        "You are right; this was an error. The horizon sweep had been produced by an older "
        "lead-time tracker that matched each failure to the first threshold crossing without the "
        "horizon bound of the lead-time definition, and the table was not regenerated when the "
        "tracker was replaced. The sweep now passes H to the tracker, the bound is asserted in the "
        f"code and enforced by a unit test. Corrected median lead times range "
        f"{hs.cl_lead_time_median.min():.1f}–{hs.cl_lead_time_median.max():.1f} slots, never above H "
        "(Table 17, Fig. 14). The conclusions of Section 7.6 of the first version that relied on "
        "this column were re-derived (see R3.2).")
    R["R5.2"] = ("Isolate predictive information from service differentiation; direct P3–P4 comparisons.",
        "The revised design is a controlled factorial (Section 6.3): P3 and P4 share budgets, the "
        "criticality term and criticality-ordered processing; uniform variants P3u and P4u remove "
        "all three. Direct paired P3–P4 contrasts are in Table 11 for every scenario; the 2 × 2 "
        "analysis is in Section 7.6. Differentiation helps the index policy most where transient "
        "signatures are frequent (S5, S7; smaller effects in S2 and S6) and has no detectable "
        "effect in S3 and S4. With differentiation, prediction is ahead of the index in S5 and "
        "marginally in S2, and indistinguishable from it in S3, S4, S6 and S7. We also compare all risk sources at matched actuation effort (Fig. 12, "
        "Table 14).")
    R["R5.3"] = ("Confounded feedback ablation; Table 20 retains a benefit at g = 0.",
        "You are right, and the claim that two thirds of the gain came from feedback is withdrawn. "
        "The ablation is repeated with traffic held fixed: P1 is run with its endogenous load, its "
        "utilisation trajectory is recorded, and P3, P4, P1b and P6 are evaluated with the "
        "degradation process driven by that replayed trajectory on the same seeds (Section 7.9, "
        f"Table 18). At nominal coupling the two arms differ by at most {fbmax:.2f} slot-equivalents "
        "for any policy: the feedback contributes nothing in the calibrated model, consistent with "
        "Table 20 of the first version. A direct check confirms that element-failure counts agree "
        "across policies to within 0.01% at nominal coupling. The closed loop is kept as the correct "
        "evaluation structure, not as a source of benefit.")
    R["R5.4"] = ("97.66% availability vs 41% SLA compliance; per-class SLA-compliant availability; "
                 "separate failure- and migration-induced recovery.",
        "Both added. Table 13 reports availability and SLA-compliant availability per service "
        "class under collapse: connectivity is largely maintained but the surviving "
        "non-terrestrial paths cannot meet the latency/loss budgets of the tighter classes, so SLA "
        "compliance is low for every policy — recovery policies restore connectivity, they cannot "
        "restore removed capacity (Section 7.3). Table 12 separates recovery time by cause. Your "
        f"concern was justified: P4's all-interruption mean recovery ({REC.loc[S4, 'proactive_ai']:.2f} vs "
        f"{REC.loc[S4, 'reactive']:.2f} slots for P1 under collapse) is a composition effect; its "
        f"failure-induced recovery is {REF.loc[S4, 'proactive_ai']:.2f} vs {REF.loc[S4, 'reactive']:.2f}. "
        f"Protection and P6 shorten failure-induced recovery to {REF.loc[S4, 'protection']:.2f} and "
        f"{REF.loc[S4, 'protection_predictive']:.2f}. Mean recovery over all interruptions is no longer "
        "used as a headline.")

    R["R3.1"] = ("Predictive performance derived from the same synthetic latent-health model; "
                 "distinguish framework and predictor validation; independently generated degradation.",
        "See R1.3. We implemented an independently structured degradation world (gamma wear-out, "
        "lagged misspecified observation map) and report both offline transfer and the closed-loop "
        "consequence (Section 7.7, Table 15). The manuscript now distinguishes explicitly between "
        "validation of the framework/decomposition (which transfers) and of the predictor (valid "
        "only within its training family), in Section 7.7, Limitation 2 and the conclusion. A "
        "trace-driven evaluation remains future work; we did not have access to operational "
        "failure traces.")
    R["R3.2"] = ("Table 16 inconsistency with Eq. (18).",
        "See R5.1 — corrected, with the cause explained. The re-derived sweep shows that the loss "
        f"of P4 rises with H ({hs.cl_resilience_loss.iloc[0]:.2f} at H = 3 to "
        f"{hs.cl_resilience_loss.max():.2f}, against {hz_ref:.2f} for P1), because longer horizons "
        "trigger more false-alarm migrations; P6 is insensitive to H. The recommendation to keep "
        "the horizon short is therefore retained for migration-based control, with the corrected "
        "reason (Section 7.8).")
    R["R3.3"] = ("Different flow counts between policies; common random numbers.",
        "Thank you — this identified a genuine defect. Traffic, environment and controller shared "
        "one RNG, so policies consuming different numbers of draws saw different arrivals and "
        "failures. The revision implements common random numbers (Section 6.4): separate "
        "SeedSequence streams for traffic, controller, common-cause process and every element; "
        "each element draws a fixed-size random block per slot and every random quantity "
        "(diffusion, jump occurrence and size, repair time, post-repair health, handover outcome, "
        "observation noise) is read from a fixed position. The number of generated flows is now "
        f"identical across policies for every (scenario, seed) ({'verified for all' if flows_equal else 'NOT verified'} "
        "168 scenario-seed pairs), and a unit test asserts that the random blocks coincide across "
        "policies. All results were recomputed.")
    R["R3.4"] = ("Specify P2 fully; compare P2 and P4 at matched alarm or migration rate.",
        "P2 is now fully specified (Section 6.3): the KPI severity rule, the nominal operator "
        "limits, the same hysteresis and k = 3 candidates as P4, and exclusion of alarmed "
        "elements from routing. We added P2m with the alarm scale matched on validation runs to "
        f"the predictor's per-element alarm rate (scale {kpi['kpi_scale_at_operating_point']:.3f}), and "
        "a frontier sweeping the operating point of every risk source (Fig. 12, Table 14). Your "
        "suspicion was correct: at matched effort P2 is comparable to P4 (collapse, best points "
        f"{frb(S4, 'P2'):.2f} vs {frb(S4, 'P4'):.2f}). The claim that proactive action on "
        "conventional thresholds is worse than reactive restoration has been removed; it was a "
        "property of one aggressive configuration.")
    R["R3.5"] = ("On which data was the P3 alarm-rate matching performed?",
        "It was not documented, and we re-did it under an explicit protocol (Section 6.3): the "
        "index quantile and the KPI scale are estimated on dedicated validation runs of P1 in "
        "S2–S4 on six seeds (9000 + 13k, k = 0…5) disjoint from every evaluation seed, at the predictor's "
        f"per-element alarm rate ({deg['alarm_rate']:.4f}), and frozen before any evaluation run. "
        "The source of each scale is recorded in the result files.")
    R["R3.6"] = ("P5 mapping undefined; perfect state with a mis-calibrated map is not an oracle.",
        "Agreed. The original map is now defined (linear in health between 0.35 and 0.18) and "
        "retained as P5 to show its consequence. We added P5c, the calibrated perfect-state "
        "reference: the exact probability that an element at health h fails within H, computed by "
        "Monte Carlo simulation of the generative model per scenario and element class. P5c "
        "does not over-actuate the way P5 did "
        f"({PR.loc[(S4, 'oracle_calibrated')]:.0f} vs {PR.loc[(S4, 'oracle')]:.0f} migrations per run "
        f"under collapse) and is better than P4 (loss {RL.loc[S4, 'oracle_calibrated']:.2f} vs "
        f"{RL.loc[S4, 'proactive_ai']:.2f}), but stays behind plain protection "
        f"({RL.loc[S4, 'protection']:.2f}). The statement 'perfect state knowledge over-actuates' has "
        "been removed; it applied only to the uncalibrated map.")
    R["R3.7"] = ("Eq. (12) presented too strongly.",
        "Section 4.5 is rewritten as 'An ex-post economic reading of the budget': the rule is "
        "stated to be a heuristic, the thresholds are not derived from Eq. (12), F1 contains no "
        "costs and R_inf is a proxy. Cost-derived actuation is named as future work.")
    R["R3.8"] = ("Physical meaning of a slot; fractional C_m.",
        "See R4.2. One slot = 1 s; C_m is an integer in {0, 1} (the first text said U(0,1), which "
        "was wrong); interruptions are counted in whole slots on the same grid for availability, "
        "recovery and Kaplan–Meier, so no fractional interruption arises (Section 3.4).")
    R["R3.9"] = ("'All policies indistinguishable' under S1 is inconsistent with P2.",
        "Corrected; the sentence was removed and the whole Results and Conclusion text was "
        "rewritten. Every number in the text is now generated from the result files by the same "
        "script that produces the tables, and the qualitative statements were checked against "
        "the result files in an independent pass before resubmission.")
    R["R3.10"] = ("State the incremental value of prediction more narrowly.",
        "Fully agreed, and the corrected results go further than the reviewer suggested: the "
        "paper is now framed as a decomposition in which prediction adds little over a "
        "contemporaneous index, and risk information pays mainly through protection. The abstract "
        "and conclusion state this decomposition directly.")
    R["R3.11"] = ("Criticality-weighted result is exploratory.",
        "The class-weighted aggregates are no longer used for any conclusion (Section 5.2); "
        "per-class availability and SLA-compliant availability are reported instead (Table 13). "
        "The corrected design did not reproduce the weighted-loss advantage, and the claim has been "
        "removed from the abstract and conclusions.")
    R["R3.12"] = ("S7 is externally parameterised stress testing, not validation.",
        "Agreed; this wording is used throughout (Sections 6.2, 7.7). S7 is also given more "
        "prominence as the scenario where the limits of anticipatory migration are clearest.")

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


def write_cover(out_dir):
    r, RL, rel, pc, _ = load()
    text = [
        "30 September 2026",
        "Editor, Reliability Engineering & System Safety",
        "Re: Revised manuscript JRESS-D-26-06464",
        "Dear Editor,",
        "We submit the revised version of our manuscript, now titled 'Where Does Predictive "
        "Resilience Come From? A Simulation-Based Digital-Twin Assessment of Failure "
        "Anticipation, Protection and Risk-Aware Recovery in Hybrid 5G/6G-Class and LEO Satellite "
        "Networks'.",
        "The reviewers' comments led us to an end-to-end audit of the simulation. It found that "
        "the paired design was not correctly paired, that one table used an outdated lead-time "
        "computation, and that migrations away from already-failed paths were scored with the "
        "switch-over instead of the restoration cost. With these corrected and with the "
        "additional baselines the reviewers requested — above all 1:1 protection — the headline "
        "of the first submission (a 33% reduction in resilience loss from failure anticipation) "
        f"does not hold: predictive migration yields {-rel(S4, 'proactive_ai'):.1f}% under "
        f"infrastructure collapse, plain protection {-rel(S4, 'protection'):.1f}%, and risk-aware "
        f"protection {-rel(S4, 'protection_predictive'):.1f}%.",
        "Rather than defend the original claim, we have rewritten the paper around what the "
        "corrected evidence supports: a controlled decomposition of where the resilience benefit "
        "of risk information comes from. Its main findings are that most of the benefit "
        "attributed to anticipation is a detection-delay effect, that protection dominates "
        "pre-emptive migration, and that risk information pays best when spent on decisions that "
        "interrupt no traffic. We believe this is a more useful and more reliable contribution "
        "for RESS readers, and the response letter documents every change point by point.",
        "The revised manuscript has not been submitted elsewhere. The updated code and result "
        "files are provided as supplementary material.",
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
        "Closed-loop digital twin with common random numbers compares 15 recovery policies",
        f"Predictive migration cuts resilience loss by only {-rel(S4, 'proactive_ai'):.0f}% under collapse",
        "Most of that gain comes from bypassing the detection delay, not foresight",
        f"1:1 protection with risk-refreshed standby paths cuts loss by {-rel(S4, 'protection_predictive'):.0f}%",
        "Perfect state knowledge adds under 2% once risk is used for standby selection",
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
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    write_response(a.out)
    write_cover(a.out)
    print("written to", a.out)


if __name__ == "__main__":
    main()
