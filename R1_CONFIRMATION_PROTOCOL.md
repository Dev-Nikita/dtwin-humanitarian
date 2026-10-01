# Revision-confirmatory experiment — protocol (frozen before execution)

Manuscript JRESS-D-26-06464, revision 1. This file is committed together with
the code it refers to, before any confirmation seed is run, and is not edited
after the confirmation results exist. The commit that adds this file is the
code freeze; its hash is recorded in `data/results/confirmation_manifest.json`
by the analysis script.

## Why this experiment exists

The predictive-protection policies (P6, P6i) and the 1:1 protection baseline
(P1b) were introduced during the revision, after the reviewers' comments. All
comparisons involving them on the development seeds (seeds 1000 + 37k and the
sweep seeds) are therefore exploratory. This experiment tests the revision's
central comparisons on seeds that have never been used for development or
analysis.

## Frozen code and model

- Code: the commit containing this file. No change to policies, parameters,
  predictor, thresholds or generative model is allowed after it.
- Predictor: logistic regression with isotonic calibration, as selected by
  validation PR-AUC (`data/results/best_model.json`); operating point theta* as
  stored there. Matched index scale and KPI scale as stored in
  `degradation_index_scale.json` and `kpi_alarm_scale.json`.
- Protection model: `feasible` (pre-signalled standby path, unreserved
  capacity, switch-over only if the standby path has enough residual capacity at
  the moment of the switch; flows processed in decreasing criticality then flow
  identifier; otherwise reactive restoration).
- P6 refresh discipline: cooldown 6 slots per flow, replacement only if it is
  observed up, capacity-feasible and lowers the standby bottleneck risk by at
  least 0.05, or if the current standby path is down.
- Interruption accounting: an action in slot t with duration C interrupts the
  flow in slots t, ..., t + C − 1 (exactly C slots). C_r ~ U{4, ..., 12},
  C_m ~ U{0, 1}. A migration away from a path with a physically failed element
  is charged C_r. All durations are keyed draws: C_r and the protection C_m by
  (seed, flow, failed element, failure onset); a live migration C_m by
  (seed, flow, slot).
- Detection delay D = 3, horizon H = 10, kappa = 1, nominal budget factors.

## Design

- Seeds: 64 new seeds, 20000 + 101 k, k = 0, ..., 63.
- Scenarios: S3, S4, S5, S6 (S7 remains an exploratory stress test).
- Policies: P1 reactive, P1b protection, P3 degradation-aware migration,
  P4 predictive migration, P6i protection + index, P6 predictive protection;
  P6o (protection + calibrated perfect state) is run as an exploratory
  reference only.
- Episode length 500 slots. 64 × 4 × 7 = 1,792 runs.

## Endpoint

Cumulative resilience loss L (slot-equivalents) per run. All contrasts are
paired by seed.

## Family B1 — primary (2 tests)

Hypothesis: risk-informed standby selection adds resilience on top of 1:1
protection.

- B1.1: L(P6) − L(P1b) in S4.
- B1.2: L(P6) − L(P1b) in S6.

Two-sided paired t-test, Holm correction over the two tests, alpha = 0.05.
The hypothesis is supported in a scenario if the Holm-adjusted p < 0.05 and the
mean difference is negative. Reported with 95% CI and Wilcoxon signed-rank
p-value as a check.

## Family B2 — key secondary (16 tests, Holm over all 16)

For each of S3, S4, S5, S6:

- P1b − P1 (value of protection)
- P4 − P1 (value of predictive migration)
- P4 − P1b (migration versus protection)
- P6 − P4 (action mechanism: standby selection versus live migration)

Two-sided paired t-tests, Holm over the 16 tests, alpha = 0.05.

## Family B3 — equivalence of risk sources inside protection

P6 versus P6i in S4 and S6, two one-sided paired t-tests (TOST), alpha = 0.05
each. Equivalence margins, fixed here as 3% of the development-campaign P1b
loss of the first revision (S4: 13.26, S6: 26.86):

- S4: ±0.40 slot-equivalents
- S6: ±0.81 slot-equivalents

Equivalence is declared if both one-sided tests reject. If not, the result is
reported as "no practically large difference observed; equivalence not
established".

## Exploratory (no inference claims)

- P6o − P6 and P6o − P6i: paired 95% CIs (headroom of perfect-state
  information under the same controller), all four scenarios.
- All other contrasts.

## Analysis

`python -m python.experiments.confirmation_analysis` (committed with this
protocol) computes every quantity above from
`data/results/confirmation.csv` and writes
`data/results/confirmation_tests.csv` and `confirmation_manifest.json`.
