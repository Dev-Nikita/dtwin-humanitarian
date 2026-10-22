# Where Does Predictive Resilience Come From?

Code and data for *"Where Does Predictive Resilience Come From? A Simulation-Based Digital-Twin
Assessment of Failure Anticipation, Protection and Risk-Aware Recovery in Hybrid 5G/6G-Class and
LEO Satellite Networks"* (N. Tarasov, R. Zinko; Reliability Engineering & System Safety,
manuscript JRESS-D-26-06464, revision R1).

Every number, figure and table of the revised manuscript is generated from `data/results/` by
the scripts in this repository.

---

## 1. What the study does

A simulation-based digital twin of a hybrid terrestrial–LEO network hides the degradation state
of every element behind noisy telemetry and a detection delay. Fifteen recovery policies are
compared in closed loop on common random numbers. They are a controlled set of mechanistic
contrasts in three factors: the **risk source** (none, static KPI alarm, contemporaneous
degradation index, learned predictor, perfect state), **criticality differentiation**, and the
**restoration mechanism** (reactive restoration or 1:1 protection).

| id | policy key | mechanism |
|---|---|---|
| P0 | `static` | path fixed at admission |
| P1 | `reactive` | recompute the path after the twin observes a failure |
| P1b | `protection` | 1:1 protection, disjoint standby path, capacity-checked switch-over |
| P2 / P2m | `proactive_threshold` / `_matched` | migration on a static KPI alarm (nominal / matched rate) |
| P3 / P3u | `degradation_aware` / `_uniform` | migration on a degradation index (differentiated / uniform) |
| P4 / P4u | `proactive_ai` / `_uniform` | migration on the calibrated failure probability |
| P5 / P5c | `oracle` / `oracle_calibrated` | migration on the true latent state (band map / Monte Carlo estimate) |
| P6i / P6 / P6o | `protection_index` / `_predictive` / `_oracle` | 1:1 protection with risk-aware primary path and standby refresh |
| P6m | `protection_predictive_mig` | P6 plus P4's migrations |
| P6-route / P6-standby | `protection_predictive_route` / `_standby` | post-confirmation ablation of P6 |

## 2. Main results (revision R1)

Frozen confirmation on 64 previously unused seeds (protocol: `R1_CONFIRMATION_PROTOCOL.md`,
freeze record: `CODE_FREEZE.md`):

* Plain 1:1 protection reduces cumulative resilience loss by 2.9–18.5 % relative to reactive
  restoration (S3–S6).
* Risk-informed protection (P6) adds a confirmed 3.6 % (infrastructure collapse) and 3.5 %
  (common-cause disruption) on top of protection (B1, Holm p < 0.0001).
* For this role a contemporaneous degradation index is statistically equivalent to the learned
  predictor (B3, TOST); perfect state adds at most 1.5 % (exploratory).
* Predictive migration (P4) is not significantly different from protection under collapse and
  satellite-segment degradation after multiplicity correction, and worse under handover
  dynamics and common-cause disruption.
* An exploratory ablation attributes the P6 effect almost entirely to standby-path selection.

## 3. Repository layout

```
python/
  simulator/     configuration, topologies (19- and 98-element), traffic, jump-diffusion and
                 gamma-wear degradation, handover and common-cause mechanisms, dataset generation
  predictor/     feature contract, model comparison and calibration, streaming features
  orchestrator/  digital twin core and the recovery policies
  experiments/   closed-loop runner, resilience metrics, development campaign
                 (campaign_chunks), revision sweeps, frozen confirmation analysis,
                 figures and tables (make_figures_r1)
  paper/         manuscript, response-letter and LaTeX builders, consistency checks (qa_r1)
  api/           FastAPI predictor service
go/              digital-twin, orchestrator and telemetry services
k8s/             deployment manifests
data/results/    every result table used by the paper
paper-assets/    generated figures and tables
tests/           unit tests (common random numbers, keyed durations, interruption accounting,
                 capacity-feasible protection, lead-time bound, twin isolation, ...)
R1_CONFIRMATION_PROTOCOL.md   frozen confirmation protocol (committed before the run)
CODE_FREEZE.md                freeze commit and what changed afterwards
```

## 4. Reproducing the results

```bash
pip install -r requirements.txt
make test                 # unit tests
make dataset heldout train
make campaign             # development campaign: 15 policies x 7 scenarios x 24 seeds
make sweeps               # revision experiments (detection delay, frontier, misspecification, ...)
make confirmation         # frozen confirmation: 64 seeds x 4 scenarios x 7 policies
make figures              # figures and tables of the revision
make paper                # manuscript (.docx), response letter, cover letter, highlights
make qa                   # consistency checks on the generated documents
```

The campaign and the sweeps run in independent chunks and can be distributed across machines:

```bash
python -m python.experiments.campaign_chunks run 0 4 4      # tasks 0-3 with 4 workers
python -m python.experiments.campaign_chunks merge
python -m python.experiments.revision_sweeps run frontier 0 12 4
python -m python.experiments.revision_sweeps merge frontier
```

The freeze commit is tagged `r1b-freeze`.

## 5. Control plane (Go + FastAPI)

```bash
docker compose up --build      # twin :8080, predictor :8081, orchestrator :8082,
                               # telemetry :8083, Prometheus :9090
kubectl apply -f k8s/
```

The control-plane services demonstrate that the twin and controller can run independently of
the scientific simulator; they are not needed to reproduce the paper.

## 6. Scope

The evaluation is entirely simulation-based. Both degradation families are modelling
assumptions, and the predictor is validated only within the family it was trained on.

## 7. Licence and citation

Released for academic use. Please cite the accompanying paper.
