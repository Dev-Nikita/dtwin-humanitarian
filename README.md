# Predictive Resilience of Hybrid 5G/6G and LEO Satellite Networks

Reproducible research artefact for *"Predictive Resilience of Hybrid 5G/6G and LEO Satellite
Networks: A Digital-Twin Framework for Failure Anticipation and Risk-Aware Recovery"*.

Everything needed to regenerate every number, figure and table in the manuscript: the degradation
environment, the failure-anticipation pipeline, the observation-only digital twin, the risk-aware
recovery controller, the closed-loop assessment harness, a Go control-plane prototype and
Kubernetes manifests.

---

## 1. What this is

A **closed-loop** experimental environment: recovery decisions change the offered load, the load
changes the hazard rate, so a policy shapes the failure process it will later face.

```
hidden degradation  ->  noisy telemetry  ->  observation-only twin  ->  calibrated failure risk
                                                        |
                                             risk-aware recovery
                                                        |
                                   routed flows -> offered load -> hazard rate
                                                        |
                                            (back into the environment)
```

Six recovery policies on identical paired seeds. Consecutive policies differ in exactly one
mechanism, which is what lets each mechanism be attributed separately:

| policy | id | description |
|---|---|---|
| `static` | P0 | path fixed at admission, never revised |
| `reactive` | P1 | path recomputed after the twin *observes* a failure (industry baseline) |
| `proactive_threshold` | P2 | same actuator, triggered by a static KPI alarm — isolates the value of acting early from the value of the evidence |
| `degradation_aware` | P3 | risk from a contemporaneous degradation index at a matched alarm rate, no prediction — isolates exposure control from anticipation |
| `proactive_ai` | P4 | **proposed**: calibrated risk over the prediction horizon, criticality-dependent budgets, make-before-break migration |
| `oracle` | P5 | same controller, risk from the true latent health (information ablation) |

Seven disruption scenarios. **S5 (LEO handover dynamics), S6 (correlated common-cause shocks) and
S7 (externally anchored non-terrestrial dynamics) are excluded from the training corpus** and act
as out-of-distribution tests; S7 takes its reconfiguration cadence and link-latency statistics from
published measurement studies.

## 2. Headline results

1008 closed-loop runs = 7 scenarios × 6 policies × 24 paired seeds, 500 slots each.

**Anticipation** (episode-disjoint test fold, prevalence 1.20 %):

| model | F1 | ROC-AUC | PR-AUC | lift | false-alarm rate |
|---|---|---|---|---|---|
| operator threshold rule | 0.313 | 0.934 | 0.162 | 13.5 | 0.57 % |
| logistic regression | **0.378** | **0.974** | **0.298** | **24.8** | 0.85 % |
| random forest | 0.330 | 0.973 | 0.272 | 22.7 | 0.92 % |
| XGBoost | 0.333 | 0.971 | 0.264 | 22.0 | 0.68 % |
| LightGBM | 0.365 | 0.966 | 0.268 | 22.4 | 1.05 % |

**Closed-loop resilience** (proposed vs. reactive restoration):

| scenario | availability | resilience loss | mean recovery |
|---|---|---|---|
| S3 NTN degradation | 97.68 → 97.97 % | 10.91 → 9.58 | 17.05 → 7.67 slots (−55 %) |
| S4 infrastructure collapse | 96.49 → 97.66 % | 16.97 → 11.44 (−33 %) | 14.25 → 3.70 slots (−74 %) |
| S6 common cause (unseen) | 95.38 → 96.30 % | 24.14 → 19.75 (−18 %) | 15.79 → 7.75 slots (−51 %) |

10 comparisons survive Holm correction over the full family of tests (|d_z| = 1.3 … 7.4).

**Three findings that matter more than the headline numbers**

1. *Acting early is not enough.* `proactive_threshold` uses the same actuator and acts early, yet
   drops S4 availability to 91.6 % (below `static`) and more than doubles resilience loss, because
   a static KPI alarm cannot separate transient degradation from impending failure.
2. *Better information is not enough.* The perfect-state controller has the shortest recovery times
   but **worse** availability than the calibrated one — it over-actuates (433 vs 114 migrations).
3. *Anticipation has a price boundary.* The benefit vanishes once a pre-emptive migration costs
   more than ≈ 4 slots — an explicit deployment criterion, consistent with the break-even
   probability `p* = C_m / C_f`.

## 3. Repository layout

```
python/
  simulator/        config, topology, traffic, jump-diffusion degradation, handover and
                    common-cause mechanisms, dataset generation
  predictor/        feature contract, training/comparison, streaming feature builder
  orchestrator/     digital twin core, six recovery policies
  experiments/      closed-loop runner, resilience metrics (survival, hazard, resilience loss,
                    lead time), campaign driver, robustness suite, figure/table generation
  api/              FastAPI predictor-service + Dockerfile
  paper/            manuscript builders (numbers injected from data/results)
go/                 digital-twin, orchestrator and telemetry services (+ Dockerfiles)
k8s/                Deployments, Services, DaemonSet, HPA, Ingress, ServiceMonitor
data/               processed datasets and all result CSVs
paper-assets/       figures/ (main F01-F14 + supplementary S01-S05) and tables/ (CSV + LaTeX)
tests/              regression tests (label causality, no leakage, twin isolation, ...)
```

## 4. Quick start

```bash
pip install -r requirements.txt

make dataset       # training corpus from S1-S4 only        (~40 s)
make heldout       # S5/S6 corpus, never used for training  (~20 s)
make train         # model comparison, calibration, theta*  (~2 min)
make experiments   # main campaign                          (~30 min)
make robustness    # ablation, OOD, noise, horizon, migration cost
make figures       # 14 main + 5 supplementary figures, 14 tables
make paper         # rebuild the manuscript from data/results
make test
```

`make all` runs the whole chain. The campaign can be chunked:

```bash
python -m python.experiments.run_experiments --seeds 8 --steps 500 \
    --scenarios S4_disaster_collapse --seed-offset 8      # appends by default
python -m python.experiments.run_experiments --overwrite ...   # start a fresh run table
python -m python.experiments.run_experiments --finalise        # recompute aggregates only
```

## 5. Control plane (Go + FastAPI)

```bash
docker compose up --build      # twin :8080, predictor :8081, orchestrator :8082,
                               # telemetry :8083, Prometheus :9090
kubectl apply -f k8s/          # namespace dtwin
```

| service | endpoint | purpose |
|---|---|---|
| digital-twin | `POST /api/v1/nodes/{id}/telemetry` | ingest a KPI sample |
| digital-twin | `POST /api/v1/risk` | bulk risk update |
| digital-twin | `GET /api/v1/topology` | elements + links + risk + load |
| predictor | `POST /api/v1/predict/batch` | calibrated P(failure ≤ H) |
| orchestrator | `POST /api/v1/orchestrate` | place a service flow |
| orchestrator | `POST /api/v1/reroute` | force a control cycle |
| all | `GET /metrics` | Prometheus exposition |

## 6. Reproducibility notes

* Seeds are paired across policies, so environment randomness cancels in every comparison.
* The corpus is split **by episode**, never by row — overlapping feature windows cannot leak.
* Calibration and the operating point are fitted on validation and frozen before the test fold.
* The latent health variable is visible only to the environment and the labelling function;
  `tests/test_framework.py::test_twin_never_sees_latent_state` enforces this.
* S5/S6 are absent from the training corpus by construction (`config.TRAIN_SCENARIOS`).
* The manuscript is generated from `data/results`, so text and results cannot diverge.

## 7. Licence

Released for academic use. Please cite the accompanying paper.
