"""
Operating-point matching of the non-predictive baselines (Reviewers 3.4, 3.5).

P3 (degradation-aware) and P2m (threshold-proactive at a matched rate) are
compared with the predictor at the same per-element alarm rate. The matching
uses VALIDATION information only:

    * closed-loop validation runs of the training scenarios S2-S4 under the
      passive reactive policy (which does not act on any risk score),
    * validation seeds 9000 + 13 k, disjoint from every seed used for testing,
    * the predictor at its frozen, validation-selected operating point theta*.

On every observed element-slot (element not failed) the calibrated predicted
risk p, the degradation index d and the KPI severity s are recorded. The
predictor's alarm rate a = P(p >= theta*) defines the target; the scales are the
(1 - a) quantiles of d and s. Both are written to data/results and frozen before
any test run is executed.
"""

from __future__ import annotations

import json
import os

import numpy as np

from ..simulator.failure_generator import STATUS_FAILED
from .run_experiments import DATA_DEFAULT, prepare_context
from .runner import ExperimentRun

RESULTS = os.path.join("data", "results")
VAL_SCENARIOS = ("S2_bs_overload", "S3_leo_degradation", "S4_disaster_collapse")


def main(seeds: int = 6, steps: int = 500) -> None:
    ctx = prepare_context(DATA_DEFAULT)
    theta = float(ctx["meta"]["threshold"])
    P, D, S = [], [], []
    for scen in VAL_SCENARIOS:
        for k in range(seeds):
            run = ExperimentRun(scen, "reactive", 9000 + 13 * k, steps=steps,
                                model=ctx["model"], feature_columns=ctx["feature_columns"],
                                latency_medians=ctx["latency_medians"], theta=theta)
            original = run.twin.set_risk

            def hook(risk, run=run, original=original):
                original(risk)
                pred = run._predicted_risk()
                for nid, node in run.env.nodes.items():
                    if node.status == STATUS_FAILED or nid not in pred:
                        continue
                    P.append(pred[nid])
                    D.append(run.degradation_index(nid))
                    S.append(run.orch.kpi_severity(nid))

            run.twin.set_risk = hook
            run.run()
        print(f"{scen} done", flush=True)
    P, D, S = map(np.asarray, (P, D, S))
    rate = float((P >= theta).mean())
    q = 1.0 - rate
    d_scale = float(np.quantile(D, q))
    s_scale = float(np.quantile(S, q))
    meta = {"alarm_rate": rate, "quantile": q, "theta": theta, "n_element_slots": int(P.size),
            "source": "closed-loop validation runs (reactive policy), scenarios "
                      + ",".join(VAL_SCENARIOS) + f", seeds 9000+13k, k<{seeds}"}
    with open(os.path.join(RESULTS, "degradation_index_scale.json"), "w") as fh:
        json.dump(dict(meta, index_at_operating_point=d_scale), fh, indent=2)
    with open(os.path.join(RESULTS, "kpi_alarm_scale.json"), "w") as fh:
        json.dump(dict(meta, kpi_scale_at_operating_point=s_scale,
                       nominal_alarm_rate=float((S > 1.0).mean())), fh, indent=2)
    print(f"alarm rate {rate:.4f}; index scale {d_scale:.4f}; KPI severity scale {s_scale:.3f}; "
          f"nominal KPI alarm rate {(S > 1.0).mean():.4f}")


if __name__ == "__main__":
    main()
