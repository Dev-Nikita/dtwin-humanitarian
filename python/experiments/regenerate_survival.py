"""Regenerate data/results/survival_records.csv from scratch.

The survival table is per-flow rather than per-run, so it must be produced by a
single deterministic pass: appending to it across campaign chunks duplicates the
configurations that are re-run. This module writes exactly one block per
(scenario, policy, seed) and refuses to append.

    python -m python.experiments.regenerate_survival [scenario ...]

With no arguments every scenario is regenerated. Pass APPEND=1 in the
environment to add scenarios to an existing file, which is only safe because
each call rewrites whole (scenario, policy, seed) blocks.
"""
from __future__ import annotations

import os
import sys

import pandas as pd

from ..simulator.config import SCENARIO_NAMES
from .run_experiments import POLICIES, prepare_context
from .runner import ExperimentRun

RESULTS = os.path.join("data", "results")
DATA = os.path.join("data", "processed", "telemetry_dataset.parquet")
SEEDS = (1000, 1037, 1074)          # the three paired seeds used for survival
STEPS = 500
KEY = ["scenario", "policy", "seed"]


def main(scenarios=None) -> pd.DataFrame:
    ctx = prepare_context(DATA)
    theta = float(ctx["meta"]["threshold"])
    scenarios = scenarios or list(SCENARIO_NAMES)
    frames = []
    for scen in scenarios:
        for policy in POLICIES:
            for seed in SEEDS:
                run = ExperimentRun(
                    scen, policy, seed, steps=STEPS,
                    model=ctx["model"], feature_columns=ctx["feature_columns"],
                    latency_medians=ctx["latency_medians"], theta=theta,
                )
                run.run()
                if not run.survival:
                    continue
                frames.append(pd.DataFrame({
                    "scenario": scen, "policy": policy, "seed": seed,
                    "duration": [r.duration for r in run.survival],
                    "event": [r.event for r in run.survival],
                    "duration_sustained": [r.duration for r in run.survival_sustained],
                    "event_sustained": [r.event for r in run.survival_sustained],
                }))
        print(f"{scen} done", flush=True)

    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    path = os.path.join(RESULTS, "survival_records.csv")
    if os.environ.get("APPEND") == "1" and os.path.exists(path) and not out.empty:
        previous = pd.read_csv(path)
        done = set(map(tuple, out[KEY].drop_duplicates().to_numpy()))
        keep = ~previous[KEY].apply(tuple, axis=1).isin(done)
        out = pd.concat([previous[keep], out], ignore_index=True)
    out.to_csv(path, index=False)
    print("rows", len(out))
    return out


if __name__ == "__main__":
    main(sys.argv[1:] or None)
