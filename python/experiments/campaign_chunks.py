"""Chunked execution of the main campaign across machines.

    python -m python.experiments.campaign_chunks run LO HI [WORKERS]
    python -m python.experiments.campaign_chunks merge

Tasks are the (scenario, seed index) blocks of run_experiments._block, enumerated
in a fixed order; every chunk writes a pickle to data/results/chunks/.
"""
import os
import pickle
import sys

import pandas as pd

from ..simulator.config import SCENARIO_NAMES
from . import run_experiments as RE

SEEDS = 24
STEPS = 500
CHUNKS = os.path.join(RE.RESULTS, "chunks")


def tasks():
    return [(scen, k, 0, STEPS, list(RE.POLICIES)) for scen in SCENARIO_NAMES for k in range(SEEDS)]


def run(lo, hi, workers):
    import multiprocessing as mp
    RE._CTX = RE.prepare_context(RE.DATA_DEFAULT)
    t = tasks()[lo:hi]
    with mp.get_context("fork").Pool(workers) as pool:
        res = pool.map(RE._block, t, chunksize=1)
    os.makedirs(CHUNKS, exist_ok=True)
    with open(os.path.join(CHUNKS, f"chunk_{lo:03d}_{hi:03d}.pkl"), "wb") as fh:
        pickle.dump(res, fh)
    print("ok", lo, hi, flush=True)


def merge():
    res = []
    for f in sorted(os.listdir(CHUNKS)):
        with open(os.path.join(CHUNKS, f), "rb") as fh:
            res.extend(pickle.load(fh))
    runs = pd.DataFrame([r for x in res for r in x[0]])
    keys = runs[["scenario", "seed", "policy"]].drop_duplicates()
    assert len(keys) == len(runs) == len(tasks()) * len(RE.POLICIES), (len(keys), len(runs))
    tl = pd.concat([t for x in res for t in x[1]], ignore_index=True)
    sv = pd.concat([t for x in res for t in x[2]], ignore_index=True)
    runs.to_csv(os.path.join(RE.RESULTS, "experiment_runs.csv"), index=False)
    tl.to_csv(os.path.join(RE.RESULTS, "experiment_timeline.csv"), index=False)
    sv.to_csv(os.path.join(RE.RESULTS, "survival_records.csv"), index=False)
    RE.aggregate(runs).to_csv(os.path.join(RE.RESULTS, "policy_comparison.csv"), index=False)
    RE.significance(runs).to_csv(os.path.join(RE.RESULTS, "significance_tests.csv"), index=False)
    RE.contrasts(runs).to_csv(os.path.join(RE.RESULTS, "contrasts.csv"), index=False)
    print(runs.pivot_table(index="scenario", columns="policy", values="resilience_loss").round(2).T)


if __name__ == "__main__":
    if sys.argv[1] == "run":
        run(int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]) if len(sys.argv) > 4 else 2)
    else:
        merge()
