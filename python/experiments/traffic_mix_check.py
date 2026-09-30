"""Verify that the nominal traffic mix used to weight Eq. (17) matches the mix
actually carried in the closed loop. Writes data/results/traffic_mix_check.csv."""
from __future__ import annotations

import os
import numpy as np
import pandas as pd

from ..simulator.config import SERVICE_PROFILES
from .runner import ExperimentRun

RESULTS = os.path.join("data", "results")
SCENARIOS = ["S2_bs_overload", "S4_disaster_collapse", "S6_common_cause"]
SEEDS = (1000, 1037)


def main() -> pd.DataFrame:
    rows = []
    for scen in SCENARIOS:
        for seed in SEEDS:
            r = ExperimentRun(scenario=scen, policy="reactive", seed=seed, steps=500).run()
            rows.append({k[len("slots_"):]: v for k, v in r.items() if k.startswith("slots_")})
    df = pd.DataFrame(rows)
    observed = df.sum() / df.sum().sum()
    nominal = {c: SERVICE_PROFILES[c].arrival_weight
               * float(np.mean(SERVICE_PROFILES[c].duration_steps)) for c in df.columns}
    total = sum(nominal.values())
    out = pd.DataFrame({"nominal_share": pd.Series({k: v / total for k, v in nominal.items()}),
                        "observed_share": observed})
    out["difference_pp"] = 100.0 * (out.observed_share - out.nominal_share)
    out = out.reset_index().rename(columns={"index": "service_class"})
    os.makedirs(RESULTS, exist_ok=True)
    out.to_csv(os.path.join(RESULTS, "traffic_mix_check.csv"), index=False)
    print(out.round(4).to_string(index=False))
    print("max |difference|:", round(out.difference_pp.abs().max(), 2), "pp")
    return out


if __name__ == "__main__":
    main()
