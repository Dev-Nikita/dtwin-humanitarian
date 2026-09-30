"""
Telemetry dataset generation for the failure-prediction task.

The generator runs a set of independent episodes (each episode = one scenario,
one random seed) and records the per-node telemetry vector at every slot.
Afterwards it derives

    * temporal features  : rolling mean / std / slope / delta over a window W
    * the target variable: y = 1 if the node enters the FAILED state within the
      next H slots, evaluated only on slots where the node is currently up.

Because the label refers to the *future* and the latent health is hidden, the
resulting problem is a genuine early-warning task rather than a threshold rule.

Usage
-----
    python -m python.simulator.dataset_generator --episodes 32 --steps 500
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .config import (
    EPISODE_STEPS,
    FEATURE_WINDOW,
    PREDICTION_HORIZON,
    RNG_SEED,
    SCENARIO_NAMES,
    TRAIN_SCENARIOS,
)
from .failure_generator import STATUS_FAILED, TELEMETRY_FIELDS, NetworkEnvironment
from .network_topology import NODE_CATALOGUE

ROLLING_FIELDS = [
    "latency_ms",
    "jitter_ms",
    "packet_loss",
    "throughput_mbps",
    "cpu_usage",
    "memory_usage",
    "signal_strength",
    "availability",
    "retx_rate",
    "queue_depth",
]


def run_episode(episode_id: int, scenario: str, steps: int, seed: int,
                **env_kwargs) -> pd.DataFrame:
    """Simulate one episode and return the raw telemetry trace."""
    env = NetworkEnvironment(NODE_CATALOGUE, scenario, seed=seed, **env_kwargs)
    records: List[Dict[str, float]] = []
    for t in range(steps):
        snapshot = env.step()
        for node_id, obs in snapshot.items():
            row = {
                "episode": episode_id,
                "scenario": scenario,
                "t": t,
                "node_id": node_id,
                "node_type": env.nodes[node_id].node_type,
                "status": obs["_status"],
                "latent_health": obs["_health"],
            }
            for key in TELEMETRY_FIELDS:
                row[key] = obs[key]
            records.append(row)
    return pd.DataFrame.from_records(records)


def add_temporal_features(df: pd.DataFrame, window: int = FEATURE_WINDOW) -> pd.DataFrame:
    """Add rolling statistics computed per (episode, node)."""
    df = df.sort_values(["episode", "node_id", "t"]).reset_index(drop=True)
    grouped = df.groupby(["episode", "node_id"], sort=False)
    new_cols: Dict[str, pd.Series] = {}

    ramp = np.arange(window) - (window - 1) / 2.0
    denom = float((ramp ** 2).sum())

    for field in ROLLING_FIELDS:
        roll = grouped[field].rolling(window, min_periods=2)
        new_cols[f"{field}_mean{window}"] = roll.mean().reset_index(level=[0, 1], drop=True)
        new_cols[f"{field}_std{window}"] = (
            roll.std().reset_index(level=[0, 1], drop=True).fillna(0.0)
        )
        new_cols[f"{field}_slope{window}"] = (
            grouped[field]
            .rolling(window, min_periods=window)
            .apply(lambda v: float(np.dot(v - v.mean(), ramp) / denom), raw=True)
            .reset_index(level=[0, 1], drop=True)
            .fillna(0.0)
        )
        new_cols[f"{field}_delta"] = grouped[field].diff().fillna(0.0)

    feat = pd.DataFrame(new_cols, index=df.index)
    df = pd.concat([df, feat], axis=1)

    # normalised departure of the instantaneous value from its own baseline
    for field in ("latency_ms", "packet_loss", "cpu_usage"):
        base = df[f"{field}_mean{window}"].replace(0.0, np.nan)
        df[f"{field}_ratio"] = (df[field] / base).fillna(1.0).clip(0.0, 20.0)

    df["degradation_index"] = (
        0.35 * df["packet_loss"].clip(0, 1)
        + 0.25 * (df["latency_ms"] / df["latency_ms"].groupby(df["node_type"]).transform("median"))
        + 0.20 * (1.0 - df["availability"])
        + 0.20 * df["retx_rate"]
    )
    return df


def add_labels(df: pd.DataFrame, horizon: int = PREDICTION_HORIZON) -> pd.DataFrame:
    """y = 1 if the node enters FAILED within the next `horizon` slots."""
    df = df.sort_values(["episode", "node_id", "t"]).reset_index(drop=True)
    is_failed = (df["status"] == STATUS_FAILED).astype(int)
    grouped = is_failed.groupby([df["episode"], df["node_id"]])
    future = (
        grouped.transform(
            lambda s: s.shift(-1).rolling(horizon, min_periods=1).max().shift(-(horizon - 1))
        )
        .fillna(0.0)
        .astype(int)
    )
    df["failure_within_H"] = future
    df["currently_failed"] = is_failed
    # evaluation is restricted to slots where the node is still operational
    df["eligible"] = (df["currently_failed"] == 0).astype(int)
    return df


def build_dataset(
    episodes_per_scenario: int = 8,
    steps: int = EPISODE_STEPS,
    seed: int = RNG_SEED,
    scenarios: Optional[List[str]] = None,
    **env_kwargs,
) -> pd.DataFrame:
    """Build the training corpus.

    Only the scenarios in TRAIN_SCENARIOS are used by default: S5 and S6 are
    held out so that both the predictor and the closed-loop policy can be
    evaluated on degradation mechanisms they have never observed.
    """
    frames: List[pd.DataFrame] = []
    episode_id = 0
    for scenario in scenarios or TRAIN_SCENARIOS:
        for k in range(episodes_per_scenario):
            frames.append(run_episode(episode_id, scenario, steps, seed + 977 * episode_id + k,
                                      **env_kwargs))
            episode_id += 1
    raw = pd.concat(frames, ignore_index=True)
    raw = add_temporal_features(raw)
    raw = add_labels(raw)
    return raw


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the telemetry dataset")
    parser.add_argument("--episodes", type=int, default=8, help="episodes per scenario")
    parser.add_argument("--steps", type=int, default=EPISODE_STEPS)
    parser.add_argument("--seed", type=int, default=RNG_SEED)
    parser.add_argument("--scenarios", nargs="*", default=None)
    parser.add_argument(
        "--out",
        type=str,
        default=os.path.join("data", "processed", "telemetry_dataset.parquet"),
    )
    args = parser.parse_args()

    df = build_dataset(args.episodes, args.steps, args.seed, args.scenarios)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    if args.out.endswith(".parquet"):
        try:
            df.to_parquet(args.out, index=False)
        except Exception:  # pragma: no cover - fallback when pyarrow is absent
            args.out = args.out.replace(".parquet", ".csv")
            df.to_csv(args.out, index=False)
    else:
        df.to_csv(args.out, index=False)

    eligible = df[df["eligible"] == 1]
    print(f"rows                : {len(df):,}")
    print(f"eligible rows       : {len(eligible):,}")
    print(f"positive rate       : {eligible['failure_within_H'].mean():.4f}")
    print(f"failed node-slots   : {int(df['currently_failed'].sum()):,}")
    print(f"episodes            : {df['episode'].nunique()}")
    print(f"features            : {df.shape[1]}")
    print(f"saved               : {args.out}")


if __name__ == "__main__":
    main()
