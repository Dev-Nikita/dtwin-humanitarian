"""Feature contract shared by training, evaluation and on-line inference."""

from __future__ import annotations

from typing import List, Tuple

import pandas as pd

# Columns that must never be used as predictors.
LEAKY_COLUMNS = {
    "latent_health",       # hidden ground-truth state
    "status",              # derived from the hidden state
    "failure_within_H",    # target
    "currently_failed",
    "eligible",
    "episode",
    "t",
    "node_id",
    "scenario",            # not observable at inference time in the field
}

CATEGORICAL_FEATURES: List[str] = ["node_type"]


def feature_columns(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """Return (numeric_features, categorical_features) for a telemetry frame."""
    numeric = [
        c
        for c in df.columns
        if c not in LEAKY_COLUMNS
        and c not in CATEGORICAL_FEATURES
        and pd.api.types.is_numeric_dtype(df[c])
    ]
    categorical = [c for c in CATEGORICAL_FEATURES if c in df.columns]
    return numeric, categorical


def select_eligible(df: pd.DataFrame) -> pd.DataFrame:
    """Restrict to slots in which the node is still operational."""
    return df[df["eligible"] == 1].reset_index(drop=True)


def episode_split(
    df: pd.DataFrame, train_frac: float = 0.60, val_frac: float = 0.20, seed: int = 7
):
    """Split by episode so that no temporal window is shared across folds."""
    rng_order = (
        df[["episode", "scenario"]]
        .drop_duplicates()
        .sample(frac=1.0, random_state=seed)
        .reset_index(drop=True)
    )
    train_eps, val_eps, test_eps = [], [], []
    for _, group in rng_order.groupby("scenario"):
        eps = group["episode"].tolist()
        n = len(eps)
        n_tr = max(1, int(round(train_frac * n)))
        n_va = max(1, int(round(val_frac * n))) if n - n_tr > 1 else 0
        train_eps += eps[:n_tr]
        val_eps += eps[n_tr : n_tr + n_va]
        test_eps += eps[n_tr + n_va :]
    return set(train_eps), set(val_eps), set(test_eps)
