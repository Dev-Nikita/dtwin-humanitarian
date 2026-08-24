"""Calibration diagnostics for the anticipation model, and a check that the
operating point does not depend on reusing the calibration fold for threshold
selection.

Writes data/results/calibration_audit.csv and calibration_curve.csv.
"""
from __future__ import annotations

import os
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ..predictor.features import episode_split, feature_columns, select_eligible

RESULTS = os.path.join("data", "results")
DATA = os.path.join("data", "processed", "telemetry_dataset.parquet")
GRID = np.linspace(0.02, 0.95, 187)


def _fit(df, cols, m_tr, m_cal):
    prep = ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                          ("scale", StandardScaler())]), cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), ["node_type"]),
    ])
    pipe = Pipeline([("prep", prep),
                     ("model", LogisticRegression(max_iter=2000, C=0.8,
                                                  class_weight="balanced", n_jobs=-1))])
    X = df[cols + ["node_type"]]
    y = df["failure_within_H"].to_numpy()
    pipe.fit(X[m_tr], y[m_tr])
    model = CalibratedClassifierCV(pipe, method="isotonic", cv="prefit")
    model.fit(X[m_cal], y[m_cal])
    return model, X, y


def _ece(y, p, bins: int = 15) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    total = 0.0
    for b in range(bins):
        m = idx == b
        if not m.any():
            continue
        total += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(total)


def _slope_intercept(y, p) -> Dict[str, float]:
    """Cox calibration: logistic regression of the outcome on the logit of the
    predicted probability. Slope 1 and intercept 0 indicate perfect calibration."""
    eps = 1e-6
    logit = np.log(np.clip(p, eps, 1 - eps) / (1 - np.clip(p, eps, 1 - eps)))
    lr = LogisticRegression(max_iter=1000)
    lr.fit(logit.reshape(-1, 1), y)
    return {"calibration_slope": float(lr.coef_[0][0]),
            "calibration_intercept": float(lr.intercept_[0])}


def main() -> pd.DataFrame:
    df = select_eligible(pd.read_parquet(DATA))
    numeric, _ = feature_columns(df)
    tr, va, te = episode_split(df, seed=42)

    # split the validation episodes into a calibration half and a disjoint
    # threshold-selection half, so the operating point never sees the fold that
    # fitted the isotonic map.
    va = sorted(va)
    rng = np.random.default_rng(17)
    perm = rng.permutation(len(va))
    cal_eps = {va[i] for i in perm[: len(va) // 2]}
    thr_eps = {va[i] for i in perm[len(va) // 2:]}

    m_tr = df["episode"].isin(tr).to_numpy()
    m_va = df["episode"].isin(set(va)).to_numpy()
    m_cal = df["episode"].isin(cal_eps).to_numpy()
    m_thr = df["episode"].isin(thr_eps).to_numpy()
    m_te = df["episode"].isin(te).to_numpy()

    rows = []
    curve_rows = []
    for name, cal_mask, thr_mask in (
        ("shared fold (as reported)", m_va, m_va),
        ("disjoint folds", m_cal, m_thr),
    ):
        model, X, y = _fit(df, numeric, m_tr, cal_mask)
        p_thr = model.predict_proba(X[thr_mask])[:, 1]
        theta = float(GRID[int(np.argmax(
            [f1_score(y[thr_mask], (p_thr >= t).astype(int), zero_division=0) for t in GRID]))])
        p_te = model.predict_proba(X[m_te])[:, 1]
        y_te = y[m_te]
        pred = (p_te >= theta).astype(int)
        row = {
            "setting": name,
            "threshold": theta,
            "alarm_rate": float(pred.mean()),
            "roc_auc": float(roc_auc_score(y_te, p_te)),
            "brier": float(brier_score_loss(y_te, p_te)),
            "ece": _ece(y_te, p_te),
            "f1": float(f1_score(y_te, pred, zero_division=0)),
        }
        row.update(_slope_intercept(y_te, p_te))
        rows.append(row)
        print(f"{name:26s} theta={theta:.3f} alarm={row['alarm_rate']:.4f} "
              f"ECE={row['ece']:.5f} slope={row['calibration_slope']:.3f}", flush=True)

        if name == "shared fold (as reported)":
            edges = np.linspace(0.0, 1.0, 16)
            idx = np.clip(np.digitize(p_te, edges) - 1, 0, 14)
            for b in range(15):
                m = idx == b
                if m.sum() < 20:
                    continue
                k = int(y_te[m].sum())
                n = int(m.sum())
                # Wilson interval for the observed frequency
                z, ph = 1.96, k / n
                denom = 1 + z * z / n
                centre = (ph + z * z / (2 * n)) / denom
                half = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / denom
                curve_rows.append({"bin_mid": float((edges[b] + edges[b + 1]) / 2),
                                   "predicted": float(p_te[m].mean()),
                                   "observed": float(ph), "n": n,
                                   "ci_low": float(max(0.0, centre - half)),
                                   "ci_high": float(min(1.0, centre + half))})

    out = pd.DataFrame(rows)
    os.makedirs(RESULTS, exist_ok=True)
    out.to_csv(os.path.join(RESULTS, "calibration_audit.csv"), index=False)
    pd.DataFrame(curve_rows).to_csv(os.path.join(RESULTS, "calibration_curve.csv"), index=False)
    return out


if __name__ == "__main__":
    main()
