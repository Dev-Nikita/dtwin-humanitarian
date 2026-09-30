"""
Failure-prediction model training and comparison.

Five predictors are compared on the identical episode-disjoint split:

    0. Threshold rule  - the operator heuristic used as a non-learning baseline
    1. Logistic regression (L2, standardised inputs)
    2. Random forest
    3. XGBoost
    4. LightGBM

The decision threshold of every learned model is selected on the *validation*
episodes by maximising F1, then frozen before touching the test episodes.
Probabilities are isotonically calibrated on the validation fold, which matters
because the orchestrator consumes probabilities, not labels.

    python -m python.predictor.train --data data/processed/telemetry_dataset.csv
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .features import episode_split, feature_columns, select_eligible

try:
    from lightgbm import LGBMClassifier

    HAS_LGBM = True
except Exception:  # pragma: no cover
    HAS_LGBM = False

try:
    from xgboost import XGBClassifier

    HAS_XGB = True
except Exception:  # pragma: no cover
    HAS_XGB = False

RESULTS_DIR = os.path.join("data", "results")


# ---------------------------------------------------------------- heuristic
class ThresholdRule:
    """Operator heuristic: alarm when any KPI crosses a static threshold.

    This is the mechanism most production NMS platforms actually implement and
    is therefore the fair non-learning reference point.
    """

    def __init__(self, loss=0.12, latency_ratio=1.9, cpu=88.0, availability=0.80):
        self.loss = loss
        self.latency_ratio = latency_ratio
        self.cpu = cpu
        self.availability = availability

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        score = (
            (df["packet_loss"] > self.loss).astype(float) * 0.34
            + (df["latency_ms_ratio"] > self.latency_ratio).astype(float) * 0.28
            + (df["cpu_usage"] > self.cpu).astype(float) * 0.18
            + (df["availability"] < self.availability).astype(float) * 0.20
        )
        return np.column_stack([1.0 - score.values, score.values])


# --------------------------------------------------------------- utilities
def build_preprocessor(numeric: List[str], categorical: List[str], scale: bool) -> ColumnTransformer:
    steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        steps.append(("scale", StandardScaler()))
    num_tf = Pipeline(steps)
    return ColumnTransformer(
        transformers=[
            ("num", num_tf, numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ]
    )


def best_threshold(y_true: np.ndarray, prob: np.ndarray) -> float:
    grid = np.linspace(0.02, 0.95, 187)
    scores = [f1_score(y_true, (prob >= th).astype(int), zero_division=0) for th in grid]
    return float(grid[int(np.argmax(scores))])


def score_all(y_true: np.ndarray, prob: np.ndarray, threshold: float) -> Dict[str, float]:
    pred = (prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(y_true, pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "mcc": matthews_corrcoef(y_true, pred) if (tp + fp) and (tp + fn) else 0.0,
        "roc_auc": roc_auc_score(y_true, prob),
        "pr_auc": average_precision_score(y_true, prob),
        "brier": brier_score_loss(y_true, prob),
        "threshold": threshold,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "false_alarm_rate": fp / max(1, fp + tn),
    }


def make_models(seed: int) -> Dict[str, object]:
    models: Dict[str, object] = {
        "logistic_regression": LogisticRegression(
            max_iter=2000, C=0.8, class_weight="balanced", n_jobs=-1
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=400,
            max_depth=16,
            min_samples_leaf=8,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=seed,
        ),
    }
    if HAS_XGB:
        models["xgboost"] = XGBClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.8,
            reg_lambda=1.5,
            eval_metric="aucpr",
            tree_method="hist",
            n_jobs=-1,
            random_state=seed,
        )
    if HAS_LGBM:
        models["lightgbm"] = LGBMClassifier(
            n_estimators=600,
            learning_rate=0.05,
            num_leaves=63,
            min_child_samples=30,
            subsample=0.85,
            subsample_freq=1,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            class_weight="balanced",
            n_jobs=-1,
            random_state=seed,
            verbose=-1,
        )
    return models


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join("data", "processed", "telemetry_dataset.csv"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--results", default=RESULTS_DIR)
    args = ap.parse_args()

    os.makedirs(args.results, exist_ok=True)
    df = pd.read_parquet(args.data) if args.data.endswith(".parquet") else pd.read_csv(args.data)
    df = select_eligible(df)

    numeric, categorical = feature_columns(df)
    train_eps, val_eps, test_eps = episode_split(df, seed=args.seed)

    tr = df[df["episode"].isin(train_eps)]
    va = df[df["episode"].isin(val_eps)]
    te = df[df["episode"].isin(test_eps)]

    X_cols = numeric + categorical
    y_tr, y_va, y_te = tr["failure_within_H"].values, va["failure_within_H"].values, te["failure_within_H"].values

    print(f"features: {len(numeric)} numeric + {len(categorical)} categorical")
    print(f"train {len(tr):,} ({y_tr.mean():.4f})  val {len(va):,} ({y_va.mean():.4f})  "
          f"test {len(te):,} ({y_te.mean():.4f})")

    rows: List[Dict[str, float]] = []
    test_probs: Dict[str, np.ndarray] = {}

    # ---- non-learning heuristic
    rule = ThresholdRule()
    p_va = rule.predict_proba(va)[:, 1]
    p_te = rule.predict_proba(te)[:, 1]
    th = best_threshold(y_va, p_va)
    m = score_all(y_te, p_te, th)
    m.update({"model": "threshold_rule", "train_time_s": 0.0, "inference_ms_per_1k": 0.0,
              "val_pr_auc": float(average_precision_score(y_va, p_va)),
              "val_brier": float(brier_score_loss(y_va, p_va))})
    rows.append(m)
    test_probs["threshold_rule"] = p_te

    # ---- learned models
    for name, estimator in make_models(args.seed).items():
        scale = name == "logistic_regression"
        pipe = Pipeline(
            [("prep", build_preprocessor(numeric, categorical, scale)), ("model", estimator)]
        )
        t0 = time.perf_counter()
        pipe.fit(tr[X_cols], y_tr)
        train_time = time.perf_counter() - t0

        calibrated = CalibratedClassifierCV(pipe, method="isotonic", cv="prefit")
        calibrated.fit(va[X_cols], y_va)

        p_va = calibrated.predict_proba(va[X_cols])[:, 1]
        th = best_threshold(y_va, p_va)

        t0 = time.perf_counter()
        p_te = calibrated.predict_proba(te[X_cols])[:, 1]
        infer_ms = (time.perf_counter() - t0) * 1000.0 / max(1, len(te)) * 1000.0

        m = score_all(y_te, p_te, th)
        m.update({"model": name, "train_time_s": train_time, "inference_ms_per_1k": infer_ms,
                  "val_pr_auc": float(average_precision_score(y_va, p_va)),
                  "val_brier": float(brier_score_loss(y_va, p_va))})
        rows.append(m)
        test_probs[name] = p_te

        joblib.dump(calibrated, os.path.join(args.results, f"{name}_model.joblib"))
        print(f"{name:22s} F1={m['f1']:.3f} ROC-AUC={m['roc_auc']:.4f} "
              f"PR-AUC={m['pr_auc']:.3f} rec={m['recall']:.3f} prec={m['precision']:.3f}")

        if hasattr(estimator, "feature_importances_"):
            names = pipe.named_steps["prep"].get_feature_names_out()
            imp = pd.DataFrame(
                {"feature": names, "importance": estimator.feature_importances_}
            ).sort_values("importance", ascending=False)
            imp.to_csv(os.path.join(args.results, f"{name}_feature_importance.csv"), index=False)

    results = pd.DataFrame(rows)[
        ["model", "accuracy", "balanced_accuracy", "precision", "recall", "f1", "mcc",
         "roc_auc", "pr_auc", "brier", "false_alarm_rate", "threshold",
         "tn", "fp", "fn", "tp", "train_time_s", "inference_ms_per_1k",
         "val_pr_auc", "val_brier"]
    ]
    results.to_csv(os.path.join(args.results, "model_comparison.csv"), index=False)

    np.savez_compressed(
        os.path.join(args.results, "test_predictions.npz"),
        y_true=y_te,
        node_type=te["node_type"].values,
        **{f"prob_{k}": v for k, v in test_probs.items()},
    )

    # Model selection uses VALIDATION information only (Reviewer 4.3): the
    # learned model with the highest validation PR-AUC is chosen; models within
    # SELECTION_TOL of the best are treated as tied and the tie is broken by the
    # validation Brier score and then by inference cost. Test metrics are
    # computed once, after the choice, and never enter it.
    SELECTION_TOL = 0.01
    learned = results[results["model"] != "threshold_rule"].copy()
    top = learned["val_pr_auc"].max()
    tied = learned[learned["val_pr_auc"] >= top - SELECTION_TOL]
    best = tied.sort_values(["val_brier", "inference_ms_per_1k"]).iloc[0]["model"]
    with open(os.path.join(args.results, "best_model.json"), "w") as fh:
        json.dump(
            {
                "best_model": best,
                "selection_metric": "validation PR-AUC (tol 0.01), then validation Brier, then inference cost",
                "tied_models": sorted(tied["model"].tolist()),
                "threshold": float(results.set_index("model").loc[best, "threshold"]),
                "n_train": int(len(tr)),
                "n_val": int(len(va)),
                "n_test": int(len(te)),
                "positive_rate_test": float(y_te.mean()),
                "n_features": len(numeric) + len(categorical),
            },
            fh,
            indent=2,
        )
    print("\nbest model:", best)
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
