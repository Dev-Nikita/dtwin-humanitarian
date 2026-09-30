"""
predictor-service: on-line failure-risk inference.

The service keeps a per-node ring buffer of the last W telemetry samples,
reconstructs exactly the feature vector used during training and returns the
calibrated probability that the element enters a failed state within the next H
slots, together with the operating point selected on the validation episodes.

    uvicorn python.api.predictor_api:app --host 0.0.0.0 --port 8081
"""

from __future__ import annotations

import json
import os
import time
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

from ..predictor.online_features import OnlineFeatureBuilder
from ..simulator.config import FEATURE_WINDOW, PREDICTION_HORIZON

RESULTS = os.environ.get("RESULTS_DIR", os.path.join("data", "results"))

app = FastAPI(
    title="AI Digital Twin - failure predictor",
    version="1.0.0",
    description="Calibrated early-warning inference for hybrid 5G/6G + LEO humanitarian networks",
)

_state: Dict[str, object] = {}
_counters = {"requests": 0, "alarms": 0, "unknown_node": 0}


class TelemetryIn(BaseModel):
    node_id: str
    node_type: str
    latency_ms: float
    jitter_ms: float
    packet_loss: float = Field(ge=0.0, le=1.0)
    throughput_mbps: float
    cpu_usage: float
    memory_usage: float
    signal_strength: float
    availability: float = Field(ge=0.0, le=1.0)
    utilisation: float = Field(ge=0.0, le=1.0)
    queue_depth: float
    retx_rate: float = Field(ge=0.0, le=1.0)
    energy_headroom: float = Field(ge=0.0, le=1.0)


class BatchIn(BaseModel):
    samples: List[TelemetryIn]


class RiskOut(BaseModel):
    node_id: str
    failure_probability: float
    risk_status: str
    threshold: float
    horizon_slots: int
    warmup: bool


@app.on_event("startup")
def _load() -> None:
    with open(os.path.join(RESULTS, "best_model.json")) as fh:
        meta = json.load(fh)
    _state["meta"] = meta
    _state["model"] = joblib.load(os.path.join(RESULTS, f"{meta['best_model']}_model.joblib"))
    with open(os.path.join(RESULTS, "feature_columns.json")) as fh:
        _state["columns"] = json.load(fh)
    medians_path = os.path.join(RESULTS, "node_type_latency_median.json")
    medians = json.load(open(medians_path)) if os.path.exists(medians_path) else {}
    _state["builder"] = OnlineFeatureBuilder(latency_medians=medians)
    _state["started"] = time.time()


def _status(prob: float, threshold: float) -> str:
    if prob >= max(0.5, 2.5 * threshold):
        return "critical"
    if prob >= threshold:
        return "high_risk"
    if prob >= 0.5 * threshold:
        return "watch"
    return "low_risk"


def _predict(samples: List[TelemetryIn]) -> List[RiskOut]:
    builder: OnlineFeatureBuilder = _state["builder"]  # type: ignore[assignment]
    model = _state["model"]
    columns: List[str] = _state["columns"]  # type: ignore[assignment]
    threshold = float(_state["meta"]["threshold"])  # type: ignore[index]

    rows, ids, warm = [], [], []
    for s in samples:
        payload = s.model_dump()
        node_id = payload.pop("node_id")
        node_type = payload.pop("node_type")
        builder.push(node_id, payload)
        rows.append(builder.features(node_id, node_type))
        ids.append(node_id)
        warm.append(len(builder.buffers[node_id]) < FEATURE_WINDOW)

    frame = pd.DataFrame(rows)
    for col in columns:
        if col not in frame.columns:
            frame[col] = np.nan
    frame = frame[columns]
    probs = model.predict_proba(frame)[:, 1]

    _counters["requests"] += len(ids)
    _counters["alarms"] += int((probs >= threshold).sum())
    return [
        RiskOut(
            node_id=nid,
            failure_probability=float(p),
            risk_status=_status(float(p), threshold),
            threshold=threshold,
            horizon_slots=PREDICTION_HORIZON,
            warmup=bool(w),
        )
        for nid, p, w in zip(ids, probs, warm)
    ]


@app.get("/healthz")
def healthz() -> Dict[str, object]:
    return {"status": "ok", "model": _state.get("meta", {}).get("best_model")}


@app.get("/readyz")
def readyz() -> Dict[str, object]:
    if "model" not in _state:
        raise HTTPException(status_code=503, detail="model not loaded")
    return {"status": "ready", "features": len(_state["columns"])}  # type: ignore[arg-type]


@app.post("/api/v1/predict", response_model=RiskOut)
def predict(sample: TelemetryIn) -> RiskOut:
    return _predict([sample])[0]


@app.post("/api/v1/predict/batch", response_model=List[RiskOut])
def predict_batch(batch: BatchIn) -> List[RiskOut]:
    if not batch.samples:
        raise HTTPException(status_code=400, detail="empty batch")
    return _predict(batch.samples)


@app.post("/api/v1/reset")
def reset() -> Dict[str, str]:
    _state["builder"].reset()  # type: ignore[union-attr]
    return {"status": "buffers cleared"}


@app.get("/api/v1/model")
def model_info() -> Dict[str, object]:
    return dict(_state.get("meta", {}))  # type: ignore[arg-type]


@app.get("/metrics", response_class=PlainTextResponse)
def metrics() -> str:
    uptime = time.time() - float(_state.get("started", time.time()))  # type: ignore[arg-type]
    return (
        "# HELP predictor_requests_total Telemetry samples scored\n"
        "# TYPE predictor_requests_total counter\n"
        f"predictor_requests_total {_counters['requests']}\n"
        "# HELP predictor_alarms_total Samples above the operating point\n"
        "# TYPE predictor_alarms_total counter\n"
        f"predictor_alarms_total {_counters['alarms']}\n"
        f"predictor_uptime_seconds {uptime:.1f}\n"
    )
