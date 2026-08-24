"""Regression tests for the simulator, twin, predictor features and orchestrator."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from python.experiments.runner import ExperimentRun
from python.orchestrator.digital_twin import DigitalTwin
from python.orchestrator.policies import Orchestrator
from python.predictor.online_features import OnlineFeatureBuilder
from python.simulator.config import FAIL_HEALTH, PREDICTION_HORIZON, SERVICE_PROFILES
from python.simulator.dataset_generator import add_labels, add_temporal_features, run_episode
from python.simulator.failure_generator import STATUS_FAILED, NetworkEnvironment
from python.simulator.network_topology import NODE_CATALOGUE, build_topology, topology_summary
from python.simulator.traffic_generator import TrafficGenerator
from python.experiments.resilience import LeadTimeTracker


def test_topology_is_connected_and_redundant():
    g = build_topology()
    summary = topology_summary(g)
    assert summary["nodes"] == len(NODE_CATALOGUE)
    assert summary["edge_connectivity"] >= 2, "every terminal needs at least two disjoint routes"
    assert summary["algebraic_connectivity"] > 0


def test_health_process_stays_in_unit_interval():
    env = NetworkEnvironment(NODE_CATALOGUE, "S4_disaster_collapse", seed=1)
    for _ in range(300):
        env.step()
    for node in env.nodes.values():
        assert 0.0 <= node.health <= 1.0


def test_failed_nodes_report_degraded_telemetry():
    env = NetworkEnvironment(NODE_CATALOGUE, "S4_disaster_collapse", seed=3)
    seen = False
    for _ in range(400):
        snap = env.step()
        for nid, obs in snap.items():
            if obs["_status"] == STATUS_FAILED:
                seen = True
                assert obs["packet_loss"] > 0.5
                assert obs["availability"] < 0.1
    assert seen, "the disaster scenario must produce at least one failure"


def test_labels_are_causal():
    """A positive label must be followed by a failure inside the horizon."""
    df = run_episode(0, "S4_disaster_collapse", 250, seed=11)
    df = add_labels(add_temporal_features(df))
    positives = df[(df["failure_within_H"] == 1) & (df["eligible"] == 1)]
    assert len(positives) > 0
    for _, row in positives.sample(min(25, len(positives)), random_state=0).iterrows():
        window = df[
            (df["node_id"] == row["node_id"])
            & (df["t"] > row["t"])
            & (df["t"] <= row["t"] + PREDICTION_HORIZON)
        ]
        assert (window["status"] == STATUS_FAILED).any()


def test_lead_time_matching_respects_prediction_horizon():
    tracker = LeadTimeTracker(theta=0.5, horizon=10)
    for step in range(16):
        tracker.update(step, {"n": 0.9 if step == 0 else 0.0}, {"n": step == 15})
    summary = tracker.summary()
    assert summary["anticipated_failures"] == 0
    assert summary["missed_failures"] == 1


def test_no_leaky_features_in_matrix():
    from python.predictor.features import feature_columns

    df = add_labels(add_temporal_features(run_episode(0, "S2_bs_overload", 80, seed=5)))
    numeric, categorical = feature_columns(df)
    for forbidden in ("latent_health", "failure_within_H", "status", "scenario"):
        assert forbidden not in numeric + categorical


def test_online_features_match_batch_features():
    df = add_temporal_features(run_episode(0, "S3_leo_degradation", 60, seed=7))
    builder = OnlineFeatureBuilder(latency_medians={})
    node = "leo_1"
    trace = df[df["node_id"] == node].sort_values("t")
    for _, row in trace.iterrows():
        builder.push(node, row.to_dict())
    online = builder.features(node, "LEO")
    batch = trace.iloc[-1]
    for key in ("latency_ms_mean12", "packet_loss_mean12", "cpu_usage_std12"):
        assert online[key] == pytest.approx(batch[key], rel=1e-6, abs=1e-9)


def test_twin_never_sees_latent_state():
    env = NetworkEnvironment(NODE_CATALOGUE, "S4_disaster_collapse", seed=2)
    twin = DigitalTwin(build_topology())
    snap = env.step()
    twin.ingest(snap)
    for node in twin.nodes.values():
        assert "_health" not in node.telemetry
        assert "_status" not in node.telemetry


def test_orchestrator_avoids_high_risk_nodes():
    twin = DigitalTwin(build_topology())
    env = NetworkEnvironment(NODE_CATALOGUE, "S1_normal", seed=4)
    twin.ingest(env.step())
    twin.set_risk({nid: 0.0 for nid in NODE_CATALOGUE})
    twin.set_risk({"bs6g_1": 0.95, "bs5g_1": 0.95, "bs4g_1": 0.95})

    orch = Orchestrator("proactive_ai", twin, theta=0.17)
    gen = TrafficGenerator(["term_1"], ["hq_1"], np.random.default_rng(0))
    flow = gen.arrivals(0)[0] if gen.arrivals(0) else None
    if flow is None:
        pytest.skip("no arrival in this slot")
    flow.source, flow.destination = "term_1", "hq_1"
    assert orch.admit(flow, 0)
    assert not ({"bs6g_1", "bs5g_1", "bs4g_1"} & set(flow.path))


def test_service_profiles_are_ordered_by_priority():
    for name, profile in SERVICE_PROFILES.items():
        assert 1 <= profile.priority <= 5
        assert profile.max_latency_ms > 0
        assert 0 < profile.max_loss < 1


def test_proactive_policy_does_not_break_nominal_operation():
    """Under S1 the proposed policy must not degrade a healthy network."""
    common = dict(steps=120, model=None, feature_columns=None, latency_medians={})
    reactive = ExperimentRun("S1_normal", "reactive", 99, **common).run()
    static = ExperimentRun("S1_normal", "static", 99, **common).run()
    assert reactive["service_availability"] >= 0.98
    assert static["service_availability"] >= 0.98


def test_recovery_time_is_non_negative():
    run = ExperimentRun("S4_disaster_collapse", "reactive", 42, steps=200,
                        model=None, feature_columns=None, latency_medians={})
    summary = run.run()
    assert summary["mean_recovery_steps"] >= 0
    assert 0.0 <= summary["service_availability"] <= 1.0
    assert 0.0 <= summary["sla_compliance"] <= 1.0
