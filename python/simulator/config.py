"""
Global configuration for the hybrid 5G/6G + LEO humanitarian network simulator.

All physical constants are expressed in engineering units:
    latency          : ms
    bandwidth        : Mbit/s
    packet loss      : fraction in [0, 1]
    signal strength  : dBm
    slot duration    : s

The numbers below are order-of-magnitude consistent with 3GPP TR 38.821
(NTN), 3GPP TS 22.261 (5G service requirements) and published Starlink /
OneWeb measurement studies. They are *not* claimed to reproduce any single
deployment; they define a reproducible synthetic environment.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

# ----------------------------------------------------------------------------
# Simulation-wide constants
# ----------------------------------------------------------------------------

SLOT_SECONDS: float = 1.0          # duration of one discrete simulation step
EPISODE_STEPS: int = 500           # steps per episode (~8.3 min of wall time)
PREDICTION_HORIZON: int = 10       # H: predict failure within next H steps
FEATURE_WINDOW: int = 12           # W: rolling window used for temporal features
FAIL_HEALTH: float = 0.15          # h below which a node is declared FAILED
DEGRADED_HEALTH: float = 0.45      # h below which a node is DEGRADED
REPAIR_STEPS: Tuple[int, int] = (12, 38)   # uniform repair duration range
RNG_SEED: int = 20250815

# ----------------------------------------------------------------------------
# Node type catalogue
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class NodeProfile:
    """Static physical profile of a class of network nodes."""

    node_type: str
    base_latency_ms: float          # L0: propagation + processing floor
    base_jitter_ms: float           # J0
    capacity_mbps: float            # C: nominal capacity
    base_loss: float                # p0: loss floor under no stress
    base_signal_dbm: float          # S0 (0 for wired/backhaul nodes)
    wireless: bool
    hazard_scale: float             # relative susceptibility to degradation
    mtbf_steps: float               # mean time between failures under nominal load
    battery_backed: bool


NODE_PROFILES: Dict[str, NodeProfile] = {
    "4G": NodeProfile("4G", 45.0, 8.0, 120.0, 0.0018, -72.0, True, 1.30, 4200, True),
    "5G": NodeProfile("5G", 18.0, 4.0, 600.0, 0.0008, -68.0, True, 1.00, 5200, True),
    "6G": NodeProfile("6G", 7.0, 2.0, 1400.0, 0.0006, -64.0, True, 0.85, 6000, True),
    "LEO": NodeProfile("LEO", 42.0, 11.0, 320.0, 0.0025, -92.0, True, 1.55, 3400, True),
    "EDGE": NodeProfile("EDGE", 6.0, 1.5, 900.0, 0.0004, 0.0, False, 0.75, 8000, False),
    "HQ": NodeProfile("HQ", 4.0, 1.0, 2000.0, 0.0003, 0.0, False, 0.45, 12000, False),
    "UAV": NodeProfile("UAV", 22.0, 6.0, 180.0, 0.002, -78.0, True, 1.70, 2600, True),
    "MOBILE": NodeProfile("MOBILE", 3.0, 1.0, 200.0, 0.0012, -80.0, True, 1.10, 7000, True),
}

NODE_TYPES: List[str] = list(NODE_PROFILES.keys())

# ----------------------------------------------------------------------------
# Humanitarian service classes  (Table 2 of the paper)
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class ServiceProfile:
    name: str
    priority: int                 # 5 = highest
    max_latency_ms: float
    max_loss: float
    min_bandwidth_mbps: float
    arrival_weight: float         # relative arrival intensity
    duration_steps: Tuple[int, int]


SERVICE_PROFILES: Dict[str, ServiceProfile] = {
    "emergency_message": ServiceProfile("emergency_message", 5, 180.0, 0.06, 0.2, 0.18, (1, 3)),
    "command_control": ServiceProfile("command_control", 5, 150.0, 0.030, 1.0, 0.10, (3, 12)),
    "medical_telemetry": ServiceProfile("medical_telemetry", 4, 200.0, 0.040, 2.0, 0.17, (5, 25)),
    "voice_call": ServiceProfile("voice_call", 3, 250.0, 0.05, 0.5, 0.22, (8, 40)),
    "video_stream": ServiceProfile("video_stream", 2, 320.0, 0.08, 8.0, 0.18, (10, 60)),
    "sensor_data": ServiceProfile("sensor_data", 1, 800.0, 0.12, 0.3, 0.15, (1, 6)),
}

SERVICE_TYPES: List[str] = list(SERVICE_PROFILES.keys())

# ----------------------------------------------------------------------------
# Stress scenarios (Section 8 of the paper)
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class Scenario:
    """A stress programme applied to the environment during an episode."""

    name: str
    description: str
    load_multiplier: float                     # global traffic scaling
    hazard_multiplier: Dict[str, float] = field(default_factory=dict)
    shock_prob: float = 0.0                    # per-step probability of a shock
    shock_targets: Tuple[str, ...] = ()
    shock_magnitude: float = 0.0               # instantaneous health drop
    burst_period: int = 0                      # 0 = no periodic burst
    destroyed_nodes: Tuple[str, ...] = ()      # elements physically destroyed at t=0
    # --- S5: measurement-informed non-terrestrial dynamics --------------------
    handover_period: int = 0                   # slots between LEO reconfigurations
    handover_duration: int = 0                 # slots of transient degradation
    handover_fail_prob: float = 0.0            # probability that a handover fails
    handover_phase: int = 0                    # slot offset of the reconfiguration grid
    latency_override: Tuple[Tuple[str, float, float], ...] = ()  # (node_type, base_ms, jitter_ms)
    # --- S6: common-cause (correlated) disruptions ---------------------------
    common_cause_rate: float = 0.0             # per-slot probability of a regional shock
    common_cause_regions: Tuple[str, ...] = ()
    common_cause_magnitude: float = 0.0        # mean intensity of the shared shock


SCENARIOS: Dict[str, Scenario] = {
    "S1_normal": Scenario(
        "S1_normal",
        "Nominal humanitarian operation, moderate load, no external shocks.",
        load_multiplier=1.00,
        hazard_multiplier={},
        shock_prob=0.0003,
        shock_targets=("4G", "5G", "6G", "LEO", "UAV", "EDGE"),
        shock_magnitude=0.22,
    ),
    "S2_bs_overload": Scenario(
        "S2_bs_overload",
        "Population influx into a camp; terrestrial base stations congested.",
        load_multiplier=1.85,
        hazard_multiplier={"4G": 4.5, "5G": 4.2, "6G": 3.8, "EDGE": 3.2},
        shock_prob=0.0009,
        shock_targets=("4G", "5G", "6G"),
        shock_magnitude=0.25,
        burst_period=90,
    ),
    "S3_leo_degradation": Scenario(
        "S3_leo_degradation",
        "Ionospheric scintillation and handover churn degrade the LEO segment.",
        load_multiplier=1.25,
        hazard_multiplier={"LEO": 4.0, "UAV": 3.0},
        shock_prob=0.0012,
        shock_targets=("LEO", "UAV"),
        shock_magnitude=0.25,
        burst_period=0,
        destroyed_nodes=("bs5g_3", "bs4g_2", "bs6g_2"),
    ),
    "S4_disaster_collapse": Scenario(
        "S4_disaster_collapse",
        "Disaster/conflict zone: cascading loss of terrestrial and power infrastructure.",
        load_multiplier=2.15,
        hazard_multiplier={"4G": 6.0, "5G": 6.8, "6G": 6.5, "EDGE": 5.5, "LEO": 3.4, "UAV": 4.2},
        shock_prob=0.0020,
        shock_targets=("4G", "5G", "6G", "EDGE", "UAV"),
        shock_magnitude=0.30,
        burst_period=60,
        destroyed_nodes=("bs5g_3", "bs4g_2", "bs6g_2", "bs4g_1", "bs5g_2"),
    ),
    # ---------------------------------------------------------------- S5, S6
    # Both scenarios are held out of model training: the predictor never sees a
    # handover-driven telemetry signature (S5) nor a common-cause disruption
    # (S6). They therefore act as out-of-distribution stress tests of the whole
    # closed loop rather than as additional training conditions.
    "S5_leo_handover": Scenario(
        "S5_leo_handover",
        "Non-terrestrial operation with periodic handover reconfiguration; timing and "
        "transient-degradation magnitudes are informed by published LEO measurement studies.",
        load_multiplier=1.35,
        hazard_multiplier={"LEO": 2.4, "UAV": 2.0},
        shock_prob=0.0006,
        shock_targets=("LEO",),
        shock_magnitude=0.25,
        destroyed_nodes=("bs5g_3", "bs4g_2", "bs6g_2"),
        handover_period=15,
        handover_duration=2,
        handover_fail_prob=0.045,
    ),
    "S6_common_cause": Scenario(
        "S6_common_cause",
        "Regional common-cause disruption: shared power and backhaul dependencies inject a "
        "correlated shock into all elements of an affected region.",
        load_multiplier=1.70,
        hazard_multiplier={"4G": 2.6, "5G": 2.4, "6G": 2.2, "EDGE": 2.2, "UAV": 2.0, "LEO": 1.4},
        shock_prob=0.0004,
        shock_targets=("4G", "5G", "6G"),
        shock_magnitude=0.20,
        common_cause_rate=0.014,
        common_cause_regions=("west", "east"),
        common_cause_magnitude=0.78,
    ),
    "S7_external_ntn": Scenario(
        "S7_external_ntn",
        "Externally anchored non-terrestrial operation: the reconfiguration cadence, its "
        "phase within the minute and the satellite-link latency statistics are taken from "
        "published Starlink measurement studies rather than from the link budget of this "
        "model. Held out of the training corpus.",
        load_multiplier=1.75,
        hazard_multiplier={"LEO": 3.4, "UAV": 2.4, "5G": 2.0, "6G": 1.9, "4G": 2.0},
        shock_prob=0.0012,
        shock_targets=("LEO", "5G"),
        shock_magnitude=0.30,
        destroyed_nodes=("bs5g_3", "bs4g_2", "bs6g_2"),
        # Michel et al. (IMC 2022) report a globally coordinated reconfiguration of the
        # dish-satellite-gateway path every 15 s, at the 12th, 27th, 42nd and 57th second
        # of every minute, with throughput dropping at the interval boundaries.
        handover_period=15,
        handover_phase=12,
        handover_duration=2,
        handover_fail_prob=0.040,
        # Median RTT ~50 ms with an interquartile range of roughly 40-60 ms to European
        # anchors; the one-way base latency and jitter below reproduce that spread.
        latency_override=(("LEO", 50.0, 10.0),),
    ),
}

SCENARIO_NAMES: List[str] = list(SCENARIOS.keys())
# Scenarios used to build the training corpus. S5 and S6 are deliberately excluded
# so that the predictor and the control policy are evaluated out of distribution. S7 is
# additionally anchored to published measurements and is likewise never trained on.
TRAIN_SCENARIOS: List[str] = ["S1_normal", "S2_bs_overload", "S3_leo_degradation",
                              "S4_disaster_collapse"]
HELDOUT_SCENARIOS: List[str] = ["S5_leo_handover", "S6_common_cause", "S7_external_ntn"]

# Region membership used by the common-cause model. Elements that share a region
# share power and backhaul infrastructure; susceptibility scales the shared shock.
NODE_REGION: Dict[str, str] = {
    "term_1": "west", "term_2": "west", "bs6g_1": "west", "bs5g_1": "west",
    "bs5g_2": "west", "bs4g_1": "west", "uav_1": "west", "edge_1": "west",
    "term_3": "east", "term_4": "east", "bs6g_2": "east", "bs5g_3": "east",
    "bs4g_2": "east", "uav_2": "east", "edge_2": "east",
    "hq_1": "core", "leo_1": "space", "leo_2": "space", "leo_3": "space",
}
REGION_SUSCEPTIBILITY: Dict[str, float] = {
    "4G": 1.15, "5G": 1.00, "6G": 0.95, "EDGE": 0.80, "UAV": 1.25,
    "MOBILE": 0.45, "LEO": 0.10, "HQ": 0.25,
}

# --- S5 observation-level handover model (Section 4.4) ----------------------
# Reported LEO systems reconfigure user-terminal association on a fixed cadence
# and exhibit transient latency inflation and loss bursts around those events.
HANDOVER_LATENCY_GAIN = (1.8, 3.6)
HANDOVER_LOSS_ADD = (0.02, 0.12)
HANDOVER_JITTER_GAIN = (2.0, 3.4)
HANDOVER_FAIL_HEALTH_DROP = (0.18, 0.40)

# ----------------------------------------------------------------------------
# Observation-model coefficients (Section 4.3)
# ----------------------------------------------------------------------------

ALPHA_LAT_HEALTH = 2.60     # a1: latency inflation from health loss
ALPHA_LAT_LOAD = 0.90       # a2: queueing term weight
BETA_LOSS_HEALTH = 0.25     # b : loss growth from health loss
BETA_LOSS_LOAD = 0.55       # c : loss growth from congestion
GAMMA_SIGNAL = 38.0         # dBm lost at h -> 0
RHO_CAP = 0.945             # utilisation clamp for the queueing term
QUEUE_TERM_CAP = 8.0        # saturation of the M/M/1 delay term (finite buffers)
OBS_NOISE = 0.055           # multiplicative log-normal observation noise sigma

# ----------------------------------------------------------------------------
# Orchestration
# ----------------------------------------------------------------------------

RISK_THRESHOLD = 0.60       # tau: proactive re-routing threshold
RISK_EXCLUDE = 0.85         # tau_hard: node excluded from candidate paths
W_LATENCY = 1.0             # routing cost weights (Eq. 12)
W_LOSS = 220.0
W_RISK = 140.0
W_LOAD = 45.0
HYSTERESIS_STEPS = 6        # minimum steps between two re-routes of one flow

# Control-plane timing (Section 6.4). A break-before-make restoration after an
# observed failure costs detection + path computation + session re-establishment.
# A pre-emptive migration is executed make-before-make, i.e. the replacement path
# is installed while the old one still carries traffic.
DETECTION_DELAY = 3                    # slots of KPI aggregation before an alarm
REACTIVE_RECONFIG_STEPS = (4, 12)       # uniform restoration delay, break-before-make
PROACTIVE_RECONFIG_STEPS = (0, 1)      # make-before-break switch-over cost
