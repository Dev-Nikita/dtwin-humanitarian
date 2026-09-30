"""
Hybrid terrestrial-satellite topology for humanitarian operations.

The reference topology models a humanitarian area of responsibility (AoR)
containing:

    * 4 mobile terminals (field teams, clinics, camp gateways)
    * 2 LTE (4G) macro cells kept as legacy fallback
    * 3 5G cells providing the main terrestrial capacity
    * 2 6G cells covering the operational core with URLLC-class latency
    * 2 UAV relays acting as opportunistic aerial backhaul
    * 3 LEO satellites (non-terrestrial segment) with a gateway to HQ
    * 2 edge nodes hosting humanitarian applications
    * 1 headquarters / command centre

The graph is undirected; each edge carries a static propagation latency and a
nominal capacity. Dynamic state (health, load, loss) lives in the digital twin,
not in the graph.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import networkx as nx

from .config import NODE_PROFILES

# node_id -> node_type
NODE_CATALOGUE: Dict[str, str] = {
    "hq_1": "HQ",
    "edge_1": "EDGE",
    "edge_2": "EDGE",
    "leo_1": "LEO",
    "leo_2": "LEO",
    "leo_3": "LEO",
    "bs6g_1": "6G",
    "bs6g_2": "6G",
    "bs5g_1": "5G",
    "bs5g_2": "5G",
    "bs5g_3": "5G",
    "bs4g_1": "4G",
    "bs4g_2": "4G",
    "uav_1": "UAV",
    "uav_2": "UAV",
    "term_1": "MOBILE",
    "term_2": "MOBILE",
    "term_3": "MOBILE",
    "term_4": "MOBILE",
}

# (u, v, propagation latency ms, capacity Mbit/s)
LINK_CATALOGUE: List[Tuple[str, str, float, float]] = [
    # terminals -> terrestrial access
    ("term_1", "bs6g_1", 2.0, 480.0),
    ("term_1", "bs5g_1", 3.0, 260.0),
    ("term_1", "bs4g_1", 5.0, 70.0),
    ("term_2", "bs5g_1", 3.0, 260.0),
    ("term_2", "bs5g_2", 3.5, 240.0),
    ("term_2", "uav_1", 6.0, 120.0),
    ("term_3", "bs5g_3", 4.0, 220.0),
    ("term_3", "bs4g_2", 5.5, 70.0),
    ("term_3", "uav_2", 6.5, 120.0),
    ("term_4", "bs6g_2", 2.5, 460.0),
    ("term_4", "bs5g_3", 4.0, 220.0),
    # terminals -> non-terrestrial access (direct-to-device / VSAT terminal)
    ("term_1", "leo_1", 24.0, 90.0),
    ("term_2", "leo_2", 26.0, 90.0),
    ("term_3", "leo_2", 25.0, 90.0),
    ("term_4", "leo_3", 27.0, 90.0),
    # UAV relays -> terrestrial and NTN
    ("uav_1", "bs5g_2", 4.0, 150.0),
    ("uav_1", "edge_1", 8.0, 150.0),
    ("uav_1", "leo_1", 20.0, 80.0),
    ("uav_2", "bs5g_3", 4.5, 150.0),
    ("uav_2", "edge_2", 9.0, 150.0),
    ("uav_2", "leo_3", 21.0, 80.0),
    # access -> edge
    ("bs6g_1", "edge_1", 3.0, 1200.0),
    ("bs6g_2", "edge_2", 3.0, 1200.0),
    ("bs5g_1", "edge_1", 6.0, 520.0),
    ("bs5g_2", "edge_1", 6.5, 520.0),
    ("bs5g_3", "edge_2", 6.0, 520.0),
    ("bs4g_1", "edge_1", 12.0, 110.0),
    ("bs4g_2", "edge_2", 12.0, 110.0),
    # edge interconnect and backhaul to HQ
    ("edge_1", "edge_2", 5.0, 900.0),
    ("edge_1", "hq_1", 9.0, 1400.0),
    ("edge_2", "hq_1", 10.0, 1400.0),
    # NTN feeder links to the HQ gateway
    ("leo_1", "hq_1", 22.0, 260.0),
    ("leo_2", "hq_1", 23.0, 260.0),
    ("leo_3", "hq_1", 24.0, 260.0),
    # inter-satellite links (optical ISL)
    ("leo_1", "leo_2", 12.0, 300.0),
    ("leo_2", "leo_3", 12.0, 300.0),
]

TERMINALS: List[str] = [n for n, t in NODE_CATALOGUE.items() if t == "MOBILE"]
SINKS: List[str] = ["hq_1", "edge_1", "edge_2"]


def build_topology() -> nx.Graph:
    """Return the reference hybrid topology as an undirected NetworkX graph."""
    g = nx.Graph()
    for node_id, node_type in NODE_CATALOGUE.items():
        profile = NODE_PROFILES[node_type]
        g.add_node(
            node_id,
            node_type=node_type,
            capacity_mbps=profile.capacity_mbps,
            wireless=profile.wireless,
            layer=_layer_of(node_type),
        )
    for u, v, latency, capacity in LINK_CATALOGUE:
        g.add_edge(u, v, prop_latency_ms=latency, capacity_mbps=capacity)
    return g


def _layer_of(node_type: str) -> str:
    if node_type == "MOBILE":
        return "user"
    if node_type in ("4G", "5G", "6G"):
        return "terrestrial_access"
    if node_type in ("UAV",):
        return "aerial"
    if node_type == "LEO":
        return "non_terrestrial"
    if node_type == "EDGE":
        return "edge"
    return "core"


def topology_summary(g: nx.Graph) -> Dict[str, float]:
    """Structural descriptors reported in Table 1 of the paper."""
    degrees = [d for _, d in g.degree()]
    return {
        "nodes": g.number_of_nodes(),
        "links": g.number_of_edges(),
        "mean_degree": sum(degrees) / len(degrees),
        "diameter": nx.diameter(g),
        "algebraic_connectivity": float(nx.algebraic_connectivity(g)),
        "avg_path_latency_ms": float(nx.average_shortest_path_length(g, weight="prop_latency_ms")),
        "edge_connectivity": nx.edge_connectivity(g),
    }


if __name__ == "__main__":  # pragma: no cover
    graph = build_topology()
    for key, value in topology_summary(graph).items():
        print(f"{key:24s} {value}")
