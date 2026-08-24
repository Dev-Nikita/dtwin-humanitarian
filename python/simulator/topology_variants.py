"""
Topology variants for the redundancy-sensitivity study.

The resilience benefit of anticipatory recovery cannot be independent of how much
alternative capacity the system has: with no diversity there is nowhere to migrate
a flow, and with overwhelming diversity individual failures barely matter. This
module derives five topology classes from the reference graph by adding or
removing alternative routes while keeping the element catalogue, the traffic model
and the degradation model unchanged, so that the only variable is structural
redundancy.

    sparse            minimal connectivity, one route per gateway where possible
    lean              reference minus the aerial and legacy alternatives
    reference         the topology used in the main experiments
    redundant         reference plus additional cross-links
    satellite_heavy   reference plus a full non-terrestrial access mesh
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import networkx as nx

from .network_topology import LINK_CATALOGUE, NODE_CATALOGUE, build_topology

Link = Tuple[str, str, float, float]

# Links removed to create the sparser classes (kept minimal so the graph stays
# connected and every gateway retains a path to the core).
_SPARSE_REMOVE = {
    ("term_1", "bs5g_1"), ("term_1", "bs4g_1"), ("term_1", "leo_1"),
    ("term_2", "uav_1"), ("term_2", "bs5g_2"),
    ("term_3", "bs4g_2"), ("term_3", "uav_2"),
    ("term_4", "bs5g_3"),
    ("edge_1", "edge_2"), ("leo_1", "leo_2"), ("leo_2", "leo_3"),
    ("uav_1", "leo_1"), ("uav_2", "leo_3"),
}
_LEAN_REMOVE = {
    ("term_1", "bs4g_1"), ("term_3", "bs4g_2"),
    ("term_2", "uav_1"), ("term_3", "uav_2"),
    ("uav_1", "leo_1"), ("uav_2", "leo_3"),
}
# Extra links added to create the richer classes.
_REDUNDANT_ADD: List[Link] = [
    ("term_1", "bs5g_2", 4.5, 240.0),
    ("term_2", "bs6g_1", 3.5, 460.0),
    ("term_3", "uav_1", 7.5, 120.0),
    ("term_4", "uav_2", 5.0, 120.0),
    ("bs5g_1", "edge_2", 8.0, 520.0),
    ("bs5g_3", "edge_1", 8.5, 520.0),
    ("bs6g_1", "edge_2", 6.0, 1200.0),
    ("uav_1", "uav_2", 9.0, 120.0),
]
_SATELLITE_ADD: List[Link] = [
    ("term_1", "leo_2", 25.0, 90.0),
    ("term_2", "leo_1", 25.0, 90.0),
    ("term_3", "leo_3", 26.0, 90.0),
    ("term_4", "leo_2", 26.0, 90.0),
    ("leo_1", "leo_3", 24.0, 300.0),
    ("edge_1", "leo_1", 26.0, 120.0),
    ("edge_2", "leo_3", 26.0, 120.0),
]

TOPOLOGY_CLASSES: List[str] = [
    "sparse", "lean", "reference", "redundant", "satellite_heavy",
]


def _norm(u: str, v: str) -> Tuple[str, str]:
    return (u, v) if u <= v else (v, u)


def variant_links(name: str) -> List[Link]:
    """Return the link catalogue of a topology class."""
    links = list(LINK_CATALOGUE)
    if name == "sparse":
        drop = {_norm(*e) for e in _SPARSE_REMOVE}
        links = [l for l in links if _norm(l[0], l[1]) not in drop]
    elif name == "lean":
        drop = {_norm(*e) for e in _LEAN_REMOVE}
        links = [l for l in links if _norm(l[0], l[1]) not in drop]
    elif name == "redundant":
        links = links + _REDUNDANT_ADD
    elif name == "satellite_heavy":
        links = links + _SATELLITE_ADD
    elif name != "reference":
        raise ValueError(f"unknown topology class {name}")
    return links


def build_variant(name: str) -> nx.Graph:
    """Build the graph of a topology class with the reference element catalogue."""
    g = build_topology()
    g.remove_edges_from(list(g.edges()))
    for u, v, latency, capacity in variant_links(name):
        g.add_edge(u, v, prop_latency_ms=latency, capacity_mbps=capacity)
    isolated = [n for n in g.nodes if g.degree(n) == 0]
    g.remove_nodes_from(isolated)
    return g


def redundancy_descriptors(g: nx.Graph) -> Dict[str, float]:
    """Structural redundancy measures used as the independent variable."""
    from .network_topology import SINKS, TERMINALS

    terminals = [t for t in TERMINALS if t in g]
    sinks = [s for s in SINKS if s in g]
    diversity = []
    for t in terminals:
        for s in sinks:
            if t == s or not nx.has_path(g, t, s):
                continue
            diversity.append(len(list(nx.node_disjoint_paths(g, t, s))))
    degrees = [d for _, d in g.degree()]
    return {
        "nodes": g.number_of_nodes(),
        "links": g.number_of_edges(),
        "mean_degree": sum(degrees) / len(degrees),
        "edge_connectivity": nx.edge_connectivity(g),
        "algebraic_connectivity": float(nx.algebraic_connectivity(g)),
        "mean_route_diversity": float(sum(diversity) / len(diversity)) if diversity else 0.0,
        "min_route_diversity": float(min(diversity)) if diversity else 0.0,
    }


if __name__ == "__main__":  # pragma: no cover
    for cls in TOPOLOGY_CLASSES:
        graph = build_variant(cls)
        desc = redundancy_descriptors(graph)
        print(f"{cls:16s} " + "  ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
                                        for k, v in desc.items()))
