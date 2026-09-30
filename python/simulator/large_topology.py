"""
Large hybrid topology used for the scale experiment (Section 7.7, Reviewer 1.2).

Eight terrestrial regions, each with one 6G-class cell, two 5G cells, one LTE
cell, one UAV relay, one edge node and three user gateways; two core/gateway
sites; and a LEO segment of three orbital planes with eight satellites each,
connected by intra-plane ring ISLs and inter-plane ISLs between neighbouring
planes. Every user gateway can reach at least one satellite directly, every UAV
has a satellite uplink, and each core site acts as feeder gateway for four
satellites per plane.

    98 elements, 8 regions, 24 LEO satellites in 3 planes.

The element classes, degradation model, observation map and traffic classes are
exactly those of the reference study; only the graph and the corresponding
region, destruction and endpoint maps change.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import networkx as nx

from .config import NODE_PROFILES
from .network_topology import _layer_of

N_REGIONS = 8
N_PLANES = 3
SATS_PER_PLANE = 8


def _catalogue() -> Tuple[Dict[str, str], List[Tuple[str, str, float, float]], Dict[str, str]]:
    cat: Dict[str, str] = {"hq_1": "HQ", "hq_2": "HQ"}
    region: Dict[str, str] = {"hq_1": "core", "hq_2": "core"}
    links: List[Tuple[str, str, float, float]] = [("hq_1", "hq_2", 8.0, 2000.0)]

    for p in range(N_PLANES):
        for k in range(SATS_PER_PLANE):
            sid = f"leo_p{p}_{k}"
            cat[sid] = "LEO"
            region[sid] = "space"
    for p in range(N_PLANES):
        for k in range(SATS_PER_PLANE):
            a = f"leo_p{p}_{k}"
            links.append((a, f"leo_p{p}_{(k + 1) % SATS_PER_PLANE}", 12.0, 300.0))
            if p + 1 < N_PLANES:
                links.append((a, f"leo_p{p + 1}_{k}", 16.0, 240.0))
        # feeder links: each core site sees four satellites of every plane
        for k in range(SATS_PER_PLANE):
            gw = "hq_1" if k < SATS_PER_PLANE // 2 else "hq_2"
            links.append((f"leo_p{p}_{k}", gw, 22.0 + 0.5 * k, 260.0))

    for r in range(1, N_REGIONS + 1):
        half = "west" if r <= N_REGIONS // 2 else "east"
        ids = {
            "6g": f"r{r}_bs6g", "5ga": f"r{r}_bs5g_a", "5gb": f"r{r}_bs5g_b",
            "4g": f"r{r}_bs4g", "uav": f"r{r}_uav", "edge": f"r{r}_edge",
        }
        types = {"6g": "6G", "5ga": "5G", "5gb": "5G", "4g": "4G", "uav": "UAV", "edge": "EDGE"}
        for key, nid in ids.items():
            cat[nid] = types[key]
            region[nid] = half
        terms = [f"r{r}_term_{i}" for i in range(1, 4)]
        for t in terms:
            cat[t] = "MOBILE"
            region[t] = half
        plane = (r - 1) % N_PLANES
        sat = lambda j: f"leo_p{plane}_{(r + j) % SATS_PER_PLANE}"  # noqa: E731
        # terminals -> access (each gateway has two or three terrestrial options)
        links += [
            (terms[0], ids["6g"], 2.0, 480.0), (terms[0], ids["5ga"], 3.0, 260.0),
            (terms[0], ids["4g"], 5.0, 70.0),
            (terms[1], ids["5ga"], 3.0, 260.0), (terms[1], ids["5gb"], 3.5, 240.0),
            (terms[1], ids["uav"], 6.0, 120.0),
            (terms[2], ids["5gb"], 4.0, 220.0), (terms[2], ids["4g"], 5.5, 70.0),
            (terms[2], ids["uav"], 6.5, 120.0),
        ]
        # terminals -> LEO (direct-to-device / VSAT)
        links += [(t, sat(i), 24.0 + i, 90.0) for i, t in enumerate(terms)]
        # UAV relay
        links += [(ids["uav"], ids["5gb"], 4.0, 150.0), (ids["uav"], ids["edge"], 8.0, 150.0),
                  (ids["uav"], sat(1), 20.0, 80.0)]
        # access -> edge
        links += [(ids["6g"], ids["edge"], 3.0, 1200.0), (ids["5ga"], ids["edge"], 6.0, 520.0),
                  (ids["5gb"], ids["edge"], 6.5, 520.0), (ids["4g"], ids["edge"], 12.0, 110.0)]
        # edge -> core and edge ring
        links.append((ids["edge"], "hq_1" if r <= N_REGIONS // 2 else "hq_2", 9.0 + 0.3 * r, 1400.0))
        nxt = r % N_REGIONS + 1
        links.append((ids["edge"], f"r{nxt}_edge", 5.0, 900.0))
    return cat, links, region


LARGE_CATALOGUE, LARGE_LINKS, LARGE_REGION = _catalogue()
LARGE_TERMINALS: List[str] = [n for n, t in LARGE_CATALOGUE.items() if t == "MOBILE"]
LARGE_SINKS: List[str] = ["hq_1", "hq_2"] + [f"r{r}_edge" for r in range(1, N_REGIONS + 1)]
# Traffic scales with the number of user gateways (24 against 4 in the reference).
LARGE_ARRIVAL_FACTOR = len(LARGE_TERMINALS) / 4.0


def large_destroyed(scenario: str) -> Tuple[str, ...]:
    """Destroyed elements, mirroring the proportions of the reference scenarios:
    S3/S5/S7 remove three of the four cell types in the eastern half (3 of 7
    cells per affected region in the reference); S4 additionally removes the LTE
    and one 5G cell in the western half."""
    east = range(N_REGIONS // 2 + 1, N_REGIONS + 1)
    west = range(1, N_REGIONS // 2 + 1)
    base = tuple(n for r in east for n in (f"r{r}_bs6g", f"r{r}_bs5g_b", f"r{r}_bs4g"))
    if scenario in ("S3_leo_degradation", "S5_leo_handover", "S7_external_ntn"):
        return base
    if scenario == "S4_disaster_collapse":
        return base + tuple(n for r in west for n in (f"r{r}_bs4g", f"r{r}_bs5g_a"))
    return ()


def build_large() -> nx.Graph:
    g = nx.Graph()
    for nid, ntype in LARGE_CATALOGUE.items():
        prof = NODE_PROFILES[ntype]
        g.add_node(nid, node_type=ntype, capacity_mbps=prof.capacity_mbps,
                   wireless=prof.wireless, layer=_layer_of(ntype))
    for u, v, lat, cap in LARGE_LINKS:
        g.add_edge(u, v, prop_latency_ms=lat, capacity_mbps=cap)
    return g


if __name__ == "__main__":  # pragma: no cover
    g = build_large()
    print(g.number_of_nodes(), "nodes", g.number_of_edges(), "links",
          "edge connectivity", nx.edge_connectivity(g), "diameter", nx.diameter(g))
