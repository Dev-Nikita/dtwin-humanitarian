"""Manifest test: the 98-element topology must match its description in the manuscript."""

from __future__ import annotations

from collections import Counter

import networkx as nx

from python.simulator.large_topology import LARGE_ARRIVAL_FACTOR, LARGE_TERMINALS, build_large


def test_large_topology_manifest():
    g = build_large()
    assert g.number_of_nodes() == 98
    assert g.number_of_edges() == 233
    counts = Counter(nx.get_node_attributes(g, "node_type").values())
    assert counts == {"LEO": 24, "MOBILE": 24, "5G": 16, "6G": 8, "4G": 8,
                      "UAV": 8, "EDGE": 8, "HQ": 2}
    assert len(LARGE_TERMINALS) == 24
    assert LARGE_ARRIVAL_FACTOR == 6.0
    assert nx.edge_connectivity(g) == 2
    assert nx.diameter(g) == 5
