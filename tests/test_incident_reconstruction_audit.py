"""Independent bounded-search oracle and reconstruction invariants."""

import random

import networkx as nx
import pandas as pd
import pytest

from pidsmaker.incidents.builder import build_incidents
from pidsmaker.incidents.reachability import find_temporal_path


@pytest.mark.parametrize("seed", range(25))
def test_minimum_cost_temporal_search_matches_exhaustive_oracle(seed):
    rng = random.Random(seed)
    graph = nx.MultiDiGraph()
    graph.add_nodes_from(range(4))
    for key in range(9):
        graph.add_edge(rng.randrange(4), rng.randrange(4), key=key,
                       time=rng.randrange(6), weight=rng.randrange(4))
    expected = []

    def enumerate_walks(node, arrival, remaining, cost):
        if node == 3:
            expected.append(cost)
        if not remaining:
            return
        for _, dst, _, data in graph.out_edges(node, keys=True, data=True):
            if arrival <= data["time"] <= 5:
                enumerate_walks(dst, data["time"], remaining-1, cost+data["weight"])

    enumerate_walks(0, 0, 4, 0)
    path = find_temporal_path(graph, 0, 3, 0, 5, 4,
                              edge_cost=lambda u, v, k, d: d["weight"])
    if not expected:
        assert path is None
    else:
        assert path is not None
        assert sum(graph[u][v][k]["weight"] for u, v, k in path) == min(expected)
        timestamps = [graph[u][v][k]["time"] for u, v, k in path]
        assert timestamps == sorted(timestamps)
        assert len(path) <= 4


def test_overlapping_independent_stories_and_parallel_events_keep_exact_evidence():
    graph = nx.MultiDiGraph()
    rows = []
    # Two simultaneous but disconnected chains, plus an isolated seed.
    for start in (0, 10):
        for step in range(5):
            score = 10. if step in (0, 4) else .1
            graph.add_edge(start+step, start+step+1, key=step, time=step, edge_type=1, payload=f"event-{start}-{step}")
            rows.append(dict(srcnode=start+step, dstnode=start+step+1, time=step, edge_type=1, key=step, loss=score))
    graph.add_edge(0, 1, key=99, time=0, edge_type=1, payload="parallel")
    rows.append(dict(srcnode=0, dstnode=1, time=0, edge_type=1, key=99, loss=11.))
    graph.add_edge(100, 101, key=7, time=2, edge_type=1, payload="isolated")
    rows.append(dict(srcnode=100, dstnode=101, time=2, edge_type=1, key=7, loss=12.))
    original = graph.copy()
    incidents = build_incidents(graph, pd.DataFrame(rows), 5.)
    assert len(incidents) == 3
    seeds = [(e.srcnode, e.dstnode, e.key) for i in incidents for e in i.seed_edges]
    assert len(seeds) == len(set(seeds)) == 6
    assert sum(len(i.connector_edges) for i in incidents) == 6
    assert {frozenset(i.graph.nodes) for i in incidents} == {
        frozenset(range(6)), frozenset(range(10, 16)), frozenset((100, 101))}
    for incident in incidents:
        assert nx.is_weakly_connected(incident.graph)
        for src, dst, key, data in incident.graph.edges(keys=True, data=True):
            assert graph.has_edge(src, dst, key)
            assert data["time"] == original[src][dst][key]["time"]
            assert data["payload"] == original[src][dst][key]["payload"]
    assert nx.utils.graphs_equal(original, graph)
