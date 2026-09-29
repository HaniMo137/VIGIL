import random
from copy import deepcopy

import networkx as nx
import pandas as pd
import pytest

from pidsmaker.incidents import (
    IncidentBuilderConfig, build_incidents, calculate_seed_threshold, select_seed_edges,
)
from pidsmaker.incidents.builder import prepare_provenance_graph, prune_incident_graph
from pidsmaker.incidents.clustering import seeds_are_related
from pidsmaker.incidents.models import EdgeRef, Incident
from pidsmaker.incidents.reachability import PathSearchLimitExceeded, find_temporal_path


def region(events):
    graph = nx.MultiDiGraph(source="synthetic")
    rows = []
    for src, dst, timestamp, edge_type, score in events:
        key = graph.add_edge(
            src, dst, time=timestamp, edge_type=edge_type,
            event_uuid=f"event-{len(rows)}", evidence={"original": True},
        )
        graph.nodes[src]["node_type"] = "subject"
        graph.nodes[dst]["node_type"] = "subject"
        rows.append(dict(srcnode=src, dstnode=dst, time=timestamp,
                         edge_type=edge_type, loss=score, key=key))
    scores = pd.DataFrame(rows, columns=["srcnode", "dstnode", "time", "edge_type", "loss", "key"])
    return graph, scores


def bridge():
    return region([(1, 2, 1, 1, 10), (2, 3, 2, 2, 0.1),
                   (3, 4, 3, 2, 0.2), (4, 5, 4, 1, 11)])


def test_builder_recovers_bridge_and_preserves_evidence_without_mutation():
    graph, scores = bridge()
    original_graph, original_scores = deepcopy(graph), scores.copy(deep=True)
    incidents = build_incidents(graph, scores, threshold=5)

    assert len(incidents) == 1
    incident = incidents[0]
    assert {(e.srcnode, e.dstnode) for e in incident.seed_edges} == {(1, 2), (4, 5)}
    assert {(e.srcnode, e.dstnode) for e in incident.connector_edges} == {(2, 3), (3, 4)}
    assert len(incident.edges) == incident.graph.number_of_edges() == 4
    assert incident.context_edges == ()
    assert (incident.start_time, incident.end_time) == (1, 4)
    assert not nx.is_frozen(incident.graph)
    for edge in incident.edges:
        attrs = incident.graph[edge.srcnode][edge.dstnode][edge.key]
        assert attrs["time"] == edge.time
        assert attrs["score"] == edge.score
        assert attrs["event_uuid"] == graph[edge.srcnode][edge.dstnode][edge.key]["event_uuid"]
    incident.graph[2][3][0]["evidence"]["original"] = False
    assert nx.utils.graphs_equal(graph, original_graph)
    pd.testing.assert_frame_equal(scores, original_scores)


def test_builder_keeps_independent_seed_as_its_own_incident():
    graph, scores = region([(1, 2, 1, 1, 10), (2, 3, 2, 2, 0.1),
                            (3, 4, 3, 2, 0.2), (4, 5, 4, 1, 11), (8, 9, 3, 1, 12)])
    incidents = build_incidents(graph, scores, 5)
    assert sorted(len(incident.seed_edges) for incident in incidents) == [1, 2]
    assert sum(len(incident.seed_edges) for incident in incidents) == 3
    singleton = next(incident for incident in incidents if len(incident.seed_edges) == 1)
    assert singleton.graph.number_of_edges() == 1
    assert singleton.connector_edges == ()


@pytest.mark.parametrize("reason", ["backward", "time_limit", "hop_limit"])
def test_builder_separates_seeds_when_bridge_is_invalid(reason):
    graph, scores = bridge()
    settings = {}
    if reason == "backward":
        graph[2][3][0]["time"] = 3
        graph[3][4][0]["time"] = 2
        scores.loc[scores.srcnode == 2, "time"] = 3
        scores.loc[scores.srcnode == 3, "time"] = 2
    elif reason == "time_limit":
        settings["max_time_gap_ns"] = 2
    else:
        settings["max_connector_hops"] = 1
    incidents = build_incidents(graph, scores, 5, IncidentBuilderConfig(**settings))
    assert len(incidents) == 2
    assert all(len(incident.seed_edges) == 1 and not incident.connector_edges for incident in incidents)


def test_builder_retains_all_seeds_in_a_transitive_cluster():
    graph, scores = region([(1, 2, 1, 1, 10), (2, 3, 2, 2, 0.1),
                            (3, 4, 3, 2, 0.1), (4, 5, 4, 1, 10),
                            (5, 6, 5, 2, 0.1), (6, 8, 6, 2, 0.1), (8, 9, 7, 1, 10)])
    incidents = build_incidents(graph, scores, 5, IncidentBuilderConfig(max_time_gap_ns=3))
    assert len(incidents) == 1
    assert len(incidents[0].seed_edges) == 3
    assert len(incidents[0].connector_edges) == 4
    assert len(set(incidents[0].edges)) == 7


def test_builder_prefers_a_lower_cost_bridge_not_necessarily_fewer_hops():
    graph, scores = region([(1, 2, 1, 1, 10), (4, 5, 4, 1, 10),
                            (2, 3, 2, 2, 0.1), (3, 4, 3, 2, 0.1),
                            (2, 6, 2, 2, 4.9), (6, 7, 2, 2, 4.9), (7, 4, 3, 2, 4.9)])
    incident, = build_incidents(graph, scores, 5)
    assert {(e.srcnode, e.dstnode) for e in incident.connector_edges} == {(2, 6), (6, 7), (7, 4)}
    limited, = build_incidents(graph, scores, 5, IncidentBuilderConfig(max_connector_hops=2))
    assert {(e.srcnode, e.dstnode) for e in limited.connector_edges} == {(2, 3), (3, 4)}


def test_unscored_original_events_can_be_connectors():
    graph, scores = bridge()
    incident, = build_incidents(graph, scores[scores.loss >= 5], 5)
    assert len(incident.connector_edges) == 2
    assert all(edge.score is None for edge in incident.connector_edges)


def test_builder_no_seeds_returns_empty_tuple():
    graph, scores = bridge()
    assert build_incidents(graph, scores, 100) == ()
    assert build_incidents(nx.MultiDiGraph(), scores.iloc[:0], 5) == ()


@pytest.mark.parametrize("threshold", [-1, float("nan"), float("inf")])
def test_builder_rejects_invalid_thresholds(threshold):
    graph, scores = bridge()
    with pytest.raises(ValueError, match="Threshold"):
        build_incidents(graph, scores, threshold)


def test_seed_at_threshold_is_included():
    graph, scores = region([(1, 2, 1, 1, 5)])
    incident, = build_incidents(graph, scores, 5)
    assert incident.seed_edges[0].score == 5


def test_adapter_accepts_original_label_schema_and_string_node_ids():
    graph = nx.MultiDiGraph()
    graph.add_edge("1", "2", key="original", time=10, label="EVENT_WRITE", event_uuid="abc")
    scores = pd.DataFrame([dict(srcnode=1, dstnode=2, time=10, edge_type=9, loss=5)])
    incident, = build_incidents(graph, scores, 5, relation_to_id={"EVENT_WRITE": 9})
    assert incident.seed_edges[0].key == "original"
    assert incident.graph[1][2]["original"]["label"] == "EVENT_WRITE"
    assert "edge_type" not in graph["1"]["2"]["original"]


def test_adapter_requires_relation_mapping_for_original_graphs():
    graph = nx.MultiDiGraph()
    graph.add_edge(1, 2, time=10, label="EVENT_WRITE")
    scores = pd.DataFrame([dict(srcnode=1, dstnode=2, time=10, edge_type=9, loss=5)])
    with pytest.raises(ValueError, match="Missing relation mapping"):
        build_incidents(graph, scores, 5)


@pytest.mark.parametrize("identity_column", ["key", "event_uuid"])
def test_adapter_disambiguates_identical_parallel_events(identity_column):
    graph = nx.MultiDiGraph()
    graph.add_edge(1, 2, key="a", time=10, edge_type=1, event_uuid="uuid-a")
    graph.add_edge(1, 2, key="b", time=10, edge_type=1, event_uuid="uuid-b")
    row = dict(srcnode=1, dstnode=2, time=10, edge_type=1, loss=5)
    row[identity_column] = "b" if identity_column == "key" else "uuid-b"
    incident, = build_incidents(graph, pd.DataFrame([row]), 5)
    assert incident.seed_edges[0].key == "b"
    assert incident.graph.number_of_edges() == 1


def test_adapter_rejects_ambiguous_parallel_events_without_identity():
    graph = nx.MultiDiGraph()
    graph.add_edge(1, 2, time=10, edge_type=1)
    graph.add_edge(1, 2, time=10, edge_type=1)
    scores = pd.DataFrame([dict(srcnode=1, dstnode=2, time=10, edge_type=1, loss=5)])
    with pytest.raises(ValueError, match="Ambiguous scored event"):
        build_incidents(graph, scores, 5)


def test_adapter_does_not_attach_scores_to_wrong_timestamps():
    graph, scores = bridge()
    scores.loc[0, "time"] = 100
    with pytest.raises(ValueError, match="not found"):
        build_incidents(graph, scores, 5)


def test_adapter_rejects_duplicate_score_rows():
    graph, scores = bridge()
    with pytest.raises(ValueError, match="Duplicate score"):
        build_incidents(graph, pd.concat([scores, scores.iloc[:1]]), 5)


def test_adapter_rejects_colliding_node_ids():
    graph, scores = bridge()
    graph.add_node("1")
    with pytest.raises(ValueError, match="collide"):
        prepare_provenance_graph(graph, scores)


@pytest.mark.parametrize("bad_time", [1.5, float(1_700_000_000_000_000_000)])
def test_adapter_rejects_noninteger_or_imprecise_timestamps(bad_time):
    graph, scores = bridge()
    graph[1][2][0]["time"] = bad_time
    with pytest.raises(ValueError, match="integer|imprecise float"):
        prepare_provenance_graph(graph, scores)


def test_metadata_boundaries_are_optional_and_not_affinity_by_themselves():
    graph, scores = bridge()
    graph[1][2][0]["host"] = "host-a"
    graph[4][5][0]["host"] = "host-b"
    assert len(build_incidents(graph, scores, 5)) == 1
    assert len(build_incidents(graph, scores, 5, IncidentBuilderConfig(group_by=("host",)))) == 2
    graph[4][5][0]["host"] = "host-a"
    graph.remove_edge(2, 3, 0)
    scores = scores[scores.srcnode != 2]
    assert len(build_incidents(graph, scores, 5, IncidentBuilderConfig(group_by=("host",)))) == 2


def test_shared_ancestry_is_opt_in_and_preserves_ancestor_events():
    graph, scores = region([(1, 2, 2, 1, 10), (3, 4, 3, 1, 10),
                            (9, 1, 1, 10, 0), (9, 3, 1, 10, 0)])
    assert len(build_incidents(graph, scores, 5)) == 2
    config = IncidentBuilderConfig(ancestry_edge_types=(10,), max_ancestry_hops=1)
    incident, = build_incidents(graph, scores, 5, config)
    assert len(incident.seed_edges) == 2
    assert {(edge.srcnode, edge.dstnode) for edge in incident.connector_edges} == {(9, 1), (9, 3)}


def test_context_is_separate_bounded_and_relation_filtered():
    graph, scores = region([(1, 2, 1, 1, 10), (2, 3, 2, 2, 0.1),
                            (3, 4, 3, 2, 0.2), (4, 5, 4, 1, 11),
                            (10, 1, 1, 9, 0), (5, 6, 4, 9, 0), (3, 20, 2, 8, 0)])
    config = IncidentBuilderConfig(context_hops=1, max_context_edges=1, context_edge_types=(9,))
    incident, = build_incidents(graph, scores, 5, config)
    assert len(incident.context_edges) == 1
    assert len(incident.connector_edges) == 2
    assert incident.graph.number_of_edges() == 5
    assert incident.graph.graph["context_limit_reached"]
    assert all(edge.edge_type == 9 for edge in incident.context_edges)
    assert 20 not in incident.graph


def test_pruning_removes_unprotected_leaves_and_disconnected_cycles():
    graph = nx.MultiDiGraph()
    graph.add_edges_from([(1, 2), (2, 3), (2, 4), (4, 5), (10, 11), (11, 12), (12, 10)])
    result = prune_incident_graph(graph, terminal_nodes={1, 3}, required_edges=((1, 2, 0), (2, 3, 0)))
    assert set(result.nodes) == {1, 2, 3}
    assert graph.number_of_nodes() == 8


def test_equal_time_seed_relationship_checks_both_directions():
    first = EdgeRef(1, 2, time=10, edge_type=1, score=5)
    second = EdgeRef(3, 4, time=10, edge_type=1, score=5)
    graph = nx.MultiDiGraph()
    graph.add_nodes_from([1, 2, 3, 4])
    graph.add_edge(4, 1, time=10)
    assert seeds_are_related(first, second, 0, graph, max_hops=1)
    assert seeds_are_related(second, first, 0, graph, max_hops=1)


def test_seed_relationship_reorders_distinct_timestamps():
    graph, _ = bridge()
    first = EdgeRef(1, 2, time=1, edge_type=1, score=10)
    second = EdgeRef(4, 5, time=4, edge_type=1, score=10)
    assert seeds_are_related(first, second, 3, graph, max_hops=2)
    assert seeds_are_related(second, first, 3, graph, max_hops=2)


def test_search_budget_exhaustion_is_not_silently_no_path():
    graph, scores = bridge()
    with pytest.raises(PathSearchLimitExceeded, match="max_search_states"):
        build_incidents(graph, scores, 5, IncidentBuilderConfig(max_search_states=1))


def test_weighted_search_keeps_different_arrival_states():
    graph = nx.MultiDiGraph()
    graph.add_edge(1, 3, time=4, cost=0)
    graph.add_edge(1, 2, time=2, cost=1)
    graph.add_edge(2, 3, time=3, cost=1)
    graph.add_edge(3, 4, time=3, cost=1)
    path = find_temporal_path(graph, 1, 4, 1, 5, 3, edge_cost=lambda u, v, k, d: d["cost"])
    assert path == ((1, 2, 0), (2, 3, 0), (3, 4, 0))


@pytest.mark.parametrize("bad_cost", [-1, float("nan"), float("inf")])
def test_weighted_search_rejects_invalid_costs(bad_cost):
    graph = nx.MultiDiGraph()
    graph.add_edge(1, 2, time=1)
    with pytest.raises(ValueError, match="Edge costs"):
        find_temporal_path(graph, 1, 2, 0, 2, 1, edge_cost=lambda u, v, k, d: bad_cost)


@pytest.mark.parametrize("settings", [
    {"max_connector_hops": -1}, {"max_search_states": 0}, {"max_time_gap_ns": -1},
    {"context_hops": -1}, {"alpha": -1}, {"beta": float("nan")},
    {"relation_penalties": {1: -1}}, {"max_connector_hops": 1.5},
])
def test_builder_config_rejects_invalid_values(settings):
    with pytest.raises(ValueError):
        IncidentBuilderConfig(**settings)


def test_incident_model_rejects_wrong_key_or_timestamp():
    graph = nx.MultiDiGraph()
    graph.add_edge(1, 2, key="real", time=10, edge_type=1)
    with pytest.raises(ValueError, match="key"):
        Incident(graph, (EdgeRef(1, 2, 10, 1, 5, "wrong"),))
    with pytest.raises(ValueError, match="does not match"):
        Incident(graph, (EdgeRef(1, 2, 20, 1, 5, "real"),))


def test_seed_pair_budget_is_explicit_and_preserves_no_partial_output():
    graph, scores = region([(1, 2, 1, 1, 10), (3, 4, 1, 1, 10), (5, 6, 1, 1, 10)])
    with pytest.raises(PathSearchLimitExceeded, match="max_seed_pairs"):
        build_incidents(graph, scores, 5, IncidentBuilderConfig(max_seed_pairs=1))


def test_tuple_event_keys_remain_exact():
    graph = nx.MultiDiGraph()
    graph.add_edge(1, 2, key=("window", 0), time=1, edge_type=1)
    scores = pd.DataFrame([dict(srcnode=1, dstnode=2, key=("window", 0), time=1, edge_type=1, loss=5)])
    incident, = build_incidents(graph, scores, 5)
    assert incident.seed_edges[0].key == ("window", 0)
    assert incident.graph.has_edge(1, 2, ("window", 0))


@pytest.mark.parametrize("loss", [-1, float("nan"), float("inf"), "bad", None])
def test_invalid_losses_are_rejected_for_validation_and_selection(loss):
    graph, scores = bridge()
    scores["loss"] = scores["loss"].astype(object)
    scores.loc[0, "loss"] = loss
    with pytest.raises(ValueError, match="losses"):
        build_incidents(graph, scores, 5)
    with pytest.raises(ValueError, match="losses"):
        calculate_seed_threshold(pd.DataFrame({"loss": [loss]}))


def test_numeric_string_losses_are_compared_numerically():
    assert calculate_seed_threshold(pd.DataFrame({"loss": ["9", "10"]})) == 10
    _, scores = region([(1, 2, 1, 1, 10)])
    scores["loss"] = "10"
    assert select_seed_edges(scores, 9)[0].score == 10


def test_incident_rejects_same_event_in_two_roles():
    graph = nx.MultiDiGraph()
    graph.add_edge(1, 2, time=1, edge_type=1)
    seed = EdgeRef(1, 2, 1, 1, 5, 0)
    with pytest.raises(ValueError, match="multiple incident roles"):
        Incident(graph, (seed,), (seed,))


@pytest.mark.parametrize("case", range(20))
def test_weighted_temporal_search_matches_exhaustive_small_graphs(case):
    rng = random.Random(case)
    graph = nx.MultiDiGraph()
    graph.add_nodes_from(range(5))
    for _ in range(12):
        graph.add_edge(rng.randrange(5), rng.randrange(5), time=rng.randrange(5), cost=rng.randrange(3))
    target = rng.randrange(5)
    max_hops = case % 5

    def exhaustive(node, arrival, hops):
        if node == target:
            return 0
        if hops == max_hops:
            return float("inf")
        return min(
            (data["cost"] + exhaustive(dst, data["time"], hops + 1)
             for _, dst, _, data in graph.out_edges(node, keys=True, data=True)
             if arrival <= data["time"] <= 4),
            default=float("inf"),
        )

    expected_cost = exhaustive(0, 0, 0)
    path = find_temporal_path(graph, 0, target, 0, 4, max_hops,
                              edge_cost=lambda u, v, k, d: d["cost"])
    if expected_cost == float("inf"):
        assert path is None
    else:
        assert path is not None
        assert sum(graph[u][v][k]["cost"] for u, v, k in path) == expected_cost
        assert len(path) <= max_hops
        timestamps = [graph[u][v][k]["time"] for u, v, k in path]
        assert timestamps == sorted(timestamps)


def test_ancestry_search_keeps_distinct_time_states():
    graph, scores = region([
        (1, 2, 5, 1, 10), (3, 4, 5, 1, 10),
        (9, 1, 2, 10, 0), (9, 1, 4, 10, 0),
        (8, 9, 3, 10, 0), (8, 3, 4, 10, 0),
    ])
    config = IncidentBuilderConfig(ancestry_edge_types=(10,), max_ancestry_hops=2)
    incident, = build_incidents(graph, scores, 5, config)
    assert len(incident.seed_edges) == 2
    assert {(edge.srcnode, edge.dstnode, edge.time) for edge in incident.connector_edges} == {
        (8, 9, 3), (9, 1, 4), (8, 3, 4),
    }


def test_unknown_metadata_does_not_reject_a_valid_bridge():
    graph, scores = bridge()
    graph[1][2][0]["host"] = "known-host"
    incidents = build_incidents(graph, scores, 5, IncidentBuilderConfig(group_by=("host",)))
    assert len(incidents) == 1
