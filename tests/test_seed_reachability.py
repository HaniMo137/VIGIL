import networkx as nx
import pytest

from pidsmaker.incidents import clustering
from pidsmaker.incidents.clustering import seed_can_reach
from pidsmaker.incidents.models import EdgeRef


@pytest.mark.parametrize("path_exists", [True, False])
def test_seed_can_reach_passes_seed_endpoints_and_times(monkeypatch, path_exists):
    earlier = EdgeRef(10, 20, time=100, edge_type=1, score=2.5)
    later = EdgeRef(40, 50, time=200, edge_type=2, score=3.0)
    graph = nx.MultiDiGraph()
    calls = []

    def fake_temporal_path_exists(**kwargs):
        calls.append(kwargs)
        return path_exists

    monkeypatch.setattr(clustering, "temporal_path_exists", fake_temporal_path_exists)

    assert seed_can_reach(earlier, later, graph, max_hops=3) is path_exists
    assert calls == [
        {
            "graph": graph,
            "start_node": 20,
            "target_node": 40,
            "start_time": 100,
            "end_time": 200,
            "max_hops": 3,
        }
    ]


@pytest.mark.parametrize(
    "first_time, second_time, expected",
    [
        (2_000_000_000, 3_000_000_000, True),
        (3_000_000_000, 2_000_000_000, False),
        (1_000_000_000, 4_000_000_000, True),
        (0, 3_000_000_000, False),
        (2_000_000_000, 5_000_000_000, False),
        (2_000_000_000, 2_000_000_000, True),
    ],
)
def test_seed_can_reach_uses_a_time_respecting_bridge(
    first_time, second_time, expected
):
    earlier = EdgeRef(10, 20, time=1_000_000_000, edge_type=1, score=2.5)
    later = EdgeRef(40, 50, time=4_000_000_000, edge_type=2, score=3.0)
    graph = nx.MultiDiGraph()
    graph.add_edge(10, 20, time=earlier.time)
    graph.add_edge(20, 30, time=first_time)
    graph.add_edge(30, 40, time=second_time)
    graph.add_edge(40, 50, time=later.time)
    original = graph.copy()

    assert seed_can_reach(earlier, later, graph, max_hops=2) is expected
    assert nx.utils.graphs_equal(graph, original)


@pytest.mark.parametrize("max_hops, expected", [(0, False), (1, False), (2, True), (3, True)])
def test_seed_can_reach_counts_bridge_hops_not_seed_events(max_hops, expected):
    earlier = EdgeRef(10, 20, time=1, edge_type=1, score=2.5)
    later = EdgeRef(40, 50, time=4, edge_type=2, score=3.0)
    graph = nx.MultiDiGraph()
    graph.add_edge(10, 20, time=1)
    graph.add_edge(20, 30, time=2)
    graph.add_edge(30, 40, time=3)
    graph.add_edge(40, 50, time=4)

    assert seed_can_reach(earlier, later, graph, max_hops) is expected


@pytest.mark.parametrize("wrong_bridge", [(10, 40), (20, 50)])
def test_seed_can_reach_rejects_paths_between_the_wrong_seed_endpoints(wrong_bridge):
    earlier = EdgeRef(10, 20, time=1, edge_type=1, score=2.5)
    later = EdgeRef(40, 50, time=4, edge_type=2, score=3.0)
    graph = nx.MultiDiGraph()
    graph.add_edge(10, 20, time=1)
    graph.add_edge(40, 50, time=4)
    graph.add_edge(*wrong_bridge, time=2)

    assert seed_can_reach(earlier, later, graph, max_hops=3) is False


@pytest.mark.parametrize("max_hops", [0, 3])
def test_seed_can_reach_needs_no_connector_when_boundary_nodes_match(max_hops):
    earlier = EdgeRef(10, 20, time=1, edge_type=1, score=2.5)
    later = EdgeRef(20, 30, time=4, edge_type=2, score=3.0)
    graph = nx.MultiDiGraph()
    graph.add_edge(10, 20, time=1)
    graph.add_edge(20, 30, time=4)

    assert seed_can_reach(earlier, later, graph, max_hops) is True


def test_seed_can_reach_requires_chronological_arguments():
    earlier = EdgeRef(10, 20, time=1, edge_type=1, score=2.5)
    later = EdgeRef(40, 50, time=4, edge_type=2, score=3.0)

    with pytest.raises(ValueError, match="Start time must not exceed end time"):
        seed_can_reach(later, earlier, nx.MultiDiGraph(), max_hops=2)


def test_seed_can_reach_rejects_negative_hop_limits():
    earlier = EdgeRef(10, 20, time=1, edge_type=1, score=2.5)
    later = EdgeRef(40, 50, time=4, edge_type=2, score=3.0)

    with pytest.raises(ValueError, match="Maximum hops must not be negative"):
        seed_can_reach(earlier, later, nx.MultiDiGraph(), max_hops=-1)
