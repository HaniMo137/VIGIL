import networkx as nx
import pytest

from pidsmaker.incidents.reachability import (
    eligible_outgoing_edges,
    temporal_path_exists,
)


@pytest.mark.parametrize(
    "offset, eligible",
    [(9, False), (10, True), (12, True), (14, True), (15, False)],
)
def test_outgoing_edges_use_inclusive_nanosecond_time_limits(offset, eligible):
    graph = nx.MultiDiGraph()
    base_time = 1_700_000_000_000_000_000
    graph.add_edge(10, 20, key="event", time=base_time + offset)

    result = eligible_outgoing_edges(graph, 10, base_time + 10, base_time + 14)

    expected = ((10, 20, "event"),) if eligible else ()
    assert result == expected
    assert isinstance(result, tuple)


def test_equal_time_limits_allow_only_events_at_that_time():
    graph = nx.MultiDiGraph()
    graph.add_edge(10, 20, key="before", time=19)
    graph.add_edge(10, 20, key="equal", time=20)
    graph.add_edge(10, 20, key="after", time=21)

    assert eligible_outgoing_edges(graph, 10, 20, 20) == ((10, 20, "equal"),)


def test_parallel_events_keep_their_keys_and_leave_graph_unchanged():
    graph = nx.MultiDiGraph()
    graph.add_edge(10, 20, key="first", time=2, label="EVENT_WRITE")
    graph.add_edge(10, 20, key="second", time=3, label="EVENT_READ")
    graph.add_edge(10, 20, key="late", time=5, label="EVENT_WRITE")
    original = graph.copy()

    result = eligible_outgoing_edges(graph, 10, 2, 4)

    assert len(result) == 2
    assert set(result) == {(10, 20, "first"), (10, 20, "second")}
    assert nx.utils.graphs_equal(graph, original)
    for src, dst, key in result:
        assert 2 <= graph[src][dst][key]["time"] <= 4


def test_only_outgoing_events_from_the_requested_node_are_selected():
    graph = nx.MultiDiGraph()
    graph.add_edge(20, 10, key="incoming", time=3)
    graph.add_edge(10, 30, key="outgoing", time=3)
    graph.add_edge(40, 50, key="elsewhere", time=3)

    assert eligible_outgoing_edges(graph, 10, 2, 4) == ((10, 30, "outgoing"),)


def test_node_without_outgoing_events_returns_empty_tuple():
    graph = nx.MultiDiGraph()
    graph.add_node(10)

    assert eligible_outgoing_edges(graph, 10, 2, 4) == ()


def test_no_events_in_time_range_returns_empty_tuple():
    graph = nx.MultiDiGraph()
    graph.add_edge(10, 20, time=1)
    graph.add_edge(10, 30, time=5)

    assert eligible_outgoing_edges(graph, 10, 2, 4) == ()


@pytest.mark.parametrize(
    "event_time, expected",
    [(0, False), (1, True), (2, True), (3, True), (4, False)],
)
def test_temporal_path_direct_event_obeys_inclusive_limits(event_time, expected):
    graph = nx.MultiDiGraph()
    graph.add_edge(10, 20, time=event_time)

    assert temporal_path_exists(graph, 10, 20, 1, 3, max_hops=1) is expected


@pytest.mark.parametrize(
    "first_time, second_time, expected",
    [(2, 3, True), (3, 2, False), (2, 2, True), (0, 2, False), (2, 5, False)],
)
def test_temporal_path_checks_time_order_across_events(
    first_time, second_time, expected
):
    graph = nx.MultiDiGraph()
    graph.add_edge(10, 20, time=first_time)
    graph.add_edge(20, 30, time=second_time)

    assert temporal_path_exists(graph, 10, 30, 1, 4, max_hops=2) is expected


def test_temporal_path_explores_sibling_branches_with_independent_hop_counts():
    graph = nx.MultiDiGraph()
    graph.add_edge(10, 20, time=2)  # First branch is a dead end.
    graph.add_edge(10, 30, time=2)  # Second branch also starts at hop one.
    graph.add_edge(30, 40, time=3)

    assert temporal_path_exists(graph, 10, 40, 1, 4, max_hops=2) is True


def test_temporal_path_explores_distinct_arrival_times_at_the_same_node():
    graph = nx.MultiDiGraph()
    graph.add_edge(10, 30, time=4)  # This arrival is too late to reach 40.
    graph.add_edge(10, 20, time=2)
    graph.add_edge(20, 30, time=3)  # The longer route arrives in time.
    graph.add_edge(30, 40, time=3)

    assert temporal_path_exists(graph, 10, 40, 1, 5, max_hops=3) is True


def test_temporal_path_preserves_parallel_event_timestamps():
    graph = nx.MultiDiGraph()
    graph.add_edge(10, 20, key="too-late", time=4)
    graph.add_edge(10, 20, key="early", time=2)
    graph.add_edge(20, 30, time=3)

    assert temporal_path_exists(graph, 10, 30, 1, 5, max_hops=2) is True


def test_temporal_path_terminates_on_a_cycle_at_the_hop_limit():
    graph = nx.MultiDiGraph()
    graph.add_edge(10, 20, time=2)
    graph.add_edge(20, 10, time=2)
    graph.add_node(30)

    assert temporal_path_exists(graph, 10, 30, 1, 3, max_hops=5) is False


def test_temporal_path_does_not_reverse_provenance_edges():
    graph = nx.MultiDiGraph()
    graph.add_edge(20, 10, time=2)

    assert temporal_path_exists(graph, 10, 20, 1, 3, max_hops=1) is False


@pytest.mark.parametrize("target_node, expected", [(10, True), (20, False)])
def test_temporal_path_with_zero_hops(target_node, expected):
    graph = nx.MultiDiGraph()
    graph.add_edge(10, 20, time=2)

    assert temporal_path_exists(graph, 10, target_node, 1, 3, max_hops=0) is expected


@pytest.mark.parametrize("max_hops, expected", [(1, False), (2, True), (3, True)])
def test_temporal_path_respects_the_hop_limit(max_hops, expected):
    graph = nx.MultiDiGraph()
    graph.add_edge(10, 20, time=2)
    graph.add_edge(20, 30, time=3)

    assert temporal_path_exists(graph, 10, 30, 1, 4, max_hops=max_hops) is expected


@pytest.mark.parametrize(
    "start_time, end_time, max_hops, message",
    [
        (1, 3, -1, "Maximum hops must not be negative"),
        (3, 1, 2, "Start time must not exceed end time"),
    ],
)
def test_temporal_path_rejects_invalid_limits(
    start_time, end_time, max_hops, message
):
    graph = nx.MultiDiGraph()
    graph.add_edge(10, 20, time=2)

    with pytest.raises(ValueError, match=message):
        temporal_path_exists(graph, 10, 20, start_time, end_time, max_hops)


def test_temporal_path_leaves_the_provenance_graph_unchanged():
    graph = nx.MultiDiGraph()
    graph.add_edge(10, 20, key="event", time=2, label="EVENT_WRITE")
    original = graph.copy()

    assert temporal_path_exists(graph, 10, 20, 1, 3, max_hops=1) is True
    assert nx.utils.graphs_equal(graph, original)
