import pytest

from pidsmaker.incidents.clustering import (
    build_seed_affinity_graph,
    cluster_seed_edges,
    seeds_are_related,
    seeds_share_node,
)
from pidsmaker.incidents.models import EdgeRef


@pytest.mark.parametrize(
    "first_endpoints, second_endpoints, expected",
    [
        ((10, 20), (10, 30), True),  # Same source.
        ((10, 20), (30, 20), True),  # Same destination.
        ((10, 20), (20, 30), True),  # Destination matches source.
        ((10, 20), (30, 10), True),  # Source matches destination.
        ((10, 20), (30, 40), False),  # Unrelated entities.
        ((10, 20), (20, 10), True),  # Reversed endpoints.
        ((10, 10), (10, 20), True),  # Self-loop touches another event.
        ((10, 10), (20, 20), False),  # Separate self-loops.
    ],
)
def test_seeds_share_node(first_endpoints, second_endpoints, expected):
    first = EdgeRef(*first_endpoints, time=100, edge_type=1, score=2.5)
    second = EdgeRef(*second_endpoints, time=200, edge_type=2, score=3.0)

    assert seeds_share_node(first, second) is expected
    assert seeds_share_node(second, first) is expected


@pytest.mark.parametrize(
    "second_endpoints, time_gap_ns, expected",
    [
        ((20, 30), 0, True),
        ((20, 30), 3_000_000_000, True),
        ((20, 30), 5_000_000_000, True),
        ((20, 30), 5_000_000_001, False),
        ((20, 30), 8_000_000_000, False),
        ((30, 40), 1_000_000_000, False),
        ((30, 40), 0, False),
    ],
)
def test_seeds_are_related(second_endpoints, time_gap_ns, expected):
    first = EdgeRef(10, 20, time=10_000_000_000, edge_type=1, score=2.5)
    second = EdgeRef(
        *second_endpoints,
        time=first.time + time_gap_ns,
        edge_type=2,
        score=3.0,
    )

    assert seeds_are_related(first, second, max_time_gap_ns=5_000_000_000) is expected
    assert seeds_are_related(second, first, max_time_gap_ns=5_000_000_000) is expected


@pytest.mark.parametrize("time_gap_ns, expected", [(0, True), (1, False)])
def test_zero_time_limit_requires_equal_timestamps(time_gap_ns, expected):
    first = EdgeRef(10, 20, time=100, edge_type=1, score=2.5)
    second = EdgeRef(20, 30, time=100 + time_gap_ns, edge_type=2, score=3.0)

    assert seeds_are_related(first, second, max_time_gap_ns=0) is expected


def test_seeds_are_related_rejects_negative_time_limit():
    first = EdgeRef(10, 20, time=100, edge_type=1, score=2.5)
    second = EdgeRef(20, 30, time=100, edge_type=2, score=3.0)

    with pytest.raises(ValueError, match="Maximum time gap must not be negative"):
        seeds_are_related(first, second, max_time_gap_ns=-1)


@pytest.mark.parametrize(
    "seeds, expected_nodes",
    [
        ((), set()),
        ((EdgeRef(10, 20, time=100, edge_type=1, score=2.5),), {0}),
    ],
)
def test_affinity_graph_handles_empty_and_single_seed(seeds, expected_nodes):
    affinity = build_seed_affinity_graph(seeds, max_time_gap_ns=5_000_000_000)

    assert set(affinity.nodes) == expected_nodes
    assert affinity.number_of_edges() == 0
    assert not affinity.is_directed()
    assert not affinity.is_multigraph()


def test_affinity_graph_keeps_all_unrelated_seeds():
    seeds = (
        EdgeRef(10, 20, time=100, edge_type=1, score=2.5),
        EdgeRef(30, 40, time=100, edge_type=1, score=2.5),
        EdgeRef(50, 60, time=100, edge_type=1, score=2.5),
    )

    affinity = build_seed_affinity_graph(seeds, max_time_gap_ns=5_000_000_000)

    assert set(affinity.nodes) == {0, 1, 2}
    assert affinity.number_of_edges() == 0


def test_affinity_graph_builds_chain_and_preserves_isolated_seed():
    seeds = (
        EdgeRef(10, 20, time=10_000_000_000, edge_type=1, score=2.5),
        EdgeRef(20, 30, time=14_000_000_000, edge_type=1, score=2.5),
        EdgeRef(30, 10, time=18_000_000_000, edge_type=1, score=2.5),
        EdgeRef(40, 50, time=14_000_000_000, edge_type=1, score=2.5),
    )

    affinity = build_seed_affinity_graph(seeds, max_time_gap_ns=5_000_000_000)

    assert set(affinity.nodes) == {0, 1, 2, 3}
    assert {frozenset(edge) for edge in affinity.edges} == {
        frozenset((0, 1)),
        frozenset((1, 2)),
    }
    # Seeds 0 and 2 share a node but are eight seconds apart.
    assert not affinity.has_edge(0, 2)
    assert affinity.degree(3) == 0


def test_affinity_graph_preserves_duplicate_seed_occurrences():
    seed = EdgeRef(10, 20, time=100, edge_type=1, score=2.5)

    affinity = build_seed_affinity_graph((seed, seed), max_time_gap_ns=0)

    assert set(affinity.nodes) == {0, 1}
    assert affinity.number_of_edges() == 1
    assert affinity.has_edge(0, 1)


@pytest.mark.parametrize(
    "seeds",
    [(), (EdgeRef(10, 20, time=100, edge_type=1, score=2.5),)],
)
def test_affinity_graph_rejects_negative_limit_without_any_pairs(seeds):
    with pytest.raises(ValueError, match="Maximum time gap must not be negative"):
        build_seed_affinity_graph(seeds, max_time_gap_ns=-1)


@pytest.mark.parametrize(
    "seeds, expected_clusters",
    [
        ((), ()),
        (
            (EdgeRef(10, 20, time=100, edge_type=1, score=2.5),),
            ((EdgeRef(10, 20, time=100, edge_type=1, score=2.5),),),
        ),
    ],
)
def test_cluster_seed_edges_handles_empty_and_single_seed(seeds, expected_clusters):
    assert cluster_seed_edges(seeds, max_time_gap_ns=5_000_000_000) == expected_clusters


def test_cluster_seed_edges_maps_nonconsecutive_indices_to_original_events():
    seeds = (
        EdgeRef(10, 20, time=100, edge_type=1, score=2.5),
        EdgeRef(40, 50, time=100, edge_type=2, score=3.0),
        EdgeRef(20, 30, time=101, edge_type=3, score=3.5),
        EdgeRef(50, 60, time=101, edge_type=4, score=4.0),
    )

    clusters = cluster_seed_edges(seeds, max_time_gap_ns=5_000_000_000)

    assert clusters == ((seeds[0], seeds[2]), (seeds[1], seeds[3]))
    assert clusters[0][0] is seeds[0]
    assert clusters[0][1] is seeds[2]
    assert clusters[1][0] is seeds[1]
    assert clusters[1][1] is seeds[3]


def test_cluster_seed_edges_groups_indirect_chain_and_keeps_isolated_event():
    seeds = (
        EdgeRef(10, 20, time=10_000_000_000, edge_type=1, score=2.5),
        EdgeRef(20, 30, time=14_000_000_000, edge_type=1, score=2.5),
        EdgeRef(30, 40, time=18_000_000_000, edge_type=1, score=2.5),
        EdgeRef(50, 60, time=14_000_000_000, edge_type=1, score=2.5),
    )

    clusters = cluster_seed_edges(seeds, max_time_gap_ns=5_000_000_000)

    assert clusters == (seeds[:3], (seeds[3],))


def test_cluster_seed_edges_returns_singletons_for_unrelated_events():
    seeds = (
        EdgeRef(10, 20, time=100, edge_type=1, score=2.5),
        EdgeRef(30, 40, time=100, edge_type=1, score=2.5),
        EdgeRef(50, 60, time=100, edge_type=1, score=2.5),
    )

    assert cluster_seed_edges(seeds, max_time_gap_ns=0) == tuple((seed,) for seed in seeds)


def test_cluster_seed_edges_separates_events_beyond_time_limit():
    seeds = (
        EdgeRef(10, 20, time=10_000_000_000, edge_type=1, score=2.5),
        EdgeRef(20, 30, time=16_000_000_000, edge_type=1, score=2.5),
    )

    assert cluster_seed_edges(seeds, max_time_gap_ns=5_000_000_000) == (
        (seeds[0],), (seeds[1],)
    )


def test_cluster_seed_edges_preserves_duplicate_occurrences():
    seed = EdgeRef(10, 20, time=100, edge_type=1, score=2.5)

    assert cluster_seed_edges((seed, seed), max_time_gap_ns=0) == ((seed, seed),)


def test_cluster_seed_edges_rejects_negative_time_limit_for_empty_input():
    with pytest.raises(ValueError, match="Maximum time gap must not be negative"):
        cluster_seed_edges((), max_time_gap_ns=-1)
