"""Seed affinity with optional temporal connectors and process ancestry."""

from typing import Optional, Tuple

import networkx as nx

from pidsmaker.incidents.models import EdgeRef
from pidsmaker.incidents.reachability import (
    EdgeCost,
    EventPath,
    PathSearchLimitExceeded,
    find_temporal_path,
    temporal_path_exists,
)


def seeds_share_node(first: EdgeRef, second: EdgeRef) -> bool:
    return bool({first.srcnode, first.dstnode} & {second.srcnode, second.dstnode})


def seed_can_reach(
    earlier: EdgeRef,
    later: EdgeRef,
    provenance_graph: nx.MultiDiGraph,
    max_hops: int,
) -> bool:
    return temporal_path_exists(
        graph=provenance_graph,
        start_node=earlier.dstnode,
        target_node=later.srcnode,
        start_time=earlier.time,
        end_time=later.time,
        max_hops=max_hops,
    )


def _seed_metadata(seed, graph, attribute):
    data = graph.get_edge_data(seed.srcnode, seed.dstnode, default={})
    if seed.key in data and attribute in data[seed.key]:
        return data[seed.key][attribute]
    values = {
        graph.nodes[node][attribute]
        for node in (seed.srcnode, seed.dstnode)
        if node in graph and graph.nodes[node].get(attribute) is not None
    }
    return next(iter(values)) if len(values) == 1 else None


def _ancestor_paths(seed, graph, relation_types, max_hops, lower_time, max_search_states):
    """Backward traversal of explicitly configured parent-to-child relations."""
    paths = {seed.srcnode: [()], seed.dstnode: [()]}
    pending = [(node, seed.time, ()) for node in paths]
    if len(pending) > max_search_states:
        raise PathSearchLimitExceeded("Ancestry search exceeded max_search_states")
    for node, latest_time, path in pending:
        if len(path) >= max_hops or node not in graph:
            continue
        for src, dst, key, data in graph.in_edges(node, keys=True, data=True):
            if data.get("edge_type") not in relation_types:
                continue
            timestamp = int(data["time"])
            if lower_time <= timestamp <= latest_time:
                parent_path = ((src, dst, key),) + path
                if len(pending) >= max_search_states:
                    raise PathSearchLimitExceeded("Ancestry search exceeded max_search_states")
                paths.setdefault(src, []).append(parent_path)
                pending.append((src, timestamp, parent_path))
    return paths


def _relation_path(
    first, second, max_time_gap_ns, graph, max_hops, max_search_states,
    edge_cost, group_by, ancestry_edge_types, max_ancestry_hops,
) -> Optional[EventPath]:
    if abs(first.time - second.time) > max_time_gap_ns:
        return None
    if graph is not None:
        for attribute in group_by:
            left = _seed_metadata(first, graph, attribute)
            right = _seed_metadata(second, graph, attribute)
            if left is not None and right is not None and left != right:
                return None
    if seeds_share_node(first, second):
        return ()
    if graph is None:
        return None

    earlier, later = (first, second) if first.time <= second.time else (second, first)
    orders = [(earlier, later)]
    # Equal timestamps have no unique order; check both directions.
    if first.time == second.time:
        orders.append((later, earlier))
    candidates = []
    for earlier, later in orders:
        path = find_temporal_path(
            graph, earlier.dstnode, later.srcnode, earlier.time, later.time, max_hops,
            max_search_states=max_search_states, edge_cost=edge_cost,
        )
        if path is not None:
            candidates.append(path)

    def cost(path):
        if edge_cost is None:
            return len(path)
        return sum(edge_cost(src, dst, key, graph[src][dst][key]) for src, dst, key in path)

    if ancestry_edge_types and max_ancestry_hops:
        lower_time = min(first.time, second.time) - max_time_gap_ns
        left = _ancestor_paths(
            first, graph, ancestry_edge_types, max_ancestry_hops, lower_time, max_search_states,
        )
        right = _ancestor_paths(
            second, graph, ancestry_edge_types, max_ancestry_hops, lower_time, max_search_states,
        )
        for ancestor in sorted(left.keys() & right.keys()):
            left_path = min(left[ancestor], key=lambda path: (cost(path), len(path)))
            right_path = min(right[ancestor], key=lambda path: (cost(path), len(path)))
            candidates.append(tuple(dict.fromkeys(left_path + right_path)))

    return min(candidates, key=lambda path: (cost(path), len(path))) if candidates else None


def seeds_are_related(
    first: EdgeRef, second: EdgeRef, max_time_gap_ns: int,
    provenance_graph: Optional[nx.MultiDiGraph] = None, max_hops: int = 5,
    *, max_search_states: int = 100_000, edge_cost: Optional[EdgeCost] = None,
    group_by: Tuple[str, ...] = (), ancestry_edge_types: Tuple[int, ...] = (),
    max_ancestry_hops: int = 0,
) -> bool:
    _validate_limits(max_time_gap_ns, max_hops, max_search_states, max_ancestry_hops)
    return _relation_path(
        first, second, max_time_gap_ns, provenance_graph, max_hops, max_search_states,
        edge_cost, group_by, ancestry_edge_types, max_ancestry_hops,
    ) is not None


def _validate_limits(max_time_gap_ns, max_hops, max_search_states, max_ancestry_hops):
    if max_time_gap_ns < 0:
        raise ValueError("Maximum time gap must not be negative")
    if max_hops < 0 or max_ancestry_hops < 0:
        raise ValueError("Maximum hops must not be negative")
    if max_search_states < 1:
        raise ValueError("Maximum search states must be positive")


def build_seed_affinity_graph(
    seeds: Tuple[EdgeRef, ...], max_time_gap_ns: int,
    provenance_graph: Optional[nx.MultiDiGraph] = None, max_hops: int = 5,
    *, max_search_states: int = 100_000, edge_cost: Optional[EdgeCost] = None,
    group_by: Tuple[str, ...] = (), ancestry_edge_types: Tuple[int, ...] = (),
    max_ancestry_hops: int = 0, max_seed_pairs: int = 1_000_000,
) -> nx.Graph:
    _validate_limits(max_time_gap_ns, max_hops, max_search_states, max_ancestry_hops)
    if max_seed_pairs < 1:
        raise ValueError("Maximum seed pairs must be positive")
    affinity = nx.Graph()
    affinity.add_nodes_from(range(len(seeds)))
    ordered = sorted(range(len(seeds)), key=lambda index: (seeds[index].time, index))
    pairs = 0
    for position, i in enumerate(ordered):
        for next_position in range(position + 1, len(ordered)):
            j = ordered[next_position]
            if seeds[j].time - seeds[i].time > max_time_gap_ns:
                break
            pairs += 1
            if pairs > max_seed_pairs:
                raise PathSearchLimitExceeded(
                    "Seed affinity exceeded max_seed_pairs; reduce the provenance region"
                )
            _add_affinity_edge(
                affinity, seeds, i, j, max_time_gap_ns, provenance_graph,
                max_hops, max_search_states, edge_cost, group_by,
                ancestry_edge_types, max_ancestry_hops,
            )
    return affinity


def _add_affinity_edge(
    affinity, seeds, i, j, max_time_gap_ns, provenance_graph, max_hops,
    max_search_states, edge_cost, group_by, ancestry_edge_types, max_ancestry_hops,
):
    path = _relation_path(
        seeds[i], seeds[j], max_time_gap_ns, provenance_graph, max_hops,
        max_search_states, edge_cost, group_by, ancestry_edge_types, max_ancestry_hops,
    )
    if path is not None:
        cost = len(path) if edge_cost is None else sum(
            edge_cost(src, dst, key, provenance_graph[src][dst][key])
            for src, dst, key in path
        )
        affinity.add_edge(i, j, connector_path=path, weight=float(cost))


def cluster_seed_edges(
    seeds: Tuple[EdgeRef, ...], max_time_gap_ns: int,
    provenance_graph: Optional[nx.MultiDiGraph] = None, max_hops: int = 5, **kwargs,
) -> Tuple[Tuple[EdgeRef, ...], ...]:
    affinity = build_seed_affinity_graph(
        seeds, max_time_gap_ns, provenance_graph, max_hops, **kwargs,
    )
    return tuple(
        tuple(seeds[i] for i in sorted(component))
        for component in nx.connected_components(affinity)
    )
