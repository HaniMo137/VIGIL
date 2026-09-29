"""Bounded, time-respecting searches that preserve multiedge identities."""

import heapq
import math
from itertools import count
from typing import Callable, Hashable, Optional, Tuple

import networkx as nx

EventKey = Tuple[int, int, Hashable]
EventPath = Tuple[EventKey, ...]
EdgeCost = Callable[[int, int, Hashable, dict], float]


class PathSearchLimitExceeded(RuntimeError):
    """A search exhausted its budget; this does not mean no path exists."""


def eligible_outgoing_edges(
    graph: nx.MultiDiGraph,
    node: int,
    earliest_time: int,
    latest_time: int,
) -> EventPath:
    if earliest_time > latest_time:
        raise ValueError("Start time must not exceed end time")
    if node not in graph:
        return ()
    return tuple(
        (src, dst, key)
        for src, dst, key, data in graph.out_edges(node, keys=True, data=True)
        if earliest_time <= int(data["time"]) <= latest_time
    )


def find_temporal_path(
    graph: nx.MultiDiGraph,
    start_node: int,
    target_node: int,
    start_time: int,
    end_time: int,
    max_hops: int,
    *,
    max_search_states: int = 100_000,
    edge_cost: Optional[EdgeCost] = None,
) -> Optional[EventPath]:
    """Return a minimum-cost temporal path, () for zero hops, or None.

    Unit costs give the fewest-event path (the original BFS behavior).
    With nonnegative costs this is a bounded Dijkstra search. Arrival time
    and hop count are part of the state: visiting a node once is not enough.
    """
    if not graph.is_directed() or not graph.is_multigraph():
        raise ValueError("Temporal searches require a MultiDiGraph")
    if max_hops < 0:
        raise ValueError("Maximum hops must not be negative")
    if start_time > end_time:
        raise ValueError("Start time must not exceed end time")
    if max_search_states < 1:
        raise ValueError("Maximum search states must be positive")
    if start_node not in graph or target_node not in graph:
        return None

    serial = count()
    pending = [(0.0, 0, next(serial), start_node, start_time, ())]
    best = {(start_node, start_time, 0): 0.0}
    generated = 1
    while pending:
        cost, hops, _, node, arrival_time, path = heapq.heappop(pending)
        if cost != best[(node, arrival_time, hops)]:
            continue
        if node == target_node:
            return path
        if hops >= max_hops:
            continue

        events = eligible_outgoing_edges(graph, node, arrival_time, end_time)
        for src, dst, key in events:
            data = graph[src][dst][key]
            timestamp = int(data["time"])
            step_cost = 1.0 if edge_cost is None else float(edge_cost(src, dst, key, data))
            if not math.isfinite(step_cost) or step_cost < 0:
                raise ValueError("Edge costs must be finite and nonnegative")
            next_cost = cost + step_cost
            if not math.isfinite(next_cost):
                raise ValueError("Total path cost must be finite")
            state = (dst, timestamp, hops + 1)
            if next_cost >= best.get(state, math.inf):
                continue
            generated += 1
            if generated > max_search_states:
                raise PathSearchLimitExceeded(
                    "Temporal path search exceeded max_search_states; "
                    "increase the budget or reduce the provenance region"
                )
            best[state] = next_cost
            heapq.heappush(
                pending,
                (next_cost, hops + 1, next(serial), dst, timestamp, path + ((src, dst, key),)),
            )
    return None


def temporal_path_exists(
    graph: nx.MultiDiGraph,
    start_node: int,
    target_node: int,
    start_time: int,
    end_time: int,
    max_hops: int,
    *,
    max_search_states: int = 100_000,
) -> bool:
    return find_temporal_path(
        graph, start_node, target_node, start_time, end_time, max_hops,
        max_search_states=max_search_states,
    ) is not None
