"""Construct evidence-preserving candidate incidents from scored provenance."""

import math
from copy import deepcopy
from dataclasses import dataclass, field
from numbers import Integral
from typing import Mapping, Optional, Tuple

import networkx as nx
import pandas as pd

from pidsmaker.incidents.clustering import build_seed_affinity_graph
from pidsmaker.incidents.models import EdgeRef, Incident
from pidsmaker.incidents.seeds import _has_identity, select_seed_edges


@dataclass(frozen=True)
class IncidentBuilderConfig:
    """Engineering starting values, not validation-tuned research settings.

    Context, metadata boundaries, and ancestry are opt-in. Ancestry relation
    IDs must identify parent-to-child events in the supplied graph.
    """

    max_time_gap_ns: int = 5_000_000_000
    max_connector_hops: int = 5
    max_search_states: int = 100_000
    max_seed_pairs: int = 1_000_000
    group_by: Tuple[str, ...] = ()
    ancestry_edge_types: Tuple[int, ...] = ()
    max_ancestry_hops: int = 0
    context_hops: int = 0
    max_context_edges: int = 100
    context_time_padding_ns: int = 0
    context_edge_types: Tuple[int, ...] = ()
    alpha: float = 1.0
    beta: float = 0.0
    gamma: float = 0.0
    relation_penalties: Mapping[int, float] = field(default_factory=dict)

    def __post_init__(self):
        for name in (
            "max_time_gap_ns", "max_connector_hops", "max_ancestry_hops",
            "context_hops", "max_context_edges", "context_time_padding_ns",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        for name in ("max_search_states", "max_seed_pairs"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name, value in (
            ("alpha", self.alpha), ("beta", self.beta), ("gamma", self.gamma),
            *((f"relation penalty {key}", value) for key, value in self.relation_penalties.items()),
        ):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")


def _integer(value, field_name):
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be an integer")
    try:
        number = int(value)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError(f"{field_name} must be an integer") from exc
    if not isinstance(value, str) and number != value:
        raise ValueError(f"{field_name} must be an integer")
    if field_name == "time" and isinstance(value, float) and abs(value) > 2 ** 53:
        raise ValueError("Nanosecond time must not be stored as an imprecise float")
    return number


def prepare_provenance_graph(
    provenance_graph: nx.MultiDiGraph,
    edge_scores: pd.DataFrame,
    relation_to_id: Optional[Mapping[str, int]] = None,
) -> nx.MultiDiGraph:
    """Copy a region and match score rows to exact events, never endpoints alone.

    PIDSMaker graphs use operation labels; CSVs use numeric types. Pass the
    matching label-to-score-ID mapping when graph edges lack ``edge_type``.
    Ambiguous parallel events require a CSV ``key`` or ``event_uuid`` column.
    """
    if not isinstance(provenance_graph, nx.MultiDiGraph):
        raise ValueError("Provenance graph must be a MultiDiGraph")
    required = {"srcnode", "dstnode", "time", "edge_type", "loss"}
    missing = required - set(edge_scores.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    graph = nx.MultiDiGraph()
    graph.graph.update(deepcopy(provenance_graph.graph))
    node_ids = {}
    for node, data in provenance_graph.nodes(data=True):
        node_id = _integer(node, "node ID")
        if node_id in graph:
            raise ValueError("Node IDs collide after integer conversion")
        node_ids[node] = node_id
        graph.add_node(node_id, **deepcopy(data))

    index = {}
    for src, dst, key, original in provenance_graph.edges(keys=True, data=True):
        data = deepcopy(original)
        if "time" not in data:
            raise ValueError("Every provenance event needs a time attribute")
        data["time"] = _integer(data["time"], "time")
        if "edge_type" not in data:
            label = data.get("label")
            if relation_to_id is None or label not in relation_to_id:
                raise ValueError(f"Missing relation mapping for {label!r}; provide relation_to_id")
            data["edge_type"] = _integer(relation_to_id[label], "edge_type")
        else:
            data["edge_type"] = _integer(data["edge_type"], "edge_type")
        # Use only the supplied detector scores, not stale graph annotations.
        data.pop("score", None)
        u, v = node_ids[src], node_ids[dst]
        graph.add_edge(u, v, key=key, **data)
        signature = (u, v, data["time"], data["edge_type"])
        index.setdefault(signature, []).append((u, v, key))

    assigned = set()
    for row in edge_scores.to_dict("records"):
        signature = tuple(_integer(row[name], name) for name in (
            "srcnode", "dstnode", "time", "edge_type",
        ))
        score = float(row["loss"])
        if not math.isfinite(score) or score < 0:
            raise ValueError("Anomaly losses must be finite and nonnegative")
        candidates = index.get(signature, [])
        if "key" in row and _has_identity(row["key"]):
            candidates = [event for event in candidates if event[2] == row["key"]]
        if "event_uuid" in row and _has_identity(row["event_uuid"]):
            candidates = [
                event for event in candidates
                if graph[event[0]][event[1]][event[2]].get("event_uuid") == row["event_uuid"]
            ]
        if not candidates:
            raise ValueError(f"Scored event {signature} not found in the provenance region")
        if len(candidates) > 1:
            raise ValueError(f"Ambiguous scored event {signature}; provide key or event_uuid")
        event = candidates[0]
        if event in assigned:
            raise ValueError(f"Duplicate score rows for event {signature}")
        assigned.add(event)
        graph[event[0]][event[1]][event[2]]["score"] = score
    return graph


def _edge_ref(graph, event):
    src, dst, key = event
    data = graph[src][dst][key]
    return EdgeRef(src, dst, data["time"], data["edge_type"], data.get("score"), key)


def prune_incident_graph(graph, terminal_nodes, required_edges=()):
    """Copy and iteratively remove unsupported leaves; never remove evidence."""
    result = deepcopy(graph.copy())
    protected = set(terminal_nodes)
    for src, dst, _ in required_edges:
        protected.update((src, dst))
    for component in list(nx.weakly_connected_components(result)):
        if not component & protected:
            result.remove_nodes_from(component)
    while True:
        leaves = [
            node for node in result
            if node not in protected
            and len(set(result.predecessors(node)) | set(result.successors(node))) <= 1
        ]
        if not leaves:
            return result
        result.remove_nodes_from(leaves)


def _add_context(graph, kept, cluster, config, all_seeds):
    if not config.context_hops or not config.max_context_edges:
        return ()
    lower = min(seed.time for seed in cluster) - config.context_time_padding_ns
    upper = max(seed.time for seed in cluster) + config.context_time_padding_ns
    frontier = {node for event in kept for node in event[:2]}
    expanded = set()
    context = []
    for _ in range(config.context_hops):
        candidates = {}
        for node in sorted(frontier - expanded):
            expanded.add(node)
            events = list(graph.in_edges(node, keys=True, data=True))
            events += list(graph.out_edges(node, keys=True, data=True))
            for src, dst, key, data in events:
                event = (src, dst, key)
                if event in kept or event in all_seeds:
                    continue
                if not lower <= data["time"] <= upper:
                    continue
                if config.context_edge_types and data["edge_type"] not in config.context_edge_types:
                    continue
                candidates[event] = data["time"]
        frontier = set()
        for event in sorted(candidates, key=lambda item: (candidates[item], item[:2], repr(item[2]))):
            if len(context) >= config.max_context_edges:
                return tuple(context)
            kept.add(event)
            context.append(event)
            frontier.update(event[:2])
    return tuple(context)


def build_incidents(
    provenance_graph: nx.MultiDiGraph,
    edge_scores: pd.DataFrame,
    threshold: float,
    config: Optional[IncidentBuilderConfig] = None,
    *, relation_to_id: Optional[Mapping[str, int]] = None,
) -> Tuple[Incident, ...]:
    """Select seeds, cluster, recover compact connectors, and retain context.

    Thresholds must come from validation, not test labels. Outputs are
    unverified candidate incidents, not malicious labels or memory entries.
    """
    if not math.isfinite(threshold) or threshold < 0:
        raise ValueError("Threshold must be finite and nonnegative")
    config = config or IncidentBuilderConfig()
    # Check the score schema even if no seeds are selected.
    selected = select_seed_edges(edge_scores, threshold)
    if not selected:
        return ()
    graph = prepare_provenance_graph(provenance_graph, edge_scores, relation_to_id)
    seeds = tuple(
        _edge_ref(graph, (src, dst, key))
        for src, dst, key, data in graph.edges(keys=True, data=True)
        if data.get("score", -math.inf) >= threshold
    )
    seed_events = {(seed.srcnode, seed.dstnode, seed.key) for seed in seeds}
    scale = max(threshold, 1e-12)

    def cost(src, dst, key, data):
        normalized = min(1.0, max(0.0, data.get("score", 0.0) / scale))
        distance = min(abs(data["time"] - seed.time) for seed in seeds)
        distance = min(1.0, distance / max(config.max_time_gap_ns, 1))
        penalty = config.relation_penalties.get(data["edge_type"], 1.0)
        return config.alpha * (1.0 - normalized) + config.beta * distance + config.gamma * penalty

    affinity = build_seed_affinity_graph(
        seeds, config.max_time_gap_ns, graph, config.max_connector_hops,
        max_search_states=config.max_search_states, edge_cost=cost,
        max_seed_pairs=config.max_seed_pairs,
        group_by=config.group_by, ancestry_edge_types=config.ancestry_edge_types,
        max_ancestry_hops=config.max_ancestry_hops,
    )
    incidents = []
    for component in nx.connected_components(affinity):
        indices = sorted(component)
        cluster = tuple(seeds[index] for index in indices)
        cluster_events = {(seed.srcnode, seed.dstnode, seed.key) for seed in cluster}
        kept = set(cluster_events)
        # Spanning the affinity component avoids unioning every pair's path.
        tree = nx.minimum_spanning_tree(affinity.subgraph(component), weight="weight")
        for _, _, data in tree.edges(data=True):
            kept.update(data["connector_path"])
        if (kept & seed_events) - cluster_events:
            raise ValueError("A connector path crosses another seed cluster; review affinity limits")
        connectors = kept - cluster_events
        context = _add_context(graph, kept, cluster, config, seed_events)
        region = deepcopy(graph.edge_subgraph(kept).copy())
        terminals = {node for event in cluster_events for node in event[:2]}
        region = prune_incident_graph(region, terminals, kept)

        def order(event):
            return (graph[event[0]][event[1]][event[2]]["time"], event[:2], repr(event[2]))

        connector_refs = tuple(_edge_ref(graph, event) for event in sorted(connectors, key=order))
        context_refs = tuple(_edge_ref(graph, event) for event in sorted(context, key=order))
        for kind, refs in (("seed", cluster), ("connector", connector_refs), ("context", context_refs)):
            for ref in refs:
                region[ref.srcnode][ref.dstnode][ref.key]["incident_role"] = kind
        region.graph["seed_threshold"] = threshold
        region.graph["context_limit_reached"] = bool(
            config.context_hops and len(context) >= config.max_context_edges
        )
        incidents.append(Incident(region, cluster, connector_refs, context_refs))
    return tuple(incidents)
