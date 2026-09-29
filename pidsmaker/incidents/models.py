"""Core data structures shared by the VIGIL incident-building stages."""

import math
from dataclasses import dataclass
from typing import Hashable, Optional, Tuple

import networkx as nx


@dataclass(frozen=True)
class EdgeRef:
    """Stable reference to one provenance event.

    ``key`` is optional because PIDSMaker's score CSV currently records the
    endpoints, timestamp, and edge type, but not the NetworkX multiedge key.
    """

    srcnode: int
    dstnode: int
    time: int
    edge_type: int
    score: Optional[float] = None
    key: Optional[Hashable] = None


@dataclass
class Incident:
    """An unverified candidate incident with preserved provenance evidence."""

    graph: nx.Graph
    seed_edges: Tuple[EdgeRef, ...]
    connector_edges: Tuple[EdgeRef, ...] = ()
    context_edges: Tuple[EdgeRef, ...] = ()

    def __post_init__(self) -> None:
        if not self.seed_edges:
            raise ValueError("An incident must contain at least one seed edge")

        for edge in self.seed_edges:
            if edge.score is None:
                raise ValueError("Every seed edge must have an anomaly score")
            if not math.isfinite(edge.score) or edge.score < 0:
                raise ValueError("Seed anomaly scores must be finite and nonnegative")
        identities = set()
        for edge in self.edges:
            if not self.graph.has_edge(edge.srcnode, edge.dstnode):
                raise ValueError("Every edge must belong to the incident graph")
            identity = (edge.srcnode, edge.dstnode, edge.key, edge.time, edge.edge_type)
            if identity in identities:
                raise ValueError("An event cannot have multiple incident roles")
            identities.add(identity)
            if edge.key is not None:
                if not self.graph.is_multigraph() or not self.graph.has_edge(
                    edge.srcnode, edge.dstnode, edge.key
                ):
                    raise ValueError("Every edge key must belong to the incident graph")
                data = self.graph[edge.srcnode][edge.dstnode][edge.key]
                if data.get("time", edge.time) != edge.time or data.get(
                    "edge_type", edge.edge_type
                ) != edge.edge_type:
                    raise ValueError("Edge reference does not match the incident event")
            else:
                candidates = (
                    self.graph[edge.srcnode][edge.dstnode].values()
                    if self.graph.is_multigraph()
                    else (self.graph[edge.srcnode][edge.dstnode],)
                )
                if not any(
                    data.get("time", edge.time) == edge.time
                    and data.get("edge_type", edge.edge_type) == edge.edge_type
                    for data in candidates
                ):
                    raise ValueError("Edge reference does not match the incident event")

    @property
    def edges(self) -> Tuple[EdgeRef, ...]:
        """All explicitly classified edges in the incident."""

        return self.seed_edges + self.connector_edges + self.context_edges

    @property
    def start_time(self) -> int:
        return min(edge.time for edge in self.edges)

    @property
    def end_time(self) -> int:
        return max(edge.time for edge in self.edges)
