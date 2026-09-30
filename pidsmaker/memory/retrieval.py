"""Read-only similarity search over independently verified memory entries."""

import math
from dataclasses import dataclass
from typing import Mapping, Optional, Tuple

import networkx as nx
import numpy as np

from pidsmaker.memory.models import MemoryLabel
from pidsmaker.memory.signature import SignatureSchema, build_incident_signature
from pidsmaker.memory.store import IncidentMemoryStore


@dataclass(frozen=True)
class MemoryMatch:
    entry_id: str
    label: MemoryLabel
    similarity: float
    above_threshold: Optional[bool]


@dataclass(frozen=True)
class RetrievalResult:
    matches: Tuple[MemoryMatch, ...]
    threshold: Optional[float]
    no_strong_match: bool
    schema_id: str


class IncidentRetriever:
    """Build signatures from verified graphs and rank them without relabeling.

    Reference features must come from the same frozen featurization pipeline as
    query features. The store is checked on every reference graph read; no
    detector-generated labels are accepted by this interface.
    """

    def __init__(self, store: IncidentMemoryStore, reference_features: Mapping,
                 encoder, schema: SignatureSchema, *, threshold: Optional[float] = None,
                 top_k_per_label: int = 3):
        if not isinstance(store, IncidentMemoryStore):
            raise ValueError("A verified IncidentMemoryStore is required")
        if not isinstance(schema, SignatureSchema):
            raise ValueError("A fixed SignatureSchema is required")
        if (threshold is not None and (isinstance(threshold, bool)
                or not isinstance(threshold, (int, float))
                or not math.isfinite(threshold) or not -1 <= threshold <= 1)):
            raise ValueError("Similarity threshold must be finite and between -1 and 1")
        if isinstance(top_k_per_label, bool) or not isinstance(top_k_per_label, int) or top_k_per_label < 1:
            raise ValueError("top_k_per_label must be a positive integer")
        if not isinstance(reference_features, Mapping):
            raise ValueError("Reference features must be mapped by entry ID")
        self.store = store
        self.encoder = encoder
        self.schema = schema
        self.threshold = float(threshold) if threshold is not None else None
        self.top_k_per_label = top_k_per_label
        self._references = {}
        for summary in store.list_entries():
            if summary.entry_id not in reference_features:
                raise ValueError(f"Missing features for memory entry {summary.entry_id!r}")
            entry = store.get(summary.entry_id)
            signature = build_incident_signature(
                entry.graph, reference_features[entry.entry_id], encoder, schema,
            )
            self._references[entry.entry_id] = (entry.label, signature.vector)

    def reference_graph(self, entry_id: str) -> nx.MultiDiGraph:
        """Return an independently rechecked reference graph, never a candidate."""
        if entry_id not in self._references:
            raise KeyError(entry_id)
        return self.store.get(entry_id).graph

    def search(self, graph: nx.MultiDiGraph, node_features: Mapping) -> RetrievalResult:
        query = build_incident_signature(graph, node_features, self.encoder, self.schema)
        query_norm = np.linalg.norm(query.vector)
        ranked = []
        for entry_id, (label, vector) in self._references.items():
            denominator = query_norm * np.linalg.norm(vector)
            similarity = float(np.dot(query.vector, vector) / denominator) if denominator else 0.0
            similarity = max(-1.0, min(1.0, similarity))
            ranked.append(MemoryMatch(
                entry_id, label, similarity,
                None if self.threshold is None else similarity >= self.threshold,
            ))
        ranked.sort(key=lambda item: (-item.similarity, item.entry_id))
        eligible = [item for item in ranked if item.above_threshold is not False]
        if not eligible and ranked:
            # Keep one closest graph for context, clearly marked below threshold.
            selected = (ranked[0],)
        else:
            counts = {label: 0 for label in MemoryLabel}
            selected = []
            for item in eligible:
                if counts[item.label] < self.top_k_per_label:
                    selected.append(item)
                    counts[item.label] += 1
            selected = tuple(selected)
        return RetrievalResult(
            matches=selected, threshold=self.threshold,
            no_strong_match=bool(self.threshold is not None and not eligible),
            schema_id=self.schema.schema_id,
        )
