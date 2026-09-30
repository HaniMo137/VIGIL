"""Memory-entry schema; metadata alone does not establish factual verification.

Builder incidents remain unverified candidates. An importer must check the
independent evidence named by ``VerificationRecord`` before admitting an entry.
Storage, signatures, and retrieval are separate later stages.
"""

from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from numbers import Integral
from typing import Optional, Tuple

import networkx as nx


class MemoryLabel(str, Enum):
    MALICIOUS = "malicious"
    BENIGN = "benign"


class VerificationMethod(str, Enum):
    DATASET_GROUND_TRUTH = "dataset_ground_truth"
    CONTROLLED_SIMULATION = "controlled_simulation"
    ANALYST_REVIEW = "analyst_review"


def _require_text(value, field):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a nonempty string")


@dataclass(frozen=True)
class VerificationRecord:
    """Traceable claim about independent evidence, not a detector prediction."""

    method: VerificationMethod
    evidence_ref: str
    authority: str

    def __post_init__(self):
        if not isinstance(self.method, VerificationMethod):
            raise ValueError("Verification method must be an approved independent source")
        _require_text(self.evidence_ref, "evidence_ref")
        _require_text(self.authority, "authority")


@dataclass
class MemoryEntry:
    """A proposed trusted reference with preserved provenance and label source.

    This model checks the entry's shape. It cannot establish whether the
    referenced ground truth or review is genuine; the importer must do that.
    """

    entry_id: str
    graph: nx.MultiDiGraph
    label: MemoryLabel
    source_id: str
    host_id: str
    verification: VerificationRecord
    attack_instance_id: Optional[str] = None
    techniques: Tuple[str, ...] = ()

    def __post_init__(self):
        for name in ("entry_id", "source_id", "host_id"):
            _require_text(getattr(self, name), name)
        if self.attack_instance_id is not None:
            _require_text(self.attack_instance_id, "attack_instance_id")
        if not isinstance(self.label, MemoryLabel):
            raise ValueError("Memory label must be explicitly malicious or benign")
        if not isinstance(self.verification, VerificationRecord):
            raise ValueError("A verification record is required for memory entry")
        if not isinstance(self.graph, nx.MultiDiGraph) or not self.graph.number_of_edges():
            raise ValueError("Memory entry needs a nonempty provenance MultiDiGraph")
        for _, _, _, data in self.graph.edges(keys=True, data=True):
            timestamp = data.get("time")
            if isinstance(timestamp, bool) or not isinstance(timestamp, Integral):
                raise ValueError("Every memory event needs an integer nanosecond time")
            if "edge_type" not in data and "label" not in data:
                raise ValueError("Every memory event needs an edge type or relation label")
        if (not isinstance(self.techniques, tuple)
                or any(not isinstance(item, str) or not item.strip() for item in self.techniques)
                or len(set(self.techniques)) != len(self.techniques)):
            raise ValueError("techniques must be a tuple of distinct nonempty strings")
        # Do not share the builder's mutable graph with the proposed entry.
        self.graph = deepcopy(self.graph)
