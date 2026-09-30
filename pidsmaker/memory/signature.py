"""Fixed structural-semantic incident signature from a frozen encoder.

The semantic block pools VIGIL's shared node embeddings. Role-specific source
and destination heads are not mixed into a single node vector. The other
blocks are deterministic histograms with a fixed reference vocabulary.
"""

import hashlib
import json
import math
from dataclasses import dataclass
from numbers import Integral
from typing import Mapping, Tuple

import networkx as nx
import numpy as np
import torch

_MAX_PATH_PAIRS = 1_000_000


def _relation_token(value):
    if isinstance(value, bool):
        raise ValueError("Boolean relation IDs are not supported")
    if isinstance(value, Integral):
        return "id:" + str(int(value))
    if isinstance(value, str) and value.strip():
        return "label:" + value
    raise ValueError("Relations must have an integer ID or nonempty label")


def _edge_relation(data):
    if "edge_type" in data:
        return _relation_token(data["edge_type"])
    if "label" in data:
        return _relation_token(data["label"])
    raise ValueError("Every incident event needs a relation type")


def _unit(block):
    norm = np.linalg.norm(block)
    return block / norm if norm > 0 else block


def encoder_state_fingerprint(encoder) -> str:
    """SHA-256 of the encoder class and exact parameter/buffer state."""
    if not hasattr(encoder, "state_dict"):
        raise ValueError("Encoder must expose a state_dict")
    digest = hashlib.sha256()

    def update(value):
        data = value if isinstance(value, bytes) else value.encode("utf-8")
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)

    update(type(encoder).__module__ + "." + type(encoder).__qualname__)
    for name, value in sorted(encoder.state_dict().items()):
        if not isinstance(value, torch.Tensor) or value.is_sparse:
            raise ValueError("Unsupported encoder state value")
        update(name)
        update(str(value.dtype))
        update(json.dumps(list(value.shape), separators=(",", ":")))
        update(value.detach().to("cpu").contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class SignatureSchema:
    """Vocabulary and encoder checkpoint fixed before comparing incidents."""

    semantic_dim: int
    node_types: Tuple[str, ...]
    relation_types: Tuple[object, ...]
    encoder_checkpoint_id: str

    def __post_init__(self):
        if isinstance(self.semantic_dim, bool) or not isinstance(self.semantic_dim, Integral) or self.semantic_dim < 1:
            raise ValueError("semantic_dim must be a positive integer")
        if (not isinstance(self.encoder_checkpoint_id, str)
                or len(self.encoder_checkpoint_id) != 64
                or any(character not in "0123456789abcdef"
                       for character in self.encoder_checkpoint_id)):
            raise ValueError("Encoder checkpoint ID must be a SHA-256 state fingerprint")
        if not isinstance(self.node_types, tuple) or not isinstance(self.relation_types, tuple):
            raise ValueError("Signature vocabularies must be fixed tuples")
        if any(not isinstance(kind, str) or not kind.strip() for kind in self.node_types):
            raise ValueError("Node types must be nonempty strings")
        normalized_nodes = [kind.strip().lower() for kind in self.node_types]
        normalized_relations = [_relation_token(value) for value in self.relation_types]
        if (len(set(normalized_nodes)) != len(normalized_nodes)
                or len(set(normalized_relations)) != len(normalized_relations)):
            raise ValueError("Signature vocabularies must not contain duplicates")

    @property
    def schema_id(self):
        payload = {
            "version": 1,
            "semantic_dim": int(self.semantic_dim),
            "node_types": [kind.strip().lower() for kind in self.node_types],
            "relation_types": [_relation_token(value) for value in self.relation_types],
            "encoder_checkpoint_id": self.encoder_checkpoint_id,
            "path_rule": "directed_time_respecting_length_2",
            "block_weight": 0.5,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @property
    def dimension(self):
        relations = len(self.relation_types) + 1  # final bin is unknown
        return int(self.semantic_dim) + len(self.node_types) + 1 + relations + relations ** 2


@dataclass(frozen=True)
class IncidentSignature:
    vector: np.ndarray
    schema_id: str
    encoder_checkpoint_id: str


def build_incident_signature(
    graph: nx.MultiDiGraph,
    node_features: Mapping,
    encoder,
    schema: SignatureSchema,
) -> IncidentSignature:
    """Pool frozen encoder embeddings and count structural evidence.

    ``node_features`` maps every incident node ID to its Word2Vec feature
    vector. Both reference and query incidents must use the same feature
    pipeline, encoder checkpoint, and schema.
    """
    if not isinstance(graph, nx.MultiDiGraph) or not graph.number_of_edges():
        raise ValueError("Signature needs a nonempty provenance MultiDiGraph")
    if not isinstance(schema, SignatureSchema):
        raise ValueError("A fixed SignatureSchema is required")
    if not hasattr(encoder, "encode_nodes") or not hasattr(encoder, "training"):
        raise ValueError("Encoder must expose encode_nodes and eval state")
    if encoder.training:
        raise ValueError("Encoder must be in eval mode for a fixed signature")
    if encoder_state_fingerprint(encoder) != schema.encoder_checkpoint_id:
        raise ValueError("Encoder weights do not match the signature checkpoint ID")

    nodes = list(graph.nodes)
    features = []
    for node in nodes:
        try:
            value = node_features[node]
        except (KeyError, TypeError) as exc:
            raise ValueError(f"Missing node features for {node!r}") from exc
        try:
            feature = torch.as_tensor(value, dtype=torch.float32)
        except (TypeError, ValueError, RuntimeError) as exc:
            raise ValueError(f"Invalid node features for {node!r}") from exc
        if feature.ndim != 1 or not torch.isfinite(feature).all():
            raise ValueError(f"Node features for {node!r} must be a finite vector")
        features.append(feature)
    if len({feature.numel() for feature in features}) != 1:
        raise ValueError("All node feature vectors must have the same length")
    try:
        device = next(encoder.parameters()).device
    except StopIteration:
        device = torch.device("cpu")
    with torch.no_grad():
        embeddings = encoder.encode_nodes(torch.stack(features).to(device))
    if (not isinstance(embeddings, torch.Tensor) or embeddings.ndim != 2
            or embeddings.shape != (len(nodes), schema.semantic_dim)
            or not torch.isfinite(embeddings).all()):
        raise ValueError("Encoder returned an invalid shared node embedding")
    semantic = embeddings.detach().to(device="cpu", dtype=torch.float64).numpy().mean(axis=0)

    node_vocabulary = {kind.strip().lower(): index for index, kind in enumerate(schema.node_types)}
    relation_vocabulary = {
        _relation_token(value): index for index, value in enumerate(schema.relation_types)
    }
    unknown_node = len(node_vocabulary)
    unknown_relation = len(relation_vocabulary)
    relation_bins = unknown_relation + 1
    node_hist = np.zeros(unknown_node + 1, dtype=np.float64)
    relation_hist = np.zeros(relation_bins, dtype=np.float64)
    path_hist = np.zeros(relation_bins ** 2, dtype=np.float64)

    for node in nodes:
        kind = str(graph.nodes[node].get("node_type", "")).strip().lower()
        node_hist[node_vocabulary.get(kind, unknown_node)] += 1
    event_bins = {}
    for src, dst, key, data in graph.edges(keys=True, data=True):
        if isinstance(data.get("time"), bool) or not isinstance(data.get("time"), Integral):
            raise ValueError("Every incident event needs an integer nanosecond time")
        bin_index = relation_vocabulary.get(_edge_relation(data), unknown_relation)
        event_bins[(src, dst, key)] = bin_index
        relation_hist[bin_index] += 1

    examined = 0
    for middle in graph.nodes:
        for src, _, in_key, incoming in graph.in_edges(middle, keys=True, data=True):
            for _, dst, out_key, outgoing in graph.out_edges(middle, keys=True, data=True):
                examined += 1
                if examined > _MAX_PATH_PAIRS:
                    raise ValueError("Incident has too many two-event path pairs")
                if ((src, middle, in_key) == (middle, dst, out_key)
                        or incoming["time"] > outgoing["time"]):
                    continue
                first = event_bins[(src, middle, in_key)]
                second = event_bins[(middle, dst, out_key)]
                path_hist[first * relation_bins + second] += 1

    vector = np.concatenate([
        0.5 * _unit(semantic),
        0.5 * _unit(node_hist),
        0.5 * _unit(relation_hist),
        0.5 * _unit(path_hist),
    ])
    if vector.shape != (schema.dimension,) or not np.isfinite(vector).all():
        raise ValueError("Incident signature is not finite or has the wrong dimension")
    vector.setflags(write=False)
    return IncidentSignature(vector, schema.schema_id, schema.encoder_checkpoint_id)
