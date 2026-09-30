"""Small append-only SQLite store for independently admitted incidents.

No pickle is used. Unsupported graph attributes fail explicitly instead of
being silently stringified or dropped. The API never updates or deletes rows.
"""

import hashlib
import json
import math
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path
from typing import Optional

import networkx as nx

from pidsmaker.memory.importer import (
    _admit_manifest_bytes, evidence_fingerprint, import_verified_entry,
)
from pidsmaker.memory.models import (
    MemoryEntry, MemoryLabel, VerificationMethod, VerificationRecord,
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS memory_entries (
    entry_id TEXT PRIMARY KEY,
    label TEXT NOT NULL,
    source_id TEXT NOT NULL,
    host_id TEXT NOT NULL,
    attack_instance_id TEXT,
    metadata_json TEXT NOT NULL,
    graph_json TEXT NOT NULL,
    graph_sha256 TEXT NOT NULL,
    manifest_bytes BLOB NOT NULL,
    manifest_sha256 TEXT NOT NULL
)
"""


def _json_text(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False)


def _sha256(value):
    return hashlib.sha256(value).hexdigest()


def _encode(value):
    """Tag Python types so JSON round-trips IDs, keys, and graph attributes."""
    if value is None:
        return ["none", None]
    if isinstance(value, bool):
        return ["bool", value]
    if isinstance(value, Integral):
        return ["int", int(value)]
    if isinstance(value, Real):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("Nonfinite graph attributes cannot be stored")
        return ["float", number]
    if isinstance(value, str):
        return ["str", value]
    if isinstance(value, tuple):
        return ["tuple", [_encode(item) for item in value]]
    if isinstance(value, list):
        return ["list", [_encode(item) for item in value]]
    if isinstance(value, dict):
        pairs = [[_encode(key), _encode(item)] for key, item in value.items()]
        pairs.sort(key=lambda pair: _json_text(pair[0]))
        return ["dict", pairs]
    raise ValueError(f"Unsupported graph attribute type: {type(value).__name__}")


def _decode(encoded):
    if not isinstance(encoded, list) or len(encoded) != 2:
        raise ValueError("Malformed stored graph value")
    kind, value = encoded
    if kind == "none" and value is None:
        return None
    if kind == "bool" and type(value) is bool:
        return value
    if kind == "int" and type(value) is int:
        return value
    if kind == "float" and type(value) in (float, int) and math.isfinite(value):
        return float(value)
    if kind == "str" and isinstance(value, str):
        return value
    if kind in ("tuple", "list") and isinstance(value, list):
        parts = [_decode(item) for item in value]
        return tuple(parts) if kind == "tuple" else parts
    if kind == "dict" and isinstance(value, list):
        result = {}
        for pair in value:
            if not isinstance(pair, list) or len(pair) != 2:
                raise ValueError("Malformed stored graph dictionary")
            key, item = _decode(pair[0]), _decode(pair[1])
            try:
                if key in result:
                    raise ValueError("Duplicate stored graph attribute")
                result[key] = item
            except TypeError as exc:
                raise ValueError("Unhashable stored graph attribute key") from exc
        return result
    raise ValueError("Malformed stored graph value")


def _graph_to_json(graph):
    nodes = [[_encode(node), _encode(data)] for node, data in graph.nodes(data=True)]
    edges = [
        [_encode(src), _encode(dst), _encode(key), _encode(data)]
        for src, dst, key, data in graph.edges(keys=True, data=True)
    ]
    nodes.sort(key=lambda row: _json_text(row[0]))
    edges.sort(key=lambda row: _json_text(row[:3]))
    return _json_text({
        "schema_version": 1,
        "graph_attributes": _encode(graph.graph),
        "nodes": nodes,
        "events": edges,
    })


def _graph_from_json(text):
    try:
        payload = json.loads(text)
        if (not isinstance(payload, dict)
                or type(payload.get("schema_version")) is not int
                or payload["schema_version"] != 1
                or not isinstance(payload.get("nodes"), list)
                or not isinstance(payload.get("events"), list)):
            raise ValueError("Unsupported stored graph schema")
        graph = nx.MultiDiGraph()
        attributes = _decode(payload["graph_attributes"])
        if not isinstance(attributes, dict):
            raise ValueError("Malformed stored graph attributes")
        graph.graph.update(attributes)
        for row in payload["nodes"]:
            if not isinstance(row, list) or len(row) != 2:
                raise ValueError("Malformed stored graph node")
            node, data = _decode(row[0]), _decode(row[1])
            if not isinstance(data, dict) or node in graph:
                raise ValueError("Malformed or duplicate stored graph node")
            graph.add_node(node)
            graph.nodes[node].update(data)
        for row in payload["events"]:
            if not isinstance(row, list) or len(row) != 4:
                raise ValueError("Malformed stored graph event")
            src, dst, key, data = (_decode(item) for item in row)
            if (not isinstance(data, dict) or src not in graph or dst not in graph
                    or graph.has_edge(src, dst, key)):
                raise ValueError("Malformed or duplicate stored graph event")
            graph.add_edge(src, dst, key=key)
            graph[src][dst][key].update(data)
        return graph
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError("Malformed stored graph JSON") from exc


def _entry_metadata(entry):
    return {
        "entry_id": entry.entry_id,
        "label": entry.label.value,
        "source_id": entry.source_id,
        "host_id": entry.host_id,
        "attack_instance_id": entry.attack_instance_id,
        "techniques": list(entry.techniques),
        "verification": {
            "method": entry.verification.method.value,
            "evidence_ref": entry.verification.evidence_ref,
            "authority": entry.verification.authority,
        },
    }


def _entry_from_json(metadata_json, graph_json):
    try:
        metadata = json.loads(metadata_json)
        verification = metadata["verification"]
        return MemoryEntry(
            entry_id=metadata["entry_id"], graph=_graph_from_json(graph_json),
            label=MemoryLabel(metadata["label"]),
            source_id=metadata["source_id"], host_id=metadata["host_id"],
            verification=VerificationRecord(
                method=VerificationMethod(verification["method"]),
                evidence_ref=verification["evidence_ref"],
                authority=verification["authority"],
            ),
            attack_instance_id=metadata["attack_instance_id"],
            techniques=tuple(metadata["techniques"]),
        )
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError("Malformed stored memory metadata") from exc


@dataclass(frozen=True)
class MemoryEntrySummary:
    entry_id: str
    label: MemoryLabel
    source_id: str
    host_id: str
    attack_instance_id: Optional[str]


class IncidentMemoryStore:
    """Insert-only API for local, verified incident references."""

    def __init__(self, database_path="artifacts/incident_memory.sqlite3"):
        self.path = Path(database_path)

    def _connect(self, *, create):
        if create:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        elif not self.path.is_file():
            raise FileNotFoundError(self.path)
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        if create:
            connection.execute(_SCHEMA)
        return connection

    def add(self, entry: MemoryEntry, manifest_path):
        """Recheck independent evidence and insert a new entry; never replace."""
        admitted = import_verified_entry(entry, manifest_path)
        entry = admitted.entry
        graph_json = _graph_to_json(entry.graph)
        if evidence_fingerprint(_graph_from_json(graph_json)) != evidence_fingerprint(entry.graph):
            raise ValueError("Graph serialization changed provenance event identity")
        metadata_json = _json_text(_entry_metadata(entry))
        with closing(self._connect(create=True)) as connection:
            with connection:
                try:
                    connection.execute(
                        """INSERT INTO memory_entries
                        (entry_id, label, source_id, host_id, attack_instance_id,
                         metadata_json, graph_json, graph_sha256,
                         manifest_bytes, manifest_sha256)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (
                            entry.entry_id, entry.label.value, entry.source_id,
                            entry.host_id, entry.attack_instance_id, metadata_json,
                            graph_json, _sha256(graph_json.encode("utf-8")),
                            admitted.manifest_bytes, admitted.manifest_sha256,
                        ),
                    )
                except sqlite3.IntegrityError as exc:
                    raise ValueError(f"Memory entry ID already exists: {entry.entry_id}") from exc
        return entry.entry_id

    def get(self, entry_id):
        """Load and recheck the archived graph and manifest."""
        with closing(self._connect(create=False)) as connection:
            row = connection.execute(
                "SELECT * FROM memory_entries WHERE entry_id = ?", (entry_id,)
            ).fetchone()
        if row is None:
            raise KeyError(entry_id)
        if (not isinstance(row["graph_json"], str)
                or _sha256(row["graph_json"].encode("utf-8")) != row["graph_sha256"]):
            raise ValueError("Stored graph checksum does not match")
        if not isinstance(row["manifest_bytes"], bytes):
            raise ValueError("Stored evidence manifest checksum does not match")
        manifest_bytes = row["manifest_bytes"]
        if _sha256(manifest_bytes) != row["manifest_sha256"]:
            raise ValueError("Stored evidence manifest checksum does not match")
        entry = _entry_from_json(row["metadata_json"], row["graph_json"])
        if (entry.entry_id != row["entry_id"] or entry.label.value != row["label"]
                or entry.source_id != row["source_id"] or entry.host_id != row["host_id"]
                or entry.attack_instance_id != row["attack_instance_id"]):
            raise ValueError("Stored memory metadata columns do not match")
        return _admit_manifest_bytes(entry, manifest_bytes).entry

    def list_entries(self):
        """Return metadata only; validate each stored entry before listing it."""
        with closing(self._connect(create=False)) as connection:
            identifiers = [row[0] for row in connection.execute(
                "SELECT entry_id FROM memory_entries ORDER BY entry_id"
            )]
        entries = (self.get(entry_id) for entry_id in identifiers)
        return tuple(
            MemoryEntrySummary(
                entry_id=entry.entry_id, label=entry.label,
                source_id=entry.source_id, host_id=entry.host_id,
                attack_instance_id=entry.attack_instance_id,
            )
            for entry in entries
        )
