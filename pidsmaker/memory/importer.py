"""Check proposed memory entries against separately prepared evidence manifests.

The manifest is an input from an independently curated source, not something
the detector or this importer should generate from its own prediction. These
checks establish correspondence, not the authenticity of the source itself.
"""

import hashlib
import json
from dataclasses import dataclass, field
from numbers import Integral
from pathlib import Path

import networkx as nx

from pidsmaker.memory.models import MemoryEntry, MemoryLabel


def _identity(value):
    """Give common provenance IDs an unambiguous, stable JSON representation."""
    if isinstance(value, bool):
        raise ValueError("Boolean provenance identifiers are not supported")
    if isinstance(value, Integral):
        return ["int", int(value)]
    if isinstance(value, str):
        return ["str", value]
    if value is None:
        return ["null"]
    if isinstance(value, tuple):
        return ["tuple", [_identity(part) for part in value]]
    raise ValueError(f"Unsupported provenance identifier type: {type(value).__name__}")


def evidence_fingerprint(graph: nx.MultiDiGraph) -> str:
    """Hash exact nodes and event identities, excluding detector anomaly scores."""
    if not isinstance(graph, nx.MultiDiGraph):
        raise ValueError("Evidence fingerprint needs a provenance MultiDiGraph")
    nodes = sorted(json.dumps(_identity(node), separators=(",", ":")) for node in graph)
    events = []
    for src, dst, key, data in graph.edges(keys=True, data=True):
        timestamp = data.get("time")
        if isinstance(timestamp, bool) or not isinstance(timestamp, Integral):
            raise ValueError("Every event needs an integer nanosecond time")
        if "edge_type" not in data and "label" not in data:
            raise ValueError("Every event needs an edge type or relation label")
        event = [
            _identity(src), _identity(dst), _identity(key), int(timestamp),
            _identity(data["edge_type"]) if "edge_type" in data else None,
            _identity(data["label"]) if "label" in data else None,
            _identity(data["event_uuid"]) if "event_uuid" in data else None,
        ]
        events.append(json.dumps(event, separators=(",", ":"), ensure_ascii=False))
    payload = json.dumps(
        {"nodes": nodes, "events": sorted(events)},
        separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class AdmittedEntry:
    """Entry that passed a manifest consistency check, ready for a later store."""

    entry: MemoryEntry
    manifest_sha256: str
    manifest_bytes: bytes = field(repr=False)


def import_verified_entry(entry: MemoryEntry, manifest_path) -> AdmittedEntry:
    """Admit an entry only when a separate manifest matches its evidence.

    An evidence curator must establish the manifest's origin independently.
    Matching a self-written manifest does not prove an attack or benign status.
    """
    return _admit_manifest_bytes(entry, Path(manifest_path).read_bytes())


def _admit_manifest_bytes(entry: MemoryEntry, raw: bytes) -> AdmittedEntry:
    """Validate a manifest; also used when rereading an archived store entry."""
    if not isinstance(entry, MemoryEntry):
        raise ValueError("Expected a proposed MemoryEntry")
    # Revalidate and detach the current candidate state: MemoryEntry is mutable.
    entry = MemoryEntry(
        entry_id=entry.entry_id, graph=entry.graph, label=entry.label,
        source_id=entry.source_id, host_id=entry.host_id,
        verification=entry.verification,
        attack_instance_id=entry.attack_instance_id,
        techniques=entry.techniques,
    )
    try:
        manifest = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Evidence manifest is not valid JSON") from exc
    if (not isinstance(manifest, dict)
            or type(manifest.get("schema_version")) is not int
            or manifest["schema_version"] != 1):
        raise ValueError("Unsupported evidence manifest schema")

    expected = {
        "entry_id": entry.entry_id,
        "label": entry.label.value,
        "source_id": entry.source_id,
        "host_id": entry.host_id,
        "attack_instance_id": entry.attack_instance_id,
        "techniques": sorted(entry.techniques),
        "verification": {
            "method": entry.verification.method.value,
            "evidence_ref": entry.verification.evidence_ref,
            "authority": entry.verification.authority,
        },
        "evidence_fingerprint": evidence_fingerprint(entry.graph),
    }
    missing = set(expected) | {"confirmed_attack_nodes"}
    missing -= manifest.keys()
    if missing:
        raise ValueError(f"Evidence manifest is missing fields: {sorted(missing)}")
    for field, value in expected.items():
        if manifest.get(field) != value:
            raise ValueError(f"Evidence manifest {field} does not match entry")

    attack_nodes = manifest.get("confirmed_attack_nodes")
    if (not isinstance(attack_nodes, list)
            or any(isinstance(node, bool) or not isinstance(node, (int, str))
                   for node in attack_nodes)
            or len(attack_nodes) != len(set(attack_nodes))):
        raise ValueError("confirmed_attack_nodes must be distinct provenance node IDs")
    if entry.label is MemoryLabel.MALICIOUS:
        if not entry.attack_instance_id or not attack_nodes:
            raise ValueError("Malicious evidence needs an attack instance and confirmed nodes")
        if not set(attack_nodes).issubset(entry.graph.nodes):
            raise ValueError("Confirmed attack nodes are absent from the incident graph")
    elif entry.attack_instance_id is not None or attack_nodes:
        raise ValueError("Benign evidence cannot claim an attack instance or attack nodes")

    return AdmittedEntry(
        entry=entry,
        manifest_sha256=hashlib.sha256(raw).hexdigest(),
        manifest_bytes=raw,
    )
