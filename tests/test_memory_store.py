import json
import sqlite3

import networkx as nx
import pytest

from pidsmaker.memory import (
    IncidentMemoryStore, MemoryEntry, MemoryLabel, VerificationMethod,
    VerificationRecord, evidence_fingerprint,
)


def proposed_entry(entry_id="case-1", label=MemoryLabel.MALICIOUS):
    graph = nx.MultiDiGraph()
    graph.graph["region"] = "example"
    graph.add_node(10, node_type="process", label="example process")
    graph.add_node(20, node_type="file", details={"path": "/synthetic/file"})
    graph.add_edge(
        10, 20, key=("event", 1), time=1_700_000_000_000_000_000,
        edge_type=1, label="EVENT_WRITE", score=10.0,
        tags=["synthetic", "local"],
    )
    method = (VerificationMethod.DATASET_GROUND_TRUTH
              if label is MemoryLabel.MALICIOUS else VerificationMethod.ANALYST_REVIEW)
    return MemoryEntry(
        entry_id=entry_id, graph=graph, label=label,
        source_id="example-engagement", host_id="example-host",
        verification=VerificationRecord(
            method=method, evidence_ref="annotations/example-1",
            authority="independent test fixture",
        ),
        attack_instance_id="attack-1" if label is MemoryLabel.MALICIOUS else None,
        techniques=("T1059",) if label is MemoryLabel.MALICIOUS else (),
    )


def manifest_for(entry):
    return {
        "schema_version": 1,
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
        "confirmed_attack_nodes": [10] if entry.label is MemoryLabel.MALICIOUS else [],
    }


def manifest_path(tmp_path, entry):
    path = tmp_path / f"{entry.entry_id}.json"
    path.write_text(json.dumps(manifest_for(entry)))
    return path


def test_store_round_trips_malicious_and_benign_entries_with_exact_graph(tmp_path):
    store = IncidentMemoryStore(tmp_path / "memory.sqlite3")
    malicious = proposed_entry()
    benign = proposed_entry("case-2", MemoryLabel.BENIGN)
    assert store.add(malicious, manifest_path(tmp_path, malicious)) == "case-1"
    assert store.add(benign, manifest_path(tmp_path, benign)) == "case-2"

    loaded = store.get("case-1")
    assert loaded.label is MemoryLabel.MALICIOUS
    assert loaded.graph.graph["region"] == "example"
    assert loaded.graph.nodes[20]["details"] == {"path": "/synthetic/file"}
    assert loaded.graph.has_edge(10, 20, ("event", 1))
    assert loaded.graph[10][20][("event", 1)]["time"] == 1_700_000_000_000_000_000
    assert loaded.graph[10][20][("event", 1)]["tags"] == ["synthetic", "local"]
    assert store.get("case-2").label is MemoryLabel.BENIGN
    assert [item.entry_id for item in store.list_entries()] == ["case-1", "case-2"]
    assert store.list_entries()[0].attack_instance_id == "attack-1"


def test_store_refuses_duplicate_id_without_replacing_original(tmp_path):
    store = IncidentMemoryStore(tmp_path / "memory.sqlite3")
    first = proposed_entry()
    store.add(first, manifest_path(tmp_path, first))
    other = proposed_entry("case-1", MemoryLabel.BENIGN)
    with pytest.raises(ValueError, match="already exists"):
        store.add(other, manifest_path(tmp_path, other))
    assert store.get("case-1").label is MemoryLabel.MALICIOUS
    assert len(store.list_entries()) == 1


def test_store_rechecks_manifest_before_creating_database(tmp_path):
    database = tmp_path / "memory.sqlite3"
    store = IncidentMemoryStore(database)
    entry = proposed_entry()
    path = tmp_path / "wrong.json"
    manifest = manifest_for(entry)
    manifest["host_id"] = "wrong-host"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="host_id"):
        store.add(entry, path)
    assert not database.exists()


def test_store_rejects_unsupported_graph_attribute_without_data_loss(tmp_path):
    database = tmp_path / "memory.sqlite3"
    store = IncidentMemoryStore(database)
    entry = proposed_entry()
    entry.graph.nodes[10]["unsupported"] = object()
    with pytest.raises(ValueError, match="Unsupported graph attribute"):
        store.add(entry, manifest_path(tmp_path, entry))
    assert not database.exists()


def test_stored_graph_is_detached_from_the_original_candidate(tmp_path):
    store = IncidentMemoryStore(tmp_path / "memory.sqlite3")
    entry = proposed_entry()
    store.add(entry, manifest_path(tmp_path, entry))
    entry.graph.remove_edge(10, 20, ("event", 1))
    assert store.get("case-1").graph.number_of_edges() == 1


def test_store_keeps_the_checked_manifest_after_source_file_is_removed(tmp_path):
    store = IncidentMemoryStore(tmp_path / "memory.sqlite3")
    entry = proposed_entry()
    path = manifest_path(tmp_path, entry)
    store.add(entry, path)
    path.unlink()
    assert store.get("case-1").verification.evidence_ref == "annotations/example-1"


@pytest.mark.parametrize("column", ["graph_json", "manifest_bytes"])
def test_store_detects_modified_archived_evidence(tmp_path, column):
    database = tmp_path / "memory.sqlite3"
    store = IncidentMemoryStore(database)
    entry = proposed_entry()
    store.add(entry, manifest_path(tmp_path, entry))
    with sqlite3.connect(database) as connection:
        connection.execute(
            f"UPDATE memory_entries SET {column} = ? WHERE entry_id = ?",
            ("{}", "case-1"),
        )
    with pytest.raises(ValueError, match="checksum"):
        store.get("case-1")


def test_store_read_does_not_create_database_and_unknown_id_is_clear(tmp_path):
    database = tmp_path / "memory.sqlite3"
    store = IncidentMemoryStore(database)
    with pytest.raises(FileNotFoundError):
        store.list_entries()
    assert not database.exists()
    entry = proposed_entry()
    store.add(entry, manifest_path(tmp_path, entry))
    with pytest.raises(KeyError):
        store.get("missing")
