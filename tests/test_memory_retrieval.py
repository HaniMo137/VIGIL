import json

import networkx as nx
import pytest
import torch

from pidsmaker.encoders.vigil_encoder import VigilEncoder
from pidsmaker.incidents.demo_data import create_demo
from pidsmaker.incidents.viewer import create_app
from pidsmaker.memory import (
    IncidentMemoryStore, IncidentRetriever, MemoryEntry, MemoryLabel,
    SignatureSchema, VerificationMethod, VerificationRecord,
    encoder_state_fingerprint, evidence_fingerprint,
)


def reference(entry_id, label, relation=1):
    graph = nx.MultiDiGraph()
    graph.add_node(1, node_type="process")
    graph.add_node(2, node_type="file")
    graph.add_edge(1, 2, key="event", time=10, edge_type=relation)
    return MemoryEntry(
        entry_id=entry_id, graph=graph, label=label, source_id="test-source",
        host_id="test-host",
        verification=VerificationRecord(
            method=VerificationMethod.ANALYST_REVIEW,
            evidence_ref="independent-test-record", authority="test analyst",
        ),
        attack_instance_id="attack-1" if label is MemoryLabel.MALICIOUS else None,
    )


def admit(store, tmp_path, entry):
    manifest = {
        "schema_version": 1, "entry_id": entry.entry_id, "label": entry.label.value,
        "source_id": entry.source_id, "host_id": entry.host_id,
        "attack_instance_id": entry.attack_instance_id, "techniques": [],
        "verification": {
            "method": entry.verification.method.value,
            "evidence_ref": entry.verification.evidence_ref,
            "authority": entry.verification.authority,
        },
        "evidence_fingerprint": evidence_fingerprint(entry.graph),
        "confirmed_attack_nodes": [1] if entry.label is MemoryLabel.MALICIOUS else [],
    }
    path = tmp_path / (entry.entry_id + ".json")
    path.write_text(json.dumps(manifest))
    store.add(entry, path)


def setup(tmp_path):
    torch.manual_seed(11)
    model = VigilEncoder(in_dim=3, hid_dim=5, out_dim=2)
    model.eval()
    schema = SignatureSchema(
        semantic_dim=5, node_types=("process", "file"), relation_types=(1, 2),
        encoder_checkpoint_id=encoder_state_fingerprint(model),
    )
    store = IncidentMemoryStore(tmp_path / "memory.sqlite3")
    malicious = reference("malicious-1", MemoryLabel.MALICIOUS)
    benign = reference("benign-1", MemoryLabel.BENIGN)
    admit(store, tmp_path, malicious)
    admit(store, tmp_path, benign)
    features = {1: [1.0, 0.0, 0.0], 2: [0.0, 1.0, 0.0]}
    return store, model, schema, features, malicious.graph


def test_retrieval_shows_both_verified_labels_without_auto_decision(tmp_path):
    store, model, schema, features, query = setup(tmp_path)
    retriever = IncidentRetriever(
        store, {"malicious-1": features, "benign-1": features},
        model, schema, threshold=0.99,
    )
    result = retriever.search(query, features)
    assert [item.entry_id for item in result.matches] == ["benign-1", "malicious-1"]
    assert {item.label for item in result.matches} == {MemoryLabel.MALICIOUS, MemoryLabel.BENIGN}
    assert all(item.above_threshold for item in result.matches)
    assert all(item.similarity == pytest.approx(1.0) for item in result.matches)
    assert not result.no_strong_match
    assert result.schema_id == schema.schema_id


def test_below_threshold_returns_only_closest_context_with_clear_flag(tmp_path):
    store, model, schema, features, query = setup(tmp_path)
    retriever = IncidentRetriever(
        store, {"malicious-1": features, "benign-1": features},
        model, schema, threshold=1.0,
    )
    different = query.copy()
    different.nodes[1]["node_type"] = "socket"
    different[1][2]["event"]["edge_type"] = 2
    result = retriever.search(different, features)
    assert result.no_strong_match
    assert len(result.matches) == 1
    assert result.matches[0].above_threshold is False
    assert result.matches[0].similarity < 1.0


def test_unset_threshold_returns_ranked_examples_without_a_strong_claim(tmp_path):
    store, model, schema, features, query = setup(tmp_path)
    retriever = IncidentRetriever(
        store, {"malicious-1": features, "benign-1": features}, model, schema,
    )
    result = retriever.search(query, features)
    assert len(result.matches) == 2
    assert result.threshold is None
    assert all(item.above_threshold is None for item in result.matches)
    assert not result.no_strong_match


def test_top_k_is_applied_per_label_not_globally(tmp_path):
    store, model, schema, features, query = setup(tmp_path)
    admit(store, tmp_path, reference("malicious-2", MemoryLabel.MALICIOUS))
    admit(store, tmp_path, reference("benign-2", MemoryLabel.BENIGN))
    retriever = IncidentRetriever(
        store, {entry_id: features for entry_id in
                ("malicious-1", "malicious-2", "benign-1", "benign-2")},
        model, schema, threshold=0.99, top_k_per_label=1,
    )
    result = retriever.search(query, features)
    assert [item.entry_id for item in result.matches] == ["benign-1", "malicious-1"]


def test_retriever_rechecks_store_and_does_not_accept_unadmitted_reference(tmp_path):
    store, model, schema, features, query = setup(tmp_path)
    with pytest.raises(ValueError, match="Missing features"):
        IncidentRetriever(store, {"malicious-1": features}, model, schema)
    retriever = IncidentRetriever(
        store, {"malicious-1": features, "benign-1": features,
                "not-admitted": features}, model, schema,
    )
    with pytest.raises(KeyError):
        retriever.reference_graph("not-admitted")
    assert retriever.reference_graph("malicious-1").number_of_edges() == 1


@pytest.mark.parametrize("kwargs", [
    {"threshold": float("nan")}, {"threshold": 1.01},
    {"top_k_per_label": 0}, {"top_k_per_label": True},
])
def test_retriever_rejects_invalid_search_controls(tmp_path, kwargs):
    store, model, schema, features, _ = setup(tmp_path)
    with pytest.raises(ValueError):
        IncidentRetriever(
            store, {"malicious-1": features, "benign-1": features},
            model, schema, **kwargs,
        )


def test_changed_encoder_weights_refuse_search(tmp_path):
    store, model, schema, features, query = setup(tmp_path)
    retriever = IncidentRetriever(
        store, {"malicious-1": features, "benign-1": features}, model, schema,
    )
    with torch.no_grad():
        next(model.parameters()).add_(0.1)
    with pytest.raises(ValueError, match="checkpoint"):
        retriever.search(query, features)


def test_viewer_shows_verified_matches_and_bounded_reference_graph(tmp_path):
    store, model, schema, features, _ = setup(tmp_path)
    retriever = IncidentRetriever(
        store, {"malicious-1": features, "benign-1": features},
        model, schema, threshold=0.0,
    )
    demo = create_demo(tmp_path / "demo", background_chains=1,
                       case_count=1, chain_edges=8)
    paths = (demo / "incidents.json", demo / "original_graph.pt",
             demo / "event_scores.csv")
    plain = create_app(*paths)
    plain.testing = True
    assert plain.test_client().get("/api/summary").get_json()["memory_enabled"] is False
    assert plain.test_client().get("/api/memory/0").status_code == 404

    node_ids = plain.config["VIEWER_DATA"].graph.nodes
    query_features = {0: {node: [1.0, 0.0, 0.0] for node in node_ids}}
    app = create_app(*paths, memory_retriever=retriever,
                     incident_node_features=query_features, max_edges=1)
    app.testing = True
    client = app.test_client()
    assert client.get("/api/summary").get_json()["memory_enabled"] is True
    matches = client.get("/api/memory/0").get_json()
    assert {item["label"] for item in matches["matches"]} == {"malicious", "benign"}
    assert all(item["above_threshold"] for item in matches["matches"])
    graph = client.get("/api/memory/0/graph?entry_id=malicious-1").get_json()
    assert len(graph["edges"]) == 1
    assert graph["edges"][0]["role"] == "reference"
    assert graph["nodes"][0]["original_id"] in {"1", "2"}
    assert client.get("/api/memory/0/graph?entry_id=not-admitted").status_code == 404
    assert client.get("/api/memory/9").status_code == 404


def test_viewer_rejects_partial_memory_configuration(tmp_path):
    store, model, schema, features, _ = setup(tmp_path)
    retriever = IncidentRetriever(
        store, {"malicious-1": features, "benign-1": features}, model, schema,
    )
    demo = create_demo(tmp_path / "demo", background_chains=1,
                       case_count=1, chain_edges=8)
    paths = (demo / "incidents.json", demo / "original_graph.pt",
             demo / "event_scores.csv")
    with pytest.raises(ValueError, match="both"):
        create_app(*paths, memory_retriever=retriever)
