import json

import networkx as nx
import pytest

from pidsmaker.memory import (
    MemoryEntry, MemoryLabel, VerificationMethod, VerificationRecord,
    evidence_fingerprint, import_verified_entry,
)


def proposed_entry(label=MemoryLabel.MALICIOUS, attack_instance_id="attack-1"):
    graph = nx.MultiDiGraph()
    graph.add_edge(10, 20, key=3, time=1_700_000_000_000_000_000,
                   edge_type=1, label="EVENT_WRITE", score=10.0)
    return MemoryEntry(
        entry_id="reference-1", graph=graph, label=label,
        source_id="engagement-1", host_id="host-1",
        verification=VerificationRecord(
            method=VerificationMethod.DATASET_GROUND_TRUTH,
            evidence_ref="ground-truth/attack-1", authority="dataset annotation",
        ),
        attack_instance_id=attack_instance_id,
        techniques=("T1059",) if label is MemoryLabel.MALICIOUS else (),
    )


def evidence_manifest(entry, attack_nodes=None):
    if attack_nodes is None:
        attack_nodes = [10] if entry.label is MemoryLabel.MALICIOUS else []
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
        "confirmed_attack_nodes": attack_nodes,
    }


def save_manifest(tmp_path, manifest):
    path = tmp_path / "independent_evidence.json"
    path.write_text(json.dumps(manifest))
    return path


def test_importer_admits_matching_malicious_evidence_and_snapshots_it(tmp_path):
    entry = proposed_entry()
    path = save_manifest(tmp_path, evidence_manifest(entry))
    admitted = import_verified_entry(entry, path)
    assert admitted.entry.label is MemoryLabel.MALICIOUS
    assert admitted.entry.graph.number_of_edges() == 1
    assert len(admitted.manifest_sha256) == 64
    entry.graph.remove_edge(10, 20, key=3)
    assert admitted.entry.graph.number_of_edges() == 1


def test_importer_admits_explicitly_reviewed_benign_evidence(tmp_path):
    entry = proposed_entry(MemoryLabel.BENIGN, attack_instance_id=None)
    entry.verification = VerificationRecord(
        method=VerificationMethod.ANALYST_REVIEW,
        evidence_ref="review/benign-1", authority="analyst-1",
    )
    admitted = import_verified_entry(entry, save_manifest(tmp_path, evidence_manifest(entry)))
    assert admitted.entry.label is MemoryLabel.BENIGN


@pytest.mark.parametrize("field,changed", [
    ("entry_id", "other-entry"), ("label", "benign"),
    ("source_id", "other-engagement"), ("host_id", "other-host"),
    ("attack_instance_id", "other-attack"), ("techniques", ["T9999"]),
    ("evidence_fingerprint", "0" * 64),
])
def test_importer_rejects_mismatched_manifest_fields(tmp_path, field, changed):
    entry = proposed_entry()
    manifest = evidence_manifest(entry)
    manifest[field] = changed
    with pytest.raises(ValueError, match=field):
        import_verified_entry(entry, save_manifest(tmp_path, manifest))


def test_importer_rejects_changed_verification_claim(tmp_path):
    entry = proposed_entry()
    manifest = evidence_manifest(entry)
    manifest["verification"]["evidence_ref"] = "detector/score-10"
    with pytest.raises(ValueError, match="verification"):
        import_verified_entry(entry, save_manifest(tmp_path, manifest))


@pytest.mark.parametrize("attack_nodes", [[], [99], [10, 10], [True]])
def test_malicious_entry_needs_distinct_confirmed_nodes_in_its_graph(
    tmp_path, attack_nodes,
):
    entry = proposed_entry()
    with pytest.raises(ValueError, match="confirmed|attack nodes"):
        import_verified_entry(
            entry, save_manifest(tmp_path, evidence_manifest(entry, attack_nodes))
        )


def test_malicious_entry_needs_named_attack_instance(tmp_path):
    entry = proposed_entry(attack_instance_id=None)
    with pytest.raises(ValueError, match="attack instance"):
        import_verified_entry(entry, save_manifest(tmp_path, evidence_manifest(entry)))


def test_benign_entry_cannot_claim_attack_nodes_or_instance(tmp_path):
    entry = proposed_entry(MemoryLabel.BENIGN, attack_instance_id=None)
    with pytest.raises(ValueError, match="Benign evidence"):
        import_verified_entry(
            entry, save_manifest(tmp_path, evidence_manifest(entry, [10]))
        )
    entry.attack_instance_id = "attack-1"
    with pytest.raises(ValueError, match="Benign evidence"):
        import_verified_entry(entry, save_manifest(tmp_path, evidence_manifest(entry)))


def test_fingerprint_tracks_exact_event_identity_but_not_detector_score():
    entry = proposed_entry()
    original = evidence_fingerprint(entry.graph)
    entry.graph[10][20][3]["score"] = 0.1
    assert evidence_fingerprint(entry.graph) == original
    entry.graph[10][20][3]["time"] += 1
    assert evidence_fingerprint(entry.graph) != original


def test_importer_rejects_invalid_manifest_file(tmp_path):
    entry = proposed_entry()
    path = tmp_path / "bad.json"
    path.write_text("not json")
    with pytest.raises(ValueError, match="not valid JSON"):
        import_verified_entry(entry, path)
    path.write_text(json.dumps({"schema_version": 2}))
    with pytest.raises(ValueError, match="schema"):
        import_verified_entry(entry, path)
    path.write_text(json.dumps({"schema_version": True}))
    with pytest.raises(ValueError, match="schema"):
        import_verified_entry(entry, path)


def test_importer_requires_explicit_null_for_benign_attack_instance(tmp_path):
    entry = proposed_entry(MemoryLabel.BENIGN, attack_instance_id=None)
    manifest = evidence_manifest(entry)
    del manifest["attack_instance_id"]
    with pytest.raises(ValueError, match="missing fields"):
        import_verified_entry(entry, save_manifest(tmp_path, manifest))


def test_importer_revalidates_mutated_candidate_before_admission(tmp_path):
    entry = proposed_entry()
    path = save_manifest(tmp_path, evidence_manifest(entry))
    entry.source_id = " "
    with pytest.raises(ValueError, match="source_id"):
        import_verified_entry(entry, path)
