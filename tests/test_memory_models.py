import networkx as nx
import pytest

from pidsmaker.memory import (
    MemoryEntry, MemoryLabel, VerificationMethod, VerificationRecord,
)


def graph():
    region = nx.MultiDiGraph()
    region.add_edge(10, 20, time=1_700_000_000_000_000_000, edge_type=1)
    return region


def verification():
    return VerificationRecord(
        method=VerificationMethod.DATASET_GROUND_TRUTH,
        evidence_ref="ground-truth/attack-1:node-10",
        authority="dataset annotation",
    )


def entry(**overrides):
    values = dict(
        entry_id="reference-1", graph=graph(), label=MemoryLabel.MALICIOUS,
        source_id="dataset-engagement-1", host_id="host-1",
        verification=verification(), attack_instance_id="attack-1",
        techniques=("T1059",),
    )
    values.update(overrides)
    return MemoryEntry(**values)


def test_verified_malicious_and_benign_entries_have_traceable_labels():
    malicious = entry()
    benign = entry(
        entry_id="reference-2", label=MemoryLabel.BENIGN,
        verification=VerificationRecord(
            method=VerificationMethod.ANALYST_REVIEW,
            evidence_ref="reviews/benign-2", authority="analyst-1",
        ),
        attack_instance_id=None, techniques=(),
    )
    assert malicious.label is MemoryLabel.MALICIOUS
    assert benign.label is MemoryLabel.BENIGN
    assert malicious.verification.evidence_ref == "ground-truth/attack-1:node-10"


def test_detector_output_is_not_a_verification_method():
    with pytest.raises(ValueError, match="independent source"):
        VerificationRecord(
            method="detector_prediction", evidence_ref="loss=10", authority="detector",
        )


@pytest.mark.parametrize("bad_record", [None, "verified", 10])
def test_unverified_candidates_cannot_be_memory_entries(bad_record):
    with pytest.raises(ValueError, match="verification record"):
        entry(verification=bad_record)


@pytest.mark.parametrize("field,value", [
    ("entry_id", " "), ("source_id", ""), ("host_id", None),
    ("attack_instance_id", " "), ("label", "malicious"),
])
def test_entry_rejects_invalid_identity_or_implicit_label(field, value):
    with pytest.raises(ValueError):
        entry(**{field: value})


@pytest.mark.parametrize("field,value", [
    ("method", "detector_prediction"), ("evidence_ref", " "),
    ("authority", None),
])
def test_verification_requires_a_real_record_shape(field, value):
    values = dict(
        method=VerificationMethod.CONTROLLED_SIMULATION,
        evidence_ref="simulation/run-1", authority="simulation manifest",
    )
    values[field] = value
    with pytest.raises(ValueError):
        VerificationRecord(**values)


def test_entry_rejects_missing_provenance_fields():
    with pytest.raises(ValueError, match="nonempty provenance"):
        entry(graph=nx.MultiDiGraph())
    missing_time = nx.MultiDiGraph()
    missing_time.add_edge(1, 2, edge_type=1)
    with pytest.raises(ValueError, match="nanosecond time"):
        entry(graph=missing_time)
    missing_relation = nx.MultiDiGraph()
    missing_relation.add_edge(1, 2, time=100)
    with pytest.raises(ValueError, match="relation label"):
        entry(graph=missing_relation)


def test_entry_detaches_its_graph_from_the_candidate():
    candidate = graph()
    reference = entry(graph=candidate)
    candidate.remove_edge(10, 20)
    assert reference.graph.number_of_edges() == 1


def test_entry_rejects_duplicate_technique_ids():
    with pytest.raises(ValueError, match="distinct"):
        entry(techniques=("T1059", "T1059"))
