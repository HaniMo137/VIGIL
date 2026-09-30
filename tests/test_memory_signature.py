import networkx as nx
import numpy as np
import pytest
import torch

from pidsmaker.encoders.vigil_encoder import VigilEncoder
from pidsmaker.memory.signature import (
    SignatureSchema, build_incident_signature, encoder_state_fingerprint,
)


def graph():
    region = nx.MultiDiGraph()
    region.add_node(1, node_type="process")
    region.add_node(2, node_type="file")
    region.add_node(3, node_type="process")
    region.add_edge(1, 2, key="a", time=10, edge_type=1)
    region.add_edge(2, 3, key="b", time=20, edge_type=2)
    region.add_edge(1, 3, key="c", time=30, edge_type=2)
    return region


def features():
    return {1: [1.0, 0.0, 0.0], 2: [0.0, 1.0, 0.0], 3: [0.0, 0.0, 1.0]}


def encoder():
    torch.manual_seed(13)
    model = VigilEncoder(in_dim=3, hid_dim=5, out_dim=2, dropout=0.4)
    model.eval()
    return model


def schema(model=None, **overrides):
    values = dict(
        semantic_dim=5, node_types=("process", "file"), relation_types=(1, 2),
        encoder_checkpoint_id=encoder_state_fingerprint(model if model is not None else encoder()),
    )
    values.update(overrides)
    return SignatureSchema(**values)


def test_signature_uses_shared_encoder_embeddings_and_four_fixed_blocks():
    model = encoder()
    result = build_incident_signature(graph(), features(), model, schema())
    assert result.vector.shape == (schema().dimension,) == (20,)
    assert not result.vector.flags.writeable
    assert result.schema_id == schema().schema_id
    assert result.encoder_checkpoint_id == encoder_state_fingerprint(model)

    with torch.no_grad():
        shared = model.encode_nodes(torch.tensor(list(features().values()))).mean(dim=0)
    expected_semantic = shared.numpy() / np.linalg.norm(shared.numpy()) * 0.5
    np.testing.assert_allclose(result.vector[:5], expected_semantic, atol=1e-7)
    np.testing.assert_allclose(result.vector[5:8],
                               np.array([2, 1, 0]) / np.sqrt(5) * 0.5)
    np.testing.assert_allclose(result.vector[8:11],
                               np.array([1, 2, 0]) / np.sqrt(5) * 0.5)
    assert result.vector[11 + 1] == pytest.approx(0.5)
    assert all(parameter.grad is None for parameter in model.parameters())


def test_encoder_node_api_preserves_detector_role_outputs():
    model = encoder()
    src = torch.tensor([[1.0, 0.0, 0.0]])
    dst = torch.tensor([[0.0, 1.0, 0.0]])
    assert model.encode_nodes(src).shape == (1, 5)
    src_role, dst_role = model(src, dst)
    assert src_role.shape == dst_role.shape == (1, 2)
    with pytest.raises(ValueError, match="shape"):
        model.encode_nodes(torch.ones(1, 4))


def test_signature_requires_eval_mode_and_every_node_feature():
    model = encoder()
    model.train()
    with pytest.raises(ValueError, match="eval mode"):
        build_incident_signature(graph(), features(), model, schema())
    model.eval()
    missing = features()
    del missing[2]
    with pytest.raises(ValueError, match="Missing node features"):
        build_incident_signature(graph(), missing, model, schema())
    bad = features()
    bad[2] = [float("nan"), 1.0, 0.0]
    with pytest.raises(ValueError, match="finite vector"):
        build_incident_signature(graph(), bad, model, schema())


def test_schema_and_signature_are_stable_under_graph_insertion_order():
    first = graph()
    reversed_graph = nx.MultiDiGraph()
    for node, data in reversed(list(first.nodes(data=True))):
        reversed_graph.add_node(node, **data)
    for src, dst, key, data in reversed(list(first.edges(keys=True, data=True))):
        reversed_graph.add_edge(src, dst, key=key, **data)
    model = encoder()
    a = build_incident_signature(first, features(), model, schema())
    b = build_incident_signature(reversed_graph, features(), model, schema())
    np.testing.assert_allclose(a.vector, b.vector, atol=1e-12)


def test_node_features_change_semantic_block_not_graph_histograms():
    model = encoder()
    first = build_incident_signature(graph(), features(), model, schema())
    changed = features()
    changed[1] = [8.0, 2.0, -3.0]
    second = build_incident_signature(graph(), changed, model, schema())
    assert not np.allclose(first.vector[:5], second.vector[:5])
    np.testing.assert_array_equal(first.vector[5:], second.vector[5:])


def test_two_event_paths_respect_event_order():
    region = graph()
    region[2][3]["b"]["time"] = 5
    result = build_incident_signature(region, features(), encoder(), schema())
    np.testing.assert_array_equal(result.vector[11:], np.zeros(9))


def test_unknown_node_and_relation_types_have_fixed_bins():
    region = graph()
    region.nodes[2]["node_type"] = "socket"
    region[2][3]["b"]["edge_type"] = 999
    result = build_incident_signature(region, features(), encoder(), schema())
    assert result.vector[7] > 0  # unknown node type
    assert result.vector[10] > 0  # unknown relation


def test_checkpoint_and_vocab_are_part_of_signature_schema_identity():
    base = schema()
    assert base.schema_id != schema(encoder_checkpoint_id="0" * 64).schema_id
    assert base.schema_id != schema(relation_types=(2, 1)).schema_id
    with pytest.raises(ValueError, match="duplicates"):
        schema(node_types=("Process", "process"))
    with pytest.raises(ValueError, match="checkpoint"):
        schema(encoder_checkpoint_id=" ")


def test_malformed_graph_or_encoder_output_is_rejected():
    region = graph()
    del region[1][2]["a"]["time"]
    with pytest.raises(ValueError, match="nanosecond time"):
        build_incident_signature(region, features(), encoder(), schema())
    with pytest.raises(ValueError, match="nonempty provenance"):
        build_incident_signature(nx.MultiDiGraph(), features(), encoder(), schema())
    with pytest.raises(ValueError, match="invalid shared node embedding"):
        build_incident_signature(graph(), features(), encoder(), schema(semantic_dim=2))


def test_signature_rejects_changed_encoder_weights():
    model = encoder()
    spec = schema(model)
    with torch.no_grad():
        next(model.parameters()).add_(0.1)
    with pytest.raises(ValueError, match="checkpoint"):
        build_incident_signature(graph(), features(), model, spec)
