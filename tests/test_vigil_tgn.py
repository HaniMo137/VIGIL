"""VIGIL's real TGN loader, attention encoder, score export and reconstruction."""

from pathlib import Path

import networkx as nx
import pandas as pd
import pytest
import torch

from pidsmaker.incidents import pipeline


def tgn_fixture(tmp_path):
    from pidsmaker.config import get_runtime_required_args, get_yml_cfg
    from pidsmaker.tasks.feat_inference import feat_inference
    from pidsmaker.utils.data_utils import load_all_datasets
    from pidsmaker.utils.dataset_utils import get_rel2id, get_node_map
    from pidsmaker.utils.utils import gen_relation_onehot

    args = get_runtime_required_args(args=["vigil", "ATLASV2_EDR", "--cpu", "--artifact_dir", str(tmp_path / "artifacts")])
    cfg = get_yml_cfg(args)
    cfg.batching.intra_graph_batching.edges.intra_graph_batch_size = 2
    relations = get_rel2id(cfg)
    generator = torch.Generator().manual_seed(17)
    features = {n: torch.randn(128, generator=generator).numpy() for n in (10, 20, 30, 40)}
    graphs = {}
    for split, offset in (("train", 0), ("val", 100), ("test", 200)):
        graph = nx.MultiDiGraph()
        for node in features:
            graph.add_node(node, node_type="subject")
        for index, (src, dst, key) in enumerate(((10, 20, 7), (20, 30, 3), (10, 20, 42), (30, 40, 4), (10, 30, 9))):
            graph.add_edge(src, dst, key=key, time=1658188800000000000 + offset + index,
                           label="FILE_OPENED")
        path = tmp_path / f"{split}.pt"
        torch.save(graph, path)
        graphs[split] = [path]
        target = Path(cfg.feat_inference._edge_embeds_dir) / split
        feat_inference(features, gen_relation_onehot(relations), gen_relation_onehot(get_node_map()),
                       [str(path)], str(target), cfg)
    train, val, test, max_node = load_all_datasets(cfg, torch.device("cpu"))
    return cfg, (train, val, test), graphs, max_node


@pytest.fixture
def setup(tmp_path):
    return tgn_fixture(tmp_path)


def test_configuration_preserves_user_choices_and_valid_tgn_pipeline(setup):
    cfg, _, _, _ = setup
    assert cfg.training.encoder.used_methods == "tgn, vigil"
    assert cfg.featurization.training_split == "train"
    assert cfg.batching.node_features == "node_emb, node_type"
    assert cfg.evaluation.node_evaluation.threshold_method == "max_val_loss"
    assert cfg.evaluation.node_evaluation.use_kmeans
    assert cfg.evaluation.node_evaluation.kmeans_top_K == 30
    assert cfg.batching.fix_buggy_graph_reindexer
    assert not cfg.batching.intra_graph_batching.tgn_last_neighbor.insert_neighbors_before
    assert not cfg.training.encoder.tgn.use_memory


def test_temporal_loader_uses_history_not_current_or_future_events(setup):
    _, datasets, _, _ = setup
    for dataset in datasets:
        previous_time = None
        seen = set()
        for batch in dataset[0]:
            history = batch.t_tgn
            assert not len(history) or history.max() < batch.t.min()
            if previous_time is None:
                assert batch.edge_index_tgn.shape[1] == 0
            for i in range(batch.edge_index_tgn.shape[1]):
                src, dst = batch.n_id_tgn[batch.edge_index_tgn[:, i]].tolist()
                assert (min(src, dst), max(src, dst), int(history[i])) in seen
            for i in range(batch.num_events):
                src, dst = batch.original_edge_index[:, i].tolist()
                seen.add((min(src, dst), max(src, dst), int(batch.t[i])))
            previous_time = int(batch.t.max())


def test_attention_gradients_history_sensitivity_and_checkpoint(setup, tmp_path):
    from pidsmaker.factory import build_model
    from pidsmaker.encoders.vigil_encoder import VigilTGNEncoder
    from pidsmaker.utils.data_utils import load_model, save_model
    cfg, (train, _, _), _, max_node = setup
    batch = train[0][1]
    model = build_model(train[0][0], torch.device("cpu"), cfg, max_node)
    assert isinstance(model.encoder, VigilTGNEncoder)
    assert len(model.encoder.encoder.convs) == 2
    assert model.encoder.src_linear.in_features == 131
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.lr)
    loss = model(batch)["loss"]
    assert torch.isfinite(loss).all()
    loss.sum().backward()
    for module in (model.encoder.src_linear, model.encoder.dst_linear, model.encoder.encoder):
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters())
    before = model.encoder.src_linear.weight.detach().clone()
    optimizer.step()
    assert not torch.equal(before, model.encoder.src_linear.weight)
    model.eval()
    expected = model(batch, inference=True)["loss"]
    ids, embeddings = model.encoder.encode_temporal_nodes(batch)
    assert torch.equal(ids, batch.original_n_id)
    assert embeddings.shape == (len(ids), 128)
    empty_history = batch.clone()
    empty_history.edge_index_tgn = batch.edge_index_tgn[:, :0]
    empty_history.edge_type_tgn = batch.edge_type_tgn[:0]
    changed = model.encoder.encode_temporal_nodes(empty_history)[1]
    assert not torch.allclose(embeddings, changed)
    path = tmp_path / "checkpoint"
    save_model(model, str(path), cfg)
    assert not (path / "neighbor_loader.pkl").exists()
    restored = build_model(train[0][0], torch.device("cpu"), cfg, max_node)
    load_model(restored, str(path), cfg, map_location="cpu")
    restored.eval()
    torch.testing.assert_close(expected, restored(batch, inference=True)["loss"])
    torch.testing.assert_close(embeddings, restored.encoder.encode_temporal_nodes(batch)[1])
    with pytest.raises(ValueError, match="temporal context"):
        restored.encoder.encode_nodes(torch.randn(4, 131))


def test_real_tgn_export_and_incident_reconstruction(setup, tmp_path):
    from pidsmaker.factory import build_model
    from pidsmaker.detection.training_methods.inference_loop import test_edge_level
    from pidsmaker.utils.dataset_utils import get_rel2id
    cfg, (train, val, test), graphs, max_node = setup
    model = build_model(train[0][0], torch.device("cpu"), cfg, max_node)
    mapping = pipeline.relation_score_ids(get_rel2id(cfg))
    for split, dataset in (("val", val), ("test", test)):
        for batch in dataset[0]:
            test_edge_level(batch, model, split, "model_epoch_0", cfg, torch.device("cpu"))
        folder = tmp_path / f"align_{split}"
        windows, _ = pipeline.align_split(graphs[split], pipeline.score_files(cfg.training._edge_losses_dir, split, 0), mapping, folder)
        scores = pd.read_csv(folder.parent / windows[0]["scores"])
        assert len(scores) == 5
        assert set(scores["key"]) == {7, 3, 42, 4, 9}
        original = torch.load(graphs[split][0], weights_only=False)
        # Zero is a fixture-only threshold: verify every exported identity, not detection quality.
        incidents = pipeline.build_incidents(original, scores, 0., relation_to_id=mapping)
        assert sum(len(i.seed_edges) for i in incidents) == 5
        assert {(e.srcnode, e.dstnode, e.key) for i in incidents for e in i.seed_edges} == set(original.edges(keys=True))


def test_actual_tgn_training_then_incident_stage(setup, tmp_path, monkeypatch):
    import wandb
    from pidsmaker.detection.training_methods import training_loop
    from pidsmaker.tasks import postprocessing
    cfg, datasets, graphs, max_node = setup
    cfg.training.num_epochs = 2
    cfg._save_for_viz = True
    cfg.dataset.ground_truth_relative_path = []
    monkeypatch.setattr(training_loop, "get_preprocessed_graphs", lambda cfg: (*datasets, max_node))
    monkeypatch.setattr(postprocessing, "get_split_to_files", lambda *a: {s: graphs[s] for s in ("val", "test")})
    with wandb.init(project="vigil-tgn-smoke", mode="offline", dir=str(tmp_path)):
        training_loop.main(cfg)
        report = postprocessing.main(cfg)
    assert report["selected_epoch"] == 1
    assert report["window_count"] == 1
    assert (Path(cfg.training._trained_models_dir) / "model_epoch_1/state_dict.pkl").is_file()


def test_stable_temporal_sort_keeps_event_identity_and_rejects_window_overlap():
    from pidsmaker.utils.data_utils import CollatableTemporalData, prepare_temporal_history
    data = CollatableTemporalData(src=torch.tensor([1, 2, 3]), dst=torch.tensor([2, 3, 4]),
                                  t=torch.tensor([30, 10, 10]), msg=torch.tensor([[3.], [1.], [2.]]),
                                  event_key=torch.tensor([300, 100, 200]))
    stream = [data]
    prepare_temporal_history([[stream]])
    assert stream[0].t.tolist() == [10, 10, 30]
    assert stream[0].event_key.tolist() == [100, 200, 300]
    assert stream[0].src.tolist() == [2, 3, 1]
    assert stream[0].msg.flatten().tolist() == [1., 2., 3.]
    with pytest.raises(ValueError, match="overlap"):
        prepare_temporal_history([[[stream[0], data]]])


def test_invalid_tgn_combinations_fail_early(setup):
    from pidsmaker.config.pipeline import check_edge_cases
    cfg, _, _, _ = setup
    cfg.batching.intra_graph_batching.tgn_last_neighbor.insert_neighbors_before = True
    with pytest.raises(ValueError, match="before scoring"):
        check_edge_cases(cfg)
    cfg.batching.intra_graph_batching.tgn_last_neighbor.insert_neighbors_before = False
    cfg.training.encoder.used_methods = "vigil"
    with pytest.raises(ValueError, match="tgn, vigil"):
        check_edge_cases(cfg)
