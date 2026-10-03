"""CPU-only synthetic acceptance tests; no database, real data or W&B network."""

import json
from pathlib import Path
from types import SimpleNamespace

import networkx as nx
import pandas as pd
import pytest
import torch

from pidsmaker.incidents import pipeline
from pidsmaker.incidents.tracking import log_report


def configuration(tmp_path, model="vigil_mlp"):
    from pidsmaker.config import get_runtime_required_args, get_yml_cfg
    args = get_runtime_required_args(args=[model, "ATLASV2_EDR", "--cpu", "--artifact_dir", str(tmp_path)])
    return get_yml_cfg(args)


def test_configuration_and_optional_stage(tmp_path):
    from pidsmaker.main import get_task_to_module
    from pidsmaker.config.config import TASK_DEPENDENCIES
    from pidsmaker.config.pipeline import set_task_paths
    cfg = configuration(tmp_path)
    assert cfg.featurization.training_split == "train"
    assert cfg.featurization.used_method == "word2vec"
    assert cfg.feat_inference.preserve_event_keys
    assert cfg.batching.node_features == "node_emb"
    assert cfg.batching.intra_graph_batching.used_methods == "edges"
    assert cfg.batching.intra_graph_batching.edges.intra_graph_batch_size == 1024
    assert cfg.training.node_hid_dim == cfg.training.node_out_dim == 128
    assert cfg.training.lr == .0001
    assert cfg.training.encoder.dropout == .3
    assert cfg.training.encoder.vigil.num_residual_blocks == 2
    assert cfg.training.num_epochs == 12
    assert cfg.training.decoder.used_methods == "predict_edge_type"
    assert cfg.evaluation.node_evaluation.threshold_method == "max_val_loss"
    assert not cfg.evaluation.node_evaluation.use_kmeans
    assert list(get_task_to_module(cfg))[-1] == "postprocessing"
    assert TASK_DEPENDENCIES["postprocessing"] == ["training", "transformation"]
    training_path = cfg.training._task_path
    cfg.postprocessing.incidents.enabled = False
    set_task_paths(cfg)
    assert cfg.training._task_path == training_path
    assert "postprocessing" not in get_task_to_module(cfg)
    assert not configuration(tmp_path, "velox").postprocessing.incidents.enabled


def test_encoder_training_and_checkpoint(tmp_path):
    from pidsmaker.factory import build_model
    cfg = configuration(tmp_path)
    cfg.feat_inference.event_identity_version = 1  # Legacy handcrafted batch/export coverage.
    batch = SimpleNamespace(
        x_src=torch.randn(4, 128), x_dst=torch.randn(4, 128),
        # IDs intentionally exceed the event count; role tensors must not be
        # indexed using these node identifiers.
        edge_index=torch.tensor([[10, 10, 20, 30], [20, 30, 40, 40]]),
        t=torch.arange(4), msg=torch.randn(4, 295),
        edge_type=torch.nn.functional.one_hot(torch.tensor([0, 1, 2, 32]), 33).float(),
        node_type_src=torch.ones(4, 3), node_type_dst=torch.ones(4, 3), y=torch.zeros(4),
        event_key=torch.zeros(4, dtype=torch.long),
    )
    model = build_model(batch, torch.device("cpu"), cfg, max_node_num=50)
    before = model.encoder.semantic_encoder[0].weight.detach().clone()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.lr)
    loss = model(batch)["loss"]
    assert torch.isfinite(loss).all()
    loss.sum().backward()
    for part in (model.encoder.semantic_encoder, model.encoder.src_encoder, model.encoder.dst_encoder):
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in part.parameters())
    optimizer.step()
    assert not torch.equal(before, model.encoder.semantic_encoder[0].weight)
    model.eval()
    expected = model(batch, inference=True)["loss"]
    signature = model.encoder.encode_nodes(batch.x_src).detach()
    path = tmp_path / "checkpoint.pt"
    torch.save(model.state_dict(), path)
    restored = build_model(batch, torch.device("cpu"), cfg, max_node_num=50)
    restored.load_state_dict(torch.load(path, map_location="cpu"))
    restored.eval()
    torch.testing.assert_close(expected, restored(batch, inference=True)["loss"])
    torch.testing.assert_close(signature, restored.encoder.encode_nodes(batch.x_src))

    # Use the real detector CSV writer, then match its original IDs and encoded
    # relation types back to a provenance graph (not just handcrafted scores).
    from pidsmaker.detection.training_methods.inference_loop import test_edge_level
    batch.original_edge_index = batch.edge_index.clone()
    test_edge_level(batch, restored, "test", "model_epoch_0", cfg, torch.device("cpu"))
    graph = nx.MultiDiGraph()
    for i, (src, dst) in enumerate(batch.edge_index.t().tolist()):
        graph.add_edge(src, dst, time=i, edge_type=int(batch.edge_type[i].argmax()) + 1)
    graph_path = tmp_path / "inference_graph.pt"
    torch.save(graph, graph_path)
    windows, maximum = pipeline.align_split(
        [graph_path], pipeline.score_files(cfg.training._edge_losses_dir, "test", 0),
        {}, tmp_path / "actual_inference",
    )
    assert len(windows) == 1 and maximum > 0


def graph_file(root, name, events):
    graph = nx.MultiDiGraph()
    for src, dst, timestamp, kind, score in events:
        graph.add_edge(src, dst, time=timestamp, edge_type=kind)
    path = root / (name + ".pt")
    torch.save(graph, path)
    return path


def write_scores(root, split, epoch, rows):
    directory = root / split / f"model_epoch_{epoch}"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "arbitrary-batch-name.csv"
    pd.DataFrame(rows, columns=pipeline.COLUMNS).to_csv(path, index=False)
    return path


@pytest.fixture
def inputs(tmp_path):
    ns = 1658188800000000000
    val = [(100, 101, ns, 1, 1.), (101, 102, ns + 1, 1, 2.)]
    first = [(1, 2, ns + 10, 1, 4.), (2, 3, ns + 11, 1, .2), (3, 4, ns + 12, 1, 4.)]
    second = [(1, 5, ns + 100, 1, 4.), (6, 7, ns + 101, 1, .1)]
    graphs = {"val": [graph_file(tmp_path, "val", val)], "test": [
        graph_file(tmp_path, "first", first), graph_file(tmp_path, "second", second),
        graph_file(tmp_path, "empty", []),
    ]}
    root = tmp_path / "scores"
    for epoch in (2, 10):
        write_scores(root, "val", epoch, val)
        write_scores(root, "test", epoch, first + second)
    write_scores(root, "val", 11, val)
    write_scores(root, "test", 11, first)  # incomplete latest epoch
    attack = tmp_path / "attack_A.csv"
    attack.write_text("uuid,description,node_id\nu1,a,1\nu2,b,3\nu3,c,6\nu9,absent,999\n")
    return graphs, root, tmp_path / "reports", [attack]


def test_evaluation_numeric_epoch_alignment_unions_and_cache(inputs, monkeypatch):
    graphs, root, output, labels = inputs
    real_build = pipeline.build_incidents
    calls = []
    def build(*args, **kwargs):
        calls.append(1)
        return real_build(*args, **kwargs)
    monkeypatch.setattr(pipeline, "build_incidents", build)
    report = pipeline.evaluate(graphs, root, output, {}, attack_label_paths=labels)
    assert len(calls) == 3  # one call per test window, including empty
    summary = report["summary"]
    assert summary["selected_epoch"] == 10
    assert summary["threshold"] == 2.
    assert summary["labeled_malicious_present"] == 3
    assert summary["labeled_malicious_absent"] == 1
    assert summary["malicious_coverage_seeds"] == pytest.approx(2 / 3)
    assert summary["malicious_coverage_core"] == pytest.approx(2 / 3)
    assert summary["verified_benign_fraction_core"] is None
    assert report["attacks"][0]["coverage_core"] == pytest.approx(2 / 3)
    assert report["attacks"][0]["incident_fragments"] == 2
    assert report["windows"][2]["incident_count"] == 0
    assert report["windows"][2]["malicious_coverage_core"] is None
    assert report["skipped_incomplete_epochs"]
    for path in report["local_files"]:
        assert (output / path).is_file()
    first_scores = pd.read_csv(output / report["windows"][0]["scores"])
    assert first_scores.time.iloc[0] == 1658188800000000010
    cached = pipeline.evaluate(graphs, root, output, {}, attack_label_paths=labels)
    assert cached == report and len(calls) == 3
    explicit = pipeline.evaluate(graphs, root, output, {}, epoch=2)
    assert explicit["summary"]["selected_epoch"] == 2
    assert explicit["summary"]["malicious_coverage_core"] is None
    with pytest.raises(pipeline.IncompleteEpoch, match="epoch 11"):
        pipeline.evaluate(graphs, root, output, {}, epoch=11)


def test_missing_labels_and_verified_complete_labels(inputs, tmp_path):
    graphs, root, output, labels = inputs
    missing = tmp_path / "not_there.csv"
    report = pipeline.evaluate(graphs, root, output, {}, attack_label_paths=[missing])
    assert report["missing_attack_label_files"] == [str(missing)]
    assert report["summary"]["labeled_malicious_present"] is None
    assert report["summary"]["verified_benign_fraction_core"] is None
    complete = tmp_path / "verified.csv"
    complete.write_text("node_id,is_malicious\n1,1\n2,0\n3,1\n4,0\n5,0\n6,1\n7,0\n")
    report = pipeline.evaluate(graphs, root, output, {}, attack_label_paths=labels, complete_node_labels=complete)
    assert report["summary"]["verified_benign_fraction_core"] == pytest.approx(3 / 5)
    complete.write_text("node_id,is_malicious\n1,1\n")
    with pytest.raises(ValueError, match="Complete node labels miss"):
        pipeline.evaluate(graphs, root, output, {}, complete_node_labels=complete)


def test_atlas_actual_featurization_ids_and_ambiguity(tmp_path):
    from pidsmaker.utils.dataset_utils import get_rel2id
    cfg = SimpleNamespace(dataset=SimpleNamespace(name="ATLASV2_EDR"))
    mapping = pipeline.relation_score_ids(get_rel2id(cfg))
    assert mapping["FILE_OPENED"] == 1 and mapping["REGISTRY_VALUE_DELETED"] == 33
    # Legacy ATLAS has a zero-based table; do not assume its raw IDs equal CSV IDs.
    zero_map = pipeline.relation_score_ids({0: "zero", "zero": 0, 1: "one", "one": 1})
    assert zero_map == {"zero": 2, "one": 1}
    graph = nx.MultiDiGraph()
    graph.add_edge(10, 20, time=50, label="REGISTRY_VALUE_DELETED", event_uuid="a")
    path = tmp_path / "graph.pt"
    torch.save(graph, path)
    scores = write_scores(tmp_path, "test", 1, [(10, 20, 50, 33, 4.)])
    pipeline.align_split([path], [scores], mapping, tmp_path / "ok")
    graph.add_edge(10, 20, time=50, label="REGISTRY_VALUE_DELETED", event_uuid="b")
    torch.save(graph, path)
    with pytest.raises(ValueError, match="Ambiguous"):
        pipeline.align_split([path], [scores], mapping, tmp_path / "ambiguous")
    pd.DataFrame([dict(zip(pipeline.COLUMNS, (10, 20, 50, 33, 4.)), event_uuid=x)
                  for x in ("a", "b")]).to_csv(scores, index=False)
    pipeline.align_split([path], [scores], mapping, tmp_path / "identified")


@pytest.mark.parametrize("mutation,match", [
    ("missing", "Missing event scores"), ("duplicate", "Duplicate score"),
    ("extra", "do not match"), ("wrong_type", "Missing event scores"),
])
def test_strict_alignment(tmp_path, mutation, match):
    rows = [(1, 2, 10, 1, 3.), (2, 3, 11, 1, 4.)]
    path = graph_file(tmp_path, "graph", rows)
    if mutation == "missing":
        rows = rows[:1]
    elif mutation == "duplicate":
        rows += rows[:1]
    elif mutation == "extra":
        rows += [(9, 10, 12, 1, 2.)]
    else:
        rows[0] = (1, 2, 10, 2, 3.)
    scores = write_scores(tmp_path, "test", 1, rows)
    with pytest.raises(ValueError, match=match):
        pipeline.align_split([path], [scores], {}, tmp_path / "aligned")


def test_unique_coverage_not_window_average_and_mixing():
    from pidsmaker.incidents.builder import build_incidents
    accumulator = pipeline.CoverageAccumulator({"A": {1, 2, 3}, "B": {4}, "absent": {99}})
    graph = nx.MultiDiGraph()
    graph.add_edge(1, 4, time=1, edge_type=1)
    graph.add_nodes_from([2, 3])
    scores = pd.DataFrame([(1, 4, 1, 1, 5.)], columns=pipeline.COLUMNS)
    incidents = build_incidents(graph, scores, 2.)
    accumulator.add(graph, incidents)
    accumulator.add(incidents[0].graph, incidents)
    summary, attacks = accumulator.reports()
    assert summary["malicious_coverage_core"] == .5  # not (.5 + 1)/2
    assert summary["mixed_attack_incidents"] == 2
    assert attacks[0]["coverage_core"] == pytest.approx(1 / 3)
    assert attacks[0]["incident_fragments"] == 2
    assert attacks[2]["coverage_core"] is None
    assert attacks[2]["incident_fragments"] is None


def test_coverage_improves_through_connectors_and_context(tmp_path):
    events = [(1, 2, 1, 1, 5.), (2, 3, 2, 1, .1), (3, 4, 3, 1, .1),
              (4, 5, 4, 1, 5.), (3, 6, 3, 1, .1)]
    graph = pipeline._load_graph(graph_file(tmp_path, "context", events))
    scores = pd.DataFrame(events, columns=pipeline.COLUMNS)
    incidents = pipeline.build_incidents(
        graph, scores, 2., pipeline.IncidentBuilderConfig(context_hops=1),
    )
    accumulator = pipeline.CoverageAccumulator({"attack": {1, 3, 6}})
    accumulator.add(graph, incidents)
    summary, attacks = accumulator.reports()
    assert summary["malicious_coverage_seeds"] == pytest.approx(1 / 3)
    assert summary["malicious_coverage_core"] == pytest.approx(2 / 3)
    assert summary["malicious_coverage_with_context"] == 1.
    assert attacks[0]["coverage_with_context"] == 1.


def test_cache_new_epoch_and_label_changes(inputs):
    graphs, root, output, labels = inputs
    first = pipeline.evaluate(graphs, root, output, {}, attack_label_paths=labels)
    for split in ("val", "test"):
        original = pd.read_csv(pipeline.score_files(root, split, 10)[0])
        write_scores(root, split, 20, original[pipeline.COLUMNS].itertuples(index=False, name=None))
    second = pipeline.evaluate(graphs, root, output, {}, attack_label_paths=labels)
    assert second["summary"]["selected_epoch"] == 20
    assert first["fingerprint"] != second["fingerprint"]
    labels[0].write_text("uuid,description,node_id\na,changed,6\n")
    third = pipeline.evaluate(graphs, root, output, {}, attack_label_paths=labels)
    assert third["summary"]["malicious_coverage_core"] == 0.
    assert third["fingerprint"] != second["fingerprint"]


def test_empty_test_split_and_undefined_metrics(inputs):
    graphs, root, output, labels = inputs
    graphs["test"] = []
    write_scores(root, "test", 10, [])
    report = pipeline.evaluate(graphs, root, output, {}, epoch=10, attack_label_paths=labels)
    assert report["windows"] == []
    assert report["summary"]["window_count"] == 0
    assert report["summary"]["labeled_malicious_present"] == 0
    assert report["summary"]["labeled_malicious_absent"] == 4
    assert report["summary"]["malicious_coverage_core"] is None
    assert report["summary"]["mixed_attack_incidents"] is None


def test_shared_attack_nodes_excluded_from_mixing():
    from pidsmaker.incidents.builder import build_incidents
    graph = nx.MultiDiGraph()
    graph.add_edge(1, 2, time=1, edge_type=1)
    scores = pd.DataFrame([(1, 2, 1, 1, 5.)], columns=pipeline.COLUMNS)
    incidents = build_incidents(graph, scores, 2.)
    accumulator = pipeline.CoverageAccumulator({"A": {1}, "B": {1}})
    accumulator.add(graph, incidents)
    summary, attacks = accumulator.reports()
    assert summary["malicious_coverage_core"] == 1.
    assert summary["ambiguous_attack_nodes"] == 1
    assert summary["mixed_attack_incidents"] is None
    assert all(row["incident_fragments"] is None for row in attacks)


class FakeAPI:
    class Table:
        def __init__(self, columns, data):
            self.columns, self.data = columns, data
    plot = SimpleNamespace(bar=lambda *a, **kw: (a, kw), line=lambda *a, **kw: (a, kw))


def test_logging_only_metrics_and_cached_reports(inputs):
    graphs, root, output, labels = inputs
    report = pipeline.evaluate(graphs, root, output, {}, attack_label_paths=labels)
    payloads = []
    for _ in range(2):
        run = SimpleNamespace(summary={}, log=payloads.append)
        cached = pipeline.evaluate(graphs, root, output, {}, attack_label_paths=labels)
        log_report(cached, run, FakeAPI)
        assert run.summary["incidents/selected_epoch"] == 10
    assert payloads[0].keys() == payloads[1].keys()
    assert "incidents/coverage" in payloads[0]
    assert "incidents/label_availability" in payloads[0]
    assert "incidents/fragmentation" in payloads[0]
    for name in ("windows", "attacks", "summary"):
        columns = payloads[0][f"incidents/tables/{name}"].columns
        assert not {"graph", "scores", "incidents", "srcnode", "dstnode"} & set(columns)


def test_stage_relogs_cache_and_disabled_is_noop(inputs, tmp_path, monkeypatch):
    from pidsmaker.tasks import postprocessing
    graphs, root, output, labels = inputs
    cfg = configuration(tmp_path / "artifacts")
    cfg.training._edge_losses_dir = str(root)
    cfg.postprocessing._task_path = str(output)
    cfg.dataset.ground_truth_relative_path = [str(p) for p in labels]
    monkeypatch.setattr(postprocessing, "get_split_to_files", lambda *args: graphs)
    monkeypatch.setattr(postprocessing, "relation_score_ids", lambda *args: {})
    logged = []
    monkeypatch.setattr(postprocessing, "log_report", logged.append)
    first = postprocessing.main(cfg)
    monkeypatch.setattr(pipeline, "build_incidents", lambda *a, **kw: pytest.fail("Rebuilt a cached evaluation"))
    second = postprocessing.main(cfg)
    assert first == second and len(logged) == 2
    cfg.postprocessing.incidents.enabled = False
    assert postprocessing.main(cfg) is None
    assert len(logged) == 2


@pytest.mark.parametrize("label_mode", ["present", "missing", "empty_windows"])
def test_wandb_offline_smoke(inputs, tmp_path, label_mode):
    import wandb
    graphs, root, output, labels = inputs
    if label_mode == "missing":
        labels = []
    if label_mode == "empty_windows":
        graphs["test"] = []
        write_scores(root, "test", 10, [])
    report = pipeline.evaluate(graphs, root, output, {}, epoch=10, attack_label_paths=labels)
    with wandb.init(project="vigil-local-smoke", mode="offline", dir=str(tmp_path)) as run:
        log_report(report, run, wandb)
        assert run.summary["incidents/selected_epoch"] == 10
    assert list((tmp_path / "wandb").glob("offline-run-*/run-*.wandb"))


def test_actual_training_then_incident_stage_offline(tmp_path, monkeypatch):
    """Real training/inference + postprocessing; only dataset provisioning is synthetic."""
    import wandb
    from torch_geometric.data import Data
    from pidsmaker.detection.training_methods import training_loop
    from pidsmaker.tasks import postprocessing

    cfg = configuration(tmp_path / "artifacts")
    cfg.feat_inference.event_identity_version = 1  # Legacy artifacts remain supported when unambiguous.
    cfg.training.num_epochs = 2
    cfg._save_for_viz = True
    label = tmp_path / "attack.csv"
    label.write_text("uuid,description,node_id\na,synthetic,11\n")
    cfg.dataset.ground_truth_relative_path = [str(label)]

    def batch(offset):
        edge_index = torch.tensor([[10, 11, 12], [11, 12, 13]])
        return Data(
            x_src=torch.randn(3, 128), x_dst=torch.randn(3, 128),
            edge_index=edge_index, original_edge_index=edge_index.clone(),
            t=torch.arange(3) + 1658188800000000000 + offset,
            msg=torch.randn(3, 295),
            edge_type=torch.nn.functional.one_hot(torch.tensor([0, 1, 32]), 33).float(),
            node_type_src=torch.ones(3, 3), node_type_dst=torch.ones(3, 3), y=torch.zeros(3),
            event_key=torch.zeros(3, dtype=torch.long),
        )

    train, val, test = batch(0), batch(100), batch(200)
    graphs = {}
    for split, data in (("val", val), ("test", test)):
        events = [(*data.original_edge_index[:, i].tolist(), int(data.t[i]),
                   int(data.edge_type[i].argmax()) + 1, 0.) for i in range(3)]
        graphs[split] = [graph_file(tmp_path, split, events)]
    monkeypatch.setattr(training_loop, "get_preprocessed_graphs", lambda cfg: ([[train]], [[val]], [[test]], 14))
    monkeypatch.setattr(postprocessing, "get_split_to_files", lambda *args: graphs)
    with wandb.init(project="vigil-training-smoke", mode="offline", dir=str(tmp_path)) as run:
        training_loop.main(cfg)
        report = postprocessing.main(cfg)
        assert report["selected_epoch"] == 1
        assert report["window_count"] == 1
        assert run.summary["incidents/selected_epoch"] == 1
        saved = json.loads((Path(cfg.postprocessing._task_path) / "report.json").read_text())
        assert (Path(cfg.postprocessing._task_path) / saved["windows"][0]["incidents"]).is_file()
        assert (Path(cfg.training._trained_models_dir) / "model_epoch_1/state_dict.pkl").is_file()


def test_parallel_events_survive_features_batches_export_and_alignment(tmp_path):
    from pidsmaker.tasks.feat_inference import feat_inference
    from pidsmaker.utils.data_utils import load_all_datasets, collate_temporal_data
    from pidsmaker.utils.dataset_utils import get_rel2id, get_node_map
    from pidsmaker.utils.utils import gen_relation_onehot
    from pidsmaker.detection.training_methods.inference_loop import test_edge_level
    from pidsmaker.detection.evaluation_methods.evaluation_utils import datetime_to_ns_time_US_handle_nano

    cfg = configuration(tmp_path / "artifacts")
    cfg.batching.intra_graph_batching.edges.intra_graph_batch_size = 1
    graph = nx.MultiDiGraph()
    for node in (506, 255026, 255027):
        graph.add_node(node, node_type="subject")
    timestamp = 1658031176581720500
    for dst, key in ((255026, 7), (255026, 42), (255027, 9)):
        graph.add_edge(506, dst, key=key, time=timestamp, label="FILE_OPENED")
    graph_path = tmp_path / "parallel.pt"
    torch.save(graph, graph_path)
    relations = get_rel2id(cfg)
    for split in ("train", "val", "test"):
        directory = Path(cfg.feat_inference._edge_embeds_dir) / split
        feat_inference(
            {n: torch.randn(128).numpy() for n in graph.nodes},
            gen_relation_onehot(relations), gen_relation_onehot(get_node_map()),
            [str(graph_path)], str(directory), cfg,
        )
        feature_path = directory / "parallel.pt.TemporalData.simple"
        data = torch.load(feature_path)
        assert data.event_key.tolist() == [7, 42, 9]
        # Keys must follow reordering, slicing, and concatenation, not positions
        # assigned later by the CSV writer.
        data = data[torch.tensor([2, 1, 0])]
        assert data.event_key.tolist() == [9, 42, 7]
        joined = collate_temporal_data([data[:1], data[1:]])
        assert joined.event_key.tolist() == [9, 42, 7]
        torch.save(joined, feature_path)

    _, _, test_data, _ = load_all_datasets(cfg, torch.device("cpu"))

    class IdentityScoreModel:
        def eval(self):
            pass

        def __call__(self, batch, **kwargs):
            # Distinct losses expose incorrectly paired parallel edges. This
            # is a test double, NOT a feature consumed by the actual encoder.
            return {"loss": batch.event_key.float() + 1}

    for batch in test_data[0]:
        test_edge_level(batch, IdentityScoreModel(), "test", "model_epoch_0", cfg, "cpu")
    files = pipeline.score_files(cfg.training._edge_losses_dir, "test", 0)
    assert len(files) == 3  # all three batches have the exact same timestamps
    for file in files:
        # Existing detector time-range parsing remains compatible.
        assert len([datetime_to_ns_time_US_handle_nano(t) for t in file.name.split("~")]) == 2
    output = tmp_path / "aligned"
    windows, _ = pipeline.align_split([graph_path], files, pipeline.relation_score_ids(relations), output)
    scores = pd.read_csv(output.parent / windows[0]["scores"])
    assert dict(zip(scores["key"], scores.loss)) == {7: 8., 42: 43., 9: 10.}
    prepared = pipeline.prepare_provenance_graph(graph, scores, pipeline.relation_score_ids(relations))
    assert prepared[506][255026][7]["score"] == 8.
    assert prepared[506][255026][42]["score"] == 43.
    incidents = pipeline.build_incidents(graph, scores, 0., relation_to_id=pipeline.relation_score_ids(relations))
    assert sum(len(i.seed_edges) for i in incidents) == 3
    # Re-export is deterministic, not a second copy of every score.
    for batch in test_data[0]:
        test_edge_level(batch, IdentityScoreModel(), "test", "model_epoch_0", cfg, "cpu")
    assert len(pipeline.score_files(cfg.training._edge_losses_dir, "test", 0)) == 3


def test_event_identity_setting_invalidates_only_downstream_cache(tmp_path):
    from pidsmaker.config.pipeline import set_task_paths
    cfg = configuration(tmp_path)
    tasks = ["construction", "transformation", "featurization", "feat_inference",
             "batching", "training", "evaluation", "postprocessing"]
    keyed_paths = {t: getattr(cfg, t)._task_path for t in tasks}
    cfg.feat_inference.preserve_event_keys = False
    set_task_paths(cfg)
    for task in tasks[:3]:
        assert getattr(cfg, task)._task_path == keyed_paths[task]
    for task in tasks[3:]:
        assert getattr(cfg, task)._task_path != keyed_paths[task]


def test_missing_event_keys_fail_with_rebuild_instruction(tmp_path):
    from pidsmaker.detection.training_methods.inference_loop import test_edge_level
    cfg = configuration(tmp_path)
    batch = SimpleNamespace(t=torch.tensor([1]), original_edge_index=torch.tensor([[1], [2]]),
                            edge_type=torch.tensor([[1., 0.]]))
    class Model:
        def eval(self):
            pass
        def __call__(self, *args, **kwargs):
            return {"loss": torch.tensor([1.])}
    with pytest.raises(ValueError, match="force_restart feat_inference"):
        test_edge_level(batch, Model(), "test", "model_epoch_0", cfg, "cpu")
    cfg.feat_inference.preserve_event_keys = False
    test_edge_level(batch, Model(), "test", "model_epoch_0", cfg, "cpu")
    scores = pd.read_csv(pipeline.score_files(cfg.training._edge_losses_dir, "test", 0)[0])
    assert "key" not in scores  # old detector export is unchanged when disabled
