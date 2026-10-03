"""Cross-window event-key collisions must not conflate distinct observations."""

import shutil
from pathlib import Path

import networkx as nx
import pandas as pd
import pytest
import torch

from pidsmaker.incidents import pipeline
from pidsmaker.utils.event_identity import event_uuid_hash, source_graph_id


def scoped_fixture(tmp_path):
    graphs, files = [], []
    for i, loss in enumerate((0.4868861734867096, 0.17388974130153656)):
        graph = nx.MultiDiGraph()
        graph.add_edge(506, 295276, key=0, time=1658064867804978400,
                       edge_type=32, event_uuid=f"distinct-original-event-{i}")
        path = tmp_path / f"window_{i}.pt"
        torch.save(graph, path)
        graphs.append(path)
        score = tmp_path / f"scores_{i}.csv"
        pd.DataFrame([dict(srcnode=506, dstnode=295276, time=1658064867804978400,
                           edge_type=32, key=0, loss=loss, source_graph=source_graph_id(path),
                           event_uuid_hash=event_uuid_hash(f"distinct-original-event-{i}"))]).to_csv(score, index=False)
        files.append(score)
    return graphs, files


def test_boundary_collision_keeps_both_scores_in_their_original_windows(tmp_path):
    graphs, files = scoped_fixture(tmp_path)
    windows, maximum = pipeline.align_split(graphs, files, {}, tmp_path / "aligned")
    assert len(windows) == 2
    assert maximum == pytest.approx(0.4868861734867096)
    for i, window in enumerate(windows):
        aligned = pd.read_csv(tmp_path / window["scores"])
        original = pd.read_csv(files[i])
        pd.testing.assert_frame_equal(aligned, original)
        incidents = pipeline.build_incidents(torch.load(graphs[i]), aligned, 0.)
        assert len(incidents) == 1
        assert len(incidents[0].seed_edges) == 1
        assert incidents[0].graph[506][295276][0]["event_uuid"] == f"distinct-original-event-{i}"


@pytest.mark.parametrize("second_loss", [0.4868861734867096, 0.9])
def test_true_duplicate_within_source_graph_still_fails(tmp_path, second_loss):
    graphs, files = scoped_fixture(tmp_path)
    duplicate = pd.read_csv(files[0])
    duplicate["loss"] = second_loss
    extra = tmp_path / "duplicate.csv"
    duplicate.to_csv(extra, index=False)
    with pytest.raises(ValueError, match="Duplicate score rows"):
        pipeline.align_split(graphs, files + [extra], {}, tmp_path / "aligned")


def test_uuid_mismatch_is_not_accepted_even_with_correct_key_and_graph(tmp_path):
    graphs, files = scoped_fixture(tmp_path)
    scores = pd.read_csv(files[0])
    scores["event_uuid_hash"] = event_uuid_hash("wrong-original-event")
    scores.to_csv(files[0], index=False)
    with pytest.raises(ValueError, match="not found"):
        pipeline.align_split(graphs, files, {}, tmp_path / "aligned")


@pytest.mark.parametrize("fault", ["legacy", "missing", "unknown", "wrong_event"])
def test_stale_or_invalid_scoped_scores_fail_closed(tmp_path, fault):
    graphs, files = scoped_fixture(tmp_path)
    scores = pd.read_csv(files[1])
    if fault == "legacy":
        scores = scores.drop(columns=["source_graph", "event_uuid_hash"])
    elif fault == "missing":
        scores["source_graph"] = ""
    elif fault == "unknown":
        scores["source_graph"] = "f" * 64
    else:
        scores["srcnode"] = 999
    scores.to_csv(files[1], index=False)
    with pytest.raises(ValueError):
        pipeline.align_split(graphs, files, {}, tmp_path / "aligned")


def test_identity_survives_artifact_relocation_but_rejects_repeated_source(tmp_path):
    graphs, files = scoped_fixture(tmp_path)
    relocated = tmp_path / "relocated"
    relocated.mkdir()
    copied = []
    for path in graphs:
        target = relocated / path.name
        shutil.copyfile(path, target)
        assert source_graph_id(target) == source_graph_id(path)
        copied.append(target)
    assert len(pipeline.align_split(copied, files, {}, tmp_path / "aligned")[0]) == 2
    with pytest.raises(ValueError, match="same source graph"):
        pipeline.align_split([graphs[0], copied[0]], files[:1], {}, tmp_path / "repeated")


def test_modified_graph_rejects_old_scores_instead_of_falling_back(tmp_path):
    graphs, files = scoped_fixture(tmp_path)
    graph = torch.load(graphs[0])
    graph[506][295276][0]["event_uuid"] = "replacement-event"
    torch.save(graph, graphs[0])
    with pytest.raises(ValueError, match="source graphs do not match") as error:
        pipeline.align_split(graphs, files, {}, tmp_path / "aligned")
    assert not isinstance(error.value, pipeline.IncompleteEpoch)


def test_scoped_alignment_keeps_empty_windows(tmp_path):
    graphs, files = scoped_fixture(tmp_path)
    empty = tmp_path / "empty.pt"
    torch.save(nx.MultiDiGraph(), empty)
    windows, _ = pipeline.align_split([empty] + graphs, files, {}, tmp_path / "aligned")
    assert len(windows) == 3
    assert pd.read_csv(tmp_path / windows[0]["scores"]).empty


def test_legacy_boundary_collision_is_not_guessed(tmp_path):
    graphs, files = scoped_fixture(tmp_path)
    for file in files:
        pd.read_csv(file).drop(columns=["source_graph", "event_uuid_hash"]).to_csv(file, index=False)
    with pytest.raises(ValueError, match="Duplicate score rows"):
        pipeline.align_split(graphs, files, {}, tmp_path / "aligned")


def test_legacy_string_keys_remain_supported(tmp_path):
    graph = nx.MultiDiGraph()
    graph.add_edge(1, 2, key="custom-key", time=1, edge_type=1)
    path = tmp_path / "graph.pt"
    torch.save(graph, path)
    scores = tmp_path / "scores.csv"
    pd.DataFrame([dict(srcnode=1, dstnode=2, key="custom-key", time=1, edge_type=1, loss=1.)]).to_csv(scores, index=False)
    assert len(pipeline.align_split([path], [scores], {}, tmp_path / "aligned")[0]) == 1


@pytest.mark.parametrize("profile", ["vigil_mlp", "vigil"])
def test_scope_and_uuid_survive_real_features_batching_and_score_export(tmp_path, profile):
    from pidsmaker.config import get_runtime_required_args, get_yml_cfg
    from pidsmaker.tasks.feat_inference import feat_inference
    from pidsmaker.utils.data_utils import load_all_datasets, collate_temporal_data
    from pidsmaker.utils.dataset_utils import get_rel2id, get_node_map
    from pidsmaker.utils.utils import gen_relation_onehot
    from pidsmaker.detection.training_methods.inference_loop import test_edge_level

    cfg = get_yml_cfg(get_runtime_required_args(args=[profile, "ATLASV2_EDR", "--cpu", "--artifact_dir", str(tmp_path / "artifacts")]))
    cfg.batching.intra_graph_batching.edges.intra_graph_batch_size = 1
    relations = get_rel2id(cfg)
    paths = []
    for i in range(2):
        graph = nx.MultiDiGraph()
        graph.add_node(10, node_type="subject")
        graph.add_node(20, node_type="subject")
        graph.add_edge(10, 20, key=0, time=1658064867804978400, label="FILE_OPENED", event_uuid=f"event-{i}")
        path = tmp_path / f"graph_{i}.pt"
        torch.save(graph, path)
        paths.append(path)
    for split in ("train", "val", "test"):
        directory = Path(cfg.feat_inference._edge_embeds_dir) / split
        feat_inference({n: torch.zeros(128).numpy() for n in (10, 20)},
                       gen_relation_onehot(relations), gen_relation_onehot(get_node_map()),
                       [str(p) for p in paths], str(directory), cfg)
        originals = [torch.load(directory / f"{p.name}.TemporalData.simple") for p in paths]
        reordered = collate_temporal_data(originals)[torch.tensor([1, 0])]
        assert bytes(reordered.event_source_graph[0].tolist()).hex() == source_graph_id(paths[1])
        assert bytes(reordered.event_uuid_hash[1].tolist()).hex() == event_uuid_hash("event-0")

    _, _, test, _ = load_all_datasets(cfg, torch.device("cpu"))
    class Model:
        def eval(self):
            pass
        def __call__(self, batch, **kwargs):
            return {"loss": torch.ones(len(batch.t))}
    for batch in test[0]:
        test_edge_level(batch, Model(), "test", "model_epoch_0", cfg, "cpu")
    files = pipeline.score_files(cfg.training._edge_losses_dir, "test", 0)
    # Same time range, endpoints, relation and key MUST NOT overwrite a file.
    assert len(files) == 2
    windows, _ = pipeline.align_split(paths, files, pipeline.relation_score_ids(relations), tmp_path / "aligned")
    for i, window in enumerate(windows):
        row = pd.read_csv(tmp_path / window["scores"]).iloc[0]
        assert row["source_graph"] == source_graph_id(paths[i])
        assert row["event_uuid_hash"] == event_uuid_hash(f"event-{i}")


def test_identity_version_invalidates_only_affected_cache_paths(tmp_path):
    from pidsmaker.config import get_runtime_required_args, get_yml_cfg
    from pidsmaker.config.pipeline import set_task_paths
    cfg = get_yml_cfg(get_runtime_required_args(args=["vigil", "ATLASV2_EDR", "--artifact_dir", str(tmp_path)]))
    assert cfg.feat_inference.event_identity_version == 2
    tasks = ["construction", "transformation", "featurization", "feat_inference", "batching", "training", "evaluation", "postprocessing"]
    before = {task: getattr(cfg, task)._task_path for task in tasks}
    cfg.feat_inference.event_identity_version = 1
    set_task_paths(cfg)
    for task in tasks:
        assert (getattr(cfg, task)._task_path == before[task]) == (task in tasks[:3])
    from pidsmaker.config.pipeline import check_edge_cases
    cfg.feat_inference.event_identity_version = 3
    with pytest.raises(ValueError, match="event_identity_version"):
        check_edge_cases(cfg)


@pytest.mark.parametrize("fault", ["missing", "shape", "dtype", "blank"])
def test_export_rejects_missing_or_corrupt_source_metadata(tmp_path, fault):
    from types import SimpleNamespace
    from pidsmaker.config import get_runtime_required_args, get_yml_cfg
    from pidsmaker.detection.training_methods.inference_loop import test_edge_level
    cfg = get_yml_cfg(get_runtime_required_args(args=["vigil", "ATLASV2_EDR", "--artifact_dir", str(tmp_path)]))
    batch = SimpleNamespace(t=torch.tensor([1]), original_edge_index=torch.tensor([[1], [2]]),
                            edge_type=torch.tensor([[1., 0.]]), event_key=torch.tensor([0]),
                            event_source_graph=torch.ones(1, 32, dtype=torch.uint8),
                            event_uuid_hash=torch.zeros(1, 32, dtype=torch.uint8))
    if fault == "missing":
        del batch.event_source_graph
    elif fault == "shape":
        batch.event_source_graph = torch.ones(32, dtype=torch.uint8)
    elif fault == "dtype":
        batch.event_source_graph = batch.event_source_graph.long()
    else:
        batch.event_source_graph.zero_()
    class Model:
        def eval(self):
            pass
        def __call__(self, *args, **kwargs):
            return {"loss": torch.tensor([1.])}
    with pytest.raises(ValueError):
        test_edge_level(batch, Model(), "val", "model_epoch_0", cfg, "cpu")
    assert not pipeline.score_files(cfg.training._edge_losses_dir, "val", 0)
