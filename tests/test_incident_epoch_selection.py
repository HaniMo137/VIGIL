"""Detector-selected checkpoint propagation, cache reuse and provenance."""

import json
from pathlib import Path

import networkx as nx
import pandas as pd
import pytest
import torch

from pidsmaker.incidents import pipeline
from pidsmaker.utils.epoch_selection import (
    load_detector_selection, save_detector_selection, score_inventory_fingerprint, selection_path,
)


@pytest.fixture
def setup(tmp_path, monkeypatch):
    from pidsmaker.config import get_runtime_required_args, get_yml_cfg
    from pidsmaker.tasks import evaluation, postprocessing
    cfg = get_yml_cfg(get_runtime_required_args(args=["vigil", "ATLASV2_EDR", "--cpu", "--artifact_dir", str(tmp_path / "artifacts")]))
    cfg.dataset.ground_truth_relative_path = []
    graphs = {}
    for split in ("val", "test"):
        graph = nx.MultiDiGraph()
        graph.add_edge(1, 2, key=0, time=1, edge_type=1)
        path = tmp_path / f"{split}.pt"
        torch.save(graph, path)
        graphs[split] = [path]
        for epoch in (0, 11):
            directory = Path(cfg.training._edge_losses_dir) / split / f"model_epoch_{epoch}"
            directory.mkdir(parents=True)
            loss = (1. if epoch == 0 else 3.) if split == "val" else 2.
            pd.DataFrame([dict(srcnode=1, dstnode=2, time=1, edge_type=1, key=0, loss=loss)]).to_csv(directory / "batch.csv", index=False)
    monkeypatch.setattr(evaluation, "compute_tw_labels", lambda cfg: {})
    monkeypatch.setattr(evaluation.wandb, "log", lambda *a, **kw: None)
    monkeypatch.setattr(evaluation.wandb, "save", lambda *a, **kw: None)
    monkeypatch.setattr(postprocessing, "get_split_to_files", lambda *a: graphs)
    monkeypatch.setattr(postprocessing, "relation_score_ids", lambda *a: {})
    reports = []
    monkeypatch.setattr(postprocessing, "log_report", reports.append)
    return cfg, reports


def run_detector_evaluation(cfg, metrics=None):
    from pidsmaker.tasks import evaluation
    metrics = metrics or {0: (.8, .7), 11: (.1, .2)}
    def evaluate(val, test, epoch_name, cfg, **kwargs):
        epoch = int(epoch_name.split("_")[-1])
        adp, discrimination = metrics[epoch]
        return {"adp_score": adp, "discrimination": discrimination,
                "neat_scores_img_file": "fixture.svg"}
    return evaluation.standard_evaluation(cfg, evaluate)


def test_best_epoch_propagates_and_cached_stage_does_not_use_latest(setup, monkeypatch):
    from pidsmaker.tasks import postprocessing
    cfg, reports = setup
    assert cfg.postprocessing.incidents.epoch_selection == "detector_best"
    assert run_detector_evaluation(cfg)["epoch"] == 0
    selected = load_detector_selection(cfg)
    assert selected["epoch"] == 0 and selected["test_selected"] is True
    first = postprocessing.main(cfg)
    assert first["selected_epoch"] == 0
    assert first["threshold"] == 1.
    assert first["seed_count"] == 1
    assert first["epoch_selection_policy"] == "detector_best"
    assert first["epoch_selection_split"] == "test"
    persisted = json.loads((Path(cfg.postprocessing._task_path) / "report.json").read_text())
    assert persisted["epoch_selection"]["selection_method"] == "best_adp"
    monkeypatch.setattr(pipeline, "build_incidents", lambda *a, **kw: pytest.fail("Rebuilt cached incidents"))
    assert postprocessing.main(cfg) == first
    assert len(reports) == 2  # cached results are still logged


@pytest.mark.parametrize("method,metrics,expected", [
    ("best_adp", {0: (.8, .1), 11: (.8, .9)}, 11),
    ("best_discrimination", {0: (.8, .1), 11: (.1, .9)}, 11),
])
def test_reuses_actual_detector_selection_rule(setup, method, metrics, expected):
    cfg, _ = setup
    cfg.evaluation.best_model_selection = method
    assert run_detector_evaluation(cfg, metrics)["epoch"] == expected
    assert load_detector_selection(cfg)["epoch"] == expected


def test_explicit_epoch_override_and_latest_remain_available(setup):
    from pidsmaker.tasks import postprocessing
    cfg, _ = setup
    # No selection file exists: an explicit epoch must still work.
    cfg.postprocessing.incidents.epoch = 0
    assert postprocessing.main(cfg)["selected_epoch"] == 0
    cfg.postprocessing.incidents.epoch = -1
    cfg.postprocessing.incidents.epoch_selection = "latest"
    report = postprocessing.main(cfg)
    assert report["selected_epoch"] == 11 and report["seed_count"] == 0
    assert report["epoch_test_selected"] is False


@pytest.mark.parametrize("fault", ["missing", "changed_scores", "changed_method", "corrupt"])
def test_missing_or_stale_selection_never_falls_back_to_latest(setup, fault):
    from pidsmaker.tasks import postprocessing
    cfg, _ = setup
    if fault != "missing":
        run_detector_evaluation(cfg)
    if fault == "changed_scores":
        file = pipeline.score_files(cfg.training._edge_losses_dir, "test", 0)[0]
        file.write_text(file.read_text() + "\n")
    elif fault == "changed_method":
        cfg.evaluation.best_model_selection = "best_discrimination"
    elif fault == "corrupt":
        selection_path(cfg).write_text("not JSON")
    with pytest.raises(ValueError, match="force_restart evaluation"):
        postprocessing.main(cfg)


def test_incomplete_best_epoch_never_falls_back_to_complete_latest(setup):
    from pidsmaker.tasks import postprocessing
    cfg, _ = setup
    pipeline.score_files(cfg.training._edge_losses_dir, "val", 0)[0].unlink()
    save_detector_selection(cfg, {"epoch": 0, "adp_score": .8, "discrimination": .7},
                            score_inventory_fingerprint(cfg.training._edge_losses_dir))
    with pytest.raises(pipeline.IncompleteEpoch):
        postprocessing.main(cfg)


def test_selection_provenance_invalidates_cache_even_for_same_epoch(setup):
    from pidsmaker.tasks import postprocessing
    cfg, _ = setup
    run_detector_evaluation(cfg)
    postprocessing.main(cfg)
    path = Path(cfg.postprocessing._task_path) / "report.json"
    first = json.loads(path.read_text())
    cfg.postprocessing.incidents.epoch = 0
    postprocessing.main(cfg)
    second = json.loads(path.read_text())
    assert first["summary"]["selected_epoch"] == second["summary"]["selected_epoch"]
    assert first["fingerprint"] != second["fingerprint"]
    assert second["epoch_selection"]["policy"] == "explicit"


def test_disabled_postprocessing_needs_no_selection_file(setup):
    from pidsmaker.tasks import postprocessing
    cfg, _ = setup
    cfg.postprocessing.incidents.enabled = False
    assert postprocessing.main(cfg) is None


def test_selection_policy_does_not_invalidate_training_or_detector_evaluation(setup):
    from pidsmaker.config.pipeline import set_task_paths
    cfg, _ = setup
    tasks = ["construction", "transformation", "featurization", "feat_inference", "batching", "training", "evaluation", "postprocessing"]
    before = {task: getattr(cfg, task)._task_path for task in tasks}
    cfg.postprocessing.incidents.epoch_selection = "latest"
    set_task_paths(cfg)
    for task in tasks:
        assert (getattr(cfg, task)._task_path == before[task]) == (task != "postprocessing")
