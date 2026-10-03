"""Optional cached incident stage with explicit checkpoint selection provenance."""

from pathlib import Path

from pidsmaker.config.pipeline import ROOT_GROUND_TRUTH_DIR
from pidsmaker.incidents.builder import IncidentBuilderConfig
from pidsmaker.incidents.pipeline import evaluate, relation_score_ids
from pidsmaker.incidents.tracking import log_report
from pidsmaker.utils.dataset_utils import get_rel2id
from pidsmaker.utils.utils import get_split_to_files, log
from pidsmaker.utils.epoch_selection import load_detector_selection


def resolve_incident_epoch(cfg):
    settings = cfg.postprocessing.incidents
    if settings.epoch < -1:
        raise ValueError("Incident epoch must be -1 or a nonnegative epoch index")
    if settings.epoch >= 0:
        return settings.epoch, {"policy": "explicit", "selection_split": "user_supplied", "test_selected": None}
    if settings.epoch_selection == "latest":
        return -1, {"policy": "latest", "selection_split": "none", "test_selected": False}
    if settings.epoch_selection == "detector_best":
        record = load_detector_selection(cfg)
        return record["epoch"], record
    raise ValueError(f"Unknown incident epoch selection: {settings.epoch_selection}")


def main(cfg, force=False):
    settings = cfg.postprocessing.incidents
    if not settings.enabled:
        return None
    if cfg.construction.multi_dataset != "none":
        raise ValueError("Incident postprocessing currently requires a single dataset's original node IDs")
    if cfg._is_node_level or cfg.training.decoder.predict_edge_type.use_triplet_types:
        raise ValueError("Incident postprocessing requires event scores with featurization relation IDs")
    epoch, selection = resolve_incident_epoch(cfg)
    if selection["test_selected"]:
        log("WARNING: Incident epoch follows detector selection on TEST labels; exploratory, not an unbiased held-out result")
    labels_root = Path(ROOT_GROUND_TRUTH_DIR) / cfg.evaluation.ground_truth_version
    report = evaluate(
        get_split_to_files(cfg, cfg.transformation._graphs_dir),
        cfg.training._edge_losses_dir,
        cfg.postprocessing._task_path,
        relation_score_ids(get_rel2id(cfg)),
        config=IncidentBuilderConfig(
            max_time_gap_ns=settings.max_time_gap_ns,
            max_connector_hops=settings.max_connector_hops,
            context_hops=settings.context_hops,
        ),
        epoch=epoch,
        epoch_selection=selection,
        attack_label_paths=[labels_root / p for p in cfg.dataset.ground_truth_relative_path],
        complete_node_labels=settings.complete_node_labels or None,
        force=force,
    )
    log_report(report)
    log(f"Incident epoch {report['summary']['selected_epoch']}; reports: {cfg.postprocessing._task_path}/report.json")
    if report["missing_attack_label_files"]:
        log("WARNING: Some attack label files are absent; coverage uses only available labels")
    return report["summary"]
