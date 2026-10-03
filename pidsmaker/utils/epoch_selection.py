"""Persist detector checkpoint selection independently of an active W&B run."""

import hashlib
import json
import math
import os
import tempfile
from pathlib import Path


def score_inventory_fingerprint(root):
    root = Path(root)
    inventory = [(str(path.relative_to(root)), path.stat().st_size, path.stat().st_mtime_ns)
                 for split in ("val", "test")
                 for path in sorted((root / split).glob("model_epoch_*/*.csv"))]
    return hashlib.sha256(json.dumps(inventory).encode()).hexdigest()


def selection_path(cfg):
    return Path(cfg.evaluation._task_path) / "selected_epoch.json"


def save_detector_selection(cfg, stats, fingerprint):
    if stats is None or not isinstance(stats.get("epoch"), int) or stats["epoch"] < 0:
        raise ValueError("Detector evaluation did not select a valid epoch")
    if fingerprint != score_inventory_fingerprint(cfg.training._edge_losses_dir):
        raise RuntimeError("Score artifacts changed during detector evaluation; rerun evaluation")
    record = {
        "schema_version": 1,
        "epoch": stats["epoch"],
        "policy": "detector_best",
        "selection_method": cfg.evaluation.best_model_selection,
        "selection_split": "test",
        "test_selected": True,
        "score_fingerprint": fingerprint,
        "metrics": {key: float(stats[key]) for key in ("adp_score", "discrimination")},
    }
    if not all(math.isfinite(value) for value in record["metrics"].values()):
        raise ValueError("Selected detector metrics must be finite")
    path = selection_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, suffix=".json", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            json.dump(record, stream, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return record


def load_detector_selection(cfg):
    instruction = "Rerun with --force_restart evaluation (no retraining), or choose --postprocessing.incidents.epoch explicitly."
    path = selection_path(cfg)
    if not path.is_file():
        raise ValueError(f"Missing detector epoch selection at {path}. {instruction}")
    try:
        record = json.loads(path.read_text())
    except (ValueError, OSError) as exc:
        raise ValueError(f"Cannot read detector epoch selection. {instruction}") from exc
    if (not isinstance(record, dict) or record.get("schema_version") != 1
            or type(record.get("epoch")) is not int or record["epoch"] < 0
            or record.get("policy") != "detector_best"
            or record.get("selection_split") != "test"
            or record.get("test_selected") is not True
            or record.get("selection_method") != cfg.evaluation.best_model_selection
            or record.get("score_fingerprint") != score_inventory_fingerprint(cfg.training._edge_losses_dir)):
        raise ValueError(f"Invalid or stale detector epoch selection. {instruction}")
    return record
