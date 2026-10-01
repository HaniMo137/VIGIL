"""Once-per-run incident evaluation of trusted local detector artifacts.

Score rows are spooled to SQLite, so dataset-sized score frames and provenance
graphs need not coexist in RAM. Window boundaries come from transformed graphs,
not CSV filenames (which can instead describe inference batches).
"""

import csv
import hashlib
import json
import math
import re
import sqlite3
import tempfile
import time
from collections import Counter
from contextlib import closing
from dataclasses import asdict
from pathlib import Path

import networkx as nx
import pandas as pd

from .__main__ import _json_value, _load_graph
from .benchmark import load_attack_labels, load_complete_node_labels, quality_metrics
from .builder import IncidentBuilderConfig, _integer, build_incidents, prepare_provenance_graph


COLUMNS = ["srcnode", "dstnode", "time", "edge_type", "loss"]
REGIONS = ("seeds", "core", "with_context")
SCHEMA_VERSION = 1


class IncompleteEpoch(ValueError):
    """An epoch has not produced scores for all expected events yet."""


def relation_score_ids(rel2id):
    """Use exactly the one-hot encoding and argmax+1 used by PIDSMaker.

    In particular, ATLAS relation zero maps to the LAST one-hot column.
    """
    from pidsmaker.utils.utils import gen_relation_onehot

    encoded = gen_relation_onehot(rel2id)
    return {label: int(encoded[label].argmax()) + 1
            for label in rel2id if isinstance(label, str)}


def encoded_graph(path, relation_ids):
    graph = _load_graph(path)
    if not isinstance(graph, nx.MultiDiGraph):
        raise ValueError(f"Expected a MultiDiGraph: {path}")
    for _, _, data in graph.edges(data=True):
        if "label" in data:
            if data["label"] not in relation_ids:
                raise ValueError(f"Unknown featurization relation {data['label']!r}")
            data["edge_type"] = relation_ids[data["label"]]
        elif "edge_type" not in data:
            # feat_inference encodes label-less transformed edges as all zeros;
            # inference exports argmax+1, which is 1 in that case.
            data["edge_type"] = 1
    return graph


def score_files(root, split, epoch):
    return sorted((Path(root) / split / f"model_epoch_{epoch}").glob("*.csv"))


def epoch_candidates(root, override=-1):
    if override < -1:
        raise ValueError("Incident epoch must be -1 or a nonnegative epoch index")
    if override != -1:
        return [override]
    epochs = set()
    for split in ("val", "test"):
        for path in (Path(root) / split).glob("model_epoch_*"):
            match = re.fullmatch(r"model_epoch_(\d+)", path.name)
            if match and path.is_dir():
                epochs.add(int(match.group(1)))
    return sorted(epochs, reverse=True)


def align_split(graph_paths, files, relation_ids, output):
    """Require a bijection between score rows and graph events; save each window.

    Empty windows are valid. Duplicated or ambiguous events are errors, never
    evidence for selecting a different epoch. Only incomplete epochs fall back.
    """
    output.mkdir(parents=True, exist_ok=True)
    if not files:
        raise IncompleteEpoch("Missing score CSVs")
    with tempfile.TemporaryDirectory(prefix="score-index-", dir=output) as temp:
        with closing(sqlite3.connect(str(Path(temp) / "scores.sqlite"))) as db:
            db.execute("CREATE TABLE scores (id INTEGER PRIMARY KEY, time INTEGER, payload TEXT, assigned INTEGER DEFAULT 0)")
            maximum = None
            for file in files:
                for chunk in pd.read_csv(file, chunksize=50000, dtype={
                    **{name: str for name in COLUMNS[:-1]}, "event_uuid": str,
                }):
                    missing = set(COLUMNS) - set(chunk.columns)
                    if missing:
                        raise ValueError(f"Missing required columns in {file}: {sorted(missing)}")
                    records = []
                    for row in chunk.to_dict("records"):
                        for name in COLUMNS[:-1]:
                            row[name] = _integer(row[name], name)
                        row["loss"] = float(row["loss"])
                        if not math.isfinite(row["loss"]) or row["loss"] < 0:
                            raise ValueError("Anomaly losses must be finite and nonnegative")
                        # Empty optional identity fields must remain genuinely absent.
                        row = {k: v for k, v in row.items() if not pd.isna(v)}
                        maximum = row["loss"] if maximum is None else max(maximum, row["loss"])
                        records.append((row["time"], json.dumps(row, default=_json_value)))
                    db.executemany("INSERT INTO scores(time,payload) VALUES (?,?)", records)
            db.execute("CREATE INDEX score_time ON scores(time)")
            windows = []
            for index, path in enumerate(graph_paths):
                graph = encoded_graph(path, relation_ids)
                signatures = {
                    (_integer(u, "srcnode"), _integer(v, "dstnode"),
                     _integer(d["time"], "time"), _integer(d["edge_type"], "edge_type"))
                    for u, v, d in graph.edges(data=True)
                }
                records, ids = [], []
                if signatures:
                    times = [event[2] for event in signatures]
                    for row_id, payload, assigned in db.execute(
                        "SELECT id,payload,assigned FROM scores WHERE time BETWEEN ? AND ?",
                        (min(times), max(times)),
                    ):
                        row = json.loads(payload)
                        if tuple(row[name] for name in COLUMNS[:-1]) in signatures:
                            if assigned:
                                raise ValueError("Scored event matches multiple provenance windows")
                            records.append(row)
                            ids.append((row_id,))
                scores = pd.DataFrame(records) if records else pd.DataFrame(columns=COLUMNS)
                # This checks keys/UUIDs, parallel-edge ambiguity and duplicate rows.
                prepare_provenance_graph(graph, scores)
                if len(scores) != graph.number_of_edges():
                    raise IncompleteEpoch(f"Missing event scores for graph {path}")
                db.executemany("UPDATE scores SET assigned=1 WHERE id=?", ids)
                target = output / f"window_{index:05d}.csv"
                scores.to_csv(target, index=False)
                windows.append({"window": index, "graph": str(Path(path).resolve()),
                                "scores": str(target.relative_to(output.parent))})
            if db.execute("SELECT COUNT(*) FROM scores WHERE assigned=0").fetchone()[0]:
                raise ValueError("Score rows do not match any configured provenance graph")
            return windows, maximum


def select_epoch(graphs, score_root, relation_ids, directory, override=-1):
    """Select by completeness alone, before consulting any attack labels."""
    failures = []
    for epoch in epoch_candidates(score_root, override):
        with tempfile.TemporaryDirectory(prefix=f"epoch-{epoch}-", dir=directory) as temp:
            target = Path(temp)
            try:
                _, threshold = align_split(graphs["val"], score_files(score_root, "val", epoch),
                                           relation_ids, target / "val")
                if threshold is None:
                    raise IncompleteEpoch("Validation scores are empty; cannot freeze a threshold")
                windows, _ = align_split(graphs["test"], score_files(score_root, "test", epoch),
                                         relation_ids, target / "test")
            except IncompleteEpoch as exc:
                failures.append(f"epoch {epoch}: {exc}")
                continue
            destination = directory / "aligned"
            target.rename(destination)
            return epoch, threshold, windows, failures
    raise IncompleteEpoch("No complete validation/test scored epoch. " + "; ".join(failures))


def node_sets(incidents):
    seeds, core, context = set(), set(), set()
    for incident in incidents:
        for edge in incident.seed_edges:
            seeds.update((edge.srcnode, edge.dstnode))
        for edge in incident.seed_edges + incident.connector_edges:
            core.update((edge.srcnode, edge.dstnode))
        context.update(incident.graph.nodes)
    return dict(zip(REGIONS, (seeds, core, context)))


class CoverageAccumulator:
    """Keep node unions and counters, never retain all incident graphs."""

    def __init__(self, attacks=None, complete_labels=None):
        self.attacks = attacks or {}
        self.complete_labels = complete_labels
        if complete_labels is not None and any(
            node in complete_labels and not complete_labels[node]
            for nodes in self.attacks.values() for node in nodes
        ):
            raise ValueError("Attack labels conflict with complete node labels")
        memberships = Counter(n for nodes in self.attacks.values() for n in nodes)
        self.ambiguous = {n for n, count in memberships.items() if count > 1}
        self.unique_attacks = {name: nodes - self.ambiguous for name, nodes in self.attacks.items()}
        self.region = set()
        self.nodes = {name: set() for name in REGIONS}
        self.fragmentation = Counter()
        self.mixed = 0

    def add(self, graph, incidents):
        # Reuse validation for verified complete labels and conflicting positives.
        metrics = quality_metrics(incidents, graph.nodes, self.attacks, self.complete_labels)
        region = {int(n) for n in graph.nodes}
        if not any(nodes & region for nodes in self.unique_attacks.values()):
            metrics["mixed_attack_incidents"] = None
        self.region.update(region)
        for name, nodes in node_sets(incidents).items():
            self.nodes[name].update(nodes)
        for incident in incidents:
            found = [name for name, nodes in self.unique_attacks.items() if nodes & set(incident.graph.nodes)]
            self.fragmentation.update(found)
            self.mixed += len(found) > 1
        return metrics

    def reports(self):
        known = set().union(*self.attacks.values()) if self.attacks else set()
        labeled = self.complete_labels is not None or bool(self.attacks)
        malicious = ({n for n, value in self.complete_labels.items() if value} | known
                     if self.complete_labels is not None else known)
        present = malicious & self.region
        fraction = lambda num, den: num / den if den else None
        summary = {
            "labels_available": labeled,
            "processed_nodes": len(self.region),
            "labeled_malicious_present": len(present) if labeled else None,
            "labeled_malicious_absent": len(malicious - self.region) if labeled else None,
            "ambiguous_attack_nodes": len(self.ambiguous & self.region),
            "mixed_attack_incidents": self.mixed if any(
                nodes & self.region for nodes in self.unique_attacks.values()
            ) else None,
        }
        for name, nodes in self.nodes.items():
            summary[f"nodes_{name}"] = len(nodes)
            summary[f"malicious_coverage_{name}"] = fraction(len(nodes & present), len(present)) if labeled else None
            summary[f"verified_benign_fraction_{name}"] = (
                fraction(len(nodes - malicious), len(nodes)) if self.complete_labels is not None else None
            )
        attacks = []
        for name, nodes in self.attacks.items():
            available = nodes & self.region
            row = {"attack": name, "labeled_present": len(available),
                   "labeled_absent": len(nodes - self.region),
                   "incident_fragments": self.fragmentation[name]
                   if self.unique_attacks[name] & self.region else None}
            for kind in REGIONS:
                row[f"coverage_{kind}"] = fraction(len(nodes & self.nodes[kind]), len(available))
            attacks.append(row)
        return summary, attacks


def incident_payload(incidents, threshold, config):
    return {
        "schema_version": 1, "status": "unverified_candidates",
        "threshold": threshold, "config": asdict(config),
        "incidents": [{
            "incident_id": i, "start_time": incident.start_time, "end_time": incident.end_time,
            "seed_edges": [asdict(e) for e in incident.seed_edges],
            "connector_edges": [asdict(e) for e in incident.connector_edges],
            "context_edges": [asdict(e) for e in incident.context_edges],
            "graph": nx.node_link_data(incident.graph),
        } for i, incident in enumerate(incidents)],
    }


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, default=_json_value, allow_nan=False) + "\n")


def write_csv(path, rows, columns):
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def input_fingerprint(graphs, score_root, relation_ids, config, epoch, label_paths, complete_labels):
    def inventory(paths):
        return [(str(Path(p).resolve()), Path(p).stat().st_size, Path(p).stat().st_mtime_ns) for p in paths]
    graph_inventory = {split: inventory(graphs[split]) for split in ("val", "test")}
    score_inventory = inventory(sorted(Path(score_root).glob("*/model_epoch_*/*.csv")))
    labels = [(str(Path(p).resolve()), hashlib.sha256(Path(p).read_bytes()).hexdigest()
               if Path(p).is_file() else None) for p in list(label_paths) + ([complete_labels] if complete_labels else [])]
    data = [SCHEMA_VERSION, graph_inventory, score_inventory, relation_ids, asdict(config), epoch, labels]
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


def evaluate(graphs, score_root, output, relation_ids, *, config=None, epoch=-1,
             attack_label_paths=(), complete_node_labels=None, force=False):
    """Save reports and local viewer artifacts, or reload a valid cached report.

    Missing configured attack files are reported, not treated as benign labels.
    Metric coverage then applies only to the available files. A missing explicit
    complete-label file is an error. Input graph files must be trusted pickles.
    """
    config = config or IncidentBuilderConfig()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    fingerprint = input_fingerprint(graphs, score_root, relation_ids, config, epoch,
                                    attack_label_paths, complete_node_labels)
    report_path = output / "report.json"
    if report_path.exists() and not force:
        cached = json.loads(report_path.read_text())
        if cached.get("fingerprint") == fingerprint and all(
            (output / p).is_file() for p in cached["local_files"]
        ):
            return cached
    # Publish only completed generations. A failed evaluation leaves the last
    # successful report intact and TemporaryDirectory cleans its own scratch.
    with tempfile.TemporaryDirectory(prefix="evaluation-", dir=output) as temp:
        directory = Path(temp)
        selected, threshold, windows, skipped = select_epoch(graphs, score_root, relation_ids, directory, epoch)
        available = [p for p in attack_label_paths if Path(p).is_file()]
        missing = [str(p) for p in attack_label_paths if not Path(p).is_file()]
        attacks = load_attack_labels(available)
        complete = load_complete_node_labels(complete_node_labels) if complete_node_labels else None
        accumulator = CoverageAccumulator(attacks, complete)
        rows, local_files = [], []
        for window in windows:
            graph = encoded_graph(window["graph"], relation_ids)
            scores_path = directory / "aligned" / window["scores"]
            scores = pd.read_csv(scores_path, dtype={"event_uuid": str})
            started = time.perf_counter()
            incidents = build_incidents(graph, scores, threshold, config)
            runtime = time.perf_counter() - started
            metrics = accumulator.add(graph, incidents)
            target = f"incidents_{window['window']:05d}.json"
            write_json(directory / target, incident_payload(incidents, threshold, config))
            row = {"window": window["window"], "graph": window["graph"],
                   "scores": "aligned/" + window["scores"], "incidents": target,
                   "graph_nodes": graph.number_of_nodes(), "graph_events": graph.number_of_edges(),
                   "incident_count": len(incidents),
                   "seed_count": sum(len(i.seed_edges) for i in incidents),
                   "connector_count": sum(len(i.connector_edges) for i in incidents),
                   "context_count": sum(len(i.context_edges) for i in incidents),
                   "runtime_seconds": runtime,
                   **{k: v for k, v in metrics.items() if k != "attack_fragmentation"}}
            rows.append(row)
            local_files.extend([target, row["scores"]])
        summary, attack_rows = accumulator.reports()
        for metric in ("incident_count", "seed_count", "connector_count", "context_count", "runtime_seconds"):
            summary[metric] = sum(row[metric] for row in rows)
        summary.update(selected_epoch=selected, threshold=threshold, window_count=len(rows))
        report = {
            "schema_version": SCHEMA_VERSION, "fingerprint": fingerprint,
            "summary": summary, "windows": rows, "attacks": attack_rows,
            "missing_attack_label_files": missing, "skipped_incomplete_epochs": skipped,
            "config": asdict(config), "relation_ids": relation_ids,
            "scope": "Incidents are built separately within each test window; node coverage uses dataset-wide unions. Fragment counts count separate window incidents. Shared attack nodes are excluded from fragmentation/mixing.",
            "runtime_scope": "One builder call per window, including graph preparation; excludes file I/O, metric evaluation, and W&B logging.",
        }
        generation = "results-" + directory.name.removeprefix("evaluation-")
        for row in rows:
            row["scores"] = generation + "/" + row["scores"]
            row["incidents"] = generation + "/" + row["incidents"]
        write_json(directory / "relation_ids.json", relation_ids)
        write_csv(directory / "summary.csv", [summary], list(summary))
        write_csv(directory / "windows.csv", rows, list(rows[0]) if rows else ["window", "incident_count", "runtime_seconds"])
        write_csv(directory / "attacks.csv", attack_rows, list(attack_rows[0]) if attack_rows else ["attack", "labeled_present", "labeled_absent", "incident_fragments", *[f"coverage_{r}" for r in REGIONS]])
        local_files.extend(["relation_ids.json", "summary.csv", "windows.csv", "attacks.csv"])
        report["local_files"] = [generation + "/" + p for p in local_files]
        report["results_directory"] = generation
        write_json(directory / "report.json", report)
        if fingerprint != input_fingerprint(graphs, score_root, relation_ids, config, epoch,
                                            attack_label_paths, complete_node_labels):
            raise RuntimeError("Evaluation inputs changed while processing; retry with immutable artifacts")
        directory.rename(output / generation)
        # Atomic pointer/report replacement; old generations stay recoverable.
        staging = output / (generation + ".json")
        write_json(staging, report)
        staging.replace(report_path)
        # Match JSON types on cache hits and fresh evaluations (tuple -> list).
        return json.loads(report_path.read_text())
