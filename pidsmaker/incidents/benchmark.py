"""Reproducible engineering benchmarks for candidate incident construction.

Run ``python -m pidsmaker.incidents.benchmark --help``. Ground truth is read
only after building; it is never passed to seed selection or the builder.
"""

import argparse
import json
import math
import statistics
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Mapping, Optional, Set

import networkx as nx
import pandas as pd

from pidsmaker.incidents.__main__ import _json_value, _load_graph
from pidsmaker.incidents.builder import IncidentBuilderConfig, build_incidents
from pidsmaker.incidents.seeds import calculate_seed_threshold, select_seed_edges


def synthetic_region(background_chains: int = 200):
    """Return deterministic toy data with two attacks and ordinary background."""
    if background_chains < 0:
        raise ValueError("background_chains must be nonnegative")
    base = 1_700_000_000_000_000_000
    graph = nx.MultiDiGraph()
    rows = []
    attacks = {}
    labels = {}

    def add_event(src, dst, timestamp, edge_type, score):
        graph.add_edge(src, dst, time=timestamp, edge_type=edge_type)
        rows.append(dict(srcnode=src, dstnode=dst, time=timestamp,
                         edge_type=edge_type, loss=score))

    for attack_id in range(2):
        nodes = [1 + attack_id * 10 + index for index in range(6)]
        attacks[f"attack_{attack_id}"] = set(nodes)
        labels.update({node: True for node in nodes})
        for step in range(5):
            score = 10.0 if step in (0, 4) else 1.0
            add_event(nodes[step], nodes[step + 1], base + step * 1_000_000_000, 1, score)
        distractor = 100 + attack_id
        labels[distractor] = False
        add_event(nodes[1], distractor, base + 2_000_000_000, 2, 0.1)

    for index in range(background_chains):
        nodes = [1_000 + 3 * index + offset for offset in range(3)]
        labels.update({node: False for node in nodes})
        add_event(nodes[0], nodes[1], base + 1_000_000_000, 2, 0.1)
        add_event(nodes[1], nodes[2], base + 2_000_000_000, 2, 0.1)

    scores = pd.DataFrame(rows, columns=["srcnode", "dstnode", "time", "edge_type", "loss"])
    validation = pd.DataFrame({"loss": [1.0, 5.0]})
    return graph, scores, validation, attacks, labels


def load_attack_labels(paths) -> Dict[str, Set[int]]:
    """Read PIDSMaker orthrus/reapr files: UUID, description, integer node ID."""
    attacks = {}
    for path in paths:
        name = Path(path).stem
        if name in attacks:
            raise ValueError(f"Duplicate attack label name: {name}")
        frame = pd.read_csv(path, header=None, usecols=[2], dtype=str)
        identifiers = set()
        for value in frame.iloc[:, 0]:
            if str(value).strip().lower() == "node_id":
                continue
            try:
                identifiers.add(int(value))
            except (ValueError, TypeError) as exc:
                raise ValueError(f"Attack labels in {path} need integer node IDs") from exc
        if not identifiers:
            raise ValueError(f"No attack node IDs in {path}")
        attacks[name] = identifiers
    return attacks


def load_complete_node_labels(path) -> Dict[int, bool]:
    """Read an explicitly complete node_id,is_malicious CSV (0/1 values)."""
    frame = pd.read_csv(path, dtype=str)
    if not {"node_id", "is_malicious"}.issubset(frame.columns):
        raise ValueError("Node label CSV needs node_id,is_malicious columns")
    labels = {}
    for node, value in frame[["node_id", "is_malicious"]].itertuples(index=False, name=None):
        try:
            node_id = int(node)
        except (ValueError, TypeError) as exc:
            raise ValueError("Node labels need integer node IDs") from exc
        if value not in ("0", "1"):
            raise ValueError("is_malicious must be 0 or 1")
        if node_id in labels:
            raise ValueError(f"Duplicate node label: {node_id}")
        labels[node_id] = value == "1"
    return labels


def _fraction(numerator, denominator):
    return numerator / denominator if denominator else None


def quality_metrics(incidents, region_nodes, attack_labels=None, complete_labels=None):
    """Evaluate mapped nodes; never treat missing labels as verified benign."""
    region_nodes = {int(node) for node in region_nodes}
    attack_labels = attack_labels or {}
    mapped_attacks = {
        name: set(nodes) & region_nodes for name, nodes in attack_labels.items()
    }
    membership = {}
    for nodes in mapped_attacks.values():
        for node in nodes:
            membership[node] = membership.get(node, 0) + 1
    ambiguous_attack_nodes = {node for node, count in membership.items() if count > 1}
    unique_attacks = {
        name: nodes - ambiguous_attack_nodes for name, nodes in mapped_attacks.items()
    }
    known_attack_nodes = set().union(*mapped_attacks.values()) if mapped_attacks else set()
    if complete_labels is not None:
        missing = region_nodes - complete_labels.keys()
        if missing:
            raise ValueError(f"Complete node labels miss {len(missing)} provenance nodes")
        positives = {node for node in region_nodes if complete_labels[node]}
        if not known_attack_nodes <= positives:
            raise ValueError("Attack labels conflict with complete node labels")
        malicious = positives
        known_benign = region_nodes - positives
    else:
        malicious = known_attack_nodes if attack_labels else None
        known_benign = None

    all_nodes = set().union(*(set(incident.graph.nodes) for incident in incidents)) if incidents else set()
    core_nodes = {
        node for incident in incidents
        for edge in incident.seed_edges + incident.connector_edges
        for node in (edge.srcnode, edge.dstnode)
    }
    seed_nodes = {
        node for incident in incidents for edge in incident.seed_edges
        for node in (edge.srcnode, edge.dstnode)
    }
    result = {
        "candidate_nodes": len(all_nodes),
        "core_nodes": len(core_nodes),
        "malicious_nodes_in_region": len(malicious) if malicious is not None else None,
        "malicious_coverage_seeds": _fraction(len(seed_nodes & malicious), len(malicious))
        if malicious is not None else None,
        "malicious_coverage_core": _fraction(len(core_nodes & malicious), len(malicious))
        if malicious is not None else None,
        "malicious_coverage_with_context": _fraction(len(all_nodes & malicious), len(malicious))
        if malicious is not None else None,
        "verified_benign_fraction_core": _fraction(len(core_nodes & known_benign), len(core_nodes))
        if known_benign is not None else None,
        "verified_benign_fraction_with_context": _fraction(len(all_nodes & known_benign), len(all_nodes))
        if known_benign is not None else None,
        "attack_labels_matched": sum(bool(nodes) for nodes in mapped_attacks.values()),
        "attack_labels_unmatched_nodes": len(
            set().union(*(set(nodes) - region_nodes for nodes in attack_labels.values()))
        ) if attack_labels else 0,
        "ambiguous_attack_nodes": len(ambiguous_attack_nodes),
    }
    if mapped_attacks:
        fragmentation = {
            name: sum(bool(set(incident.graph.nodes) & nodes) for incident in incidents)
            if nodes else None
            for name, nodes in unique_attacks.items() if mapped_attacks[name]
        }
        result["attack_fragmentation"] = fragmentation
        result["mixed_attack_incidents"] = sum(
            sum(bool(set(incident.graph.nodes) & nodes) for nodes in unique_attacks.values()) > 1
            for incident in incidents
        )
    else:
        result["attack_fragmentation"] = None
        result["mixed_attack_incidents"] = None
    return result


def benchmark_configs(
    graph, scores, threshold, configurations: Mapping[str, IncidentBuilderConfig],
    *, repeats=3, relation_to_id=None, attack_labels=None, complete_labels=None,
):
    """Time builder-only work; test labels are consulted only after each run."""
    if repeats < 1:
        raise ValueError("repeats must be positive")
    if not configurations:
        raise ValueError("At least one configuration is required")
    selected = select_seed_edges(scores, threshold)
    report = {}
    for name, config in configurations.items():
        runtimes = []
        first_summary = None
        for _ in range(repeats):
            started = time.perf_counter()
            incidents = build_incidents(graph, scores, threshold, config,
                                        relation_to_id=relation_to_id)
            runtimes.append(time.perf_counter() - started)
            summary = {
                "incident_count": len(incidents),
                "seed_count": sum(len(incident.seed_edges) for incident in incidents),
                "connector_count": sum(len(incident.connector_edges) for incident in incidents),
                "context_count": sum(len(incident.context_edges) for incident in incidents),
                "seed_retention": _fraction(
                    sum(len(incident.seed_edges) for incident in incidents), len(selected)
                ),
            }
            summary.update(quality_metrics(incidents, graph.nodes, attack_labels, complete_labels))
            if first_summary is not None and summary != first_summary:
                raise RuntimeError("Incident results changed between benchmark repetitions")
            first_summary = summary
        median = statistics.median(runtimes)
        report[name] = {
            "config": asdict(config),
            "metrics": first_summary,
            "runtime_seconds": {"median": median, "runs": runtimes},
            "events_per_second": graph.number_of_edges() / median if median > 0 else None,
        }
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description="Benchmark incident construction")
    sources = parser.add_mutually_exclusive_group(required=True)
    sources.add_argument("--synthetic", action="store_true", help="Deterministic toy workload")
    sources.add_argument("--graph", help="Trusted torch-saved MultiDiGraph")
    parser.add_argument("--scores", help="Matching event-score CSV")
    parser.add_argument("--validation-scores", help="Separate validation-score CSV")
    parser.add_argument("--threshold", type=float, help="Frozen validation threshold")
    parser.add_argument("--relation-map", help="Label-to-score-ID JSON")
    parser.add_argument("--attack-label", action="append", default=[],
                        help="Repeat per-attack PIDSMaker CSV (UUID,label,node_id)")
    parser.add_argument("--complete-node-labels", help="CSV: node_id,is_malicious for every region node")
    parser.add_argument("--split", choices=["validation", "test"], help="Real-data split")
    parser.add_argument("--configs", help="JSON object mapping strategy names to config fields")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--synthetic-background-chains", type=int, default=200)
    parser.add_argument("--output", required=True, help="New JSON output file")
    args = parser.parse_args(argv)
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    if args.synthetic:
        if args.synthetic_background_chains < 0:
            parser.error("--synthetic-background-chains must be nonnegative")
        if any((args.scores, args.validation_scores, args.threshold is not None,
                args.relation_map, args.attack_label, args.complete_node_labels, args.split)):
            parser.error("Synthetic mode generates its own inputs and labels")
        graph, scores, validation, attacks, labels = synthetic_region(args.synthetic_background_chains)
        threshold = calculate_seed_threshold(validation)
        source = "synthetic"
        relation_map = None
    else:
        if not args.scores or not args.split:
            parser.error("Real-data mode requires --scores and --split")
        if (args.validation_scores is None) == (args.threshold is None):
            parser.error("Choose exactly one of --validation-scores or --threshold")
        graph = _load_graph(args.graph)
        scores = pd.read_csv(args.scores)
        threshold = (
            calculate_seed_threshold(pd.read_csv(args.validation_scores))
            if args.validation_scores else args.threshold
        )
        attacks = load_attack_labels(args.attack_label) if args.attack_label else None
        labels = load_complete_node_labels(args.complete_node_labels) if args.complete_node_labels else None
        relation_map = json.loads(Path(args.relation_map).read_text()) if args.relation_map else None
        source = args.split
    configurations = {"baseline": IncidentBuilderConfig()}
    if args.configs:
        raw = json.loads(Path(args.configs).read_text())
        configurations = {
            name: IncidentBuilderConfig(**{
                **settings,
                **({"relation_penalties": {int(k): v for k, v in settings["relation_penalties"].items()}}
                   if "relation_penalties" in settings else {}),
            })
            for name, settings in raw.items()
        }
    if source == "test" and len(configurations) != 1:
        parser.error("Test split accepts one frozen configuration; compare strategies on validation")
    output = Path(args.output)
    if output.exists():
        parser.error(f"Output already exists: {output}")
    report = benchmark_configs(
        graph, scores, threshold, configurations, repeats=args.repeats,
        relation_to_id=relation_map, attack_labels=attacks, complete_labels=labels,
    )
    payload = {
        "benchmark_type": "synthetic_engineering" if args.synthetic else "real_data",
        "split": source,
        "threshold": threshold,
        "region_events": graph.number_of_edges(),
        "region_nodes": graph.number_of_nodes(),
        "repeats": args.repeats,
        "results": report,
    }
    serialized = json.dumps(payload, indent=2, default=_json_value, allow_nan=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        stream.write(serialized + "\n")
    for name, result in report.items():
        metrics = result["metrics"]
        print(
            f"{name}: incidents={metrics['incident_count']} "
            f"core_coverage={metrics['malicious_coverage_core']} "
            f"median_seconds={result['runtime_seconds']['median']:.4f}"
        )
    print(f"Saved benchmark: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
