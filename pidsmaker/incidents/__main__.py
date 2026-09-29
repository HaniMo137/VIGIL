"""Offline entry point: python -m pidsmaker.incidents --help."""

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path

import networkx as nx
import pandas as pd

from pidsmaker.incidents.builder import IncidentBuilderConfig, build_incidents
from pidsmaker.incidents.seeds import calculate_seed_threshold


def _load_graph(path):
    # torch.load can execute pickle code: accept trusted local artifacts only.
    import torch

    return torch.load(path, map_location="cpu", weights_only=False)


def _json_value(value):
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"Cannot export {type(value).__name__} as JSON")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Build candidate incidents from a trusted graph")
    parser.add_argument("--graph", required=True, help="Trusted PIDSMaker torch-saved MultiDiGraph")
    parser.add_argument("--scores", required=True, help="Score CSV for the same provenance region")
    thresholds = parser.add_mutually_exclusive_group(required=True)
    thresholds.add_argument("--validation-scores", help="Validation CSV; uses maximum loss")
    thresholds.add_argument("--threshold", type=float, help="Already frozen validation threshold")
    parser.add_argument("--relation-map", help="JSON object: operation label to score edge_type ID")
    parser.add_argument("--config", help="JSON object with IncidentBuilderConfig fields")
    parser.add_argument("--max-time-gap-seconds", type=float)
    parser.add_argument("--max-connector-hops", type=int)
    parser.add_argument("--output", required=True, help="New JSON output file; existing files are protected")
    args = parser.parse_args(argv)

    output = Path(args.output)
    if output.exists():
        parser.error(f"Output already exists: {output}; choose a new filename")
    settings = {}
    if args.config:
        settings = json.loads(Path(args.config).read_text())
        if "relation_penalties" in settings:
            settings["relation_penalties"] = {
                int(key): value for key, value in settings["relation_penalties"].items()
            }
    if args.max_time_gap_seconds is not None:
        if not math.isfinite(args.max_time_gap_seconds) or args.max_time_gap_seconds < 0:
            parser.error("Maximum time gap must be finite and nonnegative")
        settings["max_time_gap_ns"] = int(args.max_time_gap_seconds * 1_000_000_000)
    if args.max_connector_hops is not None:
        settings["max_connector_hops"] = args.max_connector_hops
    config = IncidentBuilderConfig(**settings)
    relation_map = json.loads(Path(args.relation_map).read_text()) if args.relation_map else None
    threshold = args.threshold
    if args.validation_scores:
        threshold = calculate_seed_threshold(pd.read_csv(args.validation_scores))
    incidents = build_incidents(
        _load_graph(args.graph), pd.read_csv(args.scores), threshold,
        config, relation_to_id=relation_map,
    )
    payload = {
        "schema_version": 1,
        "status": "unverified_candidates",
        "threshold": threshold,
        "config": asdict(config),
        "incidents": [
            {
                "incident_id": index,
                "start_time": incident.start_time,
                "end_time": incident.end_time,
                "seed_edges": [asdict(edge) for edge in incident.seed_edges],
                "connector_edges": [asdict(edge) for edge in incident.connector_edges],
                "context_edges": [asdict(edge) for edge in incident.context_edges],
                "graph": nx.node_link_data(incident.graph),
            }
            for index, incident in enumerate(incidents)
        ],
    }
    serialized = json.dumps(payload, indent=2, default=_json_value, allow_nan=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        stream.write(serialized + "\n")
    print(f"Built {len(incidents)} candidate incidents; threshold={threshold}; output={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
