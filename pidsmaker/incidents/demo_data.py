"""Create ignored, deterministic local files for the incident viewer."""

import argparse
from pathlib import Path

import networkx as nx
import pandas as pd

from pidsmaker.incidents.__main__ import main as build_incidents_cli


def synthetic_viewer_region(case_count=3, chain_edges=20, background_chains=36):
    """Build connected cases with nearby distractors and unrelated activity."""
    if case_count < 1 or chain_edges < 5 or background_chains < 0:
        raise ValueError("Need at least one case, five chain edges, and nonnegative background")

    graph = nx.MultiDiGraph()
    rows = []
    labels = {}
    base_time = 1_700_000_000_000_000_000
    case_stride = 3 * chain_edges + 100

    def add_event(src, dst, timestamp, edge_type, score):
        graph.add_edge(src, dst, time=timestamp, edge_type=edge_type)
        rows.append({
            "srcnode": src, "dstnode": dst, "time": timestamp,
            "edge_type": edge_type, "loss": score,
        })

    for case_index in range(case_count):
        first_node = 1 + case_index * case_stride
        first_time = base_time + case_index * (chain_edges + 12) * 1_000_000_000
        core_nodes = [first_node + step for step in range(chain_edges + 1)]
        labels.update({node: True for node in core_nodes})
        seed_steps = set(range(0, chain_edges, 4)) | {chain_edges - 1}
        for step in range(chain_edges):
            timestamp = first_time + step * 1_000_000_000
            score = 10.0 if step in seed_steps else 1.0
            add_event(core_nodes[step], core_nodes[step + 1], timestamp, 1, score)

            side_node = first_node + chain_edges + 1 + step
            labels[side_node] = False
            add_event(core_nodes[step], side_node, timestamp + 250_000_000, 2, 0.1)
            if step % 3 == 0:
                tail_node = first_node + 2 * chain_edges + 1 + step
                labels[tail_node] = False
                add_event(side_node, tail_node, timestamp + 500_000_000, 2, 0.1)

    background_first = 1 + case_count * case_stride
    for index in range(background_chains):
        nodes = [background_first + 3 * index + offset for offset in range(3)]
        labels.update({node: False for node in nodes})
        timestamp = base_time + (index % chain_edges) * 1_000_000_000
        add_event(nodes[0], nodes[1], timestamp, 2, 0.1)
        add_event(nodes[1], nodes[2], timestamp + 500_000_000, 2, 0.1)

    scores = pd.DataFrame(rows, columns=["srcnode", "dstnode", "time", "edge_type", "loss"])
    validation = pd.DataFrame({"loss": [1.0, 5.0]})
    return graph, scores, validation, labels


def create_demo(output_dir, background_chains=36, case_count=3, chain_edges=20):
    """Write a new sample region; never overwrite an existing directory."""
    target = Path(output_dir)
    graph, scores, validation, labels = synthetic_viewer_region(
        case_count, chain_edges, background_chains,
    )
    for node, data in graph.nodes(data=True):
        is_malicious = labels[node]
        data["node_type"] = "process" if node % 3 else "file"
        data["label"] = (
            f"demo activity {node}" if is_malicious else f"background activity {node}"
        )
    for index, (_, _, _, data) in enumerate(graph.edges(keys=True, data=True)):
        data["label"] = "EVENT_WRITE" if data["edge_type"] == 1 else "EVENT_READ"
        data["event_uuid"] = f"synthetic-event-{index}"

    target.mkdir(parents=True, exist_ok=False)
    import torch

    graph_path = target / "original_graph.pt"
    scores_path = target / "event_scores.csv"
    validation_path = target / "validation_scores.csv"
    labels_path = target / "synthetic_node_labels.csv"
    incidents_path = target / "incidents.json"
    torch.save(graph, graph_path)
    scores.to_csv(scores_path, index=False)
    validation.to_csv(validation_path, index=False)
    pd.DataFrame({
        "node_id": list(labels),
        "is_malicious": [int(value) for value in labels.values()],
    }).to_csv(labels_path, index=False)
    build_incidents_cli([
        "--graph", str(graph_path),
        "--scores", str(scores_path),
        "--validation-scores", str(validation_path),
        "--output", str(incidents_path),
    ])
    return target


def main(argv=None):
    parser = argparse.ArgumentParser(description="Generate an ignored local incident demo")
    parser.add_argument("--output-dir", default="artifacts/incident_demo_large")
    parser.add_argument("--cases", type=int, default=3)
    parser.add_argument("--chain-edges", type=int, default=20)
    parser.add_argument("--background-chains", type=int, default=36)
    args = parser.parse_args(argv)
    if args.cases < 1 or args.chain_edges < 5 or args.background_chains < 0:
        parser.error("Need --cases >= 1, --chain-edges >= 5, and --background-chains >= 0")
    target = create_demo(args.output_dir, args.background_chains, args.cases, args.chain_edges)
    print(f"Created synthetic viewer files in {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
