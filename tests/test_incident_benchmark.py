import json

import pandas as pd
import pytest

from pidsmaker.incidents import IncidentBuilderConfig
from pidsmaker.incidents.benchmark import (
    benchmark_configs,
    load_attack_labels,
    load_complete_node_labels,
    main,
    quality_metrics,
    synthetic_region,
)


def test_synthetic_workload_is_deterministic_and_fully_labeled():
    first = synthetic_region(background_chains=3)
    second = synthetic_region(background_chains=3)
    graph, scores, validation, attacks, labels = first
    assert graph.number_of_edges() == 18
    assert scores.equals(second[1])
    assert validation.equals(second[2])
    assert attacks == second[3]
    assert labels == second[4]
    assert set(labels) == set(graph.nodes)
    assert sum(labels.values()) == 12


def test_baseline_benchmark_measures_coverage_contamination_and_runtime():
    graph, scores, _, attacks, labels = synthetic_region(background_chains=2)
    report = benchmark_configs(
        graph, scores, 5, {"baseline": IncidentBuilderConfig()},
        repeats=2, attack_labels=attacks, complete_labels=labels,
    )["baseline"]
    metrics = report["metrics"]
    assert metrics["incident_count"] == 2
    assert metrics["seed_count"] == 4
    assert metrics["connector_count"] == 6
    assert metrics["context_count"] == 0
    assert metrics["seed_retention"] == 1
    assert metrics["malicious_coverage_seeds"] == pytest.approx(4 / 6)
    assert metrics["malicious_coverage_core"] == 1
    assert metrics["malicious_coverage_with_context"] == 1
    assert metrics["verified_benign_fraction_core"] == 0
    assert metrics["mixed_attack_incidents"] == 0
    assert metrics["attack_fragmentation"] == {"attack_0": 1, "attack_1": 1}
    assert len(report["runtime_seconds"]["runs"]) == 2
    assert report["runtime_seconds"]["median"] > 0
    assert report["events_per_second"] > 0


def test_context_reports_extra_benign_nodes_separately():
    graph, scores, _, attacks, labels = synthetic_region(background_chains=0)
    config = IncidentBuilderConfig(context_hops=1)
    metrics = benchmark_configs(
        graph, scores, 5, {"context": config}, repeats=1,
        attack_labels=attacks, complete_labels=labels,
    )["context"]["metrics"]
    assert metrics["context_count"] == 2
    assert metrics["malicious_coverage_core"] == 1
    assert metrics["verified_benign_fraction_core"] == 0
    assert metrics["verified_benign_fraction_with_context"] == pytest.approx(2 / 14)


def test_tight_hop_limit_reduces_coverage_and_splits_attacks():
    graph, scores, _, attacks, labels = synthetic_region(background_chains=0)
    config = IncidentBuilderConfig(max_connector_hops=2)
    metrics = benchmark_configs(
        graph, scores, 5, {"tight": config}, repeats=1,
        attack_labels=attacks, complete_labels=labels,
    )["tight"]["metrics"]
    assert metrics["incident_count"] == 4
    assert metrics["malicious_coverage_core"] == pytest.approx(4 / 6)
    assert metrics["attack_fragmentation"] == {"attack_0": 2, "attack_1": 2}


def test_unlabeled_nodes_are_not_called_verified_benign():
    graph, scores, _, attacks, _ = synthetic_region(background_chains=0)
    metrics = benchmark_configs(
        graph, scores, 5, {"baseline": IncidentBuilderConfig()},
        repeats=1, attack_labels=attacks,
    )["baseline"]["metrics"]
    assert metrics["malicious_coverage_core"] == 1
    assert metrics["verified_benign_fraction_core"] is None
    assert metrics["verified_benign_fraction_with_context"] is None
    unlabeled = benchmark_configs(
        graph, scores, 5, {"baseline": IncidentBuilderConfig()}, repeats=1,
    )["baseline"]["metrics"]
    assert unlabeled["malicious_coverage_core"] is None
    assert unlabeled["verified_benign_fraction_core"] is None


def test_quality_metrics_use_only_mapped_ground_truth_nodes():
    graph, scores, _, attacks, labels = synthetic_region(background_chains=0)
    attacks["attack_0"].add(999_999)
    metrics = benchmark_configs(
        graph, scores, 5, {"baseline": IncidentBuilderConfig()},
        repeats=1, attack_labels=attacks, complete_labels=labels,
    )["baseline"]["metrics"]
    assert metrics["attack_labels_unmatched_nodes"] == 1
    assert metrics["malicious_nodes_in_region"] == 12
    assert metrics["malicious_coverage_core"] == 1


def test_complete_labels_must_cover_the_whole_region():
    graph, _, _, attacks, labels = synthetic_region(background_chains=0)
    labels.pop(next(iter(labels)))
    with pytest.raises(ValueError, match="miss 1 provenance nodes"):
        quality_metrics((), graph.nodes, attacks, labels)


def test_attack_and_complete_labels_must_agree():
    graph, _, _, attacks, labels = synthetic_region(background_chains=0)
    labels[1] = False
    with pytest.raises(ValueError, match="conflict"):
        quality_metrics((), graph.nodes, attacks, labels)


def test_zero_malicious_nodes_give_undefined_coverage():
    graph, _, _, _, labels = synthetic_region(background_chains=0)
    labels = {node: False for node in graph}
    metrics = quality_metrics((), graph.nodes, complete_labels=labels)
    assert metrics["malicious_nodes_in_region"] == 0
    assert metrics["malicious_coverage_core"] is None


def test_attack_label_loader_reads_headerless_per_attack_csv(tmp_path):
    first = tmp_path / "attack_a.csv"
    second = tmp_path / "attack_b.csv"
    first.write_text('uuid-1,"file, with comma",1\nuuid-2,process,2\n')
    second.write_text('uuid-3,process,10\n')
    assert load_attack_labels([first, second]) == {"attack_a": {1, 2}, "attack_b": {10}}


def test_attack_label_loader_retains_overlap_for_explicit_accounting(tmp_path):
    first = tmp_path / "attack_a.csv"
    second = tmp_path / "attack_b.csv"
    first.write_text('uuid-1,process,1\n')
    second.write_text('uuid-2,process,1\n')
    assert load_attack_labels([first, second]) == {"attack_a": {1}, "attack_b": {1}}


def test_overlapping_attack_nodes_do_not_create_false_mixing():
    graph, scores, _, attacks, labels = synthetic_region(background_chains=0)
    attacks["attack_0"].add(11)
    metrics = benchmark_configs(
        graph, scores, 5, {"baseline": IncidentBuilderConfig()},
        repeats=1, attack_labels=attacks, complete_labels=labels,
    )["baseline"]["metrics"]
    assert metrics["ambiguous_attack_nodes"] == 1
    assert metrics["malicious_coverage_core"] == 1
    assert metrics["attack_fragmentation"] == {"attack_0": 1, "attack_1": 1}
    assert metrics["mixed_attack_incidents"] == 0


def test_complete_node_label_loader_requires_explicit_binary_labels(tmp_path):
    labels = tmp_path / "labels.csv"
    pd.DataFrame({"node_id": [1, 2], "is_malicious": [1, 0]}).to_csv(labels, index=False)
    assert load_complete_node_labels(labels) == {1: True, 2: False}
    labels.write_text("node_id,is_malicious\n1,unknown\n")
    with pytest.raises(ValueError, match="0 or 1"):
        load_complete_node_labels(labels)


def test_synthetic_cli_writes_reproducible_metrics_without_overwriting(tmp_path):
    output = tmp_path / "benchmark.json"
    argv = ["--synthetic", "--repeats", "1", "--synthetic-background-chains", "1",
            "--output", str(output)]
    assert main(argv) == 0
    payload = json.loads(output.read_text())
    assert payload["benchmark_type"] == "synthetic_engineering"
    assert payload["results"]["baseline"]["metrics"]["malicious_coverage_core"] == 1
    original = output.read_text()
    with pytest.raises(SystemExit) as exc:
        main(argv)
    assert exc.value.code == 2
    assert output.read_text() == original


def test_validation_strategy_comparison_and_test_freeze(tmp_path):
    import torch

    graph, scores, validation, attacks, labels = synthetic_region(background_chains=0)
    graph_file, scores_file = tmp_path / "graph.pt", tmp_path / "scores.csv"
    validation_file = tmp_path / "validation.csv"
    config_file = tmp_path / "configs.json"
    output = tmp_path / "result.json"
    torch.save(graph, graph_file)
    scores.to_csv(scores_file, index=False)
    validation.to_csv(validation_file, index=False)
    config_file.write_text(json.dumps({"baseline": {}, "tight": {"max_connector_hops": 2}}))
    common = ["--graph", str(graph_file), "--scores", str(scores_file),
              "--validation-scores", str(validation_file), "--configs", str(config_file),
              "--repeats", "1", "--output", str(output)]
    assert main(common + ["--split", "validation"]) == 0
    payload = json.loads(output.read_text())
    assert payload["results"]["baseline"]["metrics"]["malicious_coverage_core"] is None
    assert set(payload["results"]) == {"baseline", "tight"}
    output.unlink()
    with pytest.raises(SystemExit) as exc:
        main(common + ["--split", "test"])
    assert exc.value.code == 2
    assert not output.exists()


def test_real_data_cli_uses_mapped_attack_and_complete_node_labels(tmp_path):
    import torch

    graph, scores, _, attacks, labels = synthetic_region(background_chains=0)
    graph_file = tmp_path / "graph.pt"
    scores_file = tmp_path / "scores.csv"
    labels_file = tmp_path / "node_labels.csv"
    output = tmp_path / "benchmark.json"
    torch.save(graph, graph_file)
    scores.to_csv(scores_file, index=False)
    pd.DataFrame(
        {"node_id": list(labels), "is_malicious": [int(value) for value in labels.values()]}
    ).to_csv(labels_file, index=False)
    argv = [
        "--graph", str(graph_file), "--scores", str(scores_file),
        "--threshold", "5", "--split", "test", "--repeats", "1",
        "--complete-node-labels", str(labels_file), "--output", str(output),
    ]
    for name, nodes in attacks.items():
        path = tmp_path / f"{name}.csv"
        path.write_text("".join(f"uuid-{node},description,{node}\n" for node in sorted(nodes)))
        argv.extend(["--attack-label", str(path)])

    assert main(argv) == 0
    payload = json.loads(output.read_text())
    metrics = payload["results"]["baseline"]["metrics"]
    assert payload["benchmark_type"] == "real_data"
    assert payload["split"] == "test"
    assert metrics["malicious_coverage_core"] == 1
    assert metrics["verified_benign_fraction_core"] == 0
    assert metrics["attack_fragmentation"] == {"attack_0": 1, "attack_1": 1}
