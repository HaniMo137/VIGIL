import json

import networkx as nx
import pandas as pd
import pytest

from pidsmaker.incidents.__main__ import main


def inputs(tmp_path):
    import torch

    graph = nx.MultiDiGraph()
    rows = []
    for src, dst, time, label, edge_type, loss in [
        (1, 2, 1, "seed", 1, 10), (2, 3, 2, "bridge", 2, 0.1),
        (3, 4, 3, "bridge", 2, 0.2), (4, 5, 4, "seed", 1, 11),
    ]:
        time += 1_700_000_000_000_000_000
        graph.add_edge(src, dst, time=time, label=label, event_uuid=f"{src}-{dst}")
        rows.append(dict(srcnode=src, dstnode=dst, time=time, edge_type=edge_type, loss=loss))
    graph_path = tmp_path / "graph.pt"
    scores_path = tmp_path / "scores.csv"
    validation_path = tmp_path / "validation.csv"
    relation_path = tmp_path / "relations.json"
    torch.save(graph, graph_path)
    pd.DataFrame(rows).to_csv(scores_path, index=False)
    pd.DataFrame({"loss": [1, 5]}).to_csv(validation_path, index=False)
    relation_path.write_text(json.dumps({"seed": 1, "bridge": 2}))
    output = tmp_path / "output" / "incidents.json"
    argv = ["--graph", str(graph_path), "--scores", str(scores_path),
            "--relation-map", str(relation_path), "--output", str(output)]
    return argv, output, validation_path


def test_cli_builds_and_exports_from_actual_saved_graph(tmp_path, capsys):
    argv, output, validation_path = inputs(tmp_path)
    assert main(argv + ["--validation-scores", str(validation_path)]) == 0
    payload = json.loads(output.read_text())
    assert payload["status"] == "unverified_candidates"
    assert payload["threshold"] == 5
    assert len(payload["incidents"]) == 1
    incident = payload["incidents"][0]
    assert incident["start_time"] == 1_700_000_000_000_000_001
    assert len(incident["seed_edges"]) == 2
    assert len(incident["connector_edges"]) == 2
    restored = nx.node_link_graph(incident["graph"])
    assert restored.is_directed() and restored.is_multigraph()
    assert restored[2][3][0]["event_uuid"] == "2-3"
    assert restored[2][3][0]["time"] == 1_700_000_000_000_000_002
    assert restored[2][3][0]["incident_role"] == "connector"
    assert "Built 1 candidate incidents" in capsys.readouterr().out


def test_cli_accepts_frozen_threshold_and_config(tmp_path):
    argv, output, _ = inputs(tmp_path)
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"max_connector_hops": 1, "relation_penalties": {"2": 0.5}}))
    main(argv + ["--threshold", "5", "--config", str(config)])
    payload = json.loads(output.read_text())
    assert len(payload["incidents"]) == 2
    assert payload["config"]["max_connector_hops"] == 1


def test_cli_refuses_to_overwrite_existing_output(tmp_path):
    argv, output, _ = inputs(tmp_path)
    output.parent.mkdir()
    output.write_text("existing user result")
    with pytest.raises(SystemExit) as exc:
        main(argv + ["--threshold", "5"])
    assert exc.value.code == 2
    assert output.read_text() == "existing user result"


def test_cli_exports_empty_incident_set(tmp_path):
    argv, output, _ = inputs(tmp_path)
    main(argv + ["--threshold", "100"])
    assert json.loads(output.read_text())["incidents"] == []


def test_cli_requires_exactly_one_threshold_source(tmp_path):
    argv, _, validation_path = inputs(tmp_path)
    with pytest.raises(SystemExit) as exc:
        main(argv + ["--threshold", "5", "--validation-scores", str(validation_path)])
    assert exc.value.code == 2
