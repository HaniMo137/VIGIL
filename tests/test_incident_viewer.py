import json

import pandas as pd
import pytest

from pidsmaker.incidents.demo_data import create_demo, synthetic_viewer_region
from pidsmaker.incidents.viewer import create_app
from pidsmaker.incidents import viewer


@pytest.fixture
def demo(tmp_path):
    return create_demo(tmp_path / "viewer_demo", background_chains=3)


@pytest.fixture
def client(demo):
    app = create_app(
        demo / "incidents.json", demo / "original_graph.pt",
        demo / "event_scores.csv",
    )
    app.testing = True
    return app.test_client()


def test_demo_creates_real_csv_graph_and_incident_files(demo):
    assert (demo / "original_graph.pt").is_file()
    assert (demo / "event_scores.csv").is_file()
    assert (demo / "validation_scores.csv").is_file()
    assert (demo / "synthetic_node_labels.csv").is_file()
    scores = pd.read_csv(demo / "event_scores.csv")
    assert set(scores.columns) == {"srcnode", "dstnode", "time", "edge_type", "loss"}
    assert scores["time"].dtype.kind in "iu"
    payload = json.loads((demo / "incidents.json").read_text())
    assert payload["status"] == "unverified_candidates"
    assert len(payload["incidents"]) == 3
    assert all(len(case["seed_edges"]) == 6 for case in payload["incidents"])
    assert all(len(case["connector_edges"]) == 14 for case in payload["incidents"])


def test_larger_demo_is_scored_and_can_be_resized(tmp_path):
    small = create_demo(tmp_path / "small", background_chains=1, case_count=1, chain_edges=8)
    large = create_demo(tmp_path / "large", background_chains=5, case_count=2, chain_edges=12)
    small_scores = pd.read_csv(small / "event_scores.csv")
    large_scores = pd.read_csv(large / "event_scores.csv")
    assert len(large_scores) > len(small_scores)
    assert len(json.loads((small / "incidents.json").read_text())["incidents"]) == 1
    assert len(json.loads((large / "incidents.json").read_text())["incidents"]) == 2
    assert all(score < 5 or score >= 10 for score in large_scores["loss"])


@pytest.mark.parametrize("kwargs", [
    {"case_count": 0}, {"chain_edges": 4}, {"background_chains": -1},
])
def test_demo_rejects_invalid_sizes(kwargs):
    with pytest.raises(ValueError):
        synthetic_viewer_region(**kwargs)


def test_demo_never_overwrites_existing_directory(demo):
    original = (demo / "event_scores.csv").read_text()
    with pytest.raises(FileExistsError):
        create_demo(demo)
    assert (demo / "event_scores.csv").read_text() == original


def test_viewer_serves_local_assets_and_no_cache(client):
    page = client.get("/")
    assert page.status_code == 200
    assert b"Incident Explorer" in page.data
    assert b"/static/app.js" in page.data
    assert page.headers["Cache-Control"] == "no-store"
    assert "default-src 'self'" in page.headers["Content-Security-Policy"]
    assert client.get("/static/app.css").status_code == 200
    assert client.get("/static/app.js").status_code == 200
    assert client.get("/vendor/d3.v4.min.js").status_code == 200
    assert client.get("/static/../../README.md").status_code != 200


def test_summary_reports_only_incident_metadata(client):
    summary = client.get("/api/summary").get_json()
    assert summary["threshold"] == 5
    assert len(summary["incidents"]) == 3
    assert summary["region_events"] == 147
    assert summary["incidents"][0]["seeds"] == 6
    assert summary["incidents"][0]["connectors"] == 14
    assert isinstance(summary["incidents"][0]["start_time"], str)
    assert "nodes" not in summary


def test_incident_and_original_views_share_exact_event_ids(client):
    incident = client.get("/api/incident/0").get_json()
    original = client.get("/api/original/0?hops=1").get_json()
    assert len(incident["edges"]) == 20
    assert len(original["edges"]) == 40
    assert not incident["truncated"]
    ids = {event["id"] for event in original["edges"]}
    assert all(event["id"] in ids for event in incident["edges"])
    assert {event["role"] for event in incident["edges"]} == {"seed", "connector"}
    distractors = [edge for edge in original["edges"] if edge["role"] == "other"]
    assert len(distractors) == 20
    assert all(edge["score"] == 0.1 for edge in distractors)
    assert all(isinstance(edge["time"], str) for edge in original["edges"])
    assert all("event_uuid" in edge and "operation" in edge for edge in original["edges"])


def test_two_hops_shows_more_original_context_without_growing_incident(client):
    one = client.get("/api/original/0?hops=1").get_json()
    two = client.get("/api/original/0?hops=2").get_json()
    incident = client.get("/api/incident/0").get_json()
    assert len(one["edges"]) == 40
    assert len(two["edges"]) == 47
    assert len(incident["edges"]) == 20


def test_original_view_can_focus_and_filter_without_losing_integer_nanoseconds(client):
    summary = client.get("/api/summary").get_json()["incidents"][0]
    first = int(summary["start_time"])
    later = first + 1_000_000_000
    response = client.get(
        "/api/original/0",
        query_string={"focus": "2", "hops": "1", "start": str(first), "end": str(later)},
    )
    assert response.status_code == 200
    graph = response.get_json()
    assert graph["edges"]
    assert all(first <= int(edge["time"]) <= later for edge in graph["edges"])
    assert "2" in {node["id"] for node in graph["nodes"]}


def test_original_view_keeps_focused_node_when_no_events_match(client):
    summary = client.get("/api/summary").get_json()["incidents"][0]
    end = int(summary["view_end_time"])
    result = client.get(
        "/api/original/0", query_string={"focus": "2", "start": str(end), "end": str(end)}
    ).get_json()
    assert result["edges"] == []
    assert {node["id"] for node in result["nodes"]} == {"2"}


def test_view_limits_are_explicit_and_keep_seeds_first(demo):
    app = create_app(
        demo / "incidents.json", demo / "original_graph.pt",
        demo / "event_scores.csv", max_edges=2,
    )
    app.testing = True
    client = app.test_client()
    incident = client.get("/api/incident/0").get_json()
    original = client.get("/api/original/0").get_json()
    assert incident["truncated"] and original["truncated"]
    assert len(incident["edges"]) == len(original["edges"]) == 2
    assert all(edge["role"] == "seed" for edge in incident["edges"])


@pytest.mark.parametrize("query", [
    {"hops": "0"}, {"hops": "3"}, {"hops": "invalid"},
    {"focus": "999999"}, {"focus": "not-an-integer"},
    {"start": "0"}, {"end": "9999999999999999999"},
])
def test_viewer_rejects_invalid_or_unbounded_original_queries(client, query):
    assert client.get("/api/original/0", query_string=query).status_code == 400


def test_viewer_returns_404_for_unknown_incidents(client):
    assert client.get("/api/incident/99").status_code == 404
    assert client.get("/api/original/99").status_code == 404


def test_viewer_rejects_mismatched_scores_instead_of_showing_false_evidence(demo):
    path = demo / "incidents.json"
    payload = json.loads(path.read_text())
    payload["incidents"][0]["seed_edges"][0]["score"] += 1
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="anomaly score"):
        create_app(path, demo / "original_graph.pt", demo / "event_scores.csv")


@pytest.mark.parametrize("compose_host,expected", [
    (None, "127.0.0.1"),
    ("0.0.0.0", "0.0.0.0"),
])
def test_viewer_binds_locally_unless_compose_sets_container_host(
    monkeypatch, compose_host, expected,
):
    calls = {}

    class FakeApp:
        def run(self, **kwargs):
            calls.update(kwargs)

    monkeypatch.setattr(viewer, "create_app", lambda *args, **kwargs: FakeApp())
    if compose_host is None:
        monkeypatch.delenv("VIGIL_VIEWER_HOST", raising=False)
    else:
        monkeypatch.setenv("VIGIL_VIEWER_HOST", compose_host)
    viewer.main([
        "--incidents", "incidents.json", "--graph", "graph.pt",
        "--scores", "scores.csv",
    ])
    assert calls["host"] == expected
    assert calls["port"] == 8765
    assert calls["debug"] is False
