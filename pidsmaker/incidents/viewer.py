"""Read-only local viewer for candidate incidents and original provenance.

Run ``python -m pidsmaker.incidents.viewer --help``. Graph files must be
trusted: loading torch-saved NetworkX graphs uses pickle.
"""

import argparse
import json
import math
import os
from itertools import chain
from pathlib import Path

import pandas as pd
from flask import Flask, abort, jsonify, request, send_from_directory

from pidsmaker.incidents.__main__ import _load_graph
from pidsmaker.incidents.builder import prepare_provenance_graph

STATIC_DIR = Path(__file__).parent / "viewer_static"
ROLES = (("seed_edges", "seed"), ("connector_edges", "connector"),
         ("context_edges", "context"))


def _key(value):
    return tuple(_key(item) for item in value) if isinstance(value, list) else value


def _event_id(src, dst, key):
    return json.dumps([src, dst, key], separators=(",", ":"), default=str)


def _safe_text(value, limit=300):
    text = str(value)
    return text[:limit] + "…" if len(text) > limit else text


class IncidentViewerData:
    """Keep one provenance region on the server; send only bounded graph slices."""

    def __init__(self, incident_path, graph_path, scores_path,
                 relation_to_id=None, max_edges=300, max_nodes=400,
                 time_padding_ns=5_000_000_000):
        if max_edges < 1 or max_nodes < 2 or time_padding_ns < 0:
            raise ValueError("Viewer limits must be positive")
        self.max_edges = max_edges
        self.max_nodes = max_nodes
        self.time_padding_ns = time_padding_ns
        payload = json.loads(Path(incident_path).read_text())
        if payload.get("status") != "unverified_candidates":
            raise ValueError("Expected unverified candidate incident JSON")
        self.incidents = payload["incidents"]
        self.threshold = float(payload["threshold"])
        if not math.isfinite(self.threshold):
            raise ValueError("Incident threshold must be finite")
        graph = _load_graph(graph_path)
        scores = pd.read_csv(scores_path)
        self.graph = prepare_provenance_graph(graph, scores, relation_to_id)
        self.roles = []
        for incident in self.incidents:
            role_map = {}
            for field, role in ROLES:
                for ref in incident[field]:
                    src, dst, key = int(ref["srcnode"]), int(ref["dstnode"]), _key(ref["key"])
                    if not self.graph.has_edge(src, dst, key):
                        raise ValueError("Incident and original graph do not match: missing event")
                    original = self.graph[src][dst][key]
                    if (original["time"] != int(ref["time"])
                            or original["edge_type"] != int(ref["edge_type"])):
                        raise ValueError("Incident and original graph do not match: event attributes")
                    if ref.get("score") is not None and not math.isclose(
                        float(original.get("score", math.nan)), float(ref["score"]),
                        rel_tol=1e-9, abs_tol=1e-12,
                    ):
                        raise ValueError("Incident and original graph do not match: anomaly score")
                    identity = (src, dst, key)
                    if identity in role_map:
                        raise ValueError("Incident event has more than one role")
                    role_map[identity] = role
            self.roles.append(role_map)

    def _incident(self, index):
        if index < 0 or index >= len(self.incidents):
            raise IndexError("Unknown incident")
        return self.incidents[index]

    def _bounds(self, index):
        incident = self._incident(index)
        return (int(incident["start_time"]) - self.time_padding_ns,
                int(incident["end_time"]) + self.time_padding_ns)

    def summary(self):
        return {
            "threshold": self.threshold,
            "region_nodes": self.graph.number_of_nodes(),
            "region_events": self.graph.number_of_edges(),
            "max_view_edges": self.max_edges,
            "incidents": [
                {
                    "id": index,
                    "start_time": str(incident["start_time"]),
                    "end_time": str(incident["end_time"]),
                    "view_start_time": str(self._bounds(index)[0]),
                    "view_end_time": str(self._bounds(index)[1]),
                    "seeds": len(incident["seed_edges"]),
                    "connectors": len(incident["connector_edges"]),
                    "context": len(incident["context_edges"]),
                }
                for index, incident in enumerate(self.incidents)
            ],
        }

    def _node(self, node):
        data = self.graph.nodes[node]
        return {
            "id": str(node),
            "type": _safe_text(data.get("node_type", "unknown"), 80),
            "label": _safe_text(data.get("label", node)),
        }

    def _edge(self, event, role):
        src, dst, key = event
        data = self.graph[src][dst][key]
        score = data.get("score")
        score = float(score) if score is not None else None
        if role == "other" and score is not None and score >= self.threshold:
            role = "other_seed"
        return {
            "id": _event_id(src, dst, key),
            "source": str(src), "target": str(dst),
            "key": _safe_text(key, 120),
            "time": str(data["time"]),
            "edge_type": data["edge_type"],
            "operation": _safe_text(data.get("label", data["edge_type"]), 120),
            "event_uuid": _safe_text(data.get("event_uuid", "—"), 160),
            "score": score,
            "role": role,
        }

    def _pack(self, events, role_map, truncated=False, extra_nodes=()):
        edge_data = [self._edge(event, role_map.get(event, "other")) for event in events]
        nodes = {node for src, dst, _ in events for node in (src, dst)} | set(extra_nodes)
        return {
            "nodes": [self._node(node) for node in sorted(nodes)],
            "edges": edge_data,
            "truncated": truncated,
        }

    def incident_graph(self, index):
        self._incident(index)
        events = []
        nodes = set()
        total = 0
        role_map = self.roles[index]
        for role in ("seed", "connector", "context"):
            for event, event_role in role_map.items():
                if event_role != role:
                    continue
                total += 1
                new_nodes = set(event[:2]) - nodes
                if len(events) >= self.max_edges or len(nodes) + len(new_nodes) > self.max_nodes:
                    continue
                events.append(event)
                nodes.update(new_nodes)
        return self._pack(events, role_map, truncated=total > len(events))

    def original_slice(self, index, *, hops=1, start=None, end=None, focus=None):
        incident = self._incident(index)
        if hops not in (1, 2):
            raise ValueError("hops must be 1 or 2")
        lower, upper = self._bounds(index)
        start = lower if start is None else int(start)
        end = upper if end is None else int(end)
        if start < lower or end > upper or start > end:
            raise ValueError("Time filter is outside the incident view window")
        if focus is not None:
            if focus not in self.graph:
                raise ValueError("Focus node is not in the provenance graph")
            frontier = {focus}
        else:
            frontier = {
                node for field in ("seed_edges", "connector_edges")
                for ref in incident[field]
                for node in (int(ref["srcnode"]), int(ref["dstnode"]))
            }
        if len(frontier) > self.max_nodes:
            frontier = set(sorted(frontier)[:self.max_nodes])
            truncated = True
        else:
            truncated = False
        seen_nodes = set(frontier)
        roots = set(frontier)
        selected = []
        seen_events = set()
        for _ in range(hops):
            next_frontier = set()
            for node in sorted(frontier):
                adjacent = self.graph.in_edges(node, keys=True, data=True)
                outgoing = self.graph.out_edges(node, keys=True, data=True)
                for src, dst, key, data in chain(adjacent, outgoing):
                    event = (src, dst, key)
                    if event in seen_events or not start <= data["time"] <= end:
                        continue
                    seen_events.add(event)
                    new_nodes = {src, dst} - seen_nodes
                    if len(selected) >= self.max_edges or len(seen_nodes) + len(new_nodes) > self.max_nodes:
                        truncated = True
                        continue
                    selected.append(event)
                    seen_nodes.update(new_nodes)
                    next_frontier.update(new_nodes)
            frontier = next_frontier
            if not frontier:
                break
        return self._pack(selected, self.roles[index], truncated=truncated, extra_nodes=roots)


def create_app(incident_path, graph_path, scores_path, *, relation_to_id=None,
               max_edges=300, max_nodes=400, time_padding_ns=5_000_000_000):
    data = IncidentViewerData(
        incident_path, graph_path, scores_path, relation_to_id,
        max_edges, max_nodes, time_padding_ns,
    )
    app = Flask(__name__, static_folder=None)
    app.config["VIEWER_DATA"] = data

    @app.after_request
    def no_cache(response):
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self'; style-src 'self'; "
            "img-src 'self' data:; connect-src 'self'"
        )
        return response

    @app.route("/")
    def home():
        return send_from_directory(STATIC_DIR, "index.html")

    @app.route("/static/<path:name>")
    def static_file(name):
        return send_from_directory(STATIC_DIR, name)

    @app.route("/vendor/d3.v4.min.js")
    def d3_library():
        root = Path(__file__).parents[1] / "vizgen" / "web" / "static" / "vendor"
        return send_from_directory(root, "d3.v4.min.js")

    @app.route("/api/summary")
    def summary():
        return jsonify(data.summary())

    @app.route("/api/incident/<int:index>")
    def incident(index):
        try:
            return jsonify(data.incident_graph(index))
        except IndexError:
            abort(404)

    @app.route("/api/original/<int:index>")
    def original(index):
        try:
            hops = int(request.args.get("hops", "1"))
            focus = request.args.get("focus")
            focus = int(focus) if focus else None
            result = data.original_slice(
                index, hops=hops, start=request.args.get("start"),
                end=request.args.get("end"), focus=focus,
            )
            return jsonify(result)
        except IndexError:
            abort(404)
        except (TypeError, ValueError) as exc:
            abort(400, str(exc))

    return app


def main(argv=None):
    parser = argparse.ArgumentParser(description="View candidate incidents alongside provenance")
    parser.add_argument("--incidents", required=True, help="Incident builder JSON output")
    parser.add_argument("--graph", required=True, help="Trusted original provenance graph")
    parser.add_argument("--scores", required=True, help="Matching score CSV")
    parser.add_argument("--relation-map", help="JSON label-to-score-ID mapping, if needed")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--max-view-edges", type=int, default=300)
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535 or args.max_view_edges < 1:
        parser.error("Port and max-view-edges must be positive")
    relation_map = json.loads(Path(args.relation_map).read_text()) if args.relation_map else None
    app = create_app(
        args.incidents, args.graph, args.scores,
        relation_to_id=relation_map, max_edges=args.max_view_edges,
    )
    print(f"Open http://127.0.0.1:{args.port} (read-only local viewer)")
    app.run(
        host=os.environ.get("VIGIL_VIEWER_HOST", "127.0.0.1"),
        port=args.port, debug=False, use_reloader=False,
    )


if __name__ == "__main__":
    main()
