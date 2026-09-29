# Incident builder

This standalone, post-detection component turns scored provenance events into
**unverified candidate incidents**. It does not change the detector or label an
incident as malicious.

## How it works

1. Select high-scoring events as **seeds**, using a threshold chosen from
   validation scores.
2. Group seeds linked by shared entities or a bounded, time-respecting path.
   Seeds in unrelated activity remain in separate incidents.
3. Keep the lower-scoring events needed to connect each group. These are
   **connectors**, not additional seeds.
4. Optionally include a small amount of adjacent **context**; prune unsupported
   branches. Isolated seeds still produce incidents.

Each output retains the selected events' original timestamps, identities, and
graph attributes. A retained event is evidence, not proof of causality or harm.

## Run independently

With the project dependencies installed, provide a saved provenance graph, its
matching event-score CSV, and a separate validation-score CSV. The graph and
event scores must use the **same node IDs and provenance region**. Validation
scores are used only to choose the threshold.

```bash
python -m pidsmaker.incidents \
  --graph path/to/graph.pt \
  --scores path/to/event_scores.csv \
  --validation-scores path/to/validation_scores.csv \
  --relation-map path/to/relation_ids.json \
  --output path/to/incidents.json
```

The graph must be a NetworkX `MultiDiGraph` with nanosecond event timestamps.
The event-score CSV needs `srcnode`, `dstnode`, `time`, `edge_type`, and `loss`.
If the graph uses operation names (`label`) instead of numeric `edge_type`, the
relation-map JSON must map those names to the **same IDs used in the score CSV**.
Otherwise, omit `--relation-map`. Optional score columns `key` or `event_uuid`
identify parallel events that would otherwise be ambiguous.

Use `--threshold NUMBER` instead of `--validation-scores FILE` if the threshold
was already fixed using validation data. `--config FILE` accepts JSON settings
for time limits, connector hops, optional context, and other bounds. Run
`python -m pidsmaker.incidents --help` to see all command-line options.

The output is JSON with separate seed, connector, and context lists and their
provenance graph. The command never overwrites an existing output file. It
rejects missing or ambiguous scored events instead of guessing their identity.
Only load graph files you trust: PyTorch-saved graphs use pickle.

The standalone run verifies the construction workflow; incident quality and
parameter choices still need evaluation on real data.

## Explore incidents locally

Generate a synthetic provenance region, score CSVs, and candidate incidents
for the read-only viewer. By default it contains three connected cases with
20 retained events each, plus nearby and unrelated background activity:

```bash
python -m pidsmaker.incidents.demo_data
python -m pidsmaker.incidents.viewer \
  --incidents artifacts/incident_demo_large/incidents.json \
  --graph artifacts/incident_demo_large/original_graph.pt \
  --scores artifacts/incident_demo_large/event_scores.csv
```

With the project Compose container, open `http://127.0.0.1:8765` directly in
your browser; no VS Code port forwarding is needed. The left graph shows the
selected incident; the right
shows a bounded neighborhood in the original provenance graph. Select an event
to compare its evidence role, score, and original identity. Time and hop
controls narrow the view. The local files are under the Git-ignored
`artifacts/` directory and are not real evaluation data. The generator refuses
to overwrite an existing demo directory; use `--output-dir` for another run.
Adjust `--cases`, `--chain-edges`, or `--background-chains` to change demo size.
The older `artifacts/incident_demo/` sample is left untouched.

To inspect your own run, replace the three viewer paths with your incident
JSON, original graph, and matching score CSV. Pass `--relation-map` if the graph
needs event-label translation. The viewer loads the whole trusted graph on the
server but sends only limited slices to the browser. Compose publishes the
viewer only on the host's localhost; outside Compose, the viewer binds to
localhost by default. It does not modify the inputs. Avoid exposing the
development server publicly.

## Test and benchmark

Run the unit tests:

```bash
python -m pytest -q tests
```

Run a synthetic engineering check without external files:

```bash
python -m pidsmaker.incidents.benchmark \
  --synthetic \
  --configs config/incident_benchmark_strategies.json \
  --output path/to/synthetic_benchmark.json
```

The synthetic result checks behavior and measures builder runtime; it is **not
an attack-detection result**. Median runtime excludes graph loading. The three
example strategies compare the default builder, a shorter connector limit,
and optional context.

For a real-data comparison, use a graph and event-score CSV from the same
region. Freeze the threshold from a separate validation-score CSV. Compare
settings on validation only:

```bash
python -m pidsmaker.incidents.benchmark \
  --graph path/to/graph.pt \
  --scores path/to/event_scores.csv \
  --validation-scores path/to/validation_scores.csv \
  --relation-map path/to/relation_ids.json \
  --split validation \
  --configs config/incident_benchmark_strategies.json \
  --attack-label path/to/one_attack.csv \
  --output path/to/validation_benchmark.json
```

`--attack-label` can be repeated for each independently labeled attack. The
runner accepts the PIDSMaker `orthrus`/`reapr` format (third CSV column:
integer `node_id`). Those IDs must match the graph. It reports how many label
IDs were absent from the chosen region. UUID-only ground truth needs an
explicit UUID-to-node-ID mapping first.
Nodes labeled in more than one attack remain in coverage calculations, but
are excluded from attack mixing and fragmentation counts and reported as
`ambiguous_attack_nodes`.

The main quality metric is **malicious-node coverage** in seed-only, core
(seeds plus connectors), and context-inclusive graphs. Attack fragmentation
and mixed-attack incidents require per-attack labels. If you additionally have
a **complete, explicitly verified** `node_id,is_malicious` CSV for every graph
node, pass `--complete-node-labels FILE` for benign-node contamination. Without
that file, the benchmark leaves benign contamination undefined; an unlabeled
node is not assumed benign.

After selecting settings on validation, run a single frozen configuration on
`--split test`. The benchmark refuses a multi-strategy test run to prevent
choosing settings using test labels. `--help` lists the full command options.
