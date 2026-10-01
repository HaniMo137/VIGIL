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

The viewer also has an optional verified-memory comparison panel through its
Python `create_app(memory_retriever=..., incident_node_features=...)` API. It
shows both verified labels and their reference graphs; a closest example below
threshold is marked as context, not a match. The command-line demo does not
enable this panel because no trained encoder checkpoint or reference feature
artifacts are bundled. See `pidsmaker/memory/README.md` for the intake and
retrieval contract.

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

## Integrated VIGIL evaluation

```bash
python pidsmaker/main.py vigil ATLASV2_EDR \
  --wandb --project VIGIL --exp vigil-atlas-first
```

`config/vigil.yml` inherits VELOX. It uses training-only Word2Vec, Word2Vec-only
node inputs, 1,024-event batches, 128-dimensional VIGIL layers, two residual
blocks, dropout 0.3, learning rate 0.0001, and edge-type prediction. Training
durations remain inherited, untuned starting settings. The detector retains
validation-maximum thresholding with K-means disabled.

The optional `postprocessing.incidents` stage runs after training. It chooses
the latest **numerically ordered, complete scored epoch**, independently of the
detector's best-test-metric summary. One threshold is frozen from **all validation
losses in that epoch**. An incomplete newer epoch is skipped and recorded;
duplicate/ambiguous scored events cause an error, not a guessed association.
The report records the selected zero-based epoch and threshold.

Each transformed test graph is processed once, separately. Incidents **do not
cross window boundaries**. The defaults are a five-second seed gap, five
connector hops, and no extra context. Score matching uses original node IDs,
nanosecond timestamps, and the featurizer's relation encoding, not score CSV
filenames. Every graph event must have exactly one matching score; parallel
events with identical identities need `key` or `event_uuid` in the score CSV.
The current detector exports neither, so genuinely ambiguous data must be
resolved at export before evaluation. Graphs are trusted local pickle files.

Useful overrides (append to the run command):

```bash
--postprocessing.incidents.epoch 9
--postprocessing.incidents.enabled False
--postprocessing.incidents.complete_node_labels path/to/verified_labels.csv
```

Other builder overrides are `max_time_gap_ns`, `max_connector_hops`, and
`context_hops` under the same prefix. Disabling the stage leaves detector
training/evaluation unchanged. This integration expects single-dataset,
edge-level scores and unmodified event identities, as configured for VIGIL.

The final log prints the report location, typically:

```text
artifacts/postprocessing/postprocessing/<configuration-hash>/ATLASV2_EDR/
  report.json
  results-<id>/
    summary.csv
    windows.csv
    attacks.csv
    relation_ids.json
    incidents_00000.json
    aligned/test/window_00000.csv
```

The artifact root follows PIDSMaker's `--artifact_dir`. Each row in
`report.json` → `windows` identifies the original graph and matching incident
JSON/score CSV. Paths for incidents/scores are relative to the report directory;
the original graph path is absolute. Open any window in the local viewer:

```bash
python -m pidsmaker.incidents.viewer \
  --incidents <report-directory>/<window-incidents-path> \
  --graph <window-graph-path> \
  --scores <report-directory>/<window-scores-path> \
  --relation-map <report-directory>/<results-directory>/relation_ids.json
```

Coverage denominators contain labeled malicious nodes **present in processed
graphs**. Dataset coverage uses unique node unions across windows, not mean
window percentages. Absent labeled nodes are reported separately. Per-attack
coverage uses each attack's present nodes. Fragmentation counts distinct
incidents touching an attack across all windows (0 means no recovered incident).
Shared attack nodes are excluded from fragmentation/mixing, but included in
coverage. These are label-file attack groups, not inferred ATT&CK techniques.

Missing label files are recorded and warned about; metrics describe available
labels only. Undefined results are JSON `null` / blank CSV cells, not zero.
Unlabeled nodes are never assumed benign. Benign contamination is available only
with an explicitly verified, complete `node_id,is_malicious` (0/1) file covering
every processed node. This assertion of label completeness is the user's
responsibility; coverage and consistency are checked automatically.

Runtime times exactly one builder call per window (including graph preparation),
excluding loading, metrics and logging. Event counts sum incident-role references;
a connector reused by two incidents contributes twice. These are engineering
measurements, not claims of real-data detection quality.

W&B receives aggregate metric tables and chart-helper panels under `incidents/`,
following the [W&B chart documentation](https://docs.wandb.ai/models/track/log/plots).
Raw provenance, incident graphs, score rows and node identifiers stay local.
Attack names from label filenames are included in the per-attack table.
Cached evaluations reload and log their reports into the current run. Cache
identity includes graph/score file paths, sizes and modification times, label
content hashes, relation encoding and builder settings. Keep source files
immutable during evaluation; use `--force_restart postprocessing` to recompute.
Successful prior report generations are retained if a subsequent attempt fails.

Without `--wandb`, the same local reports and viewer files are still produced.
For CPU-only synthetic acceptance checks (including an offline W&B smoke test):

```bash
python -m pytest -q tests/test_vigil_pipeline.py
```

Memory admission and retrieval remain separate Python components; this stage
neither labels candidates as trusted nor adds them to memory automatically.
