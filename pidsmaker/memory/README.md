# Trusted incident memory: intake contract

`MemoryEntry` is a proposed reference, not proof that its label is correct.
Before admission, `import_verified_entry` compares it with a separately
prepared JSON evidence manifest. The manifest must come from independently
checked dataset ground truth, a controlled simulation record, or analyst
review—not from the detector's score or an unverified builder output.

The version-1 manifest records `entry_id`, `label`, `source_id`, `host_id`,
`attack_instance_id` (or `null`), sorted `techniques`, the matching
`verification` object, `evidence_fingerprint`, and `confirmed_attack_nodes`.
Malicious entries need a named attack instance and at least one confirmed
attack node present in the graph. Benign entries require an explicitly
verified benign scope and have no attack nodes or attack instance. The
fingerprint binds the manifest to provenance node IDs and event identities,
including exact timestamps and relations, but excludes detector scores.

The importer checks consistency only. It cannot prove that a manifest was
independently curated; that provenance must be governed by the evaluation
workflow. Synthetic manifests in tests exercise the code but are not
scientific evidence.

`IncidentMemoryStore` provides a small, insert-only SQLite store. `add(entry,
manifest_path)` reruns intake validation, archives the exact manifest and a
JSON representation of the graph, and refuses duplicate IDs. `get(entry_id)`
checks both archives before returning an entry; `list_entries()` returns
metadata only. For example, construct the store with
`IncidentMemoryStore("artifacts/incident_memory.sqlite3")` so local contents
remain Git-ignored. No memory database is populated by this repository.
The current JSON codec accepts ordinary scalar, list, tuple, and dictionary
graph attributes; unsupported types fail explicitly. This API is not a
tamper-proof database or retrieval index.

## Fixed incident signature

**Encoder migration (3 October 2026):** the detector's `vigil` profile now uses
`VigilTGNEncoder`. This feature-only signature API still supports the archived
semantic `VigilEncoder` (`vigil_mlp` profile). TGN embeddings require a temporal
neighborhood, not just a matrix of node features; calling `encode_nodes(features)`
on the TGN fails explicitly. `encode_temporal_nodes(batch)` can export original
node IDs and context-dependent embeddings in eval mode with recurrent memory
disabled, but it is not yet wired into verified-memory intake/retrieval. A shared
checkpoint plus a documented history/replay policy is required before comparing
TGN references and queries. Do not label the existing synthetic memory results
as TGN experiments. Old databases and checkpoints are left unchanged.

`build_incident_signature(graph, node_features, encoder, schema)` uses the
frozen VIGIL encoder's shared node representation. It averages those vectors
over the incident's nodes; it does not average the separate source and
destination role heads. The other blocks count node types, edge relations,
and directed, time-respecting length-2 relation paths. Each block is
L2-normalized and weighted by `0.5`, as in the research design. Unknown
types have stable fallback bins, and a schema ID records the fixed vocabulary
and encoder state fingerprint. Use `encoder_state_fingerprint(encoder)` as the
schema's `encoder_checkpoint_id`; signature construction rejects changed
weights, rather than trusting a filename or free-form label.

The caller must provide Word2Vec features for every incident node, put the
encoder in evaluation mode, and use the same trained checkpoint and schema
for memory entries and queries. The code does not train or load a checkpoint
or create a retrieval index; synthetic features only test the mathematics.

## Read-only retrieval

`IncidentRetriever` computes reference signatures from entries already admitted
to `IncidentMemoryStore`. Pass a node-feature map for each entry, the same
frozen encoder and `SignatureSchema`, and optionally a cosine-similarity
threshold and per-label result cap. `search(candidate_graph, candidate_features)`
returns ranked verified malicious **and** benign examples. Labels belong only
to references; the candidate is never relabeled or admitted. If a threshold
is set and nothing passes, the closest graph is returned once with
`above_threshold=False` and `no_strong_match=True`. With no threshold, matches
are exploratory and `above_threshold=None`.

The incident viewer accepts optional `memory_retriever` and
`incident_node_features` arguments to `create_app`. The latter maps each
incident index to its node-feature map. When supplied together, the viewer
shows the matches and a bounded read-only reference graph, including an
explicit below-threshold warning. This is currently a Python API, not a
command-line option: loading a real trained checkpoint and matching feature
artifacts is still a separate lab integration step. The reference features
are caller-supplied and not yet archived or cryptographically bound to the
stored graph. Use only a consistent, trusted feature pipeline; similarity
thresholds need validation before operational use.

For a Python integration after loading the trusted encoder and features:

```python
retriever = IncidentRetriever(
    store, reference_node_features, encoder.eval(), schema,
    threshold=validated_similarity_threshold,
)
app = create_app(
    incident_path, graph_path, scores_path,
    memory_retriever=retriever,
    incident_node_features=query_node_features_by_incident_index,
)
```

Import `IncidentRetriever` from `pidsmaker.memory` and `create_app` from
`pidsmaker.incidents.viewer`. This call reads stored references but does not
write to memory or approve a candidate.
