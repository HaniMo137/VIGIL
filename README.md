# VIGIL

VIGIL is an early-stage research project exploring how system-provenance data and historical security context can support intrusion detection and incident investigation.

The project builds on [PIDSMaker](PIDSMaker_README.md), which provides the underlying provenance-processing and experimentation framework. VIGIL focuses on a higher-level research layer for organizing suspicious activity, comparing it with previously observed behavior, and presenting useful evidence to security analysts.

## Project status

VIGIL is under active development. The repository currently contains the PIDSMaker foundation and preliminary VIGIL components. Interfaces, experiments, and implementation details may change as the research progresses.

No production-readiness or benchmark-performance claims are made at this stage.

## High-level scope

- Process and analyze host-level provenance data.
- Build compact representations of suspicious activity.
- Use historical context to assist detection and investigation.
- Produce analyst-oriented, evidence-linked outputs.

Detailed research designs, internal reports, datasets, and unpublished experimental results are intentionally not included in this public repository.

## Getting started

VIGIL currently uses the PIDSMaker environment and workflow. See the preserved [PIDSMaker documentation](PIDSMaker_README.md) for installation, supported datasets, configuration, and basic usage.

For a configured ATLASV2_EDR dataset:

```bash
python pidsmaker/main.py vigil ATLASV2_EDR \
  --wandb --project VIGIL --exp vigil-atlas-first
```

This trains the VIGIL detector and evaluates candidate incidents once after
training. The same W&B run includes `incidents/` coverage, counts, fragmentation,
runtime, and label-availability panels. JSON/CSV reports are also saved locally;
omit `--wandb` for local-only reporting. See the [evaluation guide](pidsmaker/incidents/README.md#integrated-vigil-evaluation)
for output paths and visualization. Memory storage and retrieval remain separate
components; this command does not populate or query memory.

The current `vigil` profile uses ORTHRUS-style temporal-neighbor graph attention.
The earlier semantic MLP is retained as `vigil_mlp`. Existing synthetic memory
benchmarks remain MLP-based; TGN memory retrieval requires explicit temporal
context and must not reuse the old signatures.

## Incident builder

The incident builder turns scored provenance events into smaller **candidate incidents**. It selects high-scoring events as seeds, groups seeds that share entities or have a time-respecting connection, and retains lower-scoring events when they help connect the story. Optional nearby context can be added without treating it as a seed. Unrelated activity stays separate, and every seed is retained, even when isolated.

The result is evidence for investigation, not a confirmed malicious label. The builder can run independently once you have a saved provenance graph, its matching event-score CSV, and a separate validation-score CSV.

```bash
python -m pidsmaker.incidents \
  --graph path/to/graph.pt \
  --scores path/to/event_scores.csv \
  --validation-scores path/to/validation_scores.csv \
  --relation-map path/to/relation_ids.json \
  --output path/to/incidents.json
```

The validation scores set the seed threshold; `--threshold` can supply one already chosen from validation instead. The relation map connects graph event labels to the numeric event types in the score CSV. Omit it if the graph already has numeric `edge_type` attributes. Only load graphs you trust. See the [incident builder guide](pidsmaker/incidents/README.md) for input requirements, optional settings, tests, and benchmarks.

## Acknowledgements

VIGIL is built on the open-source [PIDSMaker](https://github.com/ubc-provenance/PIDSMaker) framework and its provenance-based intrusion-detection ecosystem.

## License

See [LICENSE](LICENSE).
