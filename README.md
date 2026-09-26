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

## Acknowledgements

VIGIL is built on the open-source [PIDSMaker](https://github.com/ubc-provenance/PIDSMaker) framework and its provenance-based intrusion-detection ecosystem.

## License

See [LICENSE](LICENSE).
