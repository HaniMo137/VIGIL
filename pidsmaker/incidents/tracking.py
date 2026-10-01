"""Metric-only W&B panels. Provenance, event scores and node IDs stay local."""

from .pipeline import REGIONS


def log_report(report, run=None, api=None):
    if api is None:
        import wandb as api
    if run is None:
        run = api.run
    if run is None or getattr(run, "disabled", False):
        return

    def table(rows, columns):
        return api.Table(columns=columns, data=[[row.get(c) for c in columns] for row in rows])

    summary = report["summary"]
    # Deliberate allowlists: never upload local paths, raw graph attributes,
    # scores, signatures, or individual labeled node identifiers.
    window_columns = ["window", "graph_nodes", "graph_events", "incident_count",
                      "seed_count", "connector_count", "context_count", "runtime_seconds",
                      "mixed_attack_incidents", "malicious_nodes_in_region",
                      *[f"malicious_coverage_{r}" for r in REGIONS],
                      "verified_benign_fraction_core", "verified_benign_fraction_with_context"]
    attack_columns = ["attack", "labeled_present", "labeled_absent", "incident_fragments",
                      *[f"coverage_{r}" for r in REGIONS]]
    windows = table(report["windows"], window_columns)
    attacks = table(report["attacks"], attack_columns)
    coverage = api.Table(columns=["region", "coverage"], data=[
        [region, summary[f"malicious_coverage_{region}"]] for region in REGIONS
    ])
    availability = api.Table(columns=["location", "labeled_malicious_nodes"], data=[
        ["present", summary["labeled_malicious_present"]],
        ["absent", summary["labeled_malicious_absent"]],
    ])
    payload = {
        "incidents/tables/summary": table([summary], list(summary)),
        "incidents/tables/windows": windows,
        "incidents/tables/attacks": attacks,
        "incidents/coverage": api.plot.bar(coverage, "region", "coverage", title="Malicious-node coverage: seeds → core → context"),
        "incidents/label_availability": api.plot.bar(availability, "location", "labeled_malicious_nodes", title="Labeled malicious nodes: present vs absent"),
        "incidents/fragmentation": api.plot.bar(attacks, "attack", "incident_fragments", title="Incident fragments per attack (across windows)"),
        "incidents/runtime": api.plot.line(windows, "window", "runtime_seconds", title="Builder runtime by window (seconds)"),
        "incidents/mixed_incidents": api.plot.line(windows, "window", "mixed_attack_incidents", title="Incidents containing multiple attacks"),
    }
    for region in REGIONS:
        payload[f"incidents/coverage_per_attack_{region}"] = api.plot.bar(
            attacks, "attack", f"coverage_{region}", title=f"Attack coverage: {region}"
        )
    for metric in ("incident_count", "seed_count", "connector_count", "context_count"):
        payload[f"incidents/{metric}"] = api.plot.line(windows, "window", metric, title=f"{metric} by window")
    for name, value in summary.items():
        if value is not None:
            run.summary[f"incidents/{name}"] = value
    run.summary["incidents/missing_attack_label_files"] = len(report["missing_attack_label_files"])
    run.log(payload)
