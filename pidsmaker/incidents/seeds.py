import math
from typing import Tuple

import pandas as pd

from pidsmaker.incidents.models import EdgeRef


def _validated_losses(scores: pd.DataFrame, description: str) -> pd.Series:
    try:
        losses = pd.to_numeric(scores["loss"], errors="raise")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{description} losses must be numeric") from exc
    if not all(math.isfinite(float(loss)) and float(loss) >= 0 for loss in losses):
        raise ValueError(f"{description} losses must be finite and nonnegative")
    return losses


def _has_identity(value) -> bool:
    return (
        value is not None and value is not pd.NA
        and not (isinstance(value, float) and math.isnan(value))
    )


def calculate_seed_threshold(
    validation_scores: pd.DataFrame,
    method: str = "max",
    percentile: float = 99.0,
) -> float:
    """Calculate a seed threshold from validation losses only."""
    if validation_scores.empty:
        raise ValueError("Validation scores must not be empty")
    if "loss" not in validation_scores.columns:
        raise ValueError("Missing required columns: ['loss']")

    losses = _validated_losses(validation_scores, "Validation")
    if method == "max":
        return float(losses.max())
    if method == "percentile":
        if not 0 <= percentile <= 100:
            raise ValueError("Percentile must be between 0 and 100")
        return float(losses.quantile(percentile / 100))

    raise ValueError(f"Unknown threshold method: {method}")


def select_seed_edges(
    edge_scores: pd.DataFrame,
    threshold: float,
) -> Tuple[EdgeRef, ...]:
    required_columns = {"srcnode", "dstnode", "time", "edge_type", "loss"}
    missing = required_columns - set(edge_scores.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if not math.isfinite(threshold) or threshold < 0:
        raise ValueError("Threshold must be finite and nonnegative")
    losses = _validated_losses(edge_scores, "Anomaly")
    selected = edge_scores[losses >= threshold].copy()
    selected["loss"] = losses[losses >= threshold]

    edges = [
        EdgeRef(
            srcnode=int(row.srcnode),
            dstnode=int(row.dstnode),
            time=int(row.time),
            edge_type=int(row.edge_type),
            score=float(row.loss),
            key=getattr(row, "key", None) if _has_identity(getattr(row, "key", None)) else None,
        )
        for row in selected.itertuples(index=False)
    ]

    return tuple(edges)
