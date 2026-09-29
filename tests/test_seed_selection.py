import pandas as pd
import pytest

from pidsmaker.incidents.models import EdgeRef
from pidsmaker.incidents.seeds import calculate_seed_threshold, select_seed_edges


@pytest.mark.parametrize(
    "losses, expected_threshold",
    [
        ([0.4, 2.5, 0.8, 1.2], 2.5),
        ([1, 4, 2], 4.0),
    ],
)
def test_threshold_uses_maximum_validation_loss(losses, expected_threshold):
    validation_scores = pd.DataFrame({"loss": losses})

    threshold = calculate_seed_threshold(validation_scores)

    assert threshold == expected_threshold
    assert isinstance(threshold, float)


@pytest.mark.parametrize("method", ["max", "percentile"])
def test_threshold_rejects_empty_validation_scores(method):
    validation_scores = pd.DataFrame({"loss": []})

    with pytest.raises(ValueError, match="Validation scores must not be empty"):
        calculate_seed_threshold(validation_scores, method=method)


@pytest.mark.parametrize(
    "percentile, expected_threshold",
    [(0, 1.0), (50, 3.0), (80, 23.2), (100, 100.0)],
)
def test_percentile_threshold(percentile, expected_threshold):
    validation_scores = pd.DataFrame({"loss": [1, 2, 3, 4, 100]})

    threshold = calculate_seed_threshold(
        validation_scores, method="percentile", percentile=percentile
    )

    assert threshold == pytest.approx(expected_threshold)
    assert isinstance(threshold, float)


def test_percentile_strategy_defaults_to_99_percent():
    validation_scores = pd.DataFrame({"loss": [1, 2, 3, 4, 100]})

    threshold = calculate_seed_threshold(validation_scores, method="percentile")

    assert threshold == pytest.approx(96.16)


@pytest.mark.parametrize("percentile", [-1, 101, float("nan")])
def test_percentile_threshold_rejects_invalid_percentile(percentile):
    validation_scores = pd.DataFrame({"loss": [1.0, 2.0]})

    with pytest.raises(ValueError, match="Percentile must be between 0 and 100"):
        calculate_seed_threshold(
            validation_scores, method="percentile", percentile=percentile
        )


def test_threshold_rejects_unknown_method():
    validation_scores = pd.DataFrame({"loss": [1.0, 2.0]})

    with pytest.raises(ValueError, match="Unknown threshold method: unknown"):
        calculate_seed_threshold(validation_scores, method="unknown")


def test_threshold_rejects_missing_loss_column():
    validation_scores = pd.DataFrame({"srcnode": [10]})

    with pytest.raises(ValueError, match="Missing required columns"):
        calculate_seed_threshold(validation_scores)


def test_select_seed_edges():
    scores = pd.DataFrame({
        "srcnode": [10, 20, 30],
        "dstnode": [20, 30, 40],
        "time": [100, 200, 300],
        "edge_type": [1, 2, 3],
        "loss": [0.4, 1.5, 2.0],
    })

    seeds = select_seed_edges(scores, threshold=1.5)

    assert seeds == (
        EdgeRef(20, 30, 200, 2, score=1.5),
        EdgeRef(30, 40, 300, 3, score=2.0),
    )


def test_missing_columns_raises_error():
    scores = pd.DataFrame({"srcnode": [10]})

    with pytest.raises(ValueError, match="Missing required columns"):
        select_seed_edges(scores, threshold=1.5)


def test_no_qualifying_edges_returns_empty_tuple():
    scores = pd.DataFrame({
        "srcnode": [10],
        "dstnode": [20],
        "time": [100],
        "edge_type": [1],
        "loss": [0.4],
    })

    assert select_seed_edges(scores, threshold=3.0) == ()
