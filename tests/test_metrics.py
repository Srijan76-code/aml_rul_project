"""Tests for MAE and RMSE on values small enough to check by hand."""

import math

import numpy as np
import pandas as pd
import pytest

from rul_prediction.metrics import mae, rmse, score_predictions


def test_mae_and_rmse_hand_computed_example():
    # errors 2, 2, 3 -> MAE = 7/3, RMSE = sqrt((4 + 4 + 9) / 3)
    actual = [10, 20, 30]
    predicted = [12, 18, 33]

    assert mae(actual, predicted) == pytest.approx(7 / 3)
    assert rmse(actual, predicted) == pytest.approx(math.sqrt(17 / 3))


def test_perfect_predictions_score_zero():
    assert mae([5, 6], [5, 6]) == 0
    assert rmse([5, 6], [5, 6]) == 0


def test_rmse_punishes_one_big_miss_more_than_mae():
    # Same total error (40 cycles): spread out vs one big miss.
    spread = rmse([0, 0, 0, 0], [10, 10, 10, 10])
    one_big_miss = rmse([0, 0, 0, 0], [40, 0, 0, 0])

    assert mae([0, 0, 0, 0], [10, 10, 10, 10]) == mae([0, 0, 0, 0], [40, 0, 0, 0]) == 10
    assert spread == pytest.approx(10)
    assert one_big_miss == pytest.approx(20)


def test_over_and_under_prediction_count_equally():
    assert mae([100], [90]) == mae([100], [110]) == 10


def test_accepts_numpy_and_pandas_inputs():
    actual = pd.Series([1.0, 2.0])
    predicted = np.array([1.0, 4.0])

    assert score_predictions(actual, predicted) == {
        "mae": pytest.approx(1.0),
        "rmse": pytest.approx(math.sqrt(2)),
        "n": 2,
    }


@pytest.mark.parametrize(
    "y_true, y_pred",
    [
        ([1, 2, 3], [1, 2]),
        ([[1], [2]], [1, 2]),
        ([], []),
        ([1, float("nan")], [1, 2]),
        ([1, 2], [1, float("inf")]),
    ],
)
def test_rejects_mismatched_empty_or_non_finite_inputs(y_true, y_pred):
    with pytest.raises(ValueError):
        mae(y_true, y_pred)
    with pytest.raises(ValueError):
        rmse(y_true, y_pred)
