"""Tests for MAE and RMSE on values small enough to check by hand."""

import math

import numpy as np
import pandas as pd
import pytest

from rul_prediction.metrics import RUL_BAND_THRESHOLD, mae, rmse, score_by_rul_band, score_predictions


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


def test_score_by_rul_band_hand_computed_example():
    # Early errors 50 and 0; late errors 10 and 10.
    bands = score_by_rul_band([200, 150, 100, 20], [150, 150, 90, 30])

    assert bands["early"] == {"mae": pytest.approx(25), "rmse": pytest.approx(math.sqrt(1250)), "n": 2}
    assert bands["late"] == {"mae": pytest.approx(10), "rmse": pytest.approx(10), "n": 2}


def test_rul_band_boundary_value_counts_as_late():
    bands = score_by_rul_band([RUL_BAND_THRESHOLD, RUL_BAND_THRESHOLD + 1], [0, 0])

    assert bands["late"]["n"] == 1
    assert bands["late"]["mae"] == 125
    assert bands["early"]["n"] == 1


def test_rul_band_groups_by_true_rul_not_prediction():
    bands = score_by_rul_band([10], [300])

    assert bands["late"]["n"] == 1
    assert bands["early"]["n"] == 0


def test_empty_rul_band_reports_zero_points_and_nan():
    bands = score_by_rul_band([10, 20], [12, 18])

    assert bands["early"]["n"] == 0
    assert math.isnan(bands["early"]["mae"])
    assert math.isnan(bands["early"]["rmse"])
    assert bands["late"]["n"] == 2


def test_rul_bands_cover_every_point_once():
    rng = np.random.default_rng(0)
    actual = rng.integers(0, 330, size=50)
    predicted = rng.normal(150, 50, size=50)

    bands = score_by_rul_band(actual, predicted, threshold=100)

    assert bands["early"]["n"] + bands["late"]["n"] == 50
    assert bands["early"]["n"] == int((actual > 100).sum())


def test_rul_band_rejects_invalid_inputs():
    with pytest.raises(ValueError):
        score_by_rul_band([1, 2, 3], [1, 2])


def test_empty_late_band_reports_zero_points_and_nan():
    bands = score_by_rul_band([200, 300], [0, 0])

    assert bands["late"]["n"] == 0
    assert math.isnan(bands["late"]["mae"])
    assert bands["early"]["n"] == 2
    assert bands["early"]["mae"] == 250


def test_early_point_predicted_low_stays_early():
    bands = score_by_rul_band([300], [0])

    assert bands["early"]["n"] == 1
    assert bands["early"]["mae"] == 300
    assert bands["late"]["n"] == 0


def test_rul_band_rejects_non_finite_threshold():
    with pytest.raises(ValueError):
        score_by_rul_band([10, 200], [10, 200], threshold=float("nan"))
