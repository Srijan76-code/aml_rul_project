"""Tests for the RUL-cap experiment runner on tiny synthetic data."""

import math

import numpy as np
import pandas as pd
import pytest

from rul_prediction.cap_experiment import RESULT_COLUMNS, cap_target, run_cap_grid

VAL_RUL = [300.0, 200.0, 100.0, 10.0]
CAPS = {"no cap": None, "cap 125": 125}


def make_train():
    rul = np.arange(301, dtype=float)  # 0 .. 300
    return pd.DataFrame({"a": rul, "b": rul * 0.5, "rul": rul})


def make_val():
    rul = np.array(VAL_RUL)
    return pd.DataFrame({"a": rul, "b": rul * 0.5, "rul": rul})


class ZeroSpyModel:
    """Remembers the largest training target it saw; always predicts 0."""

    def __init__(self):
        self.max_target_seen = None

    def fit(self, X, y):
        self.max_target_seen = float(np.max(y))
        return self

    def predict(self, X):
        return np.zeros(len(X))


def run_zero_model(caps=CAPS):
    return run_cap_grid(make_train(), make_val(), {"only a": ["a"]}, {"m": ZeroSpyModel}, caps)


def test_cap_target_clips_values_above_cap():
    np.testing.assert_array_equal(cap_target([200, 125, 50], 125), [125, 125, 50])


def test_cap_target_none_returns_unchanged_values():
    np.testing.assert_array_equal(cap_target(pd.Series([300.0, 10.0]), None), [300, 10])


def test_cap_target_does_not_modify_input():
    s = pd.Series([300.0, 10.0])

    cap_target(s, 125)

    assert s.tolist() == [300.0, 10.0]


@pytest.mark.parametrize("cap", [0, -5])
def test_cap_target_rejects_non_positive_cap(cap):
    with pytest.raises(ValueError):
        cap_target([1, 2], cap)


def test_grid_has_one_row_per_combination_in_order():
    feature_sets = {"only a": ["a"], "a and b": ["a", "b"]}
    builders = {"m1": ZeroSpyModel, "m2": ZeroSpyModel}

    results, predictions = run_cap_grid(make_train(), make_val(), feature_sets, builders, CAPS)

    assert len(results) == 8
    assert list(results.columns) == RESULT_COLUMNS
    assert tuple(results.iloc[0][["cap", "features", "model"]]) == ("no cap", "only a", "m1")
    assert tuple(results.iloc[-1][["cap", "features", "model"]]) == ("cap 125", "a and b", "m2")
    assert len(predictions) == 8
    assert all(len(p) == 4 for p in predictions.values())
    assert set(results.loc[results["features"] == "only a", "n_features"]) == {1}
    assert set(results.loc[results["features"] == "a and b", "n_features"]) == {2}


def test_training_target_is_capped():
    spies = []

    def build():
        spies.append(ZeroSpyModel())
        return spies[-1]

    run_cap_grid(make_train(), make_val(), {"only a": ["a"]}, {"m": build}, CAPS)

    assert spies[0].max_target_seen == 300
    assert spies[1].max_target_seen == 125


def test_grading_uses_true_rul_not_capped_rul():
    results, _ = run_zero_model()

    # Mean of VAL_RUL. Grading against capped answers would give 90.0 for the cap row.
    assert results["mae"].tolist() == [pytest.approx(152.5), pytest.approx(152.5)]


def test_rmse_vs_capped_key_only_for_capped_runs():
    results, _ = run_zero_model()

    assert math.isnan(results.loc[0, "rmse_vs_capped_key"])
    expected = math.sqrt((125**2 + 125**2 + 100**2 + 10**2) / 4)
    assert results.loc[1, "rmse_vs_capped_key"] == pytest.approx(expected)


def test_band_columns_split_at_125_by_true_rul():
    results, _ = run_zero_model()
    row = results.iloc[0]

    assert row["early_n"] == 2
    assert row["late_n"] == 2
    assert row["early_mae"] == pytest.approx(250)
    assert row["late_mae"] == pytest.approx(55)


def test_grid_does_not_modify_inputs():
    train, val = make_train(), make_val()
    train_before, val_before = train.copy(), val.copy()

    run_cap_grid(train, val, {"only a": ["a"]}, {"m": ZeroSpyModel}, CAPS)

    pd.testing.assert_frame_equal(train, train_before)
    pd.testing.assert_frame_equal(val, val_before)
