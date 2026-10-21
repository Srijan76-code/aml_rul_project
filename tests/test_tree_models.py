"""Tests for tree model factories and importance on tiny synthetic data."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from rul_prediction.data import load_cmapss_file
from rul_prediction.features import add_rolling_features, model_feature_columns, select_informative_sensors
from rul_prediction.models.tree_models import (
    importance_table,
    make_gradient_boosting,
    make_random_forest,
)


def step_data(n=300, seed=0):
    # y jumps from 0 to 100 when x0 crosses 0.5; x1 is pure noise.
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"x0": rng.uniform(size=n), "x1": rng.uniform(size=n)})
    y = np.where(X["x0"] > 0.5, 100.0, 0.0)
    return X, y


@pytest.mark.parametrize("factory", [make_random_forest, make_gradient_boosting])
def test_factories_learn_a_step_a_straight_line_cannot(factory):
    X, y = step_data()
    model = factory(**({"n_estimators": 20} if factory is make_random_forest else {"max_iter": 50}))

    model.fit(X, y)

    probe = pd.DataFrame({"x0": [0.4, 0.6], "x1": [0.5, 0.5]})
    tree_predictions = model.predict(probe)
    line_predictions = LinearRegression().fit(X, y).predict(probe)

    assert tree_predictions[0] == pytest.approx(0, abs=5)
    assert tree_predictions[1] == pytest.approx(100, abs=5)
    # A straight line spreads the jump over the whole range, so near the step
    # it misses both sides by a wide margin.
    assert abs(line_predictions[0]) > 20
    assert abs(line_predictions[1] - 100) > 20


def test_random_forest_same_seed_identical_and_different_seed_differs():
    # Bootstrap sampling and feature subsampling make Random Forest random.
    X, y = step_data()
    X["x0"] += np.random.default_rng(1).normal(scale=0.05, size=len(X))

    first = make_random_forest(seed=7, n_estimators=10, max_features=0.5).fit(X, y).predict(X)
    second = make_random_forest(seed=7, n_estimators=10, max_features=0.5).fit(X, y).predict(X)
    other = make_random_forest(seed=8, n_estimators=10, max_features=0.5).fit(X, y).predict(X)

    np.testing.assert_array_equal(first, second)
    assert not np.array_equal(first, other)


@pytest.mark.parametrize("factory", [make_random_forest, make_gradient_boosting])
def test_factories_fit_the_agreed_fd001_feature_table(factory):
    # Smoke test on 5 real training engines: the shared features go in, one
    # finite, non-negative RUL per row comes out.
    root = Path(__file__).resolve().parents[1]
    train = load_cmapss_file(root / "data" / "raw" / "train_FD001.txt")
    train = train[train["engine_id"] <= 5].copy()
    train["rul"] = train.groupby("engine_id")["cycle"].transform("max") - train["cycle"]
    sensors = select_informative_sensors(train)
    table = add_rolling_features(train, sensors)
    features = model_feature_columns(sensors)
    params = {"n_estimators": 5} if factory is make_random_forest else {"max_iter": 10}

    predictions = factory(**params).fit(table[features], table["rul"]).predict(table[features])

    assert predictions.shape == (len(table),)
    assert np.isfinite(predictions).all()
    assert (predictions >= 0).all()


def test_random_forest_passes_tuned_params_through():
    model = make_random_forest(n_estimators=5, min_samples_leaf=3)

    assert model.n_estimators == 5
    assert model.min_samples_leaf == 3
    assert model.random_state == 42


def test_gradient_boosting_never_uses_row_level_early_stopping():
    assert make_gradient_boosting().early_stopping is False
    with pytest.raises(ValueError, match="engines"):
        make_gradient_boosting(early_stopping=True)


def test_importance_ranks_the_informative_column_first():
    X, y = step_data()
    model = make_random_forest(n_estimators=20).fit(X, y)

    table = importance_table(model, X, y, n_repeats=3)

    assert table.index[0] == "x0"
    assert table.loc["x0", "rmse_increase_mean"] > 10 * abs(table.loc["x1", "rmse_increase_mean"])
    assert list(table.columns) == ["rmse_increase_mean", "rmse_increase_std", "impurity_importance"]


def test_importance_marks_missing_impurity_as_nan_for_boosting():
    X, y = step_data()
    model = make_gradient_boosting(max_iter=20).fit(X, y)

    table = importance_table(model, X, y, n_repeats=2)

    assert table["impurity_importance"].isna().all()
