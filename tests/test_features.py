"""Tests for sensor selection and causal rolling features on tiny engines."""

import numpy as np
import pandas as pd
import pytest

from rul_prediction.data import SENSOR_COLUMNS
from rul_prediction.features import (
    add_rolling_features,
    model_feature_columns,
    rolling_feature_columns,
    select_informative_sensors,
)


def tiny_engines():
    # Engine 1: sensor_2 rises by 2 per cycle (slope 2). Engine 2: values far
    # from engine 1, so any leak across engines would show up in its features.
    return pd.DataFrame({
        "engine_id": [1, 1, 1, 1, 2, 2, 2],
        "cycle": [1, 2, 3, 4, 1, 2, 3],
        "sensor_2": [10.0, 12.0, 14.0, 16.0, 100.0, 100.0, 103.0],
    })


def test_rolling_features_by_hand_window_3():
    out = add_rolling_features(tiny_engines(), ["sensor_2"], window=3)
    engine_1 = out[out["engine_id"] == 1]

    # Cycle 1 has only itself; cycle 4 uses cycles 2-4 (12, 14, 16).
    assert engine_1["sensor_2_mean_w3"].tolist() == [10.0, 11.0, 12.0, 14.0]
    assert engine_1["sensor_2_std_w3"].tolist() == pytest.approx([0.0, np.sqrt(2), 2.0, 2.0])
    assert engine_1["sensor_2_slope_w3"].tolist() == pytest.approx([0.0, 2.0, 2.0, 2.0])


def test_rolling_features_never_cross_engines():
    out = add_rolling_features(tiny_engines(), ["sensor_2"], window=3)
    engine_2 = out[out["engine_id"] == 2]

    # Engine 2's first cycle must not see engine 1's last values.
    assert engine_2["sensor_2_mean_w3"].tolist() == pytest.approx([100.0, 100.0, 101.0])
    assert engine_2["sensor_2_std_w3"].iloc[0] == 0.0
    assert engine_2["sensor_2_slope_w3"].tolist() == pytest.approx([0.0, 0.0, 1.5])


def test_truncating_an_engine_does_not_change_earlier_rows():
    # The protocol's causality check: features at cycle c use only cycles <= c.
    full = add_rolling_features(tiny_engines(), ["sensor_2"], window=3)
    truncated_input = tiny_engines().query("not (engine_id == 1 and cycle > 2)")
    truncated = add_rolling_features(truncated_input, ["sensor_2"], window=3)

    pd.testing.assert_frame_equal(truncated, full.loc[truncated.index])


def test_rolling_features_keep_input_order_and_ids():
    shuffled = tiny_engines().sample(frac=1, random_state=0)

    out = add_rolling_features(shuffled, ["sensor_2"], window=3)
    expected = add_rolling_features(tiny_engines(), ["sensor_2"], window=3)

    assert out.index.tolist() == shuffled.index.tolist()
    pd.testing.assert_frame_equal(out.sort_index(), expected)


def test_rolling_features_do_not_modify_input():
    df = tiny_engines()
    add_rolling_features(df, ["sensor_2"], window=3)

    pd.testing.assert_frame_equal(df, tiny_engines())


def test_rolling_features_reject_duplicate_index():
    df = pd.concat([tiny_engines(), tiny_engines()])

    with pytest.raises(ValueError, match="index must be unique"):
        add_rolling_features(df, ["sensor_2"], window=3)


def test_select_informative_sensors_drops_constant_and_near_constant():
    n = 100
    train = pd.DataFrame({sensor: np.arange(n, dtype=float) for sensor in SENSOR_COLUMNS})
    train["sensor_1"] = 5.0  # constant
    train["sensor_6"] = [1.0] * 98 + [2.0] * 2  # 98% one value

    kept = select_informative_sensors(train)

    assert "sensor_1" not in kept
    assert "sensor_6" not in kept
    assert kept == [s for s in SENSOR_COLUMNS if s not in ("sensor_1", "sensor_6")]


def test_feature_columns_exclude_engine_id():
    columns = model_feature_columns(["sensor_2", "sensor_3"], window=5)

    assert "engine_id" not in columns
    assert columns[:3] == ["cycle", "sensor_2", "sensor_3"]
    assert columns[3:] == rolling_feature_columns(["sensor_2", "sensor_3"], window=5)
    assert len(columns) == 1 + 2 + 2 * 3


def test_rolling_slope_matches_polyfit_at_real_magnitudes():
    # Late cycles and a large sensor offset (sensor_9 is ~9000) are where the
    # cov/var shortcut could lose precision; compare with numpy's least squares.
    rng = np.random.default_rng(0)
    cycles = np.arange(1, 363)
    df = pd.DataFrame({
        "engine_id": 1,
        "cycle": cycles,
        "sensor_9": 9000 + 0.05 * cycles + rng.normal(0, 0.5, len(cycles)),
    })

    out = add_rolling_features(df, ["sensor_9"], window=10)

    for end in (10, 200, 362):
        window = df[(df["cycle"] > end - 10) & (df["cycle"] <= end)]
        expected = np.polyfit(window["cycle"], window["sensor_9"], 1)[0]
        assert out.loc[end - 1, "sensor_9_slope_w10"] == pytest.approx(expected, abs=1e-8)
