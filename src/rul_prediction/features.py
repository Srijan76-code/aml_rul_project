"""Feature selection and causal rolling features for C-MAPSS RUL models.

Every feature value at cycle c uses only cycles <= c of the same engine, so
truncating an engine at c never changes its row at c (the rule in
docs/EVALUATION_PROTOCOL.md). Feature selection is "fitted" on the training
engines only and then applied unchanged to validation and test engines.
"""

import numpy as np

from rul_prediction.data import ID_COLUMNS, SENSOR_COLUMNS

# A sensor whose most common value covers at least this share of training rows
# is treated as (nearly) constant. In FD001 this drops 6 sensors that never
# change plus sensor_6 (98% one value); the next highest share is 26%, so the
# result does not hinge on the exact threshold.
NEAR_CONSTANT_SHARE = 0.95

# Trailing window length in cycles. Long enough to smooth sensor noise, well
# below the shortest test history (31 cycles). A hyperparameter: compare other
# values on validation, never on the test set.
DEFAULT_WINDOW = 10


def select_informative_sensors(train_df, max_mode_share=NEAR_CONSTANT_SHARE):
    """Return the sensors that vary on the training engines, in column order.

    Pass the training split only: deciding which sensors to keep is a fitted
    step, so it must not look at validation or test engines.
    """
    keep = []
    for sensor in SENSOR_COLUMNS:
        mode_share = train_df[sensor].value_counts(normalize=True).iloc[0]
        if mode_share < max_mode_share:
            keep.append(sensor)
    return keep


def rolling_feature_columns(sensors, window=DEFAULT_WINDOW):
    """Names of the columns `add_rolling_features` creates, in a fixed order."""
    return [
        f"{sensor}_{stat}_w{window}"
        for sensor in sensors
        for stat in ("mean", "std", "slope")
    ]


def model_feature_columns(sensors, window=DEFAULT_WINDOW):
    """Model inputs: current cycle, raw sensors and their rolling features.

    `engine_id` is deliberately excluded: it is an arbitrary label, so a model
    using it would memorise training engines instead of learning degradation.
    `cycle` is allowed because it is known at prediction time for test engines.
    """
    return ["cycle"] + list(sensors) + rolling_feature_columns(sensors, window)


def add_rolling_features(df, sensors, window=DEFAULT_WINDOW):
    """Return a copy of `df` with trailing rolling mean, std and slope per sensor.

    For each engine and cycle c, the window is cycles max(1, c - window + 1)..c.
    Early cycles use the shorter history available rather than looking ahead.
    - mean: smooths sensor noise.
    - std: how much the sensor fluctuates recently.
    - slope: least-squares trend per cycle, i.e. how fast the sensor drifts.
    With a single cycle of history, std and slope are 0 (no spread, no trend).
    """
    if not df.index.is_unique:
        raise ValueError("df index must be unique so features align back to rows")
    ordered = df.sort_values(ID_COLUMNS)
    by_engine = ordered.groupby("engine_id", sort=False)

    def trailing_mean(series):
        # groupby keeps the rolling window inside one engine, so the first
        # cycles of an engine never borrow the previous engine's last cycles.
        return series.groupby(ordered["engine_id"]).transform(
            lambda s: s.rolling(window, min_periods=1).mean()
        )

    cycle = ordered["cycle"].astype(float)
    mean_t = trailing_mean(cycle)
    var_t = trailing_mean(cycle**2) - mean_t**2

    features = {}
    for sensor in sensors:
        values = ordered[sensor]
        mean_x = trailing_mean(values)
        std_x = by_engine[sensor].transform(lambda s: s.rolling(window, min_periods=1).std())
        # Least-squares slope = cov(cycle, x) / var(cycle) over the window.
        cov_tx = trailing_mean(cycle * values) - mean_t * mean_x
        slope = cov_tx / var_t.where(var_t > 1e-9)

        features[f"{sensor}_mean_w{window}"] = mean_x
        features[f"{sensor}_std_w{window}"] = std_x.fillna(0.0)
        features[f"{sensor}_slope_w{window}"] = slope.fillna(0.0)

    result = df.copy()
    for name, column in features.items():
        result[name] = column.reindex(df.index).to_numpy(dtype=np.float64)
    return result
