"""RUL-cap and feature-set experiment: fit on (optionally capped) training RUL,
grade on TRUE validation RUL, overall and per early/late band."""

import numpy as np
import pandas as pd

from rul_prediction.metrics import score_by_rul_band, score_predictions
from rul_prediction.targets import TARGET_COLUMN

RESULT_COLUMNS = [
    "cap", "features", "n_features", "model", "n",
    "mae", "rmse",
    "early_n", "early_mae", "early_rmse",
    "late_n", "late_mae", "late_rmse",
    "rmse_vs_capped_key",
]


def cap_target(y, cap):
    """Return a new float array with every value above `cap` set to `cap`.

    cap=None returns an unchanged copy. Used on TRAINING targets only.
    Validation answers are never capped.
    """
    values = np.array(y, dtype=float)  # np.array copies, so the input is never modified
    if cap is None:
        return values
    if cap <= 0:
        raise ValueError("cap must be positive")
    return np.minimum(values, cap)


def run_cap_grid(train_df, val_points, feature_sets, model_builders, caps, target_column=TARGET_COLUMN):
    """Fit every cap x feature set x model and score it on true validation RUL.

    feature_sets: name -> list of columns. model_builders: name -> no-argument
    function returning a NEW unfitted model. caps: name -> None or positive number.
    Returns (results DataFrame with RESULT_COLUMNS, {(cap, features, model): predictions}).
    """
    rows = []
    predictions = {}
    for cap_name, cap in caps.items():
        for features_name, columns in feature_sets.items():
            for model_name, build in model_builders.items():
                # A fresh model per run, so later runs cannot inherit earlier training.
                model = build()
                model.fit(train_df[columns], cap_target(train_df[target_column], cap))
                pred = np.asarray(model.predict(val_points[columns]), dtype=float)

                # Grade against the TRUE answers, never the capped ones.
                y_true = val_points[target_column].to_numpy(dtype=float)
                overall = score_predictions(y_true, pred)
                bands = score_by_rul_band(y_true, pred)

                # Capped-key score only explains why capped results look low in papers.
                if cap is None:
                    rmse_vs_capped_key = float("nan")
                else:
                    rmse_vs_capped_key = score_predictions(cap_target(y_true, cap), pred)["rmse"]

                rows.append({
                    "cap": cap_name,
                    "features": features_name,
                    "n_features": len(columns),
                    "model": model_name,
                    "n": overall["n"],
                    "mae": overall["mae"],
                    "rmse": overall["rmse"],
                    "early_n": bands["early"]["n"],
                    "early_mae": bands["early"]["mae"],
                    "early_rmse": bands["early"]["rmse"],
                    "late_n": bands["late"]["n"],
                    "late_mae": bands["late"]["mae"],
                    "late_rmse": bands["late"]["rmse"],
                    "rmse_vs_capped_key": rmse_vs_capped_key,
                })
                predictions[(cap_name, features_name, model_name)] = pred

    return pd.DataFrame(rows, columns=RESULT_COLUMNS), predictions
