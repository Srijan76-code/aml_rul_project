"""Shared regression metrics for every RUL model. Units are cycles.

All models (linear, trees, LSTM, TCN) must be scored with these functions so
their numbers are directly comparable.
"""

import numpy as np

# Boundary between "early" and "late" degradation for the error breakdown.
# Fixed in advance from the C-MAPSS literature (125-130), not tuned on our results.
RUL_BAND_THRESHOLD = 125


def mae(y_true, y_pred):
    """Mean absolute error: the average miss in cycles.

    Every cycle of error counts the same, so MAE reads as "off by X cycles on
    average".
    """
    y_true, y_pred = _checked_arrays(y_true, y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true, y_pred):
    """Root mean squared error, in cycles.

    Errors are squared before averaging, so one 100-cycle miss costs far more
    than ten 10-cycle misses. RMSE >= MAE always; a large gap means a few big
    misses.
    """
    y_true, y_pred = _checked_arrays(y_true, y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def score_predictions(y_true, y_pred):
    """Return {"mae", "rmse", "n"}: one row of the model comparison table."""
    return {"mae": mae(y_true, y_pred), "rmse": rmse(y_true, y_pred), "n": len(y_true)}


def score_by_rul_band(y_true, y_pred, threshold=RUL_BAND_THRESHOLD):
    """Score early and late points separately: {"early": {...}, "late": {...}}.

    Points are grouped by their TRUE RUL, after prediction. The model and the
    answers are unchanged; this is only a way of reading the errors.
    early = true RUL > threshold (engine assumed still healthy);
    late  = true RUL <= threshold (engine assumed degrading, near failure).
    A group with no points gives n = 0 with NaN errors instead of raising.
    """
    y_true, y_pred = _checked_arrays(y_true, y_pred)
    if not np.isfinite(threshold):
        raise ValueError("threshold must be a finite number")

    # Group by y_true only: grouping by y_pred would let the model choose its own band.
    is_early = y_true > threshold
    return {
        "early": _score_or_empty(y_true[is_early], y_pred[is_early]),
        "late": _score_or_empty(y_true[~is_early], y_pred[~is_early]),
    }


def _score_or_empty(y_true, y_pred):
    # score_predictions raises on empty input, but an empty band is a valid result.
    if len(y_true) == 0:
        return {"mae": float("nan"), "rmse": float("nan"), "n": 0}
    return score_predictions(y_true, y_pred)


def _checked_arrays(y_true, y_pred):
    # Mismatched lengths or NaNs would otherwise broadcast or propagate into a
    # metric that looks valid but compares the wrong rows.
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.ndim != 1 or y_true.shape != y_pred.shape:
        raise ValueError(
            f"y_true and y_pred must be 1-D with equal length, "
            f"got shapes {y_true.shape} and {y_pred.shape}"
        )
    if y_true.size == 0:
        raise ValueError("cannot score zero predictions")
    if not (np.isfinite(y_true).all() and np.isfinite(y_pred).all()):
        raise ValueError("y_true and y_pred must not contain NaN or infinity")
    return y_true, y_pred
