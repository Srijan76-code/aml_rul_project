"""Tree-based regressors for RUL: Random Forest and gradient boosting.

The factories fix the settings every run must share (seed, no row-level
internal validation) so notebooks only vary the few hyperparameters being
tuned on the shared engine-level validation split.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance

from rul_prediction.splits import DEFAULT_SEED


def make_random_forest(seed=DEFAULT_SEED, **params):
    """Random Forest: many deep trees on bootstrap samples, predictions averaged.

    Averaging many noisy trees (bagging) mainly lowers variance.
    """
    return RandomForestRegressor(random_state=seed, n_jobs=-1, **params)


def make_gradient_boosting(seed=DEFAULT_SEED, **params):
    """Gradient boosting: small trees added one by one, each fitting the remaining errors.

    Adding trees step by step mainly lowers bias. scikit-learn's
    HistGradientBoostingRegressor is used because XGBoost is not installed.

    Internal early stopping is turned off: it would hold out random *rows*,
    putting cycles of the same engine on both sides. Tuning uses the shared
    engine-level validation split instead.
    """
    if params.get("early_stopping"):
        raise ValueError("internal early stopping splits rows, not engines; tune on the shared split")
    params["early_stopping"] = False
    return HistGradientBoostingRegressor(random_state=seed, **params)


def importance_table(model, X, y, seed=DEFAULT_SEED, n_repeats=10):
    """Permutation importance of each column, plus impurity importance if the model has it.

    Permutation importance = how much RMSE rises when one column is shuffled,
    measured on the rows passed in (use validation rows, not training rows, so
    it reflects what generalises). Correlated columns share credit, and high
    importance shows what the model relies on, not what causes failure.
    """
    result = permutation_importance(
        model,
        X,
        y,
        scoring="neg_root_mean_squared_error",
        n_repeats=n_repeats,
        random_state=seed,
        n_jobs=-1,
    )
    impurity = getattr(model, "feature_importances_", np.full(X.shape[1], np.nan))
    table = pd.DataFrame(
        {
            "rmse_increase_mean": result.importances_mean,
            "rmse_increase_std": result.importances_std,
            "impurity_importance": impurity,
        },
        index=pd.Index(X.columns, name="feature"),
    )
    return table.sort_values("rmse_increase_mean", ascending=False)
