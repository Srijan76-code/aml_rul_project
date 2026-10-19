"""Engine-level train/validation split and test-like validation points.

Rows of one engine are near-copies of each other (cycle 50 looks almost like
cycle 51). A random row split would put the same engine on both sides, so the
model could memorise an engine and look better than it is. Every split here
therefore moves whole engines.

See docs/EVALUATION_PROTOCOL.md for the agreed settings.
"""

import numpy as np
import pandas as pd

from rul_prediction.data import ID_COLUMNS

DEFAULT_SEED = 42
DEFAULT_VAL_FRACTION = 0.2

# Official test engines are scored at one cut-off point each. We mimic that
# with several random cut-offs per validation engine: 20 validation engines x
# 5 cut-offs = 100 points, the same count as the 100 test engines.
DEFAULT_CUTOFFS_PER_ENGINE = 5

# The shortest test history in FD001 is 31 cycles, so validation never asks a
# model to predict from less history than the test set does. This uses only
# test input lengths, never test RUL labels.
DEFAULT_MIN_CUTOFF_CYCLE = 31


def split_engine_ids(engine_ids, val_fraction=DEFAULT_VAL_FRACTION, seed=DEFAULT_SEED):
    """Randomly assign whole engines to train or validation.

    Returns (train_ids, val_ids) as sorted lists of ints. The same IDs and seed
    always give the same split, so every model can share it.
    """
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be between 0 and 1")
    unique_ids = np.sort(pd.unique(np.asarray(engine_ids)))
    n_val = round(len(unique_ids) * val_fraction)
    if n_val == 0 or n_val == len(unique_ids):
        raise ValueError(
            f"{len(unique_ids)} engines with val_fraction={val_fraction} "
            "leaves one side empty"
        )

    rng = np.random.default_rng(seed)
    val_ids = np.sort(rng.choice(unique_ids, size=n_val, replace=False))
    train_ids = np.setdiff1d(unique_ids, val_ids)
    return train_ids.tolist(), val_ids.tolist()


def split_by_engine(df, val_ids):
    """Split a table into (train_df, val_df) by engine_id membership."""
    is_val = df["engine_id"].isin(val_ids)
    return df[~is_val].copy(), df[is_val].copy()


def sample_cutoff_rows(
    df,
    n_per_engine=DEFAULT_CUTOFFS_PER_ENGINE,
    min_cycle=DEFAULT_MIN_CUTOFF_CYCLE,
    seed=DEFAULT_SEED,
):
    """Pick test-like evaluation rows from validation engines.

    For each engine, pick `n_per_engine` distinct cut-off cycles between
    `min_cycle` and the engine's last cycle - 1 (so it has not failed yet), and
    keep only those rows.

    Scoring the row at cycle c is the same as truncating the engine at c,
    provided that row's features use only cycles <= c. So build features on the
    full validation table first, then call this.

    Rows are sorted by (engine_id, cycle) before sampling, so the same engines
    and seed give the same points whatever order the caller's table is in.
    """
    ordered = df.sort_values(ID_COLUMNS).reset_index(drop=True)
    rng = np.random.default_rng(seed)
    picked = []
    for engine_id, engine in ordered.groupby("engine_id", sort=True):
        last_cycle = engine["cycle"].max()
        candidates = engine[(engine["cycle"] >= min_cycle) & (engine["cycle"] < last_cycle)]
        if len(candidates) < n_per_engine:
            raise ValueError(
                f"engine {engine_id}: only {len(candidates)} cycles between "
                f"{min_cycle} and {last_cycle - 1}, need {n_per_engine}"
            )
        picked.append(candidates.sample(n=n_per_engine, random_state=rng))
    return pd.concat(picked).sort_values(ID_COLUMNS).reset_index(drop=True)
