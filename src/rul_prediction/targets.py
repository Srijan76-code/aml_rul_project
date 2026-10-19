"""Remaining Useful Life (RUL) targets for C-MAPSS train and test data.

Training engines run to failure, so every training row can be labelled from
its own engine's last cycle. Test engines stop before failure: only their last
observed cycle has a known RUL (from RUL_FD001.txt), so the test set is scored
on one row per engine.

No RUL cap is applied here. A cap changes the target, so it must be run as a
separately logged experiment, never as a silent default.
"""

from rul_prediction.data import ID_COLUMNS

TARGET_COLUMN = "rul"


def add_train_rul(df):
    """Return a copy of a run-to-failure table with a `rul` column added.

    RUL(engine, cycle) = last cycle of that engine - current cycle.
    The last cycle is looked up per engine because engines fail at very
    different times (128 to 362 cycles in FD001 train).
    """
    _require_id_columns(df)
    final_cycle = df.groupby("engine_id")["cycle"].transform("max")
    labelled = df.copy()
    labelled[TARGET_COLUMN] = final_cycle - df["cycle"]
    return labelled


def last_observed_rows(df):
    """Return each engine's last observed row, one per engine, ordered by ID.

    For test engines this is the prediction point: the model has seen history
    up to here and must say how many cycles remain.
    """
    _require_id_columns(df)
    # tail(1) after sorting works on position, so a duplicated DataFrame index
    # cannot return two rows for one engine (idxmax + loc could).
    ordered = df.sort_values(ID_COLUMNS)
    return ordered.groupby("engine_id").tail(1).reset_index(drop=True)


def attach_test_rul(last_rows, rul_table):
    """Join the official test RUL onto each test engine's last row by engine_id.

    Joining by ID (not by row position) means a missing, extra or repeated
    engine raises an error instead of silently pairing the wrong answers.
    Rows keep the order of `last_rows`, so predictions made on `last_rows`
    stay aligned with the returned `rul` column.
    """
    if TARGET_COLUMN in last_rows.columns:
        raise ValueError(f"last_rows already has a '{TARGET_COLUMN}' column")
    unmatched = sorted(set(last_rows["engine_id"]) ^ set(rul_table["engine_id"]))
    if unmatched:
        raise ValueError(f"engines missing from one side of the join: {unmatched}")
    return last_rows.merge(rul_table, on="engine_id", how="left", validate="one_to_one")


def _require_id_columns(df):
    missing = [column for column in ID_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"missing ID columns: {missing}")
