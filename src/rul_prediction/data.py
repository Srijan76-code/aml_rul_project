"""Load raw NASA C-MAPSS files and audit their structure.

This module only *reads and checks* raw data. It does not clean, drop sensors,
scale or create RUL labels. Those steps live elsewhere, so that every later
step starts from the same untouched raw table.

Raw file format (train_FD001.txt / test_FD001.txt): whitespace-separated text,
no header, 26 numeric columns per row:
    engine (unit) number, cycle, 3 operating settings, 21 sensor measurements.
"""

from pathlib import Path

import pandas as pd

# Proposed shared column names. Confirm them with Abhayanth (targets/splits)
# before other modules depend on them.
ID_COLUMNS = ["engine_id", "cycle"]
SETTING_COLUMNS = [f"setting_{i}" for i in range(1, 4)]
SENSOR_COLUMNS = [f"sensor_{i}" for i in range(1, 22)]
RAW_COLUMNS = ID_COLUMNS + SETTING_COLUMNS + SENSOR_COLUMNS


def load_cmapss_file(path):
    """Read one raw C-MAPSS train/test file into a DataFrame with named columns.

    Raises ValueError if the file does not have exactly 26 columns, because a
    wrong column count means our column names would be silently misaligned.
    """
    # sep=r"\s+" treats runs of spaces as one separator, which also absorbs
    # the trailing spaces at the end of each line in the NASA files.
    raw = pd.read_csv(Path(path), sep=r"\s+", header=None)
    if raw.shape[1] != len(RAW_COLUMNS):
        raise ValueError(
            f"{path}: expected {len(RAW_COLUMNS)} columns, found {raw.shape[1]}"
        )
    raw.columns = RAW_COLUMNS
    # IDs and cycles are counts; keeping them as integers avoids float
    # comparisons later when splitting by engine or ordering by cycle.
    raw[ID_COLUMNS] = raw[ID_COLUMNS].astype(int)
    return raw


def audit_raw_data(df):
    """Summarise structural data-quality checks for a raw C-MAPSS table.

    Returns a dict of plain numbers/booleans so the result can be printed in
    the EDA notebook and asserted in tests. Nothing in `df` is modified.
    """
    cycles_per_engine = df.groupby("engine_id")["cycle"].agg(["min", "max", "count"])

    # Each engine's cycles should be 1, 2, ..., N with no gaps or repeats.
    # That holds exactly when min == 1 and max == count, given no duplicate
    # (engine_id, cycle) pairs.
    duplicate_engine_cycles = int(df.duplicated(subset=ID_COLUMNS).sum())
    cycles_consecutive = bool(
        duplicate_engine_cycles == 0
        and (cycles_per_engine["min"] == 1).all()
        and (cycles_per_engine["max"] == cycles_per_engine["count"]).all()
    )

    # Row order matters for rolling features and sequence windows later.
    sorted_by_engine_then_cycle = bool(
        df[ID_COLUMNS].equals(df[ID_COLUMNS].sort_values(ID_COLUMNS))
    )

    return {
        "n_rows": int(df.shape[0]),
        "n_columns": int(df.shape[1]),
        "n_engines": int(df["engine_id"].nunique()),
        "engine_id_min": int(df["engine_id"].min()),
        "engine_id_max": int(df["engine_id"].max()),
        "cycles_per_engine_min": int(cycles_per_engine["count"].min()),
        "cycles_per_engine_median": float(cycles_per_engine["count"].median()),
        "cycles_per_engine_max": int(cycles_per_engine["count"].max()),
        "missing_values_total": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "duplicate_engine_cycle_pairs": duplicate_engine_cycles,
        "cycles_start_at_1_and_consecutive": cycles_consecutive,
        "sorted_by_engine_then_cycle": sorted_by_engine_then_cycle,
    }
