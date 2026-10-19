"""Tests for raw C-MAPSS loading and auditing on tiny hand-made files."""

import pandas as pd
import pytest

from rul_prediction.data import RAW_COLUMNS, audit_raw_data, load_cmapss_file


def make_row(engine_id, cycle, value=1.0):
    """One raw-format line: engine, cycle, then 24 setting/sensor values."""
    return " ".join([str(engine_id), str(cycle)] + [str(value)] * 24)


def write_raw_file(tmp_path, rows):
    path = tmp_path / "train_tiny.txt"
    # Trailing spaces mimic the real NASA files.
    path.write_text("\n".join(row + "  " for row in rows) + "\n")
    return path


def test_load_names_columns_and_keeps_integer_ids(tmp_path):
    path = write_raw_file(tmp_path, [make_row(1, 1), make_row(1, 2), make_row(2, 1)])

    df = load_cmapss_file(path)

    assert list(df.columns) == RAW_COLUMNS
    assert df.shape == (3, 26)
    assert df["engine_id"].tolist() == [1, 1, 2]
    assert df["cycle"].tolist() == [1, 2, 1]
    assert pd.api.types.is_integer_dtype(df["engine_id"])
    assert pd.api.types.is_integer_dtype(df["cycle"])


def test_load_rejects_wrong_column_count(tmp_path):
    path = tmp_path / "bad.txt"
    path.write_text("1 1 0.5 0.5\n")

    with pytest.raises(ValueError, match="expected 26 columns"):
        load_cmapss_file(path)


def test_audit_clean_two_engine_file(tmp_path):
    rows = [make_row(1, c) for c in (1, 2, 3)] + [make_row(2, c) for c in (1, 2)]
    df = load_cmapss_file(write_raw_file(tmp_path, rows))

    summary = audit_raw_data(df)

    assert summary["n_rows"] == 5
    assert summary["n_engines"] == 2
    assert summary["cycles_per_engine_min"] == 2
    assert summary["cycles_per_engine_max"] == 3
    assert summary["missing_values_total"] == 0
    assert summary["duplicate_rows"] == 0
    assert summary["cycles_start_at_1_and_consecutive"] is True
    assert summary["sorted_by_engine_then_cycle"] is True


def test_audit_flags_duplicates_gaps_and_bad_order(tmp_path):
    rows = [
        make_row(2, 1),  # engine 2 before engine 1 -> not sorted
        make_row(1, 1),
        make_row(1, 1),  # exact duplicate row
        make_row(1, 3),  # cycle 2 missing -> gap
    ]
    df = load_cmapss_file(write_raw_file(tmp_path, rows))

    summary = audit_raw_data(df)

    assert summary["duplicate_rows"] == 1
    assert summary["duplicate_engine_cycle_pairs"] == 1
    assert summary["cycles_start_at_1_and_consecutive"] is False
    assert summary["sorted_by_engine_then_cycle"] is False


def test_audit_counts_missing_values():
    df = pd.DataFrame({"engine_id": [1, 1], "cycle": [1, 2], "sensor_1": [1.0, None]})

    assert audit_raw_data(df)["missing_values_total"] == 1


def test_audit_flags_gap_without_duplicates(tmp_path):
    rows = [make_row(1, c) for c in (1, 2, 4)]
    df = load_cmapss_file(write_raw_file(tmp_path, rows))

    summary = audit_raw_data(df)

    assert summary["duplicate_engine_cycle_pairs"] == 0
    assert summary["cycles_start_at_1_and_consecutive"] is False


def test_audit_catches_repeated_cycle_with_different_values(tmp_path):
    # Not an exact duplicate row, but the same (engine, cycle) appears twice,
    # which would give one time step two different RUL rows.
    rows = [make_row(1, 1, value=1.0), make_row(1, 1, value=2.0), make_row(1, 2)]
    df = load_cmapss_file(write_raw_file(tmp_path, rows))

    summary = audit_raw_data(df)

    assert summary["duplicate_rows"] == 0
    assert summary["duplicate_engine_cycle_pairs"] == 1
