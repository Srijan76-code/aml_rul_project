"""Tests for engine-level splitting and test-like validation cut-offs."""

import pandas as pd
import pytest

from rul_prediction.splits import sample_cutoff_rows, split_by_engine, split_engine_ids


def engines(lengths):
    """Run-to-failure table: engine i has lengths[i-1] cycles."""
    rows = [
        (engine_id, cycle)
        for engine_id, length in enumerate(lengths, start=1)
        for cycle in range(1, length + 1)
    ]
    return pd.DataFrame(rows, columns=["engine_id", "cycle"])


def test_split_engine_ids_is_disjoint_complete_and_sized():
    train_ids, val_ids = split_engine_ids(range(1, 101), val_fraction=0.2, seed=42)

    assert len(val_ids) == 20
    assert set(train_ids).isdisjoint(val_ids)
    assert sorted(train_ids + val_ids) == list(range(1, 101))


def test_split_engine_ids_is_reproducible_and_seed_dependent():
    ids = range(1, 101)

    assert split_engine_ids(ids, seed=42) == split_engine_ids(ids, seed=42)
    assert split_engine_ids(ids, seed=42) != split_engine_ids(ids, seed=7)


def test_split_engine_ids_ignores_repeated_ids_from_row_level_input():
    row_level_ids = [1, 1, 1, 2, 2, 3, 3, 4, 5]

    train_ids, val_ids = split_engine_ids(row_level_ids, val_fraction=0.4)

    assert len(val_ids) == 2
    assert sorted(train_ids + val_ids) == [1, 2, 3, 4, 5]


@pytest.mark.parametrize("bad_fraction", [0, 1, 1.5])
def test_split_engine_ids_rejects_fraction_outside_zero_one(bad_fraction):
    with pytest.raises(ValueError):
        split_engine_ids(range(1, 11), val_fraction=bad_fraction)


def test_split_by_engine_never_puts_one_engine_on_both_sides():
    df = engines([5, 6, 7, 8])

    train_df, val_df = split_by_engine(df, val_ids=[2, 4])

    assert set(train_df["engine_id"]) == {1, 3}
    assert set(val_df["engine_id"]) == {2, 4}
    assert len(train_df) + len(val_df) == len(df)


def test_cutoff_rows_stay_inside_allowed_cycle_range():
    df = engines([40, 60])

    cutoffs = sample_cutoff_rows(df, n_per_engine=5, min_cycle=31, seed=0)

    assert len(cutoffs) == 10
    assert cutoffs.groupby("engine_id").size().tolist() == [5, 5]
    assert cutoffs["cycle"].min() >= 31
    # Never the failure cycle itself: a test engine has not failed yet.
    assert cutoffs.loc[cutoffs["engine_id"] == 1, "cycle"].max() <= 39
    assert not cutoffs.duplicated(["engine_id", "cycle"]).any()


def test_cutoff_rows_are_reproducible_with_seed():
    df = engines([100, 120, 140])

    first = sample_cutoff_rows(df, seed=3)
    second = sample_cutoff_rows(df, seed=3)

    pd.testing.assert_frame_equal(first, second)


def test_cutoff_rows_reject_engine_too_short_for_requested_points():
    with pytest.raises(ValueError, match="engine 1"):
        sample_cutoff_rows(engines([33]), n_per_engine=5, min_cycle=31)


def test_cutoff_rows_do_not_depend_on_input_row_order():
    df = engines([100, 120, 140])
    shuffled = df.sample(frac=1, random_state=0)

    pd.testing.assert_frame_equal(sample_cutoff_rows(df), sample_cutoff_rows(shuffled))
