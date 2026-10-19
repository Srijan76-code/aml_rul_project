"""Tests for RUL labels and test-set alignment on tiny hand-made engines."""

import pandas as pd
import pytest

from rul_prediction.targets import add_train_rul, attach_test_rul, last_observed_rows


def tiny_engines():
    # Engine 1 fails at cycle 4, engine 2 at cycle 2.
    return pd.DataFrame(
        {
            "engine_id": [1, 1, 1, 1, 2, 2],
            "cycle": [1, 2, 3, 4, 1, 2],
            "sensor_1": [10.0, 11.0, 12.0, 13.0, 20.0, 21.0],
        }
    )


def test_train_rul_counts_down_to_zero_per_engine():
    labelled = add_train_rul(tiny_engines())

    assert labelled["rul"].tolist() == [3, 2, 1, 0, 1, 0]


def test_train_rul_uses_each_engines_own_last_cycle_even_if_unsorted():
    shuffled = tiny_engines().sample(frac=1, random_state=0)

    labelled = add_train_rul(shuffled)

    by_row = labelled.set_index(["engine_id", "cycle"])["rul"]
    assert by_row[(1, 1)] == 3
    assert by_row[(2, 1)] == 1


def test_train_rul_does_not_modify_input_or_cap():
    original = tiny_engines()
    long_engine = pd.DataFrame({"engine_id": [7] * 300, "cycle": range(1, 301)})

    add_train_rul(original)

    assert "rul" not in original.columns
    assert add_train_rul(long_engine)["rul"].max() == 299


def test_train_rul_requires_id_columns():
    with pytest.raises(ValueError, match="missing ID columns"):
        add_train_rul(pd.DataFrame({"engine_id": [1], "sensor_1": [1.0]}))


def test_last_observed_rows_returns_one_final_row_per_engine():
    last = last_observed_rows(tiny_engines().iloc[::-1])

    assert last["engine_id"].tolist() == [1, 2]
    assert last["cycle"].tolist() == [4, 2]
    assert last["sensor_1"].tolist() == [13.0, 21.0]


def test_attach_test_rul_joins_by_engine_id_not_row_order():
    last = last_observed_rows(tiny_engines())
    rul_table = pd.DataFrame({"engine_id": [2, 1], "rul": [50, 112]})

    joined = attach_test_rul(last, rul_table)

    assert joined.set_index("engine_id")["rul"].to_dict() == {1: 112, 2: 50}


def test_attach_test_rul_rejects_missing_engine():
    last = last_observed_rows(tiny_engines())
    rul_table = pd.DataFrame({"engine_id": [1], "rul": [112]})

    with pytest.raises(ValueError, match=r"\[2\]"):
        attach_test_rul(last, rul_table)


def test_attach_test_rul_rejects_more_than_one_row_per_engine():
    rul_table = pd.DataFrame({"engine_id": [1, 2], "rul": [112, 50]})

    with pytest.raises(pd.errors.MergeError):
        attach_test_rul(tiny_engines(), rul_table)


def test_attach_test_rul_keeps_the_callers_row_order():
    last = last_observed_rows(tiny_engines()).iloc[::-1].reset_index(drop=True)
    rul_table = pd.DataFrame({"engine_id": [1, 2], "rul": [112, 50]})

    joined = attach_test_rul(last, rul_table)

    assert joined["engine_id"].tolist() == [2, 1]
    assert joined["rul"].tolist() == [50, 112]


def test_attach_test_rul_rejects_extra_engine_in_answer_key():
    last = last_observed_rows(tiny_engines())
    rul_table = pd.DataFrame({"engine_id": [1, 2, 3], "rul": [112, 50, 7]})

    with pytest.raises(ValueError, match=r"\[3\]"):
        attach_test_rul(last, rul_table)


def test_last_observed_rows_survives_duplicated_index():
    df = tiny_engines()
    df.index = [0, 0, 1, 1, 2, 2]

    last = last_observed_rows(df)

    assert last["engine_id"].tolist() == [1, 2]
    assert last["cycle"].tolist() == [4, 2]
