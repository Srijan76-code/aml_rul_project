"""Pin the agreed FD001 split and validation points to the committed data.

If a library upgrade or code change shifts the split or cut-offs, these fail
loudly instead of every model silently being scored on different engines.
"""

from pathlib import Path

import pandas as pd

from rul_prediction.data import load_cmapss_file
from rul_prediction.splits import sample_cutoff_rows, split_by_engine, split_engine_ids
from rul_prediction.targets import add_train_rul

ROOT = Path(__file__).resolve().parents[1]
AGREED_VAL_IDS = [8, 9, 13, 18, 37, 46, 48, 50, 55, 62, 64, 69, 72, 74, 76, 84, 85, 87, 90, 95]


def fd001_train():
    return add_train_rul(load_cmapss_file(ROOT / "data" / "raw" / "train_FD001.txt"))


def test_fd001_validation_engines_match_protocol():
    train = fd001_train()

    train_ids, val_ids = split_engine_ids(train["engine_id"])
    train_df, val_df = split_by_engine(train, val_ids)

    assert val_ids == AGREED_VAL_IDS
    assert len(train_ids) == 80
    assert (len(train_df), len(val_df)) == (16390, 4241)


def test_fd001_cutoff_points_match_committed_csv():
    _, val_df = split_by_engine(fd001_train(), AGREED_VAL_IDS)
    expected = pd.read_csv(ROOT / "docs" / "fd001_validation_cutoffs.csv")

    cutoffs = sample_cutoff_rows(val_df)[["engine_id", "cycle", "rul"]]

    pd.testing.assert_frame_equal(cutoffs, expected)
