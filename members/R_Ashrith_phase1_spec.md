# R Ashrith — Phase 1 spec: early/late error breakdown and RUL-cap experiment

This file is a step-by-step build spec. Follow it **in order**. Every step says which file to touch, what the code must do, which tests to write and how to check it. If anything here conflicts with `docs/EVALUATION_PROTOCOL.md`, stop and ask Abhayanth.

## 0. Background (read this first)

- Each validation point has a **true RUL** (cycles left before the engine breaks).
- Healthy engines look the same whether they have 300 or 200 cycles left, so every model makes big errors on them. These few big errors dominate the overall RMSE.
- **Team decision (2026-10-21):** the training target and the answer key stay **true RUL**. To show how good a model is near failure, we split the error into two groups by true RUL, **after** predicting:
  - **early:** true RUL > 125 (engine assumed still healthy);
  - **late:** true RUL <= 125 (engine assumed degrading, near failure).
- 125 is borrowed from the capped target used in C-MAPSS papers. It is fixed; do not try other thresholds.
- On the 100 validation points this gives **22 early points (from 14 engines) and 78 late points (from all 20)**.
- The model never knows about the groups. Grouping is used only for grading.

You build three things:

| Part | What | Files |
|---|---|---|
| A | The early/late scoring function | `src/rul_prediction/metrics.py`, `tests/test_metrics.py` |
| B | A reusable experiment runner (cap × features × models) | `src/rul_prediction/cap_experiment.py`, `tests/test_cap_experiment.py` |
| C | The experiment notebook and saved results | `notebooks/03_rul_cap_experiment.ipynb`, `reports/results/`, `reports/figures/` |

Then D (docs), E (review and PR).

The experiment answers two questions:

1. **Cap question:** how does *training* on `min(RUL, 125)` change the overall, early and late error, graded against **true** RUL? The decision rule is already written in `docs/EVALUATION_PROTOCOL.md` Section 1: capped training is adopted only if the overall RMSE is **no worse** than without the cap **and** the late band improves. A better late band alone is not enough.
2. **Rolling-features question:** do trees do better with all 57 features or with `cycle` + 14 raw sensors (15 features)? The team decides; differences of 1–2 cycles count as ties (only 20 validation engines).

## Hard rules (breaking any of these makes the work invalid)

1. **Never load the test set.** Do not open or import `data/raw/test_FD001.txt` or `data/raw/RUL_FD001.txt`, and do not import `load_rul_file`, `last_observed_rows` or `attach_test_rul`. Validation only.
2. **Cap only the training target.** Never change `val_points["rul"]`. All main scores are against true RUL.
3. **Do not retune.** Use exactly the model settings in Part C, step C3.
4. **Do not change existing functions** in `metrics.py`, `splits.py`, `targets.py`, `features.py`, `data.py` or `models/`. Only add the new code described here.
5. **Do not type numbers by hand** into results files or the log. Write them from code.
6. Overall MAE/RMSE stay the **primary** score. Early/late numbers are shown **next to** them, never instead of them.
7. Before each commit, `python -m pytest -q` must pass.

## Setup

```bash
cd aml_rul_project
git checkout main
git pull
git checkout -b ashrith/rul-band-and-cap
python3 -m venv .venv            # skip if .venv already exists
source .venv/bin/activate
pip install -r requirements.txt
python -m pytest -q              # expected: 64 passed
```

If the test count is not 64 on a fresh `main`, stop and tell the team.

---

## Part A — `score_by_rul_band` in `src/rul_prediction/metrics.py`

### A1. Add a constant

Below `import numpy as np` add:

```python
# Boundary between "early" and "late" degradation for the error breakdown.
# Fixed in advance from the C-MAPSS literature (125-130), not tuned on our results.
RUL_BAND_THRESHOLD = 125
```

### A2. Add the function

Place it **after** `score_predictions` and **before** `_checked_arrays`.

Signature (exact):

```python
def score_by_rul_band(y_true, y_pred, threshold=RUL_BAND_THRESHOLD):
```

Behaviour (all required):

1. First call `y_true, y_pred = _checked_arrays(y_true, y_pred)`. This gives the same input checks as `mae`/`rmse` (equal length, 1-D, not empty, no NaN/inf). Then, if `threshold` is not a finite number (`not np.isfinite(threshold)`), `raise ValueError("threshold must be a finite number")`. Without this check, a NaN threshold would silently put every point in "late".
2. `is_early = y_true > threshold`. Use **strictly greater than**: a point with true RUL exactly 125 is **late**.
3. Group by **`y_true` only**. Never use `y_pred` to decide the group.
4. Return exactly this shape:
   ```python
   {"early": {"mae": ..., "rmse": ..., "n": ...},
    "late":  {"mae": ..., "rmse": ..., "n": ...}}
   ```
   Each inner dict comes from the existing `score_predictions` on that group's rows.
5. If a group has **no points**, its dict is `{"mae": float("nan"), "rmse": float("nan"), "n": 0}`. It must not raise. Put this in a small private helper `_score_or_empty(y_true, y_pred)`, because `score_predictions` raises on empty input.
6. Write a docstring that says: points are grouped by **true** RUL after prediction, the model and the answers are unchanged, early = true RUL > threshold, late = true RUL <= threshold, and an empty group gives n = 0 with NaN errors.

### A3. Tests in `tests/test_metrics.py`

Extend the import line to:

```python
from rul_prediction.metrics import RUL_BAND_THRESHOLD, mae, rmse, score_by_rul_band, score_predictions
```

Add these 9 tests at the bottom. Use the exact inputs and expected values:

| Test name | Input `y_true`, `y_pred` | Must assert |
|---|---|---|
| `test_score_by_rul_band_hand_computed_example` | `[200, 150, 100, 20]`, `[150, 150, 90, 30]` | early: mae 25, rmse `math.sqrt(1250)`, n 2. late: mae 10, rmse 10, n 2. (Early errors 50 and 0; late errors 10 and 10.) Use `pytest.approx`. |
| `test_rul_band_boundary_value_counts_as_late` | `[RUL_BAND_THRESHOLD, RUL_BAND_THRESHOLD + 1]`, `[0, 0]` | late n == 1, late mae == 125, early n == 1 |
| `test_rul_band_groups_by_true_rul_not_prediction` | `[10]`, `[300]` | late n == 1, early n == 0 |
| `test_empty_rul_band_reports_zero_points_and_nan` | `[10, 20]`, `[12, 18]` | early n == 0, early mae and rmse are NaN (`math.isnan`), late n == 2 |
| `test_rul_bands_cover_every_point_once` | `rng = np.random.default_rng(0)`; `actual = rng.integers(0, 330, size=50)`; `predicted = rng.normal(150, 50, size=50)`; call with `threshold=100` | early n + late n == 50; early n == `int((actual > 100).sum())` |
| `test_rul_band_rejects_invalid_inputs` | `[1, 2, 3]`, `[1, 2]` | raises `ValueError` |
| `test_empty_late_band_reports_zero_points_and_nan` | `[200, 300]`, `[0, 0]` | late n == 0, late mae is NaN, early n == 2, early mae == 250 |
| `test_early_point_predicted_low_stays_early` | `[300]`, `[0]` | early n == 1, early mae == 300, late n == 0 |
| `test_rul_band_rejects_non_finite_threshold` | `[10, 200]`, `[10, 200]`, `threshold=float("nan")` | raises `ValueError` |

Then pin the threshold and the validation split in `tests/test_protocol_fd001.py`. Add one test, `test_validation_points_split_22_early_78_late`:

- assert `RUL_BAND_THRESHOLD == 125`;
- read `docs/fd001_validation_cutoffs.csv` the same way the other tests in that file read it;
- call `score_by_rul_band(points["rul"], points["rul"])` (perfect predictions, used here only for the counts);
- assert early n == 22 and late n == 78.

### A4. Check and commit

```bash
python -m pytest -q              # expected: 74 passed (64 + 9 + 1)
git add src/rul_prediction/metrics.py tests/test_metrics.py tests/test_protocol_fd001.py
git commit -m "Add early/late RUL band scoring to shared metrics"
```

---

## Part B — experiment runner `src/rul_prediction/cap_experiment.py` (new file)

Why a module and not only notebook code: the logic gets unit tests, and the notebook stays short.

### B1. File header and imports

```python
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
```

### B2. `cap_target(y, cap)`

- Returns a **new** `numpy` float array.
- `cap is None` → the same values, unchanged (a copy).
- Otherwise → `np.minimum(values, cap)`: every value above `cap` becomes `cap`, the others stay.
- `cap <= 0` → `raise ValueError("cap must be positive")`.
- Must **never** modify the input (Series or array).
- Docstring: "Used on TRAINING targets only. Validation answers are never capped."

### B3. `run_cap_grid(train_df, val_points, feature_sets, model_builders, caps, target_column=TARGET_COLUMN)`

Arguments:

- `train_df`: training rows, with feature columns and `target_column`.
- `val_points`: the 100 validation points, with feature columns and `target_column` (true RUL).
- `feature_sets`: dict `name -> list of column names`.
- `model_builders`: dict `name -> function with no arguments that returns a NEW unfitted model`.
- `caps`: dict `name -> None or a positive number`.

Loop order (must be exactly this, using dict insertion order):

```text
for cap_name, cap in caps.items():
    for features_name, columns in feature_sets.items():
        for model_name, build in model_builders.items():
```

Inside the loop:

1. `model = build()`. Call the builder every time, so each run gets a fresh model.
2. `model.fit(train_df[columns], cap_target(train_df[target_column], cap))`
3. `pred = np.asarray(model.predict(val_points[columns]), dtype=float)`
4. `y_true = val_points[target_column].to_numpy(dtype=float)`. **Not capped.**
5. `overall = score_predictions(y_true, pred)` and `bands = score_by_rul_band(y_true, pred)`.
6. `rmse_vs_capped_key`: `float("nan")` when `cap is None`; otherwise `score_predictions(cap_target(y_true, cap), pred)["rmse"]`. This is a capped-answer score, shown only to explain why capped numbers in papers look so low. It is not comparable with any of our other numbers, nor with published test-set numbers.
7. Append one row dict with exactly the keys in `RESULT_COLUMNS`:
   - `cap` = cap_name, `features` = features_name, `n_features` = `len(columns)`, `model` = model_name;
   - `n`, `mae`, `rmse` from `overall`;
   - `early_n`, `early_mae`, `early_rmse` from `bands["early"]`;
   - `late_n`, `late_mae`, `late_rmse` from `bands["late"]`;
   - `rmse_vs_capped_key` from step 6.
8. Store `predictions[(cap_name, features_name, model_name)] = pred`.

Return `results, predictions`, where `results = pd.DataFrame(rows, columns=RESULT_COLUMNS)`.

Do **not** modify `train_df` or `val_points`. Do **not** sort or reset their index.

### B4. Tests in `tests/test_cap_experiment.py` (new file)

Use this tiny synthetic data and spy model at the top of the file:

```python
import math

import numpy as np
import pandas as pd
import pytest

from rul_prediction.cap_experiment import RESULT_COLUMNS, cap_target, run_cap_grid

VAL_RUL = [300.0, 200.0, 100.0, 10.0]


def make_train():
    rul = np.arange(301, dtype=float)                    # 0 .. 300
    return pd.DataFrame({"a": rul, "b": rul * 0.5, "rul": rul})


def make_val():
    rul = np.array(VAL_RUL)
    return pd.DataFrame({"a": rul, "b": rul * 0.5, "rul": rul})


class ZeroSpyModel:
    """Remembers the largest training target it saw; always predicts 0."""

    def __init__(self):
        self.max_target_seen = None

    def fit(self, X, y):
        self.max_target_seen = float(np.max(y))
        return self

    def predict(self, X):
        return np.zeros(len(X))
```

Write these tests:

| Test name | What it does | Must assert |
|---|---|---|
| `test_cap_target_clips_values_above_cap` | `cap_target([200, 125, 50], 125)` | equals `[125, 125, 50]` (`np.testing.assert_array_equal`) |
| `test_cap_target_none_returns_unchanged_values` | `cap_target(pd.Series([300.0, 10.0]), None)` | equals `[300, 10]` |
| `test_cap_target_does_not_modify_input` | `s = pd.Series([300.0, 10.0])`; `cap_target(s, 125)` | `s.tolist() == [300.0, 10.0]` |
| `test_cap_target_rejects_non_positive_cap` | parametrize cap over `0` and `-5` | `pytest.raises(ValueError)` |
| `test_grid_has_one_row_per_combination_in_order` | 2 caps (`{"no cap": None, "cap 125": 125}`) × 2 feature sets (`{"only a": ["a"], "a and b": ["a", "b"]}`) × 2 models (`{"m1": ZeroSpyModel, "m2": ZeroSpyModel}`) | 8 rows; `list(results.columns) == RESULT_COLUMNS`; row 0 is `("no cap", "only a", "m1")`; last row is `("cap 125", "a and b", "m2")`; `len(predictions) == 8`; each prediction has length 4; `n_features` is 1 for "only a" and 2 for "a and b" |
| `test_training_target_is_capped` | builder that creates a `ZeroSpyModel`, appends it to a list and returns it; 1 feature set, caps as above | the no-cap spy saw max 300; the cap-125 spy saw max 125 |
| `test_grading_uses_true_rul_not_capped_rul` | `ZeroSpyModel`, caps as above | for **both** rows: `mae == pytest.approx(152.5)`, the mean of `VAL_RUL`. If grading wrongly used capped answers, the cap row would give 90.0. |
| `test_rmse_vs_capped_key_only_for_capped_runs` | same | no-cap row: `math.isnan(rmse_vs_capped_key)`; cap row: `pytest.approx(math.sqrt((125**2 + 125**2 + 100**2 + 10**2) / 4))` |
| `test_band_columns_split_at_125_by_true_rul` | same, no-cap row | `early_n == 2`, `late_n == 2`, `early_mae == pytest.approx(250)`, `late_mae == pytest.approx(55)` |
| `test_grid_does_not_modify_inputs` | keep `train.copy()` and `val.copy()` before the run | `pd.testing.assert_frame_equal` for both after the run |

Check and commit:

```bash
python -m pytest -q              # expected: 85 passed (74 + 11; the parametrized test counts twice)
git add src/rul_prediction/cap_experiment.py tests/test_cap_experiment.py
git commit -m "Add RUL-cap experiment runner graded on true RUL"
```

---

## Part C — notebook `notebooks/03_rul_cap_experiment.ipynb`

Create it in Jupyter or VS Code, saved in `notebooks/`. One idea per cell. Use a markdown heading before each code cell.

### C1. Title cell (markdown)

Title, owner (Ashrith), the two questions from section 0, and the line: *"Validation only. The test set is not loaded in this notebook."*

### C2. Setup and data (code; copy exactly)

```python
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path.cwd().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rul_prediction.cap_experiment import run_cap_grid
from rul_prediction.data import load_cmapss_file
from rul_prediction.features import add_rolling_features, model_feature_columns, select_informative_sensors
from rul_prediction.metrics import RUL_BAND_THRESHOLD
from rul_prediction.models.tree_models import make_gradient_boosting, make_random_forest
from rul_prediction.splits import DEFAULT_SEED, sample_cutoff_rows, split_by_engine, split_engine_ids
from rul_prediction.targets import add_train_rul

RESULTS_DIR = PROJECT_ROOT / "reports" / "results"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

train = add_train_rul(load_cmapss_file(PROJECT_ROOT / "data" / "raw" / "train_FD001.txt"))
train_ids, val_ids = split_engine_ids(train["engine_id"])
train_raw, val_raw = split_by_engine(train, val_ids)
sensors = select_informative_sensors(train_raw)
train_df = add_rolling_features(train_raw, sensors)
val_df = add_rolling_features(val_raw, sensors)
val_points = sample_cutoff_rows(val_df)

pinned = pd.read_csv(PROJECT_ROOT / "docs" / "fd001_validation_cutoffs.csv")
pd.testing.assert_frame_equal(val_points[["engine_id", "cycle", "rul"]], pinned)
print(len(train_df), "training rows,", len(val_points), "validation points,", len(sensors), "sensors")
```

Expected print: `16390 training rows, 100 validation points, 14 sensors`. If not, stop.

### C3. Experiment settings (code; copy exactly, do not change)

```python
FEATURE_SETS = {
    "v2 (57, with rolling)": model_feature_columns(sensors),
    "cycle + 14 raw (15, no rolling)": ["cycle"] + sensors,
}
# Settings chosen on validation in 03_linear_baseline and 03_tree_models. Not retuned here.
MODEL_BUILDERS = {
    "Elastic Net": lambda: make_pipeline(StandardScaler(), ElasticNet(alpha=1, l1_ratio=0.5, max_iter=50_000)),
    "Random Forest": lambda: make_random_forest(n_estimators=300, min_samples_leaf=20, max_features=0.33),
    "Gradient boosting": lambda: make_gradient_boosting(learning_rate=0.05, max_iter=200, max_depth=3),
}
CAPS = {"no cap": None, "cap 125": 125}
```

### C4. Run (code)

```python
results, predictions = run_cap_grid(train_df, val_points, FEATURE_SETS, MODEL_BUILDERS, CAPS)
results.round(2)
```

It takes about a minute (12 runs).

### C5. Sanity checks (code; must pass before you read any result)

The no-cap rows must reproduce the numbers already published in `reports/results/linear_baseline_results.csv`, `tree_models_results.csv`, `tree_models_ablation.csv` and the linear notebook's ablation (section 3b):

```python
expected_no_cap_rmse = {
    ("v2 (57, with rolling)", "Elastic Net"): 41.77,
    ("v2 (57, with rolling)", "Random Forest"): 40.17,
    ("v2 (57, with rolling)", "Gradient boosting"): 40.17,
    ("cycle + 14 raw (15, no rolling)", "Elastic Net"): 41.74,
    ("cycle + 14 raw (15, no rolling)", "Random Forest"): 37.13,
    ("cycle + 14 raw (15, no rolling)", "Gradient boosting"): 36.44,
}
no_cap = results[results["cap"] == "no cap"].set_index(["features", "model"])
for key, value in expected_no_cap_rmse.items():
    assert abs(no_cap.loc[key, "rmse"] - value) < 0.006, (key, no_cap.loc[key, "rmse"], value)
assert (results["n"] == 100).all()
assert (results["early_n"] == 22).all() and (results["late_n"] == 78).all()
print("sanity checks passed")
```

If any assert fails, **stop**. Something differs from the shared pipeline: the data, the features, the settings or the split. Do not "fix" the expected numbers.

### C6. Question 1 table: what does capped training change? (code; copy exactly)

```python
cap_table = results.pivot_table(
    index=["features", "model"], columns="cap", values=["rmse", "early_rmse", "late_rmse"]
)
for metric in ["rmse", "early_rmse", "late_rmse"]:
    cap_table[(metric, "change")] = cap_table[(metric, "cap 125")] - cap_table[(metric, "no cap")]
cap_table["passes_rule"] = (cap_table[("rmse", "change")] <= 0) & (cap_table[("late_rmse", "change")] < 0)
cap_table.round(2)
```

`change` < 0 means capped training lowered that error. `passes_rule` applies the protocol rule to each run: overall no worse **and** late better.

### C7. Question 2 table: do the rolling features help the trees? (code; copy exactly)

```python
no_cap = results[results["cap"] == "no cap"]
feature_table = no_cap.pivot_table(index="model", columns="features", values=["rmse", "late_rmse"])
for metric in ["rmse", "late_rmse"]:
    feature_table[(metric, "rolling minus no rolling")] = (
        feature_table[(metric, "v2 (57, with rolling)")]
        - feature_table[(metric, "cycle + 14 raw (15, no rolling)")]
    )
feature_table.round(2)
```

A positive "rolling minus no rolling" means the rolling features made that error **worse**.

### C8. Capped-answer score, clearly labelled (code + markdown)

Show `rmse_vs_capped_key` for the cap-125 rows only. The markdown cell above it must say: *"Capped-answer score: graded against min(RUL, 125) instead of true RUL. Shown only to explain why capped results look so low. Not comparable with any other number in this notebook, nor with published test-set results."*

### C9. Figure (code)

Pick two runs by **lowest overall `rmse`**: the best no-cap run and the best cap-125 run. Make one figure with 2 side-by-side panels, `figsize=(11, 4.5)`:

- x = true RUL (`val_points["rul"]`), y = predicted (`predictions[(cap, features, model)]`);
- a dashed diagonal line y = x;
- a dotted vertical line at x = `RUL_BAND_THRESHOLD`, labelled "early | late";
- panel title: cap, features, model and the run's overall and late RMSE (rounded to 2 decimals);
- axis labels "True RUL (cycles)" and "Predicted RUL (cycles)".

Save with `fig.savefig(FIGURES_DIR / "rul_cap_predicted_vs_actual.png", dpi=150, bbox_inches="tight")`.

### C10. Save results and log (code)

```python
results.to_csv(RESULTS_DIR / "rul_cap_experiment.csv", index=False)
```

Then write `RESULTS_DIR / "rul_cap_experiment_log.md"` from code, in the same field format as `reports/results/linear_baseline_v2_experiment_log.md`:

- Owner, Notebook, Question (both questions);
- Dataset/subset: FD001;
- Split protocol and the validation engine list (`sorted(val_ids)`);
- Target settings: "trained on true RUL or min(RUL, 125); always graded on true RUL";
- Models and hyperparameters (from C3), random seed (`DEFAULT_SEED`);
- Other settings: scikit-learn, pandas and numpy versions (`sklearn.__version__`, etc.);
- Band threshold (`RUL_BAND_THRESHOLD`) and the band counts;
- Test MAE / RMSE: "not run (validation-only experiment)";
- Plots/results files;
- Interpretation: "see the Findings section of the notebook".

### C11. Findings (markdown; write after reading the tables)

Answer both questions in plain sentences, using **only** numbers that appear in the tables above. Rules:

- Compare numbers only **within the same column**: overall with overall, late with late, true key with true key.
- Never compare `rmse_vs_capped_key` with anything else.
- There are only 20 validation engines, so differences of 1–2 cycles may be noise. Say so.
- Cap question: say for each model whether `passes_rule` is True or False, and what that means under the protocol rule. Do not invent a different rule.
- Say whether the early band (22 points, 14 engines) changed in the opposite direction to the late band.
- End with a **recommendation** for each question (adopt / do not adopt / needs more evidence). Say clearly that the team decides.

### C12. Optional (only if everything above is done)

Caps 100 and 150 for the single model and feature set with the lowest `late_rmse` among the cap-125 runs. Use the same `run_cap_grid` with one model and one feature set. Append these rows to the CSV with the cap name in the `cap` column. Do not change the main recommendation based on this alone.

### C13. Run top to bottom and check

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/03_rul_cap_experiment.ipynb
grep -n "test_FD001\|RUL_FD001\|load_rul_file\|attach_test_rul" notebooks/03_rul_cap_experiment.ipynb   # must print nothing
python -m pytest -q
git add notebooks/03_rul_cap_experiment.ipynb reports/results/rul_cap_experiment.csv reports/results/rul_cap_experiment_log.md reports/figures/rul_cap_predicted_vs_actual.png
git commit -m "Run validation-only RUL-cap and feature-set experiment"
```

---

## Part D — docs

1. In `docs/EVALUATION_PROTOCOL.md`, "Quick use" block: import `score_by_rul_band` next to `score_predictions` and replace the last line with:
   ```python
   predictions = model.predict(val_points[feature_columns])
   scores = score_predictions(val_points["rul"], predictions)          # primary
   band_scores = score_by_rul_band(val_points["rul"], predictions)     # early / late
   ```
   In Section 4, change "**Early/late breakdown** (to be added by Ashrith as `metrics.score_by_rul_band`, see `members/R_Ashrith_phase1_spec.md`)." to "**Early/late breakdown** (`metrics.score_by_rul_band`, constant `metrics.RUL_BAND_THRESHOLD`).", because the function now exists.
2. In `members/R_Ashrith.md`, tick the Phase 1 boxes you finished.
3. Commit: `git commit -am "Document early/late band scoring"`.

## Part E — review and pull request

1. Ask both reviewers (Claude Code agents in `.claude/agents/`):
   - `leakage-reviewer`: "Check that cap_experiment.py and 03_rul_cap_experiment.ipynb cap only training targets, never change validation answers, and never load test files."
   - `experiment-reviewer`: "Check the comparison is fair (same split, features and settings as the published notebooks), the sanity checks pass, and the Findings do not claim more than the tables show."
2. Fix what they find. Re-run C13.
3. Push and open the PR:
   ```bash
   git push -u origin ashrith/rul-band-and-cap
   gh pr create --title "Early/late RUL band scoring and validation-only RUL-cap experiment"
   ```
   PR body: what you added (Parts A–D), the two questions and your recommendations, "validation only, test set not loaded", and the reviewers' findings and fixes.
4. CI (`pytest`) must be green. Akhil reviews the PR (his Phase 1 human-review task). Do not merge your own PR.

## Common mistakes (check yourself)

| Mistake | Why it is wrong | How to spot it |
|---|---|---|
| Capping `val_points["rul"]` | Changes the answer key; scores look about 2× better than they are | `test_grading_uses_true_rul_not_capped_rul` fails |
| Grouping early/late by prediction | A late engine predicted at 300 would be counted as "early" | `test_rul_band_groups_by_true_rul_not_prediction` fails |
| Reusing one fitted model across runs | Later runs inherit earlier training | Each builder call must return a new model |
| Changing model settings "to improve" | That is retuning on validation, which makes the comparison unfair | C5 asserts fail, or the settings differ from C3 |
| Comparing a capped-key score with a true-key score | Different answer keys, so not comparable | Only `rmse_vs_capped_key` uses the capped key; compare within columns |
| Adopting the cap because the late band improved | The protocol rule also needs overall RMSE no worse | Read `passes_rule` in C6 |
| Computing band scores on test predictions | Band counts summarise test RUL before the decision | Not allowed until the final evaluation |
| Loading test files "just to look" | Breaks the protocol; the test set is used once at final evaluation | The C13 grep prints something |
| Typing numbers into the log by hand | Can drift from the real results | The log must be written by code in C10 |

## Ask Claude Code to help (copy-paste prompt)

```text
Read CLAUDE.md, AGENTS.md, docs/EVALUATION_PROTOCOL.md, members/R_Ashrith.md and
members/R_Ashrith_phase1_spec.md. Implement ONLY the next unfinished Part of the spec
(A, then B, then C, then D, then E), exactly as written: same file names, function
names, signatures, return shapes, test names and test values. Do not change existing
functions, model settings or the split, and never load test_FD001.txt or RUL_FD001.txt.
After each Part, run `python -m pytest -q`, show me the output, explain what the code
does in simple words, and stop before starting the next Part.
```

## Be ready to explain (viva)

- Why the early/late groups are formed from **true** RUL, after prediction.
- Why a capped model looks much better against capped answers but not against true answers.
- Why the cap is applied to training targets only.
- Why the no-cap rows must reproduce the published numbers before you trust the new ones.
- Why a 1–2 cycle difference on 20 validation engines may be noise.
