# Evaluation protocol — target, split and metrics

Owner: Abhayanth. Every model (linear, trees, LSTM, TCN) uses these settings so results are comparable. Do not change them without telling the team; changing any of them means re-running every model.

## 1. Target

- Training RUL: `rul = last cycle of that engine - current cycle` (`targets.add_train_rul`). Every engine ends at RUL 0.
- **No RUL cap (team decision, 2026-10-21).** The training target and the answer key both stay true RUL. To see how good a model is near failure, split the error into early and late points (Section 4) instead of capping the answers. A capped *training* target (e.g. `min(RUL, 125)`) may be tried only as a logged experiment, and it is always graded against true RUL.
- **Rule for the cap experiment, written before it runs:** capped training replaces the default only if, with the same features and model settings, the overall validation RMSE against true RUL is **no worse** than without the cap **and** the late band improves. A better late band alone is not enough, because a capped model is expected to trade early error for late error. Disclosure: Abhayanth ran an unlogged scratch version before this rule was written; in it, capped training made the overall RMSE worse for all three models.
- Test RUL: only the last observed cycle of each test engine has a label. Use `targets.last_observed_rows` on `test_FD001.txt`, then `targets.attach_test_rul` with `data.load_rul_file`. The join is by `engine_id` and fails on missing or repeated engines.

## 2. Train / validation split

- Split by **engine ID**, never by rows (`splits.split_engine_ids`, `splits.split_by_engine`).
- 20% of the 100 training engines go to validation, `seed = 42`.
- Validation engines (FD001, seed 42):

  `8, 9, 13, 18, 37, 46, 48, 50, 55, 62, 64, 69, 72, 74, 76, 84, 85, 87, 90, 95`

- The other 80 engines (16,390 rows) are for training; validation has 4,241 rows.
- Fit scalers, feature selection and PCA on the 80 training engines only.
- Final model fit: by default the same 80 training engines that were validated. Refitting on all 100 engines before the test run is a team decision, not yet made; if done, log it.

## 3. Validation scoring points (mimic the test set)

Test engines are cut off before failure and scored once, at their last observed cycle. Validation copies this:

- `splits.sample_cutoff_rows(val_df)` picks **5 random cut-off cycles per validation engine** → 100 points. `seed = 42`. The exact points are committed in `docs/fd001_validation_cutoffs.csv` and pinned by `tests/test_protocol_fd001.py`; check your rows against that file.
- Cut-offs lie between cycle 31 and the engine's last cycle − 1 (not failed yet). 31 is the shortest test history; it comes from test *input* lengths only, never test RUL labels. State this in the report.
- **Build features on the full validation table first, then sample cut-off rows.** This equals truncating the engine at the cut-off **only if every feature is causal**: the value at cycle c may use cycles ≤ c of the same engine and nothing else.
  - Allowed: trailing rolling mean/std/slope, expanding statistics, the current cycle number.
  - Not allowed: centred or forward rolling windows, backfill or interpolation across time, per-engine normalisation with the whole-trajectory mean/min/max, engine length or any "cycles until end" feature.
  - Feature owners should add a test: truncate an engine at cycle c, rebuild features, and check the row at c is unchanged.
- Sequence models: use the window that **ends** at each cut-off `(engine_id, cycle)`, never crossing into another engine. The window length and the padding/masking rule for cut-offs shorter than the window are the sequence owner's design, written down before tuning.
- Test features are built per engine from `test_FD001.txt` alone, using scalers fitted on the training engines.
- The **primary validation score** is MAE/RMSE on these 100 points. MAE/RMSE on all validation rows may be reported as a secondary number.

## 4. Metrics

- `metrics.mae`, `metrics.rmse`, `metrics.score_predictions` — units are cycles.
- MAE = average absolute miss. RMSE squares errors first, so big misses count more.
- Inputs must be equal-length 1-D arrays with no NaN/inf; the functions raise otherwise.
- **Early/late breakdown** (`metrics.score_by_rul_band`, constant `metrics.RUL_BAND_THRESHOLD`). Points are grouped by their **true** RUL, after prediction:
  - **early:** true RUL > 125. The engine is assumed to be still healthy, so large errors are expected.
  - **late:** true RUL <= 125. The engine is assumed to be degrading; this is the error that matters for maintenance.
  - The threshold 125 is borrowed from the capped training target used in C-MAPSS papers (125–130; the piecewise-linear target goes back to Heimes, 2008). There it is a cap value, not a measured start of degradation. It was fixed before any band scores were computed and is not tuned. The split is sensitive to it: 26 / 22 / 18 / 13 of the 100 validation points are early at 115 / 125 / 130 / 150.
  - The overall MAE/RMSE stays the primary score. The two bands are reported next to it, never instead of it.
  - Validation: 22 early points (from 14 engines) and 78 late points (from all 20). The early band is small, so its error is noisy.
  - **Test set:** no model gets test band scores or test band counts until the cap decision is made and the final model list is frozen, because band counts on the test set summarise test RUL (Section 5). Then the bands are computed once, from the same single scoring run. The test early band may be small or empty: report `n` and read NaN as "no points".

## 5. Use of the official test set

- Choose models, features and hyperparameters **on validation only**.
- Evaluate on the test set once per final model and record it. Do not change a model after seeing its test score.
- Do not summarise or plot test RUL values before final evaluation.

## Quick use

```python
from rul_prediction.data import load_cmapss_file
from rul_prediction.targets import add_train_rul
from rul_prediction.splits import split_engine_ids, split_by_engine, sample_cutoff_rows
from rul_prediction.metrics import score_by_rul_band, score_predictions

train = add_train_rul(load_cmapss_file("data/raw/train_FD001.txt"))
train_ids, val_ids = split_engine_ids(train["engine_id"])
train_df, val_df = split_by_engine(train, val_ids)
# ... build features on train_df and val_df, fit on train_df ...
val_points = sample_cutoff_rows(val_df)
predictions = model.predict(val_points[feature_columns])
scores = score_predictions(val_points["rul"], predictions)          # primary
band_scores = score_by_rul_band(val_points["rul"], predictions)     # early / late
```

## Known limitations

- The 100 points come from only 20 engines, and the 5 points of one engine are correlated, so the effective sample size is closer to 20 than 100. Small MAE/RMSE differences between models may be noise. Not yet measured for real models; Akhil measures it in Phase 1 with an engine-level bootstrap (`members/Akhil_Nath_Reddy.md`).
- One fixed split is tuned on repeatedly, so validation scores can become optimistic. Keep tuning grids small.
- Uniform cut-offs include early cycles with large RUL (up to ~330 cycles), which weigh heavily in uncapped RMSE. We deliberately do not match the cut-off distribution to the test RUL labels. The early/late error breakdown (Section 4) shows this effect.
- Seed 42 is used for both the engine split and the cut-off sampling.
