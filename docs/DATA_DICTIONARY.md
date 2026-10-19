# Data dictionary and normalization — FD001

Owner: Aryan. Evidence: `notebooks/01_eda.ipynb` (audit, sensor plots, sensor selection) and `notebooks/02_features.ipynb` (features, causality check, scaling).

## Raw files (`data/raw/`, read-only)

| File | Rows | Content |
|---|---|---|
| `train_FD001.txt` | 20,631 | 100 engines, run to failure (128–362 cycles each) |
| `test_FD001.txt` | 13,096 | 100 engines, stopped before failure (31–303 cycles each) |
| `RUL_FD001.txt` | 100 | True RUL at each test engine's last cycle; line *i* = engine *i*. Final evaluation only |

Load with `data.load_cmapss_file` (26 columns below) and `data.load_rul_file` (`engine_id`, `rul`). All structural checks pass: no missing values, no duplicates, cycles start at 1 and are consecutive and sorted.

## Columns

| Column | Meaning | Used as model input? |
|---|---|---|
| `engine_id` | Engine (unit) number | **No.** Used for splitting, grouping and ordering only |
| `cycle` | Operating cycle, starting at 1 | Yes (known at prediction time) |
| `setting_1`, `setting_2` | Operational settings | No for engineered features (design choice): \|linear corr with RUL\| < 0.01 and std ≈ 0.002 / 0.0003 on training engines (computed); NASA's readme says FD001 has one operating condition (external) |
| `setting_3` | Operational setting | No: constant |
| `sensor_1` … `sensor_21` | Sensor measurements; NASA's readme gives no names or units | Only the 14 informative sensors below |

Physical sensor names are in Table 2 of Saxena et al. (2008), *Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation*. They have not been checked yet, so they are not listed here.

## Sensor selection (decided on the 80 training engines only)

Rule (`features.select_informative_sensors`): remove a sensor if its most common value covers ≥ 95% of training rows.

- **Removed (7):** `sensor_1`, `sensor_5`, `sensor_10`, `sensor_16`, `sensor_18`, `sensor_19` (one value only); `sensor_6` (two values, 98% one value).
- **Kept (14):** `sensor_2`, `sensor_3`, `sensor_4`, `sensor_7`, `sensor_8`, `sensor_9`, `sensor_11`, `sensor_12`, `sensor_13`, `sensor_14`, `sensor_15`, `sensor_17`, `sensor_20`, `sensor_21`.

## Engineered features for AML models (`features.py`)

`add_rolling_features(df, sensors, window=10)` adds, per sensor: `<sensor>_mean_w10`, `<sensor>_std_w10` and `<sensor>_slope_w10`. Each is a trailing window over cycles max(1, c − 9)..c of the **same engine**. With one cycle of history, std and slope are 0. `model_feature_columns(sensors)` gives the 57 model inputs: `cycle` + 14 raw sensors + 42 rolling features.

## Normalization (AML and sequence models)

- Use `sklearn.preprocessing.StandardScaler`: each feature becomes (value − mean) / std.
- **Fit on the training engines' rows only**, then apply the same fitted scaler to validation and test. Keep it inside an sklearn pipeline, or fit it once and reuse it.
- **One global scaler for all engines.** Do not normalize per engine using that engine's whole-trajectory mean, min or max: that uses future cycles and is forbidden by `docs/EVALUATION_PROTOCOL.md`.
- **For sequence windows (Ashrith):** fit a **separate** `StandardScaler` on the 14 kept raw sensors of the training engines (the AML scaler is fitted on the 57 engineered columns and cannot be reused). Scale first, then cut windows that end at each `(engine_id, cycle)` and never cross engines. Windows for test engines come from `test_FD001.txt` alone.
