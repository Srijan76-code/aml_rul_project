# Experiment log — linear baseline v1

- **Owner:** Abhayanth
- **Notebook:** notebooks/03_linear_baseline.ipynb
- **Question:** How well does a linear model on raw settings + sensors predict RUL, and does regularization help?
- **Dataset/subset:** FD001
- **Split protocol / engine IDs:** engine ID 80/20, seed 42, 100 val cut-offs (docs/fd001_validation_cutoffs.csv); validation engines [8, 9, 13, 18, 37, 46, 48, 50, 55, 62, 64, 69, 72, 74, 76, 84, 85, 87, 90, 95]
- **Target settings (including any RUL cap):** RUL = last cycle - cycle, no cap
- **Features or window length:** raw settings + sensors (24): setting_1, setting_2, setting_3, sensor_1, sensor_2, sensor_3, sensor_4, sensor_5, sensor_6, sensor_7, sensor_8, sensor_9, sensor_10, sensor_11, sensor_12, sensor_13, sensor_14, sensor_15, sensor_16, sensor_17, sensor_18, sensor_19, sensor_20, sensor_21
- **Model:** StandardScaler + OLS (chosen by validation RMSE among OLS, Ridge, ElasticNet)
- **Hyperparameters:** {}; grids: Ridge alpha [0.01, 0.1, 1, 10, 100, 1000], ElasticNet alpha [0.001, 0.01, 0.1, 1] x l1_ratio [0.2, 0.5, 0.8]
- **Random seed:** 42 (split and cut-offs; models are deterministic)
- **Other settings:** StandardScaler defaults (mean/std from training engines); ElasticNet max_iter=50000; scikit-learn 1.9.1, scipy 1.18.1, pandas 3.0.6, numpy 2.5.4
- **Validation MAE / RMSE (cycles, 100 cut-offs):** 32.86 / 42.92
- **Test MAE / RMSE (cycles, 100 engines, run once):** 24.91 / 31.17
- **Plots/results file:** reports/figures/linear_baseline_validation_diagnostics.png, reports/results/linear_baseline_results.csv, reports/results/linear_baseline_grid.csv
- **What changed from previous experiment:** first model
- **Interpretation:** see the Findings section of the notebook
- **Limitations / next experiment:** RUL cap or refitting on all 100 training engines only as separate logged team decisions
