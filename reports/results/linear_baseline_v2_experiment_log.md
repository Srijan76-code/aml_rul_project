# Experiment log — linear baseline v2

- **Owner:** Abhayanth
- **Notebook:** notebooks/03_linear_baseline.ipynb
- **Question:** Do Aryan's engineered features (cycle + rolling mean/std/slope) improve the linear baseline?
- **Dataset/subset:** FD001
- **Split protocol / engine IDs:** engine ID 80/20, seed 42, 100 val cut-offs (docs/fd001_validation_cutoffs.csv); validation engines [8, 9, 13, 18, 37, 46, 48, 50, 55, 62, 64, 69, 72, 74, 76, 84, 85, 87, 90, 95]
- **Target settings (including any RUL cap):** RUL = last cycle - cycle, no cap
- **Features or window length:** cycle + 14 sensors + rolling mean/std/slope w10 (57): cycle, sensor_2, sensor_3, sensor_4, sensor_7, sensor_8, sensor_9, sensor_11, sensor_12, sensor_13, sensor_14, sensor_15, sensor_17, sensor_20, sensor_21, sensor_2_mean_w10, sensor_2_std_w10, sensor_2_slope_w10, sensor_3_mean_w10, sensor_3_std_w10, sensor_3_slope_w10, sensor_4_mean_w10, sensor_4_std_w10, sensor_4_slope_w10, sensor_7_mean_w10, sensor_7_std_w10, sensor_7_slope_w10, sensor_8_mean_w10, sensor_8_std_w10, sensor_8_slope_w10, sensor_9_mean_w10, sensor_9_std_w10, sensor_9_slope_w10, sensor_11_mean_w10, sensor_11_std_w10, sensor_11_slope_w10, sensor_12_mean_w10, sensor_12_std_w10, sensor_12_slope_w10, sensor_13_mean_w10, sensor_13_std_w10, sensor_13_slope_w10, sensor_14_mean_w10, sensor_14_std_w10, sensor_14_slope_w10, sensor_15_mean_w10, sensor_15_std_w10, sensor_15_slope_w10, sensor_17_mean_w10, sensor_17_std_w10, sensor_17_slope_w10, sensor_20_mean_w10, sensor_20_std_w10, sensor_20_slope_w10, sensor_21_mean_w10, sensor_21_std_w10, sensor_21_slope_w10
- **Model:** StandardScaler + ElasticNet (chosen by validation RMSE among OLS, Ridge, ElasticNet)
- **Hyperparameters:** {'alpha': 1, 'l1_ratio': 0.5}; grids: Ridge alpha [0.01, 0.1, 1, 10, 100, 1000], ElasticNet alpha [0.001, 0.01, 0.1, 1] x l1_ratio [0.2, 0.5, 0.8]
- **Random seed:** 42 (split and cut-offs; models are deterministic)
- **Other settings:** StandardScaler defaults (mean/std from training engines); ElasticNet max_iter=50000; scikit-learn 1.9.1, scipy 1.18.1, pandas 3.0.6, numpy 2.5.4
- **Validation MAE / RMSE (cycles, 100 cut-offs):** 31.66 / 41.77
- **Test MAE / RMSE (cycles, 100 engines, run once):** 23.61 / 29.04
- **Plots/results file:** reports/figures/linear_baseline_v2_validation_diagnostics.png, reports/results/linear_baseline_results.csv, reports/results/linear_baseline_grid.csv
- **What changed from previous experiment:** features only: v1 raw settings + sensors -> v2 engineered features; same split, grids and seed
- **Interpretation:** see the Findings section of the notebook
- **Limitations / next experiment:** RUL cap or refitting on all 100 training engines only as separate logged team decisions
