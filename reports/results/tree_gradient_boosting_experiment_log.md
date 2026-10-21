# Experiment log — GradientBoosting

- **Owner:** Srijan
- **Notebook:** notebooks/03_tree_models.ipynb
- **Question:** Does a non-linear tree ensemble beat the linear baseline on the shared protocol?
- **Dataset/subset:** FD001
- **Split protocol / engine IDs:** engine ID 80/20, seed 42, 100 val cut-offs (docs/fd001_validation_cutoffs.csv); validation engines [8, 9, 13, 18, 37, 46, 48, 50, 55, 62, 64, 69, 72, 74, 76, 84, 85, 87, 90, 95]
- **Target settings (including any RUL cap):** RUL = last cycle - cycle, no cap
- **Features or window length:** v2: cycle + 14 sensors + rolling mean/std/slope w10 (57)
- **Model:** sklearn HistGradientBoostingRegressor (XGBoost not installed; early_stopping=False)
- **Hyperparameters:** {'learning_rate': 0.05, 'max_iter': 200, 'max_depth': 3} (chosen by validation RMSE); grid: [{'learning_rate': 0.05, 'max_iter': 200, 'max_depth': 3}, {'learning_rate': 0.05, 'max_iter': 200, 'max_depth': None}, {'learning_rate': 0.05, 'max_iter': 500, 'max_depth': 3}, {'learning_rate': 0.05, 'max_iter': 500, 'max_depth': None}, {'learning_rate': 0.1, 'max_iter': 200, 'max_depth': 3}, {'learning_rate': 0.1, 'max_iter': 200, 'max_depth': None}, {'learning_rate': 0.1, 'max_iter': 500, 'max_depth': 3}, {'learning_rate': 0.1, 'max_iter': 500, 'max_depth': None}]
- **Random seed:** 42 (split, cut-offs and model)
- **Validation MAE / RMSE (cycles, 100 cut-offs):** 26.12 / 40.17
- **Test MAE / RMSE (cycles, 100 engines, run once):** 18.54 / 25.33
- **Training time / hardware:** 0.4 s fit on the 80 training engines, local CPU; scikit-learn 1.9.1
- **Plots/results file:** reports/figures/tree_vs_linear_predicted_vs_actual.png, reports/figures/tree_best_residuals.png, reports/figures/tree_permutation_importance.png, reports/results/tree_models_*.csv
- **What changed from previous experiment:** model family (linear baseline v2 -> GradientBoosting); same features, split and metrics
- **Interpretation:** see the Findings section of the notebook
- **Limitations / next experiment:** Random Forest and boosting tie on validation RMSE; 20 validation engines, so gaps of a few cycles may be noise; chosen settings sit at the edge of the grid (not widened after the test run); RUL cap or refitting on all 100 training engines only as separate logged team decisions
