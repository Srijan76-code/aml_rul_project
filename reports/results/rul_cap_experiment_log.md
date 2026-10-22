# Experiment log — RUL cap and feature set

- **Owner:** Ashrith
- **Notebook:** notebooks/03_rul_cap_experiment.ipynb
- **Question:** (1) How does training on min(RUL, 125) change overall, early and late error graded on true RUL, and does it pass the protocol rule? (2) Do trees do better with all 57 features or cycle + 14 raw sensors (15)?
- **Dataset/subset:** FD001
- **Split protocol / engine IDs:** engine ID 80/20, seed 42, 100 val cut-offs (docs/fd001_validation_cutoffs.csv); validation engines [8, 9, 13, 18, 37, 46, 48, 50, 55, 62, 64, 69, 72, 74, 76, 84, 85, 87, 90, 95]
- **Target settings (including any RUL cap):** trained on true RUL or min(RUL, 125); always graded on true RUL
- **Features or window length:** v2 (57, with rolling) (57), cycle + 14 raw (15, no rolling) (15)
- **Models and hyperparameters:** Elastic Net (StandardScaler, alpha=1, l1_ratio=0.5, max_iter=50000); Random Forest (n_estimators=300, min_samples_leaf=20, max_features=0.33); Gradient boosting (learning_rate=0.05, max_iter=200, max_depth=3); not retuned
- **Random seed:** 42
- **Other settings:** scikit-learn 1.9.1, pandas 3.0.6, numpy 2.5.4
- **Band threshold:** early = true RUL > 125, late = true RUL <= 125; band counts: early 22, late 78
- **Validation MAE / RMSE:** see reports/results/rul_cap_experiment.csv (one row per run)
- **Test MAE / RMSE:** not run (validation-only experiment)
- **Plots/results file:** reports/figures/rul_cap_predicted_vs_actual.png, reports/results/rul_cap_experiment.csv
- **Interpretation:** see the Findings section of the notebook
