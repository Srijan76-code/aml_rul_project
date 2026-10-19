# AML topics you learned — where they fit

Use this as a learning map. The project should use relevant course ideas naturally; do not force every topic into it.

| Course topic | Use in this project | Priority |
|---|---|---|
| ML project lifecycle | Problem definition → data → features → model → evaluation → report | Core |
| Simple / multiple linear regression and OLS | First regression baseline with sensor/engine features | Core |
| Evaluation metrics for regression | MAE, RMSE, residuals and predicted-vs-actual plots | Core |
| Assumptions of linear regression | Check the linear baseline's residuals: residuals vs predicted (linearity, constant variance), residual histogram/Q-Q plot (normality), correlation between sensors (multicollinearity). Explain violations instead of hiding them | Core |
| Feature selection | Remove flat sensors only after checking; compare a justified feature subset | Core |
| Bias–variance tradeoff | Explain why a complex tree can overfit and why validation matters | Core |
| Regularization | Ridge / Elastic Net as a regularized linear baseline | Core/recommended |
| Gradient descent, SGD and mini-batch GD | Understand how neural models train; do not hand-code every optimizer | Concepts used in DL |
| Time series analysis | Keep engine cycles ordered; use rolling features and sequential windows | Core |
| PCA | Optional experiment after baselines: reduce sensor dimensions and compare validation results | Optional |
| Polynomial regression (assumption fix) | Optional only if there is a clear reason to model nonlinear tabular effects | Low priority |
| MLE, logistic regression, multiclass classification | Mainly classification/probability topics; RUL is continuous regression, so do not use these as the main model | Not needed for main task |

## Formulas worth knowing

`RUL = final cycle - current cycle`

`MAE = mean(abs(actual_RUL - predicted_RUL))`

`RMSE = sqrt(mean((actual_RUL - predicted_RUL)^2))`

Students should be able to explain the formula, unit (cycles), what a smaller number means, and what each metric penalizes.
