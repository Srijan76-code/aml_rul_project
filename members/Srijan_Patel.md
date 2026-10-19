# Srijan Patel — Tree-based AML models

**Main job:** implement and compare Random Forest and a boosting model. Your earlier role is swapped with Abhayanth's: you own the tree models now.

## When to start

Start after Aryan has created the agreed feature table and Abhayanth has verified the target, split and metric functions. You can inspect the data earlier, but do not invent a separate split or target.

## Step-by-step tasks

### Phase 1 — AML

- [ ] Read `docs/PROJECT_CONTEXT.md`, `docs/AML_CONCEPT_MAP.md` and the feature/data contract.
- [ ] Run a simple Random Forest regression model on the agreed training data.
- [ ] Run XGBoost if available; otherwise use scikit-learn Gradient Boosting and record which one you used.
- [ ] Compare models with the shared MAE and RMSE functions.
- [ ] Tune only a few parameters on validation data, e.g. tree depth, number of trees and learning rate for boosting.
- [ ] Plot predicted vs actual RUL and examine residuals.
- [ ] Record feature importance, but explain that importance is not proof of causation.
- [ ] Save settings, seed, metrics and figures; do not type made-up results into the report.

### Phase 2 — support and comparison

- [ ] Compare the best tree model with the LSTM/TCN using the same metric definitions.
- [ ] Explain where tree models do well and where they miss time-order information.
- [ ] Help write the tree-model results and limitations section.

## Expected deliverables

- `notebooks/03_tree_models.ipynb`
- reusable code in `src/rul_prediction/models/tree_models.py` (adjust to the repository layout)
- a results row for every run and the key plots

## How to approach it

1. Understand what each input feature means.
2. Fit a basic model first; keep the first result as a baseline.
3. Change one or two hyperparameters at a time.
4. Use validation results to select settings.
5. Ask `experiment-reviewer` to check fair comparison and ask `leakage-reviewer` to inspect how the data reached your model.

## Ask Claude Code to help

```text
Read CLAUDE.md, AGENTS.md, docs/PROJECT_CONTEXT.md and members/Srijan_Patel.md.
Explain the next unchecked task in terms of regression and bias-variance tradeoff,
then implement only that task. Use the agreed feature table, split and metrics.
Do not change target generation or another member's files. Run checks, report results,
and ask the experiment-reviewer and leakage-reviewer to audit the work.
```

## Be ready to explain

- Difference between Random Forest and boosting.
- Why this is regression, not classification.
- Why the train/validation split matters.
- What MAE and RMSE mean.
- What feature importance does and does not tell us.
