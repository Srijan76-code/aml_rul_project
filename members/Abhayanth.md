# Abhayanth — RUL target, validation, metrics and linear baseline

**Main job:** make sure every model learns the right target and gets evaluated fairly. Your earlier role is swapped with Srijan's: you own the evaluation setup and linear baseline now.

## When to start

Start at project setup, alongside Aryan's initial data audit. Agree on target and split rules before Srijan starts serious tree-model tuning or Ashrith starts DL work.

## Step-by-step tasks

### Phase 1 — AML foundation

- [ ] Read `docs/PROJECT_CONTEXT.md` and `docs/AML_CONCEPT_MAP.md`.
- [ ] Verify the columns and engine/cycle ordering with Aryan.
- [ ] Implement training RUL labels: `final_cycle_for_engine - current_cycle`.
- [ ] Do not silently cap RUL. If the team tests a cap later, log it as a separate experiment.
- [ ] Design a validation split by engine ID, not random rows. Discuss how to simulate the last-observed-cycle prediction used for test engines.
- [ ] Implement shared MAE and RMSE functions and test them on small known values.
- [ ] Train a simple linear regression baseline; add Ridge or Elastic Net to demonstrate regularization.
- [ ] Keep actual and predicted RUL aligned by engine/cycle. For the official test set, align one prediction at the last observed cycle of each engine with the matching provided test RUL.
- [ ] Record the exact split method, target settings, metrics and seed.

### Phase 2 — evaluation owner

- [ ] Run the final comparison using the same metric functions for the tree and DL models.
- [ ] Create a single comparison table with model, features/window, MAE, RMSE and notes.
- [ ] Break errors down by early/late degradation only after defining those ranges clearly.
- [ ] Verify final test evaluation was not used to choose hyperparameters.

## Expected deliverables

- `src/rul_prediction/targets.py`, `splits.py` and `metrics.py`
- `notebooks/03_linear_baseline.ipynb`
- final comparison table and evaluation-method section in the report

## How to approach it

1. Test RUL arithmetic on a tiny made-up trajectory.
2. Write down the validation strategy before inspecting model scores.
3. Use the same split IDs for every model.
4. Check the official test engine order/IDs against its RUL file.
5. Ask `leakage-reviewer` and `experiment-reviewer` to review the pipeline.

## Ask Claude Code to help

```text
Read CLAUDE.md, AGENTS.md, docs/PROJECT_CONTEXT.md and members/Abhayanth.md.
Explain the target and split design first. Implement only the next unchecked task.
Add small tests for target arithmetic, engine separation and metrics. Do not change the agreed
validation policy without asking. Run the tests and request leakage-reviewer and
experiment-reviewer checks.
```

## Be ready to explain

- How RUL is calculated for training data versus test predictions.
- Why we split by engine ID.
- The difference between MAE and RMSE.
- What Ridge/Elastic Net regularization does.
- How your protocol avoids leakage and unfair model comparisons.
