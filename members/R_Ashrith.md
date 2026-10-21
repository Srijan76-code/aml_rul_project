# R Ashrith — sequence data and LSTM

**Main job:** build a basic DL model that learns from ordered sensor readings. In Phase 1 you run the RUL-cap and feature-set experiment, which decides the target your LSTM will use.

## When to start

The Phase 1 experiment can start now: the target, split, metrics, features and tree models are merged on `main`.

Start sequence preparation after Aryan documents the sensor columns and Abhayanth shares the agreed target/split protocol. You can prototype on a tiny sample earlier, but confirm interfaces before training.

## Step-by-step tasks

### Phase 1 — RUL cap and feature-set experiment

Two open team decisions came out of the Phase 1 results:

- **RUL cap:** train on `min(RUL, 125)` instead of raw RUL? Early in life the sensors barely change, so RUL 300 and RUL 200 look almost the same to a model.
- **Rolling features for trees:** do the 42 rolling mean/std/slope columns help Random Forest and gradient boosting, or do `cycle` + the 14 raw sensors do as well?

Your job is to produce the evidence; the team makes the decision.

- [ ] Read `docs/EVALUATION_PROTOCOL.md`, `reports/results/linear_baseline_v2_experiment_log.md` and both `tree_*_experiment_log.md` files.
- [ ] In `notebooks/03_rul_cap_experiment.ipynb`, rebuild the shared split, features and the 100 pinned validation points (assert they match `docs/fd001_validation_cutoffs.csv`).
- [ ] Run a small grid: cap {none, 125} × features {v2 57 columns, `cycle` + 14 raw sensors} × models {Elastic Net, Random Forest, gradient boosting}. Reuse the settings already chosen in the linear and tree notebooks; do not retune.
- [ ] Clip only the **training** target. Score every run twice: against true RUL and against capped RUL. Explain why the two numbers differ so much.
- [ ] Report how many validation points have true RUL above 125, and plot predicted vs actual for the best capped and uncapped runs.
- [ ] Optional sensitivity check: caps 100 and 150 for the best model only.
- [ ] **Validation only.** Do not load the test files or score the test set; that happens once, after the team decides.
- [ ] Save `reports/results/rul_cap_experiment.csv` and `reports/results/rul_cap_experiment_log.md` from code, not by hand.
- [ ] Ask `leakage-reviewer` (is the cap applied only to training targets?) and `experiment-reviewer` (is the comparison fair?) to review, then open a PR.
- [ ] Present the results with a recommendation. Changing the target is a protocol change, so Abhayanth updates `docs/EVALUATION_PROTOCOL.md` only after the team agrees.

### Phase 2 — DL

- [ ] Read `docs/PROJECT_CONTEXT.md`, `docs/AML_CONCEPT_MAP.md` and Abhayanth's target/split design.
- [ ] Sort every engine by cycle.
- [ ] Build fixed-length sliding windows **within each engine only**; never join the end of one engine to the start of another.
- [ ] Define the target as RUL at the window's last cycle.
- [ ] Use training-only fitted scaling. Apply the same fitted scaler to validation/test data.
- [ ] Implement a small PyTorch LSTM regressor before adding layers or tuning many parameters.
- [ ] Track train and validation loss; use early stopping/checkpointing if appropriate.
- [ ] Save prediction outputs and evaluate with shared MAE/RMSE functions.
- [ ] Compare results with the best AML model under the same evaluation rules.

## Expected deliverables

- sequence-window dataset code
- `notebooks/04_lstm.ipynb`
- `src/rul_prediction/models/lstm.py`
- Phase 1: `notebooks/03_rul_cap_experiment.ipynb`, results CSV and experiment log
- training/validation loss plot and recorded results

## How to approach it

1. First print one window's shape and inspect which engine/cycles it contains.
2. Check a few window-target pairs by hand.
3. Overfit a tiny sample to verify the training loop, then train on the real split.
4. Keep the first LSTM simple and reproducible.
5. Ask `leakage-reviewer` to check window boundaries and `experiment-reviewer` to check comparison fairness.

## Ask Claude Code to help

```text
Read CLAUDE.md, AGENTS.md, docs/PROJECT_CONTEXT.md and members/R_Ashrith.md.
Teach me what one LSTM input window and target mean before coding. Implement only the next
unchecked task. Do not modify the official test protocol or tree-model code. Add shape and
engine-boundary checks, run the training/evaluation checks that are available, and request a
leakage-reviewer review.
```

## Be ready to explain

- Why the order of cycles matters.
- What sequence/window length means.
- What the LSTM receives and what it predicts.
- Why the training loss and validation loss may differ.
- Why the same MAE/RMSE functions should be used for AML and DL.
- Why a capped target can look much better against capped RUL but worse against true RUL.
- Why the cap is clipped on training targets only, and why the test set is not used to choose it.
