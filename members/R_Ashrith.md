# R Ashrith — sequence data and LSTM

**Main job:** build a basic DL model that learns from ordered sensor readings.

## When to start

Start sequence preparation after Aryan documents the sensor columns and Abhayanth shares the agreed target/split protocol. You can prototype on a tiny sample earlier, but confirm interfaces before training.

## Step-by-step tasks

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
