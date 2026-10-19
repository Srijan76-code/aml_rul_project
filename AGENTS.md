# Shared rules for every team member and coding agent

## Project in one sentence
Predict an engine's **Remaining Useful Life (RUL)** in operating cycles from its sensor history, then compare engineered-feature AML models against a sequence-based DL model.

## Non-negotiable rules

1. Start with NASA C-MAPSS **FD001** only. Add other subsets only after FD001 works and the team agrees.
2. This is a **regression** task: the output is a number of cycles, not a class or failure probability.
3. Keep each row's engine ID and cycle number throughout preprocessing.
4. Generate training RUL using that engine's last training cycle minus the current cycle. Document any RUL cap; do not silently cap labels.
5. Split training engines by **engine ID**, not by randomly splitting rows. A validation design should mimic the intended prediction point as closely as practical.
6. Fit scalers, feature selection, PCA and other learned preprocessing on the training split only.
7. Do not build a sliding window across two different engines.
8. Use the same target definition and MAE/RMSE functions to compare models.
9. Tune using validation results. Use the official test RUL labels for final evaluation only.
10. A good score is not enough: explain the method, check leakage, show plots, keep a record of experiments and document limitations.
11. The raw FD001 files (`train_FD001.txt`, `test_FD001.txt`, `RUL_FD001.txt`) are committed in `data/raw/` so every member uses identical data; never edit them. Do not commit other raw/processed data, API keys, tokens, environment files containing secrets, or large model checkpoints.
12. Do not overwrite another member's work or shared interface without discussion and review.

## How AI assistance should behave

- Teach before/while coding: connect each step to the student's course terms and explain formulas in plain English.
- Be an assistant, not an invisible project owner. The student should review and run the code and be able to explain it in viva.
- Ask before making an assumption that could change labels, data split, metrics or evaluation.
- For every code task: inspect first, propose a small plan, implement, run checks, summarize.
- Use the read-only auditors for independent checks. If they disagree, show evidence and let the human decide.
- Never fabricate model scores or claim that unrun code is verified.

## Shared interfaces to agree before coding

The team should agree on the column names and input/output shapes for:

- loading FD001 files;
- generating RUL targets;
- engine-level train/validation split;
- tabular feature generation;
- sequence-window generation;
- `MAE` and `RMSE` calculation.

Put final decisions in `docs/PROJECT_CONTEXT.md` or a linked protocol file and notify the team before changing them.
