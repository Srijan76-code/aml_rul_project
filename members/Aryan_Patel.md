# Aryan Patel — EDA, preprocessing and feature engineering

**Main job:** turn raw FD001 files into clean, well-explained features for both AML and DL.

## When to start

Start first. Work with Abhayanth to agree on columns and row/engine identifiers before building target or modeling code.

## Step-by-step tasks

### Phase 1 — AML

- [x] Read `docs/PROJECT_CONTEXT.md` and `docs/AML_CONCEPT_MAP.md`.
- [x] Load the FD001 training file and inspect shape, column names, engine counts, cycle order, missing values and duplicate rows.
- [ ] Plot several sensors across engine cycles; show that sensor behavior can change during degradation.
- [ ] Check sensors for constant or nearly constant values; document the reason for keeping/removing each one.
- [ ] Create rolling mean, rolling standard deviation and/or rolling slope features using only current and past cycles.
- [ ] Keep engine ID and cycle available for splitting and ordering; exclude ID from sensor inputs unless the team has a justified reason to encode it.
- [ ] Fit scalers/feature selectors only on the training split.
- [ ] Optionally test PCA after the basic feature/model pipeline works; compare validation results rather than assuming PCA helps.

### Phase 2 — help sequence preparation

- [ ] Document the raw sensor columns and normalization approach for Ashrith.
- [ ] Check feature/window ordering on a few hand-inspected engine examples.

## Expected deliverables

- `notebooks/01_eda.ipynb`
- `notebooks/02_features.ipynb`
- reusable functions in `src/rul_prediction/data.py` and `features.py`
- data dictionary and EDA plots

## How to approach it

1. Audit raw data before cleaning it.
2. Plot before deleting sensors.
3. Write features as small functions and test on a tiny engine trajectory.
4. Ensure rolling features never use future cycles.
5. Have `data-auditor` inspect the EDA and `leakage-reviewer` review fitted preprocessing.

## Ask Claude Code to help

```text
Read CLAUDE.md, AGENTS.md, docs/PROJECT_CONTEXT.md and members/Aryan_Patel.md.
First summarize the data checks you will perform. Implement only the next unchecked task.
Do not change RUL targets or validation policy. Explain rolling features in simple AML terms,
run tests/checks, then request a data-auditor and leakage-reviewer review.
```

## Be ready to explain

- Why EDA comes before model training.
- How a rolling mean/slope is calculated.
- How feature selection differs from PCA.
- Why scaling before splitting can leak information.
- Why future sensor values must not be used to predict current RUL.
