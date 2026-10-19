# Project context — read before coding

## Problem
**Remaining Useful Life (RUL)** means how many operating cycles remain before the engine reaches its failure point. For example, if failure is at cycle 200 and the current cycle is 150, RUL is 50 cycles.

For a training engine:

`RUL(engine, cycle) = final_cycle_of_that_training_engine - current_cycle`

The C-MAPSS training trajectories run to failure. Test trajectories stop before failure; the supplied test RUL file gives the true RUL at each test engine's last observed cycle. Keep predictions aligned with those test engine IDs.

## Dataset and target scope

- Recommended dataset in the Project 7 brief: NASA C-MAPSS Turbofan Engine Degradation Dataset.
- Start with **FD001** to keep scope manageable.
- Inputs include engine ID, cycle, operating settings and sensor measurements.
- Target: one continuous RUL value (in cycles).
- Core comparison: engineered degradation features versus sequence learning.
- Do not claim that the model knows the exact future failure time; it estimates RUL from observed history.

The project brief specifically recommends cycle, rolling mean, rolling slope, operating-regime and sensor-health features for AML, and LSTM / temporal CNN / Transformer-style fixed-length sensor windows for DL. It requires engine-ID splits and RMSE/MAE, including error analysis for early versus late degradation.

## Project phases

### Phase 1 — Advanced Machine Learning (AML)

1. Literature review and problem definition.
2. Dataset checks and exploratory data analysis (EDA).
3. RUL label generation and leakage-safe validation design.
4. Feature engineering and feature selection.
5. Train and compare a linear regression baseline and tree-based models.
6. Evaluate with MAE and RMSE; inspect plots and error patterns.
7. Write an IEEE/LaTeX-style report, presentation and viva notes.

### Phase 2 — Deep Learning (DL)

1. Build fixed-length sensor windows within each engine.
2. Train a basic LSTM sequence model.
3. Compare against the strongest Phase 1 model using the same target and evaluation rules.
4. If time permits, train a TCN and perform one or two meaningful ablation studies.
5. An ensemble/hybrid is optional: only keep it if validation results and reasoning show value.

## Evaluation rules

Exact target, split, validation-point and metric settings: `docs/EVALUATION_PROTOCOL.md`.

- MAE: average absolute error in cycles.
- RMSE: square root of average squared error; large mistakes count more heavily.
- Do not compare models evaluated with different targets or data splits.
- Never use official test metrics to select hyperparameters or repeatedly change the model.
- Report exact experimental settings and random seeds where relevant.
- Do not invent final numbers while writing the report. Load numbers from saved results.

## Scope control

The goal is a sound AML baseline and one sound DL comparison—not many half-working models. First make the pipeline reproducible. Add PCA, alternate RUL caps, more architectures or extra C-MAPSS subsets only after the base experiment works.
