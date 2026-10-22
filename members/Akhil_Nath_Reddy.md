# Akhil Nath Reddy — advanced experiment, ablation and technical review

**Main job:** check with real numbers that the project's conclusions are sound (Phase 1), then contribute one advanced model (Phase 2). Do not become the person who rewrites everyone else's work.

## When to start

Phase 1 review work can start now. Claim verification starts once Ashrith's experiment runner is merged. Implement the Phase 2 advanced model only after the AML baseline and first LSTM run.

## Step-by-step tasks

### Phase 1 — claim verification and human review

**Your role in one line:** make sure every Phase 1 conclusion the team writes is backed by real numbers, not by opinion.

You have two contributions: your own code and results (claim verification), and a visible review record (human review).

#### Contribution 1 — verify our claims with numbers

**The problem.** All models are compared on 100 validation points from only **20 engines**. With so few engines, a model can win by luck, because it happened to get "easy" engines. The report currently says "small differences may be noise", but nobody has measured it (see the limitations in `docs/EVALUATION_PROTOCOL.md`).

**The idea.** Use an **engine-level bootstrap**. Build many pretend validation sets by drawing 20 engines at random from the 20, with repeats. Score both models on each pretend set and record the difference, then look at the middle 95% of those differences. If that range stays on one side of zero, the difference is real. If it crosses zero, it could be noise and should be treated as a tie.

**Claims to check** (current validation RMSE, from `reports/results/`):

| # | Pair | Claim it tests |
|---|---|---|
| 1 | Random Forest (40.17) vs linear Elastic Net v2 (41.77) | Trees beat the linear baseline. Check MAE as well (25.03 vs 31.66). |
| 2 | Gradient boosting (40.17) vs linear Elastic Net v2 (41.77) | Same, for the other tree model |
| 3 | Random Forest vs Gradient boosting (40.17 vs 40.17) | Which tree model is best |
| 4 | RF with rolling features (40.17) vs without (37.13) | The rolling-features decision |
| 5 | GB with rolling features (40.17) vs without (36.44) | Same, for GB |
| 6 | Linear v2 Elastic Net (41.77) vs linear v1 OLS on raw sensors (42.92) | Aryan's features helped the linear model |
| 7 | Same model with and without the 125 cap, overall and late band | The cap rule (numbers from Ashrith's experiment) |

**Main rules:**

- **Resample engines, not points.** The 5 points of one engine are related. Resampling single points would pretend we have 100 independent samples and make the intervals too narrow.
- **Paired.** In every round, both models are scored on the **same** resampled engines.
- **Validation only.** Use the 100 pinned points (`docs/fd001_validation_cutoffs.csv`). Never load `test_FD001.txt` or `RUL_FD001.txt`.
- **Reproduce, don't retune.** Use predictions from the published settings. Ashrith's `run_cap_grid` returns most of them. Pair 6 needs one extra run: v1 OLS on the 24 raw settings and sensors. Before trusting any interval, check that your predictions reproduce the published RMSEs above.
- **Fix your choices before looking at results:** number of resamples (at least 1,000), seed 42, a 95% percentile interval, and "real = the interval does not include 0". Write them in the notebook before the results cell.
- **Reusable code with tests.** Put the bootstrap function in `src/rul_prediction/` and test it. For example: two identical models give a difference of exactly 0; a model that is better on every engine gives an interval entirely on one side; resampling keeps all 5 points of an engine together; the same seed gives the same result.
- **Careful wording.** Use "real difference" or "could be noise", never "proves". The bootstrap cannot fix the fact that we tuned on one fixed split; say so as a limitation.
- **Results are written by code:** a results CSV, an experiment log and one figure (each difference with its interval), plus a short Findings section.
- **When a claim fails,** tell its owner and suggest new wording. Do not edit other members' report sections yourself.

**Order:** the review work (below) can start now. The bootstrap needs Ashrith's experiment runner (his spec, Part B) merged first. If that is delayed, rebuild the predictions yourself with the same published settings.

#### Contribution 2 — human review

The `leakage-reviewer` and `experiment-reviewer` agents are automated checks. Your review is the human sign-off on top of them, and it must be visible on GitHub.

- [ ] **Review Ashrith's PR** (early/late band metric and RUL-cap experiment) with GitHub's "Review changes", and approve or request changes. Check that:
  - the cap is applied to training targets only and grading uses true RUL;
  - no test file is loaded;
  - the model settings match his spec and the C5 sanity checks pass;
  - `passes_rule` is read correctly;
  - the Findings claim no more than the tables show.
- [ ] Review the merged Phase 1 code on `main`: Abhayanth's target, split and metrics; Aryan's features for future-value leakage; that Srijan's models use the shared features and validation IDs.
- [ ] Open a GitHub issue for each real problem, with evidence, and link it to the fix. Suggest small fixes; don't take over a module.
- [ ] Write `reports/review/phase1_review.md`: what you checked, what you found, what was fixed, and the claim-verification table with each claim marked "holds" or "reword".

### Phase 2 — advanced comparison

- [ ] Once the LSTM works, implement a simple Temporal Convolutional Network (TCN) if time/computation allows.
- [ ] Compare it to LSTM and the best AML baseline under the same data/metrics.
- [ ] Run one or two meaningful ablations, e.g. sequence length or engineered features, changing one factor at a time.
- [ ] If the team considers an AML+DL ensemble, tune the ensemble weight on validation data only and retain it only if the result justifies the added complexity.
- [ ] Record failures and limitations, not just the best score.
- [ ] Perform final reproducibility and code review; make sure each member can explain their own contribution.

## Expected deliverables

- `src/rul_prediction/models/tcn.py` and `notebooks/05_tcn_ablation.ipynb` if feasible
- Phase 1: bootstrap function and its tests in `src/rul_prediction/`, `notebooks/03_claim_verification.ipynb`, results CSV, log and figure
- Phase 1: `reports/review/phase1_review.md`, GitHub PR reviews (including Ashrith's PR) and issues
- ablation table and technical review checklist
- final technical validation notes

## How to approach it

1. Get the simple pipeline working first.
2. Form one specific question before each experiment.
3. Change one factor at a time and record the configuration.
4. Use validation results for choices; preserve the official test for final comparison.
5. Use `experiment-reviewer` and `leakage-reviewer` for independent checks.

## Ask Claude Code to help

Phase 1 (claim verification):

```text
Read CLAUDE.md, AGENTS.md, docs/EVALUATION_PROTOCOL.md and members/Akhil_Nath_Reddy.md.
First teach me, with a tiny made-up example, what a paired engine-level bootstrap is and why
we resample engines instead of points. Then help me design the bootstrap function and its
tests before writing the notebook. Use validation predictions only, never load the test files,
and do not retune or change other members' code. Request leakage-reviewer and
experiment-reviewer audits before I open the PR.
```

Phase 2 (TCN):

```text
Read CLAUDE.md, AGENTS.md, docs/PROJECT_CONTEXT.md and members/Akhil_Nath_Reddy.md.
First review the existing baselines and explain one useful advanced experiment. Do not rewrite
other members' modules. Implement the TCN or the next agreed task only after the LSTM baseline
is runnable. Add a clear comparison and request experiment-reviewer and leakage-reviewer audits.
```

## Be ready to explain

- Why the advanced model is worth testing.
- What an ablation study proves and what it cannot prove.
- Why a hybrid model needs validation evidence rather than just averaging two predictions.
- How the final comparison avoids using test results for model selection.
- What a human review catches that automated reviewer agents and unit tests miss.
- What a bootstrap is, why you resample engines rather than points, and why the comparison is paired.
- What it means when a 95% interval crosses zero, and what the bootstrap cannot fix.
