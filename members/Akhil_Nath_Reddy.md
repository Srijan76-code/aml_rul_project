# Akhil Nath Reddy — advanced experiment, ablation and technical review

**Main job:** contribute one advanced model and ensure the project's conclusions are technically sound. Do not become the person who rewrites everyone else's work.

## When to start

Review the shared target/split and help the team agree on a fair evaluation design from the beginning. Implement the advanced model only after the AML baseline and first LSTM run.

## Step-by-step tasks

### Phase 1 — human review

You are the team's **human reviewer** for Phase 1. The `leakage-reviewer` and `experiment-reviewer` agents are automated checks; your review is the human sign-off on top of them. Most Phase 1 code is already merged, so review it on `main` and post findings as GitHub issues or PR comments.

- [ ] Read `docs/PROJECT_CONTEXT.md`, `docs/AML_CONCEPT_MAP.md`, and the other task files when reviewing their components.
- [ ] Review Abhayanth's label, split and metric setup.
- [ ] Review Aryan's feature code for future-value leakage.
- [ ] Check that Srijan's models use the shared features and validation IDs.
- [ ] Explain risks with evidence; suggest small fixes instead of taking over a module.
- [ ] Review Ashrith's RUL-cap and feature-set experiment PR before it is merged: is the cap applied to training targets only, is the test set untouched, and are the settings reused rather than retuned?
- [ ] Approve or request changes on new Phase 1 PRs on GitHub, so the review is visible on record.
- [ ] Write the findings into `reports/review/phase1_review.md`: what you checked, what you found, and what was fixed.

### Phase 2 — advanced comparison

- [ ] Once the LSTM works, implement a simple Temporal Convolutional Network (TCN) if time/computation allows.
- [ ] Compare it to LSTM and the best AML baseline under the same data/metrics.
- [ ] Run one or two meaningful ablations, e.g. sequence length or engineered features, changing one factor at a time.
- [ ] If the team considers an AML+DL ensemble, tune the ensemble weight on validation data only and retain it only if the result justifies the added complexity.
- [ ] Record failures and limitations, not just the best score.
- [ ] Perform final reproducibility and code review; make sure each member can explain their own contribution.

## Expected deliverables

- `src/rul_prediction/models/tcn.py` and `notebooks/05_tcn_ablation.ipynb` if feasible
- Phase 1: `reports/review/phase1_review.md` and GitHub PR reviews
- ablation table and technical review checklist
- final technical validation notes

## How to approach it

1. Get the simple pipeline working first.
2. Form one specific question before each experiment.
3. Change one factor at a time and record the configuration.
4. Use validation results for choices; preserve the official test for final comparison.
5. Use `experiment-reviewer` and `leakage-reviewer` for independent checks.

## Ask Claude Code to help

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
