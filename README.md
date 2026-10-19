# T034 — Remaining Useful Life (RUL) Prediction

A simple working guide for the five-person project. It explains who owns what, when to do it, how to check the work, and how Claude Code can assist without replacing your understanding.

## Start here

1. Read `docs/PROJECT_CONTEXT.md` to understand the project.
2. Read `docs/WORKFLOW.md` for the order of work.
3. Open your own file in `members/` and follow its checklist.
4. Put this folder in the root of your GitHub project repository.
5. Install/start Claude Code from the repository root so it loads `CLAUDE.md` and its shared context automatically.
6. Ask Claude Code to read your member file and complete the first unchecked task; explicitly request the relevant named subagent if needed.

## Team work ownership

| Member | Main ownership | Main deliverables |
|---|---|---|
| **Srijan Patel** | Random Forest / Gradient Boosting and model comparison | `03_tree_models.ipynb`, model results and feature importance |
| **Aryan Patel** | Dataset EDA, preprocessing and feature engineering | `01_eda.ipynb`, `02_features.ipynb`, reusable feature functions |
| **Abhayanth** | RUL labels, engine-based validation, metrics and linear baseline | target/split/metric functions, `03_linear_baseline.ipynb` |
| **R Ashrith** | Sequence preparation and LSTM | sequence dataset code, `04_lstm.ipynb`, training plots |
| **Akhil Nath Reddy** | TCN/advanced experiment, ablation and technical review | `05_tcn_ablation.ipynb`, review checklist and final comparison |

**Srijan and Abhayanth's earlier roles have been swapped.** Srijan now owns tree models; Abhayanth owns the evaluation setup and linear baseline.

## Folder contents

- `CLAUDE.md` — Claude Code instructions; imports shared project rules.
- `AGENTS.md` — shared rules for coding assistants and human contributors.
- `agent.md` — how the main agent should select and use subagents for each task.
- `members/` — one step-by-step file for each member.
- `docs/PROJECT_CONTEXT.md` — target, dataset, evaluation and scope.
- `docs/WORKFLOW.md` — project sequence and Git collaboration.
- `docs/AML_CONCEPT_MAP.md` — maps your taught AML topics to this project.
- `docs/DEFINITION_OF_DONE.md` — checks before marking work complete.
- `.claude/agents/` — reusable Claude Code subagents.
- `templates/` — prompts and experiment log template.

## Important

These files provide shared context and guardrails; they cannot guarantee that an agent will never make a mistake. Review diffs, run tests, verify metrics, and understand your own code. Do not commit dataset files, secrets, or model checkpoints to GitHub.
