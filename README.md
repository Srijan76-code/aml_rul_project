# T034 — Remaining Useful Life (RUL) Prediction

A simple working guide for the five-person project. It explains who owns what, when to do it, how to check the work, and how Claude Code can assist without replacing your understanding.

## Start here

1. Read `docs/PROJECT_CONTEXT.md` to understand the project.
2. Read `docs/WORKFLOW.md` for the order of work.
3. Open your own file in `members/` and follow its checklist.
4. Put this folder in the root of your GitHub project repository.
5. Install/start Claude Code from the repository root so it loads `CLAUDE.md` and its shared context automatically.
6. Ask Claude Code to read your member file and complete the first unchecked task; explicitly request the relevant named subagent if needed.

## Setup (every member, once after cloning)

```bash
git clone https://github.com/Srijan76-code/aml_rul_project.git
cd aml_rul_project
python3 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .                     # makes `import rul_prediction` work in notebooks, tests and the editor
python -m pytest -q                  # all tests should pass
```

- **Data:** the NASA C-MAPSS FD001 files are already in `data/raw/` (`train_FD001.txt`, `test_FD001.txt`, `RUL_FD001.txt`). Treat them as read-only.
- **Loading:** use `rul_prediction.data.load_cmapss_file(path)`; it returns 26 named columns: `engine_id`, `cycle`, `setting_1..3`, `sensor_1..21`. For the test answer key use `load_rul_file(path)`, which returns `engine_id` and `rul` (true RUL at each test engine's last observed cycle); join predictions to it by `engine_id`.
- **Features:** `rul_prediction.features` gives the 14 informative sensors, causal rolling mean/std/slope and the model input list; column meanings and the normalization rule (fit scalers on training engines only) are in `docs/DATA_DICTIONARY.md`.
- **Check you have the right data:** run `notebooks/01_eda.ipynb` from the `notebooks/` folder. The training file should give 20,631 rows × 26 columns and 100 engines.
- In VS Code, select the `.venv` interpreter.

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

These files provide shared context and guardrails; they cannot guarantee that an agent will never make a mistake. Review diffs, run tests, verify metrics, and understand your own code. Only the three FD001 files in `data/raw/` are committed; do not commit other dataset files, processed data, secrets, or model checkpoints to GitHub.
