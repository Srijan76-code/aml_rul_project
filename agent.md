# Agent orchestration playbook

This file explains how the main Claude Code session should use the shared subagents. `CLAUDE.md` imports this file so the rules are available in each project session.

## Every time someone starts a task

1. Read `AGENTS.md`, `docs/PROJECT_CONTEXT.md`, `docs/WORKFLOW.md`, `docs/AML_CONCEPT_MAP.md`, and that member's task file.
2. Inspect the repository and `git status`; do not assume earlier chat history is available.
3. Identify the first incomplete task and state, in simple language, what it means and why it is needed.
4. Inspect existing code before proposing changes. Ask if a target definition, split, shared function interface or ownership is unclear.
5. Use the **relevant** subagent(s) below. Do not run every agent on every tiny change.
6. Implement only the assigned task and run the relevant checks.
7. Use a reviewer after changes to data processing, target logic, splitting, metrics or model comparison.
8. Summarize changes, explain the AML concept, report the exact checks actually run, and say the next step.

## Which subagent should be used?

| Situation | Subagent |
|---|---|
| Raw data, columns, EDA, sensor checks | `data-auditor` |
| RUL labels, rolling features, scaling, splits, sequence windows, test alignment | `leakage-reviewer` |
| Model scores, MAE/RMSE, fair comparisons, plots, report claims | `experiment-reviewer` |
| Need tests for a function, or a test is failing | `test-helper` |
| Need a plain-language lesson, viva notes or concise documentation | `learning-documentation-helper` |

## How each subagent must work

- Give it the relevant project files, the question to answer and the expected output.
- Review agents are independent reviewers; they must report findings, not rewrite the implementation.
- `test-helper` may edit only the assigned tests or specifically approved function.
- Do not give two agents write access to the same module at the same time.
- Resolve disagreements using code/data evidence, not by accepting the most confident answer.
- If the current Claude Code session cannot invoke a subagent, say so and provide the agent prompt for a manual review. Never claim that an agent ran when it did not.

## What the student should see

For each meaningful task, the response should include:

1. **What we're doing** — one or two plain sentences.
2. **Why it matters** — connected to regression, feature engineering, regularization, metrics or sequence modeling.
3. **Changes made** — files and main functions.
4. **Checks** — exact tests/commands and pass/fail; never guess.
5. **Understand it** — short explanation plus a viva question.
6. **Next step** — point to the next unchecked item in that member's task file.

The purpose is assisted learning: Claude should implement bounded work, explain it, and ask the student to verify key outputs. It should not silently complete or rewrite the whole project.
