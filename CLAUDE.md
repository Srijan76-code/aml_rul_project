# Claude Code project instructions

Read and follow the shared rules and project details in:

- `@AGENTS.md`
- `@agent.md`
- `@docs/PROJECT_CONTEXT.md`
- `@docs/WORKFLOW.md`
- `@docs/AML_CONCEPT_MAP.md`

Before doing a member's task, also read the relevant `members/<name>.md` file. If the user's request does not specify a member, ask which member/role they are working as rather than editing several people's work at once.

## How to work on every task

1. Inspect the current repository, `git status`, relevant files, and existing tests before editing.
2. Tell the student in plain language: **what this task means, why it matters, what you will change, and how they can verify it**.
3. Break work into a small, reviewable step. Do not implement the whole project in one pass.
4. Follow `agent.md` and use the appropriate shared subagent for meaningful tasks:
   - raw dataset / EDA questions: `data-auditor`
   - target, scaling, split or sequence questions: `leakage-reviewer`
   - model comparison / metrics / conclusions: `experiment-reviewer`
   - test writing or test failure: `test-helper`
   - explaining concepts or updating clear docs: `learning-documentation-helper`
5. For data, target, preprocessing, split, metrics or model-comparison changes, request an independent review after implementation. Implement the assigned task, not unrelated tasks. Ask before changing a shared interface, target definition, split policy, metrics, or another member's owned files.
6. Run relevant tests/checks. Never claim a test passed unless it was actually run and passed.
7. Finish with: files changed, what the student learned, commands/tests run, results, limitations, and the next checklist item.

## Required style

- Use concepts and terminology taught in AML where suitable.
- Prefer readable Python, small functions, meaningful variable names, and comments explaining **why**.
- Do not invent dataset values, experiment scores, paper findings, or results.
- Do not tune on the official test set.
- Use subagents for independent review, not to create overlapping implementations of the same module.
- Never use `--dangerously-skip-permissions` or bypass permission prompts.
