# Team workflow — follow in order

## Step 0 — Set up together

- One member creates a private GitHub repository and invites the other four.
- Add this project pack at the repository root.
- The raw FD001 files are committed in `data/raw/` (see `README.md` → Setup). Do not edit them or add other subsets/processed data without team agreement.
- Agree on target, split, feature, sequence and metric interfaces before all five code in parallel.
- Each person opens their own `members/` file.

## Step 1 — Validate data and target

Aryan audits the raw data and documents columns. Abhayanth implements target generation and the proposed validation split. Srijan, Ashrith and Akhil review the definitions before model development. Run simple tests by hand on a tiny example.

## Step 2 — Phase 1 AML

Aryan builds preprocessing and engineered features. Abhayanth implements a linear baseline and shared metrics. Srijan implements tree models. Compare results using the same split and metric function. Fix any leakage or alignment bugs before tuning.

## Step 3 — Phase 2 DL

Ashrith prepares sequence windows and trains the first LSTM. Akhil reviews the protocol and then builds a TCN/ablation only after the LSTM baseline works. Keep the best AML model as the comparison point.

## Step 4 — Final analysis and report

Abhayanth consolidates evaluation. Srijan interprets tree model results. Aryan documents data quality and features. Ashrith documents LSTM design and training. Akhil reviews experiments and leads the DL/ablation technical check. All members contribute to the report and practice viva.

## Git routine

1. Pull latest main before starting.
2. Create a branch such as `aryan/feature-engineering` or `srijan/tree-models`.
3. Work only on owned files unless agreed otherwise.
4. Run relevant tests, inspect `git diff`, commit with a meaningful message.
5. Open a pull request; request one teammate review and, when useful, a read-only Claude subagent review.
6. Merge only after the owner can explain the code and checks pass. GitHub Actions runs `pytest` on every pull request (`.github/workflows/tests.yml`); do not merge while it is red.

## Definition of ready for the next phase

Do not start serious DL tuning until the AML target, validation design, metric code and baseline are reproducible. Do not start a hybrid until the individual AML and DL models work and show complementary errors.
