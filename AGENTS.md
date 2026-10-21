# Rules for AI assistants

Read this before writing code for this repo. If your AI tool does not load `AGENTS.md` on its own, attach or paste this file at the start of the chat.

## Project

Predict a clothing product's dominant material (Task A: classification, macro-F1) and its average rating (Task B: regression, RMSE and MAE) from Amazon Reviews 2023 metadata. Stack: Python 3.12+, pandas 3, scikit-learn, XGBoost, PyTorch with transformers, and Plotly. Versions are pinned in `requirements.txt`; don't add a library without asking the team.

## Status

- Pilot (first 10k records): TF-IDF baseline macro-F1 ≈ 0.28.
- Sample milestones: 50k products by Wed Oct 28, ~250k products by Sun Nov 1.
- Report and presentation due Sun Dec 6; presentations streamed Thu Dec 17.
- Open question: do shoes stay in scope? (sole vs. upper makes their labels noisy). Yanglin and Nguyen decide.

Update this section when a milestone lands or a question is settled.

## Ownership

Edit only the files you own, and run `git pull` before you start. `ai-prompts.md` is shared.

| Member | Owns |
|---|---|
| Nguyen Ha | `analysis/material_tfidf.py`, `analysis/rating.py`, `analysis/eda.py`, `report/` |
| Yanglin Hu | `analysis/material_bert.py` |
| Rui Song | `src/`, `app/`, `report/sections/related_work.md` |

## Data

- Use `data/sample.parquet` (built by `src/make_sample.py`) with columns `parent_asin, title, text, price, department, average_rating, rating_number, material, split`. Only until `sample.parquet` exists (target Oct 28), read `data/meta_head.jsonl` with `pd.read_json(path, lines=True)`.
- Train on `split == "train"`, report on `split == "test"`, and pass `random_state=42` everywhere.
- Never commit `data/` or `models/`.

## Labels

- The labeling rule lives in one place: `dominant()`, now in `explore_sample.py` and moving to `src/labels.py`. Don't re-implement it. Change it only there, together with its tests.
- A label is the highest-percentage known fiber in a composition string such as "90% Polyester, 10% Spandex". On a tie, the first fiber listed wins. Rare fibers are merged into the class list defined in `src/labels.py`.
- Only about 46% of listings have a composition string, so only about 46% get a label. This is expected. Rows without one have `material = NaN` and are dropped for Task A. Never fill in, guess, or model labels for the rest.

## Leakage

- Task A: `text` must not contain composition strings such as "90% polyester", because the label comes from them.
- Task A: don't use `average_rating` or `rating_number` as features. Material is predicted from the listing alone.
- Task B: `average_rating` is the target. `rating_number` is not a feature; it may only be used to filter or weight rows (for example, dropping items with very few ratings).
- Task B: no feature may be derived from the item's own ratings or reviews (review text, helpful votes, review counts, and so on). Allowed inputs are listing fields and `material`.

## Evaluation and numbers

- Compute metrics only through the shared entry point `src/evaluate.py`. Until it exists, use `sklearn.metrics`: `f1_score(average="macro")`, `root_mean_squared_error`, and `mean_absolute_error`.
- Every number written to `results/`, `docs/`, or `report/` must state the command that produced it, the seed, and the data version (the `sample.parquet` row count plus the commit of `src/make_sample.py`). The report merges three people's results, and numbers without this information can't be checked against each other.

## Feature work: three agent roles

New features go through three roles, one at a time. Each role hands off through files, not through chat.

1. **Analyst** proposes the features in `docs/feature-plan.md`: the idea, the source columns, why it should help, and a leakage check against the rules above. It writes no code.
2. **A human team member** approves or edits the plan before anything is built.
3. **Engineer** implements only the approved items in the owner's file, and records each new column (name, type, formula, source) in `docs/feature-schema.md`.
4. **Auditor** is read-only. It reruns the evaluation itself, checks the code against the plan, the schema, and the leakage rules, and appends a verdict to the plan entry: `APPROVED` or `CHANGES REQUESTED` with reasons.

## Code style

Full guide with an example: `docs/code-style.md`. Hard rules:

- Write each transformation as one method chain in parentheses, one method per line.
- Put a reusable step in a function `def step(df) -> pd.DataFrame` and call it with `.pipe(step)`.
- Never use `inplace=True`.
- Never use chained assignment like `df["a"][mask] = x`; under pandas 3 it silently does nothing.

## Workflow

- Write analysis code as `.py` files with `# %%` cells; no `.ipynb` files.
- Make charts with Plotly.
- Write report text in `report/` as paragraphs, not bullet lists. The grading rubric takes points off for lists.

## Prompt log

The report must include an "AI Prompts Used" section that lists every prompt. This log is where that section comes from.

- After every prompt you receive in this repo, append one row to the end of the table in `ai-prompts.md`:
  `| YYYY-MM-DD | <member name> | <tool> | <the prompt as one short English sentence> | <file or task it was for> |`
  Use the member's name as written in the README Authors table.
- Commit the log together with your next work commit, or once at the end of the session. Don't make a separate commit and push for each prompt.
- Keep the summary short and neutral; never include secrets, API keys, or raw data.
- If your tool can't edit files (for example, a chat in the browser), the member adds the row by hand.
