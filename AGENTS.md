# Rules for AI assistants

Read this before writing code for this repo. If your AI tool does not load `AGENTS.md` on its own, attach or paste this file at the start of the chat.

## Project

Predict a clothing product's dominant material (classification, macro-F1) and its average rating (regression, RMSE and MAE) from Amazon Reviews 2023 metadata. Stack: Python 3.12+, pandas 3, scikit-learn, XGBoost, PyTorch with transformers, and Plotly. Versions are pinned in `requirements.txt`; don't add a library without asking the team.

## Data

- Use `data/sample.parquet` (built by `src/make_sample.py`) with columns `parent_asin, title, text, price, department, average_rating, rating_number, material, split`. Until it exists, read `data/meta_head.jsonl` with `pd.read_json(path, lines=True)`.
- `text` must not contain composition strings such as "90% polyester", because the label comes from them.
- Train on `split == "train"`, report on `split == "test"`, and pass `random_state=42` everywhere.
- Never commit `data/` or `models/`.

## Code style: pandas method chaining

Write each transformation as one chain in parentheses, one method per line:

```python
df = (pd.read_parquet("data/sample.parquet")
      .query("split == 'train'")
      .dropna(subset=["material"])
      .assign(has_price=lambda d: d.price.notna(),
              log_price=lambda d: np.log1p(d.price))
      .pipe(add_text_length))
```

- A reusable step is a function `def step(df) -> pd.DataFrame` called with `.pipe(step)`.
- Add columns with `.assign(col=lambda d: ...)`, using `d` instead of the outer variable name.
- Filter with `.query(...)` or `.loc[lambda d: ...]`.
- Extend the chain instead of creating `df2`, `df_clean`, and similar intermediate variables.
- Don't use `inplace=True` or chained assignment like `df["a"][mask] = x` (it does nothing under pandas 3 copy-on-write), and avoid `iterrows` or `apply(axis=1)` when a vectorized method exists.
- To inspect a chain midway, add `.pipe(lambda d: print(d.shape) or d)` and remove it before committing.

## Workflow

- Write analysis code as `.py` files with `# %%` cells; no `.ipynb` files.
- Edit only the files you own (see issue #1), and run `git pull` before you start. `ai-prompts.md` is shared.
- Make charts with Plotly.

## Prompt log

After every prompt you receive in this repo, do this before you finish your reply:

1. Append one row to the end of the table in `ai-prompts.md`:
   `| YYYY-MM-DD | <member name> | <tool> | <the prompt as one short English sentence> | <file or task it was for> |`
   Use the member's name as written in the README Authors table.
2. Commit only that file and push:

   ```bash
   git add ai-prompts.md
   git commit -m "Log AI prompt"
   git pull --rebase --autostash
   git push
   ```

- Keep the summary short and neutral; never include secrets, API keys, or raw data.
- If your tool can't edit files or run git (for example, a chat in the browser), the member adds the row by hand.
