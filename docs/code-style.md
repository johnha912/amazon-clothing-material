# Code style: pandas method chaining

The hard rules are in `AGENTS.md`. This page shows the full style with an example.

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
- Don't use `inplace=True`, or chained assignment like `df["a"][mask] = x`; under pandas 3 copy-on-write it silently does nothing. Use `.assign` or `df.loc[mask, "a"] = x`.
- Avoid `iterrows` and `apply(axis=1)` when a vectorized method exists.
