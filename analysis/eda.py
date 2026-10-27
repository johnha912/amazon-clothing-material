"""Exploratory data analysis for the Amazon clothing-material project.

Owner: Nguyen Ha (Issue #1, Week 1).

Run from the repo root:

    python analysis/eda.py

Reads `data/sample.parquet` when it exists (built by `src/make_sample.py`,
Rui Song). Until then it falls back to the first 10,000 metadata records in
`data/meta_head.jsonl` and labels them with the shared rule in
`explore_sample.dominant()` — percentage composition strings only, exactly
as AGENTS.md specifies.

Outputs six Plotly charts (interactive HTML + static PNG) in `results/eda/`
and prints the summary numbers quoted in the report's Data section.
"""

# %% Imports and config
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "results" / "eda"
SEED = 42

# %% Load the data (sample.parquet preferred, meta_head.jsonl fallback)

from load_data import load_data

df, DATA_VERSION = load_data()
print(f"Data: {DATA_VERSION}; seed={SEED}")

# %% Derived columns used by several charts

df = (df
      .assign(has_price=lambda d: d.price.notna(),
              log_price=lambda d: np.log10(d.price.where(d.price > 0)),
              price_band=lambda d: pd.cut(
                  d.price, bins=[0, 10, 25, 50, 100, 250, np.inf],
                  labels=["<$10", "$10–25", "$25–50", "$50–100", "$100–250", ">$250"])))

labeled = df.dropna(subset=["material"])

# %% Chart 1: how many listings carry each dominant material

material_counts = (labeled
                   .groupby("material", as_index=False)
                   .size()
                   .rename(columns={"size": "listings"})
                   .sort_values("listings", ascending=True))

fig1 = px.bar(material_counts, x="listings", y="material", orientation="h",
              title="Listings by dominant material (percentage-labeled rows only)",
              labels={"listings": "Listings", "material": "Dominant material"})
fig1.update_layout(template="plotly_white")

# %% Chart 2: average rating by dominant material

rating_by_material = (labeled
                      .groupby("material")
                      .agg(mean_rating=("average_rating", "mean"),
                           listings=("average_rating", "size"))
                      .reset_index()
                      .query("listings >= 30")
                      .sort_values("mean_rating"))

fig2 = px.bar(rating_by_material, x="mean_rating", y="material", orientation="h",
              text="listings",
              title="Average rating by dominant material (materials with ≥30 listings)",
              labels={"mean_rating": "Mean average rating", "material": "Dominant material",
                      "listings": "Listings"})
fig2.update_layout(template="plotly_white", xaxis_range=[3.4, 4.8])
fig2.update_traces(textposition="outside")

# %% Chart 3: average rating by department

rating_by_dept = (df
                  .groupby("department")
                  .agg(mean_rating=("average_rating", "mean"),
                       listings=("average_rating", "size"))
                  .reset_index()
                  .query("listings >= 100")
                  .sort_values("mean_rating"))

fig3 = px.bar(rating_by_dept, x="mean_rating", y="department", orientation="h",
              text="listings",
              title="Average rating by department (departments with ≥100 listings)",
              labels={"mean_rating": "Mean average rating", "department": "Department",
                      "listings": "Listings"})
fig3.update_layout(template="plotly_white", xaxis_range=[3.4, 4.8])
fig3.update_traces(textposition="outside")

# %% Chart 4: price distribution (log scale)

priced = df.dropna(subset=["log_price"])
fig4 = px.histogram(priced, x="log_price", nbins=40,
                    title="Price distribution (log10 of price, priced listings only)",
                    labels={"log_price": "log10(price in USD)"})
fig4.update_layout(template="plotly_white", yaxis_title="Listings")

# %% Chart 5: average rating by price band

rating_by_band = (df
                  .dropna(subset=["price_band"])
                  .groupby("price_band", observed=True)
                  .agg(mean_rating=("average_rating", "mean"),
                       listings=("average_rating", "size"))
                  .reset_index())

fig5 = px.bar(rating_by_band, x="price_band", y="mean_rating", text="listings",
              title="Average rating by price band",
              labels={"mean_rating": "Mean average rating", "price_band": "Price band",
                      "listings": "Listings"})
fig5.update_layout(template="plotly_white", yaxis_range=[3.8, 4.6])
fig5.update_traces(textposition="outside")

# %% Chart 6: missing prices — where they are, and whether they rate differently

missing_by_dept = (df
                   .groupby("department")
                   .agg(missing_share=("has_price", lambda s: 1 - s.mean()),
                        listings=("has_price", "size"))
                   .reset_index()
                   .query("listings >= 100")
                   .sort_values("missing_share", ascending=True))

rating_by_price_known = (df
                         .groupby("has_price")
                         .agg(mean_rating=("average_rating", "mean"),
                              listings=("average_rating", "size"))
                         .reset_index()
                         .assign(group=lambda d: np.where(d.has_price, "Price listed", "Price missing")))

fig6 = make_subplots(rows=1, cols=2,
                     subplot_titles=("Share of listings with no price, by department",
                                     "Mean rating: price listed vs missing"))
fig6.add_trace(go.Bar(x=missing_by_dept.missing_share, y=missing_by_dept.department,
                      orientation="h", showlegend=False), row=1, col=1)
fig6.add_trace(go.Bar(x=rating_by_price_known.group, y=rating_by_price_known.mean_rating,
                      text=rating_by_price_known.listings, textposition="outside",
                      showlegend=False), row=1, col=2)
fig6.update_layout(template="plotly_white", title_text="Missing-price analysis")
fig6.update_xaxes(title_text="Share with no price", row=1, col=1)
fig6.update_yaxes(title_text="Mean average rating", range=[3.8, 4.6], row=1, col=2)

# %% Save every chart as HTML + PNG and print the headline numbers

OUT_DIR.mkdir(parents=True, exist_ok=True)
charts = {"01_material_counts": fig1, "02_rating_by_material": fig2,
          "03_rating_by_department": fig3, "04_price_distribution": fig4,
          "05_rating_by_price_band": fig5, "06_missing_price": fig6}
for name, fig in charts.items():
    fig.write_html(OUT_DIR / f"{name}.html", include_plotlyjs="cdn")
    fig.write_image(OUT_DIR / f"{name}.png", width=1100, height=650, scale=1.5)
    print(f"saved results/eda/{name}.html/.png")

summary = {
    "rows": len(df),
    "labeled_share": round(float(df.material.notna().mean()), 4),
    "missing_price_share": round(float(1 - df.has_price.mean()), 4),
    "mean_rating": round(float(df.average_rating.mean()), 3),
    "top_material": material_counts.iloc[-1].material,
    "departments": int(df.department.nunique()),
    "data_version": DATA_VERSION,
    "seed": SEED,
    "command": "python analysis/eda.py",
}
print(summary)
