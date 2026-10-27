"""Task B: predict a product's average rating from its listing.

Owner: Nguyen Ha (Issue #1, Week 1).

Run from the repo root:

    python analysis/rating.py

Trains a mean baseline, a random forest, and XGBoost on `split == "train"`,
reports RMSE and MAE on `split == "test"`, and writes `results/rating.csv`.

Data: `data/sample.parquet` when it exists (built by `src/make_sample.py`,
Rui Song); until then the first 10,000 records of `data/meta_head.jsonl`.

Leakage (AGENTS.md): `average_rating` is the target. No feature is derived
from the item's own ratings or reviews. `rating_number` is not a feature —
it is only used as a sample weight (log1p), which AGENTS.md allows. Inputs
are listing fields (price, department, text lengths) plus the Task A
`material` label (NaN becomes its own "unknown" category).
"""

# %% Imports and config
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
SEED = 42

# %% Load the data (sample.parquet preferred, meta_head.jsonl fallback)

from load_data import load_data

df, DATA_VERSION = load_data()
print(f"Data: {DATA_VERSION}; seed={SEED}")

# %% Features from listing fields only (plus the material label)

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    return (df
            .assign(has_price=lambda d: d.price.notna().astype(int),
                    log_price=lambda d: np.log1p(d.price),
                    title_words=lambda d: d.title.fillna("").str.split().str.len(),
                    text_words=lambda d: d.text.fillna("").str.split().str.len(),
                    material=lambda d: d.material.fillna("unknown"),
                    department=lambda d: d.department.fillna("Unknown")))


task_b = (df
          .dropna(subset=["average_rating"])
          .pipe(add_features))

train = task_b.query("split == 'train'")
test = task_b.query("split == 'test'")
print(f"Task B rows: train={len(train):,}, test={len(test):,}")

FEATURES = ["log_price", "has_price", "title_words", "text_words",
            "department", "material"]
NUMERIC = ["log_price", "has_price", "title_words", "text_words"]
CATEGORICAL = ["department", "material"]

def make_preprocess() -> ColumnTransformer:
    """A fresh preprocessor per pipeline — sharing one fitted instance
    across pipelines would silently couple the models."""
    return ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ])

# rating_number appears ONLY here, as a weight — never as a feature.
train_weight = np.log1p(train.rating_number.fillna(0))

# %% Train the three models and score them on the test split

models = {
    "mean_baseline": Pipeline([("model", DummyRegressor(strategy="mean"))]),
    "random_forest": Pipeline([("prep", make_preprocess()),
                               ("model", RandomForestRegressor(
                                   n_estimators=300, random_state=SEED, n_jobs=-1))]),
    "xgboost": Pipeline([("prep", make_preprocess()),
                         ("model", XGBRegressor(
                             n_estimators=400, learning_rate=0.06, max_depth=6,
                             subsample=0.9, colsample_bytree=0.9,
                             random_state=SEED, n_jobs=-1))]),
}

rows = []
preds = {}
for name, model in models.items():
    # Same weights for every model, baseline included, so the comparison is fair.
    model.fit(train[FEATURES], train.average_rating,
              model__sample_weight=train_weight)
    pred = model.predict(test[FEATURES])
    preds[name] = pred
    rows.append({"model": name,
                 "rmse": round(root_mean_squared_error(test.average_rating, pred), 4),
                 "mae": round(mean_absolute_error(test.average_rating, pred), 4),
                 "n_train": len(train), "n_test": len(test),
                 "command": "python analysis/rating.py",
                 "seed": SEED, "data_version": DATA_VERSION})
    print(f"{name}: RMSE={rows[-1]['rmse']}, MAE={rows[-1]['mae']}")

# %% Save metrics

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
(pd.DataFrame(rows)
 .to_csv(RESULTS_DIR / "rating.csv", index=False))
print("\nsaved results/rating.csv")

# %% Residual analysis for the model note (docs/model-notes/rating.md)
# XGBoost residuals on the test split: error by material group, the largest
# misses as citable examples, and a predicted-vs-residual plot.

import plotly.express as px

resid = (test
         .assign(predicted=preds["xgboost"],
                 residual=lambda d: d.average_rating - d.predicted))

by_material = (resid
               .groupby("material")
               .agg(n=("residual", "size"),
                    rmse=("residual", lambda r: float(np.sqrt((r ** 2).mean()))),
                    mae=("residual", lambda r: float(r.abs().mean())),
                    bias=("residual", "mean"))
               .sort_values("rmse", ascending=False)
               .round(4))
by_material.to_csv(RESULTS_DIR / "rating_residuals_by_material.csv")

worst = (resid
         .assign(abs_residual=lambda d: d.residual.abs())
         .nlargest(200, "abs_residual")
         .loc[:, ["parent_asin", "title", "department", "material", "price",
                  "average_rating", "predicted", "residual"]])
worst.to_csv(RESULTS_DIR / "rating_error_examples.csv", index=False)

fig = px.scatter(resid, x="predicted", y="residual", opacity=0.35,
                 labels={"predicted": "predicted rating",
                         "residual": "actual − predicted"},
                 title="XGBoost rating residuals (test split)")
fig.add_hline(y=0)
fig.write_html(RESULTS_DIR / "rating_residual_plot.html", include_plotlyjs="cdn")
fig.write_image(RESULTS_DIR / "rating_residual_plot.png", width=1000, height=600, scale=2)

print("\nworst material groups by RMSE:")
print(by_material.head(6).to_string())
print("saved results/rating_residuals_by_material.csv, "
      "rating_error_examples.csv, rating_residual_plot.html")
