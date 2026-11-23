"""Task B: predict a product's average rating from its listing.

Owner: Nguyen Ha (Issue #2, Week 2: the 250k-sample rerun; Week 1 pilot
written for Issue #1).

Run from the repo root:

    python analysis/rating.py

Trains a mean baseline, a random forest, and XGBoost on `split == "train"`,
reports RMSE and MAE on `split == "test"` through the shared evaluator
`src/evaluate.py`, and writes `results/rating.csv`. Confidence intervals
are still pending: they will be computed from the saved test-set
predictions once the shared bootstrap helper lands in `src/evaluate.py`
(Rui Song's open Issue #2 task), so no retraining will be needed.
Predictions are merged back onto the test rows by `parent_asin` and
the merge is asserted one-to-one before anything is scored, and the per-row
predictions are saved to `results/rating_predictions.csv`.

Data: `data/sample.parquet` when it exists (built by `src/make_sample.py`,
Rui Song) — for Issue #2 that is the 249,762-row sample; until then the
first 10,000 records of `data/meta_head.jsonl`.

Leakage (AGENTS.md): `average_rating` is the target. No feature is derived
from the item's own ratings or reviews. `rating_number` is not a feature —
it is only used as a sample weight (log1p), which AGENTS.md allows. Inputs
are listing fields (price, department, text lengths) plus the Task A
`material` label (NaN becomes its own "unknown" category).
"""

# %% Imports and config
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.evaluate import rating_scores  # noqa: E402  shared metrics entry point (AGENTS.md)

DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
SEED = 42

# %% Load the data (sample.parquet preferred, meta_head.jsonl fallback)

from load_data import load_data, write_run_json

df, DATA_VERSION = load_data()
print(f"Data: {DATA_VERSION}; seed={SEED}")

# %% Helper: id-aligned predictions

def attach_predictions(test: pd.DataFrame, predicted) -> pd.DataFrame:
    """Merge one model's predictions back onto the test rows by parent_asin.

    Scoring reads the merged frame, never the raw prediction array, and the
    asserts fail loudly if the merge is not an exact one-to-one alignment
    (the team's merge-on-id rule).
    """
    preds = (test
             .loc[:, ["parent_asin"]]
             .assign(predicted=predicted))
    merged = test.merge(preds, on="parent_asin", how="left",
                        validate="one_to_one")
    assert len(merged) == len(test), "prediction merge changed the test row count"
    assert merged.parent_asin.is_unique, "test parent_asin values are not unique"
    assert (merged.parent_asin.to_numpy() == test.parent_asin.to_numpy()).all(), \
        "prediction merge reordered the test rows"
    assert merged.predicted.notna().all(), "some test rows got no prediction"
    return merged


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
scored = {}
for name, model in models.items():
    # Same weights for every model, baseline included, so the comparison is fair.
    model.fit(train[FEATURES], train.average_rating,
              model__sample_weight=train_weight)
    frame = attach_predictions(test, model.predict(test[FEATURES]))
    scored[name] = frame
    scores = rating_scores(frame.average_rating, frame.predicted)
    rows.append({"model": name,
                 "rmse": round(scores["rmse"], 4),
                 "mae": round(scores["mae"], 4),
                 "n_train": len(train), "n_test": len(test),
                 "command": "python analysis/rating.py",
                 "seed": SEED, "data_version": DATA_VERSION})
    print(f"{name}: RMSE={rows[-1]['rmse']}, MAE={rows[-1]['mae']}")

# %% Save metrics and per-row predictions

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
(pd.DataFrame(rows)
 .to_csv(RESULTS_DIR / "rating.csv", index=False))

# attach_predictions asserted each scored frame is in test order, so the
# positional .to_numpy() columns below are aligned by parent_asin.
predictions = (test
               .loc[:, ["parent_asin", "average_rating"]]
               .assign(mean_baseline=scored["mean_baseline"].predicted.to_numpy(),
                       random_forest=scored["random_forest"].predicted.to_numpy(),
                       xgboost=scored["xgboost"].predicted.to_numpy()))
predictions.to_csv(RESULTS_DIR / "rating_predictions.csv", index=False)
run_json = write_run_json(
    Path(__file__), DATA_VERSION, ["numpy", "pandas", "scikit-learn", "xgboost"],
    n_train=len(train), n_test=len(test), features=FEATURES,
    sample_weight="log1p(rating_number)",
    scores={row["model"]: {"rmse": row["rmse"], "mae": row["mae"]} for row in rows},
    hyperparameters={name: {k: v for k, v in m.get_params().items() if "__" in k}
                     for name, m in models.items()})
print("\nsaved results/rating.csv, results/rating_predictions.csv and "
      f"{run_json.relative_to(ROOT).as_posix()}")

# %% Residual analysis for the model note (docs/model-notes/rating.md)
# XGBoost residuals on the test split: error by material group, the largest
# misses as citable examples, and a predicted-vs-residual plot.

import plotly.express as px

resid = (scored["xgboost"]
         .assign(residual=lambda d: d.average_rating - d.predicted))

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
