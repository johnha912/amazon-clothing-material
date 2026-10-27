# Model note — Rating prediction (Task B)

_Owner: Nguyen Ha · Role: Modeling · Script: `analysis/rating.py` · Status: pilot numbers on the first 10,000 metadata records (`data/meta_head.jsonl` pilot, seed 42); rerun unchanged on Rui's 50k sample (Oct 28) and ~250k sample (Nov 1) to refresh every number below. Metrics come from `sklearn.metrics` on the untouched test split, per AGENTS.md until `src/evaluate.py` exists._

## How the models work

The task is regression: predict a product's `average_rating` (1–5) from its listing alone. No feature is derived from the item's own ratings or reviews (AGENTS.md leakage rules). The inputs are listing fields only — log price, a has-price flag, title and description word counts, department, and the Task A material label (missing labels become their own "unknown" category) — with numeric fields median-imputed and categorical fields one-hot encoded. `rating_number` is never a feature; it is used only as a sample weight, $\log(1 + \text{rating\_number})$, so products whose average rests on many ratings count more in training than products rated once.

Three models of increasing flexibility share those inputs. The mean baseline ignores the inputs and always predicts the weighted training mean (same weights as the tree models, so the comparison is fair); it is the floor every real model must beat. The random forest averages 300 regression trees, each trained on a bootstrap sample with random feature subsets at each split, minimizing squared error; averaging cancels much of each tree's variance. XGBoost builds 400 trees sequentially instead, each new tree fitting the residual errors of the ones before it (learning rate 0.06, max depth 6, row and column subsampling 0.9), minimizing squared error plus a regularization term on tree size and leaf weights.

## Why this suits the dataset

The expected relationships are nonlinear and threshold-like — price plausibly matters in bands rather than linearly, and department interacts with everything — so tree ensembles, which carve the feature space into regions, fit the problem shape better than a linear model would. They also absorb the mixed input types (a skewed price, two word counts, two high-cardinality categories) with no scaling or distributional assumptions, and they tolerate the dataset's messiness: 45.7% of listings have no price and 54% have no material label, both of which the pipeline represents explicitly instead of dropping the rows.

## Results (pilot, test split, 1,954 rows)

| Model | RMSE | MAE |
|---|---|---|
| Mean baseline | 0.5863 | 0.3834 |
| Random forest | 0.5868 | 0.3868 |
| XGBoost | 0.5777 | 0.3774 |

Residual detail is in `results/rating_residuals_by_material.csv`, the 200 largest misses in `results/rating_error_examples.csv`, and the predicted-vs-residual plot in `results/rating_residual_plot.html`. Command: `python analysis/rating.py`, seed 42.

## Error analysis

The honest headline is that listing fields barely predict ratings, and the residuals show why. XGBoost improves on the mean baseline by only about 1.5% in RMSE (0.5777 vs. 0.5863) and 1.6% in MAE, and the random forest does not beat the baseline at all (RMSE 0.5868, MAE 0.3868) — the signature of models that have learned the safe center of the distribution and little else. Average ratings concentrate tightly (the sample mean is 4.195 and most products sit between 3.5 and 5), so a model that always answers "about 4.2" is rarely very wrong, and listing fields carry almost no information about the exceptions.

The largest misses are all the same shape: products rated 1.0 that the model scores around 4.3–4.6. A St. Patrick's Day graphic tee (actual 1.0, predicted 4.55), a velvet jewelry display bust (1.0 vs. 4.41), a men's fashion tee (1.0 vs. 4.38), and a men's tracksuit (1.0 vs. 4.32) look, from price, department, material, and word counts, exactly like the thousands of similar products rated 4–5. What made buyers hate them — quality failures, wrong sizing, misleading photos — lives in the review text and the product itself, which the leakage rules correctly keep out of the features. These are not model failures so much as evidence about where the information is.

Grouped residuals add one more finding: the worst-predicted group is material = "unknown" (1,043 test rows, RMSE 0.677, bias −0.06), well above labeled groups like leather (0.482) or nylon (0.484). Listings without a parseable composition string differ systematically — thinner metadata overall — and the mild negative bias means the model slightly overpredicts them. Rare labeled groups (rubber, wool, polyurethane, each under 10 test rows) show large RMSE with positive bias, but at those counts the numbers are anecdotes, not findings; the 250k rerun will tell.

## What transfers

The transferable result is the negative one, and it is worth stating in the report rather than hiding: for marketplace data, expect listing metadata to explain only a small share of rating variance, always benchmark against the mean baseline before claiming a model works, and use grouped residuals (here, by label coverage) to find which subpopulation a model actually fails on. The pipeline itself — leakage-safe listing features, log-count sample weights, tree ensembles with a shared preprocessing step — transfers directly to other targets on the same metadata, such as predicting price bands or return-prone categories.
