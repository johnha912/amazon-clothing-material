# Model note — Rating prediction (Task B)

_Owner: Nguyen Ha · Role: Modeling · Script: `analysis/rating.py` · Status: numbers on the ~250k sample (`data/sample.parquet (249,762 rows, make_sample.py 2219c1e)`, seed 42), scored through `src/evaluate.py` on the untouched test split; confidence intervals pending. Figures from the earlier 10,000-record pilot are labeled as pilot._

## How the models work

The task is regression: predict a product's `average_rating` (1–5) from its listing alone. No feature is derived from the item's own ratings or reviews (AGENTS.md leakage rules). The inputs are listing fields only — log price, a has-price flag, title and description word counts, department, and the Task A material label (missing labels become their own "unknown" category) — with numeric fields median-imputed and categorical fields one-hot encoded. `rating_number` is never a feature; it is used only as a sample weight, $\log(1 + \text{rating\_number})$, so products whose average rests on many ratings count more in training than products rated once.

Three models of increasing flexibility share those inputs. The mean baseline ignores the inputs and always predicts the weighted training mean (same weights as the tree models, so the comparison is fair); it is the floor every real model must beat. The random forest averages 300 regression trees, each grown to full depth on a bootstrap sample of the rows (scikit-learn's regressor considers every feature at each split by default), minimizing squared error; averaging cancels part of each tree's variance. XGBoost builds 400 trees sequentially instead, each new tree fitting the residual errors of the ones before it (learning rate 0.06, max depth 6, row and column subsampling 0.9), minimizing squared error plus a regularization term on tree size and leaf weights.

## Why this suits the dataset

The expected relationships are nonlinear and threshold-like — price plausibly matters in bands rather than linearly, and department interacts with everything — so tree ensembles, which carve the feature space into regions, fit the problem shape better than a linear model would. They also absorb the mixed input types (a skewed price, two word counts, two high-cardinality categories) with no scaling or distributional assumptions, and they tolerate the dataset's messiness: 79.8% of listings have no price and 63.6% have no material label (250k sample, our arithmetic), both of which the pipeline represents explicitly instead of dropping the rows.

## Results (250k sample, test split, 49,924 rows)

| Model | RMSE | MAE |
|---|---|---|
| Mean baseline | 0.7512 | 0.5385 |
| Random forest | 0.7647 | 0.5509 |
| XGBoost | 0.7373 | 0.5234 |

Train 199,838 / test 49,924 listings. Command `python analysis/rating.py`, seed 42, data version `data/sample.parquet (249,762 rows, make_sample.py 2219c1e)`, scored through `src/evaluate.py`; confidence intervals are pending. Residual detail is in `results/rating_residuals_by_material.csv`, the 200 largest misses in `results/rating_error_examples.csv`, per-row predictions in `results/rating_predictions.csv`, and the predicted-vs-residual plot in `results/rating_residual_plot.html` (PNG: `results/rating_residual_plot.png`). The pilot (first 10,000 records, 1,954 test rows) gave RMSE 0.5863 / 0.5854 / 0.5780 for the three models; those numbers are not comparable with the 250k ones.

## Error analysis

The honest headline is that listing fields barely predict ratings, and the residuals show why. XGBoost improves on the mean baseline by only 1.9% in RMSE (0.7373 vs. 0.7512) and 2.8% in MAE (our arithmetic from the table), and the random forest is worse than the baseline on both metrics (RMSE 1.8% higher, MAE 2.3% higher; in the pilot it had tied the baseline). Average ratings concentrate near the top (the sample mean is 4.08 and the standard deviation 0.75, with 84.9% of products between 3.5 and 5, and 13.7% of test products rated exactly 5.0), so a model that always answers "about 4.1" is rarely very wrong for most products, and listing fields carry little information about the exceptions. Whether the forest's weakness comes from its full-depth trees fitting noisy averages is a hypothesis we have not tested.

The largest misses are all the same shape: all 200 are products rated exactly 1.0 that XGBoost scores between 4.16 and 4.54, and each has at most 3 ratings (median 1, computed from `data/sample.parquet`). A girl's graphic squid T-shirt (actual 1.0, predicted 4.54), a nurse-slogan T-shirt (1.0 vs. 4.48), and a punk skull ring (1.0 vs. 4.44) look, from price, department, material, and word counts, like the many similar products rated 4 to 5. What made a single buyer rate them 1 star lives in the review text and the product itself, which the leakage rules correctly keep out of the features. These are not model failures so much as evidence about where the information is.

Grouped residuals add one more finding: the worst-predicted group is material = "unknown" (31,606 test rows, RMSE 0.7846, bias -0.034), above every labeled material, whose RMSE runs from 0.5802 (other) to 0.6716 (cotton). Listings without a parseable composition string may differ systematically, but we have not tested how; the group is also 63% of the test set. The two smallest labeled groups, metal (27 test rows, RMSE 0.5965, bias +0.133) and rubber (64 rows, RMSE 0.6279, bias +0.110), are too small for their numbers to count as findings.

## What transfers

The transferable result is the negative one, and it is worth stating in the report rather than hiding: for marketplace data, expect listing metadata to explain only a small share of rating variance, always benchmark against the mean baseline before claiming a model works, and use grouped residuals (here, by label coverage) to find which subpopulation a model actually fails on. The pipeline itself — leakage-safe listing features, log-count sample weights, tree ensembles with a shared preprocessing step — transfers directly to other targets on the same metadata, such as predicting price bands or return-prone categories.
