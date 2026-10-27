# Results (pilot)

These are pilot results on the first 10,000 metadata records (data version `data/meta_head.jsonl pilot (10,000 rows)`, built by `analysis/load_data.py`; seed 42). They validate the pipeline end to end and will be replaced by the 50,000-product sample, using the same commands: `python analysis/material_tfidf.py` and `python analysis/rating.py`.

## Task A: material

| Model | Macro-F1 | Accuracy |
|---|---|---|
| TF-IDF + logistic regression | 0.360 | 0.587 |
| TF-IDF + linear SVM | 0.357 | 0.618 |
| DistilBERT | pending | pending |

The two linear models are almost tied (903 test listings, 9 classes). The per-class scores of the logistic regression show where the signal is.

| Material | Test listings | F1 |
|---|---|---|
| Leather | 115 | 0.844 |
| Cotton | 298 | 0.640 |
| Polyester | 317 | 0.554 |
| Rayon | 72 | 0.452 |
| Nylon | 59 | 0.395 |
| Acrylic | 18 | 0.357 |
| Spandex, wool, polyurethane | 15, 6, 3 | 0.000 |

Leather is easy because its listings use distinctive words (shoes, jackets, "genuine"). Polyester and cotton blur into each other: polyester was predicted as cotton 81 times and cotton as polyester 54 times, because both appear in the same basic garments described with the same marketing language. The three smallest classes score zero, a data problem rather than a model problem, which the merged class list and the larger sample address.

## Task B: rating

We measure each model's gain over the baseline as $\Delta = (\text{RMSE}_{\text{base}} - \text{RMSE}_{\text{model}}) / \text{RMSE}_{\text{base}}$, and likewise for MAE.

| Model | RMSE | MAE | Gain in RMSE | Gain in MAE |
|---|---|---|---|---|
| Weighted mean baseline | 0.586 | 0.383 | — | — |
| Random forest | 0.587 | 0.387 | −0.1% | −0.9% |
| XGBoost | 0.578 | 0.377 | +1.5% | +1.6% |

Listing fields barely predict ratings. The random forest does not beat the baseline on either metric, and XGBoost improves it by only about 1.5%. This matches the exploration: ratings cluster near 4.2, and what makes buyers unhappy (fit, sizing, quality, shipping) is not visible in a listing. The residuals confirm it. Unlabeled products, the "unknown" material group with 1,043 test listings, have the largest error (RMSE 0.677), against 0.25 to 0.48 for the five main labeled materials. We report this as a finding, not a failure.
