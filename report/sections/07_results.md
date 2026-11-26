# Results

All numbers are on the held-out test set of the ~250,000-product sample (data version `data/sample.parquet (249,762 rows, make_sample.py 2219c1e)`, seed 42), scored through `src/evaluate.py` with `python analysis/material_tfidf.py`, `python analysis/material_bert.py --overwrite`, `python analysis/rating.py`, and `python analysis/review_sentiment.py`; intervals are 95% bootstrap intervals from the saved predictions (`python analysis/ci_from_predictions.py`, `python analysis/material_bert.py --compare-only`, 1,000 resamples, seed 42). Appendix B lists the result file behind each table and the numbers not repeated here.

| Task A model | Macro-F1 | Accuracy | 95% CI of macro-F1 |
|---|---|---|---|
| TF-IDF + logistic regression | 0.5799 | 0.7326 | 0.5648–0.5957 |
| TF-IDF + linear SVM | 0.6203 | 0.7872 | 0.6027–0.6359 |
| DistilBERT, unweighted (chosen on validation) | 0.6758 | 0.8075 | 0.6512–0.6968 |
| DistilBERT, class-weighted | 0.6576 | 0.7733 | 0.6365–0.6777 |

On the 18,318 test listings (10 classes), the SVM is higher than logistic regression on macro-F1 and accuracy and has the higher F1 on every one of the ten classes (checked against the saved predictions, our arithmetic); the paired interval for the macro-F1 difference is +0.029 to +0.052. DistilBERT was scored on the same listings; the unweighted model, chosen by the validation rule before any test score existed, is 0.0555 above the SVM (paired interval 0.0325 to 0.0763) and 0.0960 above logistic regression (0.0729 to 0.1186), so both gaps exclude zero. The class-weighted setting is 0.0182 below the unweighted model (paired interval −0.0343 to −0.0018, which only just excludes zero); with one training run each, the safe reading is that class weights did not help here. Leather is nearly solved by all four models (F1 0.943 to 0.957), and the largest DistilBERT gains over the SVM are on spandex (+0.137) and metal (+0.297, where one of 27 test listings moves F1 by several points; per-class table in Appendix B, differences our arithmetic).

| Task B model | RMSE | 95% CI of RMSE | MAE | Gain in RMSE | Gain in MAE |
|---|---|---|---|---|---|
| Weighted mean baseline | 0.4955 | 0.4908–0.5001 | 0.3817 | — | — |
| Random forest | 0.5044 | 0.4998–0.5088 | 0.3853 | −1.8% | −0.9% |
| XGBoost | 0.4761 | 0.4715–0.4808 | 0.3620 | +3.9% | +5.2% |

Task B has 138,355 training and 34,836 test listings (gains relative to the baseline, our arithmetic from the rounded table values), so its errors are not comparable with the earlier unfiltered run (RMSE 0.7512, 0.7647, and 0.7373). Listing fields predict ratings only modestly: the paired RMSE interval is −0.0208 to −0.0180 for XGBoost minus the baseline and +0.007 to +0.012 for the random forest, so both the gain and the shortfall exceed test-set sampling noise.

Task C starts from how rarely reviews mention fabric: of 866,008 reviews, 154,677 contain a material sentence (209,830 sentences), and only 18.6% of 1-star reviews do (`results/review_sentiment_run.json`). The share of polar sentences scored positive rises from 17.8% in 1-star reviews to 94.8% in 5-star reviews, and DistilBERT fine-tuned on SST-2 agrees with the lexicon on 85.3% of the 146,981 polar sentences (Cohen's kappa 0.617). Across the 15,397 products with a target, the Spearman correlation between the star average and material sentiment is 0.292. Predicting both targets from the same listing features on the same 3,093 test products gives:

| Target | Model | RMSE | MAE | RMSE vs. baseline |
|---|---|---|---|---|
| Material sentiment | Mean baseline | 0.2795 | 0.2275 | 1.0000 |
| Material sentiment | XGBoost | 0.2815 | 0.2264 | 1.0072 |
| Star average | Mean baseline | 0.4695 | 0.3422 | 1.0000 |
| Star average | XGBoost | 0.4487 | 0.3194 | 0.9557 |

XGBoost did not beat the mean baseline on material sentiment (RMSE 0.7% worse, MAE 0.5% better): the paired intervals for XGBoost minus baseline, −0.0008 to +0.0045 in RMSE and −0.0036 to +0.0014 in MAE, both include zero, so it does not differ measurably from the baseline here, while on the same products it improves the star average by 4.4% in RMSE.
