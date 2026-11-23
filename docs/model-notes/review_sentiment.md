# Model note — Material sentiment from review text (Task C)

_Owner: Nguyen Ha · Role: Modeling · Script: `analysis/review_sentiment.py` · Plan: `docs/feature-plan.md` (Task C) · Status: version 1 (lexicon) on the ~250k sample (`data/sample.parquet (249,762 rows, make_sample.py 2219c1e)`, seed 42), scored through `src/evaluate.py`; confidence intervals pending. Command `python analysis/review_sentiment.py`, run on Windows 11 (CPU), 805 s for the one-time pass over the reviews file, 486 s for a rerun from the cache._

## Why this task exists

The instructor's feedback on the proposal was that an item's star average is noisy: a buyer can leave 1 star because of late shipping or the wrong size, which says nothing about the fabric. Task B predicts that noisy average. Task C builds a target from what buyers actually write about the material and asks how much of the star average it explains, which materials draw the most fabric complaints, and whether listing fields can predict it.

## How it works

The reviews come from `Clothing_Shoes_and_Jewelry.jsonl.gz` (Amazon Reviews 2023, 7,109,695,439 bytes, 66,033,346 reviews). One streamed pass keeps the first 20 reviews of each product in the sample, 866,008 reviews for 249,705 of the 249,762 products, cached in `data/reviews_sample.parquet`. Each review (title plus text) is split into sentences, and a sentence counts as material-related if it names a fabric (cotton, polyester, fabric, material, ...) or a feel word that only describes fabric (soft, scratchy, itchy, see-through, pilling, ...). Ambiguous words such as thin, thick, sturdy, and stiff do not select a sentence by themselves, because the first run's printed sample showed them describing wrists, socks, and ankles; they still score polarity inside a sentence that names a fabric. That leaves 209,830 material sentences from 154,677 reviews.

Each material sentence gets a polarity from a lexicon: positive terms (soft, breathable, comfortable, love, ...) count +1, negative terms (scratchy, itchy, thin, cheap, pilling, ...) count −1, and a negator (not, no, never, and similar) within the three words before a term flips its sign, so "not soft" is negative and "not itchy" is positive. The sign of the sum is the sentence polarity; ties are neutral and dropped. The target is $\text{material\_sentiment} = \frac{\#\text{positive}}{\#\text{positive} + \#\text{negative}}$ over a product's material sentences, kept only for products with at least 3 polar sentences: 15,397 products (12,304 train, 3,093 test, the sample's own split).

The lexicon is version 1 because every decision it makes can be read and checked. As a check that it measures something real, the share of polar material sentences that are positive rises steadily with the review's own star: 17.8% in 1-star reviews, 34.8% in 2-star, 53.8% in 3-star, 80.5% in 4-star, and 94.8% in 5-star (`results/review_sentiment_lexicon_check.csv`).

## Results

Most reviews never mention the material. Only 18.6% of 1-star reviews contain a material sentence (22.8%, 22.7%, 21.1%, and 15.8% for 2 to 5 stars), so about four in five 1-star reviews do not mention the material in words the lexicon matches (`results/review_sentiment_run.json`, key `material_mention_share_by_star`). Five random 1-star reviews with no material sentence that mention size, fit, shipping, or returns (`results/review_sentiment_noise_examples.csv`) read "The size is not advertiserd [sic] correctly they are too big", "very disappointed with how small the crotch and butt area of the suit was", "extremely narrow at the top, wouldn't even fit over my calf", "Mold on the bat wings", and "returned way to big". These five illustrate the kind of noise the instructor described; they are examples, not a measured share. Across the 15,397 Task C products, the Spearman correlation between the star average and material sentiment is 0.292: the two move together, but weakly.

By material class (labeled products only, `results/review_sentiment_by_material.csv`), cotton (2,754 products, 0.773) and polyester (3,348, 0.776) have the lowest material sentiment and leather (392, 0.843) and the "other" class (338, 0.841) the highest; metal (2) and rubber (3) are too small to count as findings. The spread between classes is small, 0.07 between cotton and leather, against a per-product standard deviation of about 0.28 (the mean baseline's RMSE).

Predicting both targets from the Task B listing features on the same 3,093 test products (no sample weights for either target, so the two are treated alike):

| Target | Model | RMSE | MAE | RMSE vs. baseline |
|---|---|---|---|---|
| Material sentiment | Mean baseline | 0.2795 | 0.2275 | 1.0000 |
| Material sentiment | XGBoost | 0.2815 | 0.2264 | 1.0072 |
| Star average | Mean baseline | 0.4695 | 0.3422 | 1.0000 |
| Star average | XGBoost | 0.4487 | 0.3194 | 0.9557 |

XGBoost did not beat the mean baseline on RMSE for material sentiment (0.7% worse, MAE 0.5% better; confidence intervals pending), while on the same products it improves the star average by 4.4% in RMSE. Listing fields carry some signal about the overall rating (plausibly price and department relate to fit and expectations), but little about how the fabric feels, which we think is only known after someone wears it (untested).

## Error analysis

The printed random sample of material sentences shows where the lexicon fails. It misses words it has no entry for: "the material is so irritating", "material is shiny- not my taste", and "quality fabric and well made" all score 0 and are dropped. It also counts shoe parts as fabric feel, as in "the sole is soft but not too soft", because soft selects the sentence. Both kinds of error push toward a smaller, slightly mis-targeted set of sentences rather than flipped labels, and the monotone rise with review stars suggests the polarity it does assign is mostly right. Version 2 (`analysis/taskc_sst2_agreement.py`, run on Kaggle) compares the lexicon with a pretrained SST-2 model on all 209,830 sentences: they agree on 85.3% of the 146,981 polar ones (Cohen's kappa 0.617, `results/taskc_sst2_agreement.csv`). That is agreement with another model, not an error rate against human labels.

## What transfers

For any review-based product target, check how often the reviews talk about the attribute at all before trusting the overall star average as a proxy for it: here four in five 1-star reviews contain no sentence the lexicon matches. A transparent lexicon with a negation window, validated against the review's own star, is a cheap first version that can be audited sentence by sentence before a heavier model is used.
