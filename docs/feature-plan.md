# Feature plan

Proposals go here first (Analyst), a human approves or edits them, then the Engineer builds only the approved items and the Auditor appends a verdict. See `AGENTS.md`, Feature pipeline.

## Task C: material sentiment from review text (proposed 2026-11-23, Issue #3)

**Why.** Instructor feedback on the proposal: an item's star average is noisy, because a 1-star review can be about late shipping or the wrong size, not the fabric. Task B predicts that noisy average. Task C builds a target that measures what buyers say about the material itself, and Task B stays as the baseline it is compared against.

**Source.** `Clothing_Shoes_and_Jewelry.jsonl.gz` from Amazon Reviews 2023 (McAuley Lab, 7,109,695,439 bytes), fields used: `parent_asin`, `rating`, `title`, `text`. Streamed line by line and kept only for the `parent_asin`s in `data/sample.parquet` (249,762 rows, `make_sample.py` 2219c1e), at most 20 reviews per product (first 20 in file order). Output: `data/reviews_sample.parquet` (not committed). Each product keeps the `split` it has in the sample, so Task C train/test rows are a subset of the existing split.

**Material sentences.** Each review (title + text) is split into sentences. A sentence counts as material-related if it names a fabric word (fabric, material, cotton, polyester, spandex, nylon, rayon, wool, linen, silk, leather, denim, knit, ...) or a feel word that only describes fabric (soft, scratchy, itchy, see-through, sheer, stretchy, breathable, pilling, ...). Ambiguous feel words (thin, thick, sturdy, stiff, smooth, rough, ...) do not select a sentence by themselves; this was changed after the first run printed "my wrists are thin" and "a size 9 is just right to wear with a thick hiking or hunting sock" (2026-11-23). The word lists live in one place in `analysis/review_sentiment.py`. Before the lists are used, the script prints 20 random matched sentences so false positives (for example "soft" in "soft sole", "material" in "marketing material") can be checked and the lists fixed.

**Sentiment.** Version 1 is a lexicon, chosen because it is transparent and runs on a CPU in minutes. Positive terms (soft, smooth, breathable, comfortable, durable, love, ...) and negative terms (scratchy, itchy, thin, see-through, cheap, pilling, flimsy, rough, shrank, faded, ...) are counted in each material sentence; a negator (not, no, never, and similar) within the three words before a term flips it. A sentence with more positive than negative terms is positive, the reverse is negative, a tie is neutral and dropped. Version 2, only if time allows: `distilbert-base-uncased-finetuned-sst-2-english` (transformers, already pinned) on the same sentences, on Colab, as a check on the lexicon.

**Target.** `material_sentiment` = positive / (positive + negative) material sentences per product, in [0, 1], for every sample product (labeled or not; only output 2 is restricted to labeled products). Products with fewer than 3 polar material sentences get no target (too few to be stable), the same way unlabeled rows get no material label.

**Features.** Exactly the Task B features from `analysis/rating.py` (`log_price`, `has_price`, `title_words`, `text_words`, `department`, `material`). No new features.

**Leakage check.** Review text, review stars, and every count derived from them build the Task C **target** only. None of them is a feature for Task A, Task B, or Task C. `rating_number` is not a feature. The test split is unchanged: Task C test rows are the sample's `split == "test"` rows that have a target.

**Outputs.**
1. Star average vs. material sentiment on the products that have both: Spearman correlation, plus 3 to 5 real 1-star reviews whose complaint is shipping, size, or fit rather than fabric.
2. Material sentiment by material class (labeled products only): which fibers draw the most negative fabric comments, with counts.
3. Mean baseline vs. XGBoost (Task B setup, seed 42) predicting `material_sentiment`, scored with RMSE/MAE through `src/evaluate.py`, next to the Task B scores.
4. Model note `docs/model-notes/review_sentiment.md`.

**Known limits.** The lexicon misses sarcasm and phrases it has no word for; "comfortable" can be about fit, not fabric; 20 reviews per product favors early reviews. Version 2 and the printed examples are how these get checked.

**Approval.** Nguyen Ha: ☑ 2026-11-23 · Yanglin Hu: ☑ 2026-11-25 · Rui Song: ☑ 2026-11-25

**Audit verdict.** (Auditor appends here.)
