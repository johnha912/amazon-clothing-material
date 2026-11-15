# Model note — TF-IDF material classifiers (Task A)

_Owner: Nguyen Ha · Role: Modeling · Script: `analysis/material_tfidf.py` · Status: numbers on the ~250k sample (`data/sample.parquet (249,762 rows, make_sample.py 2219c1e)`, seed 42), scored through `src/evaluate.py` on the untouched test split; confidence intervals pending. Figures from the earlier 10,000-record pilot are labeled as pilot._

## How the models work

Both models read the same input: the listing text (`title` + features + description, with the composition strings removed so the answer never appears in the input). A TF-IDF vectorizer turns each listing into a long sparse vector, one entry per word or two-word phrase (unigrams + bigrams, at most 60,000 of them, words appearing in fewer than 2 listings dropped). Each entry is the term's TF-IDF weight,

$$\text{tfidf}(t, d) = \text{tf}(t, d) \cdot \left(\log \frac{1 + N}{1 + \text{df}(t)} + 1\right),$$

with sublinear term frequency ($\text{tf}$ replaced by $1 + \log \text{tf}$), so a word repeated ten times in one listing counts for more than a word said once, but not ten times more. The inverse-document-frequency factor downweights words that appear in almost every listing ("size", "color") and upweights words that appear in few ("linen", "suede"), which are exactly the words that carry material information.

On top of that vector, the two classifiers differ only in how they draw the boundary between materials. Logistic regression learns one weight vector per class and predicts by a softmax,

$$P(y = k \mid x) = \frac{e^{w_k \cdot x}}{\sum_j e^{w_j \cdot x}},$$

trained by minimizing cross-entropy loss plus an L2 penalty ($C = 1.0$). The linear SVM instead learns one separating hyperplane per class (one-vs-rest) and predicts the class with the largest decision value $w \cdot x + b$, trained with the squared hinge loss, $\max(0, 1 - y\,(w \cdot x + b))^2$, again with $C = 1.0$. Both use `class_weight="balanced"`, so rare materials count more per row than common ones during training.

## Why this suits the dataset

Listing text is short, formulaic, and keyword-driven: sellers name the fabric, the garment type, and the use case in a handful of words, so a bag-of-words representation loses little — word order rarely changes what material a listing is about. The vocabulary is also highly skewed, with a few generic words everywhere and the discriminative fiber words rare, which is precisely the situation TF-IDF weighting was designed for. These models are also cheap: they train in seconds on a CPU, which makes them the right first baselines before the DistilBERT model (Yanglin) spends GPU time, and their per-class scores show where the task itself — not the model — is hard.

## Results (250k sample, test split, 18,318 labeled rows, 10 classes)

| Model | Macro-F1 | Accuracy | 95% CI of macro-F1 |
|---|---|---|---|
| TF-IDF + logistic regression | 0.5799 | 0.7326 | pending |
| TF-IDF + linear SVM | 0.6203 | 0.7872 | pending |

Train 72,661 / test 18,318 listings. Command `python analysis/material_tfidf.py`, seed 42, data version `data/sample.parquet (249,762 rows, make_sample.py 2219c1e)`, scored through `src/evaluate.py`, on the cleaned text from `analysis/load_data.py`. The SVM is higher than logistic regression on both metrics and on every class on this test set (checked against `results/material_tfidf_predictions.csv`; for metal the F1 difference is only 0.004); confidence intervals will be computed from that file once the shared bootstrap helper lands. Per-class detail is in `results/material_tfidf_per_class.csv` (logistic regression), the confusion matrix in `results/material_tfidf_confusion.csv` (plot: `results/material_tfidf_confusion.png`), and every logistic-regression error in `results/material_tfidf_errors.csv`. The class list now comes from the shared `src/labels.py` (cotton, polyester, leather, nylon, rayon, other, spandex, wool, rubber, metal); every class has at least `MIN_CLASS_ROWS = 20` labeled rows, so none is dropped. The pilot (first 10,000 records, nine classes, 903 test rows) scored 0.5572 for logistic regression and 0.5239 for the SVM; those numbers are not comparable with the 250k ones.

## Error analysis

Logistic regression misclassifies 4,899 of 18,318 test rows (26.7%); the SVM misclassifies 3,898 (21.3%). The mistakes are not random; they cluster in three patterns.

**The big fibers blur into each other.** Polyester predicted as rayon (574 rows), nylon (555), and cotton (549), and cotton predicted as polyester (528), are the largest confusion cells. Cotton and polyester dominate the data (5,904 and 5,807 test rows) and appear in the same kinds of listings (tees, dresses, basics) described with the same marketing words, so when a listing never names its fiber the model plausibly falls back on garment-type words; we did not test that explanation directly. Per-class F1 for logistic regression: leather reaches 0.943 and cotton 0.811, while metal (0.224, on 27 test rows) and spandex (0.324) are lowest. The balanced class weights push the model to over-predict smaller classes: it predicts rayon 1,726 times for 975 true rayon rows (precision 0.428, recall 0.757).

**Whether the text names the fiber decides many errors.** For the 17,603 test rows of the eight classes other than `other` and `metal`, 30.8% contain the word of their own true class in the cleaned text (word-boundary match, our arithmetic from the predictions file and the cleaned text). On those rows the logistic-regression error rate is 13.3%; on the rest it is 32.4%. A women's tank top (true label rayon, predicted cotton) in the pilot was a typical miss: after cleaning, its text was pure brand copy about enjoying life, with no fabric word left. No text classifier can recover a label the text never hints at, so these rows set an accuracy ceiling that a bigger model will not fully remove, although DistilBERT may get more from context.

**Small classes are unstable.** Metal (27 test rows, F1 0.224 for logistic regression), rubber (64, F1 0.554), and wool (169, F1 0.575) have so few test rows that one more correct row moves their F1 by several points, and confidence intervals are pending.

**A leak in the first 250k run, now fixed.** The shared `strip_composition` in `src/labels.py` removes only strings of the form "N% fiber", and the first 250k run therefore still saw percentages in 13,373 of the 90,979 labeled texts (for example "80% Polyamide"). That run scored macro-F1 0.6397 (logistic regression) and 0.6932 (SVM), so the leak had inflated macro-F1 by about 0.06 and 0.07 (our arithmetic from the two runs). `analysis/load_data.py` now blanks every clause containing a percentage when it reads the sample, its self-check passes (no digit followed by a percent sign remains in any labeled text), and the 405 labeled listings whose whole text was such a clause are kept with empty text. The check covers only that pattern: 42 labeled listings (10 in the test set) still contain a stray percent sign in forms the labeling rule does not read, such as "%100 cotton" or "lOO%", and 1,048 spell out "N percent"; a stricter cleaner would need a rerun and is left for later. All numbers above use the cleaned text.

**A data bug the pilot analysis caught.** In the pilot, an earlier version of the text cleaning erased each percentage clause together with all text back to the previous period, and because fields were joined without one, this often wiped the title and earlier bullets and left 726 labeled pilot listings (16%) with empty text. Cleaning then removed only the delimited clause that holds the percentage, and pilot macro-F1 rose from 0.360 to 0.557 with no change to the model. This ablation was run on the pilot only and has not been repeated on the 250k sample.

## What transfers

The pipeline pattern — strip the label source from the input, TF-IDF with bigrams and sublinear scaling, balanced linear classifier — transfers to any short e-commerce text classification (category, brand tier, condition) where sellers use a formulaic vocabulary. The error lesson transfers too: when two classes share a vocabulary, report the confusion pair explicitly instead of trusting the headline macro-F1, and check how much of the error is missing signal versus genuine ambiguity before reaching for a bigger model.
