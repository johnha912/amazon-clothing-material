# Model note — TF-IDF material classifiers (Task A)

_Owner: Nguyen Ha · Role: Modeling · Script: `analysis/material_tfidf.py` · Status: pilot numbers on the first 10,000 metadata records (`data/meta_head.jsonl` pilot, seed 42); rerun unchanged on Rui's 50k sample (Oct 28) and ~250k sample (Nov 1) to refresh every number below. Metrics come from `sklearn.metrics` on the untouched test split, per AGENTS.md until `src/evaluate.py` exists._

## How the models work

Both models read the same input: the listing text (`title` + features + description, with the composition strings removed so the answer never appears in the input). A TF-IDF vectorizer turns each listing into a long sparse vector, one entry per word or two-word phrase (unigrams + bigrams, at most 60,000 of them, words appearing in fewer than 2 listings dropped). Each entry is the term's TF-IDF weight,

$$\text{tfidf}(t, d) = \text{tf}(t, d) \cdot \left(\log \frac{1 + N}{1 + \text{df}(t)} + 1\right),$$

with sublinear term frequency ($\text{tf}$ replaced by $1 + \log \text{tf}$), so a word repeated ten times in one listing counts for more than a word said once, but not ten times more. The inverse-document-frequency factor downweights words that appear in almost every listing ("size", "color") and upweights words that appear in few ("linen", "suede"), which are exactly the words that carry material information.

On top of that vector, the two classifiers differ only in how they draw the boundary between materials. Logistic regression learns one weight vector per class and predicts by a softmax,

$$P(y = k \mid x) = \frac{e^{w_k \cdot x}}{\sum_j e^{w_j \cdot x}},$$

trained by minimizing cross-entropy loss plus an L2 penalty ($C = 1.0$). The linear SVM instead learns one separating hyperplane per class (one-vs-rest) and predicts the class with the largest decision value $w \cdot x + b$, trained with the squared hinge loss, $\max(0, 1 - y\,(w \cdot x + b))^2$, again with $C = 1.0$. Both use `class_weight="balanced"`, so rare materials count more per row than common ones during training.

## Why this suits the dataset

Listing text is short, formulaic, and keyword-driven: sellers name the fabric, the garment type, and the use case in a handful of words, so a bag-of-words representation loses little — word order rarely changes what material a listing is about. The vocabulary is also highly skewed, with a few generic words everywhere and the discriminative fiber words rare, which is precisely the situation TF-IDF weighting was designed for. These models are also cheap: they train in seconds on a CPU, which makes them the right first baselines before the DistilBERT model (Yanglin) spends GPU time, and their per-class scores show where the task itself — not the model — is hard.

## Results (pilot, test split, 903 labeled rows, 9 classes)

| Model | Macro-F1 | Accuracy |
|---|---|---|
| TF-IDF + logistic regression | 0.5572 | 0.677 |
| TF-IDF + linear SVM | 0.5239 | 0.736 |

Per-class detail is in `results/material_tfidf_per_class.csv`; the confusion matrix is in `results/material_tfidf_confusion.csv` (plot: `results/material_tfidf_confusion.png`); every misclassified test row is in `results/material_tfidf_errors.csv`. Command: `python analysis/material_tfidf.py`, seed 42.

## Error analysis

The logreg model misclassifies 292 of 903 test rows (32%). The mistakes are not random; they cluster in three patterns.

**The two big fibers blur into each other.** Polyester predicted as cotton (53 rows) and cotton predicted as polyester (49 rows) are the largest confusion pair, and polyester also leaks into nylon (36) and rayon (31). Cotton and polyester dominate the data (298 and 317 test rows) and appear in the same kinds of listings — tees, dresses, basics — described with the same marketing words, so when a listing never names its fiber, the model falls back on garment-type words and effectively guesses between the two. Per-class F1 shows the same story: leather reaches 0.944 because its strongest terms are distinctive ("leather", "sole", "rubber sole", "heel", "shoe": the class is mostly footwear), while nylon (0.484) and rayon (0.543) are absorbed by their bigger neighbors.

**Whether the text names the fiber decides most errors.** After the composition clauses are removed, 34.7% of test listings still name their own material somewhere else (for example a title such as "Women's Leather Loafers"). On those rows the error rate is 22.0%; on the rest it is 37.8%. A women's tank top (true label rayon, predicted cotton) is a typical miss: after cleaning, its text is pure brand copy about enjoying life, with no fabric word left. No text classifier can recover a label the text never hints at; these rows set an accuracy ceiling that a bigger model will not fully remove, although DistilBERT may get more from context.

**Rare classes are unstable.** Spandex (15 test rows, F1 0.200), wool (6, F1 0.400), and polyurethane (3, F1 0.500) have too few examples for their scores to mean much: one more correct row moves them by 0.1 or more. Spandex is nearly always a minor blend component rather than a dominant fiber, so dominant-spandex listings are rare in training too; `src/labels.py`'s merged class list (Rui) matters more than model choice here, and these classes should be read as "not enough data yet" until the 250k sample lands.

**A data bug this analysis caught.** An earlier version of the text cleaning erased each percentage clause together with all text back to the previous period, and because fields were joined without one, this often wiped the title and earlier bullets and left 726 labeled listings (16%) with empty text. Cleaning now removes only the comma- or sentence-delimited clause that holds the percentage. Macro-F1 rose from 0.360 to 0.557 with no change to the model, which shows how much of the material signal lives outside the composition string.

## What transfers

The pipeline pattern — strip the label source from the input, TF-IDF with bigrams and sublinear scaling, balanced linear classifier — transfers to any short e-commerce text classification (category, brand tier, condition) where sellers use a formulaic vocabulary. The error lesson transfers too: when two classes share a vocabulary, report the confusion pair explicitly instead of trusting the headline macro-F1, and check how much of the error is missing signal versus genuine ambiguity before reaching for a bigger model.
