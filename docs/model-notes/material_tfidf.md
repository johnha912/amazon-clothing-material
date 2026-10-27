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
| TF-IDF + logistic regression | 0.3602 | 0.587 |
| TF-IDF + linear SVM | 0.3572 | 0.618 |

Per-class detail is in `results/material_tfidf_per_class.csv`; the confusion matrix is in `results/material_tfidf_confusion.csv` (plot: `results/material_tfidf_confusion.html`); every misclassified test row is in `results/material_tfidf_errors.csv`. Command: `python analysis/material_tfidf.py`, seed 42.

## Error analysis

The logreg model misclassifies 373 of 903 test rows (41%). The mistakes are not random; they cluster in three patterns.

**The two big fibers blur into each other.** Polyester predicted as cotton (81 rows) and cotton predicted as polyester (54 rows) are the largest confusion pair by far, and polyester also leaks into nylon (36) and rayon (28). Cotton and polyester dominate the data (298 and 317 test rows) and appear in the same kinds of listings — tees, dresses, basics — described with the same marketing words, so when a listing never names its fiber, the model falls back on garment-type words and effectively guesses between the two. Per-class F1 shows the same story: leather reaches 0.844 because leather listings use distinctive words (shoes, jackets, "genuine"), while nylon (0.395) and rayon (0.452) are absorbed by their bigger neighbors.

**Some listings carry no signal at all.** A girls' velvet dress (true label polyester) has empty text once the composition string is stripped — the composition was the only fabric statement in the listing — so the model can only guess. A men's dress shirt (true label cotton, predicted polyester) offers only size-chart boilerplate. A women's tank top (true label rayon, predicted cotton) is pure brand copy about enjoying life, with no fabric word anywhere. No text classifier can recover a label the text never hints at; these rows set an accuracy ceiling that a bigger model will not remove, although DistilBERT may squeeze slightly more from context.

**Rare classes are invisible.** Polyurethane (3 test rows), spandex (15), and wool (6) all score F1 = 0. Spandex is nearly always a minor blend component rather than a dominant fiber, so dominant-spandex listings are rare in training too; with support this small, `src/labels.py`'s merged class list (Rui) matters more than model choice, and these classes should be interpreted as "not enough data yet" until the 250k sample lands.

## What transfers

The pipeline pattern — strip the label source from the input, TF-IDF with bigrams and sublinear scaling, balanced linear classifier — transfers to any short e-commerce text classification (category, brand tier, condition) where sellers use a formulaic vocabulary. The error lesson transfers too: when two classes share a vocabulary, report the confusion pair explicitly instead of trusting the headline macro-F1, and check how much of the error is missing signal versus genuine ambiguity before reaching for a bigger model.
