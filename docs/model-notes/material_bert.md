# Model note — DistilBERT material classifier (Task A)

_Owner: Yanglin Hu · Role: Modeling (lead) · Script: `analysis/material_bert.py` · Status: 250k run done 2026-11-25 (Issue #2). Numbers on `data/sample.parquet (249,762 rows, make_sample.py 2219c1e)`, seed 42, scored through `src/evaluate.py` on the untouched test split, with bootstrap 95% confidence intervals. Command `python analysis/material_bert.py --overwrite`; intervals and the four-model table from `python analysis/material_bert.py --compare-only`; error files from `python analysis/material_bert.py --error-analysis`. The earlier 10,000-record pilot is mentioned only where labeled as pilot._

## How the model works

DistilBERT (Sanh et al., 2019) is a smaller version of BERT, trained to imitate BERT's outputs: 6 transformer layers instead of 12, 768-dimensional token vectors, 12 attention heads, and a 30,522-word WordPiece vocabulary. A listing's cleaned text is split into word pieces and cut to the first 256. Each layer then updates every token's vector by looking at every other token through self-attention,

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d}}\right) V,$$

so the same word gets a different vector in a different context ("shell" in "shell: nylon" and in "shell earrings"). The classification head reads the vector of the first token, $h$, computes $z = W_2\,\text{ReLU}(W_1 h + b_1) + b_2$ with dropout 0.2 between the two layers, and turns it into probabilities $p = \text{softmax}(z)$ over the 10 materials. Every weight of the network is fine-tuned on our listings, not just the head.

The training loss is cross-entropy with an optional class weight $c_k$:

$$L = -\frac{\sum_i c_{y_i} \log p_{i, y_i}}{\sum_i c_{y_i}}, \qquad c_k = \frac{n}{K\, n_k},$$

where $n$ is the number of rows being fitted, $K = 10$, and $n_k$ is the number of rows of class $k$. Two settings were fixed before the 250k run. `distilbert` sets every $c_k = 1$, the plain loss the pilot used. `distilbert_weighted` uses $c_k = n / (K n_k)$, the `class_weight="balanced"` rule of the TF-IDF models; the weights run from 0.301 for cotton to 58.13 for metal, so a metal listing counts 193 times as much as a cotton listing (weights in `results/material_bert_run.json`; the ratio is our arithmetic). Optimization is AdamW with learning rate $2 \times 10^{-5}$ decaying linearly to 0, weight decay 0.01, batch size 16, and mixed precision (fp16).

**Choosing the number of epochs without touching test.** For each setting, 20% of the 72,661 training listings (14,533, stratified by material, seed 42) are held out for validation. The model trains for 3 epochs on the other 58,128 and is scored on validation after each epoch, and the epoch count with the highest validation macro-F1 wins (the smaller count on a tie). A fresh copy of the pretrained model is then trained on all 72,661 training listings for that many epochs, and only this final model predicts the 18,318 test listings, once. Which setting the report leads with was also fixed in advance: the one with the higher validation macro-F1. The script applies that rule after both settings have predicted the test split, but the rule reads only the validation scores. Both test scores are reported either way.

## Why it suits this dataset

TF-IDF counts words but ignores their order and neighbors. A transformer reads each word in context, which should help where the TF-IDF note found most of its errors: listings that never name their fiber, where the material has to be inferred from garment type, construction, or care words, and listings that name several fibers. DistilBERT is 40% smaller and 60% faster than BERT while keeping 97% of its language-understanding performance (Sanh et al., 2019), so 72,661 listings fine-tune on one laptop GPU in two to three minutes per epoch. The data also limits what context can add. Listings are short and formulaic, the label's source (the composition string) is removed from the input, and 19.24% of training listings (19.55% of test) are longer than 256 tokens and are cut, which can drop a fiber word that appears late in a long description (`share_longer_than_max_length` in the run file).

## Setup and runtime

The sample was rebuilt for this run with `python -m src.make_sample data/meta_Clothing_Shoes_and_Jewelry.jsonl --n 250000` (63 s of wall-clock time, timed in the shell during the session and not saved in any file). Its printed summary matches the team's numbers exactly: 249,762 rows, 36.4% labeled, 79.8% missing price, 199,838 train and 49,924 test rows. The Task A population is 72,661 training and 18,318 test listings in 10 classes. Class eligibility (at least 20 labeled listings) is counted over both splits, the same rule as `material_tfidf.py`, but all ten classes also qualify on training listings alone (the smallest, metal, has 125), and the script now refuses to run if that ever stops being true. These are the same rows the TF-IDF models were scored on: the `test_ids_sha256` of both runs is `8cc141bc…a850`, and `--compare-only` refuses to build the comparison if the hashes differ, if the prediction files cover different listings, or if a saved score does not match its saved predictions.

The run used one NVIDIA GeForce RTX 5090 Laptop GPU (24 GB) under Windows 11, with Python 3.13.14 and the versions pinned in `requirements.txt` (torch 2.14.1 built for CUDA 13.0, transformers 5.20.2, pandas 3.0.6, scikit-learn 1.9.1). Pretrained revision `12040accade4e8a0f71eabdb258fecc2e7e948be` is the same one the pilot used. Tokenizing, training, and predicting both settings took 1,719.8 s (28.7 min). That timer, in `results/material_bert_run.json`, starts after the data are loaded and stops before the error analysis and the bootstrap intervals. For each setting it covers about 385 s of epoch selection and 453 to 463 s of refitting and predicting. Hyperparameters, package versions, class weights, validation histories, the script's SHA-256, and the task fingerprint are in the same file. The SHA-256 belongs to the script as it ran; the later audit fixes (provenance columns in the output files, the eligibility guard, `--skip-png`, one reformatted line) changed no training step and are a separate commit. Weights, tokenizer, and the split manifest stay in `models/material_bert/` and are not committed.

Sources for figures not in `results/`: the architecture (6 layers, 768 dimensions, 12 heads, 30,522-word vocabulary, head dropout 0.2) is the pretrained configuration saved with the weights (`models/material_bert/*/final/config.json`); learning rate and batch size are listed under `hyperparameters` in the run file, and weight decay (0.01) and fp16 are set in `training_arguments` in the script as it ran (the first of the two commits); AdamW with linear decay and no warmup are the defaults of the transformers 5.20.2 `Trainer`; the 24 GB of GPU memory is from `nvidia-smi`; the 1,846-character example and the check of the examples against the 26 material names in `src/labels.py` use the cleaned text of `data/sample.parquet`; the 40%, 60%, and 97% figures are from Sanh et al. (2019, https://arxiv.org/abs/1910.01108).

## Results (250k sample, test split, 18,318 listings, 10 classes)

| Model | Macro-F1 | 95% CI of macro-F1 | Accuracy |
|---|---|---|---|
| **DistilBERT, unweighted (`distilbert`)** | **0.6758** | 0.6512–0.6968 | **0.8075** |
| DistilBERT, class-weighted (`distilbert_weighted`) | 0.6576 | 0.6365–0.6777 | 0.7733 |
| TF-IDF + linear SVM | 0.6203 | 0.6027–0.6359 | 0.7872 |
| TF-IDF + logistic regression | 0.5799 | 0.5648–0.5957 | 0.7326 |

Source: `results/material_comparison.csv`. Intervals are percentile bootstraps over the test listings (1,000 resamples, seed 42) from `src/evaluate.py`. They cover how much a score would move on another draw of test listings, not how much it would move if the model were retrained with another seed. Macro-F1, accuracy, and the per-class scores below come from `material_scores` and `material_per_class` in `src/evaluate.py`, rescored from the saved predictions. The training run itself computed accuracy with scikit-learn's `accuracy_score`, which `material_scores` wraps, and `--compare-only` refuses to build the table if a saved score differs from its rescored value.

Both settings chose 3 epochs, the most the search allowed. Validation macro-F1 by epoch was 0.5882, 0.6635, 0.6784 without weights and 0.5823, 0.6197, 0.6460 with them. The unweighted model scored higher on validation, so by the rule fixed before the run it is the DistilBERT result the report leads with.

| Paired gap on the same test listings | Estimate | 95% CI |
|---|---|---|
| DistilBERT − TF-IDF + SVM | +0.0555 | +0.0325 to +0.0763 |
| DistilBERT − TF-IDF + logistic regression | +0.0960 | +0.0729 to +0.1186 |
| DistilBERT weighted − TF-IDF + SVM | +0.0373 | +0.0178 to +0.0560 |
| DistilBERT weighted − DistilBERT unweighted | −0.0182 | −0.0343 to −0.0018 |

Source: `results/material_bert_ci.csv` (`bootstrap_diff_ci`, the same resampled listings for both models). DistilBERT's lead over the SVM is larger than test-sampling noise. Class weights did not help: the weighted model is lower, and its interval only just excludes zero. That interval does not include retraining noise, and we trained each setting once, so "class weights did not help on this run" is as far as the evidence goes. The pilot's DistilBERT score (0.3563 on 903 pilot test listings, nine classes, unweighted; `results/material_bert.csv` as of commit `f3f7a3e`) is not comparable with these numbers.

| Material | Test listings | TF-IDF LR | TF-IDF SVM | DistilBERT | DistilBERT weighted |
|---|---|---|---|---|---|
| Cotton | 5,904 | 0.811 | 0.842 | **0.848** | 0.832 |
| Polyester | 5,807 | 0.693 | 0.771 | **0.790** | 0.740 |
| Leather | 3,188 | 0.943 | 0.953 | **0.957** | 0.953 |
| Nylon | 1,110 | 0.589 | 0.637 | **0.657** | 0.629 |
| Rayon | 975 | 0.547 | 0.592 | **0.612** | 0.597 |
| Other | 688 | 0.540 | 0.577 | **0.586** | 0.571 |
| Spandex | 386 | 0.324 | 0.353 | 0.490 | **0.519** |
| Wool | 169 | 0.575 | **0.605** | 0.593 | 0.515 |
| Rubber | 64 | 0.554 | 0.647 | 0.702 | **0.719** |
| Metal | 27 | 0.224 | 0.227 | **0.524** | 0.500 |

F1 per class, from `results/material_bert_per_class.csv`, which also holds precision, recall, and the number of times each model predicted each class. The unweighted DistilBERT has the highest F1 on seven classes, the weighted one on spandex and rubber, and the SVM on wool. The gains are small on the big classes (+0.006 on cotton and +0.019 on polyester against the SVM) and large on the small ones (spandex +0.137, metal +0.297), but metal has only 27 test listings, so one listing moves its F1 by several points.

## Error analysis

The unweighted DistilBERT misclassifies 3,526 of 18,318 test listings (19.2%), against 3,898 (21.3%) for the SVM (`results/material_bert_error_summary.json`; shares are our arithmetic). The two models make the same prediction on 15,041 listings. DistilBERT alone is right on 1,503 and the SVM alone on 1,131, and both are wrong on 2,395, so about two thirds of DistilBERT's errors are also SVM errors: the two models share a hard core of listings. The confusion matrix is `results/material_bert_confusion.png` (`.csv`, `.html`; the weighted model's are `material_bert_weighted_confusion.*`), and every error with a 300-character excerpt is in `results/material_bert_errors.csv`.

**Errors flow into the two big classes.** The largest cells are cotton predicted as polyester (641), polyester as cotton (580), nylon as polyester (304), and rayon as polyester (272). Without class weights the model leans toward frequent classes: it predicts polyester 6,238 times for 5,807 true polyester listings, and spandex only 238 times for 386, rayon 804 times for 975, and metal 15 times for 27. Its small-class precision is therefore higher than its recall (spandex 0.643 against 0.396).

**Class weights swap one error for another.** The weighted model moves the other way, toward smaller classes. It predicts nylon 1,451 times for 1,110 true listings, rayon 1,343 times for 975, `other` 930 times for 688, and wool 266 times for 169. Recall goes up and precision goes down (nylon: recall 0.726 against 0.614, precision 0.555 against 0.707). In aggregate, the F1 losses on polyester (0.790 to 0.740), cotton (0.848 to 0.832), and wool (0.593 to 0.515) outweigh the gains on spandex and rubber, although the shift also corrects some listings that the unweighted model wrongly called cotton or polyester. One plausible reason, which we did not test: weights as large as 193 to 1 make the few rare-class listings dominate the gradient, while the unweighted pretrained model already handles rare classes better than the class-weighted SVM does (metal 0.524 against 0.227). It is the same over-prediction the TF-IDF note found for logistic regression with balanced weights (rayon predicted 1,726 times for 975 listings).

**On the eight fiber classes, most remaining errors are listings whose text omits their own fiber word.** On the 17,603 test listings of the eight classes named after one fiber word (all but `other` and `metal`), 30.8% contain the word of their own class in the cleaned text, the same split as the TF-IDF note (`fiber_word_split` in the summary file; the code reproduces that note's logistic-regression figures of 13.3% and 32.4%). Where the word is present, DistilBERT's error rate is 5.8% against the SVM's 8.8%. Where it is absent, the rate is 23.8% against 25.7%. The relative drop is larger where the word is present, about a third of the errors (477 to 317) against 7% (3,129 to 2,898). One plausible reading, which we did not test, is that context helps most on listings that name several fibers or name the right one in passing. The absolute numbers point the other way: the 12,183 listings without their own word are 69% of these rows and hold 2,898 of DistilBERT's 3,215 errors on them (90%). Of those 2,898, 520 name a different one of the eight fiber words and 2,378 name none of them (counts from `fiber_word_split`; shares are our arithmetic).

Five errors drawn at random (seed 42) from the five largest confusion cells show why:

| True → predicted | parent_asin | Cleaned listing text (excerpt) |
|---|---|---|
| cotton → polyester | B012C6B0GM | "Van Heusen Men's Short-Sleeve Oxford Dress Shirt … Button closure Machine Wash … Wrinkle-resistant and stain-repellant." |
| polyester → cotton | B072XDDX5F | "Modfine Men's Casual Long Sleeve Slim Fit Double-Breasted … Dress Shirts … Cotton,Polyester Button closure Hand Wash Only Material: , ." |
| nylon → polyester | B074KC4WWF | "AONIJIE Hydration Backpack Vest, 5.5L Capacity, Multi-Pocket Design, Breathable and Lightweight …" |
| rayon → polyester | B08HYRCJBB | "OLUOLIN Two Piece Outfits for Women … Tie Dye Sweatsuit Tracksuits Material: , .Stretchy,soft and comfy …" |
| polyester → nylon | B0070A8FLO | "Gore Women's Power So Lady Tights SHELL: ,WINDSTOPPERMEMBRANE, ,INSERT: , , Imported …" |

In three of the five, the stubs "Material: , ." and "SHELL: , INSERT: , ," show exactly where a composition clause was blanked to keep the label out of the input; the fiber information was in that clause, and the rest of the text does not say which fiber dominates. The backpack vest names none of the 26 material names in `src/labels.py` anywhere in its 1,846 characters of cleaned text. The Modfine shirt names both "Cotton" and "Polyester" with the percentages removed, so its dominant fiber cannot be read from the text. The Van Heusen shirt names none of the 26 either; "wrinkle-resistant" is the kind of cue a model can read as synthetic, but the label says cotton. These listings carry little or no text evidence of their dominant fiber, and in three of the five our leakage rule removed it. We did not measure how much of that evidence any text model could still recover, so this describes the errors; it does not establish a ceiling.

**Scope caveat.** Leather is the easiest class (F1 0.957), but shoes stay in scope by team decision, and on the pilot most leather test listings were shoes, so part of that score measures "this is a shoe".

## What transfers

The workflow transfers to any listing-attribute task: strip the label's source from the input, fine-tune a small pretrained transformer, pick epochs on a validation split of train, refit once, and compare against a TF-IDF baseline on the same test rows with paired bootstrap intervals. Two lessons generalize. First, a class-weighted loss is not automatically the fix for imbalance with a pretrained model. Here it traded precision for recall and lowered macro-F1, so test it as a setting chosen on validation rather than adopt it by default. Second, check how many errors are "the text no longer contains the answer" before reaching for a bigger model. Here, across the eight tracked fiber classes, 2,898 of 3,215 errors omitted the listing's own fiber word; of those, 520 named another tracked fiber and 2,378 named none of the eight, which points at the input more than at the classifier.

**Limits of this run.** One training seed per setting. The epoch search stopped at its upper bound (3), with unweighted validation macro-F1 still rising, so more epochs might score higher; that has not been tested, and any such search must stay on the validation split. GPU training is not bit-for-bit deterministic, so a rerun may move the last digits.
