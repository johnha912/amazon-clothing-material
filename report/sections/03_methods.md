# Methods

Both tasks follow one protocol. Models train on `split == "train"` (80%) and are scored once on `split == "test"` (20%), with `random_state=42` wherever randomness enters. Every results file records the command, seed, and data version, so numbers from three authors can be merged and rechecked.

## Task A: material from listing text

The input is the title, feature bullets, and description, with every clause that contains a percentage removed in either word order ("90% Polyester" or "Polyester 88%"). Those clauses are where the label comes from, so leaving them in would turn prediction into string matching.

Each listing becomes a TF-IDF vector over word unigrams and bigrams. The weight of term $t$ in listing $d$ is

$$w_{t,d} = \bigl(1 + \log \text{tf}_{t,d}\bigr)\cdot\left(\log\frac{1 + N}{1 + \text{df}_t} + 1\right),$$

where $\text{tf}_{t,d}$ counts $t$ in $d$, $N$ is the number of training listings, and $\text{df}_t$ is how many of them contain $t$; each vector is then scaled to unit length. The log on term frequency stops a word repeated ten times from counting ten times as much, and the inverse document frequency lowers generic words ("size", "color") while raising rare, informative ones ("linen", "suede").

Two linear classifiers read the same vectors. Logistic regression models class probabilities with a softmax, $P(y = k \mid x) = e^{w_k^\top x} / \sum_j e^{w_j^\top x}$, and minimizes cross-entropy with an L2 penalty. The linear SVM trains one classifier per class (one-vs-rest) with the squared hinge loss $\max(0,\, 1 - y\,w^\top x)^2$ and predicts the class with the largest score. Both use balanced class weights, $c_k = n / (K\, n_k)$, so each rare-class listing counts more during training.

| Setting | Value |
|---|---|
| TF-IDF | Unigrams and bigrams, minimum document frequency 2, at most 60,000 terms |
| Regularization | $C = 1.0$ for both classifiers |
| Classes | Materials with at least 20 labeled listings (9 in the pilot); the shared class list in `src/labels.py` replaces this cutoff on the full sample |

DistilBERT (Yanglin Hu) uses the same text, split, and metric and is described in its own section. The saved logistic-regression pipeline also powers the demo app.

## Task B: rating from listing fields

The target is the product's average rating. Inputs are listing fields and the Task A material label only; nothing derived from the product's own ratings or reviews is allowed.

| Feature | Definition |
|---|---|
| Log price | $\log(1 + \text{price})$, median-imputed when missing |
| Has price | 1 if a price is listed, else 0 |
| Title words, text words | Word counts of the title and of the cleaned text |
| Department | One-hot encoded |
| Material | One-hot encoded, with a separate "unknown" category for unlabeled listings |

The number of ratings $n_i$ is never a feature. It enters only as a sample weight, $s_i = \log(1 + n_i)$, applied to all three models, so an average built on many ratings counts more than one built on a single rating. The baseline therefore predicts the weighted training mean, $\hat{y} = \sum_i s_i y_i / \sum_i s_i$, for every product.

The random forest averages 300 regression trees grown on bootstrap samples. XGBoost adds 400 trees one at a time, each fitting the errors left by the previous ones (learning rate 0.06, depth 6, row and column subsampling 0.9). Both minimize squared error. We score them with

$$\text{RMSE} = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}, \qquad \text{MAE} = \frac{1}{n}\sum_{i=1}^{n}\lvert y_i - \hat{y}_i\rvert.$$

Comparing three models of rising flexibility answers the real question: do listing fields carry any rating signal beyond the average?
