"""The shared metrics entry point. Every number in results/ and report/ comes from here.

Point scores: material_scores (macro-F1), rating_scores (RMSE, MAE).
Uncertainty: bootstrap_ci gives a 95% interval for one model's score, and
bootstrap_diff_ci gives one for the gap between two models scored on the same
test rows. Both resample test rows with replacement (1,000 times, seed 42 by
default) and take the 2.5th and 97.5th percentiles of the resampled scores.

The intervals cover test-set sampling noise only: how much a score would move on
a different draw of test rows of the same size. They do not cover retraining
with another seed or another training sample.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, mean_absolute_error, root_mean_squared_error
from sklearn.utils.multiclass import unique_labels

SEED = 42
N_RESAMPLES = 1000


def _macro_f1(y_true, y_pred):
    # Macro over the classes present in y_true or y_pred, the rule f1_score uses
    # for the point score too. On the 250k test set every class appears in every
    # resample; on a test set of a few hundred rows a rare class can drop out of
    # some resamples, which widens the interval slightly.
    return f1_score(y_true, y_pred, average="macro")


def material_scores(y_true, y_pred):
    return {"macro_f1": _macro_f1(y_true, y_pred)}


def rating_scores(y_true, y_pred):
    return {"rmse": root_mean_squared_error(y_true, y_pred),
            "mae": mean_absolute_error(y_true, y_pred)}


METRICS = {
    "macro_f1": _macro_f1,
    "rmse": root_mean_squared_error,
    "mae": mean_absolute_error,
}


def _prepare(y_true, *predictions, metric):
    if metric not in METRICS:
        raise ValueError(f"metric must be one of {sorted(METRICS)}, got {metric!r}")
    arrays = [np.asarray(a) for a in (y_true, *predictions)]
    if len({len(a) for a in arrays}) != 1:
        raise ValueError("y_true and every prediction must have the same length")
    if len(arrays[0]) == 0:
        raise ValueError("cannot bootstrap an empty test set")
    if any(pd.isna(a).any() for a in arrays):
        raise ValueError("missing values in y_true or a prediction; drop unlabeled rows first")
    if metric == "macro_f1":
        unique_labels(*arrays)  # sklearn's own check: rejects mixed label types such as 1 and "1"
        # Integer codes make 1,000 f1_score calls about 20 times faster than strings,
        # and give exactly the same macro-F1 (checked in test_evaluate.py).
        _, codes = np.unique(np.concatenate(arrays), return_inverse=True)
        arrays = np.split(codes, len(arrays))
    return arrays


def _resamples(n, n_resamples, seed):
    rng = np.random.default_rng(seed)
    for _ in range(n_resamples):
        yield rng.integers(0, n, size=n)


def bootstrap_ci(y_true, y_pred, metric, n_resamples=N_RESAMPLES, seed=SEED):
    """95% percentile-bootstrap interval for one model's score on the test rows.

    metric is "macro_f1", "rmse", or "mae". Returns (ci_low, ci_high).
    """
    y_true, y_pred = _prepare(y_true, y_pred, metric=metric)
    score = METRICS[metric]
    scores = [score(y_true[i], y_pred[i]) for i in _resamples(len(y_true), n_resamples, seed)]
    low, high = np.percentile(scores, [2.5, 97.5])
    return float(low), float(high)


def bootstrap_diff_ci(y_true, pred_a, pred_b, metric, n_resamples=N_RESAMPLES, seed=SEED):
    """95% interval for score(a) - score(b), resampling the same rows for both models.

    Use it to say whether one model beats another: if the interval excludes 0,
    the gap is larger than test-set sampling noise. Returns (ci_low, ci_high).
    """
    y_true, pred_a, pred_b = _prepare(y_true, pred_a, pred_b, metric=metric)
    score = METRICS[metric]
    diffs = [score(y_true[i], pred_a[i]) - score(y_true[i], pred_b[i])
             for i in _resamples(len(y_true), n_resamples, seed)]
    low, high = np.percentile(diffs, [2.5, 97.5])
    return float(low), float(high)
