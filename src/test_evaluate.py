"""Tests for src/evaluate.py. Run from the repo root: python -m src.test_evaluate"""
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score

from src.evaluate import (bootstrap_ci, bootstrap_diff_ci, material_per_class, material_scores,
                          rating_scores)

RNG = np.random.default_rng(0)
CLASSES = np.array(["cotton", "polyester", "leather", "wool"])
Y_TRUE = RNG.choice(CLASSES, size=2000, p=[0.4, 0.4, 0.15, 0.05])
Y_PRED = np.where(RNG.random(2000) < 0.7, Y_TRUE, RNG.choice(CLASSES, size=2000))
R_TRUE = RNG.uniform(1, 5, size=2000)
R_PRED = R_TRUE + RNG.normal(0, 0.5, size=2000)


def test_interval_brackets_point_score():
    point = material_scores(Y_TRUE, Y_PRED)["macro_f1"]
    low, high = bootstrap_ci(Y_TRUE, Y_PRED, "macro_f1")
    assert low < point < high
    for metric in ("rmse", "mae"):
        point = rating_scores(R_TRUE, R_PRED)[metric]
        low, high = bootstrap_ci(R_TRUE, R_PRED, metric)
        assert low < point < high


def test_same_seed_same_interval():
    assert bootstrap_ci(Y_TRUE, Y_PRED, "macro_f1") == bootstrap_ci(Y_TRUE, Y_PRED, "macro_f1")
    assert bootstrap_ci(R_TRUE, R_PRED, "rmse", seed=1) != bootstrap_ci(R_TRUE, R_PRED, "rmse", seed=2)


def test_perfect_predictions_give_a_point_interval():
    assert bootstrap_ci(Y_TRUE, Y_TRUE, "macro_f1", n_resamples=50) == (1.0, 1.0)
    assert bootstrap_ci(R_TRUE, R_TRUE, "mae", n_resamples=50) == (0.0, 0.0)


def test_accepts_lists_and_pandas_series_paired_by_position():
    from_arrays = bootstrap_ci(Y_TRUE, Y_PRED, "macro_f1", n_resamples=50)
    assert bootstrap_ci(list(Y_TRUE), list(Y_PRED), "macro_f1", n_resamples=50) == from_arrays
    shuffled = RNG.permutation(len(Y_TRUE))
    series = (pd.Series(Y_TRUE, index=shuffled), pd.Series(Y_PRED, index=shuffled[::-1]))
    assert bootstrap_ci(*series, "macro_f1", n_resamples=50) == from_arrays


def test_integer_codes_give_the_same_macro_f1_as_strings():
    # The speed-up in _prepare scores integer codes; it must not change any score.
    _, codes = np.unique(np.concatenate([Y_TRUE, Y_PRED]), return_inverse=True)
    t, p = np.split(codes, 2)
    for seed in range(5):
        i = np.random.default_rng(seed).integers(0, len(Y_TRUE), size=len(Y_TRUE))
        assert f1_score(t[i], p[i], average="macro") == f1_score(Y_TRUE[i], Y_PRED[i], average="macro")


def test_material_scores_include_accuracy():
    scores = material_scores(Y_TRUE, Y_PRED)
    assert scores["accuracy"] == accuracy_score(Y_TRUE, Y_PRED)
    assert scores["macro_f1"] == f1_score(Y_TRUE, Y_PRED, average="macro")


def test_per_class_matches_classification_report():
    labels = sorted(CLASSES) + ["silk"]  # a class that is never predicted or present
    table = material_per_class(Y_TRUE, Y_PRED, labels)
    report = classification_report(Y_TRUE, Y_PRED, labels=labels, output_dict=True, zero_division=0)
    assert list(table.material) == labels
    for row in table.itertuples():
        expected = report[row.material]
        assert (row.precision, row.recall, row.f1, row.support) == (
            expected["precision"], expected["recall"], expected["f1-score"], expected["support"])


def test_diff_of_a_model_with_itself_is_zero():
    assert bootstrap_diff_ci(R_TRUE, R_PRED, R_PRED, "rmse", n_resamples=50) == (0.0, 0.0)


def test_diff_detects_a_clearly_better_model():
    worse = R_TRUE + RNG.normal(0, 1.0, size=2000)
    low, high = bootstrap_diff_ci(R_TRUE, R_PRED, worse, "rmse", n_resamples=200)
    assert high < 0  # a has the lower RMSE on every resample


def test_rejects_bad_input():
    with_nan = Y_TRUE.astype(object)
    with_nan[0] = np.nan
    for bad in (lambda: bootstrap_ci(Y_TRUE, Y_PRED, "accuracy"),
                lambda: bootstrap_ci(Y_TRUE, Y_PRED[:10], "macro_f1"),
                lambda: bootstrap_ci([], [], "rmse"),
                lambda: bootstrap_ci(with_nan, Y_PRED, "macro_f1"),       # unlabeled row left in
                lambda: bootstrap_ci([1, "1", 2], [1, 2, 2], "macro_f1")):  # mixed label types
        try:
            bad()
        except ValueError:
            continue
        raise AssertionError("expected ValueError")


if __name__ == "__main__":
    tests = [f for name, f in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
    print(f"{len(tests)} tests passed")
