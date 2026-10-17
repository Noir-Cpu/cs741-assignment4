"""Tests for common.metrics against hand-computed values and scikit-learn."""
import numpy as np
import pytest
from sklearn import metrics as skm

from common import metrics


def test_hand_computed_three_class_example():
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 1, 1, 1, 2, 0])
    cm = metrics.confusion_matrix(y_true, y_pred, 3)
    assert np.array_equal(cm, [[1, 1, 0], [0, 2, 0], [1, 0, 1]])
    pc = metrics.per_class(cm)
    assert np.allclose(pc["precision"], [0.5, 2 / 3, 1.0])
    assert np.allclose(pc["recall"], [0.5, 1.0, 0.5])
    assert np.allclose(pc["f1"], [0.5, 0.8, 2 / 3])
    s = metrics.scores_from_confusion(cm)
    assert s["accuracy"] == pytest.approx(4 / 6)
    assert s["macro_f1"] == pytest.approx((0.5 + 0.8 + 2 / 3) / 3)
    assert s["balanced_accuracy"] == pytest.approx(2 / 3)
    assert s["weighted_f1"] == pytest.approx((0.5 + 0.8 + 2 / 3) / 3)   # equal supports


def test_zero_division_scores_zero_and_class_stays_in_macro_average():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 0, 0])      # class 1 never predicted, class 2 absent
    cm = metrics.confusion_matrix(y_true, y_pred, 3)
    pc = metrics.per_class(cm)
    assert pc["precision"][1] == 0.0 and pc["f1"][1] == 0.0 and pc["f1"][2] == 0.0
    s = metrics.scores_from_confusion(cm)
    f1_class0 = 2 * 0.5 * 1.0 / 1.5
    assert s["macro_f1"] == pytest.approx(f1_class0 / 3)
    assert s["balanced_accuracy"] == pytest.approx(0.5)     # class 2 has no support


def test_matches_scikit_learn_on_random_predictions():
    gen = np.random.default_rng(1072417991)
    y_true = gen.integers(0, 6, 2000)
    y_pred = np.where(gen.random(2000) < 0.6, y_true, gen.integers(0, 6, 2000))
    s = metrics.score_all(y_true, y_pred, 6)
    labels = list(range(6))
    assert s["accuracy"] == pytest.approx(skm.accuracy_score(y_true, y_pred))
    assert s["macro_f1"] == pytest.approx(skm.f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))
    assert s["weighted_f1"] == pytest.approx(skm.f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0))
    assert s["balanced_accuracy"] == pytest.approx(skm.balanced_accuracy_score(y_true, y_pred))
    cm = metrics.confusion_matrix(y_true, y_pred, 6)
    assert np.array_equal(cm, skm.confusion_matrix(y_true, y_pred, labels=labels))
    p, r, f, sup = skm.precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    pc = metrics.per_class(cm)
    assert np.allclose(pc["precision"], p) and np.allclose(pc["recall"], r)
    assert np.allclose(pc["f1"], f) and np.array_equal(pc["support"], sup)


def test_table_rows_and_row_normalisation():
    cm = np.array([[3, 1], [0, 4]])
    rows = metrics.per_class_table(cm, names=("a", "b"))
    assert rows[0]["name"] == "a" and rows[1]["support"] == 4
    assert np.allclose(metrics.normalise_rows(cm), [[0.75, 0.25], [0.0, 1.0]])
    assert np.allclose(metrics.normalise_rows(np.zeros((2, 2))), 0.0)
