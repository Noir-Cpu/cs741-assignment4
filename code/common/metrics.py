"""Classification performance measures from the confusion matrix.

Conventions (identical for every model):

* Labels are the integers 0 to n_classes - 1; the label set is fixed, so a
  class that never occurs in the predictions still contributes to macro
  averages.
* Zero-division convention: a precision, recall or F1 whose denominator is
  zero scores 0.
* Macro averages are unweighted means over all n_classes classes. Weighted
  averages weight each class by its support (true count).
* Balanced accuracy is the mean recall over the classes with support > 0.
* The confusion matrix has rows for the true class and columns for the
  predicted class.
"""
from __future__ import annotations

import numpy as np

from .data import CLASS_NAMES

MEASURES = ("accuracy", "macro_f1", "weighted_f1", "balanced_accuracy")


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    """Confusion matrix (rows true, columns predicted) of int64 counts."""
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    flat = y_true * n_classes + y_pred
    return np.bincount(flat, minlength=n_classes * n_classes).reshape(n_classes, n_classes)


def _safe_divide(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    num = np.asarray(num, dtype=np.float64)
    den = np.asarray(den, dtype=np.float64)
    out = np.zeros(np.broadcast(num, den).shape)
    np.divide(num, den, out=out, where=den > 0)
    return out


def per_class(cm: np.ndarray) -> dict:
    """Per-class precision, recall, F1 and support from a confusion matrix."""
    tp = np.diag(cm).astype(np.float64)
    predicted = cm.sum(axis=0)
    support = cm.sum(axis=1)
    precision = _safe_divide(tp, predicted)
    recall = _safe_divide(tp, support)
    f1 = _safe_divide(2 * precision * recall, precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1, "support": support}


def scores_from_confusion(cm: np.ndarray) -> dict:
    """Accuracy, macro F1, weighted F1 and balanced accuracy."""
    pc = per_class(cm)
    total = cm.sum()
    support = pc["support"]
    present = support > 0
    return {
        "accuracy": float(np.trace(cm) / total) if total > 0 else 0.0,
        "macro_f1": float(pc["f1"].mean()),
        "weighted_f1": float((pc["f1"] * support).sum() / total) if total > 0 else 0.0,
        "balanced_accuracy": float(pc["recall"][present].mean()) if present.any() else 0.0,
    }


def score_all(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> dict:
    """The four headline measures for a prediction vector."""
    return scores_from_confusion(confusion_matrix(y_true, y_pred, n_classes))


def accuracy(y_true, y_pred) -> float:
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


def macro_f1(y_true, y_pred, n_classes: int) -> float:
    return score_all(y_true, y_pred, n_classes)["macro_f1"]


def balanced_accuracy(y_true, y_pred, n_classes: int) -> float:
    return score_all(y_true, y_pred, n_classes)["balanced_accuracy"]


def per_class_table(cm: np.ndarray, names=CLASS_NAMES) -> list[dict]:
    """Rows of per-class precision, recall, F1 and support, ready for a CSV."""
    pc = per_class(cm)
    rows = []
    for c in range(cm.shape[0]):
        rows.append({"class": c, "name": names[c] if c < len(names) else str(c),
                     "precision": float(pc["precision"][c]), "recall": float(pc["recall"][c]),
                     "f1": float(pc["f1"][c]), "support": int(pc["support"][c])})
    return rows


def normalise_rows(cm: np.ndarray) -> np.ndarray:
    """Confusion matrix with each row divided by its sum (recall on the diagonal)."""
    return _safe_divide(cm, cm.sum(axis=1, keepdims=True))
