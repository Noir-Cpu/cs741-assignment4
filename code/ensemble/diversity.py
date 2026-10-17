"""Measures of member strength and diversity for an ensemble of classifiers.

Inputs are the labels predicted by every member for every row, a matrix of shape
(rows, members), and the true labels. Measures follow the oracle-output view of
Kuncheva and Whitaker: member i is correct on a row when its label equals the
true label.

* member_accuracy: mean accuracy of one member (strength).
* mean_disagreement: probability that two members drawn at random without
  replacement predict different labels on a row, averaged over rows. It is
  computed exactly from the vote counts: a row with vote counts v_k has
  sum_k C(v_k, 2) agreeing pairs out of C(T, 2) pairs.
* mean_q_statistic: average pairwise Yule Q statistic of the oracle outputs,
  Q = (N11 N00 - N01 N10) / (N11 N00 + N01 N10), over all pairs of members;
  a pair with a zero denominator is left out of the mean.
* error_correlation: average pairwise Pearson correlation of the oracle outputs.
"""
from __future__ import annotations

import numpy as np


def vote_counts(members: np.ndarray, n_classes: int) -> np.ndarray:
    """Votes per class for every row, shape (rows, n_classes)."""
    rows = members.shape[0]
    out = np.zeros((rows, n_classes), dtype=np.int64)
    for k in range(n_classes):
        out[:, k] = (members == k).sum(axis=1)
    return out


def member_accuracy(members: np.ndarray, y: np.ndarray) -> float:
    return float((members == np.asarray(y)[:, None]).mean())


def mean_disagreement(members: np.ndarray, n_classes: int) -> float:
    t = members.shape[1]
    if t < 2:
        return 0.0
    v = vote_counts(members, n_classes)
    agree = (v * (v - 1) // 2).sum(axis=1)
    return float(1.0 - agree.mean() / (t * (t - 1) / 2.0))


def _oracle(members: np.ndarray, y: np.ndarray, max_members: int | None) -> np.ndarray:
    if max_members is not None and members.shape[1] > max_members:
        members = members[:, :max_members]
    return (members == np.asarray(y)[:, None]).astype(np.float64)


def mean_q_statistic(members: np.ndarray, y: np.ndarray, max_members: int | None = 200) -> float:
    o = _oracle(members, y, max_members)
    t = o.shape[1]
    if t < 2:
        return float("nan")
    n11 = o.T @ o
    n10 = o.T @ (1 - o)
    n01 = n10.T
    n00 = (1 - o).T @ (1 - o)
    num = n11 * n00 - n01 * n10
    den = n11 * n00 + n01 * n10
    iu = np.triu_indices(t, 1)
    num, den = num[iu], den[iu]
    ok = den > 0
    return float((num[ok] / den[ok]).mean()) if ok.any() else float("nan")


def error_correlation(members: np.ndarray, y: np.ndarray, max_members: int | None = 200) -> float:
    o = _oracle(members, y, max_members)
    t = o.shape[1]
    if t < 2:
        return float("nan")
    std = o.std(axis=0)
    ok = std > 0
    if ok.sum() < 2:
        return float("nan")
    c = np.corrcoef(o[:, ok], rowvar=False)
    iu = np.triu_indices(c.shape[0], 1)
    return float(c[iu].mean())


def summary(members: np.ndarray, y: np.ndarray, n_classes: int) -> dict:
    return {"member_accuracy": member_accuracy(members, y),
            "disagreement": mean_disagreement(members, n_classes),
            "q_statistic": mean_q_statistic(members, y),
            "error_correlation": error_correlation(members, y)}
