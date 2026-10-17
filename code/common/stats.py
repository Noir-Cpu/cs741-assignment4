"""Statistical tests for comparing models on paired cross-validation scores.

The paired scores of two models come from the same partitions, so every test
here is paired. The 15 scores of 5-fold cross-validation repeated 3 times are
not independent (training sets overlap), so p-values are optimistic. The
corrected resampled t-test of Nadeau and Bengio inflates the variance to
account for the overlap and is provided as a conservative companion.

Functions
---------
paired_wilcoxon              two models: Wilcoxon signed-rank test plus effect sizes
corrected_resampled_ttest    two models: variance-corrected paired t-test
friedman_test                more than two models: omnibus test and average ranks
nemenyi_posthoc              post hoc: Nemenyi critical difference and p-values
holm_wilcoxon_posthoc        post hoc: pairwise Wilcoxon with Holm correction
holm_adjust                  Holm step-down adjustment of a p-value vector
summarise                    mean and sample standard deviation
"""
from __future__ import annotations

import itertools

import numpy as np
from scipy import stats as sps


def summarise(x) -> dict:
    """Mean and sample standard deviation (ddof = 1) of a score vector."""
    x = np.asarray(x, dtype=np.float64)
    return {"mean": float(x.mean()), "std": float(x.std(ddof=1)) if x.size > 1 else 0.0,
            "min": float(x.min()), "max": float(x.max()), "n": int(x.size)}


def rank_biserial(diff) -> float:
    """Matched-pairs rank-biserial correlation in [-1, 1].

    Equal to (sum of ranks of positive differences - sum of ranks of negative
    differences) divided by the total rank sum, with zeros dropped.
    """
    d = np.asarray(diff, dtype=np.float64)
    d = d[d != 0]
    if d.size == 0:
        return 0.0
    ranks = sps.rankdata(np.abs(d))
    return float((ranks[d > 0].sum() - ranks[d < 0].sum()) / ranks.sum())


def paired_wilcoxon(a, b, alternative: str = "two-sided") -> dict:
    """Wilcoxon signed-rank test of a - b, with the effect sizes of the differences.

    Returns the statistic, the p-value, the mean and median difference, the
    paired Cohen's d (mean difference divided by the standard deviation of the
    differences) and the rank-biserial correlation. When every difference is
    zero the p-value is 1.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    diff = a - b
    if np.all(diff == 0):
        stat, p = 0.0, 1.0
    else:
        res = sps.wilcoxon(a, b, alternative=alternative)
        stat, p = float(res.statistic), float(res.pvalue)
    sd = diff.std(ddof=1) if diff.size > 1 else 0.0
    return {"statistic": stat, "p_value": p, "mean_diff": float(diff.mean()),
            "median_diff": float(np.median(diff)),
            "cohens_dz": float(diff.mean() / sd) if sd > 0 else float("nan"),
            "rank_biserial": rank_biserial(diff), "n": int(diff.size),
            "n_a_higher": int((diff > 0).sum()), "n_b_higher": int((diff < 0).sum())}


def corrected_resampled_ttest(a, b, test_train_ratio: float = 0.25) -> dict:
    """Paired t-test with the variance correction of Nadeau and Bengio.

    The variance of the mean difference over J resampled partitions is
    (1/J + n_test/n_train) * s^2 instead of s^2 / J. For 5-fold cross-validation
    n_test / n_train = 1/4.
    """
    diff = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    j = diff.size
    var = diff.var(ddof=1)
    se = np.sqrt((1.0 / j + test_train_ratio) * var)
    if se == 0:
        return {"t": float("nan"), "p_value": 1.0 if diff.mean() == 0 else 0.0,
                "mean_diff": float(diff.mean()), "df": j - 1}
    t = diff.mean() / se
    return {"t": float(t), "p_value": float(2 * sps.t.sf(abs(t), j - 1)),
            "mean_diff": float(diff.mean()), "df": j - 1}


def holm_adjust(p_values) -> np.ndarray:
    """Holm step-down adjusted p-values, in the order of the input."""
    p = np.asarray(p_values, dtype=np.float64)
    m = p.size
    order = np.argsort(p)
    adjusted = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adjusted[i] = min(1.0, running)
    return adjusted


def average_ranks(scores, higher_is_better: bool = True) -> np.ndarray:
    """Average rank of each model (rank 1 is best) over the blocks (rows).

    scores has shape (n_blocks, n_models). Ties receive the mean rank.
    """
    s = np.asarray(scores, dtype=np.float64)
    ranks = np.vstack([sps.rankdata(-row if higher_is_better else row) for row in s])
    return ranks.mean(axis=0)


def friedman_test(scores, higher_is_better: bool = True) -> dict:
    """Friedman omnibus test over blocks (rows) and models (columns).

    Needs at least three models. Returns the chi-square statistic with its
    p-value (conservative for few blocks), the Iman-Davenport F statistic with
    its p-value (the variant recommended for small numbers of blocks) and the
    average ranks (rank 1 is best).
    """
    s = np.asarray(scores, dtype=np.float64)
    if s.shape[1] < 3:
        raise ValueError("the Friedman test needs at least three models")
    res = sps.friedmanchisquare(*[s[:, j] for j in range(s.shape[1])])
    n_blocks, k = s.shape
    chi2 = float(res.statistic)
    denominator = n_blocks * (k - 1) - chi2
    if denominator <= 0:
        f_stat, f_p = float("inf"), 0.0
    else:
        f_stat = (n_blocks - 1) * chi2 / denominator
        f_p = float(sps.f.sf(f_stat, k - 1, (k - 1) * (n_blocks - 1)))
    return {"statistic": chi2, "p_value": float(res.pvalue), "iman_davenport_f": float(f_stat),
            "iman_davenport_p": f_p, "average_ranks": average_ranks(s, higher_is_better),
            "n_blocks": int(n_blocks)}


def nemenyi_posthoc(scores, alpha: float = 0.05, higher_is_better: bool = True) -> dict:
    """Nemenyi post hoc test after a Friedman test.

    Two models differ significantly when their average ranks differ by more
    than the critical difference q_alpha / sqrt(2) * sqrt(k (k + 1) / (6 N)),
    where q_alpha is the upper critical value of the studentised range for k
    models with infinite degrees of freedom and N is the number of blocks.
    Returns the critical difference, the average ranks and a matrix of
    pairwise p-values.
    """
    s = np.asarray(scores, dtype=np.float64)
    n, k = s.shape
    ranks = average_ranks(s, higher_is_better)
    se = np.sqrt(k * (k + 1) / (6.0 * n))
    q_alpha = float(sps.studentized_range.ppf(1 - alpha, k, np.inf))
    p = np.ones((k, k))
    for i, j in itertools.combinations(range(k), 2):
        q = abs(ranks[i] - ranks[j]) / se * np.sqrt(2.0)
        p[i, j] = p[j, i] = float(sps.studentized_range.sf(q, k, np.inf))
    return {"critical_difference": q_alpha / np.sqrt(2.0) * se, "average_ranks": ranks,
            "p_values": p, "alpha": alpha}


def holm_wilcoxon_posthoc(scores, names=None, alternative: str = "two-sided") -> list[dict]:
    """All pairwise Wilcoxon signed-rank tests with Holm-adjusted p-values."""
    s = np.asarray(scores, dtype=np.float64)
    k = s.shape[1]
    names = list(names) if names is not None else [str(j) for j in range(k)]
    rows = []
    for i, j in itertools.combinations(range(k), 2):
        res = paired_wilcoxon(s[:, i], s[:, j], alternative)
        res.update({"model_a": names[i], "model_b": names[j]})
        rows.append(res)
    adjusted = holm_adjust([r["p_value"] for r in rows])
    for r, p_adj in zip(rows, adjusted):
        r["p_holm"] = float(p_adj)
    return rows
