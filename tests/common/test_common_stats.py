"""Tests for common.stats against hand-computed values and scipy."""
import numpy as np
import pytest
from scipy import stats as sps

from common import stats


def test_summarise():
    s = stats.summarise([1.0, 2.0, 3.0, 4.0])
    assert s["mean"] == 2.5 and s["std"] == pytest.approx(np.std([1, 2, 3, 4], ddof=1))
    assert s["n"] == 4


def test_rank_biserial_hand_example():
    # |d| ranks: 0.5 -> 1, 1 -> 2, 2 -> 3, 3 -> 4; positives 9, negatives 1.
    assert stats.rank_biserial([1.0, 2.0, 3.0, -0.5]) == pytest.approx(0.8)
    assert stats.rank_biserial([0.0, 0.0]) == 0.0


def test_wilcoxon_matches_scipy_and_reports_effects():
    gen = np.random.default_rng(1072417991)
    a = gen.normal(0.6, 0.02, 15)
    b = a - gen.normal(0.03, 0.01, 15)
    res = stats.paired_wilcoxon(a, b)
    ref = sps.wilcoxon(a, b)
    assert res["p_value"] == pytest.approx(ref.pvalue)
    assert res["statistic"] == pytest.approx(ref.statistic)
    assert res["mean_diff"] == pytest.approx((a - b).mean())
    assert res["n_a_higher"] + res["n_b_higher"] == 15
    assert res["rank_biserial"] == pytest.approx(1.0)


def test_wilcoxon_smallest_two_sided_p_for_fifteen_pairs():
    a = np.arange(15) + 10.0
    res = stats.paired_wilcoxon(a, a - 1.0 - np.arange(15) * 1e-3)
    assert res["p_value"] == pytest.approx(2 / 2 ** 15)


def test_wilcoxon_identical_scores():
    res = stats.paired_wilcoxon([1.0, 2.0], [1.0, 2.0])
    assert res["p_value"] == 1.0 and res["mean_diff"] == 0.0


def test_corrected_resampled_ttest_hand_example():
    a = np.array([0.01, 0.02, 0.03, 0.04, 0.05])
    res = stats.corrected_resampled_ttest(a, np.zeros(5), 0.25)
    se = np.sqrt((1 / 5 + 0.25) * np.var(a, ddof=1))
    assert res["t"] == pytest.approx(0.03 / se)
    assert res["p_value"] == pytest.approx(2 * sps.t.sf(0.03 / se, 4))
    plain = sps.ttest_rel(a, np.zeros(5))
    assert res["p_value"] > plain.pvalue          # the correction is conservative


def test_holm_adjust_hand_example():
    adj = stats.holm_adjust([0.01, 0.04, 0.03])
    assert np.allclose(adj, [0.03, 0.06, 0.06])
    assert np.allclose(stats.holm_adjust([0.2, 0.9]), [0.4, 0.9])


def test_friedman_perfect_ordering():
    n = 10
    base = np.arange(n)[:, None] * 0.001
    scores = base + np.array([0.3, 0.2, 0.1])
    res = stats.friedman_test(scores)
    assert np.allclose(res["average_ranks"], [1.0, 2.0, 3.0])
    assert res["statistic"] == pytest.approx(20.0)      # 12 N / (k (k + 1)) * sum (R - 2)^2
    assert res["p_value"] == pytest.approx(sps.chi2.sf(20.0, 2))
    assert res["iman_davenport_f"] == float("inf") and res["iman_davenport_p"] == 0.0
    mixed = np.array([[3.0, 2.0, 1.0], [3.0, 1.0, 2.0], [2.0, 3.0, 1.0], [3.0, 2.0, 1.0]])
    chi2 = stats.friedman_test(mixed)["statistic"]            # ranks 1.5-ish, hand value below
    f_hand = (4 - 1) * chi2 / (4 * 2 - chi2)
    got = stats.friedman_test(mixed)
    assert got["iman_davenport_f"] == pytest.approx(f_hand)
    assert got["iman_davenport_p"] == pytest.approx(sps.f.sf(f_hand, 2, 6))
    low = stats.friedman_test(scores, higher_is_better=False)
    assert np.allclose(low["average_ranks"], [3.0, 2.0, 1.0])
    with pytest.raises(ValueError):
        stats.friedman_test(scores[:, :2])


def test_nemenyi_critical_difference_matches_demsar_table():
    scores = np.random.default_rng(1072417991).random((10, 3))
    res = stats.nemenyi_posthoc(scores)
    # Demsar: q_0.05 = 2.343 for k = 3 (already divided by sqrt 2); CD = 2.343 * sqrt(k (k + 1) / (6 N))
    assert res["critical_difference"] == pytest.approx(2.343 * np.sqrt(12 / 60), abs=2e-3)
    p = res["p_values"]
    assert np.allclose(p, p.T) and np.allclose(np.diag(p), 1.0)


def test_nemenyi_p_value_crosses_alpha_at_the_critical_difference():
    n, k = 12, 4
    gen = np.random.default_rng(1072417991)
    scores = gen.random((n, k)) + np.arange(k) * 0.6
    res = stats.nemenyi_posthoc(scores)
    ranks, cd, p = res["average_ranks"], res["critical_difference"], res["p_values"]
    for i in range(k):
        for j in range(i + 1, k):
            assert (abs(ranks[i] - ranks[j]) > cd) == (p[i, j] < 0.05)


def test_holm_wilcoxon_posthoc_lists_all_pairs():
    gen = np.random.default_rng(1072417991)
    scores = gen.normal(size=(15, 3)) * 0.01 + np.array([0.5, 0.52, 0.7])
    rows = stats.holm_wilcoxon_posthoc(scores, names=["a", "b", "c"])
    assert len(rows) == 3
    assert all(r["p_holm"] >= r["p_value"] for r in rows)
    assert {(r["model_a"], r["model_b"]) for r in rows} == {("a", "b"), ("a", "c"), ("b", "c")}
