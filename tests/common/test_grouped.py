"""Tests of the leakage-aware grouped protocol (code/common/grouped.py)."""
import numpy as np
import pytest

from common import data, grouped, protocol

# Documented tolerance: every class has within TOL_ROWS rows of the ideal share (class total
# divided by 5) in every test fold, and the fold sizes differ by at most TOL_SIZE of their mean.
TOL_ROWS = 8.0
TOL_SIZE = 0.002


@pytest.fixture(scope="module")
def table():
    return data.load_table()


@pytest.fixture(scope="module")
def split(table):
    return grouped.grouped_split(table)


def test_groups_match_numpy_unique(table, split):
    """The hash grouping equals exact grouping of the 42 feature columns (no collision)."""
    matrix = np.concatenate([table.numeric, table.nominal.astype(np.float64)], axis=1)
    _, inverse = np.unique(matrix, axis=0, return_inverse=True)
    inverse = inverse.reshape(-1)
    assert split.groups.max() + 1 == inverse.max() + 1
    # same partition of the rows: the pair (group, exact id) is one-to-one
    pairs = np.unique(np.stack([split.groups, inverse], axis=1), axis=0)
    assert pairs.shape[0] == split.groups.max() + 1


def test_holdout_keeps_groups_whole_and_matches_shared_g(table, split):
    assert np.intersect1d(split.groups_dev, split.groups_gen).size == 0
    assert split.dev_idx.size + split.gen_idx.size == table.n
    assert np.intersect1d(split.dev_idx, split.gen_idx).size == 0
    _, shared_gen = protocol.stratified_holdout(table.y)
    want = data.class_counts(table.y[shared_gen])
    got = data.class_counts(table.y[split.gen_idx])
    assert np.abs(got - want).max() <= 10         # the group structure is balanced as well (typically within 5)
    assert got[7] >= 30                           # class 7 stays usable


def _check_partitions(y, groups, parts, n_per_repeat=5):
    n = y.size
    for p in parts:
        assert np.intersect1d(groups[p.train], groups[p.test]).size == 0
        assert np.intersect1d(p.train, p.test).size == 0
        assert p.train.size + p.test.size == n
    for r in range(len(parts) // n_per_repeat):
        tests = np.concatenate([p.test for p in parts if p.repeat == r])
        assert np.array_equal(np.sort(tests), np.arange(n))      # every row in exactly one test fold


def test_holdout_has_the_group_structure_of_the_development_set(split):
    """G_g is not a collection of the largest groups: its group sizes resemble those of D_g."""
    def shares(groups):
        size = np.unique(groups, return_inverse=True, return_counts=True)
        size = size[2][size[1].reshape(-1)]
        return np.array([(size == 1).mean(), (size >= 10).mean()])
    assert np.abs(shares(split.groups_dev) - shares(split.groups_gen)).max() < 0.02


def test_tuning_partitions(table, split):
    y = table.y[split.dev_idx]
    parts = grouped.grouped_tuning_partitions(y, split.groups_dev)
    assert len(parts) == 5
    _check_partitions(y, split.groups_dev, parts)
    assert grouped.twin_share(split.groups_dev, parts[0]) == 0.0


def test_comparison_partitions(table, split):
    y = table.y[split.dev_idx]
    parts = grouped.grouped_comparison_partitions(y, split.groups_dev)
    assert len(parts) == 15
    _check_partitions(y, split.groups_dev, parts)
    assert [p.repeat for p in parts] == [r for r in range(3) for _ in range(5)]
    # the repeats differ from each other
    assert not np.array_equal(parts[0].test, parts[5].test)
    assert not np.array_equal(parts[5].test, parts[10].test)


def test_determinism(table, split):
    y = table.y[split.dev_idx]
    a = grouped.grouped_comparison_partitions(y, split.groups_dev)
    b = grouped.grouped_comparison_partitions(y, split.groups_dev)
    assert all(np.array_equal(p.test, q.test) for p, q in zip(a, b))
    d1, g1 = grouped.grouped_holdout(table)
    assert np.array_equal(d1, split.dev_idx) and np.array_equal(g1, split.gen_idx)


def test_class_balance_within_tolerance(table, split):
    y = table.y[split.dev_idx]
    for parts in (grouped.grouped_tuning_partitions(y, split.groups_dev),
                  grouped.grouped_comparison_partitions(y, split.groups_dev)):
        counts = grouped.balance_table(y, parts)
        ideal = data.class_counts(y) / 5
        assert np.abs(counts - ideal[None, :]).max() <= TOL_ROWS
        sizes = counts.sum(axis=1)
        assert (sizes.max() - sizes.min()) / sizes.mean() <= TOL_SIZE
        assert counts[:, 7].min() >= 25            # class 7 present in every test fold


def test_conflicting_groups_stay_whole(table, split):
    y = table.y[split.dev_idx]
    groups = split.groups_dev
    summary = grouped.group_summary(groups, y)
    assert summary["n_conflicting_groups"] > 0
    parts = grouped.grouped_tuning_partitions(y, groups)
    for p in parts:
        assert grouped.groups_cross(groups, p) == 0


def test_assign_groups_small_example():
    groups = np.array([0, 0, 1, 2, 2, 2, 3, 4, 5, 5])
    y = np.array([0, 0, 1, 1, 1, 1, 0, 0, 1, 1])
    gen = protocol.rng("test-grouped-small")
    bins = grouped.assign_groups(groups, y, [0.5, 0.5], gen)
    for g in np.unique(groups):
        assert np.unique(bins[groups == g]).size == 1
    assert np.unique(bins).size == 2


def test_twin_share_primary_protocol_is_high(table):
    """The primary protocol has many twins; the grouped one has none."""
    dev_idx, _ = protocol.stratified_holdout(table.y)
    groups = grouped.feature_groups(table)[dev_idx]
    part = protocol.tuning_partitions(table.y[dev_idx])[0]
    assert grouped.twin_share(groups, part) > 0.3
