"""Leakage-aware (grouped) evaluation protocol: identical feature vectors stay together.

Phase 1 found that 39.0 percent of the development set are repeated feature
vectors and that 44 percent of the rows of a cross-validation fold have an exact
twin (an identical descriptive feature vector) in the training part. The
Assignment 2 compatible protocol (common.protocol) is kept as the PRIMARY
protocol. This module adds a second protocol in which no test row has a twin in
the training part, so that models that memorise are not rewarded.

Groups
------
Two rows belong to the same group when all 42 descriptive features are
identical: the 39 numeric columns (float64) and the 3 nominal columns (integer
codes of the strings proto, service, state). The identifier and the target are
excluded. The 42 values of a row are written as 42 float64 numbers (negative
zero is normalised to zero), the 336 bytes are hashed with BLAKE2b (128-bit
digest) and rows with equal digests form a group. Group numbers follow the first
occurrence of the digest in the table, so they do not depend on the hash
function. The tests check against numpy.unique(axis=0) that no collision
occurs. Groups with conflicting labels (same vector, different attack category)
stay whole like every other group.

Partitions
----------
`grouped_holdout` splits the table into a development set D_g and a
generalisation set G_g, `grouped_tuning_partitions` returns 5 folds inside D_g,
`grouped_comparison_partitions` returns 5 folds repeated 3 times (15
partitions). Every partition keeps every group whole. The returned objects are
the same `protocol.Partition` objects as in the primary protocol, so existing
code runs unchanged.

Algorithm (greedy stratification under group constraints)
---------------------------------------------------------
A group is an indivisible item with a vector m of per-class row counts (more
than one non-zero entry for a conflicting group). The K bins (folds, or D_g and
G_g) have target totals T_b,c for every class c, for the number of rows, and for
five structure columns (rows in groups of size 1, 2 to 9, 10 to 49, 50 or more, and
rows in groups with conflicting labels), so that bin sizes and the group structure
balance as well. The groups are processed
in this order: first by their rarest class (the class with the smallest total
among the classes present in the group, rarest first), then by size class
(floor(log2 of the group size), larger first), then at random inside the same
rarest class and size class. The random order is the only source of variation
between repeats and comes from a named stream of common.protocol. Each group
goes to the bin that minimises the increase of

    sum_c ((a_b,c + m_c - T_b,c)^2 - (a_b,c - T_b,c)^2) / T_b,c^2

where a is the content already assigned to the bin; the squared deviation is
relative, so a rare class counts as much as a large one. Ties are broken by a
tiny random number. A local search then visits the groups in random order and
moves a group to another bin whenever the move lowers the same total cost
(squared relative deviation of all bins), until a pass makes no move (at most
20 passes). Fold labels are finally permuted at random. Large groups are
placed while the bins are empty and the many singleton groups fill the
remaining gaps, which gives a close match. `balance_table` and
`class_balance_report` measure the achieved balance.
"""
from __future__ import annotations

import hashlib

import numpy as np
from numba import njit

from . import data, protocol

N_CLASSES = data.N_CLASSES

PURPOSE_GROUPED_SPLIT = "grouped-generalisation-split"
PURPOSE_GROUPED_TUNING = "grouped-tuning-cv"
PURPOSE_GROUPED_COMPARISON = "grouped-comparison-cv"


# ---------------------------------------------------------------------------
# Groups of identical feature vectors
# ---------------------------------------------------------------------------

def feature_matrix(table: data.Table) -> np.ndarray:
    """The 42 descriptive features as one float64 matrix (n, 42): 39 numeric, then 3 nominal codes."""
    matrix = np.concatenate([table.numeric, table.nominal.astype(np.float64)], axis=1)
    return np.ascontiguousarray(matrix + 0.0)          # + 0.0 turns -0.0 into 0.0


def row_digests(table: data.Table) -> list[bytes]:
    """128-bit BLAKE2b digest of the 336 bytes of every row of `feature_matrix`."""
    matrix = feature_matrix(table)
    return [hashlib.blake2b(row.tobytes(), digest_size=16).digest() for row in matrix]


def feature_groups(table: data.Table) -> np.ndarray:
    """Group number (0 to n_groups - 1) of every row; equal feature vectors share a number.

    Numbers follow the first occurrence in the table. The result for the full
    table is subset with `groups[idx]` for development or hold-out rows; the
    numbers of such a subset are not contiguous, which every function here accepts.
    """
    seen: dict[bytes, int] = {}
    out = np.empty(table.n, dtype=np.int64)
    for i, digest in enumerate(row_digests(table)):
        out[i] = seen.setdefault(digest, len(seen))
    return out


def group_summary(groups: np.ndarray, y: np.ndarray) -> dict:
    """Counts of groups, duplicates and conflicting groups for a set of rows."""
    inverse = np.unique(groups, return_inverse=True)[1].reshape(-1)
    n_groups = int(inverse.max()) + 1
    size = np.bincount(inverse, minlength=n_groups)
    labels = np.zeros((n_groups, N_CLASSES), dtype=bool)
    labels[inverse, y] = True
    conflict = labels.sum(axis=1) > 1
    return {"n_rows": int(y.size), "n_groups": n_groups, "n_excess_rows": int(y.size - n_groups),
            "share_rows_in_repeated_groups": float(size[size > 1].sum() / y.size),
            "n_conflicting_groups": int(conflict.sum()),
            "share_rows_in_conflicting_groups": float(size[conflict].sum() / y.size),
            "largest_group": int(size.max())}


# ---------------------------------------------------------------------------
# Greedy stratified assignment of groups to bins
# ---------------------------------------------------------------------------

@njit(cache=True)
def _greedy(items, order, targets, noise):
    """Assign items (n_groups, n_cols) to bins in the given order; returns the bin per group."""
    n_bins = targets.shape[0]
    n_cols = items.shape[1]
    filled = np.zeros((n_bins, n_cols))
    assigned = np.empty(items.shape[0], dtype=np.int64)
    for pos in range(order.shape[0]):
        g = order[pos]
        best = 0
        best_cost = np.inf
        for b in range(n_bins):
            cost = noise[g, b]
            for c in range(n_cols):
                m = items[g, c]
                if m > 0.0:
                    t = targets[b, c]
                    cost += (m * m + 2.0 * m * (filled[b, c] - t)) / (t * t)
            if cost < best_cost:
                best_cost = cost
                best = b
        assigned[g] = best
        for c in range(n_cols):
            filled[best, c] += items[g, c]
    return assigned


@njit(cache=True)
def _refine(items, assigned, targets, order, max_passes):
    """Local search: move single groups between bins while the squared relative deviation falls."""
    n_bins = targets.shape[0]
    n_cols = items.shape[1]
    filled = np.zeros((n_bins, n_cols))
    for g in range(items.shape[0]):
        for c in range(n_cols):
            filled[assigned[g], c] += items[g, c]
    for _ in range(max_passes):
        moved = 0
        for pos in range(order.shape[0]):
            g = order[pos]
            a = assigned[g]
            out_gain = 0.0
            for c in range(n_cols):
                m = items[g, c]
                if m > 0.0:
                    t = targets[a, c]
                    out_gain += (m * m - 2.0 * m * (filled[a, c] - t)) / (t * t)
            best = a
            best_delta = -1e-12
            for b in range(n_bins):
                if b == a:
                    continue
                add = 0.0
                for c in range(n_cols):
                    m = items[g, c]
                    if m > 0.0:
                        t = targets[b, c]
                        add += (m * m + 2.0 * m * (filled[b, c] - t)) / (t * t)
                delta = add + out_gain
                if delta < best_delta:
                    best_delta = delta
                    best = b
            if best != a:
                for c in range(n_cols):
                    filled[a, c] -= items[g, c]
                    filled[best, c] += items[g, c]
                assigned[g] = best
                moved += 1
        if moved == 0:
            break
    return assigned


def _group_matrix(groups: np.ndarray, y: np.ndarray):
    """Per-group class counts (n_groups, n_classes + 1; last column the row count) and the row-to-group map."""
    ids, inverse = np.unique(groups, return_inverse=True)
    inverse = inverse.reshape(-1)
    items = np.zeros((ids.size, N_CLASSES + 1))
    np.add.at(items, (inverse, y), 1.0)
    items[:, N_CLASSES] = items[:, :N_CLASSES].sum(axis=1)
    return items, inverse


SIZE_BINS = ((1, 1), (2, 9), (10, 49), (50, 10 ** 9))


def _with_structure_columns(items: np.ndarray) -> np.ndarray:
    """Append balancing columns for the group structure.

    Columns after the class counts and the row count: the number of rows that lie in groups of
    size 1, 2 to 9, 10 to 49 and 50 or more, and the number of rows in groups with conflicting
    labels. They make every bin a representative sample of the group structure, not only of the
    class counts (without them the hold-out collected the largest groups, which are also the
    hardest, and was much harder than the folds).
    """
    size = items[:, N_CLASSES]
    extra = [np.where((size >= lo) & (size <= hi), size, 0.0) for lo, hi in SIZE_BINS]
    extra.append(np.where((items[:, :N_CLASSES] > 0).sum(axis=1) > 1, size, 0.0))
    return np.concatenate([items, np.stack(extra, axis=1)], axis=1)


def assign_groups(groups: np.ndarray, y: np.ndarray, shares, gen: np.random.Generator,
                  class_targets: np.ndarray | None = None) -> np.ndarray:
    """Bin number (row-wise) such that every group lies in one bin and class shares match `shares`.

    shares: relative size of the bins (for 5 folds: five times 0.2). The target
    class counts of bin b are shares[b] times the class total, unless
    class_targets (n_bins, n_classes) is given. Returns an int array (n_rows,).
    """
    items, inverse = _group_matrix(groups, y)
    items = _with_structure_columns(items)
    shares = np.asarray(shares, dtype=np.float64)
    shares = shares / shares.sum()
    totals = items.sum(axis=0)
    targets = shares[:, None] * totals[None, :]
    if class_targets is not None:
        targets[:, :N_CLASSES] = class_targets
        targets[:, N_CLASSES] = class_targets.sum(axis=1)
    targets = np.maximum(targets, 1.0)
    present = items[:, :N_CLASSES] > 0
    big = np.iinfo(np.int64).max
    rare_rank = np.where(present, totals[None, :N_CLASSES], big).argmin(axis=1)
    rarest_total = totals[rare_rank]
    size_class = np.floor(np.log2(items[:, N_CLASSES])).astype(np.int64)
    order = np.lexsort((gen.random(items.shape[0]), -size_class, rarest_total))
    noise = gen.random((items.shape[0], shares.size)) * 1e-9
    order = order.astype(np.int64)
    bins = _greedy(items, order, targets, noise)
    bins = _refine(items, bins, targets, gen.permutation(items.shape[0]).astype(np.int64), 20)
    return bins[inverse]


def grouped_holdout(table: data.Table, purpose: str = PURPOSE_GROUPED_SPLIT,
                    groups: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(dev_idx, gen_idx) for the grouped development set D_g and generalisation set G_g, both sorted.

    No group is split. The class counts of G_g are matched to the class counts of
    the shared generalisation set G of the primary protocol (20 percent of every
    class of the full table, as `protocol.stratified_holdout` fixes them).
    """
    groups = feature_groups(table) if groups is None else groups
    target_g = np.array([int(np.clip(round(protocol.HOLDOUT_FRACTION * n), 1, n - 1))
                         for n in data.class_counts(table.y)], dtype=np.float64)
    totals = data.class_counts(table.y).astype(np.float64)
    class_targets = np.stack([totals - target_g, target_g])
    bins = assign_groups(groups, table.y, [1 - protocol.HOLDOUT_FRACTION, protocol.HOLDOUT_FRACTION],
                         protocol.rng(purpose), class_targets)
    return np.flatnonzero(bins == 0), np.flatnonzero(bins == 1)


def grouped_kfold(y: np.ndarray, groups: np.ndarray, n_folds: int = protocol.N_FOLDS,
                  purpose: str = PURPOSE_GROUPED_TUNING, repeat: int = 0) -> list[protocol.Partition]:
    """One grouped, stratified k-fold partitioning (a single repeat); indices refer to y."""
    y = np.asarray(y)
    gen = protocol.rng(f"{purpose}/repeat{repeat}")
    fold = assign_groups(np.asarray(groups), y, np.ones(n_folds), gen)
    perm = gen.permutation(n_folds)
    fold = perm[fold]
    return [protocol.Partition(repeat, f, np.flatnonzero(fold != f), np.flatnonzero(fold == f))
            for f in range(n_folds)]


def grouped_tuning_partitions(y_dev: np.ndarray, groups_dev: np.ndarray) -> list[protocol.Partition]:
    """Grouped tuning folds: 5 folds inside D_g (purpose "grouped-tuning-cv")."""
    return grouped_kfold(y_dev, groups_dev, protocol.N_FOLDS, PURPOSE_GROUPED_TUNING, 0)


def grouped_comparison_partitions(y_dev: np.ndarray, groups_dev: np.ndarray) -> list[protocol.Partition]:
    """Grouped comparison partitions: 5 folds x 3 repeats = 15, repeat-major order (purpose "grouped-comparison-cv")."""
    out: list[protocol.Partition] = []
    for r in range(protocol.N_REPEATS):
        out.extend(grouped_kfold(y_dev, groups_dev, protocol.N_FOLDS, PURPOSE_GROUPED_COMPARISON, r))
    return out


class GroupedSplit:
    """Result of `grouped_split`: indices into the full table and the group numbers of D_g and G_g."""

    def __init__(self, groups, dev_idx, gen_idx):
        self.groups, self.dev_idx, self.gen_idx = groups, dev_idx, gen_idx
        self.groups_dev, self.groups_gen = groups[dev_idx], groups[gen_idx]


def grouped_split(table: data.Table) -> GroupedSplit:
    """Groups of the full table and the grouped hold-out in one call."""
    groups = feature_groups(table)
    dev_idx, gen_idx = grouped_holdout(table, groups=groups)
    return GroupedSplit(groups, dev_idx, gen_idx)


# ---------------------------------------------------------------------------
# Measurements
# ---------------------------------------------------------------------------

def twin_share(groups: np.ndarray, partition: protocol.Partition) -> float:
    """Share of the test rows that have an identical feature vector among the training rows."""
    in_train = np.isin(groups[partition.test], np.unique(groups[partition.train]))
    return float(in_train.mean())


def twin_share_same_label(groups: np.ndarray, y: np.ndarray, partition: protocol.Partition) -> float:
    """Share of the test rows with a training twin that also carries the same label."""
    key_train = set(zip(groups[partition.train].tolist(), y[partition.train].tolist()))
    hit = [(g, c) in key_train for g, c in zip(groups[partition.test].tolist(), y[partition.test].tolist())]
    return float(np.mean(hit))


def groups_cross(groups: np.ndarray, partition: protocol.Partition) -> int:
    """Number of groups that occur both in the training and in the test part (0 for a grouped partition)."""
    return int(np.intersect1d(groups[partition.train], groups[partition.test]).size)


def balance_table(y: np.ndarray, partitions: list[protocol.Partition]) -> np.ndarray:
    """Class counts of every test fold: array (n_partitions, n_classes)."""
    return np.stack([np.bincount(y[p.test], minlength=N_CLASSES) for p in partitions])


def class_balance_report(y: np.ndarray, partitions: list[protocol.Partition]) -> dict:
    """Achieved class balance of the test folds.

    For every class the ideal count is (class total) / n_folds. Reports the
    largest relative deviation from the ideal per class, the smallest test count
    per class and the relative spread of the fold sizes.
    """
    counts = balance_table(y, partitions)
    n_folds = len({p.fold for p in partitions})
    ideal = np.bincount(y, minlength=N_CLASSES) / n_folds
    deviation = np.abs(counts - ideal[None, :]) / ideal[None, :]
    sizes = counts.sum(axis=1)
    return {"ideal_per_class": ideal.tolist(),
            "max_relative_deviation_per_class": deviation.max(axis=0).tolist(),
            "min_test_count_per_class": counts.min(axis=0).tolist(),
            "max_test_count_per_class": counts.max(axis=0).tolist(),
            "fold_size_min": int(sizes.min()), "fold_size_max": int(sizes.max()),
            "fold_size_relative_spread": float((sizes.max() - sizes.min()) / sizes.mean())}
