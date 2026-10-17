"""Classification tree from scratch (numpy and Numba), built for use in ensembles.

Algorithm
---------
CART-style binary tree grown breadth first. Every feature is quantised once
into at most max_bins ordered bins (BinMapper). A feature with at most
max_bins distinct training values keeps one bin per value, so its splits are
exact and the threshold is the midpoint of two adjacent distinct training
values, as in CART. A feature with more distinct values uses quantile bins
and the threshold is the midpoint between the two training values on either
side of a bin boundary. At a node the class-weight histogram of each candidate
feature is accumulated in one pass over the node rows and scanned for the best
bin boundary. Nodes with few rows sort their bin codes instead, which gives the
identical split at lower cost.

Features
--------
* criterion "gini" or "entropy"; weighted by sample and class weights
* max_depth, min_samples_leaf (rows, not weight), min_samples_split
* max_features: random subset of non-constant features per split (an int,
  a fraction, "sqrt", "log2" or None for all features)
* class_weight None, "balanced", a dict or an array; sample_weight per row
  (rows with weight 0 are left out, so a bootstrap sample can be passed as
  multiplicity weights)
* deterministic: the result depends only on the data, the parameters and
  seed; with max_features = all features the seed has no effect
* the fit kernel releases the GIL, so several trees can be fitted in
  parallel from Python threads (ThreadPoolExecutor), each single-threaded
* prediction on raw (unbinned) features of any float dtype; leaves store the
  weighted class totals, and predict returns the class with the largest total
  (lowest label on ties)

Differences from scikit-learn (documented deviations)
-----------------------------------------------------
Thresholds are global bin boundaries and not node-local midpoints, and
features with more than max_bins distinct values are quantised. Both affect
only test points that fall in a gap between training values.
"""
from __future__ import annotations

import numpy as np
from numba import njit, prange

from . import protocol

_GINI, _ENTROPY = 0, 1
_SORT_RATIO = 3.0      # use the sort path when rows * ratio < bins * (classes + 1)


# ---------------------------------------------------------------------------
# Binning
# ---------------------------------------------------------------------------

class BinMapper:
    """Per-feature ordered bins with exact or quantile boundaries.

    After fit, edges[f] holds the ascending thresholds of feature f. A value x
    falls in bin searchsorted(edges[f], x, side="left"), so x <= edges[f][b]
    exactly when its bin is at most b.
    """

    def __init__(self, max_bins: int = 65535):
        if not 2 <= max_bins <= 65535:
            raise ValueError("max_bins must lie in [2, 65535]")
        self.max_bins = max_bins

    def fit(self, X) -> "BinMapper":
        X = np.asarray(X)
        self.edges_ = []
        for f in range(X.shape[1]):
            values, counts = np.unique(X[:, f].astype(np.float64), return_counts=True)
            if values.size <= 1:
                edges = np.empty(0)
            elif values.size <= self.max_bins:
                edges = self._midpoints(values[:-1], values[1:])
            else:
                cum = np.cumsum(counts)
                targets = cum[-1] * np.arange(1, self.max_bins) / self.max_bins
                pos = np.unique(np.searchsorted(cum, targets, side="left"))
                pos = pos[pos < values.size - 1]
                edges = self._midpoints(values[pos], values[pos + 1])
            self.edges_.append(edges)
        self.n_bins_ = np.array([e.size + 1 for e in self.edges_], dtype=np.int64)
        self.dtype_ = np.uint8 if self.n_bins_.max() <= 256 else np.uint16
        return self

    @staticmethod
    def _midpoints(lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
        mid = lo + (hi - lo) / 2.0
        return np.where(mid >= hi, lo, mid)   # adjacent doubles: keep lo <= edge < hi

    def transform(self, X) -> np.ndarray:
        """Binned data in feature-major layout, shape (n_features, n_rows)."""
        X = np.asarray(X)
        out = np.empty((X.shape[1], X.shape[0]), dtype=self.dtype_)
        for f, edges in enumerate(self.edges_):
            out[f] = np.searchsorted(edges, X[:, f].astype(np.float64), side="left")
        return out

    def fit_transform(self, X) -> np.ndarray:
        return self.fit(X).transform(X)


# ---------------------------------------------------------------------------
# Numba kernels
# ---------------------------------------------------------------------------

@njit(cache=True)
def _parent_cost(value, total, criterion):
    """Weighted impurity times total weight of a node with class totals value."""
    if criterion == 0:
        s = 0.0
        for c in range(value.shape[0]):
            s += value[c] * value[c]
        return total - s / total
    h = 0.0
    for c in range(value.shape[0]):
        if value[c] > 0.0:
            h -= value[c] * np.log(value[c] / total)
    return h


@njit(cache=True)
def _split_cost(left, value, total, criterion):
    """Sum over both children of weight times impurity for the left totals left."""
    wl = 0.0
    for c in range(left.shape[0]):
        wl += left[c]
    wr = total - wl
    if wl <= 0.0 or wr <= 0.0:
        return np.inf
    if criterion == 0:
        sl = 0.0
        sr = 0.0
        for c in range(left.shape[0]):
            r = value[c] - left[c]
            if r < 0.0:
                r = 0.0
            sl += left[c] * left[c]
            sr += r * r
        return total - (sl / wl + sr / wr)
    h = 0.0
    for c in range(left.shape[0]):
        lc = left[c]
        rc = value[c] - lc
        if lc > 0.0:
            h -= lc * np.log(lc / wl)
        if rc > 0.0:
            h -= rc * np.log(rc / wr)
    return h


@njit(cache=True)
def _grow1(a, cap, fill):
    out = np.full(cap, fill, dtype=a.dtype)
    out[:a.shape[0]] = a
    return out


@njit(nogil=True, cache=True)
def _build(Xb_t, y, w, rows, n_bins, n_classes, criterion, max_depth, min_leaf, min_split,
           max_features, seed, sort_ratio):
    """Grow one tree. Returns node arrays trimmed to the number of nodes."""
    p = Xb_t.shape[0]
    C = n_classes
    C1 = C + 1
    m_all = rows.shape[0]
    idx = rows.copy()
    tmp = np.empty(m_all, dtype=np.int32)
    yb = np.empty(m_all, dtype=np.int64)
    wb = np.empty(m_all, dtype=np.float64)
    codes = np.empty(m_all, dtype=np.int64)
    max_nb = 1
    for f in range(p):
        if n_bins[f] > max_nb:
            max_nb = n_bins[f]
    hist = np.empty(max_nb * C1, dtype=np.float64)
    left_w = np.empty(C, dtype=np.float64)
    best_left_w = np.empty(C, dtype=np.float64)
    perm = np.arange(p)
    np.random.seed(seed)

    cap = 1024
    feat = np.full(cap, -1, dtype=np.int32)
    thr = np.zeros(cap, dtype=np.int32)
    lc = np.full(cap, -1, dtype=np.int32)
    rc = np.full(cap, -1, dtype=np.int32)
    start = np.zeros(cap, dtype=np.int64)
    end = np.zeros(cap, dtype=np.int64)
    depth = np.zeros(cap, dtype=np.int32)
    nsamp = np.zeros(cap, dtype=np.int64)
    value = np.zeros((cap, C), dtype=np.float64)
    importance = np.zeros(p, dtype=np.float64)

    n_nodes = 1
    end[0] = m_all
    node = 0
    max_d = 0
    while node < n_nodes:
        s = start[node]
        e = end[node]
        m = e - s
        total = 0.0
        for i in range(m):
            r = idx[s + i]
            yb[i] = y[r]
            wb[i] = w[r]
            value[node, yb[i]] += wb[i]
            total += wb[i]
        nsamp[node] = m
        if depth[node] > max_d:
            max_d = depth[node]
        n_present = 0
        for c in range(C):
            if value[node, c] > 0.0:
                n_present += 1
        if (n_present <= 1 or m < min_split or m < 2 * min_leaf
                or (max_depth >= 0 and depth[node] >= max_depth)):
            node += 1
            continue

        parent = _parent_cost(value[node], total, criterion)
        best_cost = np.inf
        best_f = -1
        best_b = -1
        n_valid = 0
        use_all = max_features >= p
        for j in range(p):
            if use_all:
                f = j
            else:
                k = np.random.randint(j, p)
                tmp_f = perm[j]
                perm[j] = perm[k]
                perm[k] = tmp_f
                f = perm[j]
            nb = n_bins[f]
            if nb < 2:
                continue
            col = Xb_t[f]
            non_constant = False
            if m * sort_ratio < nb * C1:
                for i in range(m):
                    codes[i] = col[idx[s + i]]
                order = np.argsort(codes[:m])
                for c in range(C):
                    left_w[c] = 0.0
                ln = 0
                if codes[order[0]] != codes[order[m - 1]]:
                    non_constant = True
                for t in range(m - 1):
                    o = order[t]
                    left_w[yb[o]] += wb[o]
                    ln += 1
                    if codes[order[t + 1]] == codes[o]:
                        continue
                    rn = m - ln
                    if rn < min_leaf:
                        break
                    if ln < min_leaf:
                        continue
                    cost = _split_cost(left_w, value[node], total, criterion)
                    if cost < best_cost:
                        best_cost = cost
                        best_f = f
                        best_b = codes[o]
                        for c in range(C):
                            best_left_w[c] = left_w[c]
            else:
                for q in range(nb * C1):
                    hist[q] = 0.0
                for i in range(m):
                    b = col[idx[s + i]]
                    hist[b * C1 + yb[i]] += wb[i]
                    hist[b * C1 + C] += 1.0
                for c in range(C):
                    left_w[c] = 0.0
                ln = 0
                n_nonempty = 0
                for b in range(nb):
                    if hist[b * C1 + C] > 0.0:
                        n_nonempty += 1
                if n_nonempty >= 2:
                    non_constant = True
                    seen = 0
                    for b in range(nb - 1):
                        cnt = int(hist[b * C1 + C])
                        if cnt == 0:
                            continue
                        seen += 1
                        for c in range(C):
                            left_w[c] += hist[b * C1 + c]
                        ln += cnt
                        if seen == n_nonempty:
                            break
                        rn = m - ln
                        if rn < min_leaf:
                            break
                        if ln < min_leaf:
                            continue
                        cost = _split_cost(left_w, value[node], total, criterion)
                        if cost < best_cost:
                            best_cost = cost
                            best_f = f
                            best_b = b
                            for c in range(C):
                                best_left_w[c] = left_w[c]
            if non_constant:
                n_valid += 1
                if (not use_all) and n_valid >= max_features:
                    break

        if best_f < 0:
            node += 1
            continue

        importance[best_f] += parent - best_cost
        col = Xb_t[best_f]
        nl = 0
        nr = 0
        for i in range(m):
            r = idx[s + i]
            if col[r] <= best_b:
                idx[s + nl] = r
                nl += 1
            else:
                tmp[nr] = r
                nr += 1
        for i in range(nr):
            idx[s + nl + i] = tmp[i]

        if n_nodes + 2 > cap:
            cap = cap * 2
            feat = _grow1(feat, cap, -1)
            thr = _grow1(thr, cap, 0)
            lc = _grow1(lc, cap, -1)
            rc = _grow1(rc, cap, -1)
            start = _grow1(start, cap, 0)
            end = _grow1(end, cap, 0)
            depth = _grow1(depth, cap, 0)
            nsamp = _grow1(nsamp, cap, 0)
            value2 = np.zeros((cap, C), dtype=np.float64)
            value2[:value.shape[0]] = value
            value = value2
        feat[node] = best_f
        thr[node] = best_b
        li = n_nodes
        ri = n_nodes + 1
        lc[node] = li
        rc[node] = ri
        start[li] = s
        end[li] = s + nl
        start[ri] = s + nl
        end[ri] = e
        depth[li] = depth[node] + 1
        depth[ri] = depth[node] + 1
        n_nodes += 2
        node += 1

    return (feat[:n_nodes].copy(), thr[:n_nodes].copy(), lc[:n_nodes].copy(),
            rc[:n_nodes].copy(), value[:n_nodes].copy(), nsamp[:n_nodes].copy(),
            importance, max_d)


@njit(nogil=True, cache=True)
def _apply_serial(X, feature, threshold, left, right):
    out = np.empty(X.shape[0], dtype=np.int32)
    for i in range(X.shape[0]):
        node = 0
        while feature[node] >= 0:
            if X[i, feature[node]] <= threshold[node]:
                node = left[node]
            else:
                node = right[node]
        out[i] = node
    return out


@njit(parallel=True, cache=True)
def _apply_parallel(X, feature, threshold, left, right):
    out = np.empty(X.shape[0], dtype=np.int32)
    for i in prange(X.shape[0]):
        node = 0
        while feature[node] >= 0:
            if X[i, feature[node]] <= threshold[node]:
                node = left[node]
            else:
                node = right[node]
        out[i] = node
    return out


# ---------------------------------------------------------------------------
# Public estimator
# ---------------------------------------------------------------------------

def resolve_max_features(max_features, n_features: int) -> int:
    """Number of features tried per split from an int, a fraction, a name or None."""
    if max_features is None:
        return n_features
    if isinstance(max_features, str):
        if max_features == "sqrt":
            return max(1, int(np.sqrt(n_features)))
        if max_features == "log2":
            return max(1, int(np.log2(n_features)))
        raise ValueError("max_features must be an int, a float, 'sqrt', 'log2' or None")
    if isinstance(max_features, (float, np.floating)):
        return max(1, min(n_features, int(max_features * n_features)))
    return max(1, min(n_features, int(max_features)))


def class_weight_array(class_weight, y: np.ndarray, weights: np.ndarray, n_classes: int) -> np.ndarray:
    """Per-class multipliers. "balanced" uses W / (n_present * W_c) from weighted counts."""
    if class_weight is None:
        return np.ones(n_classes)
    if isinstance(class_weight, str):
        if class_weight != "balanced":
            raise ValueError("class_weight must be None, 'balanced', a dict or an array")
        totals = np.bincount(y, weights=weights, minlength=n_classes)
        present = totals > 0
        out = np.zeros(n_classes)
        out[present] = totals.sum() / (present.sum() * totals[present])
        return out
    if isinstance(class_weight, dict):
        out = np.ones(n_classes)
        for c, v in class_weight.items():
            out[int(c)] = v
        return out
    return np.asarray(class_weight, dtype=np.float64)


class DecisionTreeClassifier:
    """Classification tree (see the module docstring).

    Parameters
    ----------
    criterion:         "gini" or "entropy"
    max_depth:         maximum depth of the tree (root has depth 0); None for unlimited
    min_samples_leaf:  minimum number of rows in each child of a split
    min_samples_split: minimum number of rows in a node that may be split
    max_features:      features tried per split (int, fraction, "sqrt", "log2", None)
    class_weight:      None, "balanced", dict or array
    max_bins:          bins per feature for the histogram search (at most 65535); the
                       default keeps the tree close to an exact CART split search.
                       Values near 255 are faster and use uint8 binned data, but cost
                       about 0.01 macro-F1 on the network traffic data (see results/common)
    seed:              integer seed for the random feature subsets
                       (default: derived from the master seed, purpose "tree-default")
    """

    def __init__(self, criterion: str = "gini", max_depth: int | None = None,
                 min_samples_leaf: int = 1, min_samples_split: int = 2, max_features=None,
                 class_weight=None, max_bins: int = 65535, seed: int | None = None):
        if criterion not in ("gini", "entropy"):
            raise ValueError("criterion must be 'gini' or 'entropy'")
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.class_weight = class_weight
        self.max_bins = max_bins
        self.seed = seed

    def fit(self, X, y, sample_weight=None, n_classes: int | None = None,
            binner: BinMapper | None = None, Xb_t: np.ndarray | None = None
            ) -> "DecisionTreeClassifier":
        """Fit on raw features X (n, p) and integer labels y.

        To fit many trees on the same data (an ensemble), build one BinMapper
        and its binned data once and pass both as binner and Xb_t; X is then
        used only for its shape and may be the same array.
        """
        if Xb_t is not None and binner is None:
            raise ValueError("Xb_t needs the BinMapper that produced it (binner)")
        X = np.asarray(X) if Xb_t is None else X
        y = np.ascontiguousarray(y, dtype=np.int64)
        n_rows = y.shape[0]
        self.n_classes_ = int(y.max()) + 1 if n_classes is None else int(n_classes)
        if binner is None:
            binner = BinMapper(self.max_bins).fit(X)
        if Xb_t is None:
            Xb_t = binner.transform(X)
        self.binner_ = binner
        self.n_features_ = Xb_t.shape[0]
        sw = np.ones(n_rows) if sample_weight is None else np.asarray(sample_weight, dtype=np.float64)
        cw = class_weight_array(self.class_weight, y, sw, self.n_classes_)
        w = np.ascontiguousarray(sw * cw[y])
        rows = np.flatnonzero(w > 0).astype(np.int32)
        if rows.size == 0:
            raise ValueError("no row has a positive weight")
        seed = protocol.seed_int("tree-default") if self.seed is None else int(self.seed)
        out = _build(Xb_t, y, w, rows, np.ascontiguousarray(binner.n_bins_, dtype=np.int64),
                     self.n_classes_, 0 if self.criterion == "gini" else 1,
                     -1 if self.max_depth is None else int(self.max_depth),
                     int(self.min_samples_leaf), int(self.min_samples_split),
                     resolve_max_features(self.max_features, self.n_features_),
                     seed % (2 ** 32), _SORT_RATIO)
        (self.feature_, thr_bin, self.children_left_, self.children_right_, self.value_,
         self.n_node_samples_, importance, self.depth_) = out
        self.threshold_ = np.zeros(self.feature_.shape[0])
        internal = self.feature_ >= 0
        for node in np.flatnonzero(internal):
            self.threshold_[node] = binner.edges_[self.feature_[node]][thr_bin[node]]
        self.n_nodes_ = int(self.feature_.shape[0])
        self.n_leaves_ = int((~internal).sum())
        total = importance.sum()
        self.feature_importances_ = importance / total if total > 0 else importance
        return self

    def apply(self, X, parallel: bool = False) -> np.ndarray:
        """Leaf index of every row. Use parallel=False inside Python threads."""
        X = np.asarray(X)
        if X.dtype not in (np.float32, np.float64):
            X = X.astype(np.float64)
        kernel = _apply_parallel if parallel else _apply_serial
        return kernel(X, self.feature_, self.threshold_, self.children_left_, self.children_right_)

    def predict_proba(self, X, parallel: bool = False) -> np.ndarray:
        """Normalised weighted class totals of the leaf reached by each row."""
        leaf = self.apply(X, parallel)
        value = self.value_[leaf]
        total = value.sum(axis=1, keepdims=True)
        return np.divide(value, total, out=np.zeros_like(value), where=total > 0)

    def predict(self, X, parallel: bool = False) -> np.ndarray:
        """Class with the largest weighted total in the leaf (lowest label on ties)."""
        return np.argmax(self.value_[self.apply(X, parallel)], axis=1)
