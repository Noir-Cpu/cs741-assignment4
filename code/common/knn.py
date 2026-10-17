"""k-nearest neighbours classifier from scratch (numpy and Numba).

The search is exact brute force. Queries are processed in blocks in parallel
(Numba prange) and the training matrix is streamed in cache-sized tiles, so
memory use is the training matrix plus the (n_query, k) result arrays; the
(n_query, n_train) distance matrix is never formed. This keeps 200,000
training rows within a few hundred megabytes.

Metrics: "euclidean" and "manhattan". Votes: "uniform", "distance"
(weight 1 / max(distance, 1e-12), Shepard interpolation) and an optional
class weight that multiplies the vote of each class. class_weight "balanced"
uses n / (n_classes * n_c) from the training labels; a float alpha in (0, 1]
uses that quantity raised to the power alpha (alpha = 1 is "balanced").

Ties in distance are broken in favour of the lower training index. Ties in
the vote are broken in favour of the lower class label.

Typical tuning use: compute the neighbours once with the largest k, then call
KNNClassifier.vote for every candidate k, weighting and class weight.
"""
from __future__ import annotations

import numpy as np
from numba import njit, prange

_EPS = 1e-12
_TILE = 1024          # training rows per tile
_QUERY_BLOCK = 16     # queries per parallel work item
_CHECK = 16           # features between early-abandon checks
_BIG = np.float32(3.0e38)  # finite sentinel: fastmath must not see infinities


@njit(parallel=True, fastmath=True, cache=True)
def _search(train, query, k, p, out_idx, out_dist):
    """k nearest training rows of every query (squared Euclidean or Manhattan)."""
    n_train, d = train.shape
    n_query = query.shape[0]
    n_blocks = (n_query + _QUERY_BLOCK - 1) // _QUERY_BLOCK
    for blk in prange(n_blocks):
        q0 = blk * _QUERY_BLOCK
        q1 = min(q0 + _QUERY_BLOCK, n_query)
        best_d = np.full((q1 - q0, k), _BIG, dtype=np.float32)
        best_i = np.full((q1 - q0, k), -1, dtype=np.int32)
        for t0 in range(0, n_train, _TILE):
            t1 = min(t0 + _TILE, n_train)
            for qi in range(q1 - q0):
                q = query[q0 + qi]
                worst = best_d[qi, k - 1]
                for t in range(t0, t1):
                    row = train[t]
                    s = np.float32(0.0)
                    f = 0
                    abandoned = False
                    while f < d:
                        stop = min(f + _CHECK, d)
                        if p == 1:
                            for j in range(f, stop):
                                s += abs(q[j] - row[j])
                        else:
                            for j in range(f, stop):
                                diff = q[j] - row[j]
                                s += diff * diff
                        f = stop
                        if s >= worst:
                            abandoned = True
                            break
                    if abandoned:
                        continue
                    # insert into the sorted list of the k best (strictly smaller only)
                    pos = k - 1
                    while pos > 0 and best_d[qi, pos - 1] > s:
                        best_d[qi, pos] = best_d[qi, pos - 1]
                        best_i[qi, pos] = best_i[qi, pos - 1]
                        pos -= 1
                    best_d[qi, pos] = s
                    best_i[qi, pos] = t
                    worst = best_d[qi, k - 1]
        for qi in range(q1 - q0):
            for j in range(k):
                out_idx[q0 + qi, j] = best_i[qi, j]
                out_dist[q0 + qi, j] = best_d[qi, j]


@njit(parallel=True, cache=True)
def _vote(neigh_idx, neigh_dist, y_train, k, distance_weighted, class_weight, scores):
    n_query = neigh_idx.shape[0]
    for i in prange(n_query):
        for j in range(k):
            c = y_train[neigh_idx[i, j]]
            w = class_weight[c]
            if distance_weighted:
                w = w / max(neigh_dist[i, j], _EPS)
            scores[i, c] += w


def class_weight_vector(y: np.ndarray, n_classes: int, class_weight) -> np.ndarray:
    """Per-class vote weights from training labels (see module docstring)."""
    if class_weight is None:
        return np.ones(n_classes)
    counts = np.bincount(y, minlength=n_classes).astype(np.float64)
    balanced = np.zeros(n_classes)
    present = counts > 0
    balanced[present] = y.size / (present.sum() * counts[present])
    if isinstance(class_weight, str):
        if class_weight != "balanced":
            raise ValueError("class_weight must be None, 'balanced' or a float power")
        return balanced
    alpha = float(class_weight)
    return np.where(present, balanced, 0.0) ** alpha


class KNNClassifier:
    """Exact k-nearest-neighbours classifier.

    Parameters
    ----------
    n_neighbors:   k
    metric:        "euclidean" or "manhattan"
    weights:       "uniform" or "distance"
    class_weight:  None, "balanced" or a float power in (0, 1]
    n_classes:     number of classes (labels 0 to n_classes - 1); inferred if None
    """

    def __init__(self, n_neighbors: int = 5, metric: str = "euclidean",
                 weights: str = "uniform", class_weight=None, n_classes: int | None = None):
        if metric not in ("euclidean", "manhattan"):
            raise ValueError("metric must be 'euclidean' or 'manhattan'")
        if weights not in ("uniform", "distance"):
            raise ValueError("weights must be 'uniform' or 'distance'")
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.weights = weights
        self.class_weight = class_weight
        self.n_classes = n_classes

    def fit(self, X, y) -> "KNNClassifier":
        self.X_ = np.ascontiguousarray(X, dtype=np.float32)
        self.y_ = np.ascontiguousarray(y, dtype=np.int64)
        if self.n_classes is None:
            self.n_classes = int(self.y_.max()) + 1
        self.class_weight_ = class_weight_vector(self.y_, self.n_classes, self.class_weight)
        return self

    def kneighbors(self, X, k: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Distances (float32) and training indices (int32) of the k nearest rows, nearest first."""
        k = self.n_neighbors if k is None else k
        if k > self.X_.shape[0]:
            raise ValueError("k exceeds the number of training rows")
        query = np.ascontiguousarray(X, dtype=np.float32)
        idx = np.empty((query.shape[0], k), dtype=np.int32)
        dist = np.empty((query.shape[0], k), dtype=np.float32)
        _search(self.X_, query, k, 1 if self.metric == "manhattan" else 2, idx, dist)
        if self.metric == "euclidean":
            np.sqrt(dist, out=dist)
        return dist, idx

    @staticmethod
    def vote(neigh_idx: np.ndarray, neigh_dist: np.ndarray, y_train: np.ndarray, k: int,
             weights: str, class_weight_vec: np.ndarray, n_classes: int) -> np.ndarray:
        """Class scores (n_query, n_classes) from the first k columns of a neighbour list."""
        scores = np.zeros((neigh_idx.shape[0], n_classes), dtype=np.float64)
        _vote(neigh_idx, neigh_dist, y_train, k, weights == "distance",
              np.asarray(class_weight_vec, dtype=np.float64), scores)
        return scores

    def decision_scores(self, X) -> np.ndarray:
        """Weighted vote totals per class."""
        dist, idx = self.kneighbors(X)
        return self.vote(idx, dist, self.y_, self.n_neighbors, self.weights,
                         self.class_weight_, self.n_classes)

    def predict_proba(self, X) -> np.ndarray:
        scores = self.decision_scores(X)
        total = scores.sum(axis=1, keepdims=True)
        return np.divide(scores, total, out=np.zeros_like(scores), where=total > 0)

    def predict(self, X) -> np.ndarray:
        return np.argmax(self.decision_scores(X), axis=1)

    @staticmethod
    def predict_from_neighbours(neigh_idx, neigh_dist, y_train, k, weights, class_weight_vec,
                                n_classes) -> np.ndarray:
        """Predicted labels for stored neighbour lists (used in tuning grids)."""
        return np.argmax(KNNClassifier.vote(neigh_idx, neigh_dist, y_train, k, weights,
                                            class_weight_vec, n_classes), axis=1)
