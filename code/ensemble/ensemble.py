"""Random forest and adaptive boosting (SAMME) from scratch on common.tree.

Both ensembles grow the classification trees of common.tree. One BinMapper and
one binned matrix are built per training set and shared by every tree. Trees of
a random forest are independent, so they are fitted in parallel with a
ThreadPoolExecutor (the fit kernel releases the GIL). Boosting rounds depend on
each other, so one boosting model runs on one thread and parallelism comes from
fitting several models at once.

Random forest
-------------
* Each tree is grown on its own random sample of the training rows. The sample
  is stored as per-row multiplicities that are passed to the tree as sample
  weights, so rows with multiplicity zero form the out-of-bag set of the tree.
* sampling="bootstrap" draws n rows with replacement. A row of class j has
  draw probability proportional to n_j ** (-sampling_power): zero gives the
  standard bootstrap, one gives a class-balanced bootstrap, with equal expected
  counts for all classes.
* sampling="under" draws the same number of rows with replacement from every
  class (class_sample_size, default the size of the smallest class), as in
  exploratory undersampling.
* class_weight_power sets class weights b_j = (n / (m n_j)) ** power in the
  tree, the same form as the class-weight power of the nearest-neighbour vote.
* The ensemble averages the normalised leaf distributions of the trees.

Adaptive boosting
-----------------
Multi-class SAMME (Zhu et al., 2009): instance weights start proportional to
the class weight of the instance, a tree is grown on the weighted data, the
weighted error e gives the model weight a = rate * (ln((1 - e) / e) + ln(m - 1)),
the weights of misclassified instances are multiplied by exp(a) and the
weights are normalised. The ensemble predicts the class with the largest sum of
model weights.

Randomness. Every stream comes from common.protocol: tree i of an ensemble with
purpose P uses P + "/tree<i>/sample" and P + "/tree<i>/features".
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from common import protocol
from common.tree import BinMapper, DecisionTreeClassifier

N_CLASSES = 10


def class_weight_vector(y: np.ndarray, n_classes: int, power: float) -> np.ndarray:
    """Class weights b_j = (n / (m n_j)) ** power; m counts the classes present, absent classes get zero."""
    counts = np.bincount(np.asarray(y, dtype=np.int64), minlength=n_classes).astype(np.float64)
    present = counts > 0
    out = np.zeros(n_classes)
    if power == 0:
        out[present] = 1.0
    else:
        out[present] = (counts.sum() / (present.sum() * counts[present])) ** power
    return out


def draw_multiplicities(y: np.ndarray, counts: np.ndarray, sampling: str, power: float,
                        per_class: int | None, gen: np.random.Generator) -> np.ndarray:
    """Per-row multiplicities of one random sample (float64, zero for rows left out).

    "bootstrap": n draws with probability proportional to n_class ** -power.
    "under": per_class draws with replacement from every class present.
    """
    n = y.shape[0]
    if sampling == "bootstrap":
        if power == 0:
            return np.bincount(gen.integers(0, n, size=n), minlength=n).astype(np.float64)
        p = np.maximum(counts, 1.0)[y] ** (-float(power))
        return gen.multinomial(n, p / p.sum()).astype(np.float64)
    if sampling == "under":
        out = np.zeros(n)
        size = int(counts[counts > 0].min()) if per_class is None else int(per_class)
        order = np.argsort(y, kind="stable")
        bounds = np.concatenate([[0], np.cumsum(np.bincount(y, minlength=counts.size))])
        for j in range(counts.size):
            members = order[bounds[j]:bounds[j + 1]]
            if members.size:
                out[members] = np.bincount(gen.integers(0, members.size, size=size),
                                           minlength=members.size)
        return out
    raise ValueError("sampling must be 'bootstrap' or 'under'")


def _checkpoint_set(checkpoints, n_estimators: int) -> set[int]:
    cps = {int(c) for c in (checkpoints or ()) if 1 <= int(c) <= n_estimators}
    cps.add(n_estimators)
    return cps


def _argmax_rows(a: np.ndarray) -> np.ndarray:
    return np.argmax(a, axis=1).astype(np.uint8)


class RandomForestClassifier:
    """Random forest on common.tree (see the module docstring).

    Parameters
    ----------
    n_estimators:        number of trees
    criterion:           "gini" or "entropy"
    max_features:        features tried per split (int, fraction, "sqrt", "log2", None)
    max_depth:           depth limit of every tree (None for unlimited)
    min_samples_leaf:    minimum rows in each child of a split
    sampling:            "bootstrap" or "under"
    sampling_power:      class-balance exponent of the bootstrap (0 standard, 1 balanced)
    class_sample_size:   draws per class for sampling="under" (default: smallest class)
    class_weight_power:  class weights in the trees, b_j = (n / (m n_j)) ** power
    max_bins:            bins per feature of the shared BinMapper
    n_jobs:              threads used to fit and apply trees
    purpose:             name of the random stream (see common.protocol)
    keep_trees:          keep the fitted trees so that predict_proba works after fit
    """

    def __init__(self, n_estimators: int = 100, criterion: str = "gini", max_features="sqrt",
                 max_depth: int | None = None, min_samples_leaf: int = 1,
                 sampling: str = "bootstrap", sampling_power: float = 0.0,
                 class_sample_size: int | None = None, class_weight_power: float = 0.0,
                 max_bins: int = 4095, n_jobs: int = 10, purpose: str = "opt1/forest",
                 keep_trees: bool = True):
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_features = max_features
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.sampling = sampling
        self.sampling_power = sampling_power
        self.class_sample_size = class_sample_size
        self.class_weight_power = class_weight_power
        self.max_bins = max_bins
        self.n_jobs = n_jobs
        self.purpose = purpose
        self.keep_trees = keep_trees

    # ------------------------------------------------------------------
    def fit(self, X, y, n_classes: int = N_CLASSES, probes: dict | None = None,
            checkpoints=None, track_oob: bool = False) -> "RandomForestClassifier":
        """Fit the forest.

        probes:      mapping name -> feature matrix evaluated while fitting. After fit,
                     probe_pred_[name][T] holds the predicted labels of the first T trees
                     for each checkpoint T, probe_proba_[name] the final averaged
                     distribution, and probe_members_[name] the label predicted by every
                     tree, shape (rows, trees), uint8.
        track_oob:   accumulate out-of-bag distributions; oob_pred_[T] holds the labels
                     predicted from the out-of-bag trees of each training row after T trees
                     (255 where no tree has the row out of bag).
        """
        started = time.perf_counter()
        X = np.asarray(X)
        y = np.ascontiguousarray(y, dtype=np.int64)
        n = y.shape[0]
        self.n_classes_ = n_classes
        counts = np.bincount(y, minlength=n_classes).astype(np.float64)
        binner = BinMapper(self.max_bins).fit(X)
        Xb_t = binner.transform(X)
        self.binner_ = binner
        cw = class_weight_vector(y, n_classes, self.class_weight_power)
        probes = dict(probes or {})
        cps = _checkpoint_set(checkpoints, self.n_estimators)
        T = self.n_estimators

        acc = {k: np.zeros((v.shape[0], n_classes)) for k, v in probes.items()}
        members = {k: np.empty((v.shape[0], T), dtype=np.uint8) for k, v in probes.items()}
        self.probe_pred_ = {k: {} for k in probes}
        oob_sum = np.zeros((n, n_classes), dtype=np.float32) if track_oob else None
        oob_votes = np.zeros(n, dtype=np.int32) if track_oob else None
        self.oob_pred_ = {}
        trees, depths, nodes, leaves = [], np.empty(T, np.int32), np.empty(T, np.int64), np.empty(T, np.int64)
        importance = np.zeros(Xb_t.shape[0])
        oob_fraction = np.empty(T)

        def work(i: int):
            gen = protocol.rng(f"{self.purpose}/tree{i}/sample")
            w = draw_multiplicities(y, counts, self.sampling, self.sampling_power,
                                    self.class_sample_size, gen)
            tree = DecisionTreeClassifier(
                criterion=self.criterion, max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf, max_features=self.max_features,
                class_weight=cw, max_bins=self.max_bins,
                seed=protocol.seed_int(f"{self.purpose}/tree{i}/features"))
            tree.fit(X, y, sample_weight=w, n_classes=n_classes, binner=binner, Xb_t=Xb_t)
            out_probe = {k: tree.predict_proba(v).astype(np.float32) for k, v in probes.items()}
            rows = oob_p = None
            if track_oob:
                rows = np.flatnonzero(w == 0)
                oob_p = tree.predict_proba(X[rows]).astype(np.float32)
            return i, tree, out_probe, rows, oob_p, float((w == 0).mean())

        wave = max(2 * self.n_jobs, 1)
        with ThreadPoolExecutor(max_workers=self.n_jobs) as pool:
            for first in range(0, T, wave):
                for i, tree, out_probe, rows, oob_p, oobf in pool.map(work, range(first, min(T, first + wave))):
                    depths[i], nodes[i], leaves[i] = tree.depth_, tree.n_nodes_, tree.n_leaves_
                    importance += tree.feature_importances_
                    oob_fraction[i] = oobf
                    for k, p in out_probe.items():
                        acc[k] += p
                        members[k][:, i] = _argmax_rows(p)
                    if track_oob:
                        oob_sum[rows] += oob_p
                        oob_votes[rows] += 1
                    if self.keep_trees:
                        trees.append(tree)
                    done = i + 1
                    if done in cps:
                        for k in probes:
                            self.probe_pred_[k][done] = _argmax_rows(acc[k])
                        if track_oob:
                            pred = _argmax_rows(oob_sum)
                            pred[oob_votes == 0] = 255
                            self.oob_pred_[done] = pred
        self.trees_ = trees
        self.tree_depths_, self.tree_nodes_, self.tree_leaves_ = depths, nodes, leaves
        self.feature_importances_ = importance / T
        self.oob_fraction_ = oob_fraction
        self.probe_proba_ = {k: (acc[k] / T).astype(np.float32) for k in probes}
        self.probe_members_ = members
        if track_oob:
            self.oob_proba_ = oob_sum
            self.oob_votes_ = oob_votes
        self.fit_seconds_ = time.perf_counter() - started
        return self

    # ------------------------------------------------------------------
    def predict_proba(self, X) -> np.ndarray:
        """Average of the normalised leaf distributions of all trees (needs keep_trees)."""
        if not self.trees_:
            raise RuntimeError("the trees were not kept; fit with keep_trees=True")
        X = np.asarray(X)
        total = np.zeros((X.shape[0], self.n_classes_))
        wave = max(2 * self.n_jobs, 1)
        with ThreadPoolExecutor(max_workers=self.n_jobs) as pool:
            for first in range(0, len(self.trees_), wave):
                for p in pool.map(lambda t: t.predict_proba(X), self.trees_[first:first + wave]):
                    total += p
        return total / len(self.trees_)

    def predict(self, X) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)


class AdaBoostClassifier:
    """Multi-class adaptive boosting (SAMME) with classification trees as weak learners.

    Parameters
    ----------
    n_estimators:        maximum number of boosting rounds
    max_depth:           depth of the weak learners (1 is a decision stump)
    min_samples_leaf:    minimum rows in each child of a split
    criterion:           "gini" or "entropy"
    max_features:        features tried per split (None: all)
    learning_rate:       shrinkage of the model weights
    class_weight_power:  initial instance weights proportional to (n / (m n_j)) ** power
    max_bins:            bins per feature of the shared BinMapper
    purpose:             name of the random stream
    """

    def __init__(self, n_estimators: int = 100, max_depth: int | None = 3, min_samples_leaf: int = 1,
                 criterion: str = "gini", max_features=None, learning_rate: float = 1.0,
                 class_weight_power: float = 0.0, max_bins: int = 4095,
                 purpose: str = "opt1/boost"):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.criterion = criterion
        self.max_features = max_features
        self.learning_rate = learning_rate
        self.class_weight_power = class_weight_power
        self.max_bins = max_bins
        self.purpose = purpose

    def fit(self, X, y, n_classes: int = N_CLASSES, probes: dict | None = None,
            checkpoints=None, keep_trees: bool = False,
            weight_checkpoints=None) -> "AdaBoostClassifier":
        """Fit by SAMME. probes and checkpoints work as in RandomForestClassifier.fit.

        Besides probe_pred_, probe_members_ and probe_scores_ (final summed model weights),
        the training run records the weighted error and the model weight of every round
        (errors_, alphas_) and the unweighted training error of the ensemble at the
        checkpoints (train_error_).
        """
        started = time.perf_counter()
        X = np.asarray(X)
        y = np.ascontiguousarray(y, dtype=np.int64)
        n = y.shape[0]
        K = n_classes
        T = self.n_estimators
        binner = BinMapper(self.max_bins).fit(X)
        Xb_t = binner.transform(X)
        w = class_weight_vector(y, K, self.class_weight_power)[y]
        w = w / w.sum()
        probes = dict(probes or {})
        cps = _checkpoint_set(checkpoints, T)
        score = np.zeros((n, K))
        pscore = {k: np.zeros((v.shape[0], K)) for k, v in probes.items()}
        members = {k: np.zeros((v.shape[0], T), dtype=np.uint8) for k, v in probes.items()}
        self.probe_pred_ = {k: {} for k in probes}
        self.train_error_ = {}
        self.weights_at_ = {}
        wcps = set(weight_checkpoints or ())
        errors, alphas, trees = [], [], []
        depths, nodes = [], []
        rows_all = np.arange(n)
        done = 0
        stopped = False
        for t in range(T):
            tree = DecisionTreeClassifier(
                criterion=self.criterion, max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf, max_features=self.max_features,
                max_bins=self.max_bins, seed=protocol.seed_int(f"{self.purpose}/round{t}/features"))
            tree.fit(X, y, sample_weight=w, n_classes=K, binner=binner, Xb_t=Xb_t)
            pred = tree.predict(X)
            wrong = pred != y
            err = float(w[wrong].sum() / w.sum())
            if err >= 1.0 - 1.0 / K:
                stopped = True
                break
            alpha = self.learning_rate * (np.log((1.0 - err) / max(err, 1e-12)) + np.log(K - 1.0))
            score[rows_all, pred] += alpha
            for k, v in probes.items():
                pp = tree.predict(v)
                pscore[k][np.arange(v.shape[0]), pp] += alpha
                members[k][:, t] = pp
            errors.append(err)
            alphas.append(alpha)
            depths.append(tree.depth_)
            nodes.append(tree.n_nodes_)
            if keep_trees:
                trees.append(tree)
            done = t + 1
            if done in cps:
                for k in probes:
                    self.probe_pred_[k][done] = _argmax_rows(pscore[k])
                self.train_error_[done] = float((np.argmax(score, axis=1) != y).mean())
            if err <= 0.0:
                stopped = True
                break
            w = w * np.exp(alpha * wrong)
            w = w / w.sum()
            if done in wcps:
                self.weights_at_[done] = w.astype(np.float32)
        if done == 0:
            raise RuntimeError("the first weak learner is no better than random guessing")
        # Checkpoints that were not reached repeat the final state.
        for k in probes:
            last = _argmax_rows(pscore[k])
            for c in cps:
                self.probe_pred_[k].setdefault(c, last)
        for c in cps:
            self.train_error_.setdefault(c, float((np.argmax(score, axis=1) != y).mean()))
        self.n_rounds_ = done
        self.stopped_early_ = stopped
        self.errors_ = np.array(errors)
        self.alphas_ = np.array(alphas)
        self.tree_depths_ = np.array(depths)
        self.tree_nodes_ = np.array(nodes)
        self.trees_ = trees
        self.probe_members_ = {k: v[:, :done] for k, v in members.items()}
        self.probe_scores_ = pscore
        self.n_classes_ = K
        self.fit_seconds_ = time.perf_counter() - started
        return self

    def decision_scores(self, X) -> np.ndarray:
        """Summed model weights per class (needs keep_trees=True in fit)."""
        if not self.trees_:
            raise RuntimeError("the trees were not kept; fit with keep_trees=True")
        X = np.asarray(X)
        out = np.zeros((X.shape[0], self.n_classes_))
        rows = np.arange(X.shape[0])
        for tree, alpha in zip(self.trees_, self.alphas_):
            out[rows, tree.predict(X)] += alpha
        return out

    def predict(self, X) -> np.ndarray:
        return np.argmax(self.decision_scores(X), axis=1)
