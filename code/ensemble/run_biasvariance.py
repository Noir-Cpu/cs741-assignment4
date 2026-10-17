"""Bias and variance of single trees, bagging, random forests and boosting by repeated training sets.

Design. The first tuning fold of D supplies a fixed test pool (the test part, about
41,000 flows) and a training pool (the remaining flows). Twenty training sets are drawn
from the training pool as class-proportional random subsets of half its size. Every
model type is fitted once on each training set and predicts the test pool. For a test
row x with label y and predictions h_1(x), ..., h_R(x):

  main prediction    the most frequent prediction over the R training sets
  bias               1 if the main prediction differs from y, averaged over test rows
  variance           the share of the R predictions that differ from the main prediction
  error              mean over rows and training sets of 1 if h_r(x) differs from y
  net variance       error minus bias: the part of the error that the dependence on the
                     training set adds (positive) or removes (negative)
  unbiased variance  variance on rows where the main prediction is correct
  biased variance    variance on rows where the main prediction is wrong

The label noise of the data (repeated feature vectors with conflicting labels) sits in
the bias term, because y is the observed label.

Usage: nohup .venv/bin/python code/ensemble/run_biasvariance.py > logs/bv.log 2>&1 &
"""
from __future__ import annotations

import os

for _name in ("NUMBA_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_name, "10")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import metrics, protocol  # noqa: E402
from common.tree import BinMapper, DecisionTreeClassifier  # noqa: E402
from ensemble import ensemble, harness  # noqa: E402
from ensemble.harness import N_CLASSES, RESULTS, log  # noqa: E402

N_SETS = 20
SUBSET_FRACTION = 0.5
MAX_BINS = 4095
N_TREES = 8 if harness.SMOKE else 100


def models(tuned: dict, tree_params: dict):
    """Model factories: name -> function(purpose) returning an object with fit and probe predictions."""
    rf = dict(tuned["plain"]["params"])
    bag = dict(rf)
    bag["max_features"] = None
    boost = tuned["boost"]
    out = {
        "tree": ("tree", tree_params["plain"]),
        "tree_balanced": ("tree", tree_params["balanced"]),
        "bagging": ("forest", bag),
        "rf": ("forest", rf),
        "rf_weighted": ("forest", dict(tuned["weighted"]["params"])),
        "rf_sampled": ("forest", dict(tuned["sampled"]["params"])),
        "boost": ("boost", boost),
    }
    return out


def predict_with(kind: str, params, Xtr, ytr, Xte, purpose: str) -> np.ndarray:
    if kind == "tree":
        tree = DecisionTreeClassifier(**params, seed=protocol.seed_int("baseline-tree"), max_bins=MAX_BINS)
        tree.fit(Xtr, ytr, n_classes=N_CLASSES)
        return tree.predict(Xte).astype(np.int64)
    if kind == "forest":
        rf = ensemble.RandomForestClassifier(n_estimators=N_TREES, max_bins=MAX_BINS, n_jobs=harness.N_JOBS, purpose=purpose,
                                             keep_trees=False, **params)
        rf.fit(Xtr, ytr, probes={"t": Xte}, checkpoints=[N_TREES])
        return rf.probe_pred_["t"][N_TREES].astype(np.int64)
    ab = ensemble.AdaBoostClassifier(n_estimators=params["n_estimators"], max_bins=MAX_BINS, purpose=purpose,
                                     **params["params"])
    ab.fit(Xtr, ytr, probes={"t": Xte}, checkpoints=[params["n_estimators"]])
    return ab.probe_pred_["t"][params["n_estimators"]].astype(np.int64)


def optimal_labels(dev, test) -> np.ndarray:
    """Estimate of the best possible label of every test row: the most frequent label among the rows of D
    with the same complete feature vector (the observed label when the vector is unique)."""
    frame = pd.DataFrame(np.hstack([dev.numeric, dev.nominal.astype(np.float64)]))
    key = pd.util.hash_pandas_object(frame, index=False).to_numpy()
    table = pd.DataFrame({"key": key, "y": dev.y})
    counts = table.groupby(["key", "y"]).size().reset_index(name="n")
    counts = counts.sort_values(["key", "n", "y"], ascending=[True, False, True]).drop_duplicates("key")
    best = dict(zip(counts["key"], counts["y"]))
    frame_t = pd.DataFrame(np.hstack([test.numeric, test.nominal.astype(np.float64)]))
    key_t = pd.util.hash_pandas_object(frame_t, index=False).to_numpy()
    return np.array([best[k] for k in key_t], dtype=np.int64)


def decompose(preds: np.ndarray, y: np.ndarray, y_opt: np.ndarray) -> dict:
    """preds has shape (R, rows). y_opt is the estimated best label of every row."""
    R, n = preds.shape
    votes = np.stack([(preds == k).sum(axis=0) for k in range(N_CLASSES)], axis=1)
    main = np.argmax(votes, axis=1)
    biased = main != y
    disagree = (preds != main[None, :]).mean(axis=0)
    err = (preds != y[None, :]).mean(axis=0)
    out = {
        "error": float(err.mean()),
        "bias": float(biased.mean()),
        "net_variance": float(err.mean() - biased.mean()),
        "variance": float(disagree.mean()),
        "unbiased_variance": float(disagree[~biased].sum() / n),
        "biased_variance": float(disagree[biased].sum() / n),
        "noise": float((y != y_opt).mean()),
        "bias_opt": float((main != y_opt).mean()),
        "macro_f1_mean": float(np.mean([metrics.score_all(y, p, N_CLASSES)["macro_f1"] for p in preds])),
        "accuracy_mean": float(np.mean([(p == y).mean() for p in preds])),
        "main_macro_f1": float(metrics.score_all(y, main, N_CLASSES)["macro_f1"]),
    }
    for j in range(N_CLASSES):
        m = y == j
        out[f"bias_{j}"] = float(biased[m].mean())
        out[f"error_{j}"] = float(err[m].mean())
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sets", type=int, default=N_SETS)
    ap.add_argument("--fold", type=int, default=0)
    args = ap.parse_args()
    enc = json.loads((RESULTS / "selected_encoding.json").read_text())["encoding"]
    tuned = json.loads((RESULTS / "tuned_forests.json").read_text())
    tuned["boost"] = json.loads((RESULTS / "tuned_boosting.json").read_text())
    tb = json.loads((harness.data.RESULTS_COMMON / "tree_best_params.json").read_text())

    def tp(d):
        return {"criterion": d["criterion"], "max_depth": d["max_depth"],
                "min_samples_leaf": d["min_samples_leaf"], "class_weight": d["class_weight"]}
    tree_params = {"plain": tp(tb), "balanced": tp(tb["balanced_variant"])}

    table, dev, _ = harness.load_split()
    part = protocol.tuning_partitions(dev.y)[args.fold]
    pool, test = dev.take(part.train), dev.take(part.test)
    log(f"training pool {pool.n} rows, test pool {test.n} rows, encoding {enc}")
    specs = models(tuned, tree_params)
    y_opt = optimal_labels(dev, test)
    log(f"estimated noise of the test pool: {(y_opt != test.y).mean():.4f}")
    preds = {name: [] for name in specs}
    started = time.perf_counter()
    for r in range(args.sets):
        sub = protocol.stratified_subsample(pool.y, int(SUBSET_FRACTION * pool.n), f"opt1/bv/fold{args.fold}/subset{r}", min_per_class=5)
        train = pool.take(sub)
        Xtr, Xte = harness.prepare(train, test, enc)
        for name, (kind, params) in specs.items():
            t0 = time.perf_counter()
            preds[name].append(predict_with(kind, params, Xtr, train.y, Xte, f"opt1/bv/fold{args.fold}/set{r}"))
            log(f"set {r} {name}: macro-F1 {metrics.score_all(test.y, preds[name][-1], N_CLASSES)['macro_f1']:.4f} "
                f"({time.perf_counter() - t0:.0f}s)")
    rows = []
    for name in specs:
        row = {"model": name, "n_sets": args.sets, "train_rows": int(SUBSET_FRACTION * pool.n)}
        row.update(decompose(np.stack(preds[name]), test.y, y_opt))
        rows.append(row)
    for r in rows:
        r["fold"] = args.fold
    pd.DataFrame(rows).to_csv(RESULTS / f"bias_variance_fold{args.fold}.csv", index=False)
    log(f"finished in {(time.perf_counter() - started) / 60:.1f} minutes")


if __name__ == "__main__":
    main()
