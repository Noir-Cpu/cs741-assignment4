"""Final comparison under the compatible and the grouped (leakage-aware) protocol, with dissimilar ensembles.

For every one of the 15 comparison partitions of a protocol the script fits

  * three tuned random forests (plain, class-weighted, class-sampled) with FOREST_RUNS independent runs,
  * the tuned boosting ensemble (deterministic, one run),
  * the four baselines of the second assignment (tree, balanced tree, k-nearest neighbours with and without class weights),
  * dissimilar ensembles of run zero of the forests, the tree and the weighted k-nearest-neighbour model:
    soft votes (equal weights) of several member sets, the leave-one-member-out ablation of the
    four-member vote, and one stacked ensemble. The stacked ensemble trains a small forest on the
    out-of-fold class distributions of its members, obtained from five grouped folds inside the
    training part, so that no member sees an identical feature vector of the row it predicts.

Test rows are split by twin category: no identical training vector; identical training vectors that all
carry one label (consistent); identical training vectors with at least two labels (conflicting).

protocol compatible  the Assignment 2 compatible partitions (results/ensemble/protocol_compatible.csv)
protocol grouped     identical feature vectors never straddle training and test (protocol_grouped.csv)

Usage: nohup .venv/bin/python code/ensemble/run_protocols.py --protocol grouped > logs/grouped.log 2>&1 &
"""
from __future__ import annotations

import os

for _name in ("NUMBA_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_name, "10")

import argparse  # noqa: E402
import itertools  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import data, grouped, metrics, preprocess, protocol  # noqa: E402
from common.knn import KNNClassifier  # noqa: E402
from common.tree import DecisionTreeClassifier  # noqa: E402
from ensemble import ensemble, harness  # noqa: E402
from ensemble.harness import RESULTS, log  # noqa: E402

N_CLASSES = data.N_CLASSES
COMMON = data.RESULTS_COMMON
FOREST_RUNS = 2 if harness.SMOKE else 5
FINAL_TREES = 10 if harness.SMOKE else 500
STACK_TREES = 8 if harness.SMOKE else 100
MAX_BINS = 4095
N_JOBS = harness.N_JOBS
FOREST_MODELS = {"rf": "plain", "rf_weighted": "weighted", "rf_sampled": "sampled"}
MEMBERS = ("rf", "rf_weighted", "tree", "knn_cw")      # members of the dissimilar vote
CATEGORIES = ("none", "consistent", "conflicting")


def build(protocol_name: str):
    table = data.load_table()
    if protocol_name == "compatible":
        dev_idx, _ = protocol.stratified_holdout(table.y)
        dev = table.take(dev_idx)
        parts = protocol.comparison_partitions(dev.y)
        groups = grouped.feature_groups(dev)
    else:
        gs = grouped.grouped_split(table)
        dev = table.take(gs.dev_idx)
        groups = gs.groups_dev
        parts = grouped.grouped_comparison_partitions(dev.y, groups)
    return dev, parts, groups


def twin_category(groups: np.ndarray, y: np.ndarray, part) -> np.ndarray:
    """0 no training twin, 1 training twins with one label, 2 training twins with at least two labels."""
    g_tr, y_tr = groups[part.train], y[part.train]
    pairs = np.unique(np.stack([g_tr, y_tr], axis=1), axis=0)
    labels_per_group = {}
    for g, c in pairs:
        labels_per_group.setdefault(int(g), set()).add(int(c))
    out = np.zeros(part.test.size, dtype=np.int8)
    for i, g in enumerate(groups[part.test]):
        s = labels_per_group.get(int(g))
        if s is not None:
            out[i] = 1 if len(s) == 1 else 2
    return out


def category_scores(y: np.ndarray, pred: np.ndarray, cat: np.ndarray) -> dict:
    out = {}
    for k, name in enumerate(CATEGORIES):
        mask = cat == k
        out[f"n_{name}"] = int(mask.sum())
        out[f"accuracy_{name}"] = float((pred[mask] == y[mask]).mean()) if mask.any() else float("nan")
    return out


def load_tuned() -> dict:
    tf = json.loads((RESULTS / "tuned_forests.json").read_text())
    tb = json.loads((RESULTS / "tuned_boosting.json").read_text())
    tree = json.loads((COMMON / "tree_best_params.json").read_text())
    knn = json.loads((COMMON / "knn_best_params.json").read_text())
    return {"forests": {m: tf[f]["params"] for m, f in FOREST_MODELS.items()}, "boost": tb,
            "tree": tree, "knn": knn}


def tree_model(tuned: dict, name: str) -> DecisionTreeClassifier:
    tb = tuned["tree"] if name == "tree" else tuned["tree"]["balanced_variant"]
    return DecisionTreeClassifier(tb["criterion"], tb["max_depth"], tb["min_samples_leaf"],
                                  class_weight=tb["class_weight"], seed=protocol.seed_int("baseline-tree"))


def knn_model(tuned: dict, name: str) -> KNNClassifier:
    kb = tuned["knn"][name]
    power = kb["class_weight_power"]
    return KNNClassifier(kb["k"], kb["metric"], kb["weights"], class_weight=None if power == 0 else power,
                         n_classes=N_CLASSES)


def forest(params: dict, purpose: str, trees: int) -> ensemble.RandomForestClassifier:
    return ensemble.RandomForestClassifier(n_estimators=trees, max_bins=MAX_BINS, n_jobs=N_JOBS, purpose=purpose,
                                           keep_trees=False, **params)


def soft_vote(probas: list[np.ndarray]) -> np.ndarray:
    return np.argmax(np.mean(probas, axis=0), axis=1)


def stacked_prediction(tuned, pid, train, test, y_tr, groups_tr, X_tr, X_te, Xk_tr, Xk_te, base_probas) -> np.ndarray:
    """Stacked ensemble: a small forest on out-of-fold class distributions of rf, rf_weighted, tree and knn_cw."""
    inner = grouped.grouped_kfold(y_tr, groups_tr, 5, purpose=f"opt1/stack/{pid}", repeat=0)
    oof = {k: np.zeros((y_tr.size, N_CLASSES)) for k in MEMBERS}
    for f, ip in enumerate(inner):
        for k in ("rf", "rf_weighted"):
            m = forest(tuned["forests"][k], f"opt1/stack/{pid}/{k}/fold{f}", STACK_TREES)
            m.fit(X_tr[ip.train], y_tr[ip.train], probes={"o": X_tr[ip.test]}, checkpoints=[STACK_TREES])
            oof[k][ip.test] = m.probe_proba_["o"]
        t = tree_model(tuned, "tree").fit(X_tr[ip.train], y_tr[ip.train], n_classes=N_CLASSES)
        oof["tree"][ip.test] = t.predict_proba(X_tr[ip.test])
        kn = knn_model(tuned, "knn_cw").fit(Xk_tr[ip.train], y_tr[ip.train])
        oof["knn_cw"][ip.test] = kn.predict_proba(Xk_tr[ip.test])
    meta_X = np.hstack([oof[k] for k in MEMBERS])
    meta_test = np.hstack([base_probas[k] for k in MEMBERS])
    meta = ensemble.RandomForestClassifier(n_estimators=STACK_TREES, max_features="sqrt", max_depth=12,
                                           min_samples_leaf=10, class_weight_power=0.5, max_bins=255,
                                           n_jobs=N_JOBS, purpose=f"opt1/stack/{pid}/meta", keep_trees=False)
    meta.fit(meta_X, y_tr, probes={"t": meta_test}, checkpoints=[STACK_TREES])
    return meta.probe_pred_["t"][STACK_TREES].astype(np.int64)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--protocol", required=True, choices=["compatible", "grouped"])
    ap.add_argument("--partitions", type=int, default=15)
    ap.add_argument("--no-stacking", action="store_true")
    args = ap.parse_args()
    enc = json.loads((RESULTS / "selected_encoding.json").read_text())["encoding"]
    tuned = load_tuned()
    dev, parts, groups = build(args.protocol)
    parts = parts[:args.partitions]
    (RESULTS / f"confusion_{args.protocol}").mkdir(exist_ok=True)
    writer = harness.ResultWriter(RESULTS / f"protocol_{args.protocol}.csv", ["model", "repeat", "fold", "run"])
    for part in parts:
        pid = f"{args.protocol}-{part.repeat}-{part.fold}"
        done = {(m, r) for m in ("tree", "tree_balanced", "knn", "knn_cw", "boost") for r in (0,)
                if writer.finished(model=m, repeat=part.repeat, fold=part.fold, run=0)}
        if len(done) == 5 and writer.finished(model="stack", repeat=part.repeat, fold=part.fold, run=0) \
                and all(writer.finished(model=m, repeat=part.repeat, fold=part.fold, run=FOREST_RUNS - 1)
                        for m in FOREST_MODELS):
            continue
        train, test = dev.take(part.train), dev.take(part.test)
        y_tr, y_te = train.y, test.y
        cat = twin_category(groups, dev.y, part)
        X_tr, X_te = harness.prepare(train, test, enc)
        pre = preprocess.Preprocessor.for_knn()
        Xk_tr, Xk_te = pre.fit_transform(train), pre.transform(test)
        probas, preds = {}, {}

        def record(name, run, pred, seconds, extra=None):
            row = {"model": name, "repeat": part.repeat, "fold": part.fold, "run": run, "protocol": args.protocol}
            row.update(harness.score_row(y_te, pred))
            row.update(category_scores(y_te, pred, cat))
            row["seconds"] = seconds
            row.update(extra or {})
            writer.write([row])
            np.savetxt(RESULTS / f"confusion_{args.protocol}" / f"{name}_{part.repeat}_{part.fold}_{run}.csv",
                       metrics.confusion_matrix(y_te, pred, N_CLASSES), fmt="%d", delimiter=",")

        t0 = time.perf_counter()
        for name, params in tuned["forests"].items():
            for run in range(FOREST_RUNS):
                t1 = time.perf_counter()
                m = forest(params, f"opt1/protocol/{pid}/{name}/run{run}", FINAL_TREES)
                m.fit(X_tr, y_tr, probes={"t": X_te}, checkpoints=[FINAL_TREES], track_oob=(run == 0))
                pred = m.probe_pred_["t"][FINAL_TREES].astype(np.int64)
                if run == 0:
                    probas[name] = m.probe_proba_["t"].astype(np.float64)
                    preds[name] = pred
                if not writer.finished(model=name, repeat=part.repeat, fold=part.fold, run=run):
                    extra = {}
                    if run == 0:
                        extra.update(harness.oob_row(y_tr, m.oob_pred_[FINAL_TREES]))
                    extra.update(harness.diversity_row(m.probe_members_["t"], y_te))
                    record(name, run, pred, time.perf_counter() - t1, extra)
        for name in ("tree", "tree_balanced"):
            t1 = time.perf_counter()
            m = tree_model(tuned, name).fit(X_tr, y_tr, n_classes=N_CLASSES)
            pred = m.predict(X_te)
            probas[name], preds[name] = m.predict_proba(X_te), pred
            if not writer.finished(model=name, repeat=part.repeat, fold=part.fold, run=0):
                record(name, 0, pred, time.perf_counter() - t1)
        for name in ("knn", "knn_cw"):
            t1 = time.perf_counter()
            m = knn_model(tuned, name).fit(Xk_tr, y_tr)
            probas[name] = m.predict_proba(Xk_te)
            preds[name] = np.argmax(probas[name], axis=1)
            if not writer.finished(model=name, repeat=part.repeat, fold=part.fold, run=0):
                record(name, 0, preds[name], time.perf_counter() - t1)
        if not writer.finished(model="boost", repeat=part.repeat, fold=part.fold, run=0):
            t1 = time.perf_counter()
            b = tuned["boost"]
            ab = ensemble.AdaBoostClassifier(n_estimators=b["n_estimators"], max_bins=MAX_BINS,
                                             purpose=f"opt1/protocol/{pid}/boost", **b["params"])
            ab.fit(X_tr, y_tr, probes={"t": X_te}, checkpoints=[b["n_estimators"]])
            pred = ab.probe_pred_["t"][b["n_estimators"]].astype(np.int64)
            record("boost", 0, pred, time.perf_counter() - t1, harness.diversity_row(ab.probe_members_["t"], y_te))
        # dissimilar ensembles
        votes = {"vote_forests": ("rf", "rf_weighted"), "vote_rf_knn": ("rf", "knn_cw"),
                 "vote_all": MEMBERS}
        for drop in MEMBERS:
            votes[f"vote_without_{drop}"] = tuple(k for k in MEMBERS if k != drop)
        for name, members in votes.items():
            if not writer.finished(model=name, repeat=part.repeat, fold=part.fold, run=0):
                record(name, 0, soft_vote([probas[k] for k in members]), 0.0)
        if not args.no_stacking and not writer.finished(model="stack", repeat=part.repeat, fold=part.fold, run=0):
            t1 = time.perf_counter()
            pred = stacked_prediction(tuned, pid, train, test, y_tr, groups[part.train], X_tr, X_te, Xk_tr, Xk_te, probas)
            record("stack", 0, pred, time.perf_counter() - t1)
        done_rows = len(writer.done)
        log(f"{pid}: {time.perf_counter() - t0:.0f}s; rf {harness.score_row(y_te, preds['rf'])['macro_f1']:.4f} "
            f"vote_all {harness.score_row(y_te, soft_vote([probas[k] for k in MEMBERS]))['macro_f1']:.4f} rows {done_rows}")


if __name__ == "__main__":
    main()
