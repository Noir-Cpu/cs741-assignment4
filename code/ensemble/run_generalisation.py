"""Final fit on the development set and one evaluation on the generalisation set, with independent runs.

protocol compatible  development set D and generalisation set G of the shared protocol
protocol grouped     development set D_g and generalisation set G_g (identical feature vectors stay together)

The three tuned forests run RF_RUNS times with independent random streams, boosting and the baselines once.
Dissimilar ensembles (votes and one stacked ensemble) use run zero of the forests. Predicted labels of every
run are stored as .npy files (not tracked) so that bootstrap intervals over the test instances can be computed
by analyse_generalisation.py; confusion matrices and scores are written to CSV.

Usage: nohup .venv/bin/python code/ensemble/run_generalisation.py --protocol compatible > logs/G.log 2>&1 &
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import data, grouped, metrics, preprocess, protocol  # noqa: E402
from ensemble import ensemble, harness  # noqa: E402
from ensemble import run_protocols as rp  # noqa: E402
from ensemble.harness import RESULTS, log  # noqa: E402

N_CLASSES = data.N_CLASSES
RF_RUNS = {"compatible": 2 if harness.SMOKE else 30, "grouped": 2 if harness.SMOKE else 10}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--protocol", required=True, choices=["compatible", "grouped"])
    ap.add_argument("--no-stacking", action="store_true")
    args = ap.parse_args()
    enc = json.loads((RESULTS / "selected_encoding.json").read_text())["encoding"]
    tuned = rp.load_tuned()
    table = data.load_table()
    if args.protocol == "compatible":
        dev_idx, gen_idx = protocol.stratified_holdout(table.y)
        groups_all = grouped.feature_groups(table)
        groups_dev = groups_all[dev_idx]
    else:
        gs = grouped.grouped_split(table)
        dev_idx, gen_idx, groups_dev = gs.dev_idx, gs.gen_idx, gs.groups_dev
    dev, gen = table.take(dev_idx), table.take(gen_idx)
    pred_dir = RESULTS / f"predG_{args.protocol}"
    pred_dir.mkdir(exist_ok=True)
    writer = harness.ResultWriter(RESULTS / f"generalisation_{args.protocol}.csv", ["model", "run"])
    X_tr, X_te = harness.prepare(dev, gen, enc)
    pre = preprocess.Preprocessor.for_knn()
    Xk_tr, Xk_te = pre.fit_transform(dev), pre.transform(gen)
    probas, runs = {}, RF_RUNS[args.protocol]

    def record(name, run, pred, fit_s, pred_s, extra=None):
        row = {"model": name, "run": run, "protocol": args.protocol}
        row.update(harness.score_row(gen.y, pred))
        row.update(extra or {})
        row["fit_seconds"], row["predict_seconds"] = fit_s, pred_s
        writer.write([row])
        np.save(pred_dir / f"{name}_run{run}.npy", pred.astype(np.uint8))
        np.savetxt(RESULTS / f"confusion_G_{args.protocol}_{name}_run{run}.csv",
                   metrics.confusion_matrix(gen.y, pred, N_CLASSES), fmt="%d", delimiter=",")
        log(f"G {args.protocol} {name} run {run}: macro-F1 {row['macro_f1']:.4f} acc {row['accuracy']:.4f} "
            f"fit {fit_s:.0f}s")

    for name, params in tuned["forests"].items():
        for run in range(runs):
            if writer.finished(model=name, run=run) and not (run == 0 and name not in probas):
                continue
            m = ensemble.RandomForestClassifier(n_estimators=rp.FINAL_TREES, max_bins=rp.MAX_BINS, n_jobs=rp.N_JOBS,
                                                purpose=f"opt1/G/{args.protocol}/{name}/run{run}", keep_trees=True,
                                                **params)
            m.fit(X_tr, dev.y, track_oob=(run == 0), checkpoints=[rp.FINAL_TREES])
            t0 = time.perf_counter()
            proba = m.predict_proba(X_te)
            pred = np.argmax(proba, axis=1)
            pred_s = time.perf_counter() - t0
            if run == 0:
                probas[name] = proba
            if not writer.finished(model=name, run=run):
                extra = {"mean_nodes": float(m.tree_nodes_.mean())}
                if run == 0:
                    extra.update(harness.oob_row(dev.y, m.oob_pred_[rp.FINAL_TREES]))
                record(name, run, pred, m.fit_seconds_, pred_s, extra)
            del m
    for name in ("tree", "tree_balanced"):
        t0 = time.perf_counter()
        m = rp.tree_model(tuned, name).fit(X_tr, dev.y, n_classes=N_CLASSES)
        fit_s = time.perf_counter() - t0
        probas[name] = m.predict_proba(X_te)
        if not writer.finished(model=name, run=0):
            record(name, 0, m.predict(X_te), fit_s, 0.0)
    for name in ("knn", "knn_cw"):
        t0 = time.perf_counter()
        m = rp.knn_model(tuned, name).fit(Xk_tr, dev.y)
        t1 = time.perf_counter()
        probas[name] = m.predict_proba(Xk_te)
        if not writer.finished(model=name, run=0):
            record(name, 0, np.argmax(probas[name], axis=1), t1 - t0, time.perf_counter() - t1)
    if not writer.finished(model="boost", run=0):
        b = tuned["boost"]
        ab = ensemble.AdaBoostClassifier(n_estimators=b["n_estimators"], max_bins=rp.MAX_BINS,
                                         purpose=f"opt1/G/{args.protocol}/boost", **b["params"])
        ab.fit(X_tr, dev.y, keep_trees=True)
        t0 = time.perf_counter()
        pred = ab.predict(X_te)
        record("boost", 0, pred, ab.fit_seconds_, time.perf_counter() - t0)
    votes = {"vote_forests": ("rf", "rf_weighted"), "vote_rf_knn": ("rf", "knn_cw"), "vote_all": rp.MEMBERS}
    for drop in rp.MEMBERS:
        votes[f"vote_without_{drop}"] = tuple(k for k in rp.MEMBERS if k != drop)
    for name, members in votes.items():
        if not writer.finished(model=name, run=0):
            record(name, 0, rp.soft_vote([probas[k] for k in members]), 0.0, 0.0)
    if not args.no_stacking and not writer.finished(model="stack", run=0):
        t0 = time.perf_counter()
        pred = rp.stacked_prediction(tuned, f"G-{args.protocol}", dev, gen, dev.y, groups_dev, X_tr, X_te, Xk_tr, Xk_te,
                                     probas)
        record("stack", 0, pred, time.perf_counter() - t0, 0.0)


if __name__ == "__main__":
    main()
