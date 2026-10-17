"""Tuning and ensemble-size curve of adaptive boosting (SAMME) on the five tuning folds of D.

stage grid   tree depth x shrinkage x class-weight power; every configuration is
             fitted for up to 200 rounds and scored at the checkpoints 25, 50, 100, 150
             and 200 rounds, so the number of rounds is tuned from the same fit.
             The best configuration (highest mean macro-F1 over the folds) goes to
             results/ensemble/tuned_boosting.json.
stage size   the tuned configuration for up to 500 rounds with scores, training error
             and model diversity at many checkpoints.

Boosting rounds depend on each other, so each fit is single-threaded and the five
folds run in parallel, one thread per fold. Weak learners use all features at every
split, so the fit is deterministic and one run per fold is enough.

Usage: nohup .venv/bin/python code/ensemble/run_boosting.py --stage grid > logs/boost_grid.log 2>&1 &
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
from concurrent.futures import ThreadPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import protocol  # noqa: E402
from ensemble import ensemble, harness  # noqa: E402
from ensemble.harness import RESULTS, log  # noqa: E402

DEPTHS = (3, 6, 9, 12, 18)
RATES = (0.1, 0.3, 1.0)
CLASS_POWERS = (0.0, 1.0)
GRID_ROUNDS = 200
GRID_CHECKPOINTS = [25, 50, 100, 150, 200]
SIZE_ROUNDS = 1000
SIZE_CHECKPOINTS = [1, 2, 3, 5, 7, 10, 15, 20, 30, 50, 70, 100, 150, 200, 300, 400, 500, 700, 1000]
MAX_BINS = 4095
DEPTHS2, RATES2, CLASS_POWERS2 = (24, 32), (0.03, 0.1, 0.3), (0.0, 0.5)
GRID2_ROUNDS, GRID2_CHECKPOINTS = 300, [50, 100, 200, 300]
if harness.SMOKE:
    DEPTHS, RATES, GRID_ROUNDS, GRID_CHECKPOINTS, SIZE_ROUNDS = (2,), (0.3,), 6, [3, 6], 8
    DEPTHS2, RATES2, CLASS_POWERS2, GRID2_ROUNDS, GRID2_CHECKPOINTS = (3,), (0.3,), (0.0,), 6, [3, 6]


def fit_fold(params: dict, rounds: int, checkpoints, fold_data, fold: int, purpose: str):
    Xtr, ytr, Xte, yte = fold_data
    ab = ensemble.AdaBoostClassifier(n_estimators=rounds, max_bins=MAX_BINS, purpose=purpose, **params)
    ab.fit(Xtr, ytr, probes={"test": Xte}, checkpoints=checkpoints)
    rows = []
    for t in sorted(ab.probe_pred_["test"]):
        pred = ab.probe_pred_["test"][t].astype(np.int64)
        row = {"fold": fold, "n_estimators": t, "n_rounds_fitted": ab.n_rounds_, **params}
        row.update(harness.score_row(yte, pred))
        row["train_error"] = ab.train_error_[t]
        row.update(harness.diversity_row(ab.probe_members_["test"][:, :min(t, ab.n_rounds_)], yte))
        row["mean_alpha"] = float(ab.alphas_[:t].mean())
        row["mean_weighted_error"] = float(ab.errors_[:t].mean())
        row["fit_seconds"] = ab.fit_seconds_
        rows.append(row)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True, choices=["grid", "grid2", "size"])
    args = ap.parse_args()
    enc = json.loads((RESULTS / "selected_encoding.json").read_text())["encoding"]
    table, dev, _ = harness.load_split()
    parts = protocol.tuning_partitions(dev.y)
    fold_data = []
    for part in parts:
        train, test = dev.take(part.train), dev.take(part.test)
        Xtr, Xte = harness.prepare(train, test, enc)
        fold_data.append((Xtr, train.y, Xte, test.y))

    if args.stage in ("grid", "grid2"):
        second = args.stage == "grid2"
        depths, rates, powers = (DEPTHS2, RATES2, CLASS_POWERS2) if second else (DEPTHS, RATES, CLASS_POWERS)
        rounds, cps = (GRID2_ROUNDS, GRID2_CHECKPOINTS) if second else (GRID_ROUNDS, GRID_CHECKPOINTS)
        writer = harness.ResultWriter(RESULTS / ("boosting_grid2.csv" if second else "boosting_grid.csv"),
                                      ["max_depth", "learning_rate", "class_weight_power", "fold", "n_estimators"])
        started = time.perf_counter()
        for depth, rate, alpha in itertools.product(depths, rates, powers):
            params = {"max_depth": depth, "learning_rate": rate, "class_weight_power": alpha}
            if all(writer.finished(**params, fold=f, n_estimators=rounds) for f in range(len(parts))):
                continue
            t0 = time.perf_counter()
            with ThreadPoolExecutor(len(parts)) as pool:
                futures = [pool.submit(fit_fold, params, rounds, cps, fold_data[f], f,
                                       f"opt1/boost-grid{'2' if second else ''}/fold{f}") for f in range(len(parts))]
                rows = [r for fut in futures for r in fut.result()]
            writer.write(rows)
            last = [r for r in rows if r["n_estimators"] == rounds]
            log(f"grid depth {depth} rate {rate} alpha {alpha}: macro-F1 at {rounds} rounds "
                f"{np.mean([r['macro_f1'] for r in last]):.4f} ({time.perf_counter() - t0:.0f}s)")
        frames = [pd.read_csv(RESULTS / n) for n in ("boosting_grid.csv", "boosting_grid2.csv")
                  if (RESULTS / n).exists()]
        df = pd.concat(frames, ignore_index=True)
        keys = ["max_depth", "learning_rate", "class_weight_power", "n_estimators"]
        agg = df.groupby(keys)[["macro_f1", "accuracy", "balanced_accuracy"]].agg(["mean", "std"])
        agg.columns = [f"{a}_{b}" for a, b in agg.columns]
        agg = agg.reset_index()
        agg.to_csv(RESULTS / "boosting_grid_summary.csv", index=False)
        best = agg.loc[agg["macro_f1_mean"].idxmax()]
        out = {"params": {"max_depth": int(best["max_depth"]), "learning_rate": float(best["learning_rate"]),
                          "class_weight_power": float(best["class_weight_power"])},
               "n_estimators": int(best["n_estimators"]), "macro_f1_mean": float(best["macro_f1_mean"]),
               "macro_f1_std": float(best["macro_f1_std"]), "n_configurations": int(len(agg)),
               "grid_rounds": GRID_ROUNDS, "grid2_rounds": GRID2_ROUNDS, "elapsed_minutes": (time.perf_counter() - started) / 60}
        (RESULTS / "tuned_boosting.json").write_text(json.dumps(out, indent=2))
        log(f"selected {out}")
    else:
        tuned = json.loads((RESULTS / "tuned_boosting.json").read_text())
        writer = harness.ResultWriter(RESULTS / "boosting_size.csv", ["fold", "n_estimators"])
        if writer.finished(fold=0, n_estimators=SIZE_ROUNDS):
            return
        t0 = time.perf_counter()
        with ThreadPoolExecutor(len(parts)) as pool:
            futures = [pool.submit(fit_fold, tuned["params"], SIZE_ROUNDS, SIZE_CHECKPOINTS, fold_data[f], f,
                                   f"opt1/boost-size/fold{f}") for f in range(len(parts))]
            rows = [r for fut in futures for r in fut.result()]
        writer.write(rows)
        log(f"size curve finished in {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main()
