"""One-factor and two-factor sweeps of the random forest on the five tuning folds of D.

Experiments (see the Empirical Procedure of the report):

  encoding   nominal-feature encoding: one-hot against ordinal frequency rank
  size       ensemble size: test and out-of-bag scores and diversity against the
             number of trees (up to 500), base configuration or tuned configuration
  features   number of features tried per split, strength and diversity of the trees
  structure  tree depth against minimum leaf size
  balance    class-weight power against sampling scheme

Every configuration is fitted on the training part of each of the five tuning folds
and evaluated on the test part, with two independent runs (random streams) per fold.
Results are appended to results/ensemble/<experiment>.csv, and finished units are
skipped when the script restarts.

Usage (from the repository root, thread counts are set below):
    nohup .venv/bin/python code/ensemble/run_forest_sweeps.py --experiment features \
        > logs/features.log 2>&1 &
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

from common import protocol  # noqa: E402
from ensemble import diversity, ensemble, harness  # noqa: E402
from ensemble.harness import RESULTS, log  # noqa: E402

N_RUNS = 2
N_TREES = 8 if harness.SMOKE else 200
SIZE_TREES = 20 if harness.SMOKE else 2000
SIZE_CHECKPOINTS = [1, 2, 3, 5, 7, 10, 15, 20, 30, 50, 70, 100, 150, 200, 300, 400, 500, 700, 1000, 1500, 2000]
MAX_BINS = 4095
N_JOBS = harness.N_JOBS

DEPTHS = (3, 5, 8, 12, 20, None)
LEAVES = (1, 2, 5, 10, 25, 50)
CLASS_POWERS = (0.0, 0.5, 1.0)
SAMPLINGS = (("bootstrap", 0.0, None), ("bootstrap", 0.5, None), ("bootstrap", 1.0, None),
             ("under", 0.0, None), ("under", 0.0, 1000))


def encoding_choice() -> str:
    path = RESULTS / "selected_encoding.json"
    if not path.exists():
        raise SystemExit("run the encoding experiment first (results/ensemble/selected_encoding.json missing)")
    return json.loads(path.read_text())["encoding"]


def feature_counts(p: int) -> list:
    root = max(1, int(np.sqrt(p)))
    return sorted({1, 2, 4, root, 16, 32, p})


def configs_for(experiment: str, n_features: int, size_config: str):
    """List of (name, forest parameter overrides, n_trees, checkpoints)."""
    base = harness.default_forest_params()
    out = []
    if experiment == "encoding":
        for enc in harness.features.ENCODINGS:
            out.append((enc, dict(base), N_TREES, [N_TREES], enc))
        return out
    enc = encoding_choice()
    if experiment == "size":
        params = dict(base)
        if size_config != "base":
            params.update(json.loads((RESULTS / "tuned_forests.json").read_text())[size_config]["params"])
        out.append((size_config, params, SIZE_TREES, SIZE_CHECKPOINTS, enc))
    elif experiment == "features":
        for m in feature_counts(n_features):
            p = dict(base)
            p["max_features"] = m
            out.append((f"m={m}", p, N_TREES, [N_TREES], enc))
    elif experiment == "structure":
        for d in DEPTHS:
            for leaf in LEAVES:
                p = dict(base)
                p["max_depth"], p["min_samples_leaf"] = d, leaf
                out.append((f"depth={d},leaf={leaf}", p, N_TREES, [N_TREES], enc))
    elif experiment == "balance":
        for sampling, power, per_class in SAMPLINGS:
            for alpha in CLASS_POWERS:
                p = dict(base)
                p["sampling"], p["sampling_power"], p["class_sample_size"] = sampling, power, per_class
                p["class_weight_power"] = alpha
                tag = (f"bootstrap{power}" if sampling == "bootstrap"
                       else f"under{per_class if per_class else 'min'}")
                out.append((f"{tag},alpha={alpha}", p, N_TREES, [N_TREES], enc))
    else:
        raise SystemExit(f"unknown experiment {experiment}")
    return out


def run_unit(name, params, n_trees, checkpoints, fold_data, experiment, fold, run, encoding):
    Xtr, ytr, Xte, yte = fold_data
    purpose = f"opt1/{experiment}/fold{fold}/run{run}"
    rf = ensemble.RandomForestClassifier(n_estimators=n_trees, max_bins=MAX_BINS, n_jobs=N_JOBS,
                                         purpose=purpose, keep_trees=False, **params)
    rf.fit(Xtr, ytr, probes={"test": Xte}, checkpoints=checkpoints, track_oob=True)
    rows = []
    members = rf.probe_members_["test"]
    for t in sorted(rf.probe_pred_["test"]):
        pred = rf.probe_pred_["test"][t].astype(np.int64)
        row = {"experiment": experiment, "config": name, "encoding": encoding, "fold": fold, "run": run,
               "n_estimators": t, **harness.param_columns(params)}
        row.update(harness.score_row(yte, pred))
        row.update(harness.oob_row(ytr, rf.oob_pred_[t]))
        row.update(harness.diversity_row(members[:, :t], yte))
        row["mean_nodes"] = float(rf.tree_nodes_[:t].mean())
        row["mean_depth"] = float(rf.tree_depths_[:t].mean())
        row["fit_seconds"] = rf.fit_seconds_ if t == n_trees else float("nan")
        rows.append(row)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experiment", required=True,
                    choices=["encoding", "size", "features", "structure", "balance"])
    ap.add_argument("--size-config", default="base", choices=["base", "plain", "weighted", "sampled", "both"])
    ap.add_argument("--folds", type=int, default=protocol.N_FOLDS, help="number of tuning folds to use")
    ap.add_argument("--runs", type=int, default=N_RUNS)
    ap.add_argument("--tag", default="", help="suffix of the output file name")
    args = ap.parse_args()

    table, dev, _ = harness.load_split()
    parts = protocol.tuning_partitions(dev.y)[:args.folds]
    experiment = args.experiment
    name_out = experiment + (f"_{args.size_config}" if experiment == "size" else "") + args.tag
    if experiment == "size" and args.size_config == "base" and not args.tag:
        name_out = "size_base2000"
    writer = harness.ResultWriter(RESULTS / f"{name_out}.csv",
                                  ["experiment", "config", "fold", "run", "n_estimators"])
    started = time.perf_counter()
    cache: dict[str, tuple] = {}
    n_feat = 0
    if experiment == "features":
        n_feat = harness.features.ForestFeatures(encoding_choice()).fit(dev).n_features_
    for part in parts:
        train, test = dev.take(part.train), dev.take(part.test)
        cache.clear()
        for name, params, n_trees, checkpoints, enc in configs_for(experiment, n_feat, args.size_config):
            if enc not in cache:
                Xtr, Xte = harness.prepare(train, test, enc)
                cache[enc] = (Xtr, train.y, Xte, test.y)
            for run in range(args.runs):
                if writer.finished(experiment=experiment, config=name, fold=part.fold, run=run,
                                   n_estimators=n_trees):
                    continue
                t0 = time.perf_counter()
                rows = run_unit(name, params, n_trees, checkpoints, cache[enc], experiment,
                                part.fold, run, enc)
                writer.write(rows)
                last = rows[-1]
                log(f"{experiment} fold {part.fold} run {run} {name}: macro-F1 {last['macro_f1']:.4f} "
                    f"acc {last['accuracy']:.4f} oob {last['oob_macro_f1']:.4f} ({time.perf_counter() - t0:.0f}s)")
    log(f"{experiment} finished in {(time.perf_counter() - started) / 60:.1f} minutes")

    if experiment == "encoding":
        import pandas as pd
        df = pd.read_csv(RESULTS / f"{name_out}.csv")
        summary = df.groupby("config")["macro_f1"].agg(["mean", "std"]).reset_index()
        log("\n" + summary.to_string())
        means = dict(zip(summary["config"], summary["mean"]))
        choice = "ordinal" if means.get("ordinal", 0) > means.get("onehot", 0) + 0.005 else "onehot"
        (RESULTS / "selected_encoding.json").write_text(json.dumps(
            {"encoding": choice, "rule": "ordinal if its mean macro-F1 exceeds the one-hot mean by more than 0.005",
             "means": means}, indent=2))
        log(f"selected encoding: {choice}")


if __name__ == "__main__":
    main()
