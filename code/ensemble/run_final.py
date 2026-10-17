"""Final evaluation on the shared protocol: 15 comparison partitions and the generalisation set G.

stage compare       the tuned ensembles and the four baselines on the 15 comparison
                    partitions of D (5-fold cross-validation repeated 3 times). The
                    baselines are refitted here with the control parameters of
                    results/common, so that their predictions can be split into test
                    rows with an exact twin in the training part and rows without; the
                    refitted scores are checked against results/common.
stage generalise    final fit on all of D and one evaluation on G. The random forests
                    repeat the fit with independent random streams (ten runs); the
                    baselines keep their scores from results/common.

Models: rf (tuned, no class treatment), rf_balanced (tuned, with class treatment),
boost (tuned SAMME), tree, tree_balanced, knn, knn_cw.

Usage: nohup .venv/bin/python code/ensemble/run_final.py --stage compare > logs/final.log 2>&1 &
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

from common import data, metrics, preprocess, protocol  # noqa: E402
from common.knn import KNNClassifier  # noqa: E402
from common.tree import DecisionTreeClassifier  # noqa: E402
from ensemble import ensemble, harness  # noqa: E402
from ensemble.harness import RESULTS, log  # noqa: E402

N_CLASSES = data.N_CLASSES
N_TREES_FINAL = 10 if harness.SMOKE else 500
RF_RUNS_COMPARE = 3
RF_RUNS_G = 2 if harness.SMOKE else 30
MAX_BINS = 4095
N_JOBS = harness.N_JOBS
COMMON = data.RESULTS_COMMON
ENSEMBLES = ("rf", "rf_balanced", "boost")
BASELINES = ("tree", "tree_balanced", "knn", "knn_cw")


def twin_mask(train: data.Table, test: data.Table) -> np.ndarray:
    """True for test rows whose complete feature vector also occurs in the training rows."""
    def digest(t: data.Table) -> np.ndarray:
        frame = pd.DataFrame(np.hstack([t.numeric, t.nominal.astype(np.float64)]))
        return pd.util.hash_pandas_object(frame, index=False).to_numpy()
    return np.isin(digest(test), digest(train))


def group_scores(y: np.ndarray, pred: np.ndarray, twin: np.ndarray) -> dict:
    out = {"twin_share": float(twin.mean())}
    for label, mask in (("twin", twin), ("new", ~twin)):
        cm = metrics.confusion_matrix(y[mask], pred[mask], N_CLASSES)
        s = metrics.scores_from_confusion(cm)
        out[f"accuracy_{label}"] = s["accuracy"]
        out[f"balanced_accuracy_{label}"] = s["balanced_accuracy"]
    return out


def baseline_model(name: str):
    if name.startswith("tree"):
        tb = json.loads((COMMON / "tree_best_params.json").read_text())
        tb = tb if name == "tree" else tb["balanced_variant"]
        return DecisionTreeClassifier(tb["criterion"], tb["max_depth"], tb["min_samples_leaf"],
                                      class_weight=tb["class_weight"], seed=protocol.seed_int("baseline-tree"))
    kb = json.loads((COMMON / "knn_best_params.json").read_text())[name]
    power = kb["class_weight_power"]
    return KNNClassifier(kb["k"], kb["metric"], kb["weights"], class_weight=None if power == 0 else power,
                         n_classes=N_CLASSES)


def load_tuned() -> dict:
    forest = json.loads((RESULTS / "tuned_forest.json").read_text())
    boost = json.loads((RESULTS / "tuned_boosting.json").read_text())
    return {"rf": forest["plain"]["params"], "rf_balanced": forest["balanced"]["params"], "boost": boost}


def ensemble_model(name: str, tuned: dict, purpose: str, rounds_or_trees: int | None = None):
    if name == "boost":
        b = tuned["boost"]
        return ensemble.AdaBoostClassifier(n_estimators=b["n_estimators"], max_bins=MAX_BINS, purpose=purpose,
                                           **b["params"])
    return ensemble.RandomForestClassifier(n_estimators=rounds_or_trees or N_TREES_FINAL, max_bins=MAX_BINS,
                                           n_jobs=N_JOBS, purpose=purpose, keep_trees=False, **tuned[name])


def compare(dev: data.Table, enc: str, tuned: dict, only: list[str] | None, n_parts: int = 15) -> None:
    parts = protocol.comparison_partitions(dev.y)[:n_parts]
    names = list(only) if only else list(ENSEMBLES + BASELINES)
    writer = harness.ResultWriter(RESULTS / "comparison_runs.csv", ["model", "repeat", "fold", "run"])
    for part in parts:
        pending = [(n, r) for n in names for r in range(RF_RUNS_COMPARE if n.startswith("rf") else 1)
                   if not writer.finished(model=n, repeat=part.repeat, fold=part.fold, run=r)]
        if not pending:
            continue
        train, test = dev.take(part.train), dev.take(part.test)
        twin = twin_mask(train, test)
        pid = f"part{part.repeat}-{part.fold}"
        cache = {}

        def features_for(kind: str):
            if kind not in cache:
                if kind == "ens":
                    cache[kind] = harness.prepare(train, test, enc)
                else:
                    pre = (preprocess.Preprocessor.for_knn() if kind == "knn"
                           else preprocess.Preprocessor.for_tree())
                    cache[kind] = (pre.fit_transform(train), pre.transform(test))
            return cache[kind]

        for name, run in pending:
            t0 = time.perf_counter()
            extra = {}
            if name in ENSEMBLES:
                Xtr, Xte = features_for("ens")
                model = ensemble_model(name, tuned, f"opt1/final/{pid}/run{run}")
                if name == "boost":
                    model.fit(Xtr, train.y, probes={"test": Xte})
                else:
                    model.fit(Xtr, train.y, probes={"test": Xte}, track_oob=True)
                    extra.update(harness.oob_row(train.y, model.oob_pred_[N_TREES_FINAL]))
                last = max(model.probe_pred_["test"])
                pred = model.probe_pred_["test"][last].astype(np.int64)
                members = model.probe_members_["test"]
                extra.update(harness.diversity_row(members, test.y))
                fit_seconds = model.fit_seconds_
            else:
                kind = "knn" if name.startswith("knn") else "tree"
                Xtr, Xte = features_for(kind)
                model = baseline_model(name)
                t1 = time.perf_counter()
                if name.startswith("tree"):
                    model.fit(Xtr, train.y, n_classes=N_CLASSES)
                else:
                    model.fit(Xtr, train.y)
                fit_seconds = time.perf_counter() - t1
                pred = model.predict(Xte)
            row = {"model": name, "repeat": part.repeat, "fold": part.fold, "run": run, "encoding": enc}
            row.update(harness.score_row(test.y, pred))
            row.update(group_scores(test.y, pred, twin))
            row.update(extra)
            row["fit_seconds"] = fit_seconds
            row["total_seconds"] = time.perf_counter() - t0
            writer.write([row])
            np.savetxt(RESULTS / "confusion_runs" / f"{name}_{part.repeat}_{part.fold}_{run}.csv",
                       metrics.confusion_matrix(test.y, pred, N_CLASSES), fmt="%d", delimiter=",")
            log(f"compare {name} {pid} run {run}: macro-F1 {row['macro_f1']:.4f} acc {row['accuracy']:.4f} "
                f"({row['total_seconds']:.0f}s)")


def generalise(dev: data.Table, gen: data.Table, enc: str, tuned: dict) -> None:
    writer = harness.ResultWriter(RESULTS / "generalisation_runs.csv", ["model", "run"])
    Xd, Xg = harness.prepare(dev, gen, enc)
    for name in ENSEMBLES:
        for run in range(RF_RUNS_G if name.startswith("rf") else 1):
            if writer.finished(model=name, run=run):
                continue
            model = ensemble_model(name, tuned, f"opt1/final-G/run{run}")
            if name == "boost":
                model.fit(Xd, dev.y, keep_trees=True)
                fit_seconds = model.fit_seconds_
            else:
                model.keep_trees = True
                model.fit(Xd, dev.y, track_oob=True, checkpoints=[N_TREES_FINAL])
                fit_seconds = model.fit_seconds_
            t0 = time.perf_counter()
            if name == "boost":
                pred = model.predict(Xg)
            else:
                proba = model.predict_proba(Xg)
                pred = np.argmax(proba, axis=1)
            predict_seconds = time.perf_counter() - t0
            cm = metrics.confusion_matrix(gen.y, pred, N_CLASSES)
            row = {"model": name, "run": run, "encoding": enc}
            row.update(harness.score_row(gen.y, pred))
            if name.startswith("rf"):
                row.update(harness.oob_row(dev.y, model.oob_pred_[N_TREES_FINAL]))
                row["mean_nodes"] = float(model.tree_nodes_.mean())
                row["mean_depth"] = float(model.tree_depths_.mean())
                imp = pd.DataFrame({"feature": range(len(model.feature_importances_)),
                                    "importance": model.feature_importances_})
                if run == 0:
                    ff = harness.features.ForestFeatures(enc).fit(dev)
                    imp["feature"] = ff.feature_names_
                    imp.sort_values("importance", ascending=False).to_csv(
                        RESULTS / f"{name}_feature_importances.csv", index=False)
            row["fit_seconds"] = fit_seconds
            row["predict_seconds"] = predict_seconds
            writer.write([row])
            np.savetxt(RESULTS / f"confusion_G_{name}_run{run}.csv", cm, fmt="%d", delimiter=",")
            log(f"G {name} run {run}: macro-F1 {row['macro_f1']:.4f} acc {row['accuracy']:.4f} "
                f"fit {fit_seconds:.0f}s predict {predict_seconds:.1f}s")
            del model


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True, choices=["compare", "generalise"])
    ap.add_argument("--models", nargs="*", help="restrict the compare stage to these models")
    ap.add_argument("--partitions", type=int, default=15, help="number of comparison partitions to use")
    args = ap.parse_args()
    (RESULTS / "confusion_runs").mkdir(parents=True, exist_ok=True)
    enc = json.loads((RESULTS / "selected_encoding.json").read_text())["encoding"]
    table, dev, gen = harness.load_split()
    tuned = load_tuned()
    log(f"encoding {enc}; tuned: {json.dumps(tuned, default=str)}")
    if args.stage == "compare":
        compare(dev, enc, tuned, args.models, args.partitions)
    else:
        generalise(dev, gen, enc, tuned)


if __name__ == "__main__":
    main()
