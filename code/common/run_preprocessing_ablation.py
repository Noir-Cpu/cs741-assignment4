"""Preprocessing ablation on the shared tuning folds (inside D, never G).

Each preprocessing step that the report justifies is switched off once, and
the tuned baseline model is re-evaluated by stratified 5-fold cross-validation
on the tuning folds. The results back the justification of every step with
measurements.

k-nearest neighbours uses the tuned configuration of results/common/knn_best_params.json
(model "knn"); the classification tree uses results/common/tree_best_params.json.

Usage (from the repository root, after run_baselines.py):
    nohup .venv/bin/python code/common/run_preprocessing_ablation.py > logs/ablation.log 2>&1 &
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

for _name in ("NUMBA_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_name, "20")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import data, metrics, preprocess, protocol  # noqa: E402
from common.knn import KNNClassifier  # noqa: E402
from common.tree import DecisionTreeClassifier  # noqa: E402

N_CLASSES = data.N_CLASSES

KNN_VARIANTS = {
    "full pipeline": dict(drop_redundant=True, log_transform=True, standardise=True, group_rare=True),
    "no logarithm": dict(drop_redundant=True, log_transform=False, standardise=True, group_rare=True),
    "no standardisation": dict(drop_redundant=True, log_transform=True, standardise=False, group_rare=True),
    "no logarithm and no standardisation": dict(drop_redundant=True, log_transform=False, standardise=False,
                                                group_rare=True),
    "redundant columns kept": dict(drop_redundant=False, log_transform=True, standardise=True, group_rare=True),
    "no rare-category grouping": dict(drop_redundant=True, log_transform=True, standardise=True, group_rare=False),
}

TREE_VARIANTS = {
    "raw numeric columns": dict(drop_redundant=False, log_transform=False, standardise=False, group_rare=True),
    "logarithm and standardisation": dict(drop_redundant=False, log_transform=True, standardise=True,
                                          group_rare=True),
    "redundant columns dropped": dict(drop_redundant=True, log_transform=False, standardise=False,
                                      group_rare=True),
    "no rare-category grouping": dict(drop_redundant=False, log_transform=False, standardise=False,
                                      group_rare=False),
    "255 bins per feature": dict(drop_redundant=False, log_transform=False, standardise=False,
                                 group_rare=True, max_bins=255),
    "4095 bins per feature": dict(drop_redundant=False, log_transform=False, standardise=False,
                                  group_rare=True, max_bins=4095),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(data.RESULTS_COMMON))
    args = parser.parse_args()
    out = Path(args.out)
    knn_best = json.loads((out / "knn_best_params.json").read_text())["knn"]
    tree_best = json.loads((out / "tree_best_params.json").read_text())

    table = data.load_table()
    dev_index, _ = protocol.stratified_holdout(table.y)
    dev = table.take(dev_index)
    parts = protocol.tuning_partitions(dev.y)
    rows = []
    for model, variants in (("knn", KNN_VARIANTS), ("tree", TREE_VARIANTS)):
        for name, flags in variants.items():
            start = time.perf_counter()
            per_fold = []
            flags = dict(flags)
            bins = flags.pop("max_bins", 65535)
            for part in parts:
                train, test = dev.take(part.train), dev.take(part.test)
                pre = preprocess.Preprocessor(**flags)
                x_train = pre.fit_transform(train)
                x_test = pre.transform(test)
                if model == "knn":
                    clf = KNNClassifier(knn_best["k"], knn_best["metric"], knn_best["weights"],
                                        n_classes=N_CLASSES).fit(x_train, train.y)
                else:
                    clf = DecisionTreeClassifier(tree_best["criterion"], tree_best["max_depth"],
                                                 tree_best["min_samples_leaf"],
                                                 class_weight=tree_best["class_weight"], max_bins=bins,
                                                 seed=protocol.seed_int("baseline-tree"))
                    clf.fit(x_train, train.y, n_classes=N_CLASSES)
                per_fold.append({**metrics.score_all(test.y, clf.predict(x_test), N_CLASSES),
                                 "n_features": pre.n_features_})
            frame = pd.DataFrame(per_fold)
            row = {"model": model, "variant": name, "n_features": int(frame["n_features"].iloc[0]),
                   "seconds": time.perf_counter() - start}
            for measure in metrics.MEASURES:
                row[f"{measure}_mean"] = float(frame[measure].mean())
                row[f"{measure}_std"] = float(frame[measure].std(ddof=1))
            rows.append(row)
            print(f"{model} / {name}: macro-F1 {row['macro_f1_mean']:.4f} "
                  f"accuracy {row['accuracy_mean']:.4f} ({row['seconds']:.0f}s)", flush=True)
            pd.DataFrame(rows).to_csv(out / "preprocessing_ablation.csv", index=False)


if __name__ == "__main__":
    main()
