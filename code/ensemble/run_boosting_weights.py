"""Where does boosting put its instance weight? Share of the weight on rows of conflicting vectors.

For each tuning fold the tuned boosting ensemble is fitted and the normalised instance weights are
recorded after selected rounds. Training rows fall into three kinds: rows of feature vectors with at least
two labels in the training part (conflicting), rows of repeated vectors with one label (consistent), and rows
of vectors that occur once (unique). The script writes the share of the weight and the share of the rows of
each kind, and the mean weight per row relative to the uniform weight, to results/ensemble/boosting_weights.csv.

Usage: nohup .venv/bin/python code/ensemble/run_boosting_weights.py > logs/boost_weights.log 2>&1 &
"""
from __future__ import annotations

import os

for _name in ("NUMBA_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_name, "10")

import json  # noqa: E402
import sys  # noqa: E402
from concurrent.futures import ThreadPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import grouped, protocol  # noqa: E402
from ensemble import ensemble, harness  # noqa: E402
from ensemble.harness import RESULTS, log  # noqa: E402

ROUNDS = [1, 5, 10, 25, 50, 100]


def row_kinds(groups: np.ndarray, y: np.ndarray) -> np.ndarray:
    """0 unique vector, 1 repeated vector with one label, 2 vector with at least two labels."""
    n_rows = np.bincount(groups, minlength=groups.max() + 1)
    pairs = np.unique(np.stack([groups, y], axis=1), axis=0)
    n_labels = np.bincount(pairs[:, 0], minlength=groups.max() + 1)
    kind = np.zeros(groups.size, dtype=np.int8)
    kind[(n_rows[groups] > 1) & (n_labels[groups] == 1)] = 1
    kind[n_labels[groups] > 1] = 2
    return kind


def main() -> None:
    enc = json.loads((RESULTS / "selected_encoding.json").read_text())["encoding"]
    tuned = json.loads((RESULTS / "tuned_boosting.json").read_text())
    table, dev, _ = harness.load_split()
    groups_dev = grouped.feature_groups(dev)
    parts = protocol.tuning_partitions(dev.y)

    def work(part):
        train, test = dev.take(part.train), dev.take(part.test)
        Xtr, _ = harness.prepare(train, test, enc)
        kind = row_kinds(groups_dev[part.train], train.y)
        ab = ensemble.AdaBoostClassifier(n_estimators=max(ROUNDS), max_bins=4095, purpose=f"opt1/boost-weights/fold{part.fold}",
                                         **tuned["params"])
        ab.fit(Xtr, train.y, weight_checkpoints=ROUNDS)
        rows = []
        for r in ROUNDS:
            w = ab.weights_at_.get(r)
            if w is None:
                continue
            for k, name in enumerate(("unique", "consistent", "conflicting")):
                m = kind == k
                rows.append({"fold": part.fold, "round": r, "kind": name, "row_share": float(m.mean()),
                             "weight_share": float(w[m].sum() / w.sum()),
                             "relative_weight": float(w[m].mean() * w.size)})
        return rows

    with ThreadPoolExecutor(5) as pool:
        rows = [r for rs in pool.map(work, parts) for r in rs]
    pd.DataFrame(rows).to_csv(RESULTS / "boosting_weights.csv", index=False)
    log("finished")


if __name__ == "__main__":
    main()
