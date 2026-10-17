"""Helpers of the ensemble experiment scripts.

* load_split(): development set D and generalisation set G of the shared protocol
* prepare(): feature matrices of one partition, fitted on the training rows only
* score_row(): accuracy, macro-F1, weighted F1, balanced accuracy and per-class recall and F1
* ResultWriter: append rows to a CSV and skip work that a previous run already finished
* purpose(): name of the random stream of one (experiment, configuration, fold, run)

Random streams. A forest or boosting model receives the purpose
"opt1/<experiment>/<fold or partition>/run<r>". Configurations of one experiment
share the purpose of a fold and a run, so the compared configurations use the same
random numbers (common random numbers) and differences are not seed noise.
"""
from __future__ import annotations

import csv
import os
import time
from pathlib import Path

import numpy as np

from common import data, metrics, protocol
from . import diversity, features

N_CLASSES = data.N_CLASSES
REPO_ROOT = data.REPO_ROOT
# ENSEMBLE_SMOKE=1 shrinks every script to a few seconds and ENSEMBLE_RESULTS_DIR redirects the output;
# both exist only for the smoke test of the scripts and are unset in the real runs.
SMOKE = os.environ.get("ENSEMBLE_SMOKE") == "1"
N_JOBS = int(os.environ.get("ENSEMBLE_NJOBS", "10"))     # threads for fitting the trees of one forest
RESULTS = Path(os.environ["ENSEMBLE_RESULTS_DIR"]) if os.environ.get("ENSEMBLE_RESULTS_DIR") \
    else REPO_ROOT / "results" / "ensemble"
FIGURES = REPO_ROOT / "figures" / "ensemble"
CLASS_COLUMNS = [f"recall_{j}" for j in range(N_CLASSES)] + [f"f1_{j}" for j in range(N_CLASSES)]


def log(message: str) -> None:
    print(time.strftime("%H:%M:%S"), message, flush=True)


def load_split():
    """Table, development set D, generalisation set G of the shared protocol."""
    table = data.load_table()
    dev_idx, gen_idx = protocol.stratified_holdout(table.y)
    return table, table.take(dev_idx), table.take(gen_idx)


def prepare(train: data.Table, test: data.Table, encoding: str):
    """Fit the encoding on the training rows, return (X_train, X_test)."""
    ff = features.ForestFeatures(encoding).fit(train)
    return ff.transform(train), ff.transform(test)


def score_row(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Overall scores plus per-class recall and F1 for one set of predictions."""
    cm = metrics.confusion_matrix(y_true, y_pred, N_CLASSES)
    row = dict(metrics.scores_from_confusion(cm))
    pc = metrics.per_class(cm)
    for j in range(N_CLASSES):
        row[f"recall_{j}"] = float(pc["recall"][j])
        row[f"f1_{j}"] = float(pc["f1"][j])
    return row


def oob_row(y_train: np.ndarray, oob_pred: np.ndarray, prefix: str = "oob_") -> dict:
    """Scores of the out-of-bag predictions on the rows that have at least one out-of-bag tree."""
    covered = oob_pred != 255
    s = metrics.score_all(y_train[covered], oob_pred[covered].astype(np.int64), N_CLASSES)
    out = {prefix + k: v for k, v in s.items() if k in ("accuracy", "macro_f1", "balanced_accuracy")}
    out[prefix + "coverage"] = float(covered.mean())
    return out


def diversity_row(members: np.ndarray, y: np.ndarray) -> dict:
    return diversity.summary(members, y, N_CLASSES)


class ResultWriter:
    """CSV writer that appends rows and remembers which keys are finished.

    key_columns identify one unit of work. A unit is skipped when a row with the
    same key values already exists, so an interrupted script resumes where it stopped.
    """

    def __init__(self, path: Path, key_columns: list[str]):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.key_columns = key_columns
        self.fields: list[str] | None = None
        self.done: set[tuple] = set()
        if self.path.exists() and self.path.stat().st_size > 0:
            with open(self.path, newline="") as fh:
                reader = csv.DictReader(fh)
                self.fields = list(reader.fieldnames or [])
                for row in reader:
                    self.done.add(tuple(row[k] for k in key_columns))

    @staticmethod
    def _fmt(v):
        if v is None:
            return "None"
        return repr(v) if isinstance(v, float) else str(v)

    def finished(self, **key) -> bool:
        return tuple(self._fmt(key[k]) for k in self.key_columns) in self.done

    def write(self, rows: list[dict]) -> None:
        if not rows:
            return
        if self.fields is None:
            self.fields = list(rows[0].keys())
            with open(self.path, "w", newline="") as fh:
                csv.writer(fh).writerow(self.fields)
        with open(self.path, "a", newline="") as fh:
            w = csv.writer(fh)
            for r in rows:
                w.writerow([self._fmt(r.get(f)) for f in self.fields])
                self.done.add(tuple(self._fmt(r[k]) for k in self.key_columns))


def default_forest_params() -> dict:
    """Base configuration of the sweeps: entropy criterion as in the tuned baseline tree."""
    return dict(criterion="entropy", max_features="sqrt", max_depth=None, min_samples_leaf=1,
                sampling="bootstrap", sampling_power=0.0, class_sample_size=None,
                class_weight_power=0.0)


def param_columns(p: dict) -> dict:
    """Flat, CSV-friendly view of forest parameters."""
    return {"criterion": p["criterion"], "max_features": p["max_features"],
            "max_depth": p["max_depth"], "min_samples_leaf": p["min_samples_leaf"],
            "sampling": p["sampling"], "sampling_power": p["sampling_power"],
            "class_sample_size": p["class_sample_size"],
            "class_weight_power": p["class_weight_power"]}
