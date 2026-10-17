"""Random search over the joint control-parameter space of the random forest.

The one-factor sweeps show the effect of each parameter on its own. This script
searches the joint space with random configurations (Bergstra and Bengio) on the
five tuning folds of D, one forest of 200 trees per configuration and fold, and
selects

  plain      the configuration without any class treatment (class_weight_power 0,
             standard bootstrap) with the highest mean macro-F1
  balanced   the configuration with a class treatment (class weights, balanced
             bootstrap or undersampling) with the highest mean macro-F1
  overall    the better of the two

The base configuration is always evaluated as reference. Results go to
results/ensemble/random_search.csv and the selection to results/ensemble/tuned_forest.json.

Usage: nohup .venv/bin/python code/ensemble/run_forest_search.py > logs/search.log 2>&1 &
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

from common import protocol  # noqa: E402
from ensemble import harness  # noqa: E402
from ensemble.harness import RESULTS, log  # noqa: E402
from ensemble.run_forest_sweeps import N_TREES, encoding_choice, run_unit  # noqa: E402

N_CONFIGS = 60
N_PLAIN_EXTRA = 24
SPACE = {
    "criterion": ("gini", "entropy"),
    "max_features": (2, 4, 8, 12, 16, 24, 32),
    "max_depth": (8, 12, 16, 20, 30, None),
    "min_samples_leaf": (1, 2, 3, 5, 10),
    "class_weight_power": (0.0, 0.25, 0.5, 0.75, 1.0),
    "treatment": (("bootstrap", 0.0, None), ("bootstrap", 0.5, None), ("bootstrap", 1.0, None),
                  ("under", 0.0, 1000)),
}


def sample_configs(n: int) -> list[dict]:
    gen = protocol.rng("opt1/search/configs")
    out = [harness.default_forest_params()]
    seen = {json.dumps(out[0], sort_keys=True, default=str)}
    while len(out) < n:
        p = harness.default_forest_params()
        p["criterion"] = SPACE["criterion"][gen.integers(len(SPACE["criterion"]))]
        p["max_features"] = int(SPACE["max_features"][gen.integers(len(SPACE["max_features"]))])
        d = SPACE["max_depth"][gen.integers(len(SPACE["max_depth"]))]
        p["max_depth"] = None if d is None else int(d)
        p["min_samples_leaf"] = int(SPACE["min_samples_leaf"][gen.integers(len(SPACE["min_samples_leaf"]))])
        p["class_weight_power"] = float(SPACE["class_weight_power"][gen.integers(len(SPACE["class_weight_power"]))])
        s, power, per_class = SPACE["treatment"][gen.integers(len(SPACE["treatment"]))]
        p["sampling"], p["sampling_power"], p["class_sample_size"] = s, power, per_class
        key = json.dumps(p, sort_keys=True, default=str)
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out


def sample_plain_configs(n: int) -> list[dict]:
    """Extra configurations without class treatment (the main search drew only six of them)."""
    gen = protocol.rng("opt1/search-plain/configs")
    out, seen = [], set()
    while len(out) < n:
        p = harness.default_forest_params()
        p["criterion"] = SPACE["criterion"][gen.integers(len(SPACE["criterion"]))]
        p["max_features"] = int(SPACE["max_features"][gen.integers(len(SPACE["max_features"]))])
        d = SPACE["max_depth"][gen.integers(len(SPACE["max_depth"]))]
        p["max_depth"] = None if d is None else int(d)
        p["min_samples_leaf"] = int(SPACE["min_samples_leaf"][gen.integers(len(SPACE["min_samples_leaf"]))])
        key = json.dumps(p, sort_keys=True, default=str)
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out


def has_treatment(p: dict) -> bool:
    return p["class_weight_power"] > 0 or p["sampling_power"] > 0 or p["sampling"] != "bootstrap"


def name_of(i: int) -> str:
    return f"cfg{i:02d}"


def select() -> None:
    df = pd.read_csv(RESULTS / "random_search.csv")
    configs = sample_configs(N_CONFIGS)
    agg = df.groupby("config")[["macro_f1", "accuracy", "balanced_accuracy", "oob_macro_f1"]].agg(["mean", "std"])
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    agg = agg.reset_index()
    plain_extra = sample_plain_configs(N_PLAIN_EXTRA)
    lookup = {name_of(i): c for i, c in enumerate(configs)}
    lookup.update({f"pln{i:02d}": c for i, c in enumerate(plain_extra)})
    path_extra = RESULTS / "random_search_plain.csv"
    if path_extra.exists():
        extra = pd.read_csv(path_extra)
        agg_e = extra.groupby("config")[["macro_f1", "accuracy", "balanced_accuracy", "oob_macro_f1"]].agg(["mean", "std"])
        agg_e.columns = [f"{a}_{b}" for a, b in agg_e.columns]
        agg = pd.concat([agg, agg_e.reset_index()], ignore_index=True)
    agg["has_treatment"] = [has_treatment(lookup[c]) for c in agg["config"]]
    agg.to_csv(RESULTS / "random_search_summary.csv", index=False)
    out = {"n_configs": int(len(agg)), "n_trees": N_TREES, "encoding": encoding_choice()}
    for key, mask in (("plain", ~agg["has_treatment"]), ("balanced", agg["has_treatment"])):
        sub = agg[mask]
        best = sub.loc[sub["macro_f1_mean"].idxmax()]
        cfg = lookup[best["config"]]
        out[key] = {"config": best["config"], "params": cfg,
                    "macro_f1_mean": float(best["macro_f1_mean"]), "macro_f1_std": float(best["macro_f1_std"])}
    top = "plain" if out["plain"]["macro_f1_mean"] >= out["balanced"]["macro_f1_mean"] else "balanced"
    out["overall"] = dict(out[top])
    out["overall"]["which"] = top
    out["params"] = out["overall"]["params"]
    (RESULTS / "tuned_forest.json").write_text(json.dumps(out, indent=2, default=str))
    log(f"selected: plain {out['plain']['config']} {out['plain']['macro_f1_mean']:.4f}, "
        f"balanced {out['balanced']['config']} {out['balanced']['macro_f1_mean']:.4f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-configs", type=int, default=N_CONFIGS)
    ap.add_argument("--select-only", action="store_true")
    ap.add_argument("--plain-extra", action="store_true",
                    help="evaluate the extra configurations without class treatment, then select")
    args = ap.parse_args()
    if args.select_only:
        select()
        return
    enc = encoding_choice()
    table, dev, _ = harness.load_split()
    parts = protocol.tuning_partitions(dev.y)
    if args.plain_extra:
        configs, prefix, out_name = sample_plain_configs(N_PLAIN_EXTRA), "pln", "random_search_plain.csv"
    else:
        configs, prefix, out_name = sample_configs(args.n_configs), "cfg", "random_search.csv"
    writer = harness.ResultWriter(RESULTS / out_name, ["experiment", "config", "fold", "run", "n_estimators"])
    started = time.perf_counter()
    for part in parts:
        train, test = dev.take(part.train), dev.take(part.test)
        Xtr, Xte = harness.prepare(train, test, enc)
        fold_data = (Xtr, train.y, Xte, test.y)
        for i, params in enumerate(configs):
            name = f"{prefix}{i:02d}"
            if writer.finished(experiment="search", config=name, fold=part.fold, run=0, n_estimators=N_TREES):
                continue
            t0 = time.perf_counter()
            rows = run_unit(name, params, N_TREES, [N_TREES], fold_data, "search", part.fold, 0, enc)
            writer.write(rows)
            log(f"search fold {part.fold} {name}: macro-F1 {rows[-1]['macro_f1']:.4f} ({time.perf_counter() - t0:.0f}s)")
    log(f"search finished in {(time.perf_counter() - started) / 60:.1f} minutes")
    select()


if __name__ == "__main__":
    main()
