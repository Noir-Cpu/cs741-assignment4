"""Joint search of the random forest over all interacting control parameters (successive halving).

The staged design of round one (one-factor sweeps, then a random search that drew only a few
configurations of each kind) is replaced by one joint design.

Space: criterion, number of features per split, depth limit, minimum leaf size, class-weight
power alpha, and the sampler (standard or class-balanced bootstrap with exponent omega, or
undersampling to a fixed number of instances per class). The space splits into four families
by class treatment: plain (alpha = 0, standard bootstrap), weighted (alpha > 0, standard
bootstrap), sampled (alpha = 0, balanced bootstrap or undersampling) and both.

Stage 1  FIRST_STAGE random configurations per family, 100 trees, tuning folds 0 and 1, one run.
Stage 2  the best KEEP_PER_FAMILY of each family plus the forced configurations (the optima of
         the one-factor sweeps and of round one), 200 trees, five tuning folds, three runs.
Selection: the configuration with the highest mean macro-F1 of stage 2 within each family.

Results: results/ensemble/joint_stage1.csv, joint_stage2.csv, tuned_forests.json.

Usage: nohup .venv/bin/python code/ensemble/run_joint_search.py > logs/joint.log 2>&1 &
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
from ensemble.run_forest_sweeps import run_unit  # noqa: E402

FIRST_STAGE = 40
KEEP_PER_FAMILY = 4
STAGE1_TREES, STAGE2_TREES = 100, 200
STAGE1_FOLDS, STAGE2_RUNS = 2, 3
if harness.SMOKE:
    FIRST_STAGE, KEEP_PER_FAMILY, STAGE1_TREES, STAGE2_TREES, STAGE1_FOLDS, STAGE2_RUNS = 3, 1, 6, 8, 1, 1

FEATURES = (4, 8, 12, 16, 24, 32, 48)
DEPTHS = (16, 20, 24, 32, None)
LEAVES = (1, 2, 3, 5)
CRITERIA = ("gini", "entropy")
ALPHAS = (0.25, 0.5, 0.75, 1.0)
SAMPLERS = (("bootstrap", 0.25, None), ("bootstrap", 0.5, None), ("bootstrap", 0.75, None),
            ("bootstrap", 1.0, None), ("under", 0.0, 1000), ("under", 0.0, 3000))
FAMILIES = ("plain", "weighted", "sampled", "both")


def family_of(p: dict) -> str:
    weighted = p["class_weight_power"] > 0
    sampled = p["sampling"] != "bootstrap" or p["sampling_power"] > 0
    return {(False, False): "plain", (True, False): "weighted", (False, True): "sampled", (True, True): "both"}[
        (weighted, sampled)]


def draw(gen, family: str) -> dict:
    p = harness.default_forest_params()
    p["criterion"] = CRITERIA[gen.integers(len(CRITERIA))]
    p["max_features"] = int(FEATURES[gen.integers(len(FEATURES))])
    d = DEPTHS[gen.integers(len(DEPTHS))]
    p["max_depth"] = None if d is None else int(d)
    p["min_samples_leaf"] = int(LEAVES[gen.integers(len(LEAVES))])
    if family in ("weighted", "both"):
        p["class_weight_power"] = float(ALPHAS[gen.integers(len(ALPHAS))])
    if family in ("sampled", "both"):
        s, power, size = SAMPLERS[gen.integers(len(SAMPLERS))]
        p["sampling"], p["sampling_power"], p["class_sample_size"] = s, float(power), size
    return p


def stage1_configs() -> list[dict]:
    gen = protocol.rng("opt1/joint/configs")
    out, seen = [], set()
    for fam in FAMILIES:
        n = 0
        while n < FIRST_STAGE:
            p = draw(gen, fam)
            key = json.dumps(p, sort_keys=True, default=str)
            if key not in seen:
                seen.add(key)
                out.append(p)
                n += 1
    return out


def forced_configs() -> list[dict]:
    """Optima of the one-factor sweeps and of round one, evaluated in stage 2 next to the survivors."""
    base = harness.default_forest_params()
    out = []
    for m, leaf, depth in ((16, 1, None), (32, 1, None), (16, 2, 20), (32, 2, 20)):
        p = dict(base)
        p.update(max_features=m, min_samples_leaf=leaf, max_depth=depth)
        out.append(p)
    p = dict(base)
    p.update(max_features=16, sampling_power=0.5)                       # balanced bootstrap of round one
    out.append(p)
    p = dict(base)
    p.update(max_features=24, max_depth=30, min_samples_leaf=2, class_weight_power=0.5)   # round-one winner
    out.append(p)
    p = dict(base)
    p.update(max_features=16, class_weight_power=0.5)
    out.append(p)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--select-only", action="store_true")
    args = ap.parse_args()
    enc = json.loads((RESULTS / "selected_encoding.json").read_text())["encoding"]
    table, dev, _ = harness.load_split()
    parts = protocol.tuning_partitions(dev.y)
    s1 = stage1_configs()
    s1_names = [f"j{i:03d}" for i in range(len(s1))]
    started = time.perf_counter()

    def fold_data(part):
        train, test = dev.take(part.train), dev.take(part.test)
        Xtr, Xte = harness.prepare(train, test, enc)
        return (Xtr, train.y, Xte, test.y)

    if not args.select_only:
        w1 = harness.ResultWriter(RESULTS / "joint_stage1.csv", ["experiment", "config", "fold", "run", "n_estimators"])
        for part in parts[:STAGE1_FOLDS]:
            fd = fold_data(part)
            for name, p in zip(s1_names, s1):
                if w1.finished(experiment="joint1", config=name, fold=part.fold, run=0, n_estimators=STAGE1_TREES):
                    continue
                rows = run_unit(name, p, STAGE1_TREES, [STAGE1_TREES], fd, "joint1", part.fold, 0, enc)
                for r in rows:
                    r["family"] = family_of(p)
                w1.write(rows)
                log(f"stage 1 fold {part.fold} {name} {family_of(p)}: {rows[-1]['macro_f1']:.4f}")
        d1 = pd.read_csv(RESULTS / "joint_stage1.csv")
        agg = d1.groupby("config")["macro_f1"].mean()
        fam = dict(zip(s1_names, [family_of(p) for p in s1]))
        survivors = []
        for f in FAMILIES:
            names = [n for n in agg.index if fam[n] == f]
            survivors += list(agg[names].sort_values(ascending=False).index[:KEEP_PER_FAMILY])
        cands = [(n, s1[s1_names.index(n)]) for n in survivors]
        cands += [(f"f{i}", p) for i, p in enumerate(forced_configs())]
        json.dump({n: p for n, p in cands}, open(RESULTS / "joint_stage2_configs.json", "w"), indent=1, default=str)
        w2 = harness.ResultWriter(RESULTS / "joint_stage2.csv", ["experiment", "config", "fold", "run", "n_estimators"])
        for part in parts:
            fd = fold_data(part)
            for name, p in cands:
                for run in range(STAGE2_RUNS):
                    if w2.finished(experiment="joint2", config=name, fold=part.fold, run=run, n_estimators=STAGE2_TREES):
                        continue
                    t0 = time.perf_counter()
                    rows = run_unit(name, p, STAGE2_TREES, [STAGE2_TREES], fd, "joint2", part.fold, run, enc)
                    for r in rows:
                        r["family"] = family_of(p)
                    w2.write(rows)
                    log(f"stage 2 fold {part.fold} run {run} {name} {family_of(p)}: {rows[-1]['macro_f1']:.4f} "
                        f"({time.perf_counter() - t0:.0f}s)")
    d2 = pd.read_csv(RESULTS / "joint_stage2.csv", keep_default_na=False, na_values=["", "nan"])
    cfgs = json.loads((RESULTS / "joint_stage2_configs.json").read_text())
    g = d2.groupby("config")[["macro_f1", "accuracy", "balanced_accuracy", "oob_macro_f1"]].agg(["mean", "std"])
    g.columns = [f"{a}_{b}" for a, b in g.columns]
    g = g.reset_index()
    g["family"] = [family_of(cfgs[c]) for c in g["config"]]
    g.to_csv(RESULTS / "joint_stage2_summary.csv", index=False)
    out = {"stage1_configs": len(s1), "stage2_configs": len(cfgs), "stage2_runs": STAGE2_RUNS,
           "stage1_trees": STAGE1_TREES, "stage2_trees": STAGE2_TREES, "encoding": enc}
    for f in FAMILIES:
        sub = g[g["family"] == f]
        best = sub.loc[sub["macro_f1_mean"].idxmax()]
        out[f] = {"config": best["config"], "params": cfgs[best["config"]], "macro_f1_mean": float(best["macro_f1_mean"]),
                  "macro_f1_std": float(best["macro_f1_std"])}
    (RESULTS / "tuned_forests.json").write_text(json.dumps(out, indent=2, default=str))
    log(f"selected {[(f, out[f]['config'], round(out[f]['macro_f1_mean'], 4)) for f in FAMILIES]} "
        f"after {(time.perf_counter() - started) / 60:.0f} minutes")


if __name__ == "__main__":
    main()
