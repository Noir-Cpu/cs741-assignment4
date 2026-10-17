"""Tune and evaluate the baseline models (k-nearest neighbours, classification tree).

Everything uses the shared protocol of common.protocol:

1. hold-out split: G (20 percent, never used for tuning) and D (80 percent)
2. tuning: stratified 5-fold cross-validation inside D, grids below, selection
   by mean macro-averaged F1
3. comparison: stratified 5-fold cross-validation repeated 3 times inside D
   (15 paired scores per model), then a final fit on all of D and one
   evaluation on G
4. statistics on the 15 paired scores

Models
------
knn      grid of Assignment 2 (k, vote weighting, metric), no class weights
knn_cw   the same grid extended with a class-weight power for the votes
tree     criterion, depth, minimum leaf size, class weights

Stages can be run separately (--stages) and each stage skips its work when its
output exists, unless --force is given.

Usage (from the repository root):
    nohup .venv/bin/python code/common/run_baselines.py > logs/baselines.log 2>&1 &
"""
from __future__ import annotations

import argparse
import os

# thread counts must be set before numpy and numba are imported
_THREADS = "20"
for _name in ("NUMBA_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_name, _THREADS)

import itertools  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import resource  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ThreadPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import data, metrics, preprocess, protocol, stats  # noqa: E402
from common.knn import KNNClassifier, class_weight_vector  # noqa: E402
from common.tree import BinMapper, DecisionTreeClassifier  # noqa: E402

N_CLASSES = data.N_CLASSES
MEASURES = metrics.MEASURES

KNN_K = (1, 3, 5, 9, 15, 25, 35, 51)
KNN_WEIGHTS = ("uniform", "distance")
KNN_METRICS = ("euclidean", "manhattan")
KNN_CLASS_POWER = (0.0, 0.25, 0.5, 0.75, 1.0)

TREE_CRITERIA = ("gini", "entropy")
TREE_DEPTHS = (10, 15, 20, 25, 30, 40, None)
TREE_LEAVES = (1, 2, 5, 10, 20, 50)
TREE_CLASS_WEIGHTS = (None, "balanced")

# Assignment 2 reference (results/model_comparison.csv of Assignment 2, 15 paired scores)
ASSIGNMENT2 = {
    "knn": {"accuracy": 0.8019, "macro_f1": 0.5250, "weighted_f1": 0.7995, "balanced_accuracy": 0.5072},
    "tree": {"accuracy": 0.7592, "macro_f1": 0.5757, "weighted_f1": 0.7903, "balanced_accuracy": 0.6086},
}


def log(message: str) -> None:
    print(time.strftime("%H:%M:%S"), message, flush=True)


def write_json(path: Path, obj) -> None:
    def default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(type(o))
    path.write_text(json.dumps(obj, indent=2, default=default))


def mean_std(values) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    return float(values.mean()), float(values.std(ddof=1)) if values.size > 1 else 0.0


# ---------------------------------------------------------------------------
# Folds
# ---------------------------------------------------------------------------

def knn_arrays(dev: data.Table, part: protocol.Partition):
    start = time.perf_counter()
    train, test = dev.take(part.train), dev.take(part.test)
    pre = preprocess.Preprocessor.for_knn()
    x_train = pre.fit_transform(train)
    x_test = pre.transform(test)
    return x_train, train.y, x_test, test.y, time.perf_counter() - start


def tree_arrays(dev: data.Table, part: protocol.Partition):
    start = time.perf_counter()
    train, test = dev.take(part.train), dev.take(part.test)
    pre = preprocess.Preprocessor.for_tree()
    x_train = pre.fit_transform(train)
    x_test = pre.transform(test)
    return x_train, train.y, x_test, test.y, time.perf_counter() - start


# ---------------------------------------------------------------------------
# Tuning
# ---------------------------------------------------------------------------

def tune_knn(dev: data.Table, out: Path) -> dict:
    parts = protocol.tuning_partitions(dev.y)
    kmax = max(KNN_K)
    records = []
    started = time.perf_counter()
    for part in parts:
        x_train, y_train, x_test, y_test, prep_time = knn_arrays(dev, part)
        for metric in KNN_METRICS:
            model = KNNClassifier(kmax, metric, n_classes=N_CLASSES).fit(x_train, y_train)
            t0 = time.perf_counter()
            dist, idx = model.kneighbors(x_test, kmax)
            search_time = time.perf_counter() - t0
            log(f"knn tuning fold {part.fold} {metric}: search {search_time:.1f}s")
            for power in KNN_CLASS_POWER:
                cw = class_weight_vector(y_train, N_CLASSES, None if power == 0 else power)
                for weights, k in itertools.product(KNN_WEIGHTS, KNN_K):
                    pred = KNNClassifier.predict_from_neighbours(idx, dist, y_train, k, weights,
                                                                 cw, N_CLASSES)
                    scores = metrics.score_all(y_test, pred, N_CLASSES)
                    records.append({"fold": part.fold, "metric": metric, "k": k, "weights": weights,
                                    "class_weight_power": power, "search_seconds": search_time,
                                    **scores})
    frame = pd.DataFrame(records)
    frame.to_csv(out / "knn_tuning_folds.csv", index=False)
    keys = ["metric", "k", "weights", "class_weight_power"]
    grid = frame.groupby(keys, sort=False)[list(MEASURES)].agg(["mean", "std"])
    grid.columns = [f"{a}_{b}" for a, b in grid.columns]
    grid = grid.reset_index()
    grid.to_csv(out / "knn_tuning_grid.csv", index=False)
    plain = grid[grid["class_weight_power"] == 0.0]
    best_plain = plain.loc[plain["macro_f1_mean"].idxmax()]
    best_any = grid.loc[grid["macro_f1_mean"].idxmax()]

    def params(row):
        return {"metric": row["metric"], "k": int(row["k"]), "weights": row["weights"],
                "class_weight_power": float(row["class_weight_power"]),
                "cv_macro_f1_mean": float(row["macro_f1_mean"]),
                "cv_macro_f1_std": float(row["macro_f1_std"])}
    best = {"knn": params(best_plain), "knn_cw": params(best_any),
            "n_configurations": int(len(grid)), "n_folds": len(parts),
            "wall_seconds": time.perf_counter() - started}
    write_json(out / "knn_best_params.json", best)
    log(f"knn best: {best['knn']}  with class weights: {best['knn_cw']}")
    return best


def tune_tree(dev: data.Table, out: Path, workers: int) -> dict:
    parts = protocol.tuning_partitions(dev.y)
    started = time.perf_counter()
    folds = []
    for part in parts:
        x_train, y_train, x_test, y_test, _ = tree_arrays(dev, part)
        binner = BinMapper().fit(x_train)
        folds.append((x_train, y_train, binner, binner.transform(x_train), x_test, y_test))
    configs = list(itertools.product(TREE_CRITERIA, TREE_DEPTHS, TREE_LEAVES, TREE_CLASS_WEIGHTS))
    seed = protocol.seed_int("baseline-tree")
    log(f"tree tuning: {len(configs)} configurations x {len(folds)} folds on {workers} threads")

    def job(args):
        (criterion, depth, leaf, cw), fold = args
        x_train, y_train, binner, xb, x_test, y_test = folds[fold]
        t0 = time.perf_counter()
        model = DecisionTreeClassifier(criterion, depth, leaf, class_weight=cw, seed=seed)
        model.fit(x_train, y_train, n_classes=N_CLASSES, binner=binner, Xb_t=xb)
        fit_time = time.perf_counter() - t0
        pred = model.predict(x_test)
        scores = metrics.score_all(y_test, pred, N_CLASSES)
        return {"fold": fold, "criterion": criterion,
                "max_depth": -1 if depth is None else depth, "min_samples_leaf": leaf,
                "class_weight": "none" if cw is None else cw, "n_nodes": model.n_nodes_,
                "fit_seconds": fit_time, **scores}

    jobs = [(c, f) for c in configs for f in range(len(folds))]
    with ThreadPoolExecutor(workers) as pool:
        records = list(pool.map(job, jobs))
    frame = pd.DataFrame(records)
    frame.to_csv(out / "tree_tuning_folds.csv", index=False)
    keys = ["criterion", "max_depth", "min_samples_leaf", "class_weight"]
    grid = frame.groupby(keys, sort=False)[list(MEASURES) + ["n_nodes", "fit_seconds"]].agg(["mean", "std"])
    grid.columns = [f"{a}_{b}" for a, b in grid.columns]
    grid = grid.reset_index()
    grid.to_csv(out / "tree_tuning_grid.csv", index=False)
    best_row = grid.loc[grid["macro_f1_mean"].idxmax()]
    best = {"criterion": best_row["criterion"],
            "max_depth": None if int(best_row["max_depth"]) < 0 else int(best_row["max_depth"]),
            "min_samples_leaf": int(best_row["min_samples_leaf"]),
            "class_weight": None if best_row["class_weight"] == "none" else best_row["class_weight"],
            "cv_macro_f1_mean": float(best_row["macro_f1_mean"]),
            "cv_macro_f1_std": float(best_row["macro_f1_std"]),
            "n_configurations": int(len(grid)), "n_folds": len(folds),
            "wall_seconds": time.perf_counter() - started}
    balanced = grid[grid["class_weight"] == "balanced"]
    unweighted = grid[grid["class_weight"] == "none"]
    row_b = balanced.loc[balanced["macro_f1_mean"].idxmax()]
    best["balanced_variant"] = {
        "criterion": row_b["criterion"],
        "max_depth": None if int(row_b["max_depth"]) < 0 else int(row_b["max_depth"]),
        "min_samples_leaf": int(row_b["min_samples_leaf"]), "class_weight": "balanced",
        "cv_macro_f1_mean": float(row_b["macro_f1_mean"]), "cv_macro_f1_std": float(row_b["macro_f1_std"])}
    best["best_macro_f1_balanced"] = float(balanced["macro_f1_mean"].max())
    best["best_macro_f1_unweighted"] = float(unweighted["macro_f1_mean"].max())
    write_json(out / "tree_best_params.json", best)
    log(f"tree best: {best}")
    return best


# ---------------------------------------------------------------------------
# Comparison and final evaluation
# ---------------------------------------------------------------------------

def make_model(name: str, best: dict):
    if name in ("tree", "tree_balanced"):
        tb = best[name]
        return DecisionTreeClassifier(tb["criterion"], tb["max_depth"], tb["min_samples_leaf"],
                                      class_weight=tb["class_weight"],
                                      seed=protocol.seed_int("baseline-tree"))
    kb = best[name]
    power = kb["class_weight_power"]
    return KNNClassifier(kb["k"], kb["metric"], kb["weights"],
                         class_weight=None if power == 0 else power, n_classes=N_CLASSES)


def fit_predict(name: str, best: dict, x_train, y_train, x_test):
    model = make_model(name, best)
    t0 = time.perf_counter()
    if name.startswith("tree"):
        model.fit(x_train, y_train, n_classes=N_CLASSES)
    else:
        model.fit(x_train, y_train)
    fit_time = time.perf_counter() - t0
    t0 = time.perf_counter()
    pred = model.predict(x_test)
    return model, pred, fit_time, time.perf_counter() - t0


def compare(dev: data.Table, best: dict, out: Path, names: list[str]) -> None:
    parts = protocol.comparison_partitions(dev.y)
    score_rows, class_rows = [], []
    confusion = {n: np.zeros((N_CLASSES, N_CLASSES), dtype=np.int64) for n in names}
    for part in parts:
        arrays = {"knn": None, "tree": None}
        for name in names:
            kind = "tree" if name.startswith("tree") else "knn"
            if arrays[kind] is None:
                arrays[kind] = (knn_arrays if kind == "knn" else tree_arrays)(dev, part)
            x_train, y_train, x_test, y_test, prep_time = arrays[kind]
            _, pred, fit_time, predict_time = fit_predict(name, best, x_train, y_train, x_test)
            cm = metrics.confusion_matrix(y_test, pred, N_CLASSES)
            confusion[name] += cm
            scores = metrics.scores_from_confusion(cm)
            score_rows.append({"model": name, "repeat": part.repeat, "fold": part.fold,
                               **scores, "prep_seconds": prep_time, "fit_seconds": fit_time,
                               "predict_seconds": predict_time})
            for row in metrics.per_class_table(cm):
                class_rows.append({"model": name, "repeat": part.repeat, "fold": part.fold, **row})
            log(f"compare {name} repeat {part.repeat} fold {part.fold}: "
                f"macro-F1 {scores['macro_f1']:.4f} fit {fit_time:.1f}s predict {predict_time:.1f}s")
        pd.DataFrame(score_rows).to_csv(out / "comparison_scores.csv", index=False)   # checkpoint
    pd.DataFrame(class_rows).to_csv(out / "comparison_per_class.csv", index=False)
    for name in names:
        np.savetxt(out / f"confusion_cv_{name}.csv", confusion[name], fmt="%d", delimiter=",")


def final_fit(dev: data.Table, gen: data.Table, best: dict, out: Path, names: list[str]) -> None:
    score_rows, class_rows = [], []
    arrays = {}
    for kind, builder in (("knn", preprocess.Preprocessor.for_knn), ("tree", preprocess.Preprocessor.for_tree)):
        t0 = time.perf_counter()
        pre = builder()
        x_dev = pre.fit_transform(dev)
        x_gen = pre.transform(gen)
        arrays[kind] = (x_dev, x_gen, time.perf_counter() - t0, pre)
    for name in names:
        kind = "tree" if name.startswith("tree") else "knn"
        x_dev, x_gen, prep_time, pre = arrays[kind]
        model, pred, fit_time, predict_time = fit_predict(name, best, x_dev, dev.y, x_gen)
        cm = metrics.confusion_matrix(gen.y, pred, N_CLASSES)
        scores = metrics.scores_from_confusion(cm)
        score_rows.append({"model": name, **scores, "prep_seconds": prep_time,
                           "fit_seconds": fit_time, "predict_seconds": predict_time,
                           "n_train": dev.n, "n_test": gen.n})
        for row in metrics.per_class_table(cm):
            class_rows.append({"model": name, **row})
        np.savetxt(out / f"confusion_G_{name}.csv", cm, fmt="%d", delimiter=",")
        log(f"G {name}: {scores} fit {fit_time:.1f}s predict {predict_time:.1f}s")
        if name.startswith("tree"):
            importance = pd.DataFrame({"feature": pre.feature_names_,
                                       "importance": model.feature_importances_})
            importance.sort_values("importance", ascending=False).to_csv(
                out / f"{name}_feature_importances.csv", index=False)
            write_json(out / f"{name}_structure.json", {"n_nodes": model.n_nodes_, "n_leaves": model.n_leaves_,
                                                     "depth": model.depth_})
    pd.DataFrame(score_rows).to_csv(out / "generalisation_scores.csv", index=False)
    pd.DataFrame(class_rows).to_csv(out / "generalisation_per_class.csv", index=False)


# ---------------------------------------------------------------------------
# Statistics and summary
# ---------------------------------------------------------------------------

def analyse(out: Path, names: list[str]) -> None:
    scores = pd.read_csv(out / "comparison_scores.csv")
    gen = pd.read_csv(out / "generalisation_scores.csv").set_index("model")
    summary = []
    for name in names:
        sub = scores[scores["model"] == name].sort_values(["repeat", "fold"])
        row = {"model": name, "n_scores": len(sub)}
        for measure in MEASURES:
            m, s = mean_std(sub[measure])
            row[f"{measure}_mean"], row[f"{measure}_std"] = m, s
            row[f"{measure}_G"] = float(gen.loc[name, measure])
        for column in ("prep_seconds", "fit_seconds", "predict_seconds"):
            row[f"{column}_mean"] = float(sub[column].mean())
            row[f"{column}_G"] = float(gen.loc[name, column])
        summary.append(row)
    pd.DataFrame(summary).to_csv(out / "comparison_summary.csv", index=False)

    pivot = {m: {n: scores[scores["model"] == n].sort_values(["repeat", "fold"])[m].to_numpy()
                 for n in names} for m in MEASURES}
    pair_rows = []
    pairs_to_test = [("tree", "knn"), ("tree", "knn_cw"), ("knn_cw", "knn"),
                     ("tree_balanced", "knn"), ("tree_balanced", "knn_cw"), ("tree_balanced", "tree")]
    for a, b in [pr for pr in pairs_to_test if pr[0] in names and pr[1] in names]:
        for measure in MEASURES:
            wil = stats.paired_wilcoxon(pivot[measure][a], pivot[measure][b])
            cor = stats.corrected_resampled_ttest(pivot[measure][a], pivot[measure][b])
            pair_rows.append({"model_a": a, "model_b": b, "measure": measure,
                              "mean_a": float(pivot[measure][a].mean()),
                              "mean_b": float(pivot[measure][b].mean()),
                              "mean_diff": wil["mean_diff"], "median_diff": wil["median_diff"],
                              "wilcoxon_statistic": wil["statistic"], "wilcoxon_p": wil["p_value"],
                              "cohens_dz": wil["cohens_dz"], "rank_biserial": wil["rank_biserial"],
                              "n_a_higher": wil["n_a_higher"], "n_b_higher": wil["n_b_higher"],
                              "corrected_t": cor["t"], "corrected_t_p": cor["p_value"]})
    pairs = pd.DataFrame(pair_rows)
    pairs["wilcoxon_p_holm"] = np.nan
    for measure in MEASURES:
        sel = pairs["measure"] == measure
        pairs.loc[sel, "wilcoxon_p_holm"] = stats.holm_adjust(pairs.loc[sel, "wilcoxon_p"])
    pairs.to_csv(out / "comparison_stats.csv", index=False)

    if len(names) >= 3:
        rows = []
        for measure in MEASURES:
            matrix = np.column_stack([pivot[measure][n] for n in names])
            fr = stats.friedman_test(matrix)
            nem = stats.nemenyi_posthoc(matrix)
            rows.append({"measure": measure, "friedman_statistic": fr["statistic"],
                         "friedman_p": fr["p_value"], "iman_davenport_f": fr["iman_davenport_f"],
                         "iman_davenport_p": fr["iman_davenport_p"], "nemenyi_critical_difference": nem["critical_difference"],
                         **{f"rank_{n}": float(r) for n, r in zip(names, fr["average_ranks"])}})
        pd.DataFrame(rows).to_csv(out / "comparison_friedman.csv", index=False)

    # does the Assignment 2 conclusion reproduce? (tree higher macro-F1 and balanced accuracy,
    # k-nearest neighbours marginally higher accuracy)
    s = {r["model"]: r for r in summary}
    repro = {"assignment2_reference": ASSIGNMENT2, "comparisons": {}}
    for tree_name in [n for n in ("tree", "tree_balanced") if n in s]:
        for knn_name in [n for n in ("knn", "knn_cw") if n in s]:
            sel = pairs[(pairs["model_a"] == tree_name) & (pairs["model_b"] == knn_name)].set_index("measure")
            entry = {
                "tree_macro_f1_higher": bool(s[tree_name]["macro_f1_mean"] > s[knn_name]["macro_f1_mean"]),
                "tree_balanced_accuracy_higher": bool(
                    s[tree_name]["balanced_accuracy_mean"] > s[knn_name]["balanced_accuracy_mean"]),
                "knn_accuracy_higher": bool(s[knn_name]["accuracy_mean"] > s[tree_name]["accuracy_mean"]),
                "wilcoxon_p": {m: float(sel.loc[m, "wilcoxon_p"]) for m in MEASURES},
                "mean_difference_tree_minus_knn": {m: float(sel.loc[m, "mean_diff"]) for m in MEASURES},
                "on_G_tree_macro_f1_higher": bool(s[tree_name]["macro_f1_G"] > s[knn_name]["macro_f1_G"]),
                "on_G_tree_balanced_accuracy_higher": bool(
                    s[tree_name]["balanced_accuracy_G"] > s[knn_name]["balanced_accuracy_G"]),
                "on_G_knn_accuracy_higher": bool(s[knn_name]["accuracy_G"] > s[tree_name]["accuracy_G"])}
            entry["full_conclusion_reproduces"] = bool(
                entry["tree_macro_f1_higher"] and entry["tree_balanced_accuracy_higher"]
                and entry["knn_accuracy_higher"])
            repro["comparisons"][f"{tree_name}_vs_{knn_name}"] = entry
    write_json(out / "assignment2_reproduction.json", repro)
    for key, entry in repro["comparisons"].items():
        log(f"Assignment 2 conclusion {key}: macro-F1 {entry['tree_macro_f1_higher']}, balanced accuracy "
            f"{entry['tree_balanced_accuracy_higher']}, kNN accuracy higher {entry['knn_accuracy_higher']}")


# ---------------------------------------------------------------------------

def hardware_info() -> dict:
    import numba
    import scipy
    cpu = ""
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    return {"cpu_model": cpu, "logical_cores": os.cpu_count(),
            "numba_threads": numba.config.NUMBA_NUM_THREADS, "blas_threads_env": os.environ.get("OPENBLAS_NUM_THREADS"),
            "gpu": "none", "python": platform.python_version(), "numpy": np.__version__,
            "numba": numba.__version__, "scipy": scipy.__version__, "pandas": pd.__version__}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=str(data.RESULTS_COMMON))
    parser.add_argument("--stages", default="tune_knn,tune_tree,compare,final,analyse")
    parser.add_argument("--workers", type=int, default=20, help="threads for the tree tuning grid")
    parser.add_argument("--force", action="store_true", help="recompute stages whose output exists")
    parser.add_argument("--quick", type=int, default=0,
                        help="smoke test: use a stratified subsample of D with this many rows")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stages = args.stages.split(",")
    runtimes_path = out / "runtimes.json"
    runtimes = json.loads(runtimes_path.read_text()) if runtimes_path.exists() else {}

    table = data.load_table()
    dev_index, gen_index = protocol.stratified_holdout(table.y)
    dev, gen = table.take(dev_index), table.take(gen_index)
    if args.quick:
        keep = protocol.stratified_subsample(dev.y, args.quick, min_per_class=60)
        dev = dev.take(keep)
    write_json(out / "split.json", {
        "master_seed": protocol.MASTER_SEED, "holdout_fraction": protocol.HOLDOUT_FRACTION,
        "n_total": table.n, "n_development": dev_index.size, "n_generalisation": gen_index.size,
        "generalisation_index_digest": protocol.index_digest(gen_index),
        "development_index_digest": protocol.index_digest(dev_index),
        "class_counts_development": data.class_counts(table.y[dev_index]),
        "class_counts_generalisation": data.class_counts(gen.y),
        "tuning_purpose": protocol.PURPOSE_TUNING, "comparison_purpose": protocol.PURPOSE_COMPARISON,
        "n_folds": protocol.N_FOLDS, "n_repeats": protocol.N_REPEATS, "quick_subsample": args.quick})
    write_json(out / "hardware.json", hardware_info())
    log(f"D {dev.n} rows, G {gen.n} rows, threads {os.environ['NUMBA_NUM_THREADS']}")

    def run(stage, fn):
        if stage not in stages:
            return
        t0 = time.perf_counter()
        fn()
        runtimes[stage] = {"wall_seconds": time.perf_counter() - t0}
        runtimes["peak_rss_mb"] = max(runtimes.get("peak_rss_mb", 0.0), resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
        write_json(runtimes_path, runtimes)
        log(f"stage {stage} done in {runtimes[stage]['wall_seconds']:.0f}s")

    def need(path: Path) -> bool:
        return args.force or not path.exists()

    best_path_knn, best_path_tree = out / "knn_best_params.json", out / "tree_best_params.json"
    if need(best_path_knn):
        run("tune_knn", lambda: tune_knn(dev, out))
    if need(best_path_tree):
        run("tune_tree", lambda: tune_tree(dev, out, args.workers))
    if not (best_path_knn.exists() and best_path_tree.exists()):
        return
    tree_best = json.loads(best_path_tree.read_text())
    best = {"tree": tree_best}
    best.update({k: v for k, v in json.loads(best_path_knn.read_text()).items() if k in ("knn", "knn_cw")})
    names = ["knn", "knn_cw", "tree"]
    if tree_best["class_weight"] != "balanced":
        best["tree_balanced"] = tree_best["balanced_variant"]
        names.append("tree_balanced")
    if need(out / "comparison_per_class.csv"):
        run("compare", lambda: compare(dev, best, out, names))
    if need(out / "generalisation_scores.csv"):
        run("final", lambda: final_fit(dev, gen, best, out, names))
    if "analyse" in stages and (out / "generalisation_scores.csv").exists():
        run("analyse", lambda: analyse(out, names))


if __name__ == "__main__":
    main()
