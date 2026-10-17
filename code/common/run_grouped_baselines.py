"""Baselines on the leakage-aware grouped protocol (identical feature vectors stay together).

The primary protocol (results/common/*.csv, written by run_baselines.py) is not touched. This
script adds the grouped protocol of common.grouped and writes results/common/grouped_*.

Stages (each skips its work when its output exists, unless --force)
-------
split     grouped hold-out D_g / G_g, class balance of the folds, twin shares of both protocols
tune      the grids of run_baselines.py on the grouped tuning folds; reports whether the best
          configuration differs from the one tuned on the standard protocol
compare   the four Assignment 2 models with the parameters of the STANDARD protocol on the 15
          grouped partitions (primary result), the models whose retuned configuration differs
          (suffix "_retuned"), and the final fit on D_g with one evaluation on G_g
analyse   statistics, side-by-side ranking against the primary protocol, conclusion

Usage (from the repository root):
    nohup .venv/bin/python code/common/run_grouped_baselines.py > logs/grouped_baselines.log 2>&1 &
"""
from __future__ import annotations

import argparse
import os

_THREADS = os.environ.get("GROUPED_THREADS", "8")
for _name in ("NUMBA_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = _THREADS

import itertools  # noqa: E402
import json  # noqa: E402
import resource  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ThreadPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import data, grouped, metrics, protocol, run_baselines as rb, stats  # noqa: E402
from common.knn import KNNClassifier, class_weight_vector  # noqa: E402
from common.tree import BinMapper, DecisionTreeClassifier  # noqa: E402

N_CLASSES = data.N_CLASSES
MEASURES = metrics.MEASURES
RES = data.RESULTS_COMMON
BASE_NAMES = ["knn", "knn_cw", "tree", "tree_balanced"]
log = rb.log
write_json = rb.write_json


def load_context():
    table = data.load_table()
    split = grouped.grouped_split(table)
    dev, gen = table.take(split.dev_idx), table.take(split.gen_idx)
    return table, split, dev, gen


def standard_best() -> dict:
    """Parameters tuned on the standard protocol (read only), in the layout of run_baselines.make_model."""
    tree = json.loads((RES / "tree_best_params.json").read_text())
    knn = json.loads((RES / "knn_best_params.json").read_text())
    best = {"tree": tree, "knn": knn["knn"], "knn_cw": knn["knn_cw"]}
    if tree["class_weight"] != "balanced":
        best["tree_balanced"] = tree["balanced_variant"]
    return best


# ---------------------------------------------------------------------------
# Stage split
# ---------------------------------------------------------------------------

def stage_split(table, split, dev, gen, out: Path) -> None:
    y_all = table.y
    _, shared_gen = protocol.stratified_holdout(y_all)
    shared_dev = np.setdiff1d(np.arange(table.n), shared_gen)
    parts_t = grouped.grouped_tuning_partitions(dev.y, split.groups_dev)
    parts_c = grouped.grouped_comparison_partitions(dev.y, split.groups_dev)
    shared_groups_dev = split.groups[shared_dev]
    shared_y_dev = y_all[shared_dev]
    std_t = protocol.tuning_partitions(shared_y_dev)
    std_c = protocol.comparison_partitions(shared_y_dev)

    rows = []
    for label, parts, groups, y in (("standard_tuning", std_t, shared_groups_dev, shared_y_dev),
                                    ("standard_comparison", std_c, shared_groups_dev, shared_y_dev),
                                    ("grouped_tuning", parts_t, split.groups_dev, dev.y),
                                    ("grouped_comparison", parts_c, split.groups_dev, dev.y)):
        for p in parts:
            rows.append({"protocol": label, "repeat": p.repeat, "fold": p.fold, "n_test": p.test.size,
                         "twin_share": grouped.twin_share(groups, p),
                         "twin_share_same_label": grouped.twin_share_same_label(groups, y, p),
                         "groups_crossing": grouped.groups_cross(groups, p)})
    twins = pd.DataFrame(rows)
    twins.to_csv(out / "grouped_twin_share.csv", index=False)

    shared_g_twin = float(np.isin(split.groups[shared_gen], np.unique(shared_groups_dev)).mean())
    grouped_g_twin = float(np.isin(split.groups_gen, np.unique(split.groups_dev)).mean())
    counts = {"total": data.class_counts(y_all), "shared_G": data.class_counts(y_all[shared_gen]),
              "shared_D": data.class_counts(shared_y_dev), "G_g": data.class_counts(gen.y),
              "D_g": data.class_counts(dev.y)}
    pd.DataFrame({"class": range(N_CLASSES), "name": data.CLASS_NAMES, **counts}).to_csv(
        out / "grouped_class_counts.csv", index=False)
    balance_t = grouped.class_balance_report(dev.y, parts_t)
    balance_c = grouped.class_balance_report(dev.y, parts_c)
    w7 = {"class7_total": int(counts["total"][7]), "class7_shared_G": int(counts["shared_G"][7]),
          "class7_G_g": int(counts["G_g"][7]), "class7_D_g": int(counts["D_g"][7]),
          "class7_groups_total": int(np.unique(split.groups[y_all == 7]).size),
          "class7_test_count_per_fold_min": balance_c["min_test_count_per_class"][7],
          "class7_test_count_per_fold_max": balance_c["max_test_count_per_class"][7]}
    classes_too_small = [int(c) for c in range(N_CLASSES)
                         if min(counts["G_g"][c], balance_c["min_test_count_per_class"][c]) < 20]
    summary = {
        "master_seed": protocol.MASTER_SEED,
        "purposes": {"holdout": grouped.PURPOSE_GROUPED_SPLIT, "tuning": grouped.PURPOSE_GROUPED_TUNING,
                     "comparison": grouped.PURPOSE_GROUPED_COMPARISON},
        "n_total": table.n, "n_development_grouped": dev.n, "n_generalisation_grouped": gen.n,
        "n_development_standard": int(shared_dev.size), "n_generalisation_standard": int(shared_gen.size),
        "groups_full_table": grouped.group_summary(split.groups, y_all),
        "groups_development_standard": grouped.group_summary(shared_groups_dev, shared_y_dev),
        "groups_development_grouped": grouped.group_summary(split.groups_dev, dev.y),
        "groups_shared_G": grouped.group_summary(split.groups[shared_gen], y_all[shared_gen]),
        "class_counts": {k: v for k, v in counts.items()},
        "max_abs_class_count_difference_G_g_vs_shared_G": int(np.abs(counts["G_g"] - counts["shared_G"]).max()),
        "class7": w7, "classes_with_fewer_than_20_test_rows": classes_too_small,
        "twin_share_G_standard": shared_g_twin, "twin_share_G_grouped": grouped_g_twin,
        "tuning_balance": balance_t, "comparison_balance": balance_c,
        "grouped_development_index_digest": protocol.index_digest(split.dev_idx),
        "grouped_generalisation_index_digest": protocol.index_digest(split.gen_idx),
        "mean_twin_share": {k: float(v) for k, v in twins.groupby("protocol")["twin_share"].mean().items()},
        "mean_twin_share_same_label": {k: float(v) for k, v in
                                       twins.groupby("protocol")["twin_share_same_label"].mean().items()},
        "max_groups_crossing_grouped": int(twins[twins["protocol"].str.startswith("grouped")]["groups_crossing"].max())}
    write_json(out / "grouped_split.json", summary)
    log(f"split: D_g {dev.n}, G_g {gen.n}, class 7 in G_g {w7['class7_G_g']}; mean twin share "
        f"{summary['mean_twin_share']}")


# ---------------------------------------------------------------------------
# Stage tune
# ---------------------------------------------------------------------------

def tune_knn_grouped(dev, parts, out: Path) -> dict:
    kmax = max(rb.KNN_K)
    records, started = [], time.perf_counter()
    for part in parts:
        x_train, y_train, x_test, y_test, _ = rb.knn_arrays(dev, part)
        for metric in rb.KNN_METRICS:
            model = KNNClassifier(kmax, metric, n_classes=N_CLASSES).fit(x_train, y_train)
            t0 = time.perf_counter()
            dist, idx = model.kneighbors(x_test, kmax)
            log(f"knn grouped tuning fold {part.fold} {metric}: search {time.perf_counter() - t0:.1f}s")
            for power in rb.KNN_CLASS_POWER:
                cw = class_weight_vector(y_train, N_CLASSES, None if power == 0 else power)
                for weights, k in itertools.product(rb.KNN_WEIGHTS, rb.KNN_K):
                    pred = KNNClassifier.predict_from_neighbours(idx, dist, y_train, k, weights, cw, N_CLASSES)
                    records.append({"fold": part.fold, "metric": metric, "k": k, "weights": weights,
                                    "class_weight_power": power,
                                    **metrics.score_all(y_test, pred, N_CLASSES)})
    frame = pd.DataFrame(records)
    frame.to_csv(out / "grouped_knn_tuning_folds.csv", index=False)
    keys = ["metric", "k", "weights", "class_weight_power"]
    grid = frame.groupby(keys, sort=False)[list(MEASURES)].agg(["mean", "std"])
    grid.columns = [f"{a}_{b}" for a, b in grid.columns]
    grid = grid.reset_index()
    grid.to_csv(out / "grouped_knn_tuning_grid.csv", index=False)
    plain = grid[grid["class_weight_power"] == 0.0]

    def params(row):
        return {"metric": row["metric"], "k": int(row["k"]), "weights": row["weights"],
                "class_weight_power": float(row["class_weight_power"]),
                "cv_macro_f1_mean": float(row["macro_f1_mean"]), "cv_macro_f1_std": float(row["macro_f1_std"])}
    best = {"knn": params(plain.loc[plain["macro_f1_mean"].idxmax()]),
            "knn_cw": params(grid.loc[grid["macro_f1_mean"].idxmax()]),
            "n_configurations": int(len(grid)), "wall_seconds": time.perf_counter() - started}
    return best


def tune_tree_grouped(dev, parts, out: Path, workers: int) -> dict:
    started = time.perf_counter()
    folds = []
    for part in parts:
        x_train, y_train, x_test, y_test, _ = rb.tree_arrays(dev, part)
        binner = BinMapper().fit(x_train)
        folds.append((x_train, y_train, binner, binner.transform(x_train), x_test, y_test))
    configs = list(itertools.product(rb.TREE_CRITERIA, rb.TREE_DEPTHS, rb.TREE_LEAVES, rb.TREE_CLASS_WEIGHTS))
    seed = protocol.seed_int("baseline-tree")

    def job(args):
        (criterion, depth, leaf, cw), fold = args
        x_train, y_train, binner, xb, x_test, y_test = folds[fold]
        model = DecisionTreeClassifier(criterion, depth, leaf, class_weight=cw, seed=seed)
        model.fit(x_train, y_train, n_classes=N_CLASSES, binner=binner, Xb_t=xb)
        pred = model.predict(x_test)
        return {"fold": fold, "criterion": criterion, "max_depth": -1 if depth is None else depth,
                "min_samples_leaf": leaf, "class_weight": "none" if cw is None else cw,
                "n_nodes": model.n_nodes_, **metrics.score_all(y_test, pred, N_CLASSES)}

    jobs = [(c, f) for c in configs for f in range(len(folds))]
    with ThreadPoolExecutor(workers) as pool:
        records = list(pool.map(job, jobs))
    frame = pd.DataFrame(records)
    frame.to_csv(out / "grouped_tree_tuning_folds.csv", index=False)
    keys = ["criterion", "max_depth", "min_samples_leaf", "class_weight"]
    grid = frame.groupby(keys, sort=False)[list(MEASURES) + ["n_nodes"]].agg(["mean", "std"])
    grid.columns = [f"{a}_{b}" for a, b in grid.columns]
    grid = grid.reset_index()
    grid.to_csv(out / "grouped_tree_tuning_grid.csv", index=False)

    def params(row):
        return {"criterion": row["criterion"], "max_depth": None if int(row["max_depth"]) < 0 else int(row["max_depth"]),
                "min_samples_leaf": int(row["min_samples_leaf"]),
                "class_weight": None if row["class_weight"] == "none" else row["class_weight"],
                "cv_macro_f1_mean": float(row["macro_f1_mean"]), "cv_macro_f1_std": float(row["macro_f1_std"])}
    best = params(grid.loc[grid["macro_f1_mean"].idxmax()])
    balanced = grid[grid["class_weight"] == "balanced"]
    best["balanced_variant"] = params(balanced.loc[balanced["macro_f1_mean"].idxmax()])
    best["n_configurations"] = int(len(grid))
    best["wall_seconds"] = time.perf_counter() - started
    return best


def same_params(a: dict, b: dict) -> bool:
    keys = [k for k in ("metric", "k", "weights", "class_weight_power", "criterion", "max_depth",
                        "min_samples_leaf", "class_weight") if k in a]
    return all(a[k] == b.get(k) for k in keys)


def grid_score_of(grid: pd.DataFrame, params: dict, kind: str) -> float | None:
    if kind == "knn":
        sel = grid[(grid["metric"] == params["metric"]) & (grid["k"] == params["k"])
                   & (grid["weights"] == params["weights"])
                   & (np.isclose(grid["class_weight_power"], params["class_weight_power"]))]
    else:
        depth = -1 if params["max_depth"] is None else params["max_depth"]
        sel = grid[(grid["criterion"] == params["criterion"]) & (grid["max_depth"] == depth)
                   & (grid["min_samples_leaf"] == params["min_samples_leaf"])
                   & (grid["class_weight"] == ("none" if params["class_weight"] is None else params["class_weight"]))]
    return float(sel["macro_f1_mean"].iloc[0]) if len(sel) else None


def stage_tune(dev, split, out: Path, workers: int) -> None:
    parts = grouped.grouped_tuning_partitions(dev.y, split.groups_dev)
    knn_best = tune_knn_grouped(dev, parts, out)
    write_json(out / "grouped_knn_best_params.json", knn_best)
    tree_best = tune_tree_grouped(dev, parts, out, workers)
    write_json(out / "grouped_tree_best_params.json", tree_best)
    std = standard_best()
    knn_grid = pd.read_csv(out / "grouped_knn_tuning_grid.csv")
    tree_grid = pd.read_csv(out / "grouped_tree_tuning_grid.csv")
    grouped_best = {"knn": knn_best["knn"], "knn_cw": knn_best["knn_cw"], "tree": tree_best}
    if "tree_balanced" in std:
        grouped_best["tree_balanced"] = tree_best["balanced_variant"]
    report = {}
    for name, p_std in std.items():
        p_new = grouped_best[name]
        kind = "tree" if name.startswith("tree") else "knn"
        grid = tree_grid if kind == "tree" else knn_grid
        report[name] = {"standard_protocol_best": {k: v for k, v in p_std.items() if not k.startswith("cv_")
                                                   and k not in ("balanced_variant", "n_configurations", "wall_seconds",
                                                                 "n_folds", "best_macro_f1_balanced",
                                                                 "best_macro_f1_unweighted")},
                        "grouped_best": {k: v for k, v in p_new.items() if not k.startswith("cv_")
                                         and k not in ("balanced_variant", "n_configurations", "wall_seconds")},
                        "configuration_changed": not same_params(p_std, p_new),
                        "grouped_cv_macro_f1_of_grouped_best": p_new["cv_macro_f1_mean"],
                        "grouped_cv_macro_f1_of_standard_best": grid_score_of(grid, p_std, kind),
                        "standard_cv_macro_f1_of_standard_best": p_std["cv_macro_f1_mean"]}
    write_json(out / "grouped_retuning.json", report)
    for name, r in report.items():
        log(f"retune {name}: changed {r['configuration_changed']}  {r['standard_protocol_best']} -> {r['grouped_best']}")


# ---------------------------------------------------------------------------
# Stage compare
# ---------------------------------------------------------------------------

def stage_compare(dev, gen, split, out: Path) -> None:
    std = standard_best()
    retune = json.loads((out / "grouped_retuning.json").read_text())
    grouped_best = {}
    knn_b = json.loads((out / "grouped_knn_best_params.json").read_text())
    tree_b = json.loads((out / "grouped_tree_best_params.json").read_text())
    grouped_best.update({"knn": knn_b["knn"], "knn_cw": knn_b["knn_cw"], "tree": tree_b})
    if "tree_balanced" in std:
        grouped_best["tree_balanced"] = tree_b["balanced_variant"]
    # model table: name -> (parameter set)
    models = {n: std[n] for n in BASE_NAMES if n in std}
    for n in list(models):
        if retune[n]["configuration_changed"]:
            models[f"{n}_retuned"] = grouped_best[n]
    def kind_of(name):
        return "tree" if name.startswith("tree") else "knn"

    def run_model(name, arrays):
        base = name.replace("_retuned", "")
        x_train, y_train, x_test = arrays[0], arrays[1], arrays[2]
        model = rb.make_model(base, {base: models[name]})
        t0 = time.perf_counter()
        if kind_of(name) == "tree":
            model.fit(x_train, y_train, n_classes=N_CLASSES)
        else:
            model.fit(x_train, y_train)
        fit_time = time.perf_counter() - t0
        t0 = time.perf_counter()
        pred = model.predict(x_test)
        return model, pred, fit_time, time.perf_counter() - t0

    parts = grouped.grouped_comparison_partitions(dev.y, split.groups_dev)
    score_rows, class_rows = [], []
    confusion = {n: np.zeros((N_CLASSES, N_CLASSES), dtype=np.int64) for n in models}
    for part in parts:
        arrays = {"knn": rb.knn_arrays(dev, part), "tree": rb.tree_arrays(dev, part)}
        for name in models:
            a = arrays[kind_of(name)]
            _, pred, fit_time, predict_time = run_model(name, a)
            cm = metrics.confusion_matrix(a[3], pred, N_CLASSES)
            confusion[name] += cm
            scores = metrics.scores_from_confusion(cm)
            score_rows.append({"model": name, "repeat": part.repeat, "fold": part.fold, **scores,
                               "prep_seconds": a[4], "fit_seconds": fit_time, "predict_seconds": predict_time})
            for row in metrics.per_class_table(cm):
                class_rows.append({"model": name, "repeat": part.repeat, "fold": part.fold, **row})
            log(f"grouped compare {name} r{part.repeat} f{part.fold}: macro-F1 {scores['macro_f1']:.4f} "
                f"fit {fit_time:.1f}s predict {predict_time:.1f}s")
        pd.DataFrame(score_rows).to_csv(out / "grouped_comparison_scores.csv", index=False)
    pd.DataFrame(class_rows).to_csv(out / "grouped_comparison_per_class.csv", index=False)
    for name in models:
        np.savetxt(out / f"grouped_confusion_cv_{name}.csv", confusion[name], fmt="%d", delimiter=",")

    # final fit on D_g, one evaluation on G_g
    arrays = {}
    for kind, builder in (("knn", preprocess_for_knn), ("tree", preprocess_for_tree)):
        t0 = time.perf_counter()
        pre = builder()
        x_dev = pre.fit_transform(dev)
        x_gen = pre.transform(gen)
        arrays[kind] = (x_dev, dev.y, x_gen, gen.y, time.perf_counter() - t0, pre)
    score_rows, class_rows = [], []
    for name in models:
        a = arrays[kind_of(name)]
        model, pred, fit_time, predict_time = run_model(name, a)
        cm = metrics.confusion_matrix(gen.y, pred, N_CLASSES)
        scores = metrics.scores_from_confusion(cm)
        score_rows.append({"model": name, **scores, "prep_seconds": a[4], "fit_seconds": fit_time,
                           "predict_seconds": predict_time, "n_train": dev.n, "n_test": gen.n})
        for row in metrics.per_class_table(cm):
            class_rows.append({"model": name, **row})
        np.savetxt(out / f"grouped_confusion_G_{name}.csv", cm, fmt="%d", delimiter=",")
        log(f"grouped G {name}: {scores}")
    pd.DataFrame(score_rows).to_csv(out / "grouped_generalisation_scores.csv", index=False)
    pd.DataFrame(class_rows).to_csv(out / "grouped_generalisation_per_class.csv", index=False)
    write_json(out / "grouped_models.json", {n: {k: v for k, v in p.items() if k != "balanced_variant"}
                                             for n, p in models.items()})


def preprocess_for_knn():
    from common import preprocess
    return preprocess.Preprocessor.for_knn()


def preprocess_for_tree():
    from common import preprocess
    return preprocess.Preprocessor.for_tree()


# ---------------------------------------------------------------------------
# Stage analyse
# ---------------------------------------------------------------------------

def ranks_desc(values: dict) -> dict:
    order = sorted(values, key=lambda k: -values[k])
    return {k: order.index(k) + 1 for k in values}


def stage_analyse(out: Path) -> None:
    sc = pd.read_csv(out / "grouped_comparison_scores.csv")
    gen = pd.read_csv(out / "grouped_generalisation_scores.csv").set_index("model")
    names_all = list(dict.fromkeys(sc["model"]))
    names = [n for n in BASE_NAMES if n in names_all]
    std_scores = pd.read_csv(RES / "comparison_scores.csv")
    std_summary = pd.read_csv(RES / "comparison_summary.csv").set_index("model")
    std_gen = pd.read_csv(RES / "generalisation_scores.csv").set_index("model")

    def vec(frame, name, measure):
        return frame[frame["model"] == name].sort_values(["repeat", "fold"])[measure].to_numpy()

    summary = []
    for name in names_all:
        row = {"model": name, "n_scores": int((sc["model"] == name).sum())}
        for m in MEASURES:
            row[f"{m}_mean"], row[f"{m}_std"] = rb.mean_std(vec(sc, name, m))
            row[f"{m}_G"] = float(gen.loc[name, m])
        for c in ("prep_seconds", "fit_seconds", "predict_seconds"):
            row[f"{c}_mean"] = float(vec(sc, name, c).mean())
            row[f"{c}_G"] = float(gen.loc[name, c])
        summary.append(row)
    pd.DataFrame(summary).to_csv(out / "grouped_comparison_summary.csv", index=False)

    # pairwise tests (same pairs as run_baselines) on the standard-parameter models
    pair_rows = []
    pairs_to_test = [("tree", "knn"), ("tree", "knn_cw"), ("knn_cw", "knn"),
                     ("tree_balanced", "knn"), ("tree_balanced", "knn_cw"), ("tree_balanced", "tree")]
    for a, b in [p for p in pairs_to_test if p[0] in names and p[1] in names]:
        for m in MEASURES:
            va, vb = vec(sc, a, m), vec(sc, b, m)
            wil, cor = stats.paired_wilcoxon(va, vb), stats.corrected_resampled_ttest(va, vb)
            pair_rows.append({"model_a": a, "model_b": b, "measure": m, "mean_a": float(va.mean()),
                              "mean_b": float(vb.mean()), "mean_diff": wil["mean_diff"],
                              "median_diff": wil["median_diff"], "wilcoxon_statistic": wil["statistic"],
                              "wilcoxon_p": wil["p_value"], "cohens_dz": wil["cohens_dz"],
                              "rank_biserial": wil["rank_biserial"], "n_a_higher": wil["n_a_higher"],
                              "n_b_higher": wil["n_b_higher"], "corrected_t": cor["t"],
                              "corrected_t_p": cor["p_value"]})
    pairs = pd.DataFrame(pair_rows)
    pairs["wilcoxon_p_holm"] = np.nan
    for m in MEASURES:
        sel = pairs["measure"] == m
        pairs.loc[sel, "wilcoxon_p_holm"] = stats.holm_adjust(pairs.loc[sel, "wilcoxon_p"])
    pairs.to_csv(out / "grouped_comparison_stats.csv", index=False)

    fr_rows, post_rows = [], []
    for m in MEASURES:
        matrix = np.column_stack([vec(sc, n, m) for n in names])
        fr, nem = stats.friedman_test(matrix), stats.nemenyi_posthoc(matrix)
        fr_rows.append({"measure": m, "friedman_statistic": fr["statistic"], "friedman_p": fr["p_value"],
                        "iman_davenport_f": fr["iman_davenport_f"], "iman_davenport_p": fr["iman_davenport_p"],
                        "nemenyi_critical_difference": nem["critical_difference"],
                        **{f"rank_{n}": float(r) for n, r in zip(names, fr["average_ranks"])}})
        holm = {(r["model_a"], r["model_b"]): r["p_holm"] for r in stats.holm_wilcoxon_posthoc(matrix, names)}
        for i, j in itertools.combinations(range(len(names)), 2):
            post_rows.append({"measure": m, "model_a": names[i], "model_b": names[j],
                              "nemenyi_p": float(nem["p_values"][i, j]),
                              "holm_wilcoxon_p": holm[(names[i], names[j])]})
    pd.DataFrame(fr_rows).to_csv(out / "grouped_comparison_friedman.csv", index=False)
    pd.DataFrame(post_rows).to_csv(out / "grouped_comparison_posthoc.csv", index=False)

    # retuned versus reused parameters (paired on the same 15 partitions)
    ret_rows = []
    for n in names:
        if f"{n}_retuned" in names_all:
            for m in MEASURES:
                a, b = vec(sc, f"{n}_retuned", m), vec(sc, n, m)
                w = stats.paired_wilcoxon(a, b)
                ret_rows.append({"model": n, "measure": m, "mean_retuned": float(a.mean()),
                                 "mean_reused": float(b.mean()), "mean_diff": w["mean_diff"],
                                 "wilcoxon_p": w["p_value"]})
    pd.DataFrame(ret_rows, columns=["model", "measure", "mean_retuned", "mean_reused", "mean_diff",
                                    "wilcoxon_p"]).to_csv(out / "grouped_retuned_vs_reused.csv", index=False)

    # side-by-side table: primary and grouped protocol
    side = []
    for m in MEASURES:
        prim = {n: float(std_summary.loc[n, f"{m}_mean"]) for n in names}
        grp = {n: float(sc[sc["model"] == n][m].mean()) for n in names}
        rp, rg = ranks_desc(prim), ranks_desc(grp)
        pg = {n: float(std_summary.loc[n, f"{m}_G"]) for n in names}
        gg = {n: float(gen.loc[n, m]) for n in names}
        rpg, rgg = ranks_desc(pg), ranks_desc(gg)
        for n in names:
            side.append({"measure": m, "model": n, "primary_mean": prim[n],
                         "primary_std": float(std_summary.loc[n, f"{m}_std"]), "primary_rank": rp[n],
                         "grouped_mean": grp[n], "grouped_std": float(sc[sc["model"] == n][m].std(ddof=1)),
                         "grouped_rank": rg[n], "difference": grp[n] - prim[n],
                         "primary_G": pg[n], "primary_G_rank": rpg[n],
                         "grouped_G": gg[n], "grouped_G_rank": rgg[n]})
    side = pd.DataFrame(side)
    side.to_csv(out / "grouped_ranking_comparison.csv", index=False)

    # conclusion
    from scipy import stats as sps
    concl = {"models": names, "per_measure": {}}
    for m in MEASURES:
        sub = side[side["measure"] == m].set_index("model")
        tau = float(sps.kendalltau(sub["primary_rank"], sub["grouped_rank"]).statistic)
        tau_g = float(sps.kendalltau(sub["primary_G_rank"], sub["grouped_G_rank"]).statistic)
        concl["per_measure"][m] = {
            "ranking_primary": sub.sort_values("primary_rank").index.tolist(),
            "ranking_grouped": sub.sort_values("grouped_rank").index.tolist(),
            "ranking_unchanged": bool((sub["primary_rank"] == sub["grouped_rank"]).all()),
            "kendall_tau": tau,
            "ranking_unchanged_on_G": bool((sub["primary_G_rank"] == sub["grouped_G_rank"]).all()),
            "kendall_tau_G": tau_g}

    def a2(frame_mean):                          # Assignment 2 conclusion on a table of means
        return {"tree_macro_f1_higher": frame_mean("tree", "macro_f1") > frame_mean("knn", "macro_f1"),
                "tree_balanced_accuracy_higher": frame_mean("tree", "balanced_accuracy") > frame_mean("knn", "balanced_accuracy"),
                "knn_accuracy_higher": frame_mean("knn", "accuracy") > frame_mean("tree", "accuracy")}
    prim_a2 = a2(lambda n, m: float(std_summary.loc[n, f"{m}_mean"]))
    grp_a2 = a2(lambda n, m: float(sc[sc["model"] == n][m].mean()))
    prim_a2_g = a2(lambda n, m: float(std_summary.loc[n, f"{m}_G"]))
    grp_a2_g = a2(lambda n, m: float(gen.loc[n, m]))
    concl["assignment2_conclusion_primary"] = {k: bool(v) for k, v in prim_a2.items()}
    concl["assignment2_conclusion_grouped"] = {k: bool(v) for k, v in grp_a2.items()}
    concl["assignment2_conclusion_primary_G"] = {k: bool(v) for k, v in prim_a2_g.items()}
    concl["assignment2_conclusion_grouped_G"] = {k: bool(v) for k, v in grp_a2_g.items()}
    concl["assignment2_conclusion_unchanged"] = bool(prim_a2 == grp_a2)
    concl["assignment2_conclusion_unchanged_on_G"] = bool(prim_a2_g == grp_a2_g)
    concl["tree_vs_knn_wilcoxon_p_grouped"] = {
        m: float(pairs[(pairs["model_a"] == "tree") & (pairs["model_b"] == "knn") & (pairs["measure"] == m)]["wilcoxon_p"].iloc[0])
        for m in MEASURES}
    concl["tree_minus_knn_mean_difference_grouped"] = {
        m: float(pairs[(pairs["model_a"] == "tree") & (pairs["model_b"] == "knn") & (pairs["measure"] == m)]["mean_diff"].iloc[0])
        for m in MEASURES}
    prim_stats = pd.read_csv(RES / "comparison_stats.csv")
    concl["tree_minus_knn_mean_difference_primary"] = {
        m: float(prim_stats[(prim_stats["model_a"] == "tree") & (prim_stats["model_b"] == "knn") & (prim_stats["measure"] == m)]["mean_diff"].iloc[0])
        for m in MEASURES}
    write_json(out / "grouped_conclusion.json", concl)
    for m, r in concl["per_measure"].items():
        log(f"{m}: primary {r['ranking_primary']}  grouped {r['ranking_grouped']}  unchanged {r['ranking_unchanged']}")
    log(f"Assignment 2 conclusion unchanged: {concl['assignment2_conclusion_unchanged']} "
        f"(primary {concl['assignment2_conclusion_primary']}, grouped {concl['assignment2_conclusion_grouped']})")


# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=str(data.RESULTS_COMMON))
    parser.add_argument("--stages", default="split,tune,compare,analyse")
    parser.add_argument("--workers", type=int, default=int(_THREADS), help="threads for the tree tuning grid")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stages = args.stages.split(",")
    rt_path = out / "grouped_runtimes.json"
    runtimes = json.loads(rt_path.read_text()) if rt_path.exists() else {}
    runtimes["threads"] = int(_THREADS)
    table, split, dev, gen = load_context()
    log(f"grouped protocol: D_g {dev.n} rows, G_g {gen.n} rows, threads {_THREADS}")

    def run(stage, fn, done: Path):
        if stage not in stages or (done.exists() and not args.force):
            return
        t0 = time.perf_counter()
        fn()
        runtimes[stage] = {"wall_seconds": time.perf_counter() - t0}
        runtimes["peak_rss_mb"] = max(runtimes.get("peak_rss_mb", 0.0),
                                      resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
        write_json(rt_path, runtimes)
        log(f"stage {stage} done in {runtimes[stage]['wall_seconds']:.0f}s")

    run("split", lambda: stage_split(table, split, dev, gen, out), out / "grouped_split.json")
    run("tune", lambda: stage_tune(dev, split, out, args.workers), out / "grouped_retuning.json")
    run("compare", lambda: stage_compare(dev, gen, split, out), out / "grouped_generalisation_scores.csv")
    run("analyse", lambda: stage_analyse(out), out / "grouped_conclusion.json")


if __name__ == "__main__":
    main()
