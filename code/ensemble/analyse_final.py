"""Statistics and summaries of the final comparison (results/ensemble/comparison_runs.csv, generalisation_runs.csv).

Outputs in results/ensemble:
  comparison_scores.csv       one row per model and partition (mean over the runs of a forest)
  comparison_summary.csv      mean and standard deviation over the 15 partitions, runs, scores on G
  comparison_stats.csv        paired tests of selected pairs: Wilcoxon, effect sizes, corrected t-test
  comparison_friedman.csv     Friedman test and Nemenyi critical difference per measure
  comparison_posthoc.csv      pairwise Wilcoxon tests with Holm adjustment
  twin_summary.csv            scores on test rows with and without an identical twin in the training part
  per_class_cv.csv, per_class_G.csv   per-class precision, recall and F1 (pooled confusion matrices)
  reproduction_check.json     difference between the refitted baselines and results/common

Usage: .venv/bin/python code/ensemble/analyse_final.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import data, metrics, stats  # noqa: E402
from ensemble.harness import RESULTS  # noqa: E402

MEASURES = list(metrics.MEASURES)
MODELS = ["knn", "knn_cw", "tree", "tree_balanced", "rf", "rf_balanced", "boost"]
PAIRS = [("rf", "tree"), ("rf_balanced", "tree_balanced"), ("boost", "tree"), ("rf", "knn_cw"),
         ("rf", "rf_balanced"), ("rf", "boost"), ("rf_balanced", "boost"), ("rf_balanced", "tree"),
         ("rf_balanced", "knn_cw")]
COMMON = data.RESULTS_COMMON
N_CLASSES = data.N_CLASSES


def main() -> None:
    runs = pd.read_csv(RESULTS / "comparison_runs.csv")
    num = [c for c in runs.columns if c not in ("model", "repeat", "fold", "run", "encoding")]
    per_part = runs.groupby(["model", "repeat", "fold"])[num].mean().reset_index()
    sd_runs = runs.groupby(["model", "repeat", "fold"])[MEASURES].std(ddof=1).reset_index()
    per_part.to_csv(RESULTS / "comparison_scores.csv", index=False)

    # check of the refitted baselines against results/common
    common = pd.read_csv(COMMON / "comparison_scores.csv")
    check = {}
    for m in ("knn", "knn_cw", "tree", "tree_balanced"):
        a = per_part[per_part["model"] == m].sort_values(["repeat", "fold"])
        b = common[common["model"] == m].sort_values(["repeat", "fold"])
        if len(a) == len(b) == 15:
            check[m] = {k: float(np.abs(a[k].to_numpy() - b[k].to_numpy()).max()) for k in MEASURES}
    (RESULTS / "reproduction_check.json").write_text(json.dumps(check, indent=2))

    # summary
    gen = pd.read_csv(RESULTS / "generalisation_runs.csv")
    gcommon = pd.read_csv(COMMON / "generalisation_scores.csv").set_index("model")
    rows = []
    for m in MODELS:
        sub = per_part[per_part["model"] == m].sort_values(["repeat", "fold"])
        if sub.empty:
            continue
        row = {"model": m, "n_scores": len(sub)}
        for k in MEASURES:
            row[f"{k}_mean"], row[f"{k}_std"] = float(sub[k].mean()), float(sub[k].std(ddof=1))
            s = sd_runs[sd_runs["model"] == m][k]
            row[f"{k}_run_std"] = float(s.mean()) if s.notna().any() else 0.0
            if m in gcommon.index:
                row[f"{k}_G"], row[f"{k}_G_std"] = float(gcommon.loc[m, k]), 0.0
            else:
                g = gen[gen["model"] == m][k]
                row[f"{k}_G"], row[f"{k}_G_std"] = float(g.mean()), float(g.std(ddof=1)) if len(g) > 1 else 0.0
        for c in ("fit_seconds", "total_seconds", "member_accuracy", "disagreement", "q_statistic",
                  "oob_accuracy", "oob_macro_f1", "oob_balanced_accuracy"):
            if c in sub.columns and sub[c].notna().any():
                row[f"{c}_mean"] = float(sub[c].mean())
        g = gen[gen["model"] == m]
        if len(g):
            row["G_fit_seconds"] = float(g["fit_seconds"].mean())
            row["G_predict_seconds"] = float(g["predict_seconds"].mean())
            row["G_n_runs"] = int(len(g))
        elif m in gcommon.index:
            row["G_fit_seconds"] = float(gcommon.loc[m, "fit_seconds"])
            row["G_predict_seconds"] = float(gcommon.loc[m, "predict_seconds"])
            row["G_n_runs"] = 1
        rows.append(row)
    pd.DataFrame(rows).to_csv(RESULTS / "comparison_summary.csv", index=False)

    # paired tests
    piv = {k: per_part.pivot_table(index=["repeat", "fold"], columns="model", values=k) for k in MEASURES}
    stat_rows = []
    for a, b in PAIRS:
        if a not in piv["macro_f1"].columns or b not in piv["macro_f1"].columns:
            continue
        for k in MEASURES:
            x, y = piv[k][a].to_numpy(), piv[k][b].to_numpy()
            w = stats.paired_wilcoxon(x, y)
            c = stats.corrected_resampled_ttest(x, y)
            stat_rows.append({"model_a": a, "model_b": b, "measure": k, "mean_a": float(x.mean()),
                              "mean_b": float(y.mean()), "mean_diff": w["mean_diff"], "cohens_dz": w["cohens_dz"],
                              "rank_biserial": w["rank_biserial"], "wilcoxon_p": w["p_value"],
                              "n_a_higher": w["n_a_higher"], "n_b_higher": w["n_b_higher"],
                              "corrected_t": c["t"], "corrected_t_p": c["p_value"]})
    st = pd.DataFrame(stat_rows)
    st["wilcoxon_p_holm"] = np.nan
    for k in MEASURES:
        mask = st["measure"] == k
        st.loc[mask, "wilcoxon_p_holm"] = stats.holm_adjust(st.loc[mask, "wilcoxon_p"].to_numpy())
    st.to_csv(RESULTS / "comparison_stats.csv", index=False)

    # omnibus
    fr_rows, post_rows = [], []
    names = [m for m in MODELS if m in piv["macro_f1"].columns]
    for k in MEASURES:
        scores = piv[k][names].to_numpy()
        fr = stats.friedman_test(scores)
        nem = stats.nemenyi_posthoc(scores)
        row = {"measure": k, "friedman_statistic": fr["statistic"], "friedman_p": fr["p_value"],
               "iman_davenport_f": fr["iman_davenport_f"], "iman_davenport_p": fr["iman_davenport_p"],
               "nemenyi_critical_difference": nem["critical_difference"]}
        for n, r in zip(names, fr["average_ranks"]):
            row[f"rank_{n}"] = float(r)
        fr_rows.append(row)
        for r in stats.holm_wilcoxon_posthoc(scores, names):
            post_rows.append({"measure": k, "model_a": r["model_a"], "model_b": r["model_b"],
                              "mean_diff": r["mean_diff"], "wilcoxon_p": r["p_value"], "p_holm": r["p_holm"]})
    pd.DataFrame(fr_rows).to_csv(RESULTS / "comparison_friedman.csv", index=False)
    pd.DataFrame(post_rows).to_csv(RESULTS / "comparison_posthoc.csv", index=False)

    # twin analysis
    tw = per_part.groupby("model")[["twin_share", "accuracy_twin", "accuracy_new", "balanced_accuracy_twin",
                                    "balanced_accuracy_new"]].mean().reset_index()
    tw.to_csv(RESULTS / "twin_summary.csv", index=False)
    diffs = []
    for a, b in PAIRS[:4]:
        if a in piv["macro_f1"].columns and b in piv["macro_f1"].columns:
            pa = per_part[per_part["model"] == a].sort_values(["repeat", "fold"]).reset_index(drop=True)
            pb = per_part[per_part["model"] == b].sort_values(["repeat", "fold"]).reset_index(drop=True)
            for k in ("accuracy", "balanced_accuracy"):
                for grp in ("twin", "new"):
                    d = pa[f"{k}_{grp}"] - pb[f"{k}_{grp}"]
                    w = stats.paired_wilcoxon(pa[f"{k}_{grp}"], pb[f"{k}_{grp}"])
                    diffs.append({"model_a": a, "model_b": b, "measure": k, "group": grp,
                                  "mean_diff": float(d.mean()), "sd_diff": float(d.std(ddof=1)),
                                  "wilcoxon_p": w["p_value"], "cohens_dz": w["cohens_dz"]})
    pd.DataFrame(diffs).to_csv(RESULTS / "twin_differences.csv", index=False)

    # per-class tables from pooled confusion matrices
    conf = RESULTS / "confusion_runs"
    cv_rows = []
    for m in MODELS:
        files = sorted(conf.glob(f"{m}_[0-9]_*.csv"))
        if not files:
            continue
        total = sum(np.loadtxt(f, delimiter=",", dtype=np.int64) for f in files)
        pc = metrics.per_class(total)
        for j in range(N_CLASSES):
            cv_rows.append({"model": m, "class": j, "name": data.CLASS_NAMES[j], "precision": pc["precision"][j],
                            "recall": pc["recall"][j], "f1": pc["f1"][j], "support": int(pc["support"][j])})
    pd.DataFrame(cv_rows).to_csv(RESULTS / "per_class_cv.csv", index=False)
    g_rows = []
    for m in ("rf", "rf_balanced", "boost"):
        files = sorted(RESULTS.glob(f"confusion_G_{m}_run*.csv"))
        if not files:
            continue
        per_run = [metrics.per_class(np.loadtxt(f, delimiter=",", dtype=np.int64)) for f in files]
        for j in range(N_CLASSES):
            row = {"model": m, "class": j, "name": data.CLASS_NAMES[j], "support": int(per_run[0]["support"][j])}
            for k in ("precision", "recall", "f1"):
                v = np.array([r[k][j] for r in per_run])
                row[k], row[f"{k}_std"] = float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0
            g_rows.append(row)
    gc = pd.read_csv(COMMON / "generalisation_per_class.csv")
    for m in ("knn", "knn_cw", "tree", "tree_balanced"):
        for _, r in gc[gc["model"] == m].iterrows():
            g_rows.append({"model": m, "class": int(r["class"]), "name": r["name"], "support": int(r["support"]),
                           "precision": r["precision"], "recall": r["recall"], "f1": r["f1"]})
    pd.DataFrame(g_rows).to_csv(RESULTS / "per_class_G.csv", index=False)
    print(pd.DataFrame(rows)[["model", "accuracy_mean", "macro_f1_mean", "macro_f1_std", "balanced_accuracy_mean",
                              "macro_f1_G"]].to_string())


if __name__ == "__main__":
    main()
