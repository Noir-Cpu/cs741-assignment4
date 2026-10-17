"""Statistics of the compatible and the grouped comparison and of the generalisation sets.

Reads results/ensemble/protocol_{compatible,grouped}.csv and generalisation_{compatible,grouped}.csv and
writes in results/ensemble:

  protocol_scores_<p>.csv       one row per model and partition (mean over the runs of a forest)
  protocol_summary_<p>.csv      mean and standard deviation over the 15 partitions, run spread, twin categories
  protocol_stats_<p>.csv        paired tests of selected pairs: mean difference, effect size, Wilcoxon, corrected
                                t-test, Holm adjustment over the family of one measure
  protocol_ranking.csv          Kendall rank correlation of the models between the two protocols
  generalisation_summary_<p>.csv, generalisation_bootstrap_<p>.csv  scores on the generalisation set with
                                bootstrap intervals over the test instances and paired bootstrap differences
  per_class_<p>.csv             per-class precision, recall and F1 from the pooled confusion matrices

Usage: .venv/bin/python code/ensemble/analyse_protocols.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import data, grouped, metrics, protocol, stats  # noqa: E402
from ensemble.harness import RESULTS  # noqa: E402

MEASURES = list(metrics.MEASURES)
N_CLASSES = data.N_CLASSES
SINGLE = ["knn", "knn_cw", "tree", "tree_balanced"]
ENSEMBLES = ["rf", "rf_weighted", "rf_sampled", "boost"]
DISSIMILAR = ["vote_all", "vote_forests", "vote_rf_knn", "stack", "vote_without_rf", "vote_without_rf_weighted",
              "vote_without_tree", "vote_without_knn_cw"]
PAIRS = [("rf", "tree"), ("rf_weighted", "tree"), ("rf_sampled", "tree"), ("boost", "tree"),
         ("rf_weighted", "tree_balanced"), ("rf_sampled", "tree_balanced"), ("rf", "knn_cw"),
         ("rf_weighted", "knn_cw"), ("rf_sampled", "knn_cw"), ("boost", "knn_cw"), ("tree", "knn_cw"),
         ("rf", "rf_weighted"), ("rf", "rf_sampled"), ("rf_weighted", "rf_sampled"), ("rf", "boost"),
         ("rf_weighted", "boost"), ("rf_sampled", "boost"),
         ("vote_all", "rf_weighted"), ("vote_all", "rf_sampled"), ("vote_forests", "rf_weighted"),
         ("vote_forests", "rf_sampled"), ("stack", "rf_weighted"), ("stack", "rf_sampled"),
         ("vote_all", "vote_forests"), ("vote_without_knn_cw", "vote_all"), ("vote_without_tree", "vote_all"),
         ("vote_without_rf", "vote_all"), ("vote_without_rf_weighted", "vote_all")]
BOOTSTRAPS = 1000
KEY_PAIRS_G = [("rf", "tree"), ("rf_weighted", "tree"), ("rf_sampled", "tree"), ("boost", "tree"),
               ("rf_weighted", "rf"), ("rf_sampled", "rf_weighted"), ("boost", "rf"), ("rf", "knn_cw"),
               ("vote_all", "rf_weighted"), ("stack", "rf_weighted")]


def per_partition(runs: pd.DataFrame):
    num = [c for c in runs.columns if c not in ("model", "repeat", "fold", "run", "protocol", "encoding")]
    mean = runs.groupby(["model", "repeat", "fold"])[num].mean().reset_index()
    sd = runs.groupby(["model", "repeat", "fold"])[MEASURES].std(ddof=1).reset_index()
    return mean, sd


def summary(mean: pd.DataFrame, sd: pd.DataFrame, runs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for m, sub in mean.groupby("model", sort=False):
        sub = sub.sort_values(["repeat", "fold"])
        row = {"model": m, "n_partitions": len(sub), "n_runs": int(runs[runs.model == m].groupby(["repeat", "fold"]).size().max())}
        for k in MEASURES:
            row[f"{k}_mean"], row[f"{k}_std"] = float(sub[k].mean()), float(sub[k].std(ddof=1))
            s = sd[sd.model == m][k]
            row[f"{k}_run_std"] = float(s.mean()) if s.notna().any() else 0.0
        for c in ("accuracy_none", "accuracy_consistent", "accuracy_conflicting", "n_none", "n_consistent",
                  "n_conflicting", "member_accuracy", "disagreement", "q_statistic", "oob_macro_f1", "oob_accuracy",
                  "oob_balanced_accuracy", "seconds"):
            if c in sub.columns and sub[c].notna().any():
                row[f"{c}_mean"] = float(sub[c].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def paired_stats(mean: pd.DataFrame) -> pd.DataFrame:
    piv = {k: mean.pivot_table(index=["repeat", "fold"], columns="model", values=k) for k in MEASURES}
    cols = set(piv["macro_f1"].columns)
    rows = []
    for a, b in PAIRS:
        if a not in cols or b not in cols:
            continue
        for k in MEASURES:
            x, y = piv[k][a].to_numpy(), piv[k][b].to_numpy()
            w = stats.paired_wilcoxon(x, y)
            c = stats.corrected_resampled_ttest(x, y)
            rows.append({"model_a": a, "model_b": b, "measure": k, "mean_a": float(x.mean()), "mean_b": float(y.mean()),
                         "mean_diff": w["mean_diff"], "cohens_dz": w["cohens_dz"], "wilcoxon_p": w["p_value"],
                         "corrected_t_p": c["p_value"], "n_a_higher": w["n_a_higher"], "n_b_higher": w["n_b_higher"]})
    out = pd.DataFrame(rows)
    for col in ("wilcoxon_p", "corrected_t_p"):
        out[col + "_holm"] = np.nan
        for k in MEASURES:
            mask = out["measure"] == k
            out.loc[mask, col + "_holm"] = stats.holm_adjust(out.loc[mask, col].to_numpy())
    return out


def per_class(prefix: str, protocol_name: str, models: list[str], pattern) -> pd.DataFrame:
    rows = []
    folder = RESULTS / f"confusion_{protocol_name}"
    for m in models:
        files = sorted(folder.glob(pattern(m)))
        if not files:
            continue
        total = sum(np.loadtxt(f, delimiter=",", dtype=np.int64) for f in files)
        pc = metrics.per_class(total)
        for j in range(N_CLASSES):
            rows.append({"model": m, "class": j, "name": data.CLASS_NAMES[j], "precision": pc["precision"][j],
                         "recall": pc["recall"][j], "f1": pc["f1"][j], "support": int(pc["support"][j])})
    return pd.DataFrame(rows)


def bootstrap_generalisation(name: str, y_gen: np.ndarray, rng: np.random.Generator) -> tuple[pd.DataFrame, pd.DataFrame]:
    pred_dir = RESULTS / f"predG_{name}"
    runs = {}
    for f in sorted(pred_dir.glob("*_run*.npy")):
        model, run = f.stem.rsplit("_run", 1)
        runs.setdefault(model, []).append(np.load(f).astype(np.int64))
    n = y_gen.size
    idx = [rng.integers(0, n, size=n) for _ in range(BOOTSTRAPS)]
    point, boots = {}, {}
    for model, preds in runs.items():
        P = np.stack(preds)
        point[model] = {k: float(np.mean([metrics.score_all(y_gen, p, N_CLASSES)[k] for p in P])) for k in MEASURES}
        b = {k: np.empty(BOOTSTRAPS) for k in MEASURES}
        for t, ix in enumerate(idx):
            vals = [metrics.score_all(y_gen[ix], p[ix], N_CLASSES) for p in P[:min(len(P), 10)]]
            for k in MEASURES:
                b[k][t] = np.mean([v[k] for v in vals])
        boots[model] = b
    rows = []
    for model in runs:
        row = {"model": model, "n_runs": len(runs[model])}
        for k in MEASURES:
            lo, hi = np.percentile(boots[model][k], [2.5, 97.5])
            P = np.stack(runs[model])
            per_run = [metrics.score_all(y_gen, p, N_CLASSES)[k] for p in P]
            row[f"{k}"] = point[model][k]
            row[f"{k}_run_std"] = float(np.std(per_run, ddof=1)) if len(per_run) > 1 else 0.0
            row[f"{k}_lo"], row[f"{k}_hi"] = float(lo), float(hi)
        rows.append(row)
    diffs = []
    for a, b in KEY_PAIRS_G:
        if a in boots and b in boots:
            for k in MEASURES:
                d = boots[a][k] - boots[b][k]
                lo, hi = np.percentile(d, [2.5, 97.5])
                diffs.append({"model_a": a, "model_b": b, "measure": k, "diff": point[a][k] - point[b][k],
                              "lo": float(lo), "hi": float(hi)})
    return pd.DataFrame(rows), pd.DataFrame(diffs)


def main() -> None:
    summaries = {}
    for name in ("compatible", "grouped"):
        path = RESULTS / f"protocol_{name}.csv"
        if not path.exists():
            continue
        runs = pd.read_csv(path)
        mean, sd = per_partition(runs)
        mean.to_csv(RESULTS / f"protocol_scores_{name}.csv", index=False)
        s = summary(mean, sd, runs)
        s.to_csv(RESULTS / f"protocol_summary_{name}.csv", index=False)
        paired_stats(mean).to_csv(RESULTS / f"protocol_stats_{name}.csv", index=False)
        pc = per_class("cv", name, SINGLE + ENSEMBLES + DISSIMILAR, lambda m: f"{m}_[0-9]_*.csv")
        pc.to_csv(RESULTS / f"per_class_{name}.csv", index=False)
        summaries[name] = s.set_index("model")
    if len(summaries) == 2:
        rows = []
        models = [m for m in SINGLE + ENSEMBLES + DISSIMILAR if m in summaries["compatible"].index and m in summaries["grouped"].index]
        for k in MEASURES:
            a = summaries["compatible"].loc[models, f"{k}_mean"].to_numpy()
            b = summaries["grouped"].loc[models, f"{k}_mean"].to_numpy()
            tau, p = sps.kendalltau(a, b)
            rows.append({"measure": k, "kendall_tau_all": float(tau), "p_value": float(p), "n_models": len(models)})
            base = [m for m in SINGLE + ENSEMBLES if m in models]
            a2 = summaries["compatible"].loc[base, f"{k}_mean"].to_numpy()
            b2 = summaries["grouped"].loc[base, f"{k}_mean"].to_numpy()
            tau2, p2 = sps.kendalltau(a2, b2)
            rows[-1].update({"kendall_tau_base": float(tau2), "p_value_base": float(p2), "n_base": len(base),
                             "order_compatible": " > ".join(np.array(base)[np.argsort(-a2)]),
                             "order_grouped": " > ".join(np.array(base)[np.argsort(-b2)])})
        pd.DataFrame(rows).to_csv(RESULTS / "protocol_ranking.csv", index=False)
    table = data.load_table()
    for name in ("compatible", "grouped"):
        path = RESULTS / f"generalisation_{name}.csv"
        if not path.exists() or not (RESULTS / f"predG_{name}").is_dir():
            continue
        if name == "compatible":
            _, gen_idx = protocol.stratified_holdout(table.y)
        else:
            gen_idx = grouped.grouped_split(table).gen_idx
        y_gen = table.y[gen_idx]
        rng = protocol.rng(f"opt1/bootstrap-G/{name}")
        pt, df = bootstrap_generalisation(name, y_gen, rng)
        pt.to_csv(RESULTS / f"generalisation_bootstrap_{name}.csv", index=False)
        df.to_csv(RESULTS / f"generalisation_differences_{name}.csv", index=False)
        g = pd.read_csv(path)
        g.groupby("model").agg({"fit_seconds": "mean", "predict_seconds": "mean"}).reset_index().to_csv(
            RESULTS / f"generalisation_time_{name}.csv", index=False)
    print("done")


if __name__ == "__main__":
    main()
