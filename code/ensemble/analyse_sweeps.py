"""Summaries of the tuning experiments (results/ensemble/*.csv -> results/ensemble/summary_*.csv).

Every summary reports the mean and the standard deviation over the units of work of an
experiment (five tuning folds times the runs) of each numeric column. The out-of-bag
analysis pools the configurations of the feature, structure, treatment, search and size
experiments at 200 trees.

Usage: .venv/bin/python code/ensemble/analyse_sweeps.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ensemble.harness import CLASS_COLUMNS, RESULTS  # noqa: E402

KEYS = ["accuracy", "macro_f1", "weighted_f1", "balanced_accuracy", "oob_accuracy", "oob_macro_f1",
        "oob_balanced_accuracy", "oob_coverage", "member_accuracy", "disagreement", "q_statistic",
        "error_correlation", "mean_nodes", "mean_depth", "fit_seconds"] + CLASS_COLUMNS


def summarise(df: pd.DataFrame, by: list[str], cols: list[str] | None = None) -> pd.DataFrame:
    cols = [c for c in (cols or KEYS) if c in df.columns]
    g = df.groupby(by, sort=False)[cols].agg(["mean", "std"])
    g.columns = [f"{a}_{b}" for a, b in g.columns]
    g["n_units"] = df.groupby(by, sort=False).size().to_numpy()
    return g.reset_index()


def load(name: str) -> pd.DataFrame | None:
    path = RESULTS / f"{name}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, keep_default_na=False, na_values=["", "nan", "NaN"])
    for col in ("max_depth", "class_sample_size"):      # the text None means no limit and no undersampling size
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].replace("None", -1))
    return df


def paired_effect(df: pd.DataFrame, a: str, b: str, col: str = "macro_f1") -> dict:
    """Mean difference and paired effect size of two configurations over the matched units."""
    x = df[df["config"] == a].set_index(["fold", "run"])[col]
    y = df[df["config"] == b].set_index(["fold", "run"])[col]
    d = (x - y).dropna()
    sd = d.std(ddof=1)
    return {"a": a, "b": b, "measure": col, "mean_diff": float(d.mean()),
            "paired_effect_size": float(d.mean() / sd) if sd > 0 else float("nan"), "n_units": int(d.size)}


def oob_analysis(frames: dict[str, pd.DataFrame]) -> dict:
    rows = []
    for name, df in frames.items():
        target = 100 if name == "search1" else 200
        sub = df[df["n_estimators"] == target]
        g = sub.groupby("config")[["macro_f1", "oob_macro_f1", "accuracy", "oob_accuracy",
                                   "balanced_accuracy", "oob_balanced_accuracy"]].mean().reset_index()
        g["experiment"] = name
        rows.append(g)
    pool = pd.concat(rows, ignore_index=True)
    pool.to_csv(RESULTS / "oob_pool.csv", index=False)
    out = {"n_configurations": int(len(pool))}
    good = pool[pool["macro_f1"] > 0.5]
    out["n_above_half"] = int(len(good))
    for m in ("macro_f1", "accuracy", "balanced_accuracy"):
        rho, _ = sps.spearmanr(pool[m], pool["oob_" + m])
        out[f"spearman_{m}"] = float(rho)
        out[f"mean_diff_oob_minus_test_{m}"] = float((pool["oob_" + m] - pool[m]).mean())
        out[f"sd_diff_oob_minus_test_{m}"] = float((pool["oob_" + m] - pool[m]).std(ddof=1))
        rho, _ = sps.spearmanr(good[m], good["oob_" + m])
        out[f"spearman_above_half_{m}"] = float(rho)
    plain = good[~good["experiment"].isin(["balance"])]
    out["spearman_above_half_without_treatment_experiment"] = float(sps.spearmanr(plain["macro_f1"], plain["oob_macro_f1"])[0])
    bal = pool[pool["experiment"] == "balance"]
    if len(bal) > 2:
        out["spearman_treatment_experiment"] = float(sps.spearmanr(bal["macro_f1"], bal["oob_macro_f1"])[0])
        out["mean_diff_oob_minus_test_macro_f1_treatment"] = float((bal["oob_macro_f1"] - bal["macro_f1"]).mean())
    out["spearman_by_experiment"] = {k: float(sps.spearmanr(g["macro_f1"], g["oob_macro_f1"])[0])
                                     for k, g in pool.groupby("experiment") if len(g) > 2}
    for name in ("search1", "search2"):
        df = frames.get(name)
        if df is None:
            continue
        g = df.groupby("config")[["macro_f1", "oob_macro_f1"]].mean()
        by_cv, by_oob = g["macro_f1"].idxmax(), g["oob_macro_f1"].idxmax()
        out[f"{name}_best_by_cv"] = {"config": by_cv, "macro_f1": float(g.loc[by_cv, "macro_f1"])}
        out[f"{name}_best_by_oob"] = {"config": by_oob, "macro_f1": float(g.loc[by_oob, "macro_f1"])}
        out[f"{name}_regret_of_oob_selection"] = float(g.loc[by_cv, "macro_f1"] - g.loc[by_oob, "macro_f1"])
        out[f"{name}_rank_of_oob_choice_by_cv"] = int((g["macro_f1"] > g.loc[by_oob, "macro_f1"]).sum() + 1)
        out[f"{name}_n"] = int(len(g))
        out[f"{name}_share_within_001_of_best_by_oob_rank"] = float(
            (g["macro_f1"][g["oob_macro_f1"].sort_values(ascending=False).index[:5]] >= g["macro_f1"].max() - 0.01).mean())
    s2 = frames.get("search2")
    if s2 is not None and "family" in s2.columns:
        fam = {}
        g = s2.groupby(["family", "config"])[["macro_f1", "oob_macro_f1"]].mean().reset_index()
        for f, sub in g.groupby("family"):
            cv, ob = sub.loc[sub["macro_f1"].idxmax()], sub.loc[sub["oob_macro_f1"].idxmax()]
            fam[f] = {"n": int(len(sub)), "same": bool(cv["config"] == ob["config"]),
                      "regret": float(cv["macro_f1"] - ob["macro_f1"])}
        out["search2_by_family"] = fam
    return out


def diminishing_returns(df: pd.DataFrame, config: str, tol: float = 0.005, col: str = "macro_f1") -> dict:
    sub = df[df["config"] == config]
    m = sub.groupby("n_estimators")[col].mean()
    final = m.iloc[-1]
    ok = m[m >= final - tol]
    first = int(ok.index[0])
    return {"config": config, "measure": col, "final_members": int(m.index[-1]), "final_score": float(final),
            "diminishing_at": first, "score_at_diminishing": float(m.loc[first])}


def main() -> None:
    frames = {}
    for name, file in (("features", "features"), ("structure", "structure"), ("balance", "balance"),
                       ("search1", "joint_stage1"), ("search2", "joint_stage2"), ("size_base", "size_base"),
                       ("size_plain", "size_plain"), ("size_weighted", "size_weighted"), ("size_sampled", "size_sampled")):
        df = load(file)
        if df is not None:
            frames[name] = df
    summaries = {}
    enc = load("encoding")
    if enc is not None:
        summaries["encoding"] = summarise(enc, ["config"])
    for name, df in frames.items():
        if name.startswith("size"):
            summaries[name] = summarise(df, ["config", "n_estimators"])
        elif name == "structure":
            summaries[name] = summarise(df, ["config", "max_depth", "min_samples_leaf"])
        elif name == "features":
            summaries[name] = summarise(df, ["config", "max_features"])
        elif name == "balance":
            summaries[name] = summarise(df, ["config", "sampling", "sampling_power", "class_sample_size",
                                             "class_weight_power"])
        else:
            summaries[name] = summarise(df, ["config"])
    bsize = load("boosting_size")
    if bsize is not None:
        summaries["boosting_size"] = summarise(
            bsize, ["n_estimators"], KEYS + ["train_error", "mean_alpha", "mean_weighted_error"])
    for name, s in summaries.items():
        s.to_csv(RESULTS / f"summary_{name}.csv", index=False)
    extra = {}
    for name in ("size_plain", "size_weighted", "size_sampled", "size_base"):
        if name in frames:
            c = frames[name]["config"].iloc[0]
            extra[f"diminishing_{name}"] = diminishing_returns(frames[name], c)
            extra[f"diminishing_{name}_oob"] = diminishing_returns(frames[name], c, col="oob_macro_f1")
    if bsize is not None:
        b = bsize.assign(config="boost")
        m = b.groupby("n_estimators")["macro_f1"].mean()
        extra["boosting_peak"] = {"round": int(m.idxmax()), "macro_f1": float(m.max()),
                                  "final_round": int(m.index[-1]), "final": float(m.iloc[-1])}
        extra["diminishing_boosting"] = diminishing_returns(b, "boost")
    if len(frames) > 1:
        extra["oob"] = oob_analysis({k: v for k, v in frames.items() if not k.startswith("size_") or k == "size_base"})
    if "balance" in frames:
        b = frames["balance"]
        base = "bootstrap0.0,alpha=0.0"
        extra["treatment_effects"] = [paired_effect(b, c, base) for c in b["config"].unique() if c != base]
    if "features" in frames:
        f = frames["features"]
        extra["feature_effects"] = [paired_effect(f, c, "m=8") for c in f["config"].unique()
                                    if c != "m=8" and "m=8" in set(f["config"])]
    (RESULTS / "analysis_sweeps.json").write_text(json.dumps(extra, indent=2, default=str))
    print(json.dumps({k: v for k, v in extra.items() if k not in ("treatment_effects", "feature_effects")},
                     indent=2, default=str)[:2500])


if __name__ == "__main__":
    main()
