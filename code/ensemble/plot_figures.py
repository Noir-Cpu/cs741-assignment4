"""Figures of the report, drawn from results/ensemble into figures/ensemble (vector PDF).

Usage: .venv/bin/python code/ensemble/plot_figures.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import plotstyle  # noqa: E402
from ensemble.harness import FIGURES, RESULTS  # noqa: E402

C = {"forest": plotstyle.PALETTE[0], "oob": plotstyle.PALETTE[5], "boost": plotstyle.PALETTE[1],
     "tuned": plotstyle.PALETTE[2], "dis": plotstyle.PALETTE[3]}


def band(ax, x, mean, sd, colour, **kw):
    ax.plot(x, mean, color=colour, **kw)
    ax.fill_between(x, mean - sd, mean + sd, color=colour, alpha=0.15, linewidth=0)


def fig_size() -> None:
    fig, ax = plotstyle.figure("single", height=2.4)
    styles = {"plain": (C["forest"], "-"), "weighted": (C["tuned"], "-"), "sampled": (C["dis"], "-")}
    for c, (col, ls) in styles.items():
        s = pd.read_csv(RESULTS / f"summary_size_{c}.csv")
        x = s["n_estimators"].to_numpy()
        band(ax, x, s["macro_f1_mean"], s["macro_f1_std"], col, linestyle=ls, label=f"{c} forest")
        if c in ("plain", "sampled"):
            ax.plot(x, s["oob_macro_f1_mean"], color=col, linestyle=":", label=f"{c}, out of bag")
    b = pd.read_csv(RESULTS / "summary_boosting_size.csv")
    band(ax, b["n_estimators"].to_numpy(), b["macro_f1_mean"], b["macro_f1_std"], C["boost"], linestyle="--",
         label="boosting")
    ax.set_xscale("log")
    ax.set_xlabel("members")
    ax.set_ylabel("macro-F1")
    ax.set_ylim(0.50, 0.65)
    ax.legend(loc="lower center", bbox_to_anchor=(0.45, 1.0), ncol=2, handlelength=1.5, columnspacing=0.8, borderaxespad=0.2)
    plotstyle.save(fig, FIGURES / "size.pdf")


def fig_features() -> None:
    s = pd.read_csv(RESULTS / "summary_features.csv")
    x = s["max_features"].to_numpy()
    panels = [("f1", "macro-F1", ["macro_f1", "oob_macro_f1"], ["test", "out of bag"]),
              ("strength", r"strength $\mathcal{S}$", ["member_accuracy"], [None]),
              ("disagreement", "disagreement $\\Theta$", ["disagreement"], [None])]
    for name, label, cols, labs in panels:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(2.05, 1.9))
        for i, (c, l) in enumerate(zip(cols, labs)):
            colour = [C["forest"], C["oob"]][i] if name == "f1" else C["forest"]
            band(ax, x, s[f"{c}_mean"], s[f"{c}_std"], colour, linestyle="-" if i == 0 else "--", label=l,
                 marker="o", markersize=3)
        ax.set_xscale("log", base=2)
        ax.set_xticks([1, 4, 16, 79])
        ax.set_xticklabels(["1", "4", "16", "79"])
        ax.minorticks_off()
        ax.set_xlabel("features per split $\\varphi$")
        ax.set_ylabel(label)
        if name == "f1":
            ax.legend(loc="lower right")
        plotstyle.save(fig, FIGURES / f"features_{name}.pdf")


def fig_structure() -> None:
    s = pd.read_csv(RESULTS / "summary_structure.csv")
    depths = [3, 5, 8, 12, 20, -1]
    leaves = [1, 2, 5, 10, 25, 50]
    grid = np.array([[s[(s.max_depth == d) & (s.min_samples_leaf == l)]["macro_f1_mean"].iloc[0] for l in leaves]
                     for d in depths])
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    im = ax.imshow(grid, origin="lower", cmap="Blues", aspect="auto", vmin=0.2, vmax=0.62)
    ax.set_xticks(range(len(leaves)))
    ax.set_xticklabels(leaves)
    ax.set_yticks(range(len(depths)))
    ax.set_yticklabels([str(d) if d > 0 else "none" for d in depths])
    ax.set_xlabel("minimum leaf size")
    ax.set_ylabel("depth limit")
    for i in range(len(depths)):
        for j in range(len(leaves)):
            ax.text(j, i, f"{grid[i, j]:.2f}", ha="center", va="center",
                    color="white" if grid[i, j] > 0.45 else "black")
    for sp in ax.spines.values():
        sp.set_visible(True)
    plotstyle.save(fig, FIGURES / "structure.pdf")


def fig_oob() -> None:
    pool = pd.read_csv(RESULTS / "oob_pool.csv")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    names = {"features": "feature sweep", "structure": "structure sweep", "balance": "treatment sweep",
             "search1": "search, stage one", "search2": "search, stage two", "size_base": "size curve"}
    marks = {"features": "o", "structure": "s", "balance": "^", "search1": "D", "search2": "P", "size_base": "v"}
    for i, (k, g) in enumerate(pool.groupby("experiment")):
        ax.scatter(g["macro_f1"], g["oob_macro_f1"], s=12, marker=marks.get(k, "o"), label=names.get(k, k),
                   color=plotstyle.PALETTE[i % 8])
    lo, hi = 0.2, 0.68
    ax.plot([lo, hi], [lo, hi], color="black", linewidth=0.6)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("test macro-F1")
    ax.set_ylabel("out-of-bag macro-F1")
    ax.legend(loc="lower center", bbox_to_anchor=(0.45, 1.0), ncol=2, handletextpad=0.2, columnspacing=0.6, borderaxespad=0.2)
    plotstyle.save(fig, FIGURES / "oob.pdf")


def fig_confusion() -> None:
    import matplotlib.pyplot as plt
    from common import data, metrics, protocol
    table = data.load_table()
    _, gen_idx = protocol.stratified_holdout(table.y)
    y = table.y[gen_idx]
    files = sorted((RESULTS / "predG_compatible").glob("rf_weighted_run*.npy"))
    if not files:
        return
    cm = sum(metrics.confusion_matrix(y, np.load(f).astype(np.int64), 10) for f in files).astype(float)
    cm = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(3.5, 3.1))
    ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    names = ["Norm.", "Rec.", "Back.", "DoS", "Expl.", "Anal.", "Fuzz.", "Worms", "Shell.", "Gen."]
    ax.set_xticks(range(10))
    ax.set_xticklabels(names, rotation=90)
    ax.set_yticks(range(10))
    ax.set_yticklabels(names)
    ax.set_xlabel("predicted class")
    ax.set_ylabel("true class")
    for i in range(10):
        for j in range(10):
            if cm[i, j] >= 0.05:
                ax.text(j, i, f"{100 * cm[i, j]:.0f}", ha="center", va="center", fontsize=10,
                        color="white" if cm[i, j] > 0.5 else "black")
    for sp in ax.spines.values():
        sp.set_visible(True)
    plotstyle.save(fig, FIGURES / "confusion.pdf")


def main() -> None:
    plotstyle.apply()
    fig_size()
    fig_features()
    fig_structure()
    fig_oob()
    fig_confusion()


if __name__ == "__main__":
    main()
