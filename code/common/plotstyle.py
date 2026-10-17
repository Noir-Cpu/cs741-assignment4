"""Matplotlib style that matches the IEEEtran body text.

Serif font Nimbus Roman (or TeX Gyre Termes), 10 pt everywhere, STIX
mathematics, vector PDF with embedded fonts, no figure titles (captions are
set in LaTeX), a colour-blind-safe palette (Okabe and Ito) and the IEEE
column widths: 3.5 in for one column, 7.16 in for two columns.

Usage
-----
    from common import plotstyle
    plotstyle.apply()
    fig, ax = plotstyle.figure("single", height=2.4)
    ax.plot(x, y, color=plotstyle.colour("tree"))
    plotstyle.save(fig, "figures/common/example.pdf")
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

SINGLE_WIDTH = 3.5
DOUBLE_WIDTH = 7.16
FONT_SIZE = 10
FONT_CANDIDATES = ("Nimbus Roman", "TeX Gyre Termes")

# Okabe and Ito palette, safe for the common forms of colour blindness.
PALETTE = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9",
           "#F0E442", "#000000")
MODEL_COLOURS = {"knn": PALETTE[0], "knn_cw": PALETTE[5], "tree": PALETTE[1],
                 "ensemble": PALETTE[2]}


def available_font() -> str:
    """First installed font of FONT_CANDIDATES, else the generic serif family."""
    installed = {f.name for f in font_manager.fontManager.ttflist}
    for name in FONT_CANDIDATES:
        if name in installed:
            return name
    return "serif"


def apply() -> str:
    """Install the style in matplotlib's rcParams and return the font name."""
    font = available_font()
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": [font, "serif"],
        "font.size": FONT_SIZE,
        "axes.labelsize": FONT_SIZE,
        "axes.titlesize": FONT_SIZE,
        "xtick.labelsize": FONT_SIZE,
        "ytick.labelsize": FONT_SIZE,
        "legend.fontsize": FONT_SIZE,
        "figure.titlesize": FONT_SIZE,
        "mathtext.fontset": "stix",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.6,
        "lines.linewidth": 1.2,
        "legend.frameon": False,
        "axes.prop_cycle": matplotlib.cycler(color=list(PALETTE)),
        "figure.dpi": 150,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })
    return font


def colour(name: str) -> str:
    """Colour for a model name, or the palette entry by position for an unknown name."""
    if name in MODEL_COLOURS:
        return MODEL_COLOURS[name]
    return PALETTE[sum(map(ord, name)) % len(PALETTE)]


def figure(width: str = "single", height: float | None = None, **subplots_kw):
    """Figure and axes with the IEEE single (3.5 in) or double (7.16 in) column width."""
    w = SINGLE_WIDTH if width == "single" else DOUBLE_WIDTH
    h = height if height is not None else (0.72 * w if width == "single" else 0.38 * w)
    return plt.subplots(figsize=(w, h), **subplots_kw)


def save(fig, path) -> Path:
    """Save as vector PDF (creating the directory) and close the figure."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="pdf")
    plt.close(fig)
    return path
