"""Shared typography and output settings for one-column TP1 paper figures."""

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
BLUE = "#0072B2"
VERMILION = "#D55E00"
GRAY = "#737373"
TIMES = "\N{MULTIPLICATION SIGN}"


def table(filename):
    with (HERE / filename).open() as source:
        return list(csv.DictReader(source))


def axes():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8,
        "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "legend.fontsize": 7, "axes.linewidth": 0.65, "lines.linewidth": 1.2,
        "lines.markersize": 3.8, "pdf.fonttype": 42, "ps.fonttype": 42,
        "hatch.linewidth": 0.45, "savefig.dpi": 300,
    })
    fig, ax = plt.subplots(figsize=(3.35, 2.3))
    fig.subplots_adjust(left=0.17, right=0.98, bottom=0.20, top=0.96)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(width=0.65, length=3, pad=2)
    ax.set_axisbelow(True)
    return fig, ax


def save(fig, stem):
    # Keep a fixed physical page size instead of bbox_inches='tight'.
    fig.savefig(HERE / f"{stem}.pdf")
    fig.savefig(HERE / f"{stem}.png", dpi=300)
    plt.close(fig)
