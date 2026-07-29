
import matplotlib.pyplot as plt
import numpy as np

FULL_WIDTH = 7.16   # \textwidth of a two-column IEEEtran page, in inches
COL_WIDTH = 3.45    # \columnwidth


def apply_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],   # IEEEtran body font
        "mathtext.fontset": "stix",                          # Times-matching math
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "lines.linewidth": 1.6,
        "lines.markersize": 4.5,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
        "grid.linewidth": 0.6,
        "figure.dpi": 110,
    })


# Circuit families -- unchanged from plots/ratio.py, gd.py, expr_ent.py.
ANSATZ = {
    "full":     ("o-",  "#1f77b4", "Full Connectivity"),
    "circular": ("s-",  "#ff7f0e", "Circular Connectivity"),
    "single":   ("^--", "#2ca02c", "Single-Z Terms"),
    "hea":      ("D-",  "#d62728", "HEA (L=2)"),
    "qaoa":     ("v-",  "#9467bd", "QAOA (p=2)"),
}

OPTIM = {
    "adam":   ("o-",  "#1f77b4", "Adam"),
    "cobyla": ("D-",  "#d55e00", "COBYLA"),
    "spsa":   ("^-",  "#009e73", "SPSA"),
    "qng":    ("s-",  "#cc79a7", "QNG"),
    "sgd":    ("v--", "#4d4d4d", "SGD"),
}


def mean_sem(values):
    """Mean and standard error of the mean of one list of per-instance values."""
    a = np.asarray(values, dtype=float)
    return a.mean(), a.std(ddof=1) / np.sqrt(a.size)


def panel_label(ax, text):
    ax.text(0.0, 1.06, text, transform=ax.transAxes, fontsize=10, va="bottom")
