import os, json
import numpy as np
import matplotlib.pyplot as plt

from iqp.experiments.depth import LAYERS
from iqp.experiments.optim import QUBITS
from iqp.plots.style import apply_style, ANSATZ, OPTIM, mean_sem, panel_label, FULL_WIDTH

GV_INSET = False

_HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(os.path.dirname(_HERE), "results")
OUT_DIR = os.path.join(RESULTS, "robust")
FULL_COLOR = "#1f77b4"   


def _depth(metric):
    stats = [mean_sem(json.load(open(os.path.join(RESULTS, "depth", f"{metric}__L{L}.json"))))
             for L in LAYERS]
    return np.array([m for m, _ in stats]), np.array([s for _, s in stats])


def _optim(opt):
    paths = [os.path.join(RESULTS, "ratio", f"normal__ising__full__n{n}.json") if opt == "adam"
             else os.path.join(RESULTS, "optim", f"{opt}__n{n}.json") for n in QUBITS]
    stats = [mean_sem(json.load(open(p))) for p in paths]
    return np.array([m for m, _ in stats]), np.array([s for _, s in stats])


apply_style()
fig, axes = plt.subplots(1, 2, figsize=(FULL_WIDTH, 2.75))

# (a) depth
ax = axes[0]
m, s = _depth("ratio")
ax.plot(LAYERS, m, "o-", color=FULL_COLOR, label="Full connectivity")
ax.fill_between(LAYERS, m - s, m + s, color=FULL_COLOR, alpha=0.2, linewidth=0)

ref, _ = mean_sem(json.load(open(os.path.join(
    RESULTS, "ratio", "normal__ising__circular__n12.json"))))
ax.axhline(ref, color=ANSATZ["circular"][1], linestyle="--", linewidth=1.2,
           label="Circular connectivity, $L=1$")
ax.set_ylim(0.0, None)
ax.set_xlabel("Number of IQP blocks $L$")
ax.set_xticks(LAYERS)
ax.legend(loc="lower left", frameon=False, handlelength=1.8, borderaxespad=0.3)
panel_label(ax, "(a)")

if GV_INSET:
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + 0.55 * (hi - lo))   
    inset = ax.inset_axes([0.52, 0.60, 0.45, 0.36])
    gm, gs = _depth("gv")
    inset.plot(LAYERS, gm, "o-", color=FULL_COLOR, linewidth=1.2, markersize=3)
    inset.fill_between(LAYERS, gm - gs, gm + gs, color=FULL_COLOR, alpha=0.2, linewidth=0)
    inset.set_yscale("log")
    inset.set_ylabel(r"$\mathrm{Var}[\partial\mathcal{C}/\partial\theta_k]$", fontsize=6)
    inset.set_xticks([1, 4, 10])
    inset.tick_params(labelsize=6)

# (b) optimizer
ax = axes[1]
for opt, (fmt, color, label) in OPTIM.items():
    m, s = _optim(opt)
    ax.plot(QUBITS, m, fmt, color=color, label=label)
    ax.fill_between(QUBITS, m - s, m + s, color=color, alpha=0.18, linewidth=0)
ax.set_xlabel("Number of qubits $n$")
ax.set_xticks(QUBITS)
ax.set_ylim(0.0, None)
ax.legend(loc="upper left", ncol=2, frameon=False, handlelength=1.8,
          columnspacing=1.0, borderaxespad=0.3)
panel_label(ax, "(b)")

fig.supylabel(r"Relative approx. ratio $r_{RA}$", fontsize=9)

fig.tight_layout(w_pad=1.5)
os.makedirs(OUT_DIR, exist_ok=True)
out = os.path.join(OUT_DIR, "IQP_robust")
fig.savefig(out + ".pdf", bbox_inches="tight")
fig.savefig(out + ".png", dpi=300, bbox_inches="tight")
print("saved", out + ".pdf")
