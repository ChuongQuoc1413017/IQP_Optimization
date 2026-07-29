import os, json
import numpy as np
import matplotlib.pyplot as plt

from iqp.experiments.depth import LAYERS, N, MODE, INIT, HAM

_HERE = os.path.dirname(os.path.abspath(__file__))
DEPTH = os.path.join(os.path.dirname(_HERE), "results", "depth")

COLOR = "#1f77b4"  


def _load(metric):
    means, sems = [], []
    for L in LAYERS:
        d = json.load(open(os.path.join(DEPTH, f"{metric}__L{L}.json")))
        d = np.asarray(d)
        means.append(d.mean())
        sems.append(d.std(ddof=1) / np.sqrt(len(d)))
    return np.array(means), np.array(sems)


r_mean, r_sem = _load("ratio")
g_mean, g_sem = _load("gv")

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

ax = axes[0]
ax.plot(LAYERS, r_mean, "o-", color=COLOR, linewidth=2, markersize=6)
ax.fill_between(LAYERS, r_mean - r_sem, r_mean + r_sem, color=COLOR, alpha=0.2, linewidth=0)
ax.set_xlabel("Number of layers $L$")
ax.set_ylabel(r"Relative approx. ratio $r_{RA}$")
ax.set_title("Optimization performance")
ax.grid(True, alpha=0.3, linestyle="--")
ax.set_xticks(LAYERS)

ax = axes[1]
ax.plot(LAYERS, g_mean, "o-", color=COLOR, linewidth=2, markersize=6)
ax.fill_between(LAYERS, g_mean - g_sem, g_mean + g_sem, color=COLOR, alpha=0.2, linewidth=0)
ax.set_xlabel("Number of layers $L$")
ax.set_ylabel(r"Gradient variance $\mathrm{Var}[\partial\mathcal{C}/\partial\theta_k]$")
ax.set_title("Trainability")
ax.set_yscale("log")
ax.grid(True, alpha=0.3, linestyle="--")
ax.set_xticks(LAYERS)

fig.suptitle(rf"IQP full connectivity, Ising, $n={N}$, $\mathcal{{N}}(0,1)$ init "
             f"({len(json.load(open(os.path.join(DEPTH, f'ratio__L{LAYERS[0]}.json'))))} instances)",
             fontsize=12)
plt.tight_layout()
out = os.path.join(DEPTH, "Depth")
plt.savefig(out + ".pdf", dpi=300, bbox_inches="tight")
plt.savefig(out + ".png", dpi=150, bbox_inches="tight")
print("saved", out + ".pdf")
