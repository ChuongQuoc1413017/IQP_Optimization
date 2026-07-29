import os, json
import numpy as np
import matplotlib.pyplot as plt

from iqp.experiments.optim import QUBITS

_HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(os.path.dirname(_HERE), "results")

STYLES = {
    "adam":   ("o-", "#1f77b4", "Adam"),
    "qng":    ("s-", "#ff7f0e", "QNG"),
    "spsa":   ("^-", "#2ca02c", "SPSA"),
    "cobyla": ("D-", "#d62728", "COBYLA"),
    "sgd":    ("v--", "#9467bd", "SGD"),
}


def _path(opt, n):
    if opt == "adam":
        return os.path.join(RESULTS, "ratio", f"normal__ising__full__n{n}.json")
    return os.path.join(RESULTS, "optim", f"{opt}__n{n}.json")


def _curve(opt):
    means, sems = [], []
    for n in QUBITS:
        d = np.asarray(json.load(open(_path(opt, n))))
        means.append(d.mean())
        sems.append(d.std(ddof=1) / np.sqrt(len(d)))
    return np.array(means), np.array(sems)


fig, ax = plt.subplots(figsize=(6.5, 4.6))
for opt, (fmt, color, label) in STYLES.items():
    m, s = _curve(opt)
    ax.plot(QUBITS, m, fmt, color=color, label=label, linewidth=2, markersize=6)
    ax.fill_between(QUBITS, m - s, m + s, color=color, alpha=0.18, linewidth=0)

ax.set_xlabel("Number of qubits $n$")
ax.set_ylabel(r"Relative approx. ratio $r_{RA}$")
ax.set_title(r"IQP full connectivity, Ising, $L=1$, $\mathcal{N}(0,1)$ init (50 instances)")
ax.set_xticks(QUBITS)
ax.grid(True, alpha=0.3, linestyle="--")
ax.legend(frameon=False, ncol=2)
plt.tight_layout()
out = os.path.join(RESULTS, "optim", "Optim")
plt.savefig(out + ".pdf", dpi=300, bbox_inches="tight")
plt.savefig(out + ".png", dpi=150, bbox_inches="tight")
print("saved", out + ".pdf")
