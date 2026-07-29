"""Ve ket qua pilot: ratio vs iteration.

Doc TRUC TIEP tu ckpt_pilot/ (khong can cho merge) -> chay duoc giua chung,
config nao co checkpoint thi ve config do, thieu thi de trong o va ghi chu.

Moi Hamiltonian 1 figure: hang = n (9, 15), cot = init (4), 5 duong/ansatz.
Duong = mean tren instances, dai mo = +-1 std. Duong ngang r=1 (Ising):
muc "khong tot hon trang thai |+>^n chua train" (E=0).
Danh dau moc iter=1000 (budget cu) de doi chieu voi ratio_5inits_*.json.

    python pilot_traj_plot.py
"""

import os
import json

import numpy as np
import matplotlib.pyplot as plt

from pilot_traj_driver import (CKPT, INITS, HAMS, ANSATZ, QUBITS,
                               N_ITERS, akey, ckpt_path)

STYLES = {
    "iqp-full":     ("-",  "#1f77b4", "Full Connectivity"),
    "iqp-circular": ("-",  "#ff7f0e", "Circular Connectivity"),
    "iqp-single":   ("--", "#2ca02c", "Single-Z Terms"),
    "hea-2":        ("-",  "#d62728", "HEA (L=2)"),
    "qaoa-2":       ("-",  "#9467bd", "QAOA (p=2)"),
}
COL_TITLES = {"normal": r"$\mathcal{N}(0, 1)$", "uniform": r"$\mathcal{U}(-\pi, \pi)$",
              "pi4": r"$\pi/4$ perturbation", "he": "He", "glorot": "Glorot"}
HAM_NAMES = {"ising": "Classical Ising", "maxcut": "MaxCut",
             "partition": "Number Partition"}
OLD_BUDGET = 1000


def load(init, ham, ak, n):
    p = ckpt_path(init, ham, ak, n)
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def main():
    n_missing = 0
    for ham in HAMS:
        nrows, ncols = len(QUBITS), len(INITS)
        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(4.2 * ncols, 3.4 * nrows),
                                 sharex=True, sharey="row", squeeze=False)
        for r, n in enumerate(QUBITS):
            for c, init in enumerate(INITS):
                ax = axes[r][c]
                has_any = False
                for ansatz, spec in ANSATZ:
                    ak = akey(ansatz, spec)
                    d = load(init, ham, ak, n)
                    if d is None:
                        n_missing += 1
                        continue
                    has_any = True
                    ratios = np.array(d["ratios"])       # (instances, moc)
                    iters = np.array(d["iters"])
                    m, s = ratios.mean(axis=0), ratios.std(axis=0)
                    ls, color, label = STYLES[ak]
                    ax.plot(iters, m, ls, color=color, label=label, lw=1.8)
                    ax.fill_between(iters, m - s, m + s, color=color, alpha=0.15)
                if ham == "ising":
                    ax.axhline(1.0, color="gray", ls=":", lw=1)
                ax.axvline(OLD_BUDGET, color="k", ls=":", lw=0.8, alpha=0.6)
                ax.grid(True, alpha=0.3, ls="--")
                if not has_any:
                    ax.text(0.5, 0.5, "chua co\ncheckpoint", ha="center",
                            va="center", transform=ax.transAxes, color="gray")
                if r == 0:
                    ax.set_title(COL_TITLES[init], fontsize=12)
                if c == 0:
                    ax.set_ylabel(f"{HAM_NAMES[ham]}, n={n}\n"
                                  r"Relative approx. ratio $r_{RA}$", fontsize=10)
                if r == nrows - 1:
                    ax.set_xlabel("Iteration")
                ax.set_xlim(0, N_ITERS)
        handles, labels = [], []
        for ax_row in axes:
            for ax in ax_row:
                h, l = ax.get_legend_handles_labels()
                if len(h) > len(handles):
                    handles, labels = h, l
        fig.legend(handles, labels, loc="upper center", ncol=5,
                   bbox_to_anchor=(0.5, 1.04), frameon=False, fontsize=10)
        fig.tight_layout()
        for ext in ("pdf", "png"):
            out = os.path.join(os.path.dirname(CKPT), f"pilot_traj_{ham}.{ext}")
            fig.savefig(out, dpi=200, bbox_inches="tight")
        print("Da ve:", f"pilot_traj_{ham}.pdf/.png")
        plt.close(fig)
    if n_missing:
        print(f"(con {n_missing} (config) chua co checkpoint -- ve lai sau khi "
              f"driver chay them)")


if __name__ == "__main__":
    main()
