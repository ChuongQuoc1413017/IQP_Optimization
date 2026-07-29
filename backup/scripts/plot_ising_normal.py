import json
import os
import numpy as np
import matplotlib.pyplot as plt

RATIO_FILE = "ising_ratio_normal.json"
LOSS_FILE  = "ising_loss_normal.json"
MODES      = ["full", "circular", "single"]

STYLE = {
    "full":     {"label": "Full Connectivity",     "marker": "o", "linestyle": "-"},
    "circular": {"label": "Circular Connectivity",  "marker": "s", "linestyle": "--"},
    "single":   {"label": "Single-Z Terms",         "marker": "^", "linestyle": "-."},
}

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 12,
    "axes.labelsize": 13,
    "axes.titlesize": 13,
    "legend.fontsize": 10,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "figure.dpi": 150,
})

def load(path):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Khong tim thay '{path}'. Chay script trong dung thu muc "
            f"da luu ket qua tu notebook Ising."
        )
    with open(path, "r") as f:
        return json.load(f)

def sorted_int_keys(d):
    return sorted(int(k) for k in d.keys())

def plot_ratio():
    data = load(RATIO_FILE)

    fig, ax = plt.subplots(figsize=(6, 5))

    print("\n=== Mean relative approx. ratio (Ising / Normal) ===")
    header = "n_qubits  " + "  ".join(f"{m:>9}" for m in MODES)
    print(header)

    table = {}
    for m in MODES:
        if m not in data:
            print(f"[canh bao] thieu mode '{m}' trong {RATIO_FILE}")
            continue
        qubits = sorted_int_keys(data[m])
        mean = np.array([np.mean(data[m][str(q)]) for q in qubits])
        std  = np.array([np.std(data[m][str(q)])  for q in qubits])
        table[m] = (qubits, mean)

        ax.plot(qubits, mean,
                marker=STYLE[m]["marker"], linestyle=STYLE[m]["linestyle"],
                linewidth=2, markersize=5, label=STYLE[m]["label"])
        ax.fill_between(qubits, mean - std, mean + std, alpha=0.15)

    # optimal reference line
    ax.axhline(y=0, color="black", linestyle=":", linewidth=2, label="Optimal Value")

    ax.set_xlabel("Number of Qubits")
    ax.set_ylabel("Relative Approx. Ratio")
    ax.set_title("Classical Ising  -  Normal (0, 1)")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig("ising_normal_approx_ratio.png", dpi=300, bbox_inches="tight")
    print("\n-> Saved ising_normal_approx_ratio.png")

    # print table aligned to the first available mode's qubit grid
    any_mode = next(iter(table))
    qubits = table[any_mode][0]
    for i, q in enumerate(qubits):
        row = f"{q:>8}  "
        for m in MODES:
            row += f"{table[m][1][i]:>9.4f}  " if m in table else f"{'--':>9}  "
        print(row)

def plot_convergence():
    data = load(LOSS_FILE)

    # pick the largest n that exists across modes
    common = None
    for m in MODES:
        if m in data:
            ks = set(data[m].keys())
            common = ks if common is None else (common & ks)
    if not common:
        print("[canh bao] khong co n_qubits chung de ve convergence")
        return
    n = str(max(int(k) for k in common))

    fig, ax = plt.subplots(figsize=(7, 5))
    for m in MODES:
        if m not in data or n not in data[m]:
            continue
        runs = np.array(data[m][n])          # (50, n_iters)
        mean = runs.mean(axis=0)
        ci = 1.96 * runs.std(axis=0) / np.sqrt(runs.shape[0])
        it = np.arange(1, len(mean) + 1)
        ax.plot(it, mean, linewidth=2,
                linestyle=STYLE[m]["linestyle"], label=STYLE[m]["label"])
        ax.fill_between(it, mean - ci, mean + ci, alpha=0.15)

    ax.set_xlabel("Iteration")
    ax.set_ylabel(r"Loss  $\langle H \rangle$")
    ax.set_title(f"Ising  -  Normal (0, 1)  -  {n} qubits")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig("ising_normal_convergence.png", dpi=300, bbox_inches="tight")
    print(f"-> Saved ising_normal_convergence.png  (n = {n})")


if __name__ == "__main__":
    plot_ratio()
    plot_convergence()
    plt.show()