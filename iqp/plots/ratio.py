import os, json
import matplotlib.pyplot as plt
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(os.path.dirname(_HERE), "results")
with open(os.path.join(RESULTS, "ratio_merged.json")) as f:
    all_results = json.load(f)

from iqp.parallel import grid
from iqp.plots.style import apply_style, FULL_WIDTH

FIG_W = 20     # inches; embedded at \textwidth (~FULL_WIDTH) as a figure*
BASE_SCALE = FIG_W / FULL_WIDTH
DENSITY = 0.7  # extra shrink for the lines and markers only: a 15-panel grid
               # reads better with thinner strokes. Text is exempt (see below).
SCALE = BASE_SCALE * DENSITY
apply_style(scale=SCALE)
LABEL_FS = 9 * BASE_SCALE  # shared x/y axis labels: same printed size as robust.py/expr_ent.py
# Every text element -- tick numbers, panel titles, row names, legend -- is set
# to LABEL_FS so the whole figure carries one uniform type size.
plt.rcParams.update({
    "font.size": LABEL_FS,
    "axes.labelsize": LABEL_FS,
    "axes.titlesize": LABEL_FS,
    "legend.fontsize": LABEL_FS,
    "xtick.labelsize": LABEL_FS,
    "ytick.labelsize": LABEL_FS,
})
SMALL_FS = LABEL_FS        # per-column init names, per-row Hamiltonian names
LINEWIDTH = 0.9 * SCALE
labels = [grid.config_label(a, s) for a, s in grid.ANSATZ_CONFIGS]
qubits = grid.QUBITS                                                 

ham_names = {'ising': 'Classical Ising', 'maxcut': 'MaxCut', 'partition': 'Number Partition'}
order  = ['full', 'circular', 'single', 'hea', 'qaoa']
styles = {'full': ('o-', '#1f77b4'), 'circular': ('s-', '#ff7f0e'), 'single': ('^--', '#2ca02c'),
          'hea': ('D-', '#d62728'), 'qaoa': ('v-', '#9467bd')}
labels = {'full': 'Full Connectivity', 'circular': 'Circular Connectivity',
          'single': 'Single-Z Terms', 'hea': 'HEA (L=2)', 'qaoa': 'QAOA (p=2)'}
init_cols  = ['normal', 'uniform', 'pi4', 'he', 'lecun']
col_titles = {'normal': r'$\mathcal{N}(0, 1)$', 'uniform': r'$\mathcal{U}(-\pi, \pi)$',
              'pi4': r'$\pi/4$ perturbation', 'he': 'He', 'lecun': 'LeCun'}

fig, axes = plt.subplots(3, 5, figsize=(FIG_W, 11), sharex=True, sharey='row')
for row, mode_ham in enumerate(['ising', 'maxcut', 'partition']):
    for col, init_name in enumerate(init_cols):
        ax  = axes[row][col]
        res = all_results[init_name][mode_ham]

        for mc in order:
            means = np.array([np.mean(res[mc][str(n)]) for n in qubits])
            stds  = np.array([np.std(res[mc][str(n)], ddof=1) for n in qubits])
            sems  = stds / np.sqrt([len(res[mc][str(n)]) for n in qubits])
            fmt, color = styles[mc]
            ax.plot(qubits, means, fmt, color=color, label=labels[mc], linewidth=LINEWIDTH)
            ax.fill_between(qubits, means - sems, means + sems, color=color, alpha=0.2, linewidth=0)

        ax.set_xticks(qubits)
        if col == 0:
            ax.set_ylabel(ham_names[mode_ham], fontsize=SMALL_FS)
        if row == 0:
            ax.set_title(col_titles[init_name], fontsize=SMALL_FS)

handles, labels_ = axes[0][0].get_legend_handles_labels()
fig.legend(handles, labels_, loc='upper center', ncol=5,
           bbox_to_anchor=(0.5, 1.05), frameon=False)
fig.supxlabel('Number of qubits', fontsize=LABEL_FS)
fig.supylabel(r'Relative approx. ratio $r_{RA}$', fontsize=LABEL_FS)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS, "Ratio.pdf"), dpi=300, bbox_inches='tight')
plt.show()