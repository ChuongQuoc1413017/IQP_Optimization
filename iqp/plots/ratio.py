import os, json
import matplotlib.pyplot as plt
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(os.path.dirname(_HERE), "results")
with open(os.path.join(RESULTS, "ratio_merged.json")) as f:
    all_results = json.load(f)

from iqp.parallel import grid
labels = [grid.config_label(a, s) for a, s in grid.ANSATZ_CONFIGS]  
qubits = grid.QUBITS                                                 

ham_names = {'ising': 'Classical Ising', 'maxcut': 'MaxCut', 'partition': 'Number Partition'}
order  = ['full', 'circular', 'single', 'hea', 'qaoa']
styles = {'full': ('o-', '#1f77b4'), 'circular': ('s-', '#ff7f0e'), 'single': ('^--', '#2ca02c'),
          'hea': ('D-', '#d62728'), 'qaoa': ('v-', '#9467bd')}
labels = {'full': 'Full Connectivity', 'circular': 'Circular Connectivity',
          'single': 'Single-Z Terms', 'hea': 'HEA (L=2)', 'qaoa': 'QAOA (p=2)'}
init_cols  = ['normal', 'uniform', 'pi4', 'he', 'glorot']
col_titles = {'normal': r'$\mathcal{N}(0, 1)$', 'uniform': r'$\mathcal{U}(-\pi, \pi)$',
              'pi4': r'$\pi/4$ perturbation', 'he': r'$\mathcal{N}(0, \frac{2}{n})$',
              'glorot': r'$\mathcal{N}(0, \frac{1}{n})$'}

fig, axes = plt.subplots(3, 5, figsize=(20, 11), sharex=True, sharey='row')
for row, mode_ham in enumerate(['ising', 'maxcut', 'partition']):
    for col, init_name in enumerate(init_cols):
        ax  = axes[row][col]
        res = all_results[init_name][mode_ham]

        for mc in order:
            means = np.array([np.mean(res[mc][str(n)]) for n in qubits])
            stds  = np.array([np.std(res[mc][str(n)], ddof=1) for n in qubits])
            sems  = stds / np.sqrt([len(res[mc][str(n)]) for n in qubits])
            fmt, color = styles[mc]
            ax.plot(qubits, means, fmt, color=color, label=labels[mc], linewidth=2, markersize=5)
            ax.fill_between(qubits, means - sems, means + sems, color=color, alpha=0.2, linewidth=0)

        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xticks(qubits)
        if row == 2:
            ax.set_xlabel('Number of Qubits')
        if col == 0:
            ax.set_ylabel(ham_names[mode_ham], fontsize=12)
        if row == 0:
            ax.set_title(col_titles[init_name], fontsize=13)

handles, labels_ = axes[0][0].get_legend_handles_labels()
fig.legend(handles, labels_, loc='upper center', ncol=5,
           bbox_to_anchor=(0.5, 1.02), frameon=False, fontsize=11)
fig.supylabel(r'Relative approx. ratio $r_{RA}$', fontsize=16, x=-0.0)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS, "Ratio.pdf"), dpi=300, bbox_inches='tight')
plt.show()