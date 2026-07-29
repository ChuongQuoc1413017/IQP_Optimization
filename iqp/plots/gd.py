import os, json
import matplotlib.pyplot as plt
import numpy as np

from iqp.parallel import grid

_HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(os.path.dirname(_HERE), "results")
with open(os.path.join(RESULTS, "gd_merged.json")) as f:
    all_results = json.load(f)

qubits = grid.QUBITS

ham_names = {'ising': 'Classical Ising', 'maxcut': 'MaxCut', 'partition': 'Number Partition'}
order  = ['full', 'circular', 'single', 'hea', 'qaoa']
styles = {'full': ('o-', '#1f77b4'), 'circular': ('s-', '#ff7f0e'), 'single': ('^--', '#2ca02c'),
          'hea': ('D-', '#d62728'), 'qaoa': ('v-', '#9467bd')}
labels = {'full': 'Full Connectivity', 'circular': 'Circular Connectivity',
          'single': 'Single-Z Terms', 'hea': 'HEA (L=2)', 'qaoa': 'QAOA (p=2)'}
init_cols  = ['normal', 'uniform', 'pi4', 'he', 'glorot']
col_titles = {'normal': r'$\mathcal{N}(0, 1)$', 'uniform': r'$\mathcal{U}(-\pi, \pi)$',
              'pi4': r'$\pi/4$ perturbation', 'he': 'He', 'glorot': 'Glorot'}

fig, axes = plt.subplots(3, 5, figsize=(20, 11), sharex=True, sharey='row')
for row, ham in enumerate(['ising', 'maxcut', 'partition']):
    for col, init_name in enumerate(init_cols):
        ax  = axes[row][col]
        res = all_results[init_name][ham]
        scale = max(v for n in qubits for v in res['qaoa'][str(n)])   
        for mc in order:
            means = np.array([np.mean(res[mc][str(n)]) for n in qubits]) / scale
            stds  = np.array([np.std(res[mc][str(n)])  for n in qubits]) / scale
            fmt, color = styles[mc]
            ax.plot(qubits, means, fmt, color=color, label=labels[mc], linewidth=2, markersize=5)
            if mc != 'qaoa':  
                lower = np.clip(means - stds, 1e-12, None)  
                ax.fill_between(qubits, lower, means + stds, color=color, alpha=0.2)
        ax.axhline(1.0, color='gray', ls=':', lw=0.8)   
        ax.set_yscale('log')
        ax.set_xticks(qubits)                            
        ax.grid(True, alpha=0.3, linestyle='--')
        if row == 2:                                    
            ax.set_xlabel('Number of Qubits')
        if col == 0:
            ax.set_ylabel(ham_names[ham], fontsize=12)
        if row == 0:
            ax.set_title(col_titles[init_name], fontsize=13)

handles, labels_ = axes[0][0].get_legend_handles_labels()
fig.legend(handles, labels_, loc='upper center', ncol=5,
           bbox_to_anchor=(0.5, 1.02), frameon=False, fontsize=11)
fig.supylabel(r'$\mathrm{Var}\left[\partial \mathcal{C}/\partial \theta_k\right]$ (scaled to QAOA max)',
              fontsize=14, x=0.0)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS, "GV.pdf"), dpi=300, bbox_inches='tight')
plt.show()
