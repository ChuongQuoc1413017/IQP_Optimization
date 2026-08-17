import os, json
import matplotlib.pyplot as plt
import numpy as np

from iqp.experiments.expr_ent import QUBITS, LABELS
from iqp.plots.style import apply_style, panel_label, FULL_WIDTH

FIG_W = 12  # inches; embedded at \textwidth (~FULL_WIDTH) as a figure*
SCALE = FIG_W / FULL_WIDTH
apply_style(scale=SCALE)

_HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(os.path.dirname(_HERE), "results")
with open(os.path.join(RESULTS, "expr_ent_merged.json")) as f:
    res = json.load(f)

order = ['full', 'circular', 'single', 'hea',
         'qaoa-ising', 'qaoa-maxcut', 'qaoa-partition']

styles = {'full':           ('o-',  '#1f77b4'),
          'circular':       ('s-',  '#ff7f0e'),
          'single':         ('^--', '#2ca02c'),
          'hea':            ('D-',  '#d62728'),
          'qaoa-ising':     ('v-',  '#9467bd'),
          'qaoa-maxcut':    ('P-',  '#8c564b'),
          'qaoa-partition': ('X-',  '#e377c2')}

labels = {'full': 'Full Connectivity', 'circular': 'Circular Connectivity',
          'single': 'Single-Z Terms', 'hea': 'HEA (L=2)',
          'qaoa-ising': 'QAOA (p=2), Ising', 'qaoa-maxcut': 'QAOA (p=2), MaxCut',
          'qaoa-partition': 'QAOA (p=2), Partition'}

fig, axes = plt.subplots(1, 2, figsize=(FIG_W, 4.6))

ax = axes[0]
for mc in order:
    fmt, color = styles[mc]
    y = [res['expr'][mc][str(n)] for n in QUBITS]
    ax.semilogy(QUBITS, y, fmt, color=color, label=labels[mc])
ax.set_xlabel('Number of qubits')
ax.set_ylabel('Jensen-Shannon distance')
panel_label(ax, '(a)', fontsize=10 * SCALE)
ax.set_xticks(QUBITS)

ax = axes[1]
for mc in order:
    fmt, color = styles[mc]
    mean = np.array([res['entropy_mean'][mc][str(n)] for n in QUBITS])
    std = np.array([res['entropy_std'][mc][str(n)] for n in QUBITS])
    ax.plot(QUBITS, mean, fmt, color=color, label=labels[mc])
    ax.fill_between(QUBITS, mean - std, mean + std, color=color, alpha=0.15,
                    linewidth=0)
ax.set_xlabel('Number of qubits')
ax.set_ylabel('Average bipartite\nentanglement entropy')
panel_label(ax, '(b)', fontsize=10 * SCALE)
ax.set_xticks(QUBITS)

handles, labels_ = axes[0].get_legend_handles_labels()
fig.legend(handles, labels_, loc='upper center', ncol=4,
           bbox_to_anchor=(0.5, 1.0), frameon=False)

plt.tight_layout(rect=[0, 0, 1, 0.85])
plt.savefig(os.path.join(RESULTS, "Expr_Ent.pdf"), dpi=300, bbox_inches='tight')
plt.show()
