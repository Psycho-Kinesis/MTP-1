"""Presentation version of the H2 summary figure, drawn from the saved H2 manifest (no recomputation).

Run from the project root:
    python experiments/plot_h2.py            # writes figures/h2_load_families_summary.png
"""
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SHORT = {'L1': 'L1\ntraining-type\nmixture', 'L2': 'L2\nuniform\npressure', 'L3': 'L3\nwide\npatch', 'L4': 'L4\nnarrow\npatch'}
ARMS = [('P-ext', 'physics only (0 labels)', 'tab:purple'), ('D-ext', 'data only', 'tab:orange'),
        ('A', 'frozen E5 pipeline (5x5 modes)', 'tab:gray'), ('C', 'E5 pipeline extended to 15x15 modes', 'tab:green')]


def main(manifest=ROOT/'results'/'h2_loads'/'manifest.json', out=ROOT/'figures'/'h2_load_families_summary.png'):
    m = json.loads(Path(manifest).read_text())
    table, refs, fams = m['table'], m['references'], list(m['families'])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), layout='constrained')
    xs = np.arange(len(fams))
    for ax, stat, target in zip(axes, ('median', 'p95'), (1.0, 3.0)):
        for j, (arm, label, color) in enumerate(ARMS):
            vals = [table[f][arm]['0' if arm == 'P-ext' else '10'][f'{stat}_percent_mean'] for f in fams]
            ax.bar(xs+(j-1.5)*0.2, vals, width=0.2, color=color, label=label+('' if arm == 'P-ext' else ', 10 labels'))
        ax.scatter(xs+0.1, [refs[f]['TSDT truncated to 5x5'][f'{stat}_percent'] for f in fams], marker='_', s=300, color='k',
                   zorder=3, label='floor of a 5x5-mode model (gray bars)')
        ax.axhline(target, color='r', ls=':', lw=1, label=f'target {target:g}%')
        ax.set(xticks=xs, xticklabels=[SHORT[f] for f in fams], yscale='log', ylim=(0.02, 5000),
               ylabel=f'{"median" if stat == "median" else "95th percentile"} error vs TSDT (%)',
               title=f'Unseen load shapes, 200 fresh designs: {"median" if stat == "median" else "95th percentile"}')
        ax.grid(alpha=.3, axis='y', which='both')
    axes[0].legend(fontsize=7, loc='upper left', ncol=2)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170)
    print(f'wrote {out}')


if __name__ == '__main__':
    main(*sys.argv[1:])
