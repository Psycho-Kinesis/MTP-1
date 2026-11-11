"""Presentation figures for H3, drawn from the saved results (no recomputation).

Run from the project root:
    python experiments/plot_h3.py      # writes figures/h3_main_problem_front.png and figures/h3_cost.png
"""
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT/'results'/'h3_inverse_design'
SEEDS = (11, 22, 33)
TOL = 1e-4
STYLE = {'C': ('tab:green', 'o', 'pipeline C (10 TSDT labels)'), 'P': ('tab:purple', 's', 'physics only (FSDT)'),
         'D': ('tab:orange', '^', 'data only (10 labels)')}


def main(results=RESULTS, out_dir=ROOT/'figures'):
    results, out_dir = Path(results), Path(out_dir)
    part_a = json.loads((results/'part_a_main_problem.json').read_text())
    manifest = json.loads((results/'manifest.json').read_text())
    errors = np.load(results/'h3_errors.npz')
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), layout='constrained')
    ax = axes[0]
    ref = np.array([(1000*r['reference']['ceramic'], r['reference']['mass']) for r in part_a])
    ax.plot(ref[:, 0], ref[:, 1], 'k-', lw=2, zorder=1, label='TSDT optimum (reference)')
    for arm, offset in (('C', -0.25), ('P', 0.25)):
        color, marker, label = STYLE[arm]
        for r in part_a:
            for s in SEEDS:
                rec = r['arms'][f'{arm} s{s}']
                bad = rec['open_violation'] > TOL
                ax.scatter(1000*rec['open']['ceramic']+offset, rec['open']['mass'], marker=marker, s=34, color=color,
                           facecolors='white' if bad else color, linewidths=1.2, zorder=3)
        ax.scatter([], [], marker=marker, color=color, label=label)
    ax.scatter([], [], marker='o', color='gray', facecolors='white', label='hollow: violates a TSDT limit (> 0.01%)')
    ax.set(xlabel='ceramic use h/(g+1) (mm)', ylabel='areal mass (kg/m$^2$)',
           title='Main problem: lightest plate for each ceramic limit\n(0.5 m x 0.75 m, 1 MPa, w ≤ 0.14 mm, f$_1$ ≥ 1200 Hz)')
    ax.text(0.03, 0.04, 'data-only designs (not shown) miss the optimum mass\nby 0.5-8% or violate a limit by 9%', transform=ax.transAxes,
            ha='left', va='bottom', fontsize=8, color='tab:orange')
    ax.grid(alpha=.3)
    ax.legend(fontsize=8, loc='upper right')

    ax = axes[1]
    data = [100*errors[f'decision_error_{arm}'].mean(axis=1) for arm in ('C', 'P', 'D')]
    parts = ax.boxplot(data, whis=(5, 95), showfliers=True, patch_artist=True, widths=0.5)
    for patch, arm in zip(parts['boxes'], ('C', 'P', 'D')):
        patch.set_facecolor(STYLE[arm][0])
        patch.set_alpha(0.5)
    ax.set_xticks([1, 2, 3], ['pipeline C\n(10 labels)', 'physics only\n(0 labels)', 'data only\n(10 labels)'])
    ax.set(yscale='log', ylabel='violation + |mass gap| vs TSDT optimum (%)',
           title=f"{manifest['problems']['kept']} random design problems, TSDT-checked\n(box 25-75%, whiskers 5-95%, mean of 3 seeds)")
    ax.grid(alpha=.3, which='both', axis='y')
    fig.savefig(out_dir/'h3_main_problem_front.png', dpi=170)
    plt.close(fig)

    q5 = manifest['decisions']['Q5']
    sec = manifest['secondary']
    rows = [('direct optimization\non TSDT (SLSQP)', q5['direct_slsqp_total_tsdt_analyses'], 'tab:gray'),
            ('data only + check\n(incl. 10 labels)', max(sec['D']['closed_loop']['total_analyses_including_labels'].values()), 'tab:orange'),
            ('physics only + check', max(sec['P']['closed_loop']['total_analyses_including_labels'].values()), 'tab:purple'),
            ('pipeline C + check\n(incl. 10 labels)', max(q5['c_total_tsdt_analyses'].values()), 'tab:green')]
    fig, ax = plt.subplots(figsize=(8, 3.6), layout='constrained')
    y = np.arange(len(rows))
    ax.barh(y, [r[1] for r in rows], color=[r[2] for r in rows])
    for k, r in enumerate(rows):
        ax.text(r[1]+60, k, f'{r[1]:,}', va='center', fontsize=9)
    ax.set_yticks(y, [r[0] for r in rows], fontsize=9)
    ax.set(xlabel=f"high-fidelity (TSDT) analyses to solve {manifest['problems']['kept']} design problems (worst seed)",
           title='Cost of reaching TSDT-feasible designs', xlim=(0, 1.12*q5['direct_slsqp_total_tsdt_analyses']))
    ax.grid(alpha=.3, axis='x')
    fig.savefig(out_dir/'h3_cost.png', dpi=170)
    plt.close(fig)
    print(f'wrote {out_dir/"h3_main_problem_front.png"} and {out_dir/"h3_cost.png"}')


if __name__ == '__main__':
    main(*sys.argv[1:])
