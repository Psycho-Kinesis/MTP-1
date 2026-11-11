"""E3a: physics-informed vs data-only learning of modal transfer functions at matched capacity.

Run from the project root:
    python experiments/run_e3a.py --output results/e3a_transfer

Uses results/pilot_200 as delivered (read only): the smooth-mixture family, its frozen
140/30/30 split and its labels. Every arm uses the same model class (fgplate.transfer):
H~(x) = tensor-Legendre polynomial of total degree d in the three design coordinates.

  D  data only     labels at N training designs (nested subsets as in E2)
  P  physics only  NO labels; FSDT energy at M unlabelled collocation designs
  Hy hybrid        labels at N designs + energy at M = 560 collocation designs

Collocation designs are scrambled-Halton points in the same transformed domain,
quantized like pilot-1.1 parameters, nested in M, and asserted to differ from every
validation and test design. Hyperparameters (degree, ridge, physics weight) are chosen
on validation only. The test partition is evaluated once, after all choices are frozen.
All decision rules below were fixed before the first run.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import scipy
from scipy.stats import qmc
import matplotlib
import matplotlib.ticker
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from fgplate.baseline import bootstrap_ci
from fgplate.fields import analytic_basis, project_analytic, reconstruct_analytic, relative_l2, scale_fields
from fgplate.pilot import quantize
from fgplate.transfer import (assemble_data, assemble_physics, design_coordinates, mode_matrices, solve_modes,
                              exponents)

MIXTURE = 2
DEGREES = (2, 3, 4, 5, 6)
RIDGES = (1e-12, 1e-10, 1e-8, 1e-6, 1e-4)
PHYSICS_WEIGHTS = (0.01, 0.1, 1.0, 10.0, 100.0)
LABEL_BUDGETS = (10, 20, 40, 70, 140)
COLLOCATION = (20, 40, 70, 140, 280, 560)
HYBRID_M = 560
SEEDS = (11, 22, 33)
TARGET = dict(median_percent=1.0, p95_percent=3.0)
RULES = dict(
    selection='per arm, budget and seed: minimize validation median field error, then validation p95',
    target=f'in-domain target: validation median <= {TARGET["median_percent"]}% and p95 <= {TARGET["p95_percent"]}% (mean over seeds)',
    H1=('H1 (label efficiency) is supported at label budget N if the 95% bootstrap interval of the median '
        'per-design paired difference (data-only error minus hybrid error, each averaged over seeds) on the '
        '30 validation designs lies entirely above 0. Evaluated at N = 10, 20 and 40; reported for all N.'),
    label_free='physics-only reaches the target label-free if any M meets it; report the smallest such M',
    capacity=('capacity curve: validation error versus degree with the best ridge, for physics-only at M = 560 '
              'and data-only at N = 140; a floor that does not move with more data or collocation indicates the '
              'model class, not the information, limits accuracy'),
)
ARMS = dict(D='data only (labels)', P='physics only (no labels)', Hy=f'hybrid (labels + {HYBRID_M} collocation)')


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def summary(errors):
    e = 100*np.asarray(errors)
    return dict(median_percent=float(np.median(e)), p95_percent=float(np.quantile(e, .95)),
                worst_percent=float(e.max()), mean_percent=float(e.mean()),
                median_percent_ci95=(100*bootstrap_ci(errors, 'median')[0]).tolist(), cases=int(len(e)))


def collocation_designs(count, seed):
    unit = qmc.Halton(d=3, scramble=True, seed=seed).random(count)
    t = qmc.scale(unit, [0, np.log(5), .5], [np.log(9), np.log(100), 2])
    return quantize(np.column_stack((np.expm1(t[:, 0]), np.exp(t[:, 1]), t[:, 2])))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--pilot', type=Path, default=ROOT/'results'/'pilot_200')
    parser.add_argument('--output', type=Path, default=ROOT/'results'/'e3a_transfer')
    args = parser.parse_args()
    out = args.output.resolve()
    if out.exists() and any(out.iterdir()):
        parser.error('output directory is not empty; choose a new experiment directory')
    out.mkdir(parents=True, exist_ok=True)
    start_all = perf_counter()

    # ---------------------------------------------------------------- data (read only)
    pilot = args.pilot.resolve()
    recorded = json.loads((pilot/'checksums.json').read_text())
    assert sha256(pilot/'dataset.npz') == recorded['dataset.npz'], 'pilot dataset differs from its recorded checksum'
    with np.load(pilot/'dataset.npz', allow_pickle=False) as z:
        P, split, ids = z['parameters'], z['split'], z['design_ids']
        Q, modes_all = z['pressure_coefficients'][:, MIXTURE], z['mode_indices']
        fields, xi, eta = z['fields'][:, MIXTURE], z['xi'], z['eta']
    active = np.flatnonzero(np.any(Q != 0, axis=0))
    modes, c = modes_all[active], Q[:, active]
    S = scale_fields(fields, 1.0, 1/P[:, 1])
    basis = analytic_basis(modes, xi, eta)
    labels = np.transpose(project_analytic(S, basis), (0, 2, 1))/c[:, :, None]      # H~ [designs, 25, 5]
    x = design_coordinates(P)
    train, val, test = (np.flatnonzero(split == s) for s in ('train', 'validation', 'test'))

    def field_errors(theta, degree, idx):
        from fgplate.transfer import features
        H = np.einsum('np,kcp->nkc', features(x[idx], degree), theta)
        prediction = reconstruct_analytic(np.transpose(H*c[idx, :, None], (0, 2, 1)), basis)
        return relative_l2(prediction, S[idx])

    # ---------------------------------------------------------------- collocation designs and physics normal equations
    colloc, phys = {}, {}
    t0 = perf_counter()
    for seed in SEEDS:
        designs = collocation_designs(max(COLLOCATION), 1000+seed)
        xc = design_coordinates(designs)
        for idx in (val, test):
            gap = np.min(np.linalg.norm(xc[:, None]-x[idx][None], axis=2))
            assert gap > 1e-6, 'collocation design coincides with a held-out design'
        K, f = mode_matrices(designs, modes)                    # mechanics only, no solve
        colloc[seed] = dict(designs=designs, x=xc, K=K, f=f)
        for M in COLLOCATION:
            for d in DEGREES:
                phys[seed, M, d] = assemble_physics(xc[:M], K[:M], f[:M], d)
    physics_assembly_seconds = perf_counter()-t0
    print(f'Physics normal equations assembled ({physics_assembly_seconds:.1f} s)', flush=True)

    # ---------------------------------------------------------------- fits (validation selection)
    runs, chosen = [], {}
    capacity = {'P': {d: [] for d in DEGREES}, 'D': {d: [] for d in DEGREES}}

    def select(arm, budget, seed, candidates):
        best = None
        for degree, ridge, weight, A, b in candidates:
            theta = solve_modes(A, b, degree, ridge)
            e = field_errors(theta, degree, val)
            key = (np.median(e), np.quantile(e, .95))
            if arm == 'P' and budget == HYBRID_M or arm == 'D' and budget == max(LABEL_BUDGETS):
                capacity[arm][degree].append((ridge, seed, key))
            if best is None or key < best[0]:
                best = (key, dict(degree=degree, ridge=ridge, physics_weight=weight), theta, e)
        _, hp, theta, e = best
        chosen[arm, budget, seed] = (hp, theta)
        runs.append(dict(arm=arm, budget=budget, seed=seed, selected=hp, validation=summary(e),
                         validation_errors=e.tolist()))

    for seed in SEEDS:
        order = np.random.default_rng(seed).permutation(train)          # identical subsets to E2
        for M in COLLOCATION:
            select('P', M, seed, [(d, r, 1.0, *phys[seed, M, d]) for d in DEGREES for r in RIDGES])
        for N in LABEL_BUDGETS:
            subset = np.sort(order[:N])
            data = {d: assemble_data(x[subset], labels[subset], d)[:2] for d in DEGREES}
            select('D', N, seed, [(d, r, 0.0, *data[d]) for d in DEGREES for r in RIDGES])
            select('Hy', N, seed, [(d, r, w, data[d][0]+w*phys[seed, HYBRID_M, d][0],
                                    data[d][1]+w*phys[seed, HYBRID_M, d][1])
                                   for d in DEGREES for r in RIDGES for w in PHYSICS_WEIGHTS])
        print(f'seed {seed} done ({perf_counter()-start_all:.0f} s)', flush=True)

    # ---------------------------------------------------------------- aggregate (validation only)
    def curve(arm, budgets):
        out_ = {}
        for n in budgets:
            rows = [r for r in runs if r['arm'] == arm and r['budget'] == n]
            med = [r['validation']['median_percent'] for r in rows]
            p95 = [r['validation']['p95_percent'] for r in rows]
            out_[str(n)] = dict(median_percent_mean=float(np.mean(med)), median_percent_range=[min(med), max(med)],
                                p95_percent_mean=float(np.mean(p95)), p95_percent_range=[min(p95), max(p95)],
                                degrees=[r['selected']['degree'] for r in rows])
        return out_
    curves = dict(D=curve('D', LABEL_BUDGETS), Hy=curve('Hy', LABEL_BUDGETS), P=curve('P', COLLOCATION))

    def meets(entry):
        return entry['median_percent_mean'] <= TARGET['median_percent'] and entry['p95_percent_mean'] <= TARGET['p95_percent']
    decisions = dict(rules=RULES)
    decisions['data_only_smallest_N_meeting_target'] = next((n for n in LABEL_BUDGETS if meets(curves['D'][str(n)])), None)
    decisions['hybrid_smallest_N_meeting_target'] = next((n for n in LABEL_BUDGETS if meets(curves['Hy'][str(n)])), None)
    decisions['physics_only_smallest_M_meeting_target'] = next((m for m in COLLOCATION if meets(curves['P'][str(m)])), None)
    h1 = {}
    for N in LABEL_BUDGETS:
        d = np.mean([[*r['validation_errors']] for r in runs if r['arm'] == 'D' and r['budget'] == N], axis=0)
        hy = np.mean([[*r['validation_errors']] for r in runs if r['arm'] == 'Hy' and r['budget'] == N], axis=0)
        diff = d-hy
        lo, hi = bootstrap_ci(diff, 'median')[0]
        h1[str(N)] = dict(median_paired_difference_percent=float(100*np.median(diff)),
                          ci95_percent=[float(100*lo), float(100*hi)], supported=bool(lo > 0),
                          hybrid_better_designs=int(np.sum(diff > 0)), designs=int(len(diff)))
    decisions['H1'] = h1
    decisions['H1_supported_at_N_10_20_40'] = all(h1[str(n)]['supported'] for n in (10, 20, 40))
    cap = {}
    for arm in ('P', 'D'):
        cap[arm] = {}
        for d in DEGREES:
            rows = capacity[arm][d]
            best_by_seed = {}
            for ridge, seed, key in rows:
                if seed not in best_by_seed or key < best_by_seed[seed]:
                    best_by_seed[seed] = key
            cap[arm][str(d)] = dict(coefficients_per_function=len(exponents(d)),
                                    median_percent_mean=float(100*np.mean([k[0] for k in best_by_seed.values()])),
                                    p95_percent_mean=float(100*np.mean([k[1] for k in best_by_seed.values()])))
    decisions['capacity'] = cap

    # ---------------------------------------------------------------- single final test evaluation
    for run in runs:
        hp, theta = chosen[run['arm'], run['budget'], run['seed']]
        run['test'] = summary(field_errors(theta, hp['degree'], test))
    test_curves = {arm: {str(n): dict(median_percent_mean=float(np.mean([r['test']['median_percent'] for r in runs
                                                                         if r['arm'] == arm and r['budget'] == n])),
                                      p95_percent_mean=float(np.mean([r['test']['p95_percent'] for r in runs
                                                                      if r['arm'] == arm and r['budget'] == n])))
                         for n in (COLLOCATION if arm == 'P' else LABEL_BUDGETS)} for arm in ARMS}

    # ---------------------------------------------------------------- figures
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout='constrained')
    for ax, stat, target in zip(axes, ('median', 'p95'), (TARGET['median_percent'], TARGET['p95_percent'])):
        for arm, marker in (('D', 'o'), ('Hy', 's')):
            mean = [curves[arm][str(n)][f'{stat}_percent_mean'] for n in LABEL_BUDGETS]
            lo = [curves[arm][str(n)][f'{stat}_percent_range'][0] for n in LABEL_BUDGETS]
            hi = [curves[arm][str(n)][f'{stat}_percent_range'][1] for n in LABEL_BUDGETS]
            ax.plot(LABEL_BUDGETS, mean, marker=marker, label=ARMS[arm])
            ax.fill_between(LABEL_BUDGETS, lo, hi, alpha=.15)
        best_p = min(curves['P'][str(m)][f'{stat}_percent_mean'] for m in COLLOCATION)
        ax.axhline(best_p, color='g', lw=1.2, ls='--', label=f'physics only, 0 labels (best M): {best_p:.2f}%')
        ax.axhline(target, color='r', lw=.8, ls=':', label=f'target {target}%')
        ax.set(xscale='log', yscale='log', xticks=LABEL_BUDGETS, xlabel='labelled training designs',
               ylabel=f'{"Median" if stat == "median" else "95th percentile"} validation field error (%)',
               title=f'E3a: {"median" if stat == "median" else "95th percentile"} over 30 designs (3 seeds)')
        ax.set_xticklabels([str(b) for b in LABEL_BUDGETS]); ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.grid(alpha=.3, which='both'); ax.legend(fontsize=7.5)
    fig.savefig(out/'e3a_label_curves.png', dpi=170); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout='constrained')
    for stat, marker in (('median', 'o'), ('p95', 's')):
        axes[0].plot(COLLOCATION, [curves['P'][str(m)][f'{stat}_percent_mean'] for m in COLLOCATION], marker=marker,
                     label=f'physics only, {stat}')
        for arm, ls in (('P', '-'), ('D', '--')):
            axes[1].plot(DEGREES, [cap[arm][str(d)][f'{stat}_percent_mean'] for d in DEGREES], marker=marker, ls=ls,
                         label=f'{"physics only, M=560" if arm == "P" else "data only, N=140"}: {stat}')
    for ax in axes:
        ax.axhline(TARGET['median_percent'], color='r', lw=.8, ls=':')
        ax.axhline(TARGET['p95_percent'], color='r', lw=.8, ls=':')
        ax.set_yscale('log'); ax.grid(alpha=.3, which='both'); ax.legend(fontsize=8)
    axes[0].set(xscale='log', xticks=COLLOCATION, xlabel='unlabelled collocation designs M',
                ylabel='validation field error (%)', title='E3a: physics only, no labels')
    axes[0].set_xticklabels([str(m) for m in COLLOCATION]); axes[0].xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    axes[1].set(xticks=DEGREES, xlabel='polynomial degree (model capacity)', ylabel='validation field error (%)',
                title='E3a: capacity (best ridge per degree)')
    fig.savefig(out/'e3a_physics_and_capacity.png', dpi=170); plt.close(fig)

    # ---------------------------------------------------------------- persist
    np.savez_compressed(out/'e3a_errors.npz', validation_ids=ids[val], test_ids=ids[test],
                        **{f'{r["arm"]}_{r["budget"]}_s{r["seed"]}_validation': np.array(r['validation_errors'])
                           for r in runs},
                        **{f'collocation_designs_s{seed}': colloc[seed]['designs'] for seed in SEEDS})
    for r in runs:
        r.pop('validation_errors')
    manifest = dict(experiment='E3a physics-informed vs data-only modal transfer at matched capacity',
                    created_utc=datetime.now(timezone.utc).isoformat(),
                    pilot=dict(path=str(args.pilot), dataset_sha256=recorded['dataset.npz']),
                    load_family='smooth_mixture (25 signed sine modes, unit norm, qscale 1 Pa)',
                    model_class='H~(x): tensor Legendre polynomial, total degree d, fixed domain box -> [-1,1]^3; '
                                'loads applied through the analytic Navier basis',
                    arms=ARMS, rules=RULES, grids=dict(degrees=DEGREES, ridges=RIDGES, physics_weights=PHYSICS_WEIGHTS),
                    label_budgets=LABEL_BUDGETS, collocation_counts=COLLOCATION, hybrid_collocation=HYBRID_M, seeds=SEEDS,
                    collocation='scrambled Halton (seed 1000+seed), transformed domain, quantized to 10 significant '
                                'digits, nested in M, min distance to validation/test designs asserted > 1e-6',
                    partitions=dict(train=len(train), validation=len(val), test=len(test)),
                    runs=runs, validation_curves=curves, decisions=decisions, test_curves=test_curves,
                    cost_note=('Labels: one 5x5 solve per mode per labelled design. Physics: one 5x5 matrix assembly per '
                               'mode per collocation design, no solve. On this simply supported family both are '
                               'microsecond-scale, so H1 here is a methodological test of label efficiency, not a '
                               'practical cost saving.'),
                    physics_assembly_seconds=physics_assembly_seconds, total_seconds=perf_counter()-start_all,
                    environment=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                                     matplotlib=matplotlib.__version__, platform=platform.platform()),
                    source_sha256={str(p.relative_to(ROOT)): sha256(p) for p in
                                   [*sorted((ROOT/'fgplate').glob('*.py')), Path(__file__)]})
    write_json(out/'manifest.json', manifest)
    write_json(out/'checksums.json', {p.name: sha256(p) for p in sorted(out.iterdir()) if p.is_file() and p.name != 'checksums.json'})
    print(json.dumps(dict(curves=curves, decisions={k: v for k, v in decisions.items() if k != 'rules'}), indent=1), flush=True)


if __name__ == '__main__':
    main()
