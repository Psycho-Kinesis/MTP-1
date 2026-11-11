"""H2: generalization of the confirmed E5 pipeline to unseen load shapes.

Run from the project root:
    python experiments/run_h2.py --output results/h2_loads

Follows docs/H2_PREREGISTRATION.md (sha256 asserted). Steps:
1. refit the frozen E5 arms (run_e5.fit_all; asserted identical to results/e5_base_model_v2);
2. extend the label-free base (P-ext), the multiplicative correction (C) and data only (D-ext)
   to all modes m, n <= 15 with hyperparameters FROZEN from E5 and the SAME labelled designs;
   assert that their 5x5 part reproduces the frozen arms exactly;
3. only then generate 200 fresh designs (pilot-1.1, seed 20261231) with four load families,
   compute the TSDT truth with modes m, n <= 25, and evaluate every arm once (exact modal L2).
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from numpy.polynomial.legendre import leggauss
import scipy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from fgplate.baseline import bootstrap_ci
from fgplate.pilot import GENERATOR_VERSION, design_loads, load_key, sample_designs
from fgplate.transfer import (assemble_data, assemble_physics, classical_prefactor, design_coordinates, mode_matrices,
                              predict_scaled, solve_modes)
from fgplate.tsdt import transfer
from run_e5 import SEEDS, SCOPE, TARGET, fit_all, kh2, sha256, shear_share, summary, write_json

PREREGISTRATION = ROOT/'docs'/'H2_PREREGISTRATION.md'
PREREGISTRATION_SHA256 = '5b8e0ef2061a8104307a818deca1089a094ab2f51838fde69aef36956b35c088'
TEST_SEED, TEST_COUNT = 20261231, 200
BUDGETS = (10, 40, 140)
PRIMARY_N = 10
EXT, TRUTH = 15, 25
FAMILIES = {'L1': 'smooth mixture (training family)', 'L2': 'uniform pressure',
            'L3': 'wide Gaussian patch, sigma 0.10-0.20', 'L4': 'narrow Gaussian patch, sigma 0.04-0.07'}
PATCH = {'L3': dict(sigma=(0.10, 0.20), centre=(0.25, 0.75)), 'L4': dict(sigma=(0.04, 0.07), centre=(0.2, 0.8))}
ARMS = {'A': 'frozen E5 pipeline M-S@P2 (5x5)', 'P-ext': 'label-free base, 15x15', 'B': 'A + label-free base outside 5x5',
        'C': 'M-S@P2 extended to 15x15 (same labelled designs)', 'D': 'data only, 5x5', 'D-ext': 'data only, 15x15'}


def square_modes(k):
    return np.array([(m, n) for m in range(1, k+1) for n in range(1, k+1)])


def embed(H, modes, target_modes):
    """Place H [N, len(modes), 5] into the target mode list (zeros elsewhere)."""
    lookup = {tuple(m): i for i, m in enumerate(target_modes)}
    out = np.zeros((len(H), len(target_modes), 5))
    out[:, [lookup[tuple(m)] for m in modes]] = H
    return out


def patch_coefficients(centre, sigma, modes, points=600):
    """Q_mn of a separable unit-peak Gaussian patch in normalized coordinates."""
    z, w = leggauss(points)
    t, w = (z+1)/2, w/2
    kmax = int(modes.max())
    sines = np.sin(np.outer(np.arange(1, kmax+1), np.pi*t))
    ix = sines@(w*np.exp(-(t-centre[0])**2/(2*sigma**2)))
    iy = sines@(w*np.exp(-(t-centre[1])**2/(2*sigma**2)))
    return 4*ix[modes[:, 0]-1]*iy[modes[:, 1]-1]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--pilot', type=Path, default=ROOT/'results'/'pilot_200')
    parser.add_argument('--development', type=Path, default=ROOT/'results'/'e5_base_model_v2'/'manifest.json')
    parser.add_argument('--output', type=Path, default=ROOT/'results'/'h2_loads')
    args = parser.parse_args()
    assert sha256(PREREGISTRATION) == PREREGISTRATION_SHA256, 'pre-registration file changed; H2 must not run'
    out = args.output.resolve()
    if out.exists() and any(out.iterdir()):
        parser.error('output directory is not empty; choose a new experiment directory')
    out.mkdir(parents=True, exist_ok=True)
    start = perf_counter()

    # ------------------------------------------------ 1. frozen E5 arms
    fit = fit_all(args.pilot.resolve(), log=lambda m: print(m, flush=True))
    assert fit['chosen'] == 'P2'
    dev = {(r['arm'], r['budget'], r['seed']): r['validation'] for r in json.loads(args.development.read_text())['runs']}
    for key, result in fit['results'].items():
        assert all(np.isclose(result['validation'][k], dev[key][k], rtol=1e-9, atol=1e-12)
                   for k in ('median_percent', 'p95_percent', 'worst_percent')), key
    P, modes5, models, res = fit['P'], fit['modes'], fit['models'], fit['results']
    train = fit['train']
    print('frozen E5 arms reproduced', flush=True)

    # ------------------------------------------------ 2. extension to 15x15 with frozen hyperparameters
    modes_ext = square_modes(EXT)
    in5 = np.array([tuple(m) in {tuple(q) for q in modes5} for m in modes_ext])
    assert np.array_equal(modes_ext[in5], modes5)
    x_pilot = design_coordinates(P)
    pref_pilot_ext = classical_prefactor(P, modes_ext)
    kh2_pilot_ext = kh2(P, modes_ext)
    H_tsdt_pilot_ext = transfer(P, modes_ext, 'TSDT')          # pilot labels for the extra modes (same designs)
    psi = shear_share(P)
    ext = {}
    for seed in SEEDS:
        hp = res['P2', 0, seed]['selected']
        colloc = fit['colloc'][seed]
        K, f = mode_matrices(colloc, modes_ext)
        theta_p = solve_modes(*assemble_physics(design_coordinates(colloc), K, f, hp['degree'],
                                                prefactor=classical_prefactor(colloc, modes_ext)), hp['degree'], hp['ridge'], SCOPE)
        assert np.allclose(theta_p[in5], models['P2', 0, seed].base_theta, rtol=1e-10, atol=1e-14), 'P-ext 5x5 part differs'
        ext['P-ext', 0, seed] = (theta_p, hp['degree'])
        H_P = predict_scaled(theta_p, hp['degree'], x_pilot, pref_pilot_ext)
        order_r = np.random.default_rng(seed).permutation(train)
        keys = np.log(np.random.default_rng([seed, 7]).random(len(train)))/psi[train]
        order_s = train[np.argsort(-keys)]
        for N in BUDGETS:
            sel = res['M-S@P2', N, seed]['selected']
            sub = np.sort(order_s[:N])
            pref = H_P*kh2_pilot_ext[:, :, None]
            A, b, _ = assemble_data(x_pilot[sub], H_tsdt_pilot_ext[sub]-H_P[sub], sel['correction_degree'], prefactor=pref[sub])
            ext['C', N, seed] = (theta_p, hp['degree'], solve_modes(A, b, sel['correction_degree'], sel['ridge'], SCOPE),
                                 sel['correction_degree'])
            sel_d = res['D', N, seed]['selected']
            sub_r = np.sort(order_r[:N])
            A, b, _ = assemble_data(x_pilot[sub_r], H_tsdt_pilot_ext[sub_r], sel_d['degree'])
            ext['D-ext', N, seed] = (solve_modes(A, b, sel_d['degree'], sel_d['ridge'], SCOPE), sel_d['degree'])
    print(f'extended arms fitted ({perf_counter()-start:.0f} s)', flush=True)

    def predict(arm, N, seed, params):
        """Transfer functions on the 25x25 truth mode list (zeros where the arm has no mode)."""
        modes25 = square_modes(TRUTH)
        if arm in ('A', 'D'):
            H = models['M-S@P2' if arm == 'A' else 'D', N, seed](params, modes5)
            return embed(H, modes5, modes25)
        x = design_coordinates(params)
        if arm in ('P-ext', 'B', 'C'):
            theta_p, dp = ext['P-ext', 0, seed]
            H_P = predict_scaled(theta_p, dp, x, classical_prefactor(params, modes_ext))
            if arm == 'P-ext':
                return embed(H_P, modes_ext, modes25)
            if arm == 'B':
                H = H_P.copy()
                H[:, in5] = models['M-S@P2', N, seed](params, modes5)
                return embed(H, modes_ext, modes25)
            _, _, theta_c, dc = ext['C', N, seed]
            return embed(H_P+predict_scaled(theta_c, dc, x, H_P*kh2(params, modes_ext)[:, :, None]), modes_ext, modes25)
        theta, d = ext['D-ext', N, seed]
        return embed(predict_scaled(theta, d, x), modes_ext, modes25)

    # ------------------------------------------------ 3. fresh test designs and load families (generated only now)
    test, test_ids, _ = sample_designs(TEST_COUNT, TEST_SEED, GENERATOR_VERSION)
    xt = design_coordinates(test)
    prior = [('pilot', P), *[(f'collocation s{s}', fit['colloc'][s]) for s in SEEDS],
             ('E4 fresh', np.load(ROOT/'results'/'e4_confirmation'/'fresh_test_set.npz')['parameters']),
             ('E5 fresh', np.load(ROOT/'results'/'e5_confirmation'/'fresh_test_set.npz')['parameters'])]
    for name, other in prior:
        assert np.min(np.linalg.norm(xt[:, None]-design_coordinates(other)[None], axis=2)) > 1e-6, f'overlap with {name}'
    modes25 = square_modes(TRUTH)
    Q = np.zeros((TEST_COUNT, 4, len(modes25)))
    patch_params = np.zeros((TEST_COUNT, 2, 3))          # [design, L3/L4, (xi0, eta0, sigma)]
    lookup25 = {tuple(m): i for i, m in enumerate(modes25)}
    for i in range(TEST_COUNT):
        for pair, q in design_loads(load_key(GENERATOR_VERSION, i, test_ids[i]), TEST_SEED)[0][2].items():
            Q[i, 0, lookup25[pair]] = q
        Q[i, 1] = np.where((modes25[:, 0] % 2 == 1) & (modes25[:, 1] % 2 == 1), 16/(np.pi**2*modes25[:, 0]*modes25[:, 1]), 0.0)
        for j, fam in enumerate(('L3', 'L4')):
            rng = np.random.default_rng([TEST_SEED, j+3, i])
            centre = rng.uniform(*PATCH[fam]['centre'], size=2)
            sigma = rng.uniform(*PATCH[fam]['sigma'])
            patch_params[i, j] = (*centre, sigma)
            Q[i, j+2] = patch_coefficients(centre, sigma, modes25)
    t0 = perf_counter()
    H_true = transfer(test, modes25, 'TSDT')
    H_fsdt = transfer(test, modes25, 'FSDT')
    print(f'truth computed ({perf_counter()-t0:.0f} s)', flush=True)
    np.savez_compressed(out/'h2_test_set.npz', parameters=test, design_ids=test_ids, modes=modes25, load_coefficients=Q,
                        families=np.array(list(FAMILIES)), patch_params=patch_params, H_tsdt=H_true, H_fsdt=H_fsdt,
                        seed=TEST_SEED, generator_version=GENERATOR_VERSION)

    def errors(H, fam):
        """Exact continuous relative L2 of the five scaled fields (modal orthogonality)."""
        k = list(FAMILIES).index(fam)
        true = Q[:, k, :, None]*H_true
        return np.linalg.norm((Q[:, k, :, None]*H-true).reshape(TEST_COUNT, -1), axis=1)/np.linalg.norm(true.reshape(TEST_COUNT, -1), axis=1)

    # identity of the 5x5 parts on the test designs (each mode fitted independently)
    for seed in SEEDS:
        for N in BUDGETS:
            HA, HC, HB = predict('A', N, seed, test), predict('C', N, seed, test), predict('B', N, seed, test)
            idx5 = [lookup25[tuple(m)] for m in modes5]
            assert np.allclose(HC[:, idx5], HA[:, idx5], rtol=1e-9, atol=1e-14) and np.allclose(HB[:, idx5], HA[:, idx5], rtol=1e-12, atol=0)
            assert np.allclose(predict('D-ext', N, seed, test)[:, idx5], predict('D', N, seed, test)[:, idx5], rtol=1e-9, atol=1e-14)

    # ------------------------------------------------ 4. single evaluation
    runs, errs = [], {}
    for fam in FAMILIES:
        for arm in ARMS:
            for N in ((0,) if arm == 'P-ext' else BUDGETS):
                for seed in SEEDS:
                    e = errors(predict(arm, N, seed, test), fam)
                    errs[fam, arm, N, seed] = e
                    runs.append(dict(family=fam, arm=arm, budget=N, seed=seed, test=summary(e)))
    trunc = {}
    for fam in FAMILIES:
        for name, k in (('TSDT truncated to 5x5', 5), ('TSDT truncated to 15x15', 15)):
            keep = ((modes25[:, 0] <= k) & (modes25[:, 1] <= k))[None, :, None]
            trunc[fam, name] = errors(H_true*keep, fam)
        trunc[fam, 'exact FSDT'] = errors(H_fsdt, fam)
    references = {fam: {name: summary(trunc[fam, name]) for name in ('TSDT truncated to 5x5', 'TSDT truncated to 15x15', 'exact FSDT')}
                  for fam in FAMILIES}

    def seed_mean(fam, arm, N):
        return np.mean([errs[fam, arm, N, s] for s in SEEDS], axis=0)

    def mean_stat(fam, arm, N, stat):
        return float(np.mean([r['test'][f'{stat}_percent'] for r in runs if (r['family'], r['arm'], r['budget']) == (fam, arm, N)]))

    def paired(reference, candidate):
        diff = reference-candidate
        lo, hi = bootstrap_ci(diff, 'median')[0]
        return dict(median_difference_percent=float(100*np.median(diff)), ci95_percent=[float(100*lo), float(100*hi)],
                    confirmed=bool(lo > 0), better_designs=int(np.sum(diff > 0)), designs=int(len(diff)))
    table = {fam: {arm: {str(N): dict(median_percent_mean=mean_stat(fam, arm, N, 'median'), p95_percent_mean=mean_stat(fam, arm, N, 'p95'))
                         for N in ((0,) if arm == 'P-ext' else BUDGETS)} for arm in ARMS} for fam in FAMILIES}

    def meets(fam):
        return (table[fam]['C'][str(PRIMARY_N)]['median_percent_mean'] <= TARGET['median_percent']
                and table[fam]['C'][str(PRIMARY_N)]['p95_percent_mean'] <= TARGET['p95_percent'])
    new = ('L2', 'L3', 'L4')
    Q3 = {fam: paired(seed_mean(fam, 'D-ext', PRIMARY_N), seed_mean(fam, 'C', PRIMARY_N)) for fam in new}
    Q4 = {fam: paired(seed_mean(fam, 'P-ext', 0), seed_mean(fam, 'C', PRIMARY_N)) for fam in new}
    Q5 = paired(seed_mean('L4', 'A', PRIMARY_N), seed_mean('L4', 'C', PRIMARY_N))
    decisions = dict(
        Q1=dict(L1=table['L1']['C'][str(PRIMARY_N)], confirmed=meets('L1')),
        Q2=dict(per_family={fam: dict(**table[fam]['C'][str(PRIMARY_N)], meets=meets(fam)) for fam in new},
                confirmed=all(meets(fam) for fam in new)),
        Q3=dict(per_family=Q3, confirmed=all(v['confirmed'] for v in Q3.values())),
        Q4=dict(per_family=Q4, confirmed=all(v['confirmed'] for v in Q4.values())),
        Q5=dict(L4=Q5, confirmed=Q5['confirmed']))
    # secondary: L4 error vs patch width in plate thicknesses
    width_in_thickness = patch_params[:, 1, 2]*test[:, 1]
    secondary = dict(
        A_vs_C_all_families={fam: paired(seed_mean(fam, 'A', PRIMARY_N), seed_mean(fam, 'C', PRIMARY_N)) for fam in FAMILIES},
        L4_by_patch_width={label: dict(designs=int(sel.sum()),
                                       C_median_percent=float(100*np.median(seed_mean('L4', 'C', PRIMARY_N)[sel])),
                                       C_p95_percent=float(100*np.quantile(seed_mean('L4', 'C', PRIMARY_N)[sel], .95)),
                                       exact_FSDT_median_percent=float(100*np.median(trunc['L4', 'exact FSDT'][sel])))
                           for label, sel in (('width < 1 thickness', width_in_thickness < 1),
                                              ('width 1-3 thicknesses', (width_in_thickness >= 1) & (width_in_thickness < 3)),
                                              ('width >= 3 thicknesses', width_in_thickness >= 3))},
        worst_C_L4=[dict(error_percent=float(100*seed_mean('L4', 'C', PRIMARY_N)[i]), g=float(test[i, 0]), a_over_h=float(test[i, 1]),
                         b_over_a=float(test[i, 2]), sigma=float(patch_params[i, 1, 2]))
                    for i in np.argsort(seed_mean('L4', 'C', PRIMARY_N))[::-1][:5]])

    # ------------------------------------------------ figures
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4), layout='constrained')
    arms_plot = ['P-ext', 'D-ext', 'A', 'B', 'C']
    colors = {'P-ext': 'tab:purple', 'D-ext': 'tab:orange', 'A': 'tab:gray', 'B': 'tab:cyan', 'C': 'tab:green'}
    xs = np.arange(4)
    for ax, stat, target in zip(axes, ('median', 'p95'), (TARGET['median_percent'], TARGET['p95_percent'])):
        for j, arm in enumerate(arms_plot):
            vals = [table[f][arm]['0' if arm == 'P-ext' else str(PRIMARY_N)][f'{stat}_percent_mean'] for f in FAMILIES]
            ax.bar(xs+(j-2)*0.16, vals, width=0.16, color=colors[arm], label=ARMS[arm]+('' if arm == 'P-ext' else ', N=10'))
        for name, mk in (('TSDT truncated to 5x5', '_'), ('exact FSDT', 'x')):
            ax.scatter(xs, [references[f][name][f'{stat}_percent'] for f in FAMILIES], marker=mk, s=120, color='k',
                       zorder=3, label=f'reference: {name}')
        ax.axhline(target, color='r', ls=':', lw=.8, label=f'target {target}%')
        ax.set(xticks=xs, xticklabels=[f'{f}\n{FAMILIES[f][:22]}' for f in FAMILIES], yscale='log',
               ylabel=f'{"Median" if stat == "median" else "95th percentile"} error vs TSDT (%)',
               title=f'H2: {TEST_COUNT} fresh designs per load family ({"median" if stat == "median" else "95th percentile"})')
        ax.grid(alpha=.3, axis='y', which='both'); ax.legend(fontsize=6.3)
    fig.savefig(out/'h2_load_families.png', dpi=170); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.8, 4.3), layout='constrained')
    ax.scatter(width_in_thickness, 100*trunc['L4', 'exact FSDT'], marker='x', c='k', s=14, label='exact FSDT (model-form gap)')
    ax.scatter(width_in_thickness, 100*seed_mean('L4', 'A', PRIMARY_N), marker='v', s=14, color='tab:gray', alpha=.6, label=ARMS['A']+', N=10')
    ax.scatter(width_in_thickness, 100*seed_mean('L4', 'C', PRIMARY_N), marker='^', s=16, color='tab:green', alpha=.85, label=ARMS['C']+', N=10')
    ax.axvline(1, color='k', ls=':', lw=.8)
    ax.set(xscale='log', yscale='log', xlabel='patch width sigma*a/h (in plate thicknesses)', ylabel='error vs TSDT (%)',
           title='H2: narrow patch (L4), error vs patch width')
    ax.grid(alpha=.3, which='both'); ax.legend(fontsize=7)
    fig.savefig(out/'h2_narrow_patch_width.png', dpi=170); plt.close(fig)

    np.savez_compressed(out/'h2_errors.npz', **{f'{f}_{a}_{n}_s{s}': e for (f, a, n, s), e in errs.items()},
                        **{f'{f}_ref_{name.replace(" ", "_")}': e for (f, name), e in trunc.items()})
    manifest = dict(experiment='H2 generalization to unseen load shapes', created_utc=datetime.now(timezone.utc).isoformat(),
                    preregistration=dict(path='docs/H2_PREREGISTRATION.md', sha256=PREREGISTRATION_SHA256),
                    development_reference=dict(path=str(args.development), sha256=sha256(args.development)),
                    test_set=dict(count=TEST_COUNT, seed=TEST_SEED, generator=GENERATOR_VERSION, truth_modes=f'{TRUTH}x{TRUTH}',
                                  extended_modes=f'{EXT}x{EXT}', file='h2_test_set.npz'),
                    families=FAMILIES, patch=PATCH, arms=ARMS, budgets=BUDGETS, seeds=SEEDS,
                    metric='exact continuous relative L2 of five scaled generalized fields from modal coefficients',
                    runs=runs, table=table, references=references, decisions=decisions, secondary=secondary,
                    total_seconds=perf_counter()-start,
                    environment=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                                     matplotlib=matplotlib.__version__, platform=platform.platform()),
                    source_sha256={str(p.relative_to(ROOT)): sha256(p) for p in
                                   [*sorted((ROOT/'fgplate').glob('*.py')), ROOT/'experiments'/'run_e5.py', Path(__file__)]})
    write_json(out/'manifest.json', manifest)
    write_json(out/'checksums.json', {p.name: sha256(p) for p in sorted(out.iterdir()) if p.is_file() and p.name != 'checksums.json'})
    print(json.dumps({k: v['confirmed'] for k, v in decisions.items()}, indent=1), flush=True)


if __name__ == '__main__':
    main()
