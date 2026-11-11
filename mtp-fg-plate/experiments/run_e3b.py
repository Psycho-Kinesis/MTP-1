"""E3b: imperfect physics (FSDT energy) + higher-fidelity labels (TSDT), at matched capacity.

Run from the project root:
    python experiments/run_e3b.py --output results/e3b_model_form

Design parameters, mixture loads and the 140/30/30 split come from results/pilot_200
(read only). The TARGET is now the Reddy TSDT response of every design (fgplate.tsdt,
verified against an independent symbolic Ritz derivation), standing in for an expensive
higher-fidelity model. The physics loss is FSDT, which differs from TSDT by a
model-form error that is largest for thick plates.

  P   physics only        FSDT energy at 560 unlabelled collocation designs (no labels; biased)
  D   data only           TSDT labels at N training designs
  Hy  hybrid              TSDT labels + weighted FSDT energy (shared polynomial coefficients)
  Hc  physics+correction  P's prediction + a polynomial correction fitted to TSDT residuals
  references (not trained): FSDT exact per design (pure model-form gap); TSDT exact (zero)

All arms share the model class of fgplate.transfer (tensor-Legendre polynomials on the fixed
domain box) with the per-component ('block') relative ridge. Hyperparameters are chosen on
validation only, with the tail-aware score fixed below; the test partition is evaluated
once, after all choices are frozen. The rules were fixed before the first run, after
measuring the FSDT-TSDT gap on TRAINING designs only (five-field median 0.36%, p95 5.6%).
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
from fgplate.fields import analytic_basis, reconstruct_analytic, relative_l2
from fgplate.pilot import quantize
from fgplate.transfer import assemble_data, assemble_physics, design_coordinates, features, mode_matrices, solve_modes
from fgplate.tsdt import transfer

MIXTURE = 2
DEGREES = (2, 3, 4, 5, 6, 7)
RIDGES = (1e-12, 1e-10, 1e-8, 1e-6, 1e-4)
PHYSICS_WEIGHTS = (0.001, 0.01, 0.1, 1.0, 10.0, 100.0)
CORRECTION_DEGREES = (0, 1, 2, 3, 4)
LABEL_BUDGETS = (10, 20, 40, 70, 140)
COLLOCATION_M = 560
SEEDS = (11, 22, 33)
SCOPE = 'block'
TARGET = dict(median_percent=1.0, p95_percent=3.0)
RULES = dict(
    target_model='Reddy TSDT (fgplate.tsdt), same designs, loads and split as pilot_200',
    metric='per-design relative L2 over all five scaled generalized fields vs TSDT (primary); w-only reported',
    selection=('tail-aware: minimize max(validation median / 1%, validation p95 / 3%); ties by median. '
               'Replaces the median-first rule of E3a, whose hybrid tails grew with labels.'),
    target=f'validation median <= {TARGET["median_percent"]}% and p95 <= {TARGET["p95_percent"]}% (mean over seeds)',
    Q1=('bias correction: at N = 10, 20, 40, arm X (Hy or Hc) corrects the physics bias if the 95% bootstrap '
        'interval of the median per-design paired difference (P error - X error, seed-averaged, 30 validation '
        'designs) lies above 0'),
    Q2=('label efficiency under model-form error: same test with (D error - X error) at N = 10, 20, 40'),
    Q3='smallest label budget meeting the target, per arm',
)
ARMS = dict(P='physics only (FSDT energy, no labels)', D='data only (TSDT labels)',
            Hy='hybrid (TSDT labels + FSDT energy)', Hc='physics + learned correction')


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def summary(errors):
    e = 100*np.asarray(errors)
    return dict(median_percent=float(np.median(e)), p95_percent=float(np.quantile(e, .95)),
                worst_percent=float(e.max()), mean_percent=float(e.mean()),
                median_percent_ci95=(100*bootstrap_ci(errors, 'median')[0]).tolist(), cases=int(len(e)))


def score(errors):
    e = 100*np.asarray(errors)
    med, p95 = np.median(e), np.quantile(e, .95)
    return (max(med/TARGET['median_percent'], p95/TARGET['p95_percent']), med)


def collocation_designs(count, seed):
    unit = qmc.Halton(d=3, scramble=True, seed=seed).random(count)
    t = qmc.scale(unit, [0, np.log(5), .5], [np.log(9), np.log(100), 2])
    return quantize(np.column_stack((np.expm1(t[:, 0]), np.exp(t[:, 1]), t[:, 2])))


def predict(theta, degree, x):
    return np.einsum('np,kcp->nkc', features(x, degree), theta)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--pilot', type=Path, default=ROOT/'results'/'pilot_200')
    parser.add_argument('--output', type=Path, default=ROOT/'results'/'e3b_model_form')
    args = parser.parse_args()
    out = args.output.resolve()
    if out.exists() and any(out.iterdir()):
        parser.error('output directory is not empty; choose a new experiment directory')
    out.mkdir(parents=True, exist_ok=True)
    start_all = perf_counter()

    # ---------------------------------------------------------------- data (read only) and TSDT targets
    pilot = args.pilot.resolve()
    recorded = json.loads((pilot/'checksums.json').read_text())
    assert sha256(pilot/'dataset.npz') == recorded['dataset.npz'], 'pilot dataset differs from its recorded checksum'
    with np.load(pilot/'dataset.npz', allow_pickle=False) as z:
        P, split, ids = z['parameters'], z['split'], z['design_ids']
        Q, modes_all, xi, eta = z['pressure_coefficients'][:, MIXTURE], z['mode_indices'], z['xi'], z['eta']
    active = np.flatnonzero(np.any(Q != 0, axis=0))
    modes, c = modes_all[active], Q[:, active]
    basis = analytic_basis(modes, xi, eta)
    t0 = perf_counter()
    H_tsdt = transfer(P, modes, 'TSDT')                      # targets (labels where allowed)
    H_fsdt = transfer(P, modes, 'FSDT')                      # reference only: exact imperfect physics
    label_seconds = perf_counter()-t0
    np.savez_compressed(out/'transfer_references.npz', design_ids=ids, modes=modes, H_tsdt=H_tsdt, H_fsdt=H_fsdt)
    x = design_coordinates(P)
    train, val, test = (np.flatnonzero(split == s) for s in ('train', 'validation', 'test'))

    def fields(H, idx):
        return reconstruct_analytic(np.transpose(H*c[idx, :, None], (0, 2, 1)), basis)
    truth = {name: fields(H_tsdt[idx], idx) for name, idx in (('val', val), ('test', test))}

    def errors(H, idx, part, w_only=False):
        F = fields(H, idx)
        return relative_l2(F[:, 2], truth[part][:, 2]) if w_only else relative_l2(F, truth[part])

    gap = {part: dict(five_field=summary(errors(H_fsdt[idx], idx, part)), w_only=summary(errors(H_fsdt[idx], idx, part, True)))
           for part, idx in (('val', val), ('test', test))}
    gap['train_five_field'] = summary(relative_l2(fields(H_fsdt[train], train), fields(H_tsdt[train], train)))
    print('FSDT-vs-TSDT gap (validation, five fields):', {k: round(v, 3) for k, v in gap['val']['five_field'].items()
                                                         if k in ('median_percent', 'p95_percent')}, flush=True)

    # ---------------------------------------------------------------- physics normal equations (FSDT)
    colloc, phys = {}, {}
    for seed in SEEDS:
        designs = collocation_designs(COLLOCATION_M, 1000+seed)
        xc = design_coordinates(designs)
        for idx in (val, test):
            assert np.min(np.linalg.norm(xc[:, None]-x[idx][None], axis=2)) > 1e-6
        K, f = mode_matrices(designs, modes)
        colloc[seed] = designs
        for d in DEGREES:
            phys[seed, d] = assemble_physics(xc, K, f, d)
    print(f'physics assembled ({perf_counter()-start_all:.0f} s)', flush=True)

    runs, chosen = [], {}

    def record(arm, budget, seed, hp, H_val, H_test_fn):
        e = errors(H_val, val, 'val')
        chosen[arm, budget, seed] = H_test_fn
        runs.append(dict(arm=arm, budget=budget, seed=seed, selected=hp, validation=summary(e),
                         validation_w_only=summary(errors(H_val, val, 'val', True)), validation_errors=e.tolist()))

    def best_of(candidates):
        best = None
        for hp, theta, degree in candidates:
            H_val = predict(theta, degree, x[val])
            key = score(errors(H_val, val, 'val'))
            if best is None or key < best[0]:
                best = (key, hp, theta, degree, H_val)
        return best[1:]

    for seed in SEEDS:
        # P: physics only
        hp, theta_p, deg_p, Hv = best_of(((dict(degree=d, ridge=r), solve_modes(*phys[seed, d], d, r, SCOPE), d)
                                          for d in DEGREES for r in RIDGES))
        base = dict(theta=theta_p, degree=deg_p)
        record('P', 0, seed, hp, Hv, lambda idx, t=theta_p, d=deg_p: predict(t, d, x[idx]))
        order = np.random.default_rng(seed).permutation(train)          # identical subsets to E2/E3a
        for N in LABEL_BUDGETS:
            sub = np.sort(order[:N])
            data = {d: assemble_data(x[sub], H_tsdt[sub], d)[:2] for d in DEGREES}
            # D: data only
            hp, th, dg, Hv = best_of(((dict(degree=d, ridge=r), solve_modes(*data[d], d, r, SCOPE), d)
                                      for d in DEGREES for r in RIDGES))
            record('D', N, seed, hp, Hv, lambda idx, t=th, d=dg: predict(t, d, x[idx]))
            # Hy: hybrid, shared coefficients
            hp, th, dg, Hv = best_of(((dict(degree=d, ridge=r, physics_weight=w),
                                       solve_modes(data[d][0]+w*phys[seed, d][0], data[d][1]+w*phys[seed, d][1], d, r, SCOPE), d)
                                      for d in DEGREES for r in RIDGES for w in PHYSICS_WEIGHTS))
            record('Hy', N, seed, hp, Hv, lambda idx, t=th, d=dg: predict(t, d, x[idx]))
            # Hc: physics prediction + correction fitted to residual labels
            residual = H_tsdt[sub]-predict(base['theta'], base['degree'], x[sub])
            best = None
            for dc in CORRECTION_DEGREES:
                A, b, _ = assemble_data(x[sub], residual, dc)
                for r in RIDGES:
                    th_c = solve_modes(A, b, dc, r, SCOPE)
                    Hv = predict(base['theta'], base['degree'], x[val])+predict(th_c, dc, x[val])
                    key = score(errors(Hv, val, 'val'))
                    if best is None or key < best[0]:
                        best = (key, dict(base_degree=base['degree'], correction_degree=dc, ridge=r), th_c, dc, Hv)
            _, hp, th_c, dc, Hv = best
            record('Hc', N, seed, hp, Hv, lambda idx, t=th_c, d=dc, b0=base: predict(b0['theta'], b0['degree'], x[idx])+predict(t, d, x[idx]))
        print(f'seed {seed} done ({perf_counter()-start_all:.0f} s)', flush=True)

    # ---------------------------------------------------------------- aggregation, decisions (validation only)
    def curve(arm, budgets):
        result = {}
        for n in budgets:
            rows = [r for r in runs if r['arm'] == arm and r['budget'] == n]
            med = [r['validation']['median_percent'] for r in rows]
            p95 = [r['validation']['p95_percent'] for r in rows]
            wmed = [r['validation_w_only']['median_percent'] for r in rows]
            result[str(n)] = dict(median_percent_mean=float(np.mean(med)), median_percent_range=[min(med), max(med)],
                                  p95_percent_mean=float(np.mean(p95)), p95_percent_range=[min(p95), max(p95)],
                                  w_only_median_percent_mean=float(np.mean(wmed)), selected=[r['selected'] for r in rows])
        return result
    curves = dict(P=curve('P', (0,)), D=curve('D', LABEL_BUDGETS), Hy=curve('Hy', LABEL_BUDGETS), Hc=curve('Hc', LABEL_BUDGETS))

    def meets(entry):
        return entry['median_percent_mean'] <= TARGET['median_percent'] and entry['p95_percent_mean'] <= TARGET['p95_percent']

    def seed_mean(arm, n):
        return np.mean([r['validation_errors'] for r in runs if r['arm'] == arm and r['budget'] == n], axis=0)

    def paired(reference, arm, n):
        diff = reference-seed_mean(arm, n)
        lo, hi = bootstrap_ci(diff, 'median')[0]
        return dict(median_difference_percent=float(100*np.median(diff)), ci95_percent=[float(100*lo), float(100*hi)],
                    supported=bool(lo > 0), better_designs=int(np.sum(diff > 0)), designs=int(len(diff)))
    decisions = dict(rules=RULES,
                     physics_only_meets_target=meets(curves['P']['0']),
                     smallest_N_meeting_target={arm: next((n for n in LABEL_BUDGETS if meets(curves[arm][str(n)])), None)
                                                for arm in ('D', 'Hy', 'Hc')})
    p_err = seed_mean('P', 0)
    decisions['Q1_bias_correction'] = {arm: {str(n): paired(p_err, arm, n) for n in LABEL_BUDGETS} for arm in ('Hy', 'Hc')}
    decisions['Q2_label_efficiency'] = {arm: {str(n): paired(seed_mean('D', n), arm, n) for n in LABEL_BUDGETS}
                                        for arm in ('Hy', 'Hc')}
    for q in ('Q1_bias_correction', 'Q2_label_efficiency'):
        decisions[q+'_supported_at_N_10_20_40'] = {arm: all(decisions[q][arm][str(n)]['supported'] for n in (10, 20, 40))
                                                   for arm in ('Hy', 'Hc')}

    # ---------------------------------------------------------------- single final test evaluation
    for run in runs:
        H = chosen[run['arm'], run['budget'], run['seed']](test)
        run['test'] = summary(errors(H, test, 'test'))
        run['test_w_only'] = summary(errors(H, test, 'test', True))
    test_curves = {arm: {str(n): dict(median_percent_mean=float(np.mean([r['test']['median_percent'] for r in runs if r['arm'] == arm and r['budget'] == n])),
                                      p95_percent_mean=float(np.mean([r['test']['p95_percent'] for r in runs if r['arm'] == arm and r['budget'] == n])))
                         for n in ((0,) if arm == 'P' else LABEL_BUDGETS)} for arm in ARMS}

    # ---------------------------------------------------------------- figures
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), layout='constrained')
    for ax, stat, target in zip(axes, ('median', 'p95'), (TARGET['median_percent'], TARGET['p95_percent'])):
        for arm, marker in (('D', 'o'), ('Hy', 's'), ('Hc', '^')):
            mean = [curves[arm][str(n)][f'{stat}_percent_mean'] for n in LABEL_BUDGETS]
            lo = [curves[arm][str(n)][f'{stat}_percent_range'][0] for n in LABEL_BUDGETS]
            hi = [curves[arm][str(n)][f'{stat}_percent_range'][1] for n in LABEL_BUDGETS]
            ax.plot(LABEL_BUDGETS, mean, marker=marker, label=ARMS[arm])
            ax.fill_between(LABEL_BUDGETS, lo, hi, alpha=.15)
        ax.axhline(curves['P']['0'][f'{stat}_percent_mean'], color='g', ls='--', lw=1.2,
                   label=f'physics only, 0 labels: {curves["P"]["0"][f"{stat}_percent_mean"]:.2f}%')
        ax.axhline(gap['val']['five_field'][f'{stat}_percent'], color='k', ls='-.', lw=1,
                   label=f'exact FSDT (model-form gap): {gap["val"]["five_field"][f"{stat}_percent"]:.2f}%')
        ax.axhline(target, color='r', ls=':', lw=.8, label=f'target {target}%')
        ax.set(xscale='log', yscale='log', xticks=LABEL_BUDGETS, xlabel='TSDT-labelled training designs',
               ylabel=f'{"Median" if stat == "median" else "95th percentile"} validation error vs TSDT (%)',
               title=f'E3b: {"median" if stat == "median" else "95th percentile"} over 30 designs (3 seeds)')
        ax.set_xticklabels([str(b) for b in LABEL_BUDGETS]); ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.grid(alpha=.3, which='both'); ax.legend(fontsize=7)
    fig.savefig(out/'e3b_label_curves.png', dpi=170); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4.2), layout='constrained')
    s11 = {arm: next(r for r in runs if r['arm'] == arm and r['seed'] == SEEDS[0] and r['budget'] == (0 if arm == 'P' else 20))
           for arm in ('P', 'D', 'Hc')}
    ax.scatter(P[val, 1], 100*errors(H_fsdt[val], val, 'val'), marker='x', c='k', label='exact FSDT (model-form gap)')
    for arm, marker in (('P', 'o'), ('D', 's'), ('Hc', '^')):
        ax.scatter(P[val, 1], 100*np.array(s11[arm]['validation_errors']), marker=marker, alpha=.7,
                   label=f'{ARMS[arm]}{"" if arm == "P" else ", N=20"} (seed {SEEDS[0]})')
    ax.set(xscale='log', yscale='log', xlabel='slenderness a/h', ylabel='validation error vs TSDT (%)',
           title='E3b: where the error is (30 validation designs)')
    ax.grid(alpha=.3, which='both'); ax.legend(fontsize=7)
    fig.savefig(out/'e3b_error_vs_thickness.png', dpi=170); plt.close(fig)

    # ---------------------------------------------------------------- persist
    np.savez_compressed(out/'e3b_errors.npz', validation_ids=ids[val], test_ids=ids[test],
                        **{f'{r["arm"]}_{r["budget"]}_s{r["seed"]}_validation': np.array(r['validation_errors']) for r in runs},
                        **{f'collocation_designs_s{s}': colloc[s] for s in SEEDS})
    for r in runs:
        r.pop('validation_errors')
    manifest = dict(experiment='E3b imperfect physics (FSDT) + higher-fidelity labels (TSDT) at matched capacity',
                    created_utc=datetime.now(timezone.utc).isoformat(),
                    pilot=dict(path=str(args.pilot), dataset_sha256=recorded['dataset.npz']),
                    arms=ARMS, rules=RULES, model_form_gap=gap, ridge_scope=SCOPE,
                    grids=dict(degrees=DEGREES, ridges=RIDGES, physics_weights=PHYSICS_WEIGHTS, correction_degrees=CORRECTION_DEGREES),
                    label_budgets=LABEL_BUDGETS, collocation_M=COLLOCATION_M, seeds=SEEDS,
                    partitions=dict(train=len(train), validation=len(val), test=len(test)),
                    runs=runs, validation_curves=curves, decisions=decisions, test_curves=test_curves,
                    label_generation_seconds=label_seconds, total_seconds=perf_counter()-start_all,
                    note=('TSDT stands in for a higher-fidelity model; it is itself a plate theory, not 3D elasticity. '
                          'Five generalized fields are compared; in both theories phi is du/dz at the mid-surface.'),
                    environment=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                                     matplotlib=matplotlib.__version__, platform=platform.platform()),
                    source_sha256={str(p.relative_to(ROOT)): sha256(p) for p in
                                   [*sorted((ROOT/'fgplate').glob('*.py')), Path(__file__)]})
    write_json(out/'manifest.json', manifest)
    write_json(out/'checksums.json', {p.name: sha256(p) for p in sorted(out.iterdir()) if p.is_file() and p.name != 'checksums.json'})
    print(json.dumps(dict(curves={a: {n: (round(v['median_percent_mean'], 3), round(v['p95_percent_mean'], 2)) for n, v in cc.items()}
                                  for a, cc in curves.items()},
                          decisions={k: v for k, v in decisions.items() if k != 'rules' and not k.startswith('Q')}), indent=1), flush=True)


if __name__ == '__main__':
    main()
