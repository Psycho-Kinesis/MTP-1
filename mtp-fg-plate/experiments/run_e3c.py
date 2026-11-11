"""E3c: physics-structured correction and physics-guided label placement (TSDT target, FSDT physics).

Run from the project root:
    python experiments/run_e3c.py --output results/e3c_structured

Same data, split, seeds, budgets, collocation, model class, ridge scope and tail-aware
selection as E3b. The physics-only arm P and the data-only arm D are rebuilt and asserted
to reproduce results/e3b_model_form exactly. New: six correction variants on top of P,

    correction form  G  global polynomial (= E3b's Hc)
                     A  (k h)^2 * polynomial                     additive, vanishes for thin plates
                     M  H_P * (k h)^2 * polynomial               multiplicative relative correction
    label placement  R  random (identical subsets to E2/E3a/E3b)
                     S  sampled without replacement with probability proportional to the
                        FSDT shear-compliance share psi of the (1,1) mode (label-free)

k^2 = pi^2 (m^2 + n^2 (a/b)^2) per mode; psi = [1/(As k^2)] / [1/(D* k^4) + 1/(As k^2)].
Pre-registered (fixed after training-only diagnostics, before any E3c fit):
PRIMARY = M-S. Q1: M-S beats P at N = 10, 20, 40. Q2: M-S beats D at N = 10, 20, 40.
The other variants and factor contrasts are secondary. The test partition is evaluated once.
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
from scipy.stats import qmc, spearmanr
import matplotlib
import matplotlib.ticker
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from fgplate.baseline import bootstrap_ci
from fgplate.fields import analytic_basis, reconstruct_analytic, relative_l2
from fgplate.material import al_al2o3, plate_stiffness
from fgplate.pilot import quantize
from fgplate.transfer import (assemble_data, assemble_physics, design_coordinates, mode_matrices, predict_scaled,
                              solve_modes)
from fgplate.tsdt import transfer

MIXTURE = 2
DEGREES = (2, 3, 4, 5, 6, 7)
RIDGES = (1e-12, 1e-10, 1e-8, 1e-6, 1e-4)
CORRECTION_DEGREES = (0, 1, 2, 3, 4)
LABEL_BUDGETS = (10, 20, 40, 70, 140)
COLLOCATION_M = 560
SEEDS = (11, 22, 33)
SCOPE = 'block'
FORMS = ('G', 'A', 'M')
PLACEMENTS = ('R', 'S')
PRIMARY = 'M-S'
TARGET = dict(median_percent=1.0, p95_percent=3.0)
RULES = dict(
    target_model='Reddy TSDT; physics loss FSDT (as E3b)',
    selection='tail-aware: minimize max(validation median / 1%, validation p95 / 3%); ties by median (as E3b)',
    primary=f'{PRIMARY}: multiplicative (k h)^2 correction with shear-share-weighted label placement',
    Q1=f'{PRIMARY} beats physics only P: 95% bootstrap CI of median paired difference (P - {PRIMARY}) > 0 at N = 10, 20, 40',
    Q2=f'{PRIMARY} beats data only D (random labels): CI of median paired difference (D - {PRIMARY}) > 0 at N = 10, 20, 40',
    secondary='Q1/Q2 for the other five variants; placement contrasts X-R vs X-S and form contrasts G vs A/M; Q3 smallest N meeting target',
    consistency='P, D and G-R must reproduce E3b (P, D, Hc) validation metrics',
)
NAMES = {'G-R': 'global correction, random labels (E3b Hc)', 'G-S': 'global correction, shear-guided labels',
         'A-R': '(kh)^2 correction, random labels', 'A-S': '(kh)^2 correction, shear-guided labels',
         'M-R': 'H_P(kh)^2 correction, random labels', 'M-S': 'H_P(kh)^2 correction, shear-guided labels (primary)'}


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


def shear_share(parameters):
    """FSDT shear-compliance share of the (1,1) mode; label-free (stiffness integrals only)."""
    out = []
    for g, ratio, aspect in parameters:
        S = plate_stiffness(al_al2o3(float(g)), 1/ratio)
        k2 = np.pi**2*(1+1/aspect**2)
        bend, shear = 1/((S.D11-S.B11**2/S.A11)*k2**2), 1/(S.As*k2)
        out.append(shear/(bend+shear))
    return np.array(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--pilot', type=Path, default=ROOT/'results'/'pilot_200')
    parser.add_argument('--e3b', type=Path, default=ROOT/'results'/'e3b_model_form'/'manifest.json')
    parser.add_argument('--output', type=Path, default=ROOT/'results'/'e3c_structured')
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
        Q, modes_all, xi, eta = z['pressure_coefficients'][:, MIXTURE], z['mode_indices'], z['xi'], z['eta']
    active = np.flatnonzero(np.any(Q != 0, axis=0))
    modes, c = modes_all[active], Q[:, active]
    basis = analytic_basis(modes, xi, eta)
    H_tsdt, H_fsdt = transfer(P, modes, 'TSDT'), transfer(P, modes, 'FSDT')
    x = design_coordinates(P)
    train, val, test = (np.flatnonzero(split == s) for s in ('train', 'validation', 'test'))
    h, aspect = 1/P[:, 1], P[:, 2]
    kh2 = np.pi**2*(modes[:, 0][None]**2+(modes[:, 1][None]/aspect[:, None])**2)*h[:, None]**2   # [designs, modes]
    psi = shear_share(P)

    # pre-registration diagnostics, training designs only (recorded, not used for fitting)
    rel = np.linalg.norm(H_tsdt[train]-H_fsdt[train], axis=2)/np.linalg.norm(H_tsdt[train], axis=2)
    ratio = rel/kh2[train]

    def fields(H, idx):
        return reconstruct_analytic(np.transpose(H*c[idx, :, None], (0, 2, 1)), basis)
    gap_train = relative_l2(fields(H_fsdt[train], train), fields(H_tsdt[train], train))
    preregistration = dict(
        note='computed on the 140 training designs only, before any E3c fit',
        kh2_scaling=dict(corr_log_rel_log_kh2=float(np.corrcoef(np.log(rel.ravel()), np.log(kh2[train].ravel()))[0, 1]),
                         rel_over_kh2_p05_p95=[float(np.quantile(ratio, .05)), float(np.quantile(ratio, .95))]),
        shear_share=dict(spearman_with_five_field_gap=float(spearmanr(psi[train], gap_train).statistic),
                         range=[float(psi[train].min()), float(psi[train].max())]))

    truth = {'val': fields(H_tsdt[val], val), 'test': fields(H_tsdt[test], test)}
    part_idx = {'val': val, 'test': test}

    def errors(H, part):
        return relative_l2(fields(H, part_idx[part]), truth[part])

    # ---------------------------------------------------------------- physics normal equations (as E3b)
    phys = {}
    for seed in SEEDS:
        designs = collocation_designs(COLLOCATION_M, 1000+seed)
        xc = design_coordinates(designs)
        for idx in (val, test):
            assert np.min(np.linalg.norm(xc[:, None]-x[idx][None], axis=2)) > 1e-6
        K, f = mode_matrices(designs, modes)
        for d in DEGREES:
            phys[seed, d] = assemble_physics(xc, K, f, d)
    print(f'physics assembled ({perf_counter()-start_all:.0f} s)', flush=True)

    e3b = json.loads(args.e3b.read_text())
    e3b_runs = {(r['arm'], r['budget'], r['seed']): r for r in e3b['runs']}

    def same_as_e3b(key, validation):
        ref = e3b_runs[key]['validation']
        return all(np.isclose(validation[k], ref[k], rtol=1e-9, atol=1e-12) for k in ('median_percent', 'p95_percent', 'worst_percent'))

    runs, consistency, predictors = [], {}, {}

    def record(arm, budget, seed, hp, H_val, test_fn, placement=None):
        e = errors(H_val, 'val')
        predictors[arm, budget, seed] = test_fn
        runs.append(dict(arm=arm, budget=budget, seed=seed, selected=hp, validation=summary(e),
                         validation_errors=e.tolist(), **({'placement_designs': placement} if placement else {})))
        return runs[-1]['validation']

    def select(candidates):
        best = None
        for hp, H_val, payload in candidates:
            key = score(errors(H_val, 'val'))
            if best is None or key < best[0]:
                best = (key, hp, H_val, payload)
        return best[1:]

    for seed in SEEDS:
        # P: physics only (identical to E3b)
        cand = []
        for d in DEGREES:
            for r in RIDGES:
                th = solve_modes(*phys[seed, d], d, r, SCOPE)
                cand.append((dict(degree=d, ridge=r), predict_scaled(th, d, x[val]), (th, d)))
        hp, Hv, (th_p, d_p) = select(cand)
        consistency[f'P_s{seed}'] = same_as_e3b(('P', 0, seed), record('P', 0, seed, hp, Hv, lambda idx, t=th_p, d=d_p: predict_scaled(t, d, x[idx])))
        H_P = predict_scaled(th_p, d_p, x)                                  # label-free base prediction, all designs
        prefactors = {'G': None, 'A': np.broadcast_to(kh2[:, :, None], H_P.shape), 'M': H_P*kh2[:, :, None]}
        rng_s = np.random.default_rng([seed, 7])
        keys = np.log(rng_s.random(len(train)))/psi[train]                  # Efraimidis-Spirakis weighted order
        orders = {'R': np.random.default_rng(seed).permutation(train), 'S': train[np.argsort(-keys)]}
        for N in LABEL_BUDGETS:
            # D: data only, random labels (identical to E3b)
            sub = np.sort(orders['R'][:N])
            data = {d: assemble_data(x[sub], H_tsdt[sub], d)[:2] for d in DEGREES}
            cand = []
            for d in DEGREES:
                for r in RIDGES:
                    th = solve_modes(*data[d], d, r, SCOPE)
                    cand.append((dict(degree=d, ridge=r), predict_scaled(th, d, x[val]), (th, d)))
            hp, Hv, (th, d) = select(cand)
            consistency[f'D_{N}_s{seed}'] = same_as_e3b(('D', N, seed), record('D', N, seed, hp, Hv, lambda idx, t=th, dd=d: predict_scaled(t, dd, x[idx])))
            # correction variants
            for placement in PLACEMENTS:
                sub = np.sort(orders[placement][:N])
                residual = H_tsdt[sub]-H_P[sub]
                for form in FORMS:
                    pref = prefactors[form]
                    cand = []
                    for dc in CORRECTION_DEGREES:
                        A, b, _ = assemble_data(x[sub], residual, dc, prefactor=None if pref is None else pref[sub])
                        for r in RIDGES:
                            th = solve_modes(A, b, dc, r, SCOPE)
                            Hv = H_P[val]+predict_scaled(th, dc, x[val], None if pref is None else pref[val])
                            cand.append((dict(base_degree=d_p, correction_degree=dc, ridge=r), Hv, (th, dc)))
                    hp, Hv, (th, dc) = select(cand)
                    arm = f'{form}-{placement}'
                    fn = (lambda idx, t=th, dd=dc, pf=pref: H_P[idx]+predict_scaled(t, dd, x[idx], None if pf is None else pf[idx]))
                    v = record(arm, N, seed, hp, Hv, fn, placement=ids[sub].tolist())
                    if arm == 'G-R':
                        consistency[f'G-R_{N}_s{seed}'] = same_as_e3b(('Hc', N, seed), v)
        print(f'seed {seed} done ({perf_counter()-start_all:.0f} s)', flush=True)
    assert all(consistency.values()), f'E3b reproduction failed: {[k for k, v in consistency.items() if not v]}'

    # ---------------------------------------------------------------- aggregation and decisions (validation)
    variants = [f'{f}-{p}' for f in FORMS for p in PLACEMENTS]

    def curve(arm, budgets):
        result = {}
        for n in budgets:
            rows = [r for r in runs if r['arm'] == arm and r['budget'] == n]
            med = [r['validation']['median_percent'] for r in rows]
            p95 = [r['validation']['p95_percent'] for r in rows]
            result[str(n)] = dict(median_percent_mean=float(np.mean(med)), median_percent_range=[min(med), max(med)],
                                  p95_percent_mean=float(np.mean(p95)), p95_percent_range=[min(p95), max(p95)],
                                  selected=[r['selected'] for r in rows])
        return result
    curves = {'P': curve('P', (0,)), 'D': curve('D', LABEL_BUDGETS), **{v: curve(v, LABEL_BUDGETS) for v in variants}}

    def seed_mean(arm, n):
        return np.mean([r['validation_errors'] for r in runs if r['arm'] == arm and r['budget'] == n], axis=0)

    def paired(reference, candidate):
        diff = reference-candidate
        lo, hi = bootstrap_ci(diff, 'median')[0]
        return dict(median_difference_percent=float(100*np.median(diff)), ci95_percent=[float(100*lo), float(100*hi)],
                    supported=bool(lo > 0), better_designs=int(np.sum(diff > 0)), designs=int(len(diff)))

    def meets(entry):
        return entry['median_percent_mean'] <= TARGET['median_percent'] and entry['p95_percent_mean'] <= TARGET['p95_percent']
    p_err = seed_mean('P', 0)
    decisions = dict(rules=RULES, consistency_with_e3b=all(consistency.values()),
                     Q1={v: {str(n): paired(p_err, seed_mean(v, n)) for n in LABEL_BUDGETS} for v in variants},
                     Q2={v: {str(n): paired(seed_mean('D', n), seed_mean(v, n)) for n in LABEL_BUDGETS} for v in variants},
                     Q3={v: next((n for n in LABEL_BUDGETS if meets(curves[v][str(n)])), None) for v in ['D', *variants]},
                     placement_effect={f: {str(n): paired(seed_mean(f'{f}-R', n), seed_mean(f'{f}-S', n)) for n in LABEL_BUDGETS}
                                       for f in FORMS},
                     form_effect={f'G-{p}_vs_{f}-{p}': {str(n): paired(seed_mean(f'G-{p}', n), seed_mean(f'{f}-{p}', n))
                                                       for n in LABEL_BUDGETS} for f in ('A', 'M') for p in PLACEMENTS})
    decisions['primary'] = dict(variant=PRIMARY,
                                Q1_supported_at_N_10_20_40=all(decisions['Q1'][PRIMARY][str(n)]['supported'] for n in (10, 20, 40)),
                                Q2_supported_at_N_10_20_40=all(decisions['Q2'][PRIMARY][str(n)]['supported'] for n in (10, 20, 40)))

    # ---------------------------------------------------------------- single final test evaluation
    for run in runs:
        run['test'] = summary(errors(predictors[run['arm'], run['budget'], run['seed']](test), 'test'))
    test_curves = {arm: {str(n): dict(median_percent_mean=float(np.mean([r['test']['median_percent'] for r in runs if r['arm'] == arm and r['budget'] == n])),
                                      p95_percent_mean=float(np.mean([r['test']['p95_percent'] for r in runs if r['arm'] == arm and r['budget'] == n])))
                         for n in ((0,) if arm == 'P' else LABEL_BUDGETS)} for arm in ['P', 'D', *variants]}

    # ---------------------------------------------------------------- figures
    colors = dict(G='tab:gray', A='tab:blue', M='tab:green')
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), layout='constrained')
    for ax, stat, target in zip(axes, ('median', 'p95'), (TARGET['median_percent'], TARGET['p95_percent'])):
        ax.plot(LABEL_BUDGETS, [curves['D'][str(n)][f'{stat}_percent_mean'] for n in LABEL_BUDGETS], 'o-', color='tab:orange',
                label='data only (random TSDT labels)')
        for v in variants:
            form, placement = v.split('-')
            ax.plot(LABEL_BUDGETS, [curves[v][str(n)][f'{stat}_percent_mean'] for n in LABEL_BUDGETS],
                    marker='^' if placement == 'S' else 'v', ls='-' if placement == 'S' else ':', color=colors[form],
                    lw=2.2 if v == PRIMARY else 1.2, label=NAMES[v])
        ax.axhline(curves['P']['0'][f'{stat}_percent_mean'], color='tab:purple', ls='--', lw=1.2,
                   label=f'physics only, 0 labels: {curves["P"]["0"][f"{stat}_percent_mean"]:.2f}%')
        ax.axhline(target, color='r', ls=':', lw=.8, label=f'target {target}%')
        ax.set(xscale='log', yscale='log', xticks=LABEL_BUDGETS, xlabel='TSDT-labelled training designs',
               ylabel=f'{"Median" if stat == "median" else "95th percentile"} validation error vs TSDT (%)',
               title=f'E3c: {"median" if stat == "median" else "95th percentile"} over 30 designs (3 seeds)')
        ax.set_xticklabels([str(b) for b in LABEL_BUDGETS]); ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.grid(alpha=.3, which='both'); ax.legend(fontsize=6.5)
    fig.savefig(out/'e3c_label_curves.png', dpi=170); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.8, 4.3), layout='constrained')
    pick = lambda arm, n: np.array(next(r for r in runs if r['arm'] == arm and r['seed'] == SEEDS[0] and r['budget'] == n)['validation_errors'])
    ax.scatter(P[val, 1], 100*pick('P', 0), marker='o', color='tab:purple', alpha=.7, label='physics only (0 labels)')
    ax.scatter(P[val, 1], 100*pick('G-R', 20), marker='v', color='tab:gray', alpha=.8, label=NAMES['G-R']+', N=20')
    ax.scatter(P[val, 1], 100*pick(PRIMARY, 20), marker='^', color='tab:green', alpha=.9, label=NAMES[PRIMARY]+', N=20')
    ax.set(xscale='log', yscale='log', xlabel='slenderness a/h', ylabel='validation error vs TSDT (%)',
           title=f'E3c: per-design error (seed {SEEDS[0]})')
    ax.grid(alpha=.3, which='both'); ax.legend(fontsize=7)
    fig.savefig(out/'e3c_error_vs_thickness.png', dpi=170); plt.close(fig)

    # ---------------------------------------------------------------- persist
    np.savez_compressed(out/'e3c_errors.npz', validation_ids=ids[val], test_ids=ids[test], shear_share=psi,
                        **{f'{r["arm"]}_{r["budget"]}_s{r["seed"]}_validation': np.array(r['validation_errors']) for r in runs})
    for r in runs:
        r.pop('validation_errors')
    manifest = dict(experiment='E3c physics-structured correction and physics-guided label placement',
                    created_utc=datetime.now(timezone.utc).isoformat(),
                    pilot=dict(path=str(args.pilot), dataset_sha256=recorded['dataset.npz']),
                    e3b_reference=dict(path=str(args.e3b), sha256=sha256(args.e3b)),
                    rules=RULES, preregistration_diagnostics=preregistration, variants=NAMES,
                    grids=dict(degrees=DEGREES, ridges=RIDGES, correction_degrees=CORRECTION_DEGREES),
                    label_budgets=LABEL_BUDGETS, collocation_M=COLLOCATION_M, seeds=SEEDS, ridge_scope=SCOPE,
                    partitions=dict(train=len(train), validation=len(val), test=len(test)),
                    consistency_with_e3b=consistency, runs=runs, validation_curves=curves, decisions=decisions,
                    test_curves=test_curves, total_seconds=perf_counter()-start_all,
                    environment=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                                     matplotlib=matplotlib.__version__, platform=platform.platform()),
                    source_sha256={str(p.relative_to(ROOT)): sha256(p) for p in
                                   [*sorted((ROOT/'fgplate').glob('*.py')), Path(__file__)]})
    write_json(out/'manifest.json', manifest)
    write_json(out/'checksums.json', {p.name: sha256(p) for p in sorted(out.iterdir()) if p.is_file() and p.name != 'checksums.json'})
    print(json.dumps(dict(primary=decisions['primary'], Q3=decisions['Q3'],
                          curves={a: {n: (round(v['median_percent_mean'], 3), round(v['p95_percent_mean'], 2)) for n, v in cc.items()}
                                  for a, cc in curves.items()}), indent=1), flush=True)


if __name__ == '__main__':
    main()
