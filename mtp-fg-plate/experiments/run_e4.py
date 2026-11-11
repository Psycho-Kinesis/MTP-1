"""E4: confirmation of the frozen E3c configurations on a fresh, locked test set.

Run from the project root:
    python experiments/run_e4.py --output results/e4_confirmation

Follows docs/E4_PREREGISTRATION.md exactly (its sha256 is asserted below):
1. refit the frozen E3c arms from the pilot (train labels, FSDT collocation, validation
   selection) and assert they reproduce E3c's validation metrics;
2. only then generate 200 fresh designs (generator pilot-1.1, seed 20261201), their
   smooth-mixture loads and TSDT labels; assert they are disjoint from every pilot and
   collocation design;
3. evaluate every frozen model ONCE on the fresh designs and answer C1-C4.
Nothing is fitted, selected or tuned on the fresh designs.
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
from fgplate.material import al_al2o3, plate_stiffness
from fgplate.pilot import GENERATOR_VERSION, design_loads, load_key, quantize, sample_designs
from fgplate.transfer import (assemble_data, assemble_physics, design_coordinates, mode_matrices, predict_scaled,
                              solve_modes)
from fgplate.tsdt import transfer

PREREGISTRATION = ROOT/'docs'/'E4_PREREGISTRATION.md'
PREREGISTRATION_SHA256 = 'a60526d7dc6bb11c74336de4ca9ffbe7bb016710689ac69dafc22abb8f03f589'
FRESH_SEED, FRESH_COUNT = 20261201, 200
MIXTURE = 2
DEGREES = (2, 3, 4, 5, 6, 7)
RIDGES = (1e-12, 1e-10, 1e-8, 1e-6, 1e-4)
CORRECTION_DEGREES = (0, 1, 2, 3, 4)
LABEL_BUDGETS = (10, 20, 40, 70, 140)
COLLOCATION_M = 560
SEEDS = (11, 22, 33)
SCOPE = 'block'
VARIANTS = ('G-R', 'A-R', 'A-S', 'M-R', 'M-S')
PRIMARY = 'M-S'
TARGET = dict(median_percent=1.0, p95_percent=3.0)
NAMES = {'P': 'physics only (FSDT energy, 0 labels)', 'D': 'data only (random TSDT labels)',
         'G-R': 'global correction, random labels (E3b Hc)', 'A-R': '(kh)^2 correction, random labels',
         'A-S': '(kh)^2 correction, shear-guided labels', 'M-R': 'H_P(kh)^2 correction, random labels',
         'M-S': 'H_P(kh)^2 correction, shear-guided labels (primary)'}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def summary(errors):
    e = 100*np.asarray(errors)
    return dict(median_percent=float(np.median(e)), p95_percent=float(np.quantile(e, .95)),
                worst_percent=float(e.max()), mean_percent=float(e.mean()),
                median_percent_ci95=(100*bootstrap_ci(errors, 'median')[0]).tolist(),
                p95_percent_ci95=(100*bootstrap_ci(errors, 'p95')[0]).tolist(), cases=int(len(e)))


def score(errors):
    e = 100*np.asarray(errors)
    med, p95 = np.median(e), np.quantile(e, .95)
    return (max(med/TARGET['median_percent'], p95/TARGET['p95_percent']), med)


def collocation_designs(count, seed):
    unit = qmc.Halton(d=3, scramble=True, seed=seed).random(count)
    t = qmc.scale(unit, [0, np.log(5), .5], [np.log(9), np.log(100), 2])
    return quantize(np.column_stack((np.expm1(t[:, 0]), np.exp(t[:, 1]), t[:, 2])))


def shear_share(parameters):
    out = []
    for g, ratio, aspect in parameters:
        S = plate_stiffness(al_al2o3(float(g)), 1/ratio)
        k2 = np.pi**2*(1+1/aspect**2)
        bend, shear = 1/((S.D11-S.B11**2/S.A11)*k2**2), 1/(S.As*k2)
        out.append(shear/(bend+shear))
    return np.array(out)


def kh2(parameters, modes):
    h, aspect = 1/parameters[:, 1], parameters[:, 2]
    return np.pi**2*(modes[:, 0][None]**2+(modes[:, 1][None]/aspect[:, None])**2)*h[:, None]**2


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--pilot', type=Path, default=ROOT/'results'/'pilot_200')
    parser.add_argument('--e3c', type=Path, default=ROOT/'results'/'e3c_structured'/'manifest.json')
    parser.add_argument('--output', type=Path, default=ROOT/'results'/'e4_confirmation')
    args = parser.parse_args()
    assert sha256(PREREGISTRATION) == PREREGISTRATION_SHA256, 'pre-registration file changed; E4 must not run'
    out = args.output.resolve()
    if out.exists() and any(out.iterdir()):
        parser.error('output directory is not empty; choose a new experiment directory')
    out.mkdir(parents=True, exist_ok=True)
    start_all = perf_counter()

    # ================================================================ 1. refit frozen arms (pilot only)
    pilot = args.pilot.resolve()
    recorded = json.loads((pilot/'checksums.json').read_text())
    assert sha256(pilot/'dataset.npz') == recorded['dataset.npz']
    with np.load(pilot/'dataset.npz', allow_pickle=False) as z:
        P, split = z['parameters'], z['split']
        Q, modes_all, xi, eta = z['pressure_coefficients'][:, MIXTURE], z['mode_indices'], z['xi'], z['eta']
    active = np.flatnonzero(np.any(Q != 0, axis=0))
    modes, c = modes_all[active], Q[:, active]
    basis = analytic_basis(modes, xi, eta)
    H_tsdt = transfer(P, modes, 'TSDT')
    x = design_coordinates(P)
    train, val = np.flatnonzero(split == 'train'), np.flatnonzero(split == 'validation')
    psi = shear_share(P)
    kh2_pilot = kh2(P, modes)

    def fields(H, coeffs):
        return reconstruct_analytic(np.transpose(H*coeffs[:, :, None], (0, 2, 1)), basis)
    truth_val = fields(H_tsdt[val], c[val])

    def val_errors(H):
        return relative_l2(fields(H, c[val]), truth_val)

    phys, colloc = {}, {}
    for seed in SEEDS:
        designs = collocation_designs(COLLOCATION_M, 1000+seed)
        colloc[seed] = designs
        K, f = mode_matrices(designs, modes)
        for d in DEGREES:
            phys[seed, d] = assemble_physics(design_coordinates(designs), K, f, d)

    def select(candidates):
        best = None
        for hp, H_val, payload in candidates:
            key = score(val_errors(H_val))
            if best is None or key < best[0]:
                best = (key, hp, H_val, payload)
        return best[1:]

    frozen = {}      # (arm, N, seed) -> dict(kind, hyperparameters, coefficients)
    for seed in SEEDS:
        cand = []
        for d in DEGREES:
            for r in RIDGES:
                th = solve_modes(*phys[seed, d], d, r, SCOPE)
                cand.append((dict(degree=d, ridge=r), predict_scaled(th, d, x[val]), (th, d)))
        hp, _, (th_p, d_p) = select(cand)
        frozen['P', 0, seed] = dict(hp=hp, base=(th_p, d_p))
        H_P = predict_scaled(th_p, d_p, x)
        prefactors = {'G': None, 'A': np.broadcast_to(kh2_pilot[:, :, None], H_P.shape), 'M': H_P*kh2_pilot[:, :, None]}
        keys = np.log(np.random.default_rng([seed, 7]).random(len(train)))/psi[train]
        orders = {'R': np.random.default_rng(seed).permutation(train), 'S': train[np.argsort(-keys)]}
        for N in LABEL_BUDGETS:
            sub = np.sort(orders['R'][:N])
            data = {d: assemble_data(x[sub], H_tsdt[sub], d)[:2] for d in DEGREES}
            cand = []
            for d in DEGREES:
                for r in RIDGES:
                    th = solve_modes(*data[d], d, r, SCOPE)
                    cand.append((dict(degree=d, ridge=r), predict_scaled(th, d, x[val]), (th, d)))
            hp, _, (th, d) = select(cand)
            frozen['D', N, seed] = dict(hp=hp, model=(th, d))
            for variant in VARIANTS:
                form, placement = variant.split('-')
                sub = np.sort(orders[placement][:N])
                pref = prefactors[form]
                cand = []
                for dc in CORRECTION_DEGREES:
                    A, b, _ = assemble_data(x[sub], H_tsdt[sub]-H_P[sub], dc, prefactor=None if pref is None else pref[sub])
                    for r in RIDGES:
                        th = solve_modes(A, b, dc, r, SCOPE)
                        Hv = H_P[val]+predict_scaled(th, dc, x[val], None if pref is None else pref[val])
                        cand.append((dict(base_degree=d_p, correction_degree=dc, ridge=r), Hv, (th, dc)))
                hp, _, (th, dc) = select(cand)
                frozen[variant, N, seed] = dict(hp=hp, base=(th_p, d_p), correction=(th, dc), form=form)
    print(f'frozen arms refitted ({perf_counter()-start_all:.0f} s)', flush=True)

    def predict(arm, N, seed, params):
        """Frozen model applied to arbitrary designs (no access to their labels)."""
        entry = frozen[arm, N, seed]
        xx = design_coordinates(params)
        if arm == 'D':
            th, d = entry['model']
            return predict_scaled(th, d, xx)
        th_p, d_p = entry['base']
        H_P = predict_scaled(th_p, d_p, xx)
        if arm == 'P':
            return H_P
        th, dc = entry['correction']
        k2 = kh2(params, modes)[:, :, None]
        pref = {'G': None, 'A': np.broadcast_to(k2, H_P.shape), 'M': H_P*k2}[entry['form']]
        return H_P+predict_scaled(th, dc, xx, pref)

    # reproduction of E3c validation metrics (frozen means identical)
    e3c = json.loads(args.e3c.read_text())
    e3c_runs = {(r['arm'], r['budget'], r['seed']): r['validation'] for r in e3c['runs']}
    consistency = {}
    for (arm, N, seed) in frozen:
        v = summary(val_errors(predict(arm, N, seed, P[val])))
        ref = e3c_runs[arm, N, seed]
        consistency[f'{arm}_{N}_s{seed}'] = all(np.isclose(v[k], ref[k], rtol=1e-9, atol=1e-12)
                                                for k in ('median_percent', 'p95_percent', 'worst_percent'))
    assert all(consistency.values()), f'frozen arms do not reproduce E3c: {[k for k, ok in consistency.items() if not ok]}'
    print(f'all {len(consistency)} frozen arms reproduce E3c validation metrics', flush=True)

    # ================================================================ 2. fresh locked test set (generated only now)
    fresh, fresh_ids, _ = sample_designs(FRESH_COUNT, FRESH_SEED, GENERATOR_VERSION)
    xf = design_coordinates(fresh)
    for name, other in [('pilot', x), *[(f'collocation s{s}', design_coordinates(colloc[s])) for s in SEEDS]]:
        gap = np.min(np.linalg.norm(xf[:, None]-other[None], axis=2))
        assert gap > 1e-6, f'fresh design coincides with a {name} design'
    assert not set(fresh_ids.tolist()) & set(np.load(pilot/'dataset.npz', allow_pickle=False)['design_ids'].tolist())
    c_f = np.array([[design_loads(load_key(GENERATOR_VERSION, i, fresh_ids[i]), FRESH_SEED)[0][2][tuple(map(int, m))]
                     for m in modes] for i in range(FRESH_COUNT)])
    H_tsdt_f, H_fsdt_f = transfer(fresh, modes, 'TSDT'), transfer(fresh, modes, 'FSDT')
    np.savez_compressed(out/'fresh_test_set.npz', parameters=fresh, design_ids=fresh_ids, modes=modes,
                        mixture_coefficients=c_f, H_tsdt=H_tsdt_f, H_fsdt=H_fsdt_f, seed=FRESH_SEED,
                        generator_version=GENERATOR_VERSION)
    truth_f = fields(H_tsdt_f, c_f)

    def fresh_errors(H, w_only=False):
        F = fields(H, c_f)
        return relative_l2(F[:, 2], truth_f[:, 2]) if w_only else relative_l2(F, truth_f)

    # ================================================================ 3. single evaluation
    runs = []
    for (arm, N, seed), entry in frozen.items():
        H = predict(arm, N, seed, fresh)
        e = fresh_errors(H)
        runs.append(dict(arm=arm, budget=N, seed=seed, selected=entry['hp'], fresh=summary(e),
                         fresh_w_only=summary(fresh_errors(H, True)), errors=e))
    gap_fresh = dict(five_field=summary(fresh_errors(H_fsdt_f)), w_only=summary(fresh_errors(H_fsdt_f, True)))

    def rows(arm, N):
        return [r for r in runs if r['arm'] == arm and r['budget'] == N]

    def seed_mean_errors(arm, N):
        return np.mean([r['errors'] for r in rows(arm, N)], axis=0)

    arms = ('P', 'D', *VARIANTS)
    curves = {}
    for arm in arms:
        curves[arm] = {}
        for N in ((0,) if arm == 'P' else LABEL_BUDGETS):
            rr = rows(arm, N)
            curves[arm][str(N)] = dict(
                median_percent_mean=float(np.mean([r['fresh']['median_percent'] for r in rr])),
                p95_percent_mean=float(np.mean([r['fresh']['p95_percent'] for r in rr])),
                median_percent_range=[min(r['fresh']['median_percent'] for r in rr), max(r['fresh']['median_percent'] for r in rr)],
                p95_percent_range=[min(r['fresh']['p95_percent'] for r in rr), max(r['fresh']['p95_percent'] for r in rr)],
                seed_averaged=summary(seed_mean_errors(arm, N)),
                w_only_median_percent_mean=float(np.mean([r['fresh_w_only']['median_percent'] for r in rr])))

    def paired(reference, candidate):
        diff = reference-candidate
        lo, hi = bootstrap_ci(diff, 'median')[0]
        return dict(median_difference_percent=float(100*np.median(diff)), ci95_percent=[float(100*lo), float(100*hi)],
                    confirmed=bool(lo > 0), better_designs=int(np.sum(diff > 0)), designs=int(len(diff)))

    p_err = seed_mean_errors('P', 0)
    C1 = {str(N): paired(p_err, seed_mean_errors(PRIMARY, N)) for N in LABEL_BUDGETS}
    C2 = {str(N): paired(seed_mean_errors('D', N), seed_mean_errors(PRIMARY, N)) for N in LABEL_BUDGETS}
    ms10 = curves[PRIMARY]['10']
    decisions = dict(
        C1=dict(per_N=C1, confirmed=all(C1[str(n)]['confirmed'] for n in (10, 20, 40))),
        C2=dict(per_N=C2, confirmed=all(C2[str(n)]['confirmed'] for n in (10, 20, 40))),
        C3=dict(median_percent_mean=ms10['median_percent_mean'], confirmed=ms10['median_percent_mean'] <= TARGET['median_percent']),
        C4=dict(p95_percent_mean=ms10['p95_percent_mean'], seed_averaged_p95_ci95=ms10['seed_averaged']['p95_percent_ci95'],
                confirmed=ms10['p95_percent_mean'] <= TARGET['p95_percent']),
        secondary=dict(
            smallest_N_meeting_both_targets={a: next((n for n in LABEL_BUDGETS if curves[a][str(n)]['median_percent_mean'] <= TARGET['median_percent']
                                                      and curves[a][str(n)]['p95_percent_mean'] <= TARGET['p95_percent']), None)
                                             for a in ('D', *VARIANTS)},
            form_contrast_GR_minus_MS={str(N): paired(seed_mean_errors('G-R', N), seed_mean_errors(PRIMARY, N)) for N in LABEL_BUDGETS},
            physics_only_meets_targets=bool(curves['P']['0']['median_percent_mean'] <= TARGET['median_percent']
                                            and curves['P']['0']['p95_percent_mean'] <= TARGET['p95_percent'])))
    pilot_test = {arm: {k: (v['median_percent_mean'], v['p95_percent_mean']) for k, v in cc.items()}
                  for arm, cc in e3c['test_curves'].items() if arm in arms}

    # ================================================================ figures
    colors = {'P': 'tab:purple', 'D': 'tab:orange', 'G-R': 'tab:gray', 'A-R': 'tab:cyan', 'A-S': 'tab:blue',
              'M-R': 'lightgreen', 'M-S': 'tab:green'}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), layout='constrained')
    for ax, stat, target in zip(axes, ('median', 'p95'), (TARGET['median_percent'], TARGET['p95_percent'])):
        for arm in ('D', *VARIANTS):
            mean = [curves[arm][str(n)][f'{stat}_percent_mean'] for n in LABEL_BUDGETS]
            lo = [curves[arm][str(n)][f'{stat}_percent_range'][0] for n in LABEL_BUDGETS]
            hi = [curves[arm][str(n)][f'{stat}_percent_range'][1] for n in LABEL_BUDGETS]
            ax.plot(LABEL_BUDGETS, mean, marker='o', color=colors[arm], lw=2.4 if arm == PRIMARY else 1.2, label=NAMES[arm])
            ax.fill_between(LABEL_BUDGETS, lo, hi, color=colors[arm], alpha=.12)
        ax.axhline(curves['P']['0'][f'{stat}_percent_mean'], color=colors['P'], ls='--', lw=1.2,
                   label=f'{NAMES["P"]}: {curves["P"]["0"][f"{stat}_percent_mean"]:.2f}%')
        ax.axhline(gap_fresh['five_field'][f'{stat}_percent'], color='k', ls='-.', lw=1,
                   label=f'exact FSDT (model-form gap): {gap_fresh["five_field"][f"{stat}_percent"]:.2f}%')
        ax.axhline(target, color='r', ls=':', lw=.8, label=f'target {target}%')
        ax.set(xscale='log', yscale='log', xticks=LABEL_BUDGETS, xlabel='TSDT-labelled training designs',
               ylabel=f'{"Median" if stat == "median" else "95th percentile"} error vs TSDT (%)',
               title=f'E4: fresh locked test set, {FRESH_COUNT} designs ({"median" if stat == "median" else "95th percentile"})')
        ax.set_xticklabels([str(b) for b in LABEL_BUDGETS]); ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.grid(alpha=.3, which='both'); ax.legend(fontsize=6.5)
    fig.savefig(out/'e4_fresh_test_curves.png', dpi=170); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.8, 4.3), layout='constrained')
    ax.scatter(fresh[:, 1], 100*fresh_errors(H_fsdt_f), marker='x', c='k', s=14, label='exact FSDT (model-form gap)')
    ax.scatter(fresh[:, 1], 100*p_err, marker='o', s=14, color=colors['P'], alpha=.6, label=NAMES['P'])
    ax.scatter(fresh[:, 1], 100*seed_mean_errors(PRIMARY, 10), marker='^', s=16, color=colors['M-S'], alpha=.8,
               label=NAMES['M-S']+', N=10')
    ax.axhline(TARGET['p95_percent'], color='r', ls=':', lw=.8)
    ax.set(xscale='log', yscale='log', xlabel='slenderness a/h', ylabel='error vs TSDT (%), seed-averaged',
           title=f'E4: per-design error on {FRESH_COUNT} fresh designs')
    ax.grid(alpha=.3, which='both'); ax.legend(fontsize=7)
    fig.savefig(out/'e4_error_vs_thickness.png', dpi=170); plt.close(fig)

    # ================================================================ persist
    np.savez_compressed(out/'e4_errors.npz', fresh_ids=fresh_ids,
                        **{f'{r["arm"]}_{r["budget"]}_s{r["seed"]}': r['errors'] for r in runs})
    for r in runs:
        r.pop('errors')
    manifest = dict(experiment='E4 confirmation on a fresh locked test set',
                    created_utc=datetime.now(timezone.utc).isoformat(),
                    preregistration=dict(path='docs/E4_PREREGISTRATION.md', sha256=PREREGISTRATION_SHA256),
                    pilot=dict(dataset_sha256=recorded['dataset.npz']),
                    e3c_reference=dict(path=str(args.e3c), sha256=sha256(args.e3c)),
                    fresh_test_set=dict(count=FRESH_COUNT, seed=FRESH_SEED, generator=GENERATOR_VERSION,
                                        file='fresh_test_set.npz',
                                        parameter_ranges=dict(g=[float(fresh[:, 0].min()), float(fresh[:, 0].max())],
                                                              a_over_h=[float(fresh[:, 1].min()), float(fresh[:, 1].max())],
                                                              b_over_a=[float(fresh[:, 2].min()), float(fresh[:, 2].max())])),
                    frozen_arms_reproduce_e3c=consistency, model_form_gap_fresh=gap_fresh, runs=runs,
                    fresh_curves=curves, decisions=decisions, pilot_test_for_comparison=pilot_test,
                    total_seconds=perf_counter()-start_all,
                    environment=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                                     matplotlib=matplotlib.__version__, platform=platform.platform()),
                    source_sha256={str(p.relative_to(ROOT)): sha256(p) for p in
                                   [*sorted((ROOT/'fgplate').glob('*.py')), Path(__file__)]})
    write_json(out/'manifest.json', manifest)
    write_json(out/'checksums.json', {p.name: sha256(p) for p in sorted(out.iterdir()) if p.is_file() and p.name != 'checksums.json'})
    print(json.dumps({k: (v if k != 'secondary' else v['smallest_N_meeting_both_targets']) for k, v in decisions.items()},
                     indent=1, default=str)[:4000], flush=True)


if __name__ == '__main__':
    main()
