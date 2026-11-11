"""E5 development: a better label-free base model, chosen on pilot train/validation only.

Run from the project root:
    python experiments/run_e5.py --output results/e5_base_model

E4 traced the remaining tail error to the label-free physics-only surrogate P (FSDT energy,
polynomial in the design coordinates), not to the physics: in the narrow / near-uniform /
slender corner P is far less accurate than exact FSDT. E5 compares four PRE-DECLARED base
variants, all trained on the FSDT energy only (no labels), same collocation (M = 560 per
seed), grids and tail-aware validation selection as E3c:

    P0  linear b/a coordinates, no prefactor            (= E3c's P, asserted)
    P1  log(b/a) coordinates,   no prefactor
    P2  linear b/a coordinates, classical-plate prefactor (fgplate.transfer.classical_prefactor)
    P3  log(b/a) coordinates,   classical-plate prefactor

Choice rule v1 (fixed before the first run): the variant with the lowest mean-over-seeds validation
score max(median/1%, p95/3%) vs TSDT; ties go to the earlier (simpler) variant. It chose P0 (see
results/e5_base_model). Rule v2 (declared after that outcome, before any test data; see CHOICE_RULE_V2)
chooses the base by the validation score of the downstream M-S correction at N = 10, 20, 40, and
decides the base carried into the pre-registered confirmation. Both outcomes are recorded.
The M-S and M-R corrections (E3c's multiplicative (k h)^2 form, correction grids as E3c) are
then rebuilt on the chosen base, using the base's coordinate mapping. The data-only arm D and
the E3c-based arms (M-S/M-R on P0) are kept for comparison and asserted to reproduce E3c.
No test data (pilot test partition, E4 fresh set) is used. Confirmation is a separate,
pre-registered run on a new fresh set (experiments/run_e5_confirm.py).
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
from fgplate.pilot import quantize
from fgplate.transfer import (assemble_data, assemble_physics, classical_prefactor, design_coordinates, mode_matrices,
                              predict_scaled, solve_modes)
from fgplate.tsdt import transfer

MIXTURE = 2
DEGREES = (2, 3, 4, 5, 6, 7)
RIDGES = (1e-12, 1e-10, 1e-8, 1e-6, 1e-4)
CORRECTION_DEGREES = (0, 1, 2, 3, 4)
LABEL_BUDGETS = (10, 20, 40, 70, 140)
COLLOCATION_M = 560
SEEDS = (11, 22, 33)
SCOPE = 'block'
TARGET = dict(median_percent=1.0, p95_percent=3.0)
BASES = {'P0': dict(aspect='linear', prefactor=False), 'P1': dict(aspect='log', prefactor=False),
         'P2': dict(aspect='linear', prefactor=True), 'P3': dict(aspect='log', prefactor=True)}
CHOICE_RULE = ('lowest mean-over-seeds validation score max(median/1%, p95/3%) vs TSDT; ties to the earlier variant '
               '(P0 < P1 < P2 < P3)')
CHOICE_RULE_V2 = ('lowest validation score max(median/1%, p95/3%) vs TSDT of the downstream M-S correction built on the '
                  'base, averaged over seeds and N = 10, 20, 40; ties to the earlier variant. Declared after rule v1 '
                  '(physics-only score) chose P0, because rule v1 rewards a base whose approximation error happens to '
                  'cancel the FSDT->TSDT bias, while the base is meant to reproduce FSDT and the correction handles '
                  'FSDT->TSDT. No test data had been used. Rule v2 decides the base carried into confirmation.')
V2_BUDGETS = (10, 20, 40)


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


class Frozen:
    """A frozen surrogate that can be applied to any designs (no access to their labels)."""

    def __init__(self, kind, base_name, base_theta, base_degree, correction=None, data=None):
        self.kind, self.base_name, self.base_theta, self.base_degree = kind, base_name, base_theta, base_degree
        self.correction, self.data = correction, data

    def base(self, params, modes):
        cfg = BASES[self.base_name]
        x = design_coordinates(params, cfg['aspect'])
        pref = classical_prefactor(params, modes) if cfg['prefactor'] else None
        return predict_scaled(self.base_theta, self.base_degree, x, pref)

    def __call__(self, params, modes):
        if self.kind == 'D':
            th, d = self.data
            return predict_scaled(th, d, design_coordinates(params), None)
        H_P = self.base(params, modes)
        if self.kind == 'P':
            return H_P
        th, dc = self.correction
        x = design_coordinates(params, BASES[self.base_name]['aspect'])
        return H_P+predict_scaled(th, dc, x, H_P*kh2(params, modes)[:, :, None])


def fit_all(pilot_dir, log=print):
    """Fit every E5 arm from the pilot (train labels, FSDT collocation, validation selection).

    Returns a dict with frozen models, validation results, the chosen base and the context needed
    to evaluate on new designs. Deterministic; used by run_e5.py and run_e5_confirm.py.
    """
    recorded = json.loads((pilot_dir/'checksums.json').read_text())
    assert sha256(pilot_dir/'dataset.npz') == recorded['dataset.npz']
    with np.load(pilot_dir/'dataset.npz', allow_pickle=False) as z:
        P, split, ids = z['parameters'], z['split'], z['design_ids']
        Q, modes_all, xi, eta = z['pressure_coefficients'][:, MIXTURE], z['mode_indices'], z['xi'], z['eta']
    active = np.flatnonzero(np.any(Q != 0, axis=0))
    modes, c = modes_all[active], Q[:, active]
    basis = analytic_basis(modes, xi, eta)
    H_tsdt, H_fsdt = transfer(P, modes, 'TSDT'), transfer(P, modes, 'FSDT')
    train, val = np.flatnonzero(split == 'train'), np.flatnonzero(split == 'validation')
    psi = shear_share(P)
    pref_pilot = classical_prefactor(P, modes)
    kh2_pilot = kh2(P, modes)

    def fields(H, coeffs):
        return reconstruct_analytic(np.transpose(H*coeffs[:, :, None], (0, 2, 1)), basis)
    truth_val, fsdt_val = fields(H_tsdt[val], c[val]), fields(H_fsdt[val], c[val])

    def val_errors(H, reference=None):
        return relative_l2(fields(H, c[val]), truth_val if reference is None else reference)

    def select(candidates):
        best = None
        for hp, H_val, payload in candidates:
            key = score(val_errors(H_val))
            if best is None or key < best[0]:
                best = (key, hp, H_val, payload)
        return best

    t0 = perf_counter()
    colloc = {s: collocation_designs(COLLOCATION_M, 1000+s) for s in SEEDS}
    mats = {s: mode_matrices(colloc[s], modes) for s in SEEDS}
    colloc_pref = {s: classical_prefactor(colloc[s], modes) for s in SEEDS}
    models, results = {}, {}
    # ------------------------------------------------ base variants (physics only, no labels)
    base_scores = {name: [] for name in BASES}
    for name, cfg in BASES.items():
        for seed in SEEDS:
            xc = design_coordinates(colloc[seed], cfg['aspect'])
            xv = design_coordinates(P[val], cfg['aspect'])
            cand = []
            for d in DEGREES:
                A, b = assemble_physics(xc, *mats[seed], d, prefactor=colloc_pref[seed] if cfg['prefactor'] else None)
                for r in RIDGES:
                    th = solve_modes(A, b, d, r, SCOPE)
                    cand.append((dict(degree=d, ridge=r), predict_scaled(th, d, xv, pref_pilot[val] if cfg['prefactor'] else None), (th, d)))
            key, hp, Hv, (th, d) = select(cand)
            models[name, 0, seed] = Frozen('P', name, th, d)
            base_scores[name].append(key[0])
            results[name, 0, seed] = dict(selected=hp, validation=summary(val_errors(Hv)),
                                          validation_vs_exact_fsdt=summary(val_errors(Hv, fsdt_val)),
                                          errors=val_errors(Hv))
        log(f'base {name} done ({perf_counter()-t0:.0f} s)')
    mean_scores = {name: float(np.mean(v)) for name, v in base_scores.items()}
    chosen_v1 = min(BASES, key=lambda name: (round(mean_scores[name], 12), list(BASES).index(name)))
    # ------------------------------------------------ data only (as E3c) and corrections on P0 and on the chosen base
    for seed in SEEDS:
        order_r = np.random.default_rng(seed).permutation(train)
        keys = np.log(np.random.default_rng([seed, 7]).random(len(train)))/psi[train]
        orders = {'R': order_r, 'S': train[np.argsort(-keys)]}
        for N in LABEL_BUDGETS:
            sub = np.sort(order_r[:N])
            x_lin = design_coordinates(P)
            data = {d: assemble_data(x_lin[sub], H_tsdt[sub], d)[:2] for d in DEGREES}
            cand = []
            for d in DEGREES:
                for r in RIDGES:
                    th = solve_modes(*data[d], d, r, SCOPE)
                    cand.append((dict(degree=d, ridge=r), predict_scaled(th, d, x_lin[val]), (th, d)))
            key, hp, Hv, (th, d) = select(cand)
            models['D', N, seed] = Frozen('D', None, None, None, data=(th, d))
            results['D', N, seed] = dict(selected=hp, validation=summary(val_errors(Hv)), errors=val_errors(Hv))
            for base_name in BASES:
                base_model = models[base_name, 0, seed]
                H_P = base_model.base(P, modes)
                x = design_coordinates(P, BASES[base_name]['aspect'])
                pref = H_P*kh2_pilot[:, :, None]
                for placement in ('R', 'S'):
                    sub = np.sort(orders[placement][:N])
                    cand = []
                    for dc in CORRECTION_DEGREES:
                        A, b, _ = assemble_data(x[sub], H_tsdt[sub]-H_P[sub], dc, prefactor=pref[sub])
                        for r in RIDGES:
                            th = solve_modes(A, b, dc, r, SCOPE)
                            cand.append((dict(base=base_name, base_degree=base_model.base_degree, correction_degree=dc, ridge=r),
                                         H_P[val]+predict_scaled(th, dc, x[val], pref[val]), (th, dc)))
                    key, hp, Hv, (th, dc) = select(cand)
                    arm = f'M-{placement}@{base_name}'
                    models[arm, N, seed] = Frozen('C', base_name, base_model.base_theta, base_model.base_degree, correction=(th, dc))
                    results[arm, N, seed] = dict(selected=hp, validation=summary(val_errors(Hv)), errors=val_errors(Hv))
        log(f'seed {seed} arms done ({perf_counter()-t0:.0f} s)')
    downstream = {name: float(np.mean([score(results[f'M-S@{name}', N, s]['errors'])[0] for s in SEEDS for N in V2_BUDGETS]))
                  for name in BASES}
    chosen = min(BASES, key=lambda name: (round(downstream[name], 12), list(BASES).index(name)))
    return dict(chosen_v1=chosen_v1, downstream_scores=downstream, P=P, ids=ids, split=split, modes=modes, c=c, basis=basis, H_tsdt=H_tsdt, H_fsdt=H_fsdt,
                val=val, train=train, colloc=colloc, models=models, results=results, chosen=chosen,
                mean_base_scores=mean_scores, dataset_sha256=recorded['dataset.npz'])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--pilot', type=Path, default=ROOT/'results'/'pilot_200')
    parser.add_argument('--e3c', type=Path, default=ROOT/'results'/'e3c_structured'/'manifest.json')
    parser.add_argument('--output', type=Path, default=ROOT/'results'/'e5_base_model')
    args = parser.parse_args()
    out = args.output.resolve()
    if out.exists() and any(out.iterdir()):
        parser.error('output directory is not empty; choose a new experiment directory')
    out.mkdir(parents=True, exist_ok=True)
    start = perf_counter()
    fit = fit_all(args.pilot.resolve(), log=lambda m: print(m, flush=True))
    res, chosen, chosen_v1 = fit['results'], fit['chosen'], fit['chosen_v1']

    # reproduction of E3c: P0 = P, D = D, M-S@P0 = M-S, M-R@P0 = M-R
    e3c = {(r['arm'], r['budget'], r['seed']): r['validation'] for r in json.loads(args.e3c.read_text())['runs']}
    pairs = [(('P0', 0, s), ('P', 0, s)) for s in SEEDS]
    pairs += [((a, N, s), (b, N, s)) for s in SEEDS for N in LABEL_BUDGETS
              for a, b in (('D', 'D'), ('M-S@P0', 'M-S'), ('M-R@P0', 'M-R'))]
    consistency = {f'{a[0]}_{a[1]}_s{a[2]}': all(np.isclose(res[a]['validation'][k], e3c[b][k], rtol=1e-9, atol=1e-12)
                                                for k in ('median_percent', 'p95_percent', 'worst_percent'))
                   for a, b in pairs}
    assert all(consistency.values()), [k for k, v in consistency.items() if not v]

    arms = ['P0', 'P1', 'P2', 'P3', 'D', *[f'M-{p}@{b}' for b in BASES for p in ('S', 'R')]]
    curves = {}
    for arm in arms:
        budgets = (0,) if arm in BASES else LABEL_BUDGETS
        curves[arm] = {str(n): dict(median_percent_mean=float(np.mean([res[arm, n, s]['validation']['median_percent'] for s in SEEDS])),
                                    p95_percent_mean=float(np.mean([res[arm, n, s]['validation']['p95_percent'] for s in SEEDS])),
                                    p95_percent_range=[min(res[arm, n, s]['validation']['p95_percent'] for s in SEEDS),
                                                       max(res[arm, n, s]['validation']['p95_percent'] for s in SEEDS)],
                                    selected=[res[arm, n, s]['selected'] for s in SEEDS])
                       for n in budgets}
    base_vs_fsdt = {name: dict(median_percent_mean=float(np.mean([res[name, 0, s]['validation_vs_exact_fsdt']['median_percent'] for s in SEEDS])),
                               p95_percent_mean=float(np.mean([res[name, 0, s]['validation_vs_exact_fsdt']['p95_percent'] for s in SEEDS])))
                    for name in BASES}

    def seed_mean(arm, n):
        return np.mean([res[arm, n, s]['errors'] for s in SEEDS], axis=0)

    def paired(reference, candidate):
        diff = reference-candidate
        lo, hi = bootstrap_ci(diff, 'median')[0]
        return dict(median_difference_percent=float(100*np.median(diff)), ci95_percent=[float(100*lo), float(100*hi)],
                    supported=bool(lo > 0), better_designs=int(np.sum(diff > 0)), designs=int(len(diff)))
    new = f'M-S@{chosen}'
    development = dict(
        chosen_base=chosen, chosen_base_rule_v1=chosen_v1, choice_rule_v1=CHOICE_RULE, choice_rule_v2=CHOICE_RULE_V2,
        mean_base_scores_rule_v1=fit['mean_base_scores'], downstream_scores_rule_v2=fit['downstream_scores'],
        base_accuracy_vs_exact_fsdt=base_vs_fsdt,
        new_base_vs_P0=paired(seed_mean('P0', 0), seed_mean(chosen, 0)) if chosen != 'P0' else None,
        new_MS_vs_old_MS={str(n): paired(seed_mean('M-S@P0', n), seed_mean(new, n)) for n in LABEL_BUDGETS} if chosen != 'P0' else None,
        new_MS_vs_new_base={str(n): paired(seed_mean(chosen, 0), seed_mean(new, n)) for n in LABEL_BUDGETS},
        new_MS_vs_D={str(n): paired(seed_mean('D', n), seed_mean(new, n)) for n in LABEL_BUDGETS})

    # figure: base variants and new vs old corrections (validation)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3), layout='constrained')
    names = list(BASES)
    for k, stat in enumerate(('median', 'p95')):
        axes[0].bar(np.arange(4)+.38*k-.19, [curves[n]['0'][f'{stat}_percent_mean'] for n in names], width=.38,
                    label=f'vs TSDT, {stat}')
    axes[0].scatter(np.arange(4), [base_vs_fsdt[n]['p95_percent_mean'] for n in names], color='k', marker='x', zorder=3,
                    label='vs exact FSDT, p95 (approximation error)')
    axes[0].axhline(TARGET['p95_percent'], color='r', ls=':', lw=.8)
    axes[0].set(xticks=range(4), xticklabels=[f'{n}\n{BASES[n]["aspect"]} b/a\n{"prefactor" if BASES[n]["prefactor"] else "no prefactor"}' for n in names],
                yscale='log', ylabel='validation error (%)', title=f'E5: label-free base variants (chosen: {chosen})')
    axes[0].grid(alpha=.3, axis='y', which='both'); axes[0].legend(fontsize=7)
    for arm, color, lw in (('D', 'tab:orange', 1.2), ('M-S@P0', 'tab:gray', 1.2), (new, 'tab:green', 2.4)):
        axes[1].plot(LABEL_BUDGETS, [curves[arm][str(n)]['p95_percent_mean'] for n in LABEL_BUDGETS], 'o-', color=color, lw=lw,
                     label={'D': 'data only', 'M-S@P0': 'M-S on old base (E3c/E4)'}.get(arm, f'M-S on new base {chosen}'))
    axes[1].axhline(curves[chosen]['0']['p95_percent_mean'], color='tab:green', ls='--', lw=1, label=f'new base alone (0 labels)')
    axes[1].axhline(curves['P0']['0']['p95_percent_mean'], color='tab:gray', ls='--', lw=1, label='old base alone (0 labels)')
    axes[1].axhline(TARGET['p95_percent'], color='r', ls=':', lw=.8, label='target 3%')
    axes[1].set(xscale='log', yscale='log', xticks=LABEL_BUDGETS, xlabel='TSDT-labelled training designs',
                ylabel='95th percentile validation error (%)', title='E5: tail error, new vs old base')
    axes[1].set_xticklabels([str(b) for b in LABEL_BUDGETS]); axes[1].xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    axes[1].grid(alpha=.3, which='both'); axes[1].legend(fontsize=7)
    fig.savefig(out/'e5_base_and_tail.png', dpi=170); plt.close(fig)

    runs = [dict(arm=a, budget=n, seed=s, selected=r['selected'], validation=r['validation'],
                 **({'validation_vs_exact_fsdt': r['validation_vs_exact_fsdt']} if 'validation_vs_exact_fsdt' in r else {}))
            for (a, n, s), r in res.items()]
    np.savez_compressed(out/'e5_validation_errors.npz', validation_ids=fit['ids'][fit['val']],
                        **{f'{a}_{n}_s{s}': r['errors'] for (a, n, s), r in res.items()})
    manifest = dict(experiment='E5 development: label-free base model variants (pilot train/validation only)',
                    created_utc=datetime.now(timezone.utc).isoformat(),
                    pilot=dict(dataset_sha256=fit['dataset_sha256']), bases=BASES, choice_rule_v1=CHOICE_RULE,
                    choice_rule_v2=CHOICE_RULE_V2, superseded_run='results/e5_base_model (rule v1 only; kept as the record)',
                    grids=dict(degrees=DEGREES, ridges=RIDGES, correction_degrees=CORRECTION_DEGREES),
                    label_budgets=LABEL_BUDGETS, collocation_M=COLLOCATION_M, seeds=SEEDS, ridge_scope=SCOPE,
                    test_data_used='none', consistency_with_e3c=consistency, runs=runs, validation_curves=curves,
                    development=development, total_seconds=perf_counter()-start,
                    environment=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                                     matplotlib=matplotlib.__version__, platform=platform.platform()),
                    source_sha256={str(p.relative_to(ROOT)): sha256(p) for p in
                                   [*sorted((ROOT/'fgplate').glob('*.py')), Path(__file__)]})
    write_json(out/'manifest.json', manifest)
    write_json(out/'checksums.json', {p.name: sha256(p) for p in sorted(out.iterdir()) if p.is_file() and p.name != 'checksums.json'})
    print(json.dumps(dict(chosen=chosen, chosen_v1=chosen_v1, downstream=fit['downstream_scores'], mean_base_scores=fit['mean_base_scores'], base_vs_fsdt=base_vs_fsdt,
                          curves={a: {n: (round(v['median_percent_mean'], 3), round(v['p95_percent_mean'], 2)) for n, v in cc.items()}
                                  for a, cc in curves.items()}), indent=1), flush=True)


if __name__ == '__main__':
    main()
