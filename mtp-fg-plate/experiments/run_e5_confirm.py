"""E5 confirmation: the improved base model on a NEW fresh, locked test set.

Run from the project root:
    python experiments/run_e5_confirm.py --output results/e5_confirmation

Follows docs/E5_PREREGISTRATION.md (sha256 asserted): refit the frozen E5 arms with
run_e5.fit_all (pilot only) and assert they reproduce results/e5_base_model_v2; only then
generate 200 fresh designs (pilot-1.1, seed 20261215), disjoint from the pilot, all
collocation designs and the E4 fresh set; evaluate every frozen arm once; answer C1-C6.
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
import scipy
import matplotlib
import matplotlib.ticker
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from fgplate.baseline import bootstrap_ci
from fgplate.fields import reconstruct_analytic, relative_l2
from fgplate.pilot import GENERATOR_VERSION, design_loads, load_key, sample_designs
from fgplate.transfer import design_coordinates
from fgplate.tsdt import transfer
from run_e5 import LABEL_BUDGETS, SEEDS, TARGET, fit_all, sha256, summary, write_json

PREREGISTRATION = ROOT/'docs'/'E5_PREREGISTRATION.md'
PREREGISTRATION_SHA256 = 'f25d9ea8964659e86be4519d543471c39f77aa222f2dbed88fe5bb056779864e'
FRESH_SEED, FRESH_COUNT = 20261215, 200
ARMS = ('P0', 'P2', 'D', 'M-S@P0', 'M-S@P2', 'M-R@P2')
PRIMARY, NEW_BASE, OLD_PIPELINE = 'M-S@P2', 'P2', 'M-S@P0'
NAMES = {'P0': 'old base alone (0 labels)', 'P2': 'new base alone (0 labels)', 'D': 'data only (random TSDT labels)',
         'M-S@P0': 'old pipeline M-S@P0 (E3c/E4)', 'M-S@P2': 'new pipeline M-S@P2 (primary)',
         'M-R@P2': 'new base + correction, random labels'}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--pilot', type=Path, default=ROOT/'results'/'pilot_200')
    parser.add_argument('--development', type=Path, default=ROOT/'results'/'e5_base_model_v2'/'manifest.json')
    parser.add_argument('--e4', type=Path, default=ROOT/'results'/'e4_confirmation')
    parser.add_argument('--output', type=Path, default=ROOT/'results'/'e5_confirmation')
    args = parser.parse_args()
    assert sha256(PREREGISTRATION) == PREREGISTRATION_SHA256, 'pre-registration file changed; E5 confirmation must not run'
    out = args.output.resolve()
    if out.exists() and any(out.iterdir()):
        parser.error('output directory is not empty; choose a new experiment directory')
    out.mkdir(parents=True, exist_ok=True)
    start = perf_counter()

    # ------------------------------------------------ 1. refit frozen arms, assert identical to development
    fit = fit_all(args.pilot.resolve(), log=lambda m: print(m, flush=True))
    assert fit['chosen'] == NEW_BASE, f'rule v2 no longer chooses {NEW_BASE}'
    dev = {(r['arm'], r['budget'], r['seed']): r['validation'] for r in json.loads(args.development.read_text())['runs']}
    consistency = {}
    for key, result in fit['results'].items():
        ref = dev[key]
        consistency[f'{key[0]}_{key[1]}_s{key[2]}'] = all(np.isclose(result['validation'][k], ref[k], rtol=1e-9, atol=1e-12)
                                                          for k in ('median_percent', 'p95_percent', 'worst_percent'))
    assert all(consistency.values()), [k for k, v in consistency.items() if not v]
    print(f'all {len(consistency)} arms reproduce E5 development', flush=True)
    modes, basis, models = fit['modes'], fit['basis'], fit['models']

    # ------------------------------------------------ 2. new fresh locked test set
    fresh, fresh_ids, _ = sample_designs(FRESH_COUNT, FRESH_SEED, GENERATOR_VERSION)
    xf = design_coordinates(fresh)
    e4_fresh = np.load(args.e4/'fresh_test_set.npz', allow_pickle=False)['parameters']
    others = [('pilot', fit['P']), ('E4 fresh set', e4_fresh), *[(f'collocation s{s}', fit['colloc'][s]) for s in SEEDS]]
    for name, other in others:
        assert np.min(np.linalg.norm(xf[:, None]-design_coordinates(other)[None], axis=2)) > 1e-6, f'overlap with {name}'
    c_f = np.array([[design_loads(load_key(GENERATOR_VERSION, i, fresh_ids[i]), FRESH_SEED)[0][2][tuple(map(int, m))]
                     for m in modes] for i in range(FRESH_COUNT)])
    H_tsdt_f, H_fsdt_f = transfer(fresh, modes, 'TSDT'), transfer(fresh, modes, 'FSDT')
    np.savez_compressed(out/'fresh_test_set.npz', parameters=fresh, design_ids=fresh_ids, modes=modes,
                        mixture_coefficients=c_f, H_tsdt=H_tsdt_f, H_fsdt=H_fsdt_f, seed=FRESH_SEED,
                        generator_version=GENERATOR_VERSION)

    def fields(H):
        return reconstruct_analytic(np.transpose(H*c_f[:, :, None], (0, 2, 1)), basis)
    truth, fsdt_fields = fields(H_tsdt_f), fields(H_fsdt_f)

    def errors(H, w_only=False, reference=None):
        F, T = fields(H), (truth if reference is None else reference)
        return relative_l2(F[:, 2], T[:, 2]) if w_only else relative_l2(F, T)

    # ------------------------------------------------ 3. single evaluation
    runs, errs = [], {}
    for arm in ARMS:
        for N in ((0,) if arm in ('P0', 'P2') else LABEL_BUDGETS):
            for seed in SEEDS:
                H = models[arm, N, seed](fresh, modes)
                e = errors(H)
                errs[arm, N, seed] = e
                runs.append(dict(arm=arm, budget=N, seed=seed, selected=fit['results'][arm, N, seed]['selected'],
                                 fresh=summary(e), fresh_w_only=summary(errors(H, True)),
                                 **({'fresh_vs_exact_fsdt': summary(errors(H, reference=fsdt_fields))} if arm in ('P0', 'P2') else {})))
    gap = dict(five_field=summary(errors(H_fsdt_f)), w_only=summary(errors(H_fsdt_f, True)))

    def seed_mean(arm, N):
        return np.mean([errs[arm, N, s] for s in SEEDS], axis=0)

    def per_seed(arm, N, stat):
        return [next(r for r in runs if (r['arm'], r['budget'], r['seed']) == (arm, N, s))['fresh'][f'{stat}_percent'] for s in SEEDS]

    curves = {arm: {str(N): dict(median_percent_mean=float(np.mean(per_seed(arm, N, 'median'))),
                                 p95_percent_mean=float(np.mean(per_seed(arm, N, 'p95'))),
                                 p95_percent_range=[min(per_seed(arm, N, 'p95')), max(per_seed(arm, N, 'p95'))],
                                 seed_averaged=summary(seed_mean(arm, N)))
                    for N in ((0,) if arm in ('P0', 'P2') else LABEL_BUDGETS)} for arm in ARMS}

    def paired(reference, candidate):
        diff = reference-candidate
        lo, hi = bootstrap_ci(diff, 'median')[0]
        return dict(median_difference_percent=float(100*np.median(diff)), ci95_percent=[float(100*lo), float(100*hi)],
                    confirmed=bool(lo > 0), better_designs=int(np.sum(diff > 0)), designs=int(len(diff)))
    C1 = {str(N): paired(seed_mean(NEW_BASE, 0), seed_mean(PRIMARY, N)) for N in LABEL_BUDGETS}
    C2 = {str(N): paired(seed_mean('D', N), seed_mean(PRIMARY, N)) for N in LABEL_BUDGETS}
    C6 = {str(N): paired(seed_mean(OLD_PIPELINE, N), seed_mean(PRIMARY, N)) for N in LABEL_BUDGETS}
    prim = curves[PRIMARY]
    decisions = dict(
        C1=dict(per_N=C1, confirmed=all(C1[str(n)]['confirmed'] for n in (10, 20, 40))),
        C2=dict(per_N=C2, confirmed=all(C2[str(n)]['confirmed'] for n in (10, 20, 40))),
        C3=dict(median_percent_mean=prim['10']['median_percent_mean'], confirmed=prim['10']['median_percent_mean'] <= TARGET['median_percent']),
        C4=dict(p95_percent_mean=prim['10']['p95_percent_mean'], confirmed=prim['10']['p95_percent_mean'] <= TARGET['p95_percent']),
        C5=dict(p95_percent_mean_by_N={str(n): prim[str(n)]['p95_percent_mean'] for n in (10, 20, 40, 70)},
                seed_averaged_p95_ci95_N10=prim['10']['seed_averaged']['p95_percent_ci95'],
                confirmed=all(prim[str(n)]['p95_percent_mean'] <= TARGET['p95_percent'] for n in (10, 20, 40, 70))
                and prim['10']['seed_averaged']['p95_percent_ci95'][1] <= TARGET['p95_percent']),
        C6=dict(per_N=C6, confirmed=all(C6[str(n)]['confirmed'] for n in (10, 20, 40))),
        secondary=dict(
            base_accuracy_vs_exact_fsdt={b: dict(median_percent_mean=float(np.mean([r['fresh_vs_exact_fsdt']['median_percent'] for r in runs if r['arm'] == b])),
                                                 p95_percent_mean=float(np.mean([r['fresh_vs_exact_fsdt']['p95_percent'] for r in runs if r['arm'] == b])))
                                         for b in ('P0', 'P2')},
            primary_vs_D_at_140=C2['140'],
            worst_primary_N10=[dict(error_percent=float(100*seed_mean(PRIMARY, 10)[i]), g=float(fresh[i, 0]),
                                    a_over_h=float(fresh[i, 1]), b_over_a=float(fresh[i, 2]))
                               for i in np.argsort(seed_mean(PRIMARY, 10))[::-1][:5]],
            e4_primary_for_comparison=json.loads((args.e4/'manifest.json').read_text())['fresh_curves']['M-S']))

    # ------------------------------------------------ figures
    colors = {'P0': 'tab:gray', 'P2': 'tab:green', 'D': 'tab:orange', 'M-S@P0': 'tab:gray', 'M-S@P2': 'tab:green', 'M-R@P2': 'lightgreen'}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), layout='constrained')
    for ax, stat, target in zip(axes, ('median', 'p95'), (TARGET['median_percent'], TARGET['p95_percent'])):
        for arm in ('D', 'M-S@P0', 'M-R@P2', 'M-S@P2'):
            mean = [curves[arm][str(n)][f'{stat}_percent_mean'] for n in LABEL_BUDGETS]
            ax.plot(LABEL_BUDGETS, mean, 'o-', color=colors[arm], lw=2.4 if arm == PRIMARY else 1.2,
                    ls='--' if arm == 'M-S@P0' else '-', label=NAMES[arm])
        for b in ('P0', 'P2'):
            ax.axhline(curves[b]['0'][f'{stat}_percent_mean'], color=colors[b], ls=':', lw=1.2,
                       label=f'{NAMES[b]}: {curves[b]["0"][f"{stat}_percent_mean"]:.2f}%')
        ax.axhline(gap['five_field'][f'{stat}_percent'], color='k', ls='-.', lw=1, label=f'exact FSDT (model-form gap): {gap["five_field"][f"{stat}_percent"]:.2f}%')
        ax.axhline(target, color='r', ls=':', lw=.8, label=f'target {target}%')
        ax.set(xscale='log', yscale='log', xticks=LABEL_BUDGETS, xlabel='TSDT-labelled training designs',
               ylabel=f'{"Median" if stat == "median" else "95th percentile"} error vs TSDT (%)',
               title=f'E5 confirmation: {FRESH_COUNT} new fresh designs ({"median" if stat == "median" else "95th percentile"})')
        ax.set_xticklabels([str(b) for b in LABEL_BUDGETS]); ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.grid(alpha=.3, which='both'); ax.legend(fontsize=6.5)
    fig.savefig(out/'e5_fresh_test_curves.png', dpi=170); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.8, 4.3), layout='constrained')
    ax.scatter(fresh[:, 1], 100*errors(H_fsdt_f), marker='x', c='k', s=14, label='exact FSDT (model-form gap)')
    ax.scatter(fresh[:, 1], 100*seed_mean('M-S@P0', 10), marker='v', s=14, color='tab:gray', alpha=.6, label=NAMES['M-S@P0']+', N=10')
    ax.scatter(fresh[:, 1], 100*seed_mean(PRIMARY, 10), marker='^', s=16, color='tab:green', alpha=.85, label=NAMES[PRIMARY]+', N=10')
    ax.axhline(TARGET['p95_percent'], color='r', ls=':', lw=.8)
    ax.set(xscale='log', yscale='log', xlabel='slenderness a/h', ylabel='error vs TSDT (%), seed-averaged',
           title=f'E5 confirmation: per-design error ({FRESH_COUNT} new fresh designs)')
    ax.grid(alpha=.3, which='both'); ax.legend(fontsize=7)
    fig.savefig(out/'e5_error_vs_thickness.png', dpi=170); plt.close(fig)

    np.savez_compressed(out/'e5_confirmation_errors.npz', fresh_ids=fresh_ids,
                        **{f'{a}_{n}_s{s}': e for (a, n, s), e in errs.items()})
    manifest = dict(experiment='E5 confirmation: improved label-free base on a new fresh locked test set',
                    created_utc=datetime.now(timezone.utc).isoformat(),
                    preregistration=dict(path='docs/E5_PREREGISTRATION.md', sha256=PREREGISTRATION_SHA256),
                    development_reference=dict(path=str(args.development), sha256=sha256(args.development)),
                    pilot=dict(dataset_sha256=fit['dataset_sha256']),
                    fresh_test_set=dict(count=FRESH_COUNT, seed=FRESH_SEED, generator=GENERATOR_VERSION, file='fresh_test_set.npz'),
                    frozen_arms_reproduce_development=consistency, model_form_gap_fresh=gap, runs=runs,
                    fresh_curves=curves, decisions=decisions, total_seconds=perf_counter()-start,
                    environment=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                                     matplotlib=matplotlib.__version__, platform=platform.platform()),
                    source_sha256={str(p.relative_to(ROOT)): sha256(p) for p in
                                   [*sorted((ROOT/'fgplate').glob('*.py')), ROOT/'experiments'/'run_e5.py', Path(__file__)]})
    write_json(out/'manifest.json', manifest)
    write_json(out/'checksums.json', {p.name: sha256(p) for p in sorted(out.iterdir()) if p.is_file() and p.name != 'checksums.json'})
    print(json.dumps({k: v.get('confirmed') for k, v in decisions.items() if k.startswith('C')}, indent=1), flush=True)


if __name__ == '__main__':
    main()
