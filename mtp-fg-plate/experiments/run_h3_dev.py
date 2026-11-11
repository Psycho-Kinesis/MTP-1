"""H3 development (pilot data and the FSDT solver only; no TSDT evaluation on any design-problem design).

Run from the project root:
    python experiments/run_h3_dev.py --output results/h3_development

1. Rebuild the frozen surrogates (h3_common.build): E5 arms asserted identical; static arms extended to 15x15 as in H2
   and asserted to reproduce H2's per-design uniform-pressure errors; frequency heads selected on pilot validation.
2. Describe surrogate accuracy on the 30 pilot validation designs (centre deflection under uniform pressure, frequency).
3. Truncation of the TSDT centre deflection on pilot training designs (odd modes <= 15, 25, 49, 99 vs 149).
4. FSDT feasibility scan for the main design problem (limits chosen so that both constraints matter).
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
from scipy.optimize import brentq
import scipy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from h3_common import (ARMS, EXT, PRIMARY_N, SEEDS, TRUTH_K, build, ceramic, centre_deflection, fsdt, mass,
                       square_modes, summary, sha256, write_json)
from fgplate.tsdt import fundamental_batch

MAIN = dict(a=0.5, aspect=1.5, q0=1.0e6, w_lim=1.4e-4, f_min=1200.0)


def h2_reproduction(sur):
    """Recompute H2's per-design errors of P-ext, C and D-ext (10 labels) on uniform pressure (L2); compare."""
    z = np.load(ROOT/'results'/'h2_loads'/'h2_test_set.npz')
    stored = np.load(ROOT/'results'/'h2_loads'/'h2_errors.npz')
    modes25, Q, H_true, test = z['modes'], z['load_coefficients'][:, 1], z['H_tsdt'], z['parameters']
    lookup = {tuple(m): i for i, m in enumerate(modes25)}
    idx = [lookup[tuple(m)] for m in square_modes(EXT)]
    true = Q[:, :, None]*H_true
    worst = 0.0
    for seed in SEEDS:
        for arm, name, N in (('P', 'P-ext', 0), ('C', 'C', PRIMARY_N), ('D', 'D-ext', PRIMARY_N)):
            H = np.zeros_like(H_true)
            H[:, idx] = sur[seed].transfer(arm, test)
            e = np.linalg.norm((Q[:, :, None]*H-true).reshape(len(test), -1), axis=1)/np.linalg.norm(true.reshape(len(test), -1), axis=1)
            ref = stored[f'L2_{name}_{N}_s{seed}']
            assert np.allclose(e, ref, rtol=1e-9, atol=1e-13), (arm, seed)
            worst = max(worst, float(np.max(np.abs(e-ref))))
    return worst


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--pilot', type=Path, default=ROOT/'results'/'pilot_200')
    parser.add_argument('--development', type=Path, default=ROOT/'results'/'e5_base_model_v2'/'manifest.json')
    parser.add_argument('--output', type=Path, default=ROOT/'results'/'h3_development')
    args = parser.parse_args()
    out = args.output
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f'{out} is not empty')
    out.mkdir(parents=True, exist_ok=True)
    start = perf_counter()

    fit, sur, selection = build(args.pilot, args.development, log=lambda m: print(m, flush=True))
    worst_h2 = h2_reproduction(sur)
    print(f'H2 uniform-pressure errors reproduced (max abs diff {worst_h2:.1e})', flush=True)
    P, val, train = fit['P'], fit['val'], fit['train']

    # ---- 2. accuracy on pilot validation designs (descriptive; nothing here is selected)
    w_T, f_T = centre_deflection(P[val], 'TSDT'), fundamental_batch(P[val], 'TSDT')
    w_F, f_F = fsdt(P[val])
    accuracy = {'exact FSDT': dict(deflection=summary(np.abs(w_F/w_T-1)), frequency=summary(np.abs(f_F/f_T-1)))}
    for arm in ARMS:
        for seed in SEEDS:
            w, f = sur[seed].evaluate(arm, P[val])
            accuracy[f'{arm} s{seed}'] = dict(deflection=summary(np.abs(w/w_T-1)), frequency=summary(np.abs(f/f_T-1)))
    print('validation accuracy done', flush=True)

    # ---- 3. truncation of the truth on pilot training designs
    ref = centre_deflection(P[train], 'TSDT', 149)
    truncation = {str(k): dict(max_relative=float(np.max(np.abs(centre_deflection(P[train], 'TSDT', k)/ref-1))),
                               median_relative=float(np.median(np.abs(centre_deflection(P[train], 'TSDT', k)/ref-1))))
                  for k in (15, 25, 49, 99)}

    # ---- 4. FSDT feasibility scan of the main problem (minimum thickness for each constraint, per g)
    a, lam, q0 = MAIN['a'], MAIN['aspect'], MAIN['q0']

    def w_phys(g, r):
        return q0*a*fsdt(np.array([[g, r, lam]]))[0][0]*r**3/380e9

    def f_phys(g, r):
        return fsdt(np.array([[g, r, lam]]))[1][0]*r/(2*np.pi*a)*1e4
    gs = np.expm1(np.linspace(0, np.log(9), 25))
    scan = []
    for g in gs:
        r_w = brentq(lambda r: w_phys(g, r)-MAIN['w_lim'], 5, 100) if w_phys(g, 5) < MAIN['w_lim'] else None
        r_f = brentq(lambda r: f_phys(g, r)-MAIN['f_min'], 5, 100) if f_phys(g, 5) > MAIN['f_min'] else None
        r = min(x for x in (r_w, r_f) if x is not None) if (r_w and r_f) else None
        scan.append(dict(g=float(g), a_over_h_deflection=r_w, a_over_h_frequency=r_f,
                         h_required=None if r is None else a/r, mass=None if r is None else float(mass(a/r, g)),
                         ceramic=None if r is None else float(ceramic(a/r, g)),
                         active=None if r is None else ('deflection' if r == r_w else 'frequency')))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout='constrained')
    ok = [s for s in scan if s['mass'] is not None]
    axes[0].plot([s['g'] for s in ok], [1000*a/s['a_over_h_deflection'] for s in ok], label='deflection limit')
    axes[0].plot([s['g'] for s in ok], [1000*a/s['a_over_h_frequency'] for s in ok], label='frequency limit')
    axes[0].set(xlabel='gradient index g', ylabel='minimum thickness h (mm)', title='Main problem, FSDT scan: thickness needed', xscale='symlog')
    axes[0].legend()
    axes[1].plot([1000*s['ceramic'] for s in ok], [s['mass'] for s in ok], 'o-')
    for s in ok[::4]:
        axes[1].annotate(f"g={s['g']:.2g}", (1000*s['ceramic'], s['mass']), fontsize=7)
    axes[1].set(xlabel='ceramic use h/(g+1) (mm)', ylabel='areal mass (kg/m$^2$)', title='FSDT: lightest plate for each g')
    for ax in axes:
        ax.grid(alpha=.3)
    fig.savefig(out/'h3_main_problem_fsdt_scan.png', dpi=150)
    plt.close(fig)

    manifest = dict(experiment='H3 development (pilot data and FSDT only)', created_utc=datetime.now(timezone.utc).isoformat(),
                    pilot_dataset_sha256=fit['dataset_sha256'], e5_development_sha256=sha256(args.development),
                    h2_reproduction_max_abs_difference=worst_h2, frequency_heads={str(s): selection[s] for s in SEEDS},
                    validation_accuracy=accuracy, truth_truncation=truncation, truth_modes=f'odd m, n <= {TRUTH_K}',
                    main_problem=MAIN, main_problem_fsdt_scan=scan, arms=ARMS, total_seconds=perf_counter()-start,
                    environment=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                                     matplotlib=matplotlib.__version__, platform=platform.platform()),
                    source_sha256={str(p.relative_to(ROOT)): sha256(p) for p in sorted((ROOT/'fgplate').glob('*.py'))
                                   + [ROOT/'experiments'/'h3_common.py', Path(__file__).resolve()]})
    write_json(out/'manifest.json', manifest)
    print(f'done ({perf_counter()-start:.0f} s)')


if __name__ == '__main__':
    main()
