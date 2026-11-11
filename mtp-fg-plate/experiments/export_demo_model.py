"""Export the frozen H3 surrogates (seed 11) and key results for the browser demo (demo/fg_plate_lab.html).

Run from the project root:
    python experiments/export_demo_model.py            # writes demo/fg_plate_lab_data.json

Only the transverse-deflection component (w) of the static transfer functions is exported, for all modes
m, n <= 15, plus the frequency heads. Reference values from the Python solvers and surrogates are included
so the JavaScript implementation can be tested against them (demo/test_demo_math.js).
"""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np

from h3_common import build, centre_deflection, fsdt, square_modes, truth
from fgplate.transfer import DOMAIN, exponents
from fgplate.tsdt import fundamental_batch, transfer_batch

SEED = 11


def rounded(a, digits=12):
    return [float(f'{v:.{digits}g}') for v in np.asarray(a, dtype=float).ravel()]


def main(out=ROOT/'demo'/'fg_plate_lab_data.json'):
    fit, sur, selection = build(ROOT/'results'/'pilot_200', ROOT/'results'/'e5_base_model_v2'/'manifest.json', log=lambda m: None)
    s = sur[SEED]
    modes = square_modes(15)
    model = dict(seed=SEED, modes=modes.tolist(), domain=DOMAIN.tolist())
    for arm in ('P', 'C', 'D'):
        theta, degree = s.static[arm]
        model[arm] = dict(degree=int(degree), exponents=[list(p) for p in exponents(degree)],
                          theta_w=[rounded(theta[k, 2]) for k in range(len(modes))])
    for arm in ('C', 'D'):
        theta, degree = s.freq[arm]
        model['freq_'+arm] = dict(degree=int(degree), exponents=[list(p) for p in exponents(degree)], theta=rounded(theta))

    # reference values for the JavaScript tests
    rng = np.random.default_rng(5)
    designs = np.column_stack([np.expm1(rng.uniform(0, np.log(9), 12)), np.exp(rng.uniform(np.log(5), np.log(100), 12)),
                               rng.uniform(0.5, 2, 12)])
    designs = np.vstack([designs, [[0, 5, 1], [8, 100, 2], [1, 10, 1], [4, 5, 1]]])
    w_T25, f_T = centre_deflection(designs, 'TSDT', 25), fundamental_batch(designs, 'TSDT')
    w_F25, f_F = centre_deflection(designs, 'FSDT', 25), fundamental_batch(designs, 'FSDT')
    tests = dict(designs=designs.tolist(), tsdt_centre_uniform_odd25=w_T25.tolist(), fsdt_centre_uniform_odd25=w_F25.tolist(),
                 tsdt_omega_hat=f_T.tolist(), fsdt_omega_hat=f_F.tolist(),
                 tsdt_H_w_modes=[[1, 1], [2, 3], [7, 5]], tsdt_H_w=transfer_batch(designs, [(1, 1), (2, 3), (7, 5)], 'TSDT')[:, :, 2].tolist())
    for arm in ('P', 'C', 'D'):
        w, f = s.evaluate(arm, designs)
        tests[f'surrogate_{arm}_centre_uniform'] = w.tolist()
        tests[f'surrogate_{arm}_omega_hat'] = f.tolist()

    # evidence data from the saved results
    e5 = json.loads((ROOT/'results'/'e5_confirmation'/'manifest.json').read_text())['fresh_curves']
    curves = {arm: {N: dict(median=v['median_percent_mean'], p95=v['p95_percent_mean']) for N, v in e5[arm].items()}
              for arm in ('P2', 'D', 'M-S@P2')}
    h2 = json.loads((ROOT/'results'/'h2_loads'/'manifest.json').read_text())
    h2_table = {fam: {arm: {N: dict(median=v['median_percent_mean'], p95=v['p95_percent_mean']) for N, v in h2['table'][fam][arm].items()}
                      for arm in ('A', 'C', 'P-ext', 'D-ext')} for fam in h2['families']}
    h3 = json.loads((ROOT/'results'/'h3_inverse_design'/'manifest.json').read_text())
    part_a = json.loads((ROOT/'results'/'h3_inverse_design'/'part_a_main_problem.json').read_text())
    front = [dict(ceramic_max_mm=None if r['problem']['ceramic_max'] is None else 1000*r['problem']['ceramic_max'],
                  mass=r['reference']['mass'], ceramic_mm=1000*r['reference']['ceramic'], g=r['reference']['g'],
                  a_over_h=r['reference']['a_over_h']) for r in part_a]
    evidence = dict(e5_fresh_curves=curves, h2_table=h2_table, h2_families=h2['families'],
                    h3_decisions=h3['decisions'], h3_secondary={k: h3['secondary'][k] for k in ('C', 'P', 'D', 'direct_slsqp')},
                    h3_main_problem=h3['main_problem'], h3_reference_front=front)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(model=model, tests=tests, evidence=evidence), separators=(',', ':')))
    print(f'wrote {out} ({out.stat().st_size/1e3:.0f} kB); degrees:',
          {k: model[k]['degree'] for k in ('P', 'C', 'D', 'freq_C', 'freq_D')})


if __name__ == '__main__':
    main()
