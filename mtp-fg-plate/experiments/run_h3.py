"""H3: surrogate-guided inverse design, verified with TSDT (pre-registered).

Run from the project root:
    python experiments/run_h3.py --output results/h3_inverse_design

Follows docs/H3_PREREGISTRATION.md (sha256 asserted):
1. rebuild the frozen surrogates (E5 arms and H2 errors asserted identical; frequency heads asserted equal to the
   development record results/h3_development);
2. Part A: main problem, Pareto front over ceramic-use limits (descriptive);
3. Part B: 100 random problems generated now (FSDT-only limits, seed 20270101); TSDT reference optimum, direct TSDT
   SLSQP, and open-/closed-loop designs of every arm and seed, each verified with TSDT; decisions Q1-Q5.
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
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import h3_common as H
from h3_common import (ARMS, FEAS_TOL, PRIMARY_N, SEEDS, build, closed_loop, direct_slsqp, max_deflection_ratio,
                       optimize, random_problems, sha256, truth, verify, write_json)
from fgplate.baseline import bootstrap_ci
from run_h3_dev import MAIN, h2_reproduction

PREREGISTRATION = ROOT/'docs'/'H3_PREREGISTRATION.md'
PREREGISTRATION_SHA256 = '83f35c3ee0526a3ee6797f3b201d120d5f90529b78fb12df4ae2813bc98f0251'
PROBLEM_SEED, PROBLEM_COUNT = 20270101, 100
BUDGETS_MM = (None, 40, 35, 30, 25, 20, 15, 10)


def jsonable(problem):
    out = {k: v for k, v in problem.items()}
    out['ceramic_max'] = None if not np.isfinite(problem['ceramic_max']) else float(problem['ceramic_max'])
    return out


def run_problem(problem, sur, tag, log):
    """Reference, direct SLSQP and all arms (open + closed loop) for one problem. Returns a record."""
    t0 = perf_counter()
    ref = optimize(truth, problem, key=('TSDT',))
    rec = dict(problem=jsonable(problem), reference=ref)
    if ref['status'] == 'infeasible':
        rec['excluded'] = True
        return rec
    p_ref = np.array([[ref['g'], ref['a_over_h'], problem['aspect']]])
    rec['reference_max_over_centre'] = float(max_deflection_ratio(p_ref)[0])
    rec['reference_w_over_h'] = ref['w']/ref['h']
    rec['reference_active'] = [bool(abs(c) < 1e-6) for c in ref['constraints']]
    rec['direct_slsqp'] = direct_slsqp(truth, problem)
    rec['direct_slsqp']['mass_gap'] = rec['direct_slsqp']['mass']/ref['mass']-1
    rec['arms'] = {}
    for arm in ARMS:
        for seed in SEEDS:
            evaluate = (lambda s, a: (lambda p: sur[s].evaluate(a, p)))(seed, arm)
            opt = optimize(evaluate, problem, key=(arm, seed))
            r = dict(open=opt)
            if opt['status'] == 'infeasible':
                r.update(open_violation=None, open_gap=None, decision_error=1.0)
            else:
                p = np.array([[opt['g'], opt['a_over_h'], problem['aspect']]])
                w, f, c, v = verify(problem, p)
                gap = opt['mass']/ref['mass']-1
                r.update(open_violation=float(v[0]), open_gap=float(gap), decision_error=float(v[0]+abs(gap)),
                         open_tsdt=dict(w=float(w[0]), f=float(f[0]), constraints=c[0].tolist()))
            cl = closed_loop(evaluate, problem, key=(arm, seed), first=opt)
            if cl['status'] == 'feasible':
                cl['final_gap'] = cl['design']['mass']/ref['mass']-1
            r['closed'] = cl
            rec['arms'][f'{arm} s{seed}'] = r
    rec['seconds'] = perf_counter()-t0
    log(f"{tag}: ref mass {ref['mass']:.4g} (g {ref['g']:.3f}, a/h {ref['a_over_h']:.2f}); "
        f"C s11 v {rec['arms']['C s11']['open_violation']}, gap {rec['arms']['C s11']['open_gap']}; {rec['seconds']:.0f} s")
    return rec


def pct(x):
    return None if x is None else 100*x


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--pilot', type=Path, default=ROOT/'results'/'pilot_200')
    parser.add_argument('--development', type=Path, default=ROOT/'results'/'e5_base_model_v2'/'manifest.json')
    parser.add_argument('--h3dev', type=Path, default=ROOT/'results'/'h3_development'/'manifest.json')
    parser.add_argument('--output', type=Path, default=ROOT/'results'/'h3_inverse_design')
    args = parser.parse_args()
    assert sha256(PREREGISTRATION) == PREREGISTRATION_SHA256, 'pre-registration file changed; H3 must not run'
    out = args.output
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f'{out} is not empty')
    out.mkdir(parents=True, exist_ok=True)
    start = perf_counter()
    log = lambda m: print(m, flush=True)

    # ------------------------------------------------ 1. frozen surrogates
    fit, sur, selection = build(args.pilot, args.development, log=log)
    h2_reproduction(sur)
    dev = json.loads(args.h3dev.read_text())['frequency_heads']
    for seed in SEEDS:
        for arm in ('C', 'D'):
            for k in ('degree', 'ridge'):
                assert dev[str(seed)][arm][k] == selection[seed][arm][k]
            assert np.isclose(dev[str(seed)][arm]['validation']['p95_percent'], selection[seed][arm]['validation']['p95_percent'], rtol=1e-12)
        assert dev[str(seed)]['labelled_designs'] == selection[seed]['labelled_designs']
    log('surrogates reproduce development (E5 arms, H2 errors, frequency heads)')

    # ------------------------------------------------ 2. Part A: main problem, Pareto front
    part_a = []
    for budget in BUDGETS_MM:
        problem = dict(MAIN, ceramic_max=np.inf if budget is None else budget/1000)
        part_a.append(run_problem(problem, sur, f'A C_max={budget} mm', log))
    H.GRID_CACHE.clear()

    # ------------------------------------------------ 3. Part B: random problems (generated only now)
    problems = random_problems(PROBLEM_COUNT, PROBLEM_SEED)
    part_b = []
    for problem in problems:
        part_b.append(run_problem(problem, sur, f"B{problem['index']:03d}", log))
        H.GRID_CACHE.clear()
    kept = [r for r in part_b if not r.get('excluded')]

    # ------------------------------------------------ 4. decisions
    def collect(arm, field):
        return np.array([[r['arms'][f'{arm} s{s}'][field] if r['arms'][f'{arm} s{s}'][field] is not None else np.nan
                          for s in SEEDS] for r in kept])
    V = {arm: collect(arm, 'open_violation') for arm in ARMS}
    G = {arm: collect(arm, 'open_gap') for arm in ARMS}
    E = {arm: collect(arm, 'decision_error') for arm in ARMS}

    def paired(ref_arm, cand_arm):
        diff = E[ref_arm].mean(axis=1)-E[cand_arm].mean(axis=1)
        lo, hi = bootstrap_ci(diff, 'median')[0]
        return dict(median_difference_percent=100*float(np.median(diff)), ci95_percent=[100*float(lo), 100*float(hi)],
                    confirmed=bool(lo > 0), better_problems=int(np.sum(diff > 0)), problems=int(len(diff)))

    def closed_stats(arm):
        rows = [[r['arms'][f'{arm} s{s}']['closed'] for s in SEEDS] for r in kept]
        feasible = np.array([[c['status'] == 'feasible' for c in row] for row in rows])
        analyses = np.array([[c['analyses'] for c in row] for row in rows], dtype=float)
        gaps = np.array([[abs(c['final_gap']) if c['status'] == 'feasible' else np.nan for c in row] for row in rows])
        return feasible, analyses, gaps
    vC, gC = V['C'].ravel(), np.abs(G['C'].ravel())
    q1 = dict(p95_violation_percent=100*float(np.nanquantile(vC, .95)), p95_abs_gap_percent=100*float(np.nanquantile(gC, .95)),
              median_violation_percent=100*float(np.nanmedian(vC)), median_abs_gap_percent=100*float(np.nanmedian(gC)),
              infeasible_surrogate_problems=int(np.isnan(vC).sum()))
    q1['confirmed'] = bool(q1['infeasible_surrogate_problems'] == 0 and q1['p95_violation_percent'] <= 0.5 and q1['p95_abs_gap_percent'] <= 0.5)
    feas, analyses, gaps = closed_stats('C')
    q4 = dict(all_feasible=bool(feas.all()), mean_analyses_per_seed={str(s): float(analyses[:, j].mean()) for j, s in enumerate(SEEDS)},
              p95_final_abs_gap_percent=100*float(np.nanquantile(gaps, .95)), max_analyses=int(analyses.max()))
    q4['confirmed'] = bool(q4['all_feasible'] and max(q4['mean_analyses_per_seed'].values()) <= 2.0 and q4['p95_final_abs_gap_percent'] <= 0.5)
    direct_total = int(sum(r['direct_slsqp']['analyses'] for r in kept))
    c_total = {str(s): int(PRIMARY_N+analyses[:, j].sum()) for j, s in enumerate(SEEDS)}
    q5 = dict(c_total_tsdt_analyses=c_total, direct_slsqp_total_tsdt_analyses=direct_total,
              worst_seed_ratio=max(c_total.values())/direct_total)
    q5['confirmed'] = bool(q5['worst_seed_ratio'] <= 0.10)
    decisions = dict(Q1=q1, Q2=paired('P', 'C'), Q3=paired('D', 'C'), Q4=q4, Q5=q5)

    # ------------------------------------------------ 5. secondary
    secondary = {}
    for arm in ARMS:
        feas_a, an_a, gap_a = closed_stats(arm)
        secondary[arm] = dict(
            open_violation_percent=dict(median=pct(float(np.nanmedian(V[arm]))), p95=pct(float(np.nanquantile(V[arm], .95))),
                                        max=pct(float(np.nanmax(V[arm]))), share_violating_over_tol=float(np.nanmean(V[arm] > FEAS_TOL))),
            open_abs_gap_percent=dict(median=pct(float(np.nanmedian(np.abs(G[arm])))), p95=pct(float(np.nanquantile(np.abs(G[arm]), .95))),
                                      max=pct(float(np.nanmax(np.abs(G[arm]))))),
            decision_error_percent=dict(median=pct(float(np.median(E[arm]))), p95=pct(float(np.quantile(E[arm], .95)))),
            infeasible_surrogate_problems=int(np.isnan(V[arm]).sum()),
            closed_loop=dict(feasible_share=float(feas_a.mean()), mean_analyses=float(an_a.mean()),
                             max_analyses=int(an_a.max()), analyses_histogram={str(k): int((an_a == k).sum()) for k in range(1, 6)},
                             p95_final_abs_gap_percent=pct(float(np.nanquantile(gap_a, .95))) if np.isfinite(gap_a).any() else None,
                             total_analyses_including_labels={str(s): int((0 if arm == 'P' else PRIMARY_N)+an_a[:, j].sum())
                                                              for j, s in enumerate(SEEDS)}))
    ds = [r['direct_slsqp'] for r in kept]
    secondary['direct_slsqp'] = dict(mean_analyses=float(np.mean([d['analyses'] for d in ds])),
                                     success_share=float(np.mean([d['feasible'] and abs(d['mass_gap']) <= 1e-3 for d in ds])),
                                     feasible_share=float(np.mean([d['feasible'] for d in ds])),
                                     median_abs_gap_percent=100*float(np.median([abs(d['mass_gap']) for d in ds])))
    secondary['reference'] = dict(mean_evaluations=float(np.mean([r['reference']['evaluations'] for r in kept])),
                                  active_constraint_counts={name: int(sum(r['reference_active'][k] for r in kept))
                                                            for k, name in enumerate(('deflection', 'frequency', 'ceramic'))},
                                  max_w_over_h=float(max(r['reference_w_over_h'] for r in kept)),
                                  max_deflection_over_centre_minus_1=float(max(r['reference_max_over_centre']-1 for r in kept)),
                                  excluded_problems=[r['problem']['index'] for r in part_b if r.get('excluded')])
    dist = {arm: np.array([[abs(np.log1p(r['arms'][f'{arm} s{s}']['open']['g'])-np.log1p(r['reference']['g'])) if r['arms'][f'{arm} s{s}']['open']['status'] == 'ok' else np.nan
                            for s in SEEDS] for r in kept]) for arm in ARMS}
    secondary['design_distance_log1p_g'] = {arm: dict(median=float(np.nanmedian(d)), p95=float(np.nanquantile(d, .95))) for arm, d in dist.items()}

    # ------------------------------------------------ 6. figures
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), layout='constrained')
    ax = axes[0]
    ref_pts = [(1000*r['reference']['ceramic'], r['reference']['mass']) for r in part_a]
    ax.plot(*zip(*ref_pts), 'k-', lw=2, label='TSDT optimum (reference)', zorder=1)
    style = {'C': ('tab:green', 'o'), 'P': ('tab:purple', 's'), 'D': ('tab:orange', '^')}
    for arm in ARMS:
        for j, s in enumerate(SEEDS):
            pts = [(1000*r['arms'][f'{arm} s{s}']['open']['ceramic'], r['arms'][f'{arm} s{s}']['open']['mass'], r['arms'][f'{arm} s{s}']['open_violation'])
                   for r in part_a if r['arms'][f'{arm} s{s}']['open']['status'] == 'ok']
            for x, y, v in pts:
                ax.scatter(x, y, color=style[arm][0], marker=style[arm][1], s=28, facecolors='none' if v > FEAS_TOL else style[arm][0],
                           label=None, zorder=2)
        ax.scatter([], [], color=style[arm][0], marker=style[arm][1], label=f'{arm}: {"physics only" if arm == "P" else ("data only" if arm == "D" else "pipeline C")}')
    ax.scatter([], [], color='gray', marker='o', facecolors='none', label='hollow = violates a TSDT limit')
    ax.set(xlabel='ceramic use h/(g+1) (mm)', ylabel='areal mass (kg/m$^2$)', title='Main problem: lightest plate for each ceramic limit')
    ax.grid(alpha=.3)
    ax.legend(fontsize=7)
    ax = axes[1]
    data = [100*E[arm].mean(axis=1) for arm in ARMS]
    ax.boxplot(data, whis=(5, 95), showfliers=True)
    ax.set_xticks(range(1, len(ARMS)+1), ['C\n10 labels', 'P\nphysics only', 'D\ndata only'])
    ax.set(yscale='log', ylabel='decision error: violation + |mass gap| (%)', title=f'{len(kept)} random problems (TSDT-verified, open loop)')
    ax.grid(alpha=.3, which='both', axis='y')
    fig.savefig(out/'h3_design_summary.png', dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4), layout='constrained')
    totals = [('direct TSDT\nSLSQP', direct_total), ('C closed loop\n(incl. 10 labels)', max(c_total.values())),
              ('P closed loop', max(secondary['P']['closed_loop']['total_analyses_including_labels'].values())),
              ('D closed loop\n(incl. 10 labels)', max(secondary['D']['closed_loop']['total_analyses_including_labels'].values()))]
    ax.bar(range(len(totals)), [t[1] for t in totals], color=['tab:gray', 'tab:green', 'tab:purple', 'tab:orange'])
    for k, (_, v) in enumerate(totals):
        ax.text(k, v, str(v), ha='center', va='bottom', fontsize=8)
    ax.set_xticks(range(len(totals)), [t[0] for t in totals], fontsize=8)
    ax.set(ylabel=f'TSDT analyses for {len(kept)} problems', title='High-fidelity cost (worst seed)', yscale='log')
    ax.grid(alpha=.3, axis='y', which='both')
    fig.savefig(out/'h3_cost.png', dpi=160)
    plt.close(fig)

    # ------------------------------------------------ 7. outputs
    write_json(out/'part_a_main_problem.json', part_a)
    write_json(out/'part_b_random_problems.json', part_b)
    np.savez_compressed(out/'h3_errors.npz', **{f'{k}_{arm}': v for arm in ARMS for k, v in (('violation', V[arm]), ('gap', G[arm]), ('decision_error', E[arm]))},
                        problem_indices=np.array([r['problem']['index'] for r in kept]), seeds=np.array(SEEDS))
    table_a = [dict(ceramic_max_mm=b, reference=dict(g=r['reference']['g'], a_over_h=r['reference']['a_over_h'], mass=r['reference']['mass'],
                                                     ceramic_mm=1000*r['reference']['ceramic'], active=r['reference_active']),
                    **{arm: dict(open_violation_percent=[pct(r['arms'][f'{arm} s{s}']['open_violation']) for s in SEEDS],
                                 open_gap_percent=[pct(r['arms'][f'{arm} s{s}']['open_gap']) for s in SEEDS],
                                 closed_analyses=[r['arms'][f'{arm} s{s}']['closed']['analyses'] for s in SEEDS]) for arm in ARMS},
                    direct_slsqp_analyses=r['direct_slsqp']['analyses'])
               for b, r in zip(BUDGETS_MM, part_a)]
    manifest = dict(experiment='H3 surrogate-guided inverse design (TSDT-verified)', created_utc=datetime.now(timezone.utc).isoformat(),
                    preregistration=dict(path='docs/H3_PREREGISTRATION.md', sha256=PREREGISTRATION_SHA256),
                    development=dict(path='results/h3_development/manifest.json', sha256=sha256(args.h3dev)),
                    problems=dict(seed=PROBLEM_SEED, count=PROBLEM_COUNT, kept=len(kept)), main_problem=MAIN,
                    budgets_mm=list(BUDGETS_MM), arms=ARMS, seeds=list(SEEDS), feasibility_tolerance=FEAS_TOL,
                    part_a=table_a, decisions=decisions, secondary=secondary,
                    total_seconds=perf_counter()-start,
                    environment=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                                     matplotlib=matplotlib.__version__, platform=platform.platform()),
                    source_sha256={str(p.relative_to(ROOT)): sha256(p) for p in sorted((ROOT/'fgplate').glob('*.py'))
                                   + [ROOT/'experiments'/'h3_common.py', ROOT/'experiments'/'run_h3_dev.py', Path(__file__).resolve()]})
    write_json(out/'manifest.json', manifest)
    write_json(out/'checksums.json', {p.name: sha256(p) for p in sorted(out.iterdir()) if p.name != 'checksums.json'})
    print(json.dumps({k: v['confirmed'] for k, v in decisions.items()}, indent=1))
    print(f'done ({perf_counter()-start:.0f} s)')


if __name__ == '__main__':
    main()
