"""E2: reduced field baseline on the delivered pilot (smooth-mixture family).

Run from the project root:
    python experiments/run_e2.py --output results/e2_reduced_field

Uses results/pilot_200 exactly as delivered (read only). No new mechanics data.

Step A  projection error: training-only POD of scaled five-field snapshots at several
        ranks versus the data-free analytic 25-mode Navier basis, on validation designs.
Step B  learning error at nested training budgets {20, 40, 70, 140} and seeds {11, 22, 33}:
        B0 exact closed form (reference/timing), B1 black-box MLP (no structure),
        B2 linear-in-load model on a training POD basis (knows linearity only),
        B3 modal transfer model (knows linearity + the diagonal Navier basis).
        Hyperparameters are chosen on validation only; the test partition is evaluated
        once, after every choice is frozen.
"""
import argparse
from datetime import datetime, timezone
import hashlib
from itertools import product
import json
from pathlib import Path
import platform
import sys
from time import perf_counter
import warnings

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import scipy
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPRegressor
import matplotlib
import matplotlib.ticker
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from fgplate import NavierFSDTPlate, al_al2o3
from fgplate.baseline import bootstrap_ci
from fgplate.fields import (POD, analytic_basis, field_scales, project_analytic, reconstruct_analytic,
                            relative_l2, scale_fields)
from fgplate.pilot import FIELDS, transformed_features

MIXTURE = 2
RANKS = (5, 10, 15, 20, 25, 30, 40, 60, 80, 100)
RANK_RULE = 'smallest rank whose validation p95 projection error (all five scaled fields) is <= 1%'
BUDGETS = (20, 40, 70, 140)
SEEDS = (11, 22, 33)
GRID = dict(B1=[dict(alpha=a) for a in (1e-5, 1e-3, 1e-1)],
            B2=[dict(degree=d, ridge=r) for d in (0, 1, 2, 3) for r in (1e-8, 1e-6, 1e-4, 1e-2, 1.)],
            B3=[dict(degree=d, ridge=r) for d in (1, 2, 3, 4) for r in (1e-8, 1e-6, 1e-4)],
            B3L=[dict(degree=d, ridge=r) for d in (1, 2, 3, 4) for r in (1e-8, 1e-6, 1e-4)])
STRUCTURE_AWARE = ('B3', 'B3L')
POST_HOC = dict(B3L='Added after the first E2 run (results/e2_reduced_field) showed B3 tail errors at narrow plates; '
                     'selected on validation only; reported separately from the pre-declared families.')
GATE = dict(median_percent=1.0, p95_percent=3.0)
NAMES = dict(B1='B1 black-box MLP', B2='B2 linear-in-load (POD)', B3='B3 modal transfer (Navier basis)',
             B3L='B3L modal transfer, log-magnitude targets (post hoc)')


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def poly(x, degree):
    powers = [p for p in product(range(degree+1), repeat=x.shape[1]) if sum(p) <= degree]
    return np.column_stack([np.prod(x**np.array(p), axis=1) for p in powers])


def ridge(A, Y, penalty, intercept_first=True):
    """Least squares with sqrt(penalty)*I augmentation; optional unpenalized intercept column 0."""
    P = np.sqrt(penalty)*np.eye(A.shape[1])
    if intercept_first:
        P[0, 0] = 0
    return np.linalg.lstsq(np.vstack((A, P)), np.vstack((Y, np.zeros((A.shape[1], Y.shape[1])))), rcond=None)[0]


def summary(errors):
    e = 100*np.asarray(errors)
    return dict(median_percent=float(np.median(e)), p95_percent=float(np.quantile(e, .95)),
                worst_percent=float(e.max()), mean_percent=float(e.mean()),
                median_percent_ci95=(100*bootstrap_ci(errors, 'median')[0]).tolist(), cases=int(len(e)))


class Context:
    """Training-subset statistics shared by all models of one (budget, seed)."""

    def __init__(self, x3, c, S, train, rank):
        self.mean, self.std = x3[train].mean(0), x3[train].std(0)
        self.std[self.std == 0] = 1
        self.x = (x3-self.mean)/self.std
        self.c, self.S, self.train = c, S, train
        self.pod = POD(S[train])
        self.rank = min(rank, len(train))
        self.a_train = self.pod.coefficients(S[train], self.rank)
        self.shape = S.shape[1:]


class B1:
    def __init__(self, ctx, alpha, seed):
        self.ctx = ctx
        X = np.column_stack((ctx.x[ctx.train], ctx.c[ctx.train]))
        self.mu, self.sd = ctx.a_train.mean(0), ctx.a_train.std(0)
        self.sd[self.sd == 0] = 1
        self.net = MLPRegressor(hidden_layer_sizes=(64, 64, 64), activation='tanh', solver='adam',
                                learning_rate_init=1e-3, batch_size=min(32, len(X)), max_iter=3000,
                                tol=1e-8, n_iter_no_change=200, alpha=alpha, random_state=seed)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always', ConvergenceWarning)
            self.net.fit(X, (ctx.a_train-self.mu)/self.sd)
        self.converged = not any(issubclass(w.category, ConvergenceWarning) for w in caught)

    def coefficients(self, idx):
        X = np.column_stack((self.ctx.x[idx], self.ctx.c[idx]))
        return self.net.predict(X).reshape(len(idx), -1)*self.sd+self.mu

    def predict(self, idx):
        return self.ctx.pod.reconstruct(self.coefficients(idx), self.ctx.shape)


class B2:
    def __init__(self, ctx, degree, ridge_penalty):
        self.ctx, self.degree = ctx, degree
        self.w = ridge(self.features(ctx.train), ctx.a_train, ridge_penalty, intercept_first=False)

    def features(self, idx):
        p = poly(self.ctx.x[idx], self.degree)
        return (self.ctx.c[idx][:, :, None]*p[:, None, :]).reshape(len(idx), -1)

    def coefficients(self, idx):
        return self.features(idx)@self.w

    def predict(self, idx):
        return self.ctx.pod.reconstruct(self.coefficients(idx), self.ctx.shape)


class B3:
    def __init__(self, ctx, degree, ridge_penalty, basis):
        self.ctx, self.degree, self.basis = ctx, degree, basis
        coeffs = project_analytic(ctx.S[ctx.train], basis)          # [n, 5, 25] from the same field labels
        H = (coeffs/ctx.c[ctx.train][:, None, :]).reshape(len(ctx.train), -1)
        self.mu, self.sd = H.mean(0), H.std(0)
        self.sd[self.sd == 0] = 1
        self.w = ridge(poly(ctx.x[ctx.train], degree), (H-self.mu)/self.sd, ridge_penalty)

    def predict(self, idx):
        H = (poly(self.ctx.x[idx], self.degree)@self.w*self.sd+self.mu).reshape(len(idx), 5, -1)
        return reconstruct_analytic(H*self.ctx.c[idx][:, None, :], self.basis)


class B3L(B3):
    """B3 with regression on log|H| and the training sign; every transfer component has one sign."""

    def __init__(self, ctx, degree, ridge_penalty, basis):
        self.ctx, self.degree, self.basis = ctx, degree, basis
        coeffs = project_analytic(ctx.S[ctx.train], basis)
        H = (coeffs/ctx.c[ctx.train][:, None, :]).reshape(len(ctx.train), -1)
        self.sign = np.sign(H[0])
        if np.any(np.sign(H) != self.sign) or np.any(H == 0):
            raise ValueError('log-magnitude transfer model needs one nonzero sign per component')
        L = np.log(np.abs(H))
        self.mu, self.sd = L.mean(0), L.std(0)
        self.sd[self.sd == 0] = 1
        self.w = ridge(poly(ctx.x[ctx.train], degree), (L-self.mu)/self.sd, ridge_penalty)

    def predict(self, idx):
        L = poly(self.ctx.x[idx], self.degree)@self.w*self.sd+self.mu
        H = (self.sign*np.exp(L)).reshape(len(idx), 5, -1)
        return reconstruct_analytic(H*self.ctx.c[idx][:, None, :], self.basis)


def evaluate(model, idx, ctx, S):
    prediction = model.predict(idx)
    result = dict(total=relative_l2(prediction, S[idx]), w=relative_l2(prediction[:, 2], S[idx, 2]))
    if hasattr(model, 'coefficients'):
        projected = ctx.pod.reconstruct(ctx.pod.coefficients(S[idx], ctx.rank), ctx.shape)
        norm = np.linalg.norm(S[idx].reshape(len(idx), -1), axis=1)
        result['projection'] = relative_l2(projected, S[idx])
        result['learning'] = np.linalg.norm((prediction-projected).reshape(len(idx), -1), axis=1)/norm
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--pilot', type=Path, default=ROOT/'results'/'pilot_200')
    parser.add_argument('--output', type=Path, default=ROOT/'results'/'e2_reduced_field')
    parser.add_argument('--families', default='B1,B2,B3', help='comma list from ' + ','.join(GRID))
    args = parser.parse_args()
    families = tuple(args.families.split(','))
    if not families or any(f not in GRID for f in families):
        parser.error('unknown model family')
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
        modes_all, Q = z['mode_indices'], z['pressure_coefficients'][:, MIXTURE]
        fields, xi, eta = z['fields'][:, MIXTURE], z['xi'], z['eta']
    active = np.flatnonzero(np.any(Q != 0, axis=0))
    modes = modes_all[active]
    assert len(modes) == 25 and np.all(Q[:, active] != 0)
    c = Q[:, active]                                   # unit-norm signed load coefficients (Pa)
    h = 1/P[:, 1]
    S = scale_fields(fields, 1.0, h)                   # a = 1 m, qscale = 1 Pa
    x3 = transformed_features(P)
    basis = analytic_basis(modes, xi, eta)
    train, val, test = (np.flatnonzero(split == s) for s in ('train', 'validation', 'test'))

    # ---------------------------------------------------------------- Step A
    pod_full = POD(S[train])
    norm_val = np.linalg.norm(S[val].reshape(len(val), -1), axis=1)
    step_a = dict(rank_rule=RANK_RULE, ranks={})
    for r in RANKS:
        rec = pod_full.reconstruct(pod_full.coefficients(S[val], r), S.shape[1:])
        per_field = {name: (np.linalg.norm((rec[:, k]-S[val, k]).reshape(len(val), -1), axis=1)/norm_val)
                     for k, name in enumerate(FIELDS)}
        step_a['ranks'][str(r)] = dict(total=summary(relative_l2(rec, S[val])),
                                       w_only=summary(relative_l2(rec[:, 2], S[val, 2])),
                                       per_field_share_median_percent={k: float(100*np.median(v)) for k, v in per_field.items()})
    chosen = next((r for r in RANKS if step_a['ranks'][str(r)]['total']['p95_percent'] <= 1.0), RANKS[-1])
    step_a['chosen_rank'] = chosen
    analytic = reconstruct_analytic(project_analytic(S[val], basis), basis)
    step_a['analytic_25_mode_basis'] = summary(relative_l2(analytic, S[val]))
    s = pod_full.singular_values
    step_a['training_singular_values'] = s[:120].tolist()
    step_a['training_numerical_rank_1e-10'] = int(np.sum(s > s[0]*1e-10))
    print(f'Step A: chosen POD rank {chosen}; numerical rank of scaled five-field training snapshots '
          f'{step_a["training_numerical_rank_1e-10"]}', flush=True)

    # ---------------------------------------------------------------- Step B
    runs, models, convergence_warnings = [], {}, 0
    for seed in SEEDS:
        order = np.random.default_rng(seed).permutation(train)
        for n in BUDGETS:
            subset = np.sort(order[:n])
            ctx = Context(x3, c, S, subset, chosen)
            for family in families:
                best = None
                t0 = perf_counter()
                for hp in GRID[family]:
                    if family == 'B1':
                        model = B1(ctx, hp['alpha'], seed)
                    elif family == 'B2':
                        model = B2(ctx, hp['degree'], hp['ridge'])
                    else:
                        if poly(ctx.x[:1], hp['degree']).shape[1] > n:
                            continue                     # more polynomial terms than designs
                        model = (B3 if family == 'B3' else B3L)(ctx, hp['degree'], hp['ridge'], basis)
                    score = evaluate(model, val, ctx, S)
                    key = (np.median(score['total']), np.quantile(score['total'], .95))
                    if best is None or key < best[0]:
                        best = (key, hp, model, score)
                _, hp, model, score = best
                if family == 'B1': convergence_warnings += int(not model.converged)
                models[family, n, seed] = (model, ctx)
                runs.append(dict(family=family, budget=n, seed=seed, rank=ctx.rank if family not in STRUCTURE_AWARE else 25,
                                 selected=hp, fit_and_selection_seconds=perf_counter()-t0,
                                 validation={k: summary(v) for k, v in score.items()},
                                 validation_errors=score['total'].tolist()))
            print(f'Step B: seed {seed}, budget {n} done', flush=True)

    # ---------------------------------------------------------------- aggregate (validation only)
    curves = {}
    for family in families:
        curves[family] = {}
        for n in BUDGETS:
            rows = [r for r in runs if r['family'] == family and r['budget'] == n]
            med = [r['validation']['total']['median_percent'] for r in rows]
            p95 = [r['validation']['total']['p95_percent'] for r in rows]
            curves[family][str(n)] = dict(median_percent_mean=float(np.mean(med)), median_percent_range=[min(med), max(med)],
                                          p95_percent_mean=float(np.mean(p95)), p95_percent_range=[min(p95), max(p95)])
    gate = {}
    for family in families:
        passing = [n for n in BUDGETS if curves[family][str(n)]['median_percent_mean'] <= GATE['median_percent']
                   and curves[family][str(n)]['p95_percent_mean'] <= GATE['p95_percent']]
        gate[family] = dict(smallest_budget_meeting_target=passing[0] if passing else None)
    reached = [gate[f]['smallest_budget_meeting_target'] for f in families if f in STRUCTURE_AWARE]
    reached = [n for n in reached if n is not None]
    b3_small = min(reached) if reached else None
    gate['decision'] = ('SSSS parametric field problem is too easy for H1: structure-aware model meets the in-domain target '
                        f'with {b3_small} designs; move the physics-vs-data comparison (E3) to a setting without a closed-form '
                        'structure (clamped family, loads outside the 5x5 basis, or scarce-label regularization).'
                        if b3_small is not None and b3_small <= 40 else
                        'Structure-aware model does not meet the target with <= 40 designs; the SSSS family remains a '
                        'meaningful testbed for the physics-vs-data comparison.')
    gate['rule'] = (f'median <= {GATE["median_percent"]}% and p95 <= {GATE["p95_percent"]}% (validation, mean over seeds); '
                    'decision threshold 40 designs; fixed before running E2')

    # ---------------------------------------------------------------- single final test evaluation
    for run in runs:
        model, ctx = models[run['family'], run['budget'], run['seed']]
        score = evaluate(model, test, ctx, S)
        run['test'] = {k: summary(v) for k, v in score.items()}
    test_curves = {f: {str(n): float(np.mean([r['test']['total']['median_percent'] for r in runs
                                               if r['family'] == f and r['budget'] == n])) for n in BUDGETS}
                   for f in families}

    # ---------------------------------------------------------------- timing (local, illustrative)
    timing = {}
    for family in families:
        model, _ = models[family, 140, SEEDS[0]]
        ts = []
        for _ in range(7):
            t0 = perf_counter(); model.predict(val); ts.append(perf_counter()-t0)
        timing[family+'_ms_per_design_batch30'] = float(1e3*np.median(ts[2:])/len(val))
    ts = []
    for i in val:
        t0 = perf_counter()
        plate = NavierFSDTPlate(1.0, float(P[i, 2]), float(h[i]), al_al2o3(float(P[i, 0])))
        sol = plate.bending_modal({tuple(map(int, m)): float(q) for m, q in zip(modes, c[i])}, method='closed_form')
        _, _, f = sol.field_grid(41, 41)
        scale_fields(np.stack([f[k] for k in FIELDS]), 1.0, h[i])
        ts.append(perf_counter()-t0)
    timing['B0_exact_closed_form_ms_per_design'] = float(1e3*np.median(ts))
    timing['note'] = ('Same output for all: five scaled 41x41 fields. Local CPU, single process, '
                      'median of repeats; B0 includes plate construction and the 25-mode closed-form solve.')

    # ---------------------------------------------------------------- figures
    fig, ax = plt.subplots(figsize=(6, 4), layout='constrained')
    ax.semilogy(RANKS, [step_a['ranks'][str(r)]['total']['median_percent'] for r in RANKS], 'o-', label='POD median')
    ax.semilogy(RANKS, [step_a['ranks'][str(r)]['total']['p95_percent'] for r in RANKS], 's--', label='POD 95th percentile')
    analytic_p95 = step_a['analytic_25_mode_basis']['p95_percent']
    ax.axhline(analytic_p95, color='k', lw=1,
               label=f'analytic Navier basis, 25 modes per field (p95 = {analytic_p95:.1e}%)')
    ax.axhline(1.0, color='r', lw=.8, ls=':', label='1% rank rule')
    ax.axvline(chosen, color='gray', lw=.8, ls=':')
    ax.set(xlabel='POD rank (five scaled fields)', ylabel='validation projection error (%)',
           title='E2 step A: projection error, 30 validation designs')
    ax.grid(alpha=.3); ax.legend(fontsize=8)
    fig.savefig(out/'e2_projection_error.png', dpi=170); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout='constrained')
    for family, marker in zip(families, ('o', 's', '^', 'v', 'D')):
        for ax, stat in zip(axes, ('median', 'p95')):
            mean = [curves[family][str(n)][f'{stat}_percent_mean'] for n in BUDGETS]
            lo = [curves[family][str(n)][f'{stat}_percent_range'][0] for n in BUDGETS]
            hi = [curves[family][str(n)][f'{stat}_percent_range'][1] for n in BUDGETS]
            ax.plot(BUDGETS, mean, marker=marker, label=NAMES[family])
            ax.fill_between(BUDGETS, lo, hi, alpha=.15)
    for ax, stat, target in zip(axes, ('Median', '95th percentile'), (GATE['median_percent'], GATE['p95_percent'])):
        ax.axhline(target, color='r', lw=.8, ls=':', label=f'target {target}%')
        ax.set(xscale='log', yscale='log', xticks=BUDGETS, xlabel='training designs',
               ylabel=f'{stat} validation field error (%)', title=f'E2 step B: {stat.lower()} over 30 designs')
        ax.set_xticklabels([str(b) for b in BUDGETS]); ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.grid(alpha=.3, which='both'); ax.legend(fontsize=8)
    fig.savefig(out/'e2_learning_curves.png', dpi=170); plt.close(fig)

    # ---------------------------------------------------------------- persist
    np.savez_compressed(out/'e2_errors.npz', validation_ids=ids[val], test_ids=ids[test],
                        **{f'{r["family"]}_n{r["budget"]}_s{r["seed"]}_validation': np.array(r['validation_errors'])
                           for r in runs})
    for r in runs:
        r.pop('validation_errors')
    manifest = dict(experiment='E2 reduced field baseline', created_utc=datetime.now(timezone.utc).isoformat(),
                    pilot=dict(path=str(args.pilot), dataset_sha256=recorded['dataset.npz'],
                               generator_version=json.loads((pilot/'manifest.json').read_text())['generator_version']),
                    load_family='smooth_mixture (25 signed sine modes, unit norm, qscale 1 Pa)',
                    field_scaling=dict(u0_v0='qscale a^3/(Ec h^2)', w0='qscale a^4/(Ec h^3)', phix_phiy='qscale a^3/(Ec h^3)'),
                    error_metric='per-design relative L2 over all five scaled fields on the 41x41 grid (w-only also stored)',
                    partitions=dict(train=len(train), validation=len(val), test=len(test)),
                    families={f: NAMES[f] for f in families}, post_hoc={f: POST_HOC[f] for f in families if f in POST_HOC},
                    budgets=BUDGETS, seeds=SEEDS, hyperparameter_grids={f: GRID[f] for f in families}, step_a=step_a, step_b_runs=runs,
                    validation_learning_curves=curves, test_median_percent_mean_over_seeds=test_curves,
                    decision_gate=gate, timing=timing, b1_convergence_warnings=convergence_warnings,
                    total_seconds=perf_counter()-start_all,
                    environment=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                                     sklearn=sklearn.__version__, matplotlib=matplotlib.__version__, platform=platform.platform()),
                    source_sha256={str(p.relative_to(ROOT)): sha256(p) for p in
                                   [*sorted((ROOT/'fgplate').glob('*.py')), Path(__file__)]})
    write_json(out/'manifest.json', manifest)
    write_json(out/'checksums.json', {p.name: sha256(p) for p in sorted(out.iterdir()) if p.is_file() and p.name != 'checksums.json'})
    print(json.dumps(dict(chosen_rank=chosen, validation_curves=curves, gate=gate, timing=timing), indent=1), flush=True)


if __name__ == '__main__':
    main()
