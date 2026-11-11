"""Shared code for H3 (surrogate-guided inverse design): frozen surrogates and the design optimizer.

Used by run_h3_dev.py (development on pilot data and FSDT only) and run_h3.py (pre-registered study).

Surrogates (all built from pilot data only, hyperparameters frozen from E5 or selected on pilot validation):
  static  centre deflection under uniform pressure from the H2-extended transfer functions (15x15 modes):
          C = M-S@P2 extended (10 shear-guided TSDT labels), P = label-free base P-ext, D = data only D-ext;
  modal   fundamental frequency heads:
          C = exact FSDT x (1 + (k h)^2 p(x)), p fitted on the SAME 10 labelled designs;
          P = exact FSDT (no labels);   D = polynomial in log frequency on the SAME 10 random designs as D-ext.
  truth   TSDT (Navier) with odd modes m, n <= TRUTH_K for deflection; TSDT (1,1) eigenproblem for frequency.

Design problem (TID): fixed footprint a x b, uniform service pressure q0, design variables h and g;
minimize areal mass h [rho_m + (rho_c - rho_m)/(g + 1)] subject to max|w| <= w_lim, f1 >= f_min,
optional ceramic use h/(g + 1) <= C_max, and the pilot domain bounds 5 <= a/h <= 100, 0 <= g <= 8.
"""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from scipy.optimize import minimize

from fgplate.transfer import (assemble_data, assemble_physics, classical_prefactor, design_coordinates, features,
                              mode_matrices, predict_scaled, solve_modes)
from fgplate.tsdt import fundamental_batch, transfer, transfer_batch
from run_e5 import SCOPE, SEEDS, fit_all, kh2, sha256, shear_share, summary, write_json

PRIMARY_N = 10
EXT = 15                       # surrogate modes m, n <= 15 (as in H2)
TRUTH_K = 49                   # truth: odd modes m, n <= 49 (convergence checked in development)
EC, RHO_C, RHO_M = 380e9, 3800.0, 2702.0
G_BOUNDS, R_BOUNDS = (0.0, 8.0), (5.0, 100.0)
FREQ_DEGREES = (0, 1, 2, 3)
FREQ_RIDGES = (1e-12, 1e-9, 1e-6, 1e-3)
FREQ_TARGET = dict(median_percent=1.0, p95_percent=2.0)     # PID frequency targets
GRID = 81                      # dense grid per axis in (log(1+g), log(a/h))
FEAS_TOL = 1e-4                # TSDT feasibility tolerance (relative) for verification
REFINE_STARTS, REFINE_SEPARATION = 4, 0.1   # local refinements from separated grid points (in scaled log(1+g))
ARMS = {'C': 'M-S@P2 extended to 15x15 + FSDT x (k h)^2 frequency correction (10 TSDT-labelled designs)',
        'P': 'physics only: label-free base P-ext + exact FSDT frequency (0 labels)',
        'D': 'data only: D-ext + polynomial log-frequency (10 random TSDT-labelled designs)'}


def square_modes(k):
    return np.array([(m, n) for m in range(1, k+1) for n in range(1, k+1)])


def odd_modes(k):
    return np.array([(m, n) for m in range(1, k+1, 2) for n in range(1, k+1, 2)])


def centre_weights(modes):
    """Uniform unit pressure Q_mn = 16/(pi^2 m n) times sin(m pi/2) sin(n pi/2) (0 for even m or n)."""
    m, n = modes[:, 0], modes[:, 1]
    odd = (m % 2 == 1) & (n % 2 == 1)
    sign = np.where(((m-1)//2+(n-1)//2) % 2 == 0, 1.0, -1.0)
    return np.where(odd, 16/(np.pi**2*m*n)*sign, 0.0)


def centre_deflection(params, theory, k=TRUTH_K, chunk=100):
    """Exact Navier centre deflection w_hat = w Ec h^3/(q0 a^4) under uniform pressure (odd modes <= k)."""
    params = np.atleast_2d(np.asarray(params, dtype=float))
    modes = odd_modes(k)
    weights = centre_weights(modes)
    return np.concatenate([transfer_batch(params[i:i+chunk], modes, theory)[:, :, 2]@weights
                           for i in range(0, len(params), chunk)])


def truth(params, k=TRUTH_K):
    """TSDT centre deflection w_hat and frequency omega_hat = omega_11 h sqrt(rho_c/Ec)."""
    return centre_deflection(params, 'TSDT', k), fundamental_batch(np.atleast_2d(params), 'TSDT')


def fsdt(params, k=TRUTH_K):
    return centre_deflection(params, 'FSDT', k), fundamental_batch(np.atleast_2d(params), 'FSDT')


def labelled_designs(train, psi, seed, N=PRIMARY_N):
    """The shear-guided (C) and random (D) labelled pilot designs, exactly as in E5/H2."""
    order_r = np.random.default_rng(seed).permutation(train)
    keys = np.log(np.random.default_rng([seed, 7]).random(len(train)))/psi[train]
    order_s = train[np.argsort(-keys)]
    return np.sort(order_s[:N]), np.sort(order_r[:N])


def ridge_solve(A, y, ridge):
    G = A.T@A
    lam = ridge*np.trace(G)/len(G)
    return np.linalg.solve(G+lam*np.eye(len(G)), A.T@y)


def freq_errors(pred, ref):
    return np.abs(pred/ref-1)


def freq_score(errors):
    e = 100*np.asarray(errors)
    med, p95 = float(np.median(e)), float(np.quantile(e, .95))
    return (max(med/FREQ_TARGET['median_percent'], p95/FREQ_TARGET['p95_percent']), med)


class Surrogates:
    """Frozen H3 surrogates for one seed: evaluate(arm, params) -> (w_hat_centre, omega_hat)."""

    def __init__(self, seed, static, freq):
        self.seed, self.static, self.freq = seed, static, freq
        self.modes = square_modes(EXT)
        self.weights = centre_weights(self.modes)
        self._cache = {}

    def _shared(self, params):
        """Label-free base transfer functions and exact FSDT frequency, shared by arms P and C (memoized)."""
        key = params.tobytes()
        if key not in self._cache:
            if len(self._cache) > 8:
                self._cache.clear()
            theta_p, dp = self.static['P']
            H_P = predict_scaled(theta_p, dp, design_coordinates(params), classical_prefactor(params, self.modes))
            self._cache[key] = (H_P, fundamental_batch(params, 'FSDT'))
        return self._cache[key]

    def transfer(self, arm, params):
        params = np.ascontiguousarray(np.atleast_2d(params), dtype=float)
        x = design_coordinates(params)
        H_P = self._shared(params)[0]
        if arm == 'P':
            return H_P
        if arm == 'C':
            theta_c, dc = self.static['C']
            return H_P+predict_scaled(theta_c, dc, x, H_P*kh2(params, self.modes)[:, :, None])
        theta_d, dd = self.static['D']
        return predict_scaled(theta_d, dd, x)

    def frequency(self, arm, params):
        params = np.ascontiguousarray(np.atleast_2d(params), dtype=float)
        if arm == 'D':
            theta, d = self.freq['D']
            return np.exp(features(design_coordinates(params), d)@theta)
        base = self._shared(params)[1]
        if arm == 'P':
            return base
        theta, d = self.freq['C']
        corr = kh2(params, np.array([[1, 1]]))[:, 0]*(features(design_coordinates(params), d)@theta)
        return base*np.sqrt(1+corr)

    def evaluate(self, arm, params):
        params = np.atleast_2d(params)
        return self.transfer(arm, params)[:, :, 2]@self.weights, self.frequency(arm, params)


def build(pilot_dir, development_manifest, log=print):
    """Refit the frozen E5 arms (asserted identical), extend to 15x15 as in H2, fit and select frequency heads.

    Returns (fit, surrogates {seed: Surrogates}, selection record). Deterministic.
    """
    fit = fit_all(Path(pilot_dir).resolve(), log=log)
    assert fit['chosen'] == 'P2'
    dev = {(r['arm'], r['budget'], r['seed']): r['validation'] for r in json.loads(Path(development_manifest).read_text())['runs']}
    for key, result in fit['results'].items():
        assert all(np.isclose(result['validation'][k], dev[key][k], rtol=1e-9, atol=1e-12)
                   for k in ('median_percent', 'p95_percent', 'worst_percent')), key
    log('frozen E5 arms reproduced')
    P, train, val, res = fit['P'], fit['train'], fit['val'], fit['results']
    modes_ext = square_modes(EXT)
    x_pilot = design_coordinates(P)
    pref_pilot_ext = classical_prefactor(P, modes_ext)
    kh2_pilot_ext = kh2(P, modes_ext)
    H_tsdt_pilot_ext = transfer(P, modes_ext, 'TSDT')
    psi = shear_share(P)
    omega_T, omega_F = fundamental_batch(P, 'TSDT'), fundamental_batch(P, 'FSDT')
    kh2_11 = kh2(P, np.array([[1, 1]]))[:, 0]
    sur, selection = {}, {}
    for seed in SEEDS:
        hp = res['P2', 0, seed]['selected']
        colloc = fit['colloc'][seed]
        K, f = mode_matrices(colloc, modes_ext)
        theta_p = solve_modes(*assemble_physics(design_coordinates(colloc), K, f, hp['degree'],
                                                prefactor=classical_prefactor(colloc, modes_ext)), hp['degree'], hp['ridge'], SCOPE)
        H_P = predict_scaled(theta_p, hp['degree'], x_pilot, pref_pilot_ext)
        sub_s, sub_r = labelled_designs(train, psi, seed)
        sel = res['M-S@P2', PRIMARY_N, seed]['selected']
        A, b, _ = assemble_data(x_pilot[sub_s], H_tsdt_pilot_ext[sub_s]-H_P[sub_s], sel['correction_degree'],
                                prefactor=(H_P*kh2_pilot_ext[:, :, None])[sub_s])
        theta_c = solve_modes(A, b, sel['correction_degree'], sel['ridge'], SCOPE)
        sel_d = res['D', PRIMARY_N, seed]['selected']
        A, b, _ = assemble_data(x_pilot[sub_r], H_tsdt_pilot_ext[sub_r], sel_d['degree'])
        theta_d = solve_modes(A, b, sel_d['degree'], sel_d['ridge'], SCOPE)
        static = dict(P=(theta_p, hp['degree']), C=(theta_c, sel['correction_degree']), D=(theta_d, sel_d['degree']))
        # frequency heads: candidates fitted on the labelled designs, selected on the 30 pilot validation designs
        freq, record = {}, {}
        for arm, sub in (('C', sub_s), ('D', sub_r)):
            best = None
            for d in FREQ_DEGREES:
                for ridge in FREQ_RIDGES:
                    F = features(x_pilot[sub], d)
                    if arm == 'C':
                        theta = ridge_solve(kh2_11[sub, None]*F, omega_T[sub]**2/omega_F[sub]**2-1, ridge)
                        pred = omega_F[val]*np.sqrt(1+kh2_11[val]*(features(x_pilot[val], d)@theta))
                    else:
                        theta = ridge_solve(F, np.log(omega_T[sub]), ridge)
                        pred = np.exp(features(x_pilot[val], d)@theta)
                    e = freq_errors(pred, omega_T[val])
                    key = (round(freq_score(e)[0], 12), round(freq_score(e)[1], 12), d, -ridge)
                    if best is None or key < best[0]:
                        best = (key, theta, d, ridge, e)
            freq[arm] = (best[1], best[2])
            record[arm] = dict(degree=best[2], ridge=best[3], validation=summary(best[4]))
        record['P'] = dict(validation=summary(freq_errors(omega_F[val], omega_T[val])))
        record['labelled_designs'] = dict(C=sub_s.tolist(), D=sub_r.tolist())
        sur[seed] = Surrogates(seed, static, freq)
        selection[seed] = record
    log('surrogates built')
    return fit, sur, selection


# ---------------------------------------------------------------- design problem
def mass(h, g):
    return h*(RHO_M+(RHO_C-RHO_M)/(g+1))


def ceramic(h, g):
    return h/(g+1)


def physical(problem, params, w_hat, omega_hat):
    """Dimensionless responses -> (max deflection m, frequency Hz)."""
    a, q0 = problem['a'], problem['q0']
    h = a/params[:, 1]
    return q0*a**4*w_hat/(EC*h**3), omega_hat/(2*np.pi*h)*np.sqrt(EC/RHO_C)


U_BOX = np.array([np.log1p(G_BOUNDS[0]), np.log1p(G_BOUNDS[1])])
V_BOX = np.array([np.log(R_BOUNDS[0]), np.log(R_BOUNDS[1])])


def to_params(s, problem):
    """s in [0, 1]^2 (rows) -> parameters [g, a/h, b/a]."""
    s = np.atleast_2d(s)
    g = np.expm1(U_BOX[0]+s[:, 0]*(U_BOX[1]-U_BOX[0]))
    r = np.exp(V_BOX[0]+s[:, 1]*(V_BOX[1]-V_BOX[0]))
    return np.column_stack([g, r, np.full(len(s), problem['aspect'])])


def constraints(problem, params, w, f):
    """Normalized constraint values (>= 0 feasible): deflection, frequency, ceramic use."""
    h, g = problem['a']/params[:, 1], params[:, 0]
    c_cer = 1-ceramic(h, g)/problem['ceramic_max'] if np.isfinite(problem['ceramic_max']) else np.ones(len(params))
    return np.column_stack([1-w/problem['w_lim'], f/problem['f_min']-1, c_cer])


GRID_CACHE = {}


def optimize(evaluate, problem, scale=(1.0, 1.0), key=None, grid=GRID):
    """Dense grid + SLSQP refinement in (log(1+g), log(a/h)); evaluate(params) -> (w_hat, omega_hat).

    scale multiplies the evaluator's (w_hat, omega_hat) (closed-loop correction). key, if given, caches the grid
    evaluation per (key, aspect) so that repeated problems on the same footprint reuse it (the evaluation count
    still includes the grid every time). Returns the optimum, its responses and the number of designs evaluated.
    """
    t = np.linspace(0, 1, grid)
    S = np.array([(p, q) for p in t for q in t])
    params = to_params(S, problem)
    ck = None if key is None else (key, float(problem['aspect']), grid)
    if ck is not None and ck in GRID_CACHE:
        w_hat, omega_hat = GRID_CACHE[ck]
    else:
        w_hat, omega_hat = evaluate(params)
        if ck is not None:
            GRID_CACHE[ck] = (w_hat, omega_hat)
    base_evaluate = evaluate
    evaluate = lambda p: tuple(s*v for s, v in zip(scale, base_evaluate(p)))
    w, f = physical(problem, params, scale[0]*w_hat, scale[1]*omega_hat)
    c = constraints(problem, params, w, f)
    feasible = np.all(c >= 0, axis=1)
    count = len(S)
    if not feasible.any():
        return dict(status='infeasible', evaluations=count)
    m = mass(problem['a']/params[:, 1], params[:, 0])
    i = int(np.argmin(np.where(feasible, m, np.inf)))
    m_ref = m[i]
    cache = {}

    def response(s):
        key = (float(s[0]), float(s[1]))
        if key not in cache:
            p = to_params(np.clip(np.array(key), 0, 1), problem)
            ww, ff = physical(problem, p, *evaluate(p))
            cache[key] = constraints(problem, p, ww, ff)[0]
        return cache[key]

    def objective(s):
        p = to_params(np.clip(s, 0, 1), problem)
        return float(mass(problem['a']/p[0, 1], p[0, 0])/m_ref)

    # local refinement from several separated good grid points (the mass landscape can be flat with two basins)
    order = np.argsort(np.where(feasible, m, np.inf))
    starts = []
    for j in order[:feasible.sum()]:
        if all(abs(S[j, 0]-S[k, 0]) >= REFINE_SEPARATION for k in starts):
            starts.append(j)
        if len(starts) == REFINE_STARTS:
            break
    best_s, best_obj, messages = S[i], 1.0, []
    for j in starts:
        s = S[j]
        for _ in range(3):                    # SLSQP, restarted from its own result while it improves
            r = minimize(objective, s, method='SLSQP', bounds=[(0, 1), (0, 1)],
                         constraints=[dict(type='ineq', fun=response)], options=dict(ftol=1e-12, maxiter=300))
            x = np.clip(r.x, 0, 1)
            messages.append(int(r.status))
            if np.all(response(x) >= -1e-9) and objective(x) < objective(s)-1e-13:
                s = x
            else:
                break
        if np.all(response(s) >= -1e-9) and objective(s) < best_obj:
            best_s, best_obj = s, objective(s)
    s_opt = best_s
    p = to_params(s_opt, problem)
    w_hat, omega_hat = evaluate(p)
    ww, ff = physical(problem, p, w_hat, omega_hat)
    return dict(status='ok', s=s_opt.tolist(), g=float(p[0, 0]), a_over_h=float(p[0, 1]),
                h=float(problem['a']/p[0, 1]), mass=float(mass(problem['a']/p[0, 1], p[0, 0])),
                ceramic=float(ceramic(problem['a']/p[0, 1], p[0, 0])), w=float(ww[0]), f=float(ff[0]),
                constraints=constraints(problem, p, ww, ff)[0].tolist(), evaluations=count+len(cache),
                grid_best_mass=float(m_ref), refined=bool(best_obj < 1.0), slsqp_status=messages)


def verify(problem, params):
    """TSDT reanalysis: (max deflection, frequency, normalized constraints, violation >= 0)."""
    w, f = physical(problem, params, *truth(params))
    c = constraints(problem, params, w, f)
    return w, f, c, np.maximum(0, -c[:, :2].min(axis=1))


def closed_loop(evaluate, problem, key=None, first=None, max_analyses=5):
    """Surrogate optimum -> TSDT reanalysis; while infeasible beyond FEAS_TOL, rescale the surrogate's deflection and
    frequency by the TSDT/surrogate ratios at the latest design (zero-order correction) and re-optimize.
    Each reanalysis is one TSDT analysis. first: the already computed uncorrected optimum (open loop)."""
    scale = (1.0, 1.0)
    history = []
    for k in range(1, max_analyses+1):
        opt = first if (k == 1 and first is not None) else optimize(evaluate, problem, scale, key)
        if opt['status'] == 'infeasible':
            return dict(status='infeasible', analyses=k-1, history=history)
        p = np.array([[opt['g'], opt['a_over_h'], problem['aspect']]])
        w, f, c, v = verify(problem, p)
        history.append(dict(g=opt['g'], a_over_h=opt['a_over_h'], mass=opt['mass'], violation=float(v[0])))
        if v[0] <= FEAS_TOL:
            return dict(status='feasible', analyses=k, design=opt, violation=float(v[0]), history=history)
        w_s, f_s = physical(problem, p, *evaluate(p))
        scale = (w[0]/w_s[0], f[0]/f_s[0])
    return dict(status='not_converged', analyses=max_analyses, design=opt, violation=float(v[0]), history=history)


def direct_slsqp(evaluate, problem, start=(0.5, 0.5)):
    """Direct optimization competitor: SLSQP on evaluate from the box centre (no grid), same options as optimize().
    Every distinct design evaluated counts as one analysis."""
    cache = {}

    def response(s):
        key = (float(s[0]), float(s[1]))
        if key not in cache:
            p = to_params(np.clip(np.array(key), 0, 1), problem)
            ww, ff = physical(problem, p, *evaluate(p))
            cache[key] = constraints(problem, p, ww, ff)[0]
        return cache[key]
    p0 = to_params(np.array(start), problem)
    m_ref = mass(problem['a']/p0[0, 1], p0[0, 0])

    def objective(s):
        p = to_params(np.clip(s, 0, 1), problem)
        return float(mass(problem['a']/p[0, 1], p[0, 0])/m_ref)
    s = np.array(start, dtype=float)
    for _ in range(3):
        r = minimize(objective, s, method='SLSQP', bounds=[(0, 1), (0, 1)],
                     constraints=[dict(type='ineq', fun=response)], options=dict(ftol=1e-12, maxiter=300))
        x = np.clip(r.x, 0, 1)
        improved = np.all(response(x) >= -1e-9) and (not np.all(response(s) >= -1e-9) or objective(x) < objective(s)-1e-13)
        if improved:
            s = x
        else:
            break
    p = to_params(s, problem)
    c = response(s)
    return dict(g=float(p[0, 0]), a_over_h=float(p[0, 1]), mass=float(mass(problem['a']/p[0, 1], p[0, 0])),
                feasible=bool(np.all(c >= -FEAS_TOL)), constraints=c.tolist(), analyses=len(cache))


def random_problems(count, seed):
    """Part B problem set (FSDT only). Returns a list of problem dicts in generation order."""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(count):
        aspect = rng.uniform(0.5, 2.0)
        g = np.expm1(rng.uniform(0.1, 0.9)*np.log(9))
        r = np.exp(rng.uniform(np.log(6), np.log(60)))
        dw, df, dc = rng.uniform(-0.1, 0.1, 3)
        problem = dict(a=1.0, aspect=float(aspect), q0=1.0e4)
        anchor = np.array([[g, r, aspect]])
        w, f = physical(problem, anchor, *fsdt(anchor))
        out.append(dict(problem, w_lim=float(w[0]*(1+dw)), f_min=float(f[0]*(1+df)),
                        ceramic_max=float(ceramic(1.0/r, g)*(1+dc)), anchor=dict(g=float(g), a_over_h=float(r)),
                        perturbation=dict(deflection=float(dw), frequency=float(df), ceramic=float(dc)), index=i))
    return out


def max_deflection_ratio(params, k=TRUTH_K, n=41):
    """TSDT max|w| over an n x n grid divided by the centre value (uniform pressure); 1 means the centre is the maximum."""
    modes = odd_modes(k)
    H = transfer_batch(params, modes, 'TSDT')[:, :, 2]*(16/(np.pi**2*modes[:, 0]*modes[:, 1]))[None]
    t = np.linspace(0, 1, n)
    S = np.sin(np.pi*np.outer(np.arange(1, k+1), t))
    field = np.einsum('dk,kx,ky->dxy', H, S[modes[:, 0]-1], S[modes[:, 1]-1])
    centre = H@(centre_weights(modes)*np.pi**2*modes[:, 0]*modes[:, 1]/16)      # sum of H sin(m pi/2) sin(n pi/2)
    return np.abs(field).reshape(len(params), -1).max(axis=1)/np.abs(centre)
