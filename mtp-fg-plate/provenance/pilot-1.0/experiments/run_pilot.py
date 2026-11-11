"""Build the 200-design pilot, inspect shape rank, fit and evaluate a scalar baseline.

Run from the project root: python experiments/run_pilot.py --output results/pilot_200
The output directory must be absent or empty: a saved split/test result is never
silently overwritten. A new seed constitutes a different experiment.
"""
import argparse
import csv
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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from fgplate import NavierFSDTPlate, al_al2o3
from fgplate.pilot import sample_designs, design_loads, FIELDS, LOADS, GENERATOR_VERSION
from fgplate.baseline import ScalarBaseline, error_metrics


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def relative(a, b):
    return float(np.linalg.norm(a-b)/max(np.linalg.norm(b), 1e-30))


def generate(count, seed, out):
    parameters, ids, split = sample_designs(count, seed)
    write_json(out/'split.json', {part: ids[split == part].tolist() for part in ('train', 'validation', 'test')})
    modes = sorted(set(design_loads(ids[0], seed)[0][1]) | set(design_loads(ids[0], seed)[0][2]))
    mode_lookup = {pair: i for i, pair in enumerate(modes)}
    coefficients = np.zeros((count, 3, len(modes), 5))
    pressures = np.zeros((count, 3, len(modes)))
    fields = np.zeros((count, 3, 5, 41, 41))
    frequencies = np.zeros((count, 5))
    frequency_modes = np.zeros((count, 5, 2), dtype=int)
    participation = np.zeros((count, 5))
    scalar_targets = np.zeros((count, 2))
    mixture_normalizers = np.zeros(count)
    convergence = []
    agreement, eigen_residual, orthogonality = [], [], []
    start = perf_counter()
    for i, (g, slenderness, aspect) in enumerate(parameters):
        plate = NavierFSDTPlate(1., aspect, 1/slenderness, al_al2o3(g))
        loads, mixture_normalizers[i] = design_loads(ids[i], seed)
        for j, load in enumerate(loads):
            sol = plate.bending_modal(load, method='closed_form')
            _, _, f = sol.field_grid()
            fields[i, j] = np.stack([f[name] for name in FIELDS])
            for pair, values in sol.coeffs.items():
                k = mode_lookup[pair]
                coefficients[i, j, k], pressures[i, j, k] = values, load[pair]
            # Deterministic crosschecks cover ten designs and every load family.
            if i % max(1, count//10) == 0:
                matrix = plate.bending_modal(load)
                a, b = np.array(list(sol.coeffs.values())), np.array(list(matrix.coeffs.values()))
                agreement.append([relative(a[:, k], b[:, k]) for k in range(5)])
        coarse = plate.flexural_modes(4, 5)
        fine = plate.flexural_modes(6, 5)
        frequencies[i] = [m['omega'] for m in fine]
        frequency_modes[i] = [[m['m'], m['n']] for m in fine]
        participation[i] = [m['participation'] for m in fine]
        scan_error = float(np.max(np.abs(np.array([m['omega'] for m in coarse])/frequencies[i]-1)))
        eig = plate.eigenpairs()
        K, M = plate.matrices(1, 1)
        eig_lhs, eig_rhs = K@eig.vectors, (M@eig.vectors)*eig.omega**2
        eigen_residual.append(float(np.max(np.linalg.norm(eig_lhs-eig_rhs, axis=0)/
                                           (np.linalg.norm(eig_lhs, axis=0)+np.linalg.norm(eig_rhs, axis=0)))))
        orthogonality.append(float(np.max(abs(eig.vectors.T@M@eig.vectors-np.eye(5)))))
        refined = plate.bending(1., 'uniform', 99, method='closed_form')
        _, _, refined_fields = refined.field_grid()
        uniform_diff = [relative(fields[i, 1, k], refined_fields[name]) for k, name in enumerate(FIELDS)]
        center_diff = abs(fields[i, 1, 2, 20, 20]/refined.center_deflection()-1)
        convergence.append([*uniform_diff, center_diff, scan_error])
        scalar_targets[i] = [10*380e9*plate.h**3*fields[i, 0, 2, 20, 20],
                             frequencies[i, 0]/plate.h*np.sqrt(3800/380e9)]
        if (i+1) % 25 == 0:
            print(f'Generated {i+1}/{count} designs', flush=True)
    convergence = np.array(convergence)
    checks = dict(max_matrix_closed_form_component_relative_error=np.max(agreement, axis=0).tolist(),
                  max_generalized_eigen_residual=max(eigen_residual),
                  max_mass_orthogonality_error=max(orthogonality),
                  max_uniform_49_vs_99_field_relative_error=convergence[:, :5].max(axis=0).tolist(),
                  max_uniform_center_relative_error=float(convergence[:, 5].max()),
                  max_scan_4_vs_6_frequency_relative_error=float(convergence[:, 6].max()),
                  minimum_retained_transverse_participation=float(participation.min()))
    if max(checks['max_matrix_closed_form_component_relative_error']) > 1e-8 or max(eigen_residual) > 1e-7:
        raise RuntimeError('solver agreement or eigen residual quality gate failed')
    if convergence[:, 2].max() > 1e-3 or convergence[:, 6].max() > 1e-8:
        raise RuntimeError('deflection or frequency convergence quality gate failed')
    np.savez_compressed(out/'dataset.npz', parameters=parameters, design_ids=ids, split=split,
                        mode_indices=np.array(modes), pressure_coefficients=pressures,
                        solution_coefficients=coefficients, fields=fields,
                        omega_rad_s=frequencies, frequency_mode_indices=frequency_modes,
                        transverse_participation=participation, scalar_targets=scalar_targets,
                        mixture_raw_norm=mixture_normalizers, convergence=convergence,
                        xi=np.linspace(0, 1, 41), eta=np.linspace(0, 1, 41))
    with (out/'designs.csv').open('w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['design_id', 'split', 'g', 'a_over_h', 'b_over_a', 'sine_wbar', 'omega1_bar', 'omega1_rad_s'])
        for i in range(count):
            writer.writerow([ids[i], split[i], *parameters[i], *scalar_targets[i], frequencies[i, 0]])
    return parameters, ids, split, fields, scalar_targets, checks, perf_counter()-start


def shape_analysis(fields, split, out):
    """Amplitude-normalized w shapes; no test data enter the training spectrum."""
    result = {}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout='constrained')
    for j, name in enumerate(LOADS):
        X = fields[split == 'train', j, 2].reshape(np.sum(split == 'train'), -1)
        X = X/np.linalg.norm(X, axis=1, keepdims=True)
        s = np.linalg.svd(X, compute_uv=False)
        energy = np.cumsum(s*s)/np.sum(s*s)
        result[name] = dict(numerical_rank_at_1e_minus_10=int(np.sum(s > s[0]*1e-10)),
                            rank_99_percent=int(np.searchsorted(energy, .99)+1),
                            rank_99_9_percent=int(np.searchsorted(energy, .999)+1),
                            singular_values=s.tolist())
        axes[0].semilogy(np.arange(1, len(s)+1), np.maximum(s/s[0], 1e-16), label=name)
        axes[1].plot(np.arange(1, len(s)+1), energy, label=name)
    axes[0].set(xlabel='Mode number', ylabel='Singular value / first value', xlim=(1, 35), ylim=(1e-12, 1.1),
                title='Training deflection shape diversity')
    axes[1].set(xlabel='Number of modes', ylabel='Cumulative squared singular values', xlim=(1, 25), ylim=(.7, 1.005),
                title='Snapshot norm captured (not strain energy)')
    for ax in axes:
        ax.grid(alpha=.25); ax.legend(fontsize=8)
    fig.savefig(out/'shape_rank.png', dpi=180); plt.close(fig)
    # Show three independent mixture realizations in addition to the two controls.
    indices = np.flatnonzero(split == 'train')[:3]
    fig, axes = plt.subplots(1, 5, figsize=(13, 3), layout='constrained')
    for ax, (i, j, title) in zip(axes, [(indices[0], 0, 'Single sine'), (indices[0], 1, 'Uniform')]+
                                [(i, 2, f'Mixture: design {i}') for i in indices]):
        w = fields[i, j, 2]; w = w/np.max(abs(w))
        im = ax.imshow(w, origin='lower', extent=(0, 1, 0, 1), cmap='RdBu_r', vmin=-1, vmax=1)
        ax.set(xlabel='x / a', ylabel='y / b', title=title)
    fig.colorbar(im, ax=axes, shrink=.7, label='w / max |w|')
    fig.savefig(out/'sample_load_responses.png', dpi=180); plt.close(fig)
    return result


def train_baseline(parameters, targets, ids, split, out):
    train, val, test = (split == name for name in ('train', 'validation', 'test'))
    candidates = [('polynomial', degree, reg) for degree in (1, 2, 3, 4) for reg in (1e-8, 1e-4, 1e-2)]
    candidates += [('rbf', 1, reg) for reg in (0., 1e-6, 1e-3, .1)]
    rows, models = [], []
    start = perf_counter()
    for kind, degree, reg in candidates:
        model = ScalarBaseline(kind, degree, reg).fit(parameters[train], targets[train])
        metrics = error_metrics(targets[val], model.predict(parameters[val]))
        score = float(np.mean(metrics['mean_percent']))
        rows.append(dict(kind=kind, degree=degree, regularization=reg, validation_mean_percent=score))
        models.append(model)
    best = int(np.argmin([row['validation_mean_percent'] for row in rows]))
    model = models[best]
    model.save(out/'scalar_model.npz')
    restored = ScalarBaseline.load(out/'scalar_model.npz')
    # Test evaluated once, after candidate selection and persistence.
    prediction = restored.predict(parameters[test])
    np.testing.assert_allclose(prediction, model.predict(parameters[test]), rtol=1e-13)
    validation_prediction = model.predict(parameters[val])
    constant = np.broadcast_to(np.exp(np.log(targets[train]).mean(axis=0)), prediction.shape)
    result = dict(target_order=['sine_wbar', 'omega1_bar'], selection='mean validation relative error across two outputs',
                  selected=rows[best], candidates=rows, fitting_and_selection_seconds=perf_counter()-start,
                  validation=error_metrics(targets[val], validation_prediction), test=error_metrics(targets[test], prediction),
                  constant_training_log_mean_test=error_metrics(targets[test], constant))
    np.savez_compressed(out/'scalar_predictions.npz', test_ids=ids[test], test_truth=targets[test], test_prediction=prediction,
                        validation_ids=ids[val], validation_truth=targets[val], validation_prediction=validation_prediction)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), layout='constrained')
    for j, (ax, name) in enumerate(zip(axes, result['target_order'])):
        ax.scatter(targets[test, j], prediction[:, j], c=parameters[test, 0], cmap='viridis', edgecolors='k', linewidths=.3)
        lo, hi = min(targets[test, j].min(), prediction[:, j].min()), max(targets[test, j].max(), prediction[:, j].max())
        ax.plot([lo, hi], [lo, hi], 'k--', linewidth=1)
        ax.set(xlabel='Reference solver', ylabel='Scalar baseline', title=f'{name}: held-out designs')
        ax.grid(alpha=.25)
    fig.savefig(out/'scalar_test_predictions.png', dpi=180); plt.close(fig)
    return result


def benchmark(parameters):
    plate = NavierFSDTPlate(1, float(parameters[0, 2]), float(1/parameters[0, 1]), al_al2o3(float(parameters[0, 0])))
    loads = design_loads('timing')[0][1]
    result = {}
    for with_grid in (False, True):
        for method in ('matrix', 'closed_form'):
            timings = []
            for repeat in range(12):
                start = perf_counter()
                solution = plate.bending_modal(loads, method=method)
                if with_grid: solution.field_grid()
                if repeat >= 2: timings.append(perf_counter()-start)
            result[method+('_five_fields_41x41' if with_grid else '_coefficients')] = dict(
                median_ms=float(1000*np.median(timings)), min_ms=float(1000*min(timings)), repeats=len(timings))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--count', type=int, default=200)
    parser.add_argument('--seed', type=int, default=20261109)
    parser.add_argument('--output', type=Path, default=ROOT/'results'/'pilot_200')
    args = parser.parse_args()
    out = args.output.resolve()
    if out.exists() and any(out.iterdir()):
        parser.error('output directory is not empty; choose a new experiment directory')
    out.mkdir(parents=True, exist_ok=True)
    parameters, ids, split, fields, targets, checks, seconds = generate(args.count, args.seed, out)
    print('Dataset complete; inspecting training shape rank and fitting scalar baselines', flush=True)
    shapes = shape_analysis(fields, split, out)
    ml = train_baseline(parameters, targets, ids, split, out)
    timings = benchmark(parameters)
    fig, ax = plt.subplots(figsize=(6, 4), layout='constrained')
    for name, marker in [('train', 'o'), ('validation', '^'), ('test', 's')]:
        sel = split == name
        ax.scatter(parameters[sel, 0], parameters[sel, 1], label=name, marker=marker, s=20, alpha=.7)
    ax.set(xlabel='Gradation g', ylabel='a / h (log scale)', yscale='log', title='200-design pilot coverage' if args.count == 200 else 'Pilot coverage')
    ax.legend(); fig.savefig(out/'design_coverage.png', dpi=180); plt.close(fig)
    manifest = dict(schema_version=1, generator_version=GENERATOR_VERSION, seed=args.seed,
                    created_utc=datetime.now(timezone.utc).isoformat(), designs=args.count, field_cases=args.count*3,
                    partitions={part: int(np.sum(split == part)) for part in ('train', 'validation', 'test')},
                    parameter_order=['g', 'a_over_h', 'b_over_a'], bounds=[[0, 8], [5, 100], [.5, 2]],
                    fixed_physics=dict(a_m=1, Ec_Pa=380e9, Em_Pa=70e9, rhoc_kg_m3=3800, rhom_kg_m3=2702,
                                       nu=.3, ks=5/6, supports='navier_mixed_ssss', model='linear_constant_nu_FSDT'),
                    load_order=LOADS, field_order=FIELDS, field_units=['m', 'm', 'm', '1', '1'],
                    qscale_Pa=1, uniform_max_mode=49, mixture_max_mode=5, grid=[41, 41],
                    pressure_definition='q=qscale*sum Qmn*sin(m*pi*xi)*sin(n*pi*eta); stored coefficients in Pa',
                    normalization=dict(wbar='10*Ec*h^3*w/(qscale*a^4)', omega_bar='omega*a^2/h*sqrt(rhoc/Ec)',
                                       fields='stored in SI; no fitted field scaling; ranks use unit-norm w snapshots'),
                    convergence_columns=[name+'_relative_L2_49_vs_99' for name in FIELDS]+['center_w_49_vs_99', 'frequency_scan_4_vs_6'],
                    generation_seconds=seconds, quality_checks=checks, shape_analysis=shapes, scalar_baseline=ml, timings=timings,
                    environment=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                                     matplotlib=matplotlib.__version__, platform=platform.platform()),
                    source_sha256={str(p.relative_to(ROOT)).replace('\\', '/'): sha256(p) for p in
                                   [*sorted((ROOT/'fgplate').glob('*.py')), Path(__file__)]},
                    limitations=['One pilot seed; only 30 test designs for count=200.',
                                 'Scalar baseline only predicts sine deflection and fundamental frequency; no arbitrary-load ML claim.',
                                 'Signed mixtures; no claim of one-sided physical pressure.',
                                 'No FEM, experimental validation, PINN, inverse design, held-out Gaussian loads or extrapolation study yet.',
                                 '41x41 grid norms/maxima are discrete; derivative/stress convergence is not established.'])
    write_json(out/'manifest.json', manifest)
    write_json(out/'checksums.json', {p.name: sha256(p) for p in sorted(out.iterdir()) if p.is_file()})
    print(json.dumps(dict(output=str(out), checks=checks, selected_model=ml['selected'], test=ml['test'],
                          shape_ranks={k: v['rank_99_9_percent'] for k, v in shapes.items()}), indent=2), flush=True)


if __name__ == '__main__':
    main()
