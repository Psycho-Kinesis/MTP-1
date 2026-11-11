"""Small positive-output scalar regression baselines; no deep-learning runtime."""
from itertools import product
import numpy as np
from scipy.interpolate import RBFInterpolator
from .pilot import transformed_features


def bootstrap_ci(errors, statistic='mean', level=.95, resamples=10000, seed=0):
    """Percentile bootstrap interval of a statistic ('mean', 'median' or 'p95') over cases (rows), per column.

    Rows must be independent cases (designs), never correlated grid points.
    """
    errors = np.asarray(errors, dtype=float)
    if errors.ndim == 1:
        errors = errors[:, None]
    func = {'mean': np.mean, 'median': np.median,
            'p95': lambda values, axis: np.quantile(values, .95, axis=axis)}[statistic]
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(errors), size=(resamples, len(errors)))
    stats = func(errors[draws], axis=1)
    tail = 100*(1-level)/2
    return np.percentile(stats, [tail, 100-tail], axis=0).T


def error_metrics(truth, prediction, ci=False):
    """Relative-error summary over cases; ci=True adds 95% bootstrap intervals (in %)."""
    rel = np.abs(prediction-truth)/np.maximum(np.abs(truth), 1e-15)
    result = dict(mean_percent=(100*rel.mean(axis=0)).tolist(),
                  median_percent=(100*np.median(rel, axis=0)).tolist(),
                  p95_percent=(100*np.quantile(rel, .95, axis=0)).tolist(),
                  worst_percent=(100*rel.max(axis=0)).tolist())
    if ci:
        result['mean_percent_ci95'] = (100*bootstrap_ci(rel, 'mean')).tolist()
        result['median_percent_ci95'] = (100*bootstrap_ci(rel, 'median')).tolist()
        result['cases'] = int(len(rel))
    return result


class ScalarBaseline:
    """Predict [sine wbar, fundamental omegabar] from [g,a/h,b/a].

    Both methods fit log labels. Statistics use fit inputs only. Hyperparameter
    selection belongs to the caller and must not use the final test partition.
    """
    def __init__(self, kind='polynomial', degree=3, regularization=1e-6):
        if kind not in ('polynomial', 'rbf'):
            raise ValueError('kind must be polynomial or rbf')
        self.kind, self.degree, self.regularization = kind, degree, regularization

    def _basis(self, x):
        powers = [p for p in product(range(self.degree+1), repeat=3) if sum(p) <= self.degree]
        return np.column_stack([np.prod(x**p, axis=1) for p in powers])

    def fit(self, parameters, labels):
        raw = transformed_features(parameters)
        labels = np.asarray(labels, dtype=float)
        if labels.shape != (len(raw), 2) or not np.isfinite(labels).all() or np.any(labels <= 0):
            raise ValueError('positive N by 2 labels required')
        self.mean, self.scale = raw.mean(axis=0), raw.std(axis=0)
        self.scale[self.scale == 0] = 1
        self.x = (raw-self.mean)/self.scale
        self.log_y = np.log(labels)
        if self.kind == 'rbf':
            self.interpolator = RBFInterpolator(self.x, self.log_y, kernel='thin_plate_spline',
                                                smoothing=self.regularization)
        else:
            A = self._basis(self.x)
            penalty = np.eye(A.shape[1])*np.sqrt(self.regularization)
            penalty[0, 0] = 0  # unpenalized intercept; least squares avoids normal equations
            self.weights = np.linalg.lstsq(np.vstack((A, penalty)),
                                           np.vstack((self.log_y, np.zeros((len(penalty), 2)))), rcond=None)[0]
        return self

    def predict(self, parameters):
        x = (transformed_features(parameters)-self.mean)/self.scale
        log_y = self.interpolator(x) if self.kind == 'rbf' else self._basis(x)@self.weights
        return np.exp(log_y)

    def save(self, path):
        np.savez_compressed(path, kind=self.kind, degree=self.degree, regularization=self.regularization,
                            mean=self.mean, scale=self.scale, x=self.x, log_y=self.log_y,
                            weights=getattr(self, 'weights', np.empty((0, 2))))

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as d:
            model = cls(str(d['kind']), int(d['degree']), float(d['regularization']))
            for name in ('mean', 'scale', 'x', 'log_y', 'weights'):
                setattr(model, name, d[name].copy())
        if model.kind == 'rbf':
            model.interpolator = RBFInterpolator(model.x, model.log_y, kernel='thin_plate_spline',
                                                 smoothing=model.regularization)
        return model
