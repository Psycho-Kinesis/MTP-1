"""Deterministic pilot sampling and design-level partitioning."""
import hashlib
import json
import numpy as np
from scipy.stats import qmc
from .material import index

GENERATOR_VERSION = 'pilot-1.0'
FIELDS = ('u0', 'v0', 'w0', 'phix', 'phiy')
LOADS = ('sine', 'uniform', 'smooth_mixture')


def transformed_features(parameters):
    """Physical columns [g,a/h,b/a] -> predeclared coordinates."""
    x = np.asarray(parameters, dtype=float)
    if x.ndim != 2 or x.shape[1] != 3 or not np.isfinite(x).all():
        raise ValueError('parameters must be finite N by 3')
    if np.any(x[:, 0] < 0) or np.any(x[:, 1:] <= 0):
        raise ValueError('invalid physical parameters')
    return np.column_stack((np.log1p(x[:, 0]), np.log(x[:, 1]), x[:, 2]))


def sample_designs(count=200, seed=20261109):
    index(count, 'count', 10)
    unit = qmc.LatinHypercube(d=3, seed=seed).random(count)
    x = qmc.scale(unit, [0, np.log(5), .5], [np.log(9), np.log(100), 2])
    parameters = np.column_stack((np.expm1(x[:, 0]), np.exp(x[:, 1]), x[:, 2]))
    ids = []
    for g, slenderness, aspect in parameters:
        physics = dict(a=1., b=float(aspect), h=float(1/slenderness), g=float(g),
                       Ec=380e9, Em=70e9, rhoc=3800., rhom=2702., nu=.3, ks=5/6,
                       support='navier_mixed_ssss')
        ids.append(hashlib.sha256(json.dumps(physics, sort_keys=True).encode()).hexdigest()[:20])
    if len(set(ids)) != count:
        raise ValueError('duplicate physical designs')
    # Labels are assigned before fields, scalers, models or rank analysis exist.
    order = np.random.default_rng(seed+1).permutation(count)
    split = np.full(count, 'test', dtype='<U10')
    split[order[:int(.7*count)]] = 'train'
    split[order[int(.7*count):int(.85*count)]] = 'validation'
    return parameters, np.array(ids), split


def design_loads(design_id, seed=20261109):
    # Independent per-design stream makes loads independent of iteration order.
    stream = int(hashlib.sha256(f'{seed}:{design_id}'.encode()).hexdigest()[:16], 16)
    rng = np.random.default_rng(stream)
    pairs = [(m, n) for m in range(1, 6) for n in range(1, 6)]
    c = np.array([rng.normal()/(m*m+n*n) for m, n in pairs])
    norm = float(np.linalg.norm(c))
    c /= norm
    uniform = {(m, n): 16/(np.pi**2*m*n) for m in range(1, 50, 2) for n in range(1, 50, 2)}
    return [{(1, 1): 1.}, uniform, dict(zip(pairs, c))], norm
