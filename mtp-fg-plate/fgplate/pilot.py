"""Deterministic pilot sampling and design-level partitioning.

Two generator versions exist:

pilot-1.0  (delivered pilot_200)  design_id hashes the raw sampled floats and the
           mixture-load RNG is seeded from design_id. A last-bit difference in the
           sampled floats (different platform/library build) changes the ID and
           therefore the mixture load. Kept only so pilot_200 can be audited.
pilot-1.1  (default for new data)  sampled parameters are quantized to
           SIGNIFICANT_DIGITS significant digits and then used as the physical
           parameters; design_id hashes their canonical decimal strings; the
           mixture-load RNG is seeded from (seed, sample index), independent of
           floating-point representation.
"""
import hashlib
import json
import numpy as np
from scipy.stats import qmc
from .material import index

GENERATOR_VERSION = 'pilot-1.1'
LEGACY_VERSION = 'pilot-1.0'
SUPPORTED_VERSIONS = (LEGACY_VERSION, GENERATOR_VERSION)
SIGNIFICANT_DIGITS = 10
FIELDS = ('u0', 'v0', 'w0', 'phix', 'phiy')
LOADS = ('sine', 'uniform', 'smooth_mixture')


def _check_version(version):
    if version not in SUPPORTED_VERSIONS:
        raise ValueError(f'unknown generator version {version!r}')


def transformed_features(parameters):
    """Physical columns [g,a/h,b/a] -> predeclared coordinates."""
    x = np.asarray(parameters, dtype=float)
    if x.ndim != 2 or x.shape[1] != 3 or not np.isfinite(x).all():
        raise ValueError('parameters must be finite N by 3')
    if np.any(x[:, 0] < 0) or np.any(x[:, 1:] <= 0):
        raise ValueError('invalid physical parameters')
    return np.column_stack((np.log1p(x[:, 0]), np.log(x[:, 1]), x[:, 2]))


def canonical_string(value):
    """Platform-independent decimal text of a quantized parameter."""
    return f'{float(value):.{SIGNIFICANT_DIGITS}g}'


def quantize(parameters):
    """Round every parameter to SIGNIFICANT_DIGITS significant digits."""
    return np.vectorize(lambda v: float(canonical_string(v)))(np.asarray(parameters, dtype=float))


def design_id(g, slenderness, aspect, version=GENERATOR_VERSION):
    """Permanent hash of the physical inputs and support identifier."""
    _check_version(version)
    if version == LEGACY_VERSION:
        values = dict(b=float(aspect), h=float(1/slenderness), g=float(g))
    else:
        values = dict(b_over_a=canonical_string(aspect), a_over_h=canonical_string(slenderness),
                      g=canonical_string(g))
    physics = dict(a=1., Ec=380e9, Em=70e9, rhoc=3800., rhom=2702., nu=.3, ks=5/6,
                   support='navier_mixed_ssss', **values)
    return hashlib.sha256(json.dumps(physics, sort_keys=True).encode()).hexdigest()[:20]


def sample_designs(count=200, seed=20261109, version=GENERATOR_VERSION):
    _check_version(version)
    index(count, 'count', 10)
    unit = qmc.LatinHypercube(d=3, seed=seed).random(count)
    x = qmc.scale(unit, [0, np.log(5), .5], [np.log(9), np.log(100), 2])
    parameters = np.column_stack((np.expm1(x[:, 0]), np.exp(x[:, 1]), x[:, 2]))
    if version != LEGACY_VERSION:
        parameters = quantize(parameters)
    ids = np.array([design_id(*row, version=version) for row in parameters])
    if len(set(ids)) != count:
        raise ValueError('duplicate physical designs')
    # Labels are assigned before fields, scalers, models or rank analysis exist.
    order = np.random.default_rng(seed+1).permutation(count)
    split = np.full(count, 'test', dtype='<U10')
    split[order[:int(.7*count)]] = 'train'
    split[order[int(.7*count):int(.85*count)]] = 'validation'
    return parameters, ids, split


def load_key(version, sample_index, design_identifier):
    """The string that seeds a design's mixture load for a generator version."""
    _check_version(version)
    return design_identifier if version == LEGACY_VERSION else f'sample-{int(sample_index)}'


def design_loads(key, seed=20261109):
    """Sine, uniform and smooth-mixture loads; key comes from load_key()."""
    # Independent per-design stream makes loads independent of iteration order.
    stream = int(hashlib.sha256(f'{seed}:{key}'.encode()).hexdigest()[:16], 16)
    rng = np.random.default_rng(stream)
    pairs = [(m, n) for m in range(1, 6) for n in range(1, 6)]
    c = np.array([rng.normal()/(m*m+n*n) for m, n in pairs])
    norm = float(np.linalg.norm(c))
    c /= norm
    uniform = {(m, n): 16/(np.pi**2*m*n) for m in range(1, 50, 2) for n in range(1, 50, 2)}
    return [{(1, 1): 1.}, uniform, dict(zip(pairs, c))], norm
