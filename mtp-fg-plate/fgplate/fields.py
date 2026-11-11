"""Field nondimensionalization and reduced bases for field-learning experiments.

Characteristic scales (fixed before any fitting; qscale in Pa, lengths in m):

    w0             ~ qscale a^4 / (Ec h^3)   (classical bending deflection)
    phix, phiy     ~ qscale a^3 / (Ec h^3)   (= w scale / a)
    u0, v0         ~ qscale a^3 / (Ec h^2)   (membrane displacement from the
                                             neutral-surface offset ~ h times rotation)

Scaled fields are dimensionless and O(1) across the pilot domain, whereas the
raw SI fields span four to five orders of magnitude (w ~ (a/h)^3).

Grids are [eta, xi] with xi = x/a, eta = y/b, matching dataset.npz.
"""
import numpy as np

from .pilot import FIELDS

EC_DEFAULT = 380e9


def field_scales(a, h, qscale=1.0, Ec=EC_DEFAULT):
    """Per-design scale factors (u0, v0, w0, phix, phiy); arrays broadcast over designs."""
    a, h = np.asarray(a, dtype=float), np.asarray(h, dtype=float)
    inplane = qscale*a**3/(Ec*h**2)
    w = qscale*a**4/(Ec*h**3)
    rotation = qscale*a**3/(Ec*h**3)
    return np.stack(np.broadcast_arrays(inplane, inplane, w, rotation, rotation), axis=-1)


def scale_fields(fields, a, h, qscale=1.0, Ec=EC_DEFAULT):
    """SI fields [..., 5, ny, nx] -> dimensionless fields."""
    s = field_scales(a, h, qscale, Ec)
    return np.asarray(fields)/s[..., :, None, None]


def unscale_fields(scaled, a, h, qscale=1.0, Ec=EC_DEFAULT):
    s = field_scales(a, h, qscale, Ec)
    return np.asarray(scaled)*s[..., :, None, None]


def analytic_basis(modes, xi, eta):
    """Navier basis on the [eta, xi] grid: array [component, mode, ny, nx].

    Component k uses the same trigonometric form as its series in fgplate.navier:
    u0 cos.sin, v0 sin.cos, w0 sin.sin, phix cos.sin, phiy sin.cos.
    """
    modes = np.asarray(modes)
    m, n = modes[:, 0][:, None], modes[:, 1][:, None]
    sx, cx = np.sin(m*np.pi*np.asarray(xi)[None]), np.cos(m*np.pi*np.asarray(xi)[None])
    sy, cy = np.sin(n*np.pi*np.asarray(eta)[None]), np.cos(n*np.pi*np.asarray(eta)[None])
    forms = [(cx, sy), (sx, cy), (sx, sy), (cx, sy), (sx, cy)]
    return np.stack([np.einsum('ky,kx->kyx', fy, fx) for fx, fy in forms])


def project_analytic(fields, basis):
    """Least-squares modal coefficients [..., 5, modes] of fields [..., 5, ny, nx]."""
    fields = np.asarray(fields)
    lead = fields.shape[:-3]
    out = np.zeros(lead+(5, basis.shape[1]))
    for k in range(5):
        B = basis[k].reshape(basis.shape[1], -1).T
        Y = fields[..., k, :, :].reshape(-1, B.shape[0]).T
        out[..., k, :] = np.linalg.lstsq(B, Y, rcond=None)[0].T.reshape(lead+(basis.shape[1],))
    return out


def reconstruct_analytic(coefficients, basis):
    """Fields [..., 5, ny, nx] from modal coefficients [..., 5, modes]."""
    return np.einsum('...km,kmyx->...kyx', coefficients, basis)


class POD:
    """Uncentred snapshot POD of flattened scaled five-field vectors (training data only)."""

    def __init__(self, snapshots):
        X = np.asarray(snapshots, dtype=float).reshape(len(snapshots), -1)
        _, self.singular_values, vt = np.linalg.svd(X, full_matrices=False)
        self.modes = vt  # rows are orthonormal modes

    def coefficients(self, snapshots, rank):
        X = np.asarray(snapshots, dtype=float).reshape(len(snapshots), -1)
        return X@self.modes[:rank].T

    def reconstruct(self, coefficients, shape):
        rank = coefficients.shape[1]
        return (coefficients@self.modes[:rank]).reshape((len(coefficients),)+tuple(shape))


def relative_l2(prediction, truth):
    """Per-case relative L2 over all trailing axes."""
    p = np.asarray(prediction).reshape(len(prediction), -1)
    t = np.asarray(truth).reshape(len(truth), -1)
    return np.linalg.norm(p-t, axis=1)/np.linalg.norm(t, axis=1)


__all__ = ['FIELDS', 'field_scales', 'scale_fields', 'unscale_fields', 'analytic_basis',
           'project_analytic', 'reconstruct_analytic', 'POD', 'relative_l2']
