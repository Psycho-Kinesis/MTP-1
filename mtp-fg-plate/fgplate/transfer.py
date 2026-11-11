"""Parametric modal transfer functions learned from labels, from physics, or both.

For the specified Navier supports the static response to the load coefficient Q_mn
is Q_mn * H_mn(design), where H_mn = K_mn^{-1} e_w (five components U,V,W,X,Y per
unit load). In the dimensionless field scaling of fgplate.fields, S = diag(scales):

    H~ = S^{-1} H,   K~ = S K S,   f~ = S e_w,   K~ H~ = f~.

Model class (shared by every training arm):  H~_c(x; theta) = phi(x) . theta_c,
phi = tensor Legendre polynomials of total degree <= d in the three design
coordinates mapped from the FIXED pilot domain box to [-1, 1] (no data statistics).

Training losses, all quadratic in theta, so every fit is one linear solve per mode:

    data      (1/N) sum_labelled sum_c (H~_c(x) - label_c)^2 / sigma_c^2
    physics   (1/M) sum_collocation [ 1/2 H~^T K^ H~ - f^ . H~ ],  K^ = K~/K~_ww, f^ = f~/K~_ww
    ridge     rho * mean(diag(Hessian)) * |theta|^2 per mode (constant term excluded);
              relative, so the same rho means the same thing for every arm and mode

The physics term needs the mode matrices at unlabelled collocation designs but no
solve; its minimizer per design is the exact transfer function. Dividing by K~_ww
fixes the relative weight of designs and modes without using any solution.
"""
from itertools import product
import numpy as np
from numpy.polynomial import legendre
from scipy.linalg import cho_factor, cho_solve

from .fields import field_scales
from .material import al_al2o3, plate_stiffness
from .navier import NavierFSDTPlate
from .pilot import transformed_features

# Pilot domain in transformed coordinates [log(1+g), log(a/h), b/a]; fixed a priori.
DOMAIN = np.array([[0.0, np.log(9.0)], [np.log(5.0), np.log(100.0)], [0.5, 2.0]])
W = 2  # index of the transverse component in (U, V, W, X, Y)


def design_coordinates(parameters, aspect='linear'):
    """Physical [g, a/h, b/a] -> coordinates in [-1, 1]^3 using the fixed domain box.

    aspect='linear' (default, E3a-E4) maps b/a linearly; aspect='log' maps log(b/a), which
    treats narrow (b/a < 1) and wide (b/a > 1) plates symmetrically.
    """
    t = transformed_features(parameters)
    box = DOMAIN.copy()
    if aspect == 'log':
        t = t.copy()
        t[:, 2] = np.log(t[:, 2])
        box[2] = np.log(DOMAIN[2])
    elif aspect != 'linear':
        raise ValueError("aspect must be 'linear' or 'log'")
    return 2*(t-box[:, 0])/(box[:, 1]-box[:, 0])-1


def classical_prefactor(parameters, modes, a=1.0):
    """Classical-plate scaling of each transfer component, [designs, modes, 5] (label-free, no solve).

    With alpha a = m pi, beta a = n pi a/b and k^4 = ((alpha a)^2 + (beta a)^2)^2:
        w0 ~ 1/k^4,  phix ~ alpha a/k^4,  phiy ~ beta a/k^4,
        u0 ~ (z0/h) alpha a/k^4,  v0 ~ (z0/h) beta a/k^4,   z0/h = B11/(A11 h)  (neutral-axis offset).
    Dividing exact FSDT transfer functions by these leaves a spread of about 3 (u, v, rotations) and
    about 40 (w, shear part) across the pilot domain, instead of up to 4e4.
    """
    parameters = np.asarray(parameters, dtype=float)
    modes = np.asarray(modes)
    aspect = parameters[:, 2]
    am = modes[:, 0][None]*np.pi*np.ones((len(parameters), 1))
    bn = modes[:, 1][None]*np.pi/aspect[:, None]
    k4 = (am**2+bn**2)**2
    z0h = np.array([(lambda S, h: S.B11/(S.A11*h))(plate_stiffness(al_al2o3(float(g)), a/r), a/r)
                    for g, r, _ in parameters])[:, None]
    return np.stack([z0h*am/k4, z0h*bn/k4, 1/k4, am/k4, bn/k4], axis=2)


def exponents(degree):
    return [p for p in product(range(degree+1), repeat=3) if sum(p) <= degree]


def features(x, degree):
    """Tensor Legendre features of total degree <= degree; column 0 is the constant."""
    x = np.asarray(x, dtype=float)
    columns = []
    for p in exponents(degree):
        col = np.ones(len(x))
        for j, power in enumerate(p):
            if power:
                col = col*legendre.legval(x[:, j], [0]*power+[1])
        columns.append(col)
    return np.column_stack(columns)


def mode_matrices(parameters, modes, a=1.0, Ec=380e9):
    """Normalized scaled stiffness K^ [N, modes, 5, 5] and load f^ [N, modes, 5].

    Uses the plate mechanics only (no linear solve). qscale = 1 Pa.
    """
    parameters = np.asarray(parameters, dtype=float)
    K = np.zeros((len(parameters), len(modes), 5, 5))
    f = np.zeros((len(parameters), len(modes), 5))
    for i, (g, slenderness, aspect) in enumerate(parameters):
        h = a/slenderness
        plate = NavierFSDTPlate(a, aspect*a, h, al_al2o3(float(g)))
        s = field_scales(a, h, Ec=Ec)
        for k, (m, n) in enumerate(modes):
            Kt = s[:, None]*plate.matrices(int(m), int(n))[0]*s[None, :]
            K[i, k] = Kt/Kt[W, W]
            f[i, k, W] = s[W]/Kt[W, W]
    return K, f


def exact_transfer(K, f):
    """Reference H~ = K^{-1} f per design and mode (for labels and tests only)."""
    return np.linalg.solve(K, f[..., None])[..., 0]


def assemble_data(x, labels, degree, prefactor=None):
    """Normal equations of the data loss: A [modes, 5P, 5P], b [modes, 5P], sigma [modes, 5].

    prefactor (optional, broadcastable to labels' shape [N, modes, 5]) makes the model
    prefactor * phi(x).theta instead of phi(x).theta; it is fixed, not learned
    (e.g. (k h)^2 or a physics prediction times (k h)^2). Evaluate with predict_scaled().
    """
    Fd = features(x, degree)
    P, modes = Fd.shape[1], labels.shape[1]
    sigma = labels.std(axis=0)                                   # labelled designs only
    sigma[sigma == 0] = 1
    pref = None if prefactor is None else np.broadcast_to(prefactor, labels.shape)
    G = Fd.T@Fd/len(Fd)
    A = np.zeros((modes, 5*P, 5*P))
    b = np.zeros((modes, 5*P))
    for k in range(modes):
        for c in range(5):
            sl = slice(c*P, (c+1)*P)
            Fk = Fd if pref is None else pref[:, k, c][:, None]*Fd
            A[k, sl, sl] = 2*(G if pref is None else Fk.T@Fk/len(Fd))/sigma[k, c]**2
            b[k, sl] = 2*Fk.T@labels[:, k, c]/len(Fd)/sigma[k, c]**2
    return A, b, sigma


def predict_scaled(theta, degree, x, prefactor=None):
    """prefactor * phi(x).theta  ->  [N, modes, 5]."""
    out = np.einsum('np,kcp->nkc', features(x, degree), theta)
    return out if prefactor is None else np.broadcast_to(prefactor, out.shape)*out


def assemble_physics(x, K, f, degree, prefactor=None):
    """Normal equations of the energy loss averaged over collocation designs (weight 1).

    prefactor (optional, [M, modes, 5]): the model is prefactor * phi(x).theta, so the
    energy stays quadratic in theta with K^_ij -> s_i K^_ij s_j and f^_i -> s_i f^_i.
    """
    Fp = features(x, degree)
    P, modes = Fp.shape[1], K.shape[1]
    if prefactor is not None:
        s = np.broadcast_to(prefactor, f.shape)
        K = s[:, :, :, None]*K*s[:, :, None, :]
        f = s*f
    A = np.zeros((modes, 5*P, 5*P))
    b = np.zeros((modes, 5*P))
    for k in range(modes):
        for i in range(5):
            b[k, i*P:(i+1)*P] = Fp.T@f[:, k, i]/len(Fp)
            for j in range(i, 5):
                block = (Fp.T*K[:, k, i, j])@Fp/len(Fp)
                A[k, i*P:(i+1)*P, j*P:(j+1)*P] = block
                A[k, j*P:(j+1)*P, i*P:(i+1)*P] = block.T
    return A, b


def solve_modes(A, b, degree, ridge, scope='mode'):
    """theta [modes, 5, P] from normal equations with the relative ridge.

    scope='mode'  (E3a): ridge scaled by the mean Hessian diagonal of the whole mode.
    scope='block' (E3b onwards): ridge scaled separately for each field component's
                  diagonal block, so the fit is equivariant to rescaling one component
                  (needed when components differ by orders of magnitude, e.g. residuals).

    Equilibrated Cholesky solve; minimum-norm least squares if the system is not
    numerically positive definite (e.g. more coefficients than informative designs).
    """
    P = len(exponents(degree))
    mask = np.ones(P)
    mask[0] = 0
    mask = np.tile(mask, 5)
    theta = np.zeros((A.shape[0], 5*P))
    for k in range(A.shape[0]):
        diag = np.diag(A[k])
        if scope == 'mode':
            scale = np.full(5*P, np.mean(diag))
        elif scope == 'block':
            scale = np.repeat([np.mean(diag[c*P:(c+1)*P]) for c in range(5)], P)
        else:
            raise ValueError("scope must be 'mode' or 'block'")
        Ak = A[k]+2*ridge*np.diag(scale*mask)                     # gradient of the relative ridge
        d = 1/np.sqrt(np.maximum(np.diag(Ak), np.finfo(float).tiny))
        Ae, be = d[:, None]*Ak*d[None, :], d*b[k]
        try:
            y = cho_solve(cho_factor(Ae), be)
        except np.linalg.LinAlgError:
            y = np.linalg.lstsq(Ae, be, rcond=None)[0]
        theta[k] = d*y
    return theta.reshape(A.shape[0], 5, P)


class TransferModel:
    """Fit H~(x) for every mode with any combination of data and physics losses."""

    def __init__(self, degree, ridge=1e-8, scope='mode'):
        if ridge < 0:
            raise ValueError('ridge must be nonnegative')
        self.degree, self.ridge, self.scope = int(degree), float(ridge), scope

    def fit(self, x_data=None, labels=None, x_phys=None, K=None, f=None, physics_weight=1.0):
        """labels [N, modes, 5]; K [M, modes, 5, 5]; f [M, modes, 5]. Either part may be None."""
        if labels is None and K is None:
            raise ValueError('need labels, physics, or both')
        parts = []
        if labels is not None:
            A, b, self.sigma = assemble_data(x_data, labels, self.degree)
            parts.append((A, b))
        if K is not None:
            A, b = assemble_physics(x_phys, K, f, self.degree)
            parts.append((physics_weight*A, physics_weight*b))
        return self.set_theta(solve_modes(sum(p[0] for p in parts), sum(p[1] for p in parts),
                                          self.degree, self.ridge, self.scope))

    def set_theta(self, theta):
        self.theta = theta
        return self

    def predict(self, x):
        """H~ [N, modes, 5]."""
        return np.einsum('np,kcp->nkc', features(x, self.degree), self.theta)
