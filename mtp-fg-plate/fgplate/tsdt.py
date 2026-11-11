"""Navier per-mode stiffness for FSDT and Reddy's third-order theory (TSDT), static bending.

Both theories use the same five generalized fields (u0, v0, w0, phix, phiy) and the
same Navier trigonometric expansions as fgplate.navier, so each half-wave pair (m, n)
gives a 5x5 system K q = Q e_w. Displacements:

    FSDT  u = u0 + z phix
    TSDT  u = u0 + z phix - c1 z^3 (phix + w0,x),   c1 = 4/(3 h^2)   (likewise v)

For one mode every strain is a single trigonometric product times a z-polynomial
that is linear in q = [U, V, W, X, Y]. With a(z) these coefficient vectors,

    K = int E(z) [ (a_ex a_ex^T + a_ey a_ey^T + nu(a_ex a_ey^T + a_ey a_ex^T))/(1-nu^2)
                   + a_gxy a_gxy^T / (2(1+nu))
                   + k_s (a_gxz a_gxz^T + a_gyz a_gyz^T) / (2(1+nu)) ] dz

(the common in-plane factor ab/4 cancels against the load). The z-integrals use the
exact power-law moments int E z^k dz, k <= 6, so there is no quadrature error.
k_s = 5/6 for FSDT and 1 for TSDT (parabolic shear strain, no correction factor).
Setting c1 = 0 reproduces fgplate.navier.NavierFSDTPlate.matrices exactly.

Free vibration uses the consistent mass matrix of the same displacement field,

    M = int rho(z) [ a_u a_u^T + a_v a_v^T + a_w a_w^T ] dz

(exact density moments, k <= 6). With c1 = 0 it reproduces the FSDT mass matrix (I0, I1, I2).
The *_batch functions evaluate many designs and modes at once (identical mechanics, vectorized).
"""
import numpy as np

from .fields import field_scales
from .material import FGMaterial, al_al2o3, positive, index

EC, RHO_C = 380e9, 3800.0   # al_al2o3 ceramic modulus and density (frequency scaling)

THEORIES = ('FSDT', 'TSDT')
W = 2


def _strain_polynomials(alpha, beta, c1):
    """Coefficient arrays [component, 5 unknowns, z-power 0..3] for ex, ey, gxy, gxz, gyz."""
    def vec(entries):
        out = np.zeros((5, 4))
        for (unknown, power), value in entries.items():
            out[unknown, power] += value
        return out
    U, V, Wd, X, Y = range(5)
    u_part = vec({(U, 0): 1, (X, 1): 1, (X, 3): -c1, (Wd, 3): -c1*alpha})      # u amplitude
    v_part = vec({(V, 0): 1, (Y, 1): 1, (Y, 3): -c1, (Wd, 3): -c1*beta})       # v amplitude
    ex = -alpha*u_part
    ey = -beta*v_part
    gxy = beta*u_part+alpha*v_part
    gxz = vec({(X, 0): 1, (Wd, 0): alpha, (X, 2): -3*c1, (Wd, 2): -3*c1*alpha})
    gyz = vec({(Y, 0): 1, (Wd, 0): beta, (Y, 2): -3*c1, (Wd, 2): -3*c1*beta})
    return ex, ey, gxy, gxz, gyz


def _quadratic(a, b, moments):
    """int E(z) (a(z) b(z)^T) dz for coefficient arrays a, b [5, 4]."""
    out = np.zeros((5, 5))
    for p in range(4):
        for q in range(4):
            out += moments[p+q]*np.outer(a[:, p], b[:, q])
    return out


def stiffness(material: FGMaterial, h, a, b, m, n, theory='TSDT', ks=5.0/6.0):
    """Static 5x5 stiffness of one Navier half-wave pair (SI units)."""
    if theory not in THEORIES:
        raise ValueError(f'theory must be one of {THEORIES}')
    for name, value in (('h', h), ('a', a), ('b', b), ('ks', ks)):
        positive(value, name)
    index(m, 'm'); index(n, 'n')
    alpha, beta = m*np.pi/a, n*np.pi/b
    c1 = 4/(3*h**2) if theory == 'TSDT' else 0.0
    shear_factor = 1.0 if theory == 'TSDT' else ks
    moments = [material.moment('E', k, h) for k in range(7)]
    ex, ey, gxy, gxz, gyz = _strain_polynomials(alpha, beta, c1)
    nu = material.nu
    K = (_quadratic(ex, ex, moments)+_quadratic(ey, ey, moments)
         + nu*(_quadratic(ex, ey, moments)+_quadratic(ey, ex, moments)))/(1-nu**2)
    K += _quadratic(gxy, gxy, moments)/(2*(1+nu))
    K += shear_factor*(_quadratic(gxz, gxz, moments)+_quadratic(gyz, gyz, moments))/(2*(1+nu))
    return 0.5*(K+K.T)


def transfer(parameters, modes, theory='TSDT', a=1.0, Ec=380e9):
    """Exact dimensionless transfer functions H~ [designs, modes, 5] (fgplate.fields scaling, qscale 1 Pa)."""
    parameters = np.asarray(parameters, dtype=float)
    H = np.zeros((len(parameters), len(modes), 5))
    load = np.zeros(5)
    load[W] = 1.0
    for i, (g, slenderness, aspect) in enumerate(parameters):
        h = a/slenderness
        material = al_al2o3(float(g))
        s = field_scales(a, h, Ec=Ec)
        for k, (m, n) in enumerate(modes):
            H[i, k] = np.linalg.solve(stiffness(material, h, a, aspect*a, int(m), int(n), theory), load)/s
    return H


# ---------------------------------------------------------------- free vibration
def _displacement_polynomials(alpha, beta, c1):
    """Coefficient arrays [5 unknowns, z-power 0..3] of the u, v, w amplitudes of one mode."""
    u, v, w = np.zeros((5, 4)), np.zeros((5, 4)), np.zeros((5, 4))
    u[0, 0], u[3, 1], u[3, 3], u[W, 3] = 1, 1, -c1, -c1*alpha
    v[1, 0], v[4, 1], v[4, 3], v[W, 3] = 1, 1, -c1, -c1*beta
    w[W, 0] = 1
    return u, v, w


def mass(material: FGMaterial, h, a, b, m, n, theory='TSDT'):
    """Consistent 5x5 mass matrix of one Navier half-wave pair (SI units, same ab/4 convention as stiffness)."""
    if theory not in THEORIES:
        raise ValueError(f'theory must be one of {THEORIES}')
    for name, value in (('h', h), ('a', a), ('b', b)):
        positive(value, name)
    index(m, 'm'); index(n, 'n')
    alpha, beta = m*np.pi/a, n*np.pi/b
    c1 = 4/(3*h**2) if theory == 'TSDT' else 0.0
    moments = [material.moment('rho', k, h) for k in range(7)]
    M = sum(_quadratic(part, part, moments) for part in _displacement_polynomials(alpha, beta, c1))
    return 0.5*(M+M.T)


def eigenpairs(material: FGMaterial, h, a, b, m=1, n=1, theory='TSDT', ks=5.0/6.0):
    """All five branches of one half-wave pair: omega (rad/s, ascending), M-normalized vectors and
    transverse participation I0 W^2 (share of kinetic energy carried by the transverse velocity)."""
    from scipy.linalg import eigh
    K = stiffness(material, h, a, b, m, n, theory, ks)
    M = mass(material, h, a, b, m, n, theory)
    vals, vecs = eigh(K, M)
    if np.any(vals <= 0):
        raise ValueError('nonpositive eigenvalue')
    return np.sqrt(vals), vecs, material.moment('rho', 0, h)*vecs[W]**2


# ---------------------------------------------------------------- vectorized evaluation (many designs, many modes)
def _moment_table(parameters, prop, a):
    out = np.empty((len(parameters), 7))
    for i, (g, ratio, _) in enumerate(parameters):
        material, h = al_al2o3(float(g)), a/ratio
        out[i] = [material.moment(prop, k, h) for k in range(7)]
    return out


def _hankel(moments):
    p = np.arange(4)
    return moments[:, p[:, None]+p[None, :]]


def _batch_parts(parameters, modes, theory, a):
    parameters, modes = np.asarray(parameters, dtype=float), np.asarray(modes)
    if theory not in THEORIES:
        raise ValueError(f'theory must be one of {THEORIES}')
    h = a/parameters[:, 1]
    alpha = np.broadcast_to(modes[None, :, 0]*np.pi/a, (len(parameters), len(modes)))
    beta = modes[None, :, 1]*np.pi/(parameters[:, 2:3]*a)
    c = np.broadcast_to((4/(3*h**2) if theory == 'TSDT' else 0*h)[:, None], alpha.shape)
    zero = lambda: np.zeros(alpha.shape+(5, 4))
    u, v, w = zero(), zero(), zero()
    u[..., 0, 0], u[..., 3, 1], u[..., 3, 3], u[..., W, 3] = 1, 1, -c, -c*alpha
    v[..., 1, 0], v[..., 4, 1], v[..., 4, 3], v[..., W, 3] = 1, 1, -c, -c*beta
    w[..., W, 0] = 1
    A, B = alpha[..., None, None], beta[..., None, None]
    gxz, gyz = zero(), zero()
    gxz[..., 3, 0], gxz[..., W, 0], gxz[..., 3, 2], gxz[..., W, 2] = 1, alpha, -3*c, -3*c*alpha
    gyz[..., 4, 0], gyz[..., W, 0], gyz[..., 4, 2], gyz[..., W, 2] = 1, beta, -3*c, -3*c*beta
    return dict(u=u, v=v, w=w, ex=-A*u, ey=-B*v, gxy=B*u+A*v, gxz=gxz, gyz=gyz)


def _batch_quadratic(x, y, hankel):
    return np.einsum('nmip,npq,nmjq->nmij', x, hankel, y, optimize=True)


def stiffness_batch(parameters, modes, theory='TSDT', a=1.0, ks=5.0/6.0, nu=0.3):
    """Stiffness [designs, modes, 5, 5] for parameters [g, a/h, b/a] (al_al2o3, SI units)."""
    s = _batch_parts(parameters, modes, theory, a)
    H = _hankel(_moment_table(parameters, 'E', a))
    Q = lambda x, y: _batch_quadratic(s[x], s[y], H)
    K = (Q('ex', 'ex')+Q('ey', 'ey')+nu*(Q('ex', 'ey')+Q('ey', 'ex')))/(1-nu**2)+Q('gxy', 'gxy')/(2*(1+nu))
    K = K+(1.0 if theory == 'TSDT' else ks)*(Q('gxz', 'gxz')+Q('gyz', 'gyz'))/(2*(1+nu))
    return 0.5*(K+np.swapaxes(K, -1, -2))


def mass_batch(parameters, modes, theory='TSDT', a=1.0, rho_moments=None):
    """Consistent mass [designs, modes, 5, 5] (al_al2o3, SI units)."""
    s = _batch_parts(parameters, modes, theory, a)
    H = _hankel(_moment_table(parameters, 'rho', a) if rho_moments is None else rho_moments)
    M = sum(_batch_quadratic(s[k], s[k], H) for k in ('u', 'v', 'w'))
    return 0.5*(M+np.swapaxes(M, -1, -2))


def transfer_batch(parameters, modes, theory='TSDT', a=1.0, Ec=EC):
    """Vectorized fgplate.tsdt.transfer: dimensionless H~ [designs, modes, 5] (qscale 1 Pa)."""
    parameters = np.asarray(parameters, dtype=float)
    K = stiffness_batch(parameters, modes, theory, a)
    rhs = np.zeros(K.shape[:-1]+(1,))
    rhs[..., W, 0] = 1.0
    return np.linalg.solve(K, rhs)[..., 0]/field_scales(a, a/parameters[:, 1], Ec=Ec)[:, None, :]


def fundamental_batch(parameters, theory='TSDT', a=1.0):
    """Dimensionless fundamental flexural frequency omega_hat = omega_11 h sqrt(rho_c/E_c) per design.

    Uses the (1,1) half-wave pair, the lowest flexural mode of a simply supported rectangular plate,
    and the branch with the largest transverse participation; raises if a lower (in-plane or shear)
    branch of the same pair exists, which does not happen in the pilot domain.
    """
    parameters = np.asarray(parameters, dtype=float)
    rho = _moment_table(parameters, 'rho', a)
    K = stiffness_batch(parameters, [(1, 1)], theory, a)[:, 0]
    M = mass_batch(parameters, [(1, 1)], theory, a, rho)[:, 0]
    L = np.linalg.cholesky(M)
    Linv = np.linalg.inv(L)
    vals, y = np.linalg.eigh(Linv@K@np.swapaxes(Linv, -1, -2))
    vecs = np.swapaxes(Linv, -1, -2)@y
    I0 = rho[:, 0]
    participation = I0[:, None]*vecs[:, W, :]**2
    flex = np.argmax(participation, axis=1)
    if np.any(flex != 0) or np.any(vals[:, 0] <= 0):
        raise ValueError('flexural branch is not the lowest branch of the (1,1) pair')
    h = a/parameters[:, 1]
    return np.sqrt(vals[:, 0])*h*np.sqrt(RHO_C/EC)
