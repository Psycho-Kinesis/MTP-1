"""
Independent cross-check of the Navier solver.

Instead of using the hand-derived 5x5 matrices in fgplate/navier.py, this
script builds the plate problem from scratch by ENERGY MINIMISATION (Ritz):

  1. write the 3D displacement field u, v, w for a plate theory,
  2. let sympy compute the strains and the strain/kinetic energy symbolically,
  3. integrate through the thickness numerically (works for any E(z), rho(z)),
  4. stiffness K = Hessian of strain energy, mass M = Hessian of kinetic energy.

Using the same sine/cosine functions as the Navier solution, the Ritz solution
is exact, so for FSDT it must reproduce fgplate.navier to machine precision.

It also implements Reddy's Third-order Shear Deformation Theory (TSDT), to show
how much the plate theory itself changes the answer.
"""

import sys
from pathlib import Path

import numpy as np
import sympy as sp
from scipy.integrate import quad

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fgplate import al_al2o3  # noqa: E402

x, y, z = sp.symbols("x y z", real=True)
U, V, W, X, Y = coeffs = sp.symbols("U V W X Y", real=True)


def _z_moments(f, h, kmax):
    """Integrals of f(z) * z^k over the thickness, k = 0..kmax.

    Uses adaptive quadrature (scipy) -- deliberately a different method from the
    closed-form moments in fgplate.material, so the cross-check stays independent.
    """
    return [quad(lambda zz: f(zz) * zz**k, -h / 2, h / 2, epsabs=0, epsrel=1e-13, limit=200)[0]
            for k in range(kmax + 1)]


def _integrate_over_plate(expr_z_poly, a, b, moments):
    """Integrate sum_k c_k(x,y) z^k * f(z) over the plate volume."""
    poly = sp.Poly(sp.expand(expr_z_poly), z)
    total = 0
    for (k,), c in poly.terms():
        cxy = sp.integrate(sp.integrate(c, (x, 0, a)), (y, 0, b))
        total += cxy * moments[k]
    return total


def ritz_matrices(theory, mat, a, b, h, m=1, n=1, ks=5.0 / 6.0):
    al, be = m * sp.pi / a, n * sp.pi / b
    u0 = U * sp.cos(al * x) * sp.sin(be * y)
    v0 = V * sp.sin(al * x) * sp.cos(be * y)
    w0 = W * sp.sin(al * x) * sp.sin(be * y)
    px = X * sp.cos(al * x) * sp.sin(be * y)
    py = Y * sp.sin(al * x) * sp.cos(be * y)

    if theory == "FSDT":
        u = u0 + z * px
        v = v0 + z * py
        k_shear = ks
    elif theory == "TSDT":  # Reddy (1984): parabolic shear strain, no correction factor
        c1 = sp.Rational(4, 3) / h**2
        u = u0 + z * px - c1 * z**3 * (px + sp.diff(w0, x))
        v = v0 + z * py - c1 * z**3 * (py + sp.diff(w0, y))
        k_shear = 1.0
    else:
        raise ValueError(theory)
    w = w0

    ex, ey = sp.diff(u, x), sp.diff(v, y)
    gxy = sp.diff(u, y) + sp.diff(v, x)
    gxz = sp.diff(u, z) + sp.diff(w, x)
    gyz = sp.diff(v, z) + sp.diff(w, y)

    nu = mat.nu
    # strain energy density divided by E(z)
    dens = (ex**2 + ey**2 + 2 * nu * ex * ey) / (1 - nu**2) \
        + (gxy**2 + k_shear * (gxz**2 + gyz**2)) / (2 * (1 + nu))
    # kinetic energy density divided by rho(z) (velocities -> amplitudes)
    kin = u**2 + v**2 + w**2

    Emom = _z_moments(lambda zz: mat.E(zz, h), h, 8)
    Rmom = _z_moments(lambda zz: mat.rho(zz, h), h, 8)
    strain_energy = sp.Rational(1, 2) * _integrate_over_plate(dens, a, b, Emom)
    kinetic = sp.Rational(1, 2) * _integrate_over_plate(kin, a, b, Rmom)

    K = np.array(sp.hessian(strain_energy, coeffs), dtype=float)
    M = np.array(sp.hessian(kinetic, coeffs), dtype=float)
    # load vector for q = q0 sin(pi x/a) sin(pi y/b), q0 = 1
    F = np.array([sp.diff(sp.integrate(sp.integrate(
        w0 * sp.sin(sp.pi * x / a) * sp.sin(sp.pi * y / b), (x, 0, a)), (y, 0, b)), c)
        for c in coeffs], dtype=float)
    return K, M, F


def center_deflection(theory, mat, a, b, h):
    K, _, F = ritz_matrices(theory, mat, a, b, h)
    sol = np.linalg.solve(K, F)
    return sol[2]  # W amplitude = centre deflection for m = n = 1


def fundamental_frequency(theory, mat, a, b, h):
    K, M, _ = ritz_matrices(theory, mat, a, b, h)
    return np.sqrt(np.min(np.real(np.linalg.eigvals(np.linalg.solve(M, K)))))


if __name__ == "__main__":
    from fgplate import NavierFSDTPlate

    a = b = 1.0
    h = 0.1
    print("Al/Al2O3, a/h = 10, sinusoidal load, wbar = 10 Ec h^3 w / (q0 a^4)\n")
    print(f"{'n':>4} | {'Navier FSDT':>11} | {'Ritz FSDT':>10} | {'Ritz TSDT':>10}")
    for n in [0, 1, 2, 4, 10]:
        mat = al_al2o3(n)
        scale = 10 * mat.E_c * h**3 / a**4
        w_nav = NavierFSDTPlate(a, b, h, mat).bending(1.0).center_deflection() * scale
        w_fsdt = center_deflection("FSDT", mat, a, b, h) * scale
        w_tsdt = center_deflection("TSDT", mat, a, b, h) * scale
        print(f"{n:>4} | {w_nav:11.4f} | {w_fsdt:10.4f} | {w_tsdt:10.4f}")
