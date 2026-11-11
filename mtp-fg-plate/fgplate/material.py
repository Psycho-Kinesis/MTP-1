"""
Functionally graded (FG) material model and plate stiffness integrals.

The plate is made of two materials: ceramic (top) and metal (bottom).
The ceramic volume fraction varies through the thickness z in [-h/2, h/2]
with the classical power law

    Vc(z) = (z/h + 1/2) ** n        (n = gradient index)

    n = 0      -> Vc = 1 everywhere   -> fully ceramic plate
    n -> inf   -> Vc -> 0             -> fully metal plate

Effective properties use the rule of mixtures (Voigt model):

    P(z) = (Pc - Pm) * Vc(z) + Pm      (P = Young's modulus E or density rho)

Poisson's ratio nu is assumed constant through the thickness (standard
assumption in FG plate benchmarks).
"""

from dataclasses import dataclass
from math import comb
from numbers import Real, Integral

import numpy as np


def positive(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive finite real number")


def index(value, name, minimum=1):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


@dataclass(frozen=True)
class FGMaterial:
    """Two-phase ceramic/metal functionally graded material."""

    E_c: float      # Young's modulus of ceramic (Pa)
    E_m: float      # Young's modulus of metal (Pa)
    rho_c: float    # density of ceramic (kg/m^3)
    rho_m: float    # density of metal (kg/m^3)
    nu: float = 0.3
    n: float = 1.0  # gradient index

    def __post_init__(self):
        for name in ('E_c', 'E_m', 'rho_c', 'rho_m'):
            positive(getattr(self, name), name)
        if isinstance(self.n, (bool, np.bool_)) or not isinstance(self.n, Real) or not np.isfinite(self.n) or self.n < 0:
            raise ValueError('gradation n must be finite and nonnegative')
        if isinstance(self.nu, (bool, np.bool_)) or not isinstance(self.nu, Real) or not np.isfinite(self.nu) or not -1 < self.nu < .5:
            raise ValueError('isotropic Poisson ratio must satisfy -1 < nu < 0.5')

    def ceramic_fraction(self, z, h):
        """Ceramic volume fraction Vc at thickness coordinate z."""
        positive(h, 'h')
        z = np.asarray(z, dtype=float)
        if not np.all(np.isfinite(z)) or np.any(np.abs(z) > h/2 + 8*np.finfo(float).eps*h):
            raise ValueError('z must lie inside the plate thickness')
        zeta = np.clip(z / h + 0.5, 0.0, 1.0)
        if self.n == 0:
            return np.ones_like(zeta)
        return zeta ** self.n

    def E(self, z, h):
        """Young's modulus at thickness coordinate z."""
        return (self.E_c - self.E_m) * self.ceramic_fraction(z, h) + self.E_m

    def rho(self, z, h):
        """Density at thickness coordinate z."""
        return (self.rho_c - self.rho_m) * self.ceramic_fraction(z, h) + self.rho_m

    def moment(self, prop: str, k: int, h: float) -> float:
        """Exact thickness integral  int_{-h/2}^{h/2} P(z) z^k dz  for P = 'E' or 'rho'.

        With zeta = z/h + 1/2 in [0, 1], z = h (zeta - 1/2), the power law gives
            int P z^k dz = h^(k+1) * sum_j C(k,j) (-1/2)^(k-j) [ (Pc-Pm)/(n+j+1) + Pm/(j+1) ]
        which is exact for any n >= 0 (no quadrature error, even when n < 1).
        """
        if prop not in ('E', 'rho'):
            raise ValueError("prop must be 'E' or 'rho'")
        index(k, 'moment order', 0)
        positive(h, 'h')
        Pc, Pm = (self.E_c, self.E_m) if prop == "E" else (self.rho_c, self.rho_m)
        # Stable forms avoid subtracting the homogeneous terms in odd moments.
        d, n = Pc-Pm, self.n
        if k == 0:
            return h*(Pm+d/(n+1))
        if k == 1:
            return h**2*d/(n+1)*(n/(n+2))/2
        if k == 2:
            return h**3*(Pm/12+d*(.25/(n+1)-1/(n+2)/(n+3)))
        total = 0.0
        for j in range(k + 1):
            total += comb(k, j) * (-0.5) ** (k - j) * ((Pc - Pm) / (self.n + j + 1) + Pm / (j + 1))
        return h ** (k + 1) * total


def al_al2o3(n):
    """Aluminium / Alumina FGM used in most FG plate benchmark papers."""
    return FGMaterial(E_c=380e9, E_m=70e9, rho_c=3800.0, rho_m=2702.0, nu=0.3, n=n)


def isotropic(E, rho, nu=0.3):
    """A homogeneous (non-graded) plate, written as an FGM with equal phases."""
    return FGMaterial(E_c=E, E_m=E, rho_c=rho, rho_m=rho, nu=nu, n=0.0)


@dataclass(frozen=True)
class PlateStiffness:
    """Stiffness and inertia resultants of an FSDT plate.

    A: stretching stiffness      (N/m)
    B: stretching-bending coupling (N)   -- nonzero only for FG plates
    D: bending stiffness         (N*m)
    As: transverse shear stiffness (N/m), includes shear correction factor
    I0, I1, I2: mass inertias  (integrals of rho * 1, z, z^2)
    """

    A11: float
    A12: float
    A66: float
    B11: float
    B12: float
    B66: float
    D11: float
    D12: float
    D66: float
    As: float
    I0: float
    I1: float
    I2: float


def plate_stiffness(mat: FGMaterial, h: float, ks: float = 5.0 / 6.0) -> PlateStiffness:
    """Compute A, B, D, shear stiffness and inertias by integrating through thickness.

    With constant nu, the plane-stress stiffnesses are
        Q11 = E/(1-nu^2),  Q12 = nu*E/(1-nu^2),  Q66 = Q44 = E/(2(1+nu))
    so every resultant is a scalar factor times the integrals
        E0 = int E dz,  E1 = int E z dz,  E2 = int E z^2 dz.
    """
    if not isinstance(mat, FGMaterial):
        raise TypeError('material must be an FGMaterial')
    positive(h, 'h')
    positive(ks, 'ks')
    nu = mat.nu
    E0, E1, E2 = (mat.moment("E", k, h) for k in range(3))

    c11 = 1.0 / (1.0 - nu**2)
    c12 = nu / (1.0 - nu**2)
    c66 = 1.0 / (2.0 * (1.0 + nu))

    return PlateStiffness(
        A11=c11 * E0, A12=c12 * E0, A66=c66 * E0,
        B11=c11 * E1, B12=c12 * E1, B66=c66 * E1,
        D11=c11 * E2, D12=c12 * E2, D66=c66 * E2,
        As=ks * c66 * E0,
        I0=mat.moment("rho", 0, h),
        I1=mat.moment("rho", 1, h),
        I2=mat.moment("rho", 2, h),
    )
