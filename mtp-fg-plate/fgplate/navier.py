"""
Navier (double sine series) solution of a simply supported rectangular FG plate
using First-order Shear Deformation Theory (FSDT / Mindlin plate theory).

Displacement field (FSDT):
    u(x,y,z) = u0(x,y) + z * phi_x(x,y)
    v(x,y,z) = v0(x,y) + z * phi_y(x,y)
    w(x,y,z) = w0(x,y)

For a plate simply supported on all four edges (SSSS), every unknown is
expanded in sine/cosine series that satisfy the boundary conditions exactly:

    u0    = sum U_mn cos(al x) sin(be y)
    v0    = sum V_mn sin(al x) cos(be y)
    w0    = sum W_mn sin(al x) sin(be y)
    phi_x = sum X_mn cos(al x) sin(be y)
    phi_y = sum Y_mn sin(al x) cos(be y)

    al = m*pi/a,  be = n*pi/b

Substituting into the equilibrium equations, every (m, n) term decouples into
a small 5x5 linear system:

    Static bending:   K_mn  [U V W X Y]^T = [0 0 Q_mn 0 0]^T
    Free vibration:   (K_mn - omega^2 M_mn) [U V W X Y]^T = 0

so the whole plate problem is solved with tiny matrices -- that is why the
Navier method runs in milliseconds.
"""

import numpy as np
from scipy.linalg import eigh, cholesky
from numpy.polynomial.legendre import leggauss
from collections.abc import Mapping
from dataclasses import dataclass

from .material import FGMaterial, PlateStiffness, plate_stiffness, positive, index


@dataclass(frozen=True)
class ModalEigenpairs:
    omega: np.ndarray
    vectors: np.ndarray
    transverse_participation: np.ndarray

    @property
    def flexural_index(self):
        return int(np.argmax(self.transverse_participation))


class NavierFSDTPlate:
    """Simply supported rectangular FG plate solved with the Navier method.

    Parameters
    ----------
    a, b : plate length (x) and width (y), in m
    h    : plate thickness, in m
    material : FGMaterial
    ks   : shear correction factor (5/6 is the standard FSDT value)
    """

    def __init__(self, a: float, b: float, h: float, material: FGMaterial, ks: float = 5.0 / 6.0):
        for name, value in [('a', a), ('b', b), ('h', h), ('ks', ks)]:
            positive(value, name)
        self.a, self.b, self.h = a, b, h
        self.material = material
        self.ks = ks
        self.S: PlateStiffness = plate_stiffness(material, h, ks)

    # ------------------------------------------------------------------ matrices
    def matrices(self, m: int, n: int):
        """Return the 5x5 stiffness K and mass M matrices for mode (m, n)."""
        index(m, 'm'); index(n, 'n')
        S = self.S
        al = m * np.pi / self.a
        be = n * np.pi / self.b

        K = np.zeros((5, 5))
        # in-plane (stretching) terms
        K[0, 0] = S.A11 * al**2 + S.A66 * be**2
        K[0, 1] = (S.A12 + S.A66) * al * be
        K[1, 1] = S.A66 * al**2 + S.A11 * be**2
        # stretching-bending coupling (zero for homogeneous plates)
        K[0, 3] = S.B11 * al**2 + S.B66 * be**2
        K[0, 4] = (S.B12 + S.B66) * al * be
        K[1, 3] = (S.B12 + S.B66) * al * be
        K[1, 4] = S.B66 * al**2 + S.B11 * be**2
        # transverse shear terms
        K[2, 2] = S.As * (al**2 + be**2)
        K[2, 3] = S.As * al
        K[2, 4] = S.As * be
        # bending terms
        K[3, 3] = S.D11 * al**2 + S.D66 * be**2 + S.As
        K[3, 4] = (S.D12 + S.D66) * al * be
        K[4, 4] = S.D66 * al**2 + S.D11 * be**2 + S.As
        K = K + np.triu(K, 1).T  # fill the symmetric lower triangle

        M = np.zeros((5, 5))
        M[0, 0] = M[1, 1] = M[2, 2] = S.I0
        M[3, 3] = M[4, 4] = S.I2
        M[0, 3] = M[3, 0] = S.I1
        M[1, 4] = M[4, 1] = S.I1
        return K, M

    # ------------------------------------------------------------------ bending
    @staticmethod
    def load_coefficient(q0: float, m: int, n: int, load: str) -> float:
        """Fourier coefficient Q_mn of the transverse load q(x, y)."""
        index(m, 'm'); index(n, 'n')
        if not np.isscalar(q0) or not np.isreal(q0) or not np.isfinite(q0):
            raise ValueError('q0 must be finite and real')
        if load == "sinusoidal":       # q = q0 sin(pi x/a) sin(pi y/b)
            return q0 if (m == 1 and n == 1) else 0.0
        if load == "uniform":          # q = q0 everywhere
            if m % 2 == 1 and n % 2 == 1:
                return 16.0 * q0 / (np.pi**2 * m * n)
            return 0.0
        raise ValueError(f"unknown load type: {load!r}")

    def bending(self, q0: float, load: str = "sinusoidal", n_terms: int = 49, *, method='matrix') -> "BendingSolution":
        """Solve static bending. Returns a BendingSolution with all series coefficients.

        n_terms: highest m and n kept in the series (only used for non-sinusoidal loads).
        """
        index(n_terms, 'n_terms')
        if load not in ('sinusoidal', 'uniform'):
            raise ValueError(f'unknown load type: {load!r}')
        self.load_coefficient(q0, 1, 1, load)
        n_max = 1 if load == "sinusoidal" else n_terms
        loads = {}
        for m in range(1, n_max + 1):
            for n in range(1, n_max + 1):
                Q = self.load_coefficient(q0, m, n, load)
                if Q == 0.0:
                    continue
                loads[(m, n)] = Q
        return self.bending_modal(loads, method=method)

    def project_load(self, pressure, max_mode=5, quadrature_order=64):
        """Project a callable pressure(x,y), in Pa, using Gauss quadrature.

        Returns Qmn=4/(ab)*integral(q*sin(alpha*x)*sin(beta*y)). The
        caller must check both quadrature and modal truncation convergence.
        """
        index(max_mode, 'max_mode'); index(quadrature_order, 'quadrature_order', 2)
        if not callable(pressure):
            raise TypeError('pressure must be callable')
        z, weights = leggauss(quadrature_order)
        t, weights = (z+1)/2, weights/2
        X, Y = np.meshgrid(self.a*t, self.b*t, indexing='ij')
        raw = np.asarray(pressure(X, Y))
        if np.iscomplexobj(raw):
            raise ValueError('pressure must return real values')
        q = np.broadcast_to(np.asarray(raw, dtype=float), X.shape)
        if not np.all(np.isfinite(q)):
            raise ValueError('pressure returned nonfinite values')
        basis = np.sin(np.pi*t[:, None]*np.arange(1, max_mode+1))
        values = 4*basis.T@(q*weights[:, None]*weights[None, :])@basis
        return {(m+1,n+1): float(values[m,n]) for m in range(max_mode) for n in range(max_mode)}

    def bending_modal(self, loads, *, method='matrix'):
        """Solve a finite signed transverse load series {(m,n): Qmn in Pa}.

        'closed_form' is valid for this constant-nu, through-thickness isotropic
        material and these Navier supports; it returns all five fields.
        """
        if not isinstance(loads, Mapping):
            raise TypeError('loads must map (m,n) to a real pressure coefficient')
        if method not in ('matrix','closed_form'):
            raise ValueError('method must be matrix or closed_form')
        clean = {}
        for pair, q in loads.items():
            if not isinstance(pair, tuple) or len(pair) != 2:
                raise ValueError('load keys must be (m,n) tuples')
            index(pair[0], 'm'); index(pair[1], 'n')
            if not np.isscalar(q) or not np.isreal(q) or not np.isfinite(q):
                raise ValueError('load coefficients must be finite real scalars')
            if q != 0:
                clean[pair] = float(q)
        clean = dict(sorted(clean.items()))
        coeffs = {}
        if method == 'matrix':
            for (m,n), q in clean.items():
                K, _ = self.matrices(m,n)
                coeffs[m,n] = np.linalg.solve(K, [0,0,q,0,0])
        elif clean:
            modes = np.array(list(clean))
            alpha, beta = modes[:,0]*np.pi/self.a, modes[:,1]*np.pi/self.b
            k2 = alpha**2+beta**2
            S = self.S
            dstar = S.D11-S.B11**2/S.A11
            if not np.isfinite(dstar) or dstar <= 0:
                raise ValueError('nonpositive or nonfinite condensed bending stiffness')
            q = np.array(list(clean.values()))
            wb = q/(dstar*k2**2)
            px, py = -alpha*wb, -beta*wb
            values = np.column_stack([-S.B11/S.A11*px, -S.B11/S.A11*py,
                                      wb+q/(S.As*k2), px, py])
            coeffs = dict(zip(clean, values))
        return BendingSolution(self, coeffs)

    def eigenpairs(self, m=1, n=1):
        """Symmetric generalized solve, M-normalized columns, frequencies rad/s.

        Transverse participation is I0*W^2/(c.T M c). It identifies the
        transverse-dominated branch here; it is not mode tracking across
        crossings in a future generalized constitutive model.
        """
        K, M = self.matrices(m,n)
        for name, mat in [('K',K),('M',M)]:
            if not np.all(np.isfinite(mat)) or not np.allclose(mat,mat.T,rtol=1e-13,atol=0):
                raise ValueError(f'{name} must be finite and symmetric')
            cholesky(mat, lower=True)
        vals, vecs = eigh(K, M, check_finite=True)
        if np.any(vals <= 0):
            raise ValueError('nonpositive eigenvalue; inspect conditioning and model')
        # Stable sign convention; degenerate modes still require subspace comparison.
        for j in range(5):
            largest = int(np.argmax(abs(vecs[:,j])))
            if vecs[largest,j] < 0: vecs[:,j] *= -1
        return ModalEigenpairs(np.sqrt(vals), vecs, self.S.I0*vecs[2,:]**2)

    # ------------------------------------------------------------------ vibration
    def frequencies(self, m: int = 1, n: int = 1) -> np.ndarray:
        """All 5 natural frequencies omega (rad/s) for half-wave numbers (m, n), sorted.

        These are branches of ONE half-wave pair, not five global bending modes.
        """
        return self.eigenpairs(m,n).omega

    def fundamental_frequency(self) -> float:
        """Fundamental (lowest) natural frequency omega_11 in rad/s."""
        return float(self.frequencies(1, 1)[0])

    def flexural_modes(self, max_mode=5, count=5):
        """Sorted transverse-dominated branches over a bounded half-wave scan.

        Increase max_mode and compare to establish global-mode convergence.
        """
        index(max_mode, 'max_mode'); index(count, 'count')
        if count > max_mode**2: raise ValueError('count exceeds scanned mode pairs')
        rows = []
        for m in range(1,max_mode+1):
            for n in range(1,max_mode+1):
                eig = self.eigenpairs(m,n); j = eig.flexural_index
                rows.append(dict(m=m,n=n,omega=float(eig.omega[j]),
                                 participation=float(eig.transverse_participation[j]),
                                 vector=eig.vectors[:,j].copy()))
        return sorted(rows,key=lambda row: row['omega'])[:count]


class BendingSolution:
    """Series solution of a bending problem; evaluate fields anywhere on the plate."""

    def __init__(self, plate: NavierFSDTPlate, coeffs: dict):
        self.plate = plate
        self.coeffs = coeffs  # {(m, n): array [U, V, W, X, Y]}

    def deflection(self, x, y):
        """Transverse deflection w0(x, y). x and y may be scalars or arrays."""
        return self.fields(x,y)['w0']

    def fields(self, x, y, derivative=(0,0)):
        """All five fields or analytical spatial derivatives at broadcast points.

        Displacements in metres; rotations dimensionless; derivatives in SI.
        """
        dx, dy = derivative
        index(dx,'dx',0); index(dy,'dy',0)
        x,y = np.broadcast_arrays(np.asarray(x,dtype=float),np.asarray(y,dtype=float))
        if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
            raise ValueError('coordinates must be finite')
        tol=1e-12*max(self.plate.a,self.plate.b)
        if np.any(x < -tol) or np.any(x > self.plate.a+tol) or np.any(y < -tol) or np.any(y > self.plate.b+tol):
            raise ValueError('coordinates outside plate')
        names=('u0','v0','w0','phix','phiy')
        values=np.zeros((5,x.size))
        if self.coeffs:
            modes=np.array(list(self.coeffs)); C=np.array(list(self.coeffs.values()))
            alpha=modes[:,0,None]*np.pi/self.plate.a
            beta=modes[:,1,None]*np.pi/self.plate.b
            for start in range(0,x.size,256):
                sl=slice(start,start+256)
                sx=np.sin(alpha*x.ravel()[sl]+dx*np.pi/2)*alpha**dx
                cx=np.cos(alpha*x.ravel()[sl]+dx*np.pi/2)*alpha**dx
                sy=np.sin(beta*y.ravel()[sl]+dy*np.pi/2)*beta**dy
                cy=np.cos(beta*y.ravel()[sl]+dy*np.pi/2)*beta**dy
                for j, basis in enumerate((cx*sy,sx*cy,sx*sy,cx*sy,sx*cy)):
                    values[j,sl]=C[:,j]@basis
        return {name: values[j].reshape(x.shape) for j,name in enumerate(names)}

    def field_grid(self, nx=41, ny=41):
        """Return X,Y and field dictionary, each array shape (ny,nx)."""
        index(nx,'nx',2); index(ny,'ny',2)
        X,Y=np.meshgrid(np.linspace(0,self.plate.a,nx),np.linspace(0,self.plate.b,ny))
        # Group coefficients into rectangular arrays for separable evaluation.
        out={name:np.zeros((ny,nx)) for name in ('u0','v0','w0','phix','phiy')}
        if self.coeffs:
            mx=max(m for m,n in self.coeffs); my=max(n for m,n in self.coeffs)
            ax=np.arange(1,mx+1)[:,None]*np.pi*X[0,:]/self.plate.a
            by=np.arange(1,my+1)[:,None]*np.pi*Y[:,0]/self.plate.b
            sx,cx,sy,cy=np.sin(ax),np.cos(ax),np.sin(by),np.cos(by)
            for j,(name,bx,by_) in enumerate(zip(out,(cx,sx,sx,cx,sx),(sy,cy,sy,sy,cy))):
                c=np.zeros((my,mx))
                for (m,n),v in self.coeffs.items(): c[n-1,m-1]=v[j]
                out[name]=by_.T@c@bx
        return X,Y,out

    def center_deflection(self) -> float:
        return float(self.deflection(self.plate.a / 2, self.plate.b / 2))

    def deflection_grid(self, nx: int = 41, ny: int = 41):
        """Deflection on a regular nx-by-ny grid. Returns (X, Y, W)."""
        X,Y,fields=self.field_grid(nx,ny)
        return X,Y,fields['w0']
