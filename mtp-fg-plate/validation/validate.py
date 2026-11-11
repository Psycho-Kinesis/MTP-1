"""
Validation suite for the Navier FSDT solver.

Run:   python validation/validate.py          (fast checks, < 1 s)
       python validation/validate.py --full   (also the symbolic Ritz cross-check, ~30 s)

Every check prints PASS/FAIL with the computed value, the reference value and the error.
"""

import sys
import json
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fgplate import NavierFSDTPlate, al_al2o3, isotropic, plate_stiffness  # noqa: E402

results = []
references = json.loads((Path(__file__).with_name('benchmark_references.json')).read_text(encoding='utf-8'))


def check(name, value, reference, rel_tol):
    err = abs(value - reference) / abs(reference)
    ok = err <= rel_tol
    results.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name:<58} got {value:.6g}  ref {reference:.6g}  err {err:.2e}")


def wbar(plate, w):
    """Standard FG benchmark normalisation: 10 Ec h^3 w / (q0 a^4), with q0 = 1."""
    return 10 * plate.material.E_c * plate.h**3 * w / plate.a**4


print("\n=== 1. Isotropic plates vs classical closed-form results ===")
E, rho, nu = 70e9, 2702.0, 0.3
a = 1.0
h = 0.001  # very thin -> FSDT must reduce to classical plate theory
D = E * h**3 / (12 * (1 - nu**2))
thin = NavierFSDTPlate(a, a, h, isotropic(E, rho))
odd = np.arange(1, 1000, 2)
mm, nn = np.meshgrid(odd, odd)
cpt_uniform = 16/np.pi**6*np.sum(np.sin(mm*np.pi/2)*np.sin(nn*np.pi/2)/(mm*nn*(mm.astype(float)**2+nn**2)**2))
check("thin plate, uniform load, w_max*D/(q a^4) (CPT series)",
      thin.bending(1.0, "uniform", n_terms=99).center_deflection() * D / a**4, cpt_uniform, 1e-4)
check("thin plate, sinusoidal load, w_max (exact CPT)",
      thin.bending(1.0).center_deflection(), a**4 / (4 * np.pi**4 * D), 1e-4)
check("thin plate, fundamental frequency (exact CPT)",
      thin.fundamental_frequency(), 2 * np.pi**2 / a**2 * np.sqrt(D / (rho * h)), 1e-4)

h = 0.1  # thick plate, a/h = 10: shear deformation matters
G = E / (2 * (1 + nu))
thick = NavierFSDTPlate(a, a, h, isotropic(E, rho))
check("thick plate a/h=10, omega*h*sqrt(rho/G) (Mindlin FSDT 0.0930)",
      thick.fundamental_frequency() * h * np.sqrt(rho / G), references['isotropic_frequency']['fsdt_value'], 5e-3)
check("thick plate vs 3D 0.0932 (different-model context)",
      thick.fundamental_frequency() * h * np.sqrt(rho / G), references['isotropic_frequency']['elasticity_value'], 5e-3)

print("\n=== 2. FG plates (Al/Al2O3) vs published benchmarks, a/h=10, sinusoidal load ===")
# Reproduced SSDT values: Nguyen et al. (2019), Table 4, citing Zenkour.
# These are different-theory comparisons, even where the discrepancy is small.
fg_refs = references['fg_sine_deflection']['values']
for n, ref in [(0, fg_refs['0']), (1, fg_refs['1'])]:
    p = NavierFSDTPlate(a, a, h, al_al2o3(n))
    check(f"wbar, n = {n}", wbar(p, p.bending(1.0).center_deflection()), ref, 5e-4)
metal_w = NavierFSDTPlate(a, a, h, isotropic(70e9, 2702.0)).bending(1.0).center_deflection()
check("wbar, pure metal (normalised with ceramic Ec)", 10 * 380e9 * h**3 * metal_w / a**4, fg_refs['metal'], 5e-4)

print("\n=== 3. FG plates vs exact neutral-surface closed form (checks B-coupling terms) ===")
# For constant nu, FSDT bending of an FG plate has the closed form
#     W = Q/(D* k^4) + Q/(As k^2),  D* = D11 - B11^2/A11,  k^2 = al^2 + be^2
for n in [0.5, 2, 4, 10]:
    for aspect, ratio in [(1.0, 5), (1.5, 20)]:
        b, hh = a * aspect, a / ratio
        p = NavierFSDTPlate(a, b, hh, al_al2o3(n))
        S = plate_stiffness(al_al2o3(n), hh)
        k2 = (np.pi / a) ** 2 + (np.pi / b) ** 2
        exact = 1.0 / ((S.D11 - S.B11**2 / S.A11) * k2**2) + 1.0 / (S.As * k2)
        check(f"n={n}, b/a={aspect}, a/h={ratio}", p.bending(1.0).center_deflection(), exact, 1e-10)

print("\n=== 4. Series convergence for uniform load (FG, n=2, a/h=10) ===")
p = NavierFSDTPlate(a, a, h, al_al2o3(2))
ref = p.bending(1.0, "uniform", n_terms=199).center_deflection()
for terms in [9, 19, 49]:
    check(f"centre deflection with m,n <= {terms}",
          p.bending(1.0, "uniform", n_terms=terms).center_deflection(), ref, 1e-3)

if "--full" in sys.argv:
    print("\n=== 5. Independent symbolic Ritz (energy) solution, FSDT ===")
    from ritz_crosscheck import center_deflection, fundamental_frequency
    for n in [0.5, 4]:
        mat = al_al2o3(n)
        p = NavierFSDTPlate(a, 1.5 * a, h, mat)
        check(f"deflection, n={n}, b/a=1.5", p.bending(1.0).center_deflection(),
              center_deflection("FSDT", mat, a, 1.5 * a, h), 1e-9)
        check(f"frequency,  n={n}, b/a=1.5", p.fundamental_frequency(),
              fundamental_frequency("FSDT", mat, a, 1.5 * a, h), 1e-9)

    print("\n=== 6. TSDT Navier stiffness (fgplate.tsdt) vs independent symbolic Ritz TSDT ===")
    from fgplate.tsdt import stiffness
    for n, ratio, aspect in [(2, 10, 1.0), (6, 5, 1.5)]:
        mat = al_al2o3(n)
        Kt = stiffness(mat, a/ratio, a, aspect*a, 1, 1, 'TSDT')
        w_navier = np.linalg.solve(Kt, [0, 0, 1, 0, 0])[2]
        check(f"TSDT centre deflection, n={n}, a/h={ratio}, b/a={aspect}", w_navier,
              center_deflection("TSDT", mat, a, aspect*a, a/ratio), 1e-9)

    print("\n=== 7. TSDT Navier frequency (fgplate.tsdt) vs independent symbolic Ritz TSDT ===")
    from fgplate.tsdt import eigenpairs
    for n, ratio, aspect in [(2, 10, 1.0), (6, 5, 1.5)]:
        mat = al_al2o3(n)
        check(f"TSDT fundamental frequency, n={n}, a/h={ratio}, b/a={aspect}",
              eigenpairs(mat, a/ratio, a, aspect*a, 1, 1, 'TSDT')[0][0],
              fundamental_frequency("TSDT", mat, a, aspect*a, a/ratio), 1e-8)

print(f"\n{sum(results)}/{len(results)} checks passed")
sys.exit(0 if all(results) else 1)
