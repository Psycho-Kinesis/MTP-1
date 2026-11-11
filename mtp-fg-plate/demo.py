"""
Demo: what the Navier FG-plate solver computes, and how fast it is.

Run:  python demo.py      -> prints tables and saves figures into figures/
"""

import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from fgplate import NavierFSDTPlate, al_al2o3  # noqa: E402

FIG = Path(__file__).parent / "figures"
FIG.mkdir(exist_ok=True)
a = 1.0  # plate length (m); all results below are non-dimensional


def wbar(plate, w):
    return 10 * plate.material.E_c * plate.h**3 * w / plate.a**4


def omega_bar(plate, omega):
    m = plate.material
    return omega * plate.h * np.sqrt(m.rho_c / m.E_c)


# ---------------------------------------------------------------- 1. material
fig, ax = plt.subplots(figsize=(5, 4))
z = np.linspace(-0.5, 0.5, 200)
for n in [0, 0.5, 1, 2, 5, 10]:
    ax.plot(al_al2o3(n).E(z, 1.0) / 1e9, z, label=f"n = {n}")
ax.set_xlabel("Young's modulus E(z)  [GPa]")
ax.set_ylabel("thickness position z / h")
ax.set_title("Al/Al$_2$O$_3$: metal (bottom) → ceramic (top)")
ax.legend()
fig.tight_layout()
fig.savefig(FIG / "1_material_profile.png", dpi=150)

# ---------------------------------------------------------------- 2. deflection shape
plate = NavierFSDTPlate(a, a, a / 10, al_al2o3(2))
sol = plate.bending(1.0, load="uniform")
X, Y, Wf = sol.deflection_grid(61, 61)
fig = plt.figure(figsize=(9, 4))
ax1 = fig.add_subplot(1, 2, 1, projection="3d")
ax1.plot_surface(X, Y, wbar(plate, Wf), cmap="viridis")
ax1.set_title("Deflected shape (uniform load)")
ax1.set_xlabel("x/a")
ax1.set_ylabel("y/b")
ax2 = fig.add_subplot(1, 2, 2)
c = ax2.contourf(X, Y, wbar(plate, Wf), 20, cmap="viridis")
fig.colorbar(c, ax=ax2, label="$\\bar w$")
ax2.set_aspect("equal")
ax2.set_title("FG plate, n = 2, a/h = 10")
fig.tight_layout()
fig.savefig(FIG / "2_deflection_shape.png", dpi=150)

# ---------------------------------------------------------------- 3. parametric study
ns = np.linspace(0, 10, 41)
fig, (axw, axf) = plt.subplots(1, 2, figsize=(10, 4))
print("\nNon-dimensional results, square Al/Al2O3 plate, sinusoidal load")
print(f"{'a/h':>5} {'n':>5} {'wbar':>8} {'omega_bar':>10}")
for ratio in [5, 10, 20, 100]:
    w_list, f_list = [], []
    for n in ns:
        p = NavierFSDTPlate(a, a, a / ratio, al_al2o3(n))
        w_list.append(wbar(p, p.bending(1.0).center_deflection()))
        f_list.append(omega_bar(p, p.fundamental_frequency()))
    axw.plot(ns, w_list, label=f"a/h = {ratio}")
    axf.plot(ns, np.array(f_list) * ratio**2, label=f"a/h = {ratio}")
    for n in [0, 1, 5, 10]:
        p = NavierFSDTPlate(a, a, a / ratio, al_al2o3(n))
        print(f"{ratio:>5} {n:>5} {wbar(p, p.bending(1.0).center_deflection()):8.4f} "
              f"{omega_bar(p, p.fundamental_frequency()):10.5f}")
axw.set_xlabel("gradient index n")
axw.set_ylabel("$\\bar w = 10 E_c h^3 w / (q_0 a^4)$")
axw.set_title("Centre deflection")
axw.legend()
axf.set_xlabel("gradient index n")
axf.set_ylabel("$\\omega a^2/h \\, \\sqrt{\\rho_c/E_c}$")
axf.set_title("Fundamental frequency")
axf.legend()
fig.tight_layout()
fig.savefig(FIG / "3_parametric_study.png", dpi=150)

# ---------------------------------------------------------------- 4. speed
rng = np.random.default_rng(0)
N = 2000
t0 = time.perf_counter()
for _ in range(N):
    p = NavierFSDTPlate(a, a * rng.uniform(0.5, 2), a / rng.uniform(5, 100), al_al2o3(rng.uniform(0, 10)))
    p.bending(1.0).center_deflection()
    p.fundamental_frequency()
dt = (time.perf_counter() - t0) / N
print(f"\nSpeed: {dt * 1e3:.3f} ms per design (bending + frequency)"
      f"  ->  {1 / dt:,.0f} designs per second")
print(f"Figures saved in {FIG}/")
