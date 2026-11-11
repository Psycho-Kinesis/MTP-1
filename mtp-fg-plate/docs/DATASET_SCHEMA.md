# Pilot data dictionary

Read results/pilot_200/dataset.npz using numpy.load(path, allow_pickle=False). pilot_200 was made with generator pilot-1.0 (design_id hashes the raw floats; mixture loads are seeded from design_id). New datasets use pilot-1.1: parameters quantized to 10 significant digits, design_id from canonical decimal strings, mixture loads seeded by (seed, sample index). In both cases the stored arrays are authoritative; see validation/check_artifacts.py. Arrays are float64 unless stated. All field cases are indexed [design, load, ...]; load order is sine, uniform, smooth_mixture. Field order is u0,v0,w0,phix,phiy. All output grids are [eta,xi], with x=a*xi and y=b*eta. Displacements are metres; rotations are dimensionless. The single scalar input qscale=1 Pa is fixed and recorded in manifest.json.

| Array | Shape in delivered pilot | Meaning |
| --- | --- | --- |
| parameters | (200,3) | g, a/h, b/a; a=1 m |
| design_ids | (200,) | Permanent hash of full physical inputs and support identifier |
| split | (200,) | train / validation / test |
| mode_indices | (641,2), integer | m,n sine/cosine half-wave indices |
| pressure_coefficients | (200,3,641) | Qmn in Pa; zeros for inactive modes |
| solution_coefficients | (200,3,641,5) | U,V,W,X,Y; metres, metres, metres, dimensionless, dimensionless |
| fields | (200,3,5,41,41) | Five physical fields, not normalized |
| omega_rad_s | (200,5) | Sorted first five transverse-dominated branches from a 6x6 scan |
| frequency_mode_indices | (200,5,2), integer | m,n for each retained frequency |
| transverse_participation | (200,5) | I0*W^2 for mass-normalized modal vectors |
| scalar_targets | (200,2) | Sine centre wbar and fundamental omegabar |
| mixture_raw_norm | (200,) | Euclidean norm of z_mn/(m^2+n^2) before normalizing mixture coefficients |
| convergence | (200,7) | Five relative field norms (49 vs99), centre w difference, frequency scan difference |
| xi, eta | (41,) each | Normalized grid coordinates in [0,1] |

The case identifier is the pair (design_id, load family). Geometry is recovered as b=parameters[:,2], h=1/parameters[:,1]. The fixed materials, supports, seed and versions are in manifest.json. The split JSON duplicates the identity partitions for easy auditing. Scalar CSV is a convenience view, not the full dataset.

For qscale != 1 Pa, multiply static coefficients and fields by qscale/(1 Pa); do not scale natural frequencies. For a negative overall multiplier, maximum absolute deflection remains the correct scalar for a general load. The modal mixture itself is signed. Uniform Qmn uses 16/(pi^2*m*n) on odd indices through49, not just49 total modes.

Field bases are U cos(alpha*x)sin(beta*y), V sin(alpha*x)cos(beta*y), W sin(alpha*x)sin(beta*y), X cos(alpha*x)sin(beta*y), Y sin(alpha*x)cos(beta*y). This gives the mixed simply-supported boundary conditions specified in the TID; it is not a fully restrained in-plane boundary.

## Loading the data and the saved scalar model

Run from the project root:

```python
import numpy as np
from fgplate.baseline import ScalarBaseline

with np.load('results/pilot_200/dataset.npz', allow_pickle=False) as data:
    first_mixture_w = data['fields'][0, 2, 2]  # metres, shape (41,41)
    first_design = data['parameters'][0].copy()

model = ScalarBaseline.load('results/pilot_200/scalar_model.npz')
wbar, omega_bar = model.predict([[2.0, 10.0, 1.0]])[0]
a, h, qscale, Ec, rhoc = 1.0, 0.1, 1.0, 380e9, 3800.0
w_centre_m = wbar*qscale*a**4/(10*Ec*h**3)
omega_rad_s = omega_bar*h/a**2*np.sqrt(Ec/rhoc)
frequency_Hz = omega_rad_s/(2*np.pi)
```

The saved model has no arbitrary-load inputs. Only use it inside the sampled parameter domain until extrapolation has been evaluated. Model arrays use no Python pickle. checksums.json covers generated artifacts (excluding itself), and manifest.json records the exact generator source hashes.
