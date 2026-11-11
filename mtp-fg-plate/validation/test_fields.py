"""Checks for field scaling, analytic/POD bases and bootstrap intervals; run with unittest."""
import sys
from pathlib import Path
import unittest
import numpy as np
from numpy.testing import assert_allclose
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fgplate import NavierFSDTPlate, al_al2o3
from fgplate.baseline import bootstrap_ci
from fgplate.fields import (POD, analytic_basis, field_scales, project_analytic, reconstruct_analytic,
                            relative_l2, scale_fields, unscale_fields)
from fgplate.pilot import FIELDS


class FieldTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(5)
        self.modes = np.array([(m, n) for m in range(1, 6) for n in range(1, 6)])
        self.loads = {tuple(map(int, k)): float(v) for k, v in zip(self.modes, rng.normal(size=25))}
        self.grid = np.linspace(0, 1, 41)

    def solve(self, g, ratio, aspect):
        plate = NavierFSDTPlate(1.0, aspect, 1/ratio, al_al2o3(g))
        sol = plate.bending_modal(self.loads, method='closed_form')
        _, _, f = sol.field_grid(41, 41)
        return plate, sol, np.stack([f[k] for k in FIELDS])

    def test_scaling_round_trip_and_removes_thickness_cube(self):
        out = {}
        for ratio in (5, 100):
            plate, _, fields = self.solve(1.5, ratio, 1.0)
            scaled = scale_fields(fields, 1.0, plate.h)
            assert_allclose(unscale_fields(scaled, 1.0, plate.h), fields, rtol=1e-15, atol=0)
            out[ratio] = (np.abs(fields[2]).max(), np.abs(scaled[2]).max())
        raw_ratio = out[100][0]/out[5][0]
        scaled_ratio = out[5][1]/out[100][1]          # thick plate is softer only through shear
        self.assertGreater(raw_ratio, 1e3)             # roughly (100/5)^3 = 8000, less the shear part
        self.assertTrue(1 < scaled_ratio < 3)
        self.assertEqual(field_scales(1.0, np.array([.1, .2])).shape, (2, 5))

    def test_analytic_basis_recovers_solver_coefficients(self):
        plate, sol, fields = self.solve(2.0, 12, 1.3)
        scaled = scale_fields(fields, 1.0, plate.h)
        basis = analytic_basis(self.modes, self.grid, self.grid)
        coefficients = project_analytic(scaled, basis)
        expected = np.array([sol.coeffs[tuple(map(int, k))] for k in self.modes]).T/field_scales(1.0, plate.h)[:, None]
        assert_allclose(coefficients, expected, rtol=1e-10, atol=1e-14*np.abs(expected).max())
        self.assertLess(relative_l2(reconstruct_analytic(coefficients, basis)[None], scaled[None])[0], 1e-12)

    def test_pod_full_rank_is_exact_and_orthonormal(self):
        rng = np.random.default_rng(1)
        X = rng.normal(size=(12, 5, 4, 3))
        pod = POD(X)
        assert_allclose(pod.modes@pod.modes.T, np.eye(12), atol=1e-12)
        assert_allclose(pod.reconstruct(pod.coefficients(X, 12), X.shape[1:]), X, atol=1e-12)

    def test_bootstrap_interval(self):
        assert_allclose(bootstrap_ci(np.full(30, .2)), [[.2, .2]])
        e = np.random.default_rng(3).exponential(size=(40, 2))
        lo, hi = bootstrap_ci(e).T
        self.assertTrue(np.all(lo <= e.mean(0)) and np.all(e.mean(0) <= hi))
        lo_q, hi_q = bootstrap_ci(e, 'p95').T
        self.assertTrue(np.all(lo_q <= np.quantile(e, .95, 0)) and np.all(np.quantile(e, .95, 0) <= hi_q))
        lo_m, hi_m = bootstrap_ci(e, 'median').T
        self.assertTrue(np.all(lo_m <= np.median(e, 0)) and np.all(np.median(e, 0) <= hi_m))


if __name__ == '__main__':
    unittest.main(verbosity=2)
