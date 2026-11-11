"""Checks for the modal transfer model (data / physics / hybrid); run with unittest."""
import sys
from pathlib import Path
import unittest
import numpy as np
from numpy.testing import assert_allclose
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fgplate import NavierFSDTPlate, al_al2o3
from fgplate.fields import field_scales
from fgplate.transfer import (TransferModel, assemble_data, assemble_physics, classical_prefactor, design_coordinates,
                              exact_transfer, exponents, features, mode_matrices, predict_scaled, solve_modes)

MODES = [(m, n) for m in range(1, 4) for n in range(1, 4)]


class TransferTests(unittest.TestCase):
    def setUp(self):
        self.params = np.array([[1.3, 12.0, 0.8], [0.05, 60.0, 1.7], [6.0, 5.5, 1.1]])
        self.K, self.f = mode_matrices(self.params, MODES)

    def test_mode_matrices_reproduce_solver(self):
        for i, (g, ratio, aspect) in enumerate(self.params):
            plate = NavierFSDTPlate(1.0, aspect, 1/ratio, al_al2o3(g))
            sol = plate.bending_modal({mode: 1.0 for mode in MODES}, method='closed_form')
            expected = np.array([sol.coeffs[mode] for mode in MODES])/field_scales(1.0, 1/ratio)
            assert_allclose(exact_transfer(self.K[i], self.f[i]), expected, rtol=1e-9)
        assert_allclose(self.K, np.swapaxes(self.K, -1, -2), rtol=1e-12, atol=0)
        assert_allclose(self.K[..., 2, 2], 1)
        self.assertTrue(np.all(np.linalg.eigvalsh(self.K) > 0))

    def test_physics_only_recovers_truth_without_labels_when_class_contains_it(self):
        # One collocation design, constant model: the energy minimizer is K^-1 f exactly.
        for i in range(3):
            model = TransferModel(0, ridge=0).fit(x_phys=design_coordinates(self.params[i:i+1]),
                                                  K=self.K[i:i+1], f=self.f[i:i+1])
            assert_allclose(model.predict(design_coordinates(self.params[i:i+1]))[0],
                            exact_transfer(self.K[i], self.f[i]), rtol=1e-10)

    def test_physics_minimizer_invariant_to_positive_design_weights(self):
        model_a = TransferModel(0, 0).fit(x_phys=design_coordinates(self.params[:1]), K=self.K[:1], f=self.f[:1])
        model_b = TransferModel(0, 0).fit(x_phys=design_coordinates(self.params[:1]), K=7*self.K[:1], f=7*self.f[:1])
        assert_allclose(model_a.theta, model_b.theta, rtol=1e-12)

    def test_data_only_interpolates_and_hybrid_agrees_with_consistent_truth(self):
        x = design_coordinates(self.params)
        truth = exact_transfer(self.K, self.f)
        # 4 coefficients, 3 designs: underdetermined, so a tiny ridge selects an interpolant.
        data = TransferModel(1, 1e-14).fit(x_data=x, labels=truth)
        assert_allclose(data.predict(x), truth, rtol=1e-8, atol=1e-12*np.abs(truth).max())
        hybrid = TransferModel(0, 0).fit(x_data=x[:1], labels=truth[:1], x_phys=x[:1], K=self.K[:1],
                                         f=self.f[:1], physics_weight=3.0)
        assert_allclose(hybrid.predict(x[:1]), truth[:1], rtol=1e-10)

    def test_block_ridge_is_equivariant_to_component_rescaling(self):
        rng = np.random.default_rng(2)
        params = np.column_stack((rng.uniform(0, 8, 12), np.exp(rng.uniform(np.log(5), np.log(100), 12)),
                                  rng.uniform(.5, 2, 12)))
        x = design_coordinates(params)
        K, f = mode_matrices(params, MODES[:2])
        truth = exact_transfer(K, f)
        scaled = truth.copy()
        scaled[..., 0] *= 1e-6
        a = TransferModel(2, 1e-3, scope='block').fit(x_data=x, labels=truth).predict(x)
        b = TransferModel(2, 1e-3, scope='block').fit(x_data=x, labels=scaled).predict(x)
        assert_allclose(b[..., 0], 1e-6*a[..., 0], rtol=1e-8)
        assert_allclose(b[..., 1:], a[..., 1:], rtol=1e-8)
        with self.assertRaises(ValueError):
            TransferModel(2, 1e-3, scope='other').fit(x_data=x, labels=truth)

    def test_prefactor_data_loss(self):
        rng = np.random.default_rng(4)
        x = rng.uniform(-1, 1, (15, 3))
        labels = rng.normal(size=(15, 2, 5))
        A0, b0, s0 = assemble_data(x, labels, 2)
        A1, b1, s1 = assemble_data(x, labels, 2, prefactor=np.ones((15, 2, 5)))
        assert_allclose(A1, A0, rtol=1e-13); assert_allclose(b1, b0, rtol=1e-13); assert_allclose(s1, s0)
        # Residuals that are exactly prefactor * (polynomial of degree 1) are recovered exactly.
        pref = rng.uniform(.1, 2, (15, 2, 5))
        theta = rng.normal(size=(2, 5, 4))
        target = predict_scaled(theta, 1, x, pref)
        A, b, _ = assemble_data(x, target, 1, prefactor=pref)
        fitted = solve_modes(A, b, 1, 0.0, 'block')
        assert_allclose(predict_scaled(fitted, 1, x, pref), target, rtol=1e-9, atol=1e-12)
        assert_allclose(fitted, theta, rtol=1e-8, atol=1e-10)

    def test_physics_prefactor_and_log_aspect(self):
        x = design_coordinates(self.params)
        A0, b0 = assemble_physics(x, self.K, self.f, 2)
        A1, b1 = assemble_physics(x, self.K, self.f, 2, prefactor=np.ones(self.f.shape))
        assert_allclose(A1, A0, rtol=1e-13); assert_allclose(b1, b0, rtol=1e-13)
        pref = classical_prefactor(self.params, MODES)
        self.assertEqual(pref.shape, self.f.shape)
        for i in range(3):   # one design, constant model: the prefactored energy minimizer is still exact
            th = solve_modes(*assemble_physics(x[i:i+1], self.K[i:i+1], self.f[i:i+1], 0, prefactor=pref[i:i+1]), 0, 0.0)
            assert_allclose(predict_scaled(th, 0, x[i:i+1], pref[i:i+1])[0], exact_transfer(self.K[i], self.f[i]), rtol=1e-10)
        homogeneous = classical_prefactor(np.array([[0.0, 10.0, 1.0]]), MODES)
        assert_allclose(homogeneous[..., :2], 0, atol=0)     # no stretching-bending coupling at g = 0
        self.assertTrue(np.all(homogeneous[..., 2:] > 0))
        xl = design_coordinates(np.array([[0, 5, .5], [8, 100, 2], [1, 20, 1]]), aspect='log')
        assert_allclose(xl[:2], [[-1, -1, -1], [1, 1, 1]], atol=1e-14)
        self.assertAlmostEqual(xl[2, 2], 0.0, places=14)    # b/a = 1 sits at the centre of the log axis
        with self.assertRaises(ValueError):
            design_coordinates(self.params, aspect='other')

    def test_features_and_domain(self):
        x = design_coordinates(np.array([[0, 5, .5], [8, 100, 2]]))
        assert_allclose(x, [[-1, -1, -1], [1, 1, 1]], atol=1e-14)
        F = features(np.random.default_rng(0).uniform(-1, 1, (50, 3)), 4)
        self.assertEqual(F.shape, (50, len(exponents(4))))
        assert_allclose(F[:, 0], 1)
        with self.assertRaises(ValueError):
            TransferModel(2).fit()


if __name__ == '__main__':
    unittest.main(verbosity=2)
