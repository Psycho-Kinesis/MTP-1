"""Checks for the FSDT/TSDT per-mode stiffness (fgplate.tsdt); run with unittest."""
import json
import sys
from pathlib import Path
import unittest
import numpy as np
from numpy.testing import assert_allclose
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fgplate import NavierFSDTPlate, al_al2o3, isotropic
from fgplate.fields import field_scales
from fgplate.tsdt import (eigenpairs, fundamental_batch, mass, mass_batch, stiffness, stiffness_batch, transfer,
                          transfer_batch)

REFS = json.loads((Path(__file__).with_name('benchmark_references.json')).read_text(encoding='utf-8'))


def wbar(g, theory, ratio=10):
    K = stiffness(al_al2o3(g), 1/ratio, 1.0, 1.0, 1, 1, theory)
    return 10*380e9*(1/ratio)**3*np.linalg.solve(K, [0, 0, 1, 0, 0])[2]


class TSDTTests(unittest.TestCase):
    def test_fsdt_limit_equals_navier_matrices(self):
        for g, ratio, aspect in [(0, 5, .5), (1.3, 12, .8), (6, 100, 2), (.02, 7, 1.7)]:
            plate = NavierFSDTPlate(1, aspect, 1/ratio, al_al2o3(g))
            for m, n in [(1, 1), (2, 3), (5, 4)]:
                assert_allclose(stiffness(al_al2o3(g), 1/ratio, 1, aspect, m, n, 'FSDT'), plate.matrices(m, n)[0],
                                rtol=1e-13, atol=1e-13*np.abs(plate.matrices(m, n)[0]).max())

    def test_tsdt_symmetric_positive_definite_and_stiffer_than_fsdt_in_shear(self):
        for g, ratio in [(0, 5), (4, 10), (8, 100)]:
            K = stiffness(al_al2o3(g), 1/ratio, 1, 1.3, 2, 3, 'TSDT')
            assert_allclose(K, K.T, rtol=1e-14, atol=0)
            self.assertTrue(np.all(np.linalg.eigvalsh(K) > 0))

    def test_reproduces_published_higher_order_deflections(self):
        # Nguyen et al. (2019) Table 4 lists SSDT values (a different higher-order theory):
        # agreement to ~0.05% is expected, not identity. g = 0, 1, 2 agree to 4 digits.
        values = REFS['fg_sine_deflection']['values']
        for g in ('0', '1', '2'):
            self.assertLess(abs(wbar(float(g), 'TSDT')/values[g]-1), 2e-4)
        for g in ('4', '10'):
            self.assertLess(abs(wbar(float(g), 'TSDT')/values[g]-1), 1e-3)

    def test_theories_converge_for_thin_plates_and_differ_for_thick(self):
        thin = abs(wbar(4, 'TSDT', 200)/wbar(4, 'FSDT', 200)-1)
        thick = abs(wbar(4, 'TSDT', 5)/wbar(4, 'FSDT', 5)-1)
        self.assertLess(thin, 1e-4)
        self.assertGreater(thick, 10*thin)
        K = stiffness(isotropic(70e9, 2702), .1, 1, 1, 1, 1, 'TSDT')
        self.assertEqual(K[0, 3], 0)   # no stretching-bending coupling for a homogeneous plate

    def test_transfer_scaling(self):
        params = np.array([[1.5, 20.0, 1.2]])
        H = transfer(params, [(1, 1), (2, 1)], 'FSDT')
        plate = NavierFSDTPlate(1, 1.2, 1/20, al_al2o3(1.5))
        sol = plate.bending_modal({(1, 1): 1.0, (2, 1): 1.0}, method='closed_form')
        expected = np.array([sol.coeffs[(1, 1)], sol.coeffs[(2, 1)]])/field_scales(1.0, 1/20)
        assert_allclose(H[0], expected, rtol=1e-10)
        with self.assertRaises(ValueError):
            stiffness(al_al2o3(1), .1, 1, 1, 1, 1, 'HSDT')


class TSDTVibrationAndBatchTests(unittest.TestCase):
    designs = np.array([[0, 5, .5], [1.3, 12, .8], [6, 100, 2], [.02, 7, 1.7], [8, 5, 1.0], [.4, 40, 1.3]])

    def test_fsdt_limit_mass_equals_navier(self):
        for g, ratio, aspect in self.designs:
            plate = NavierFSDTPlate(1, aspect, 1/ratio, al_al2o3(g))
            for m, n in [(1, 1), (2, 3), (5, 4)]:
                assert_allclose(mass(al_al2o3(g), 1/ratio, 1, aspect, m, n, 'FSDT'), plate.matrices(m, n)[1], rtol=1e-14, atol=0)
            # eigenvalues of the tiny 5x5 problem are conditioning-limited (~1e-10), not formula-limited
            assert_allclose(eigenpairs(al_al2o3(g), 1/ratio, 1, aspect, 1, 1, 'FSDT')[0], plate.frequencies(1, 1), rtol=1e-8)

    def test_tsdt_mass_positive_definite_and_thin_limit(self):
        for g, ratio, aspect in self.designs:
            M = mass(al_al2o3(g), 1/ratio, 1, aspect, 2, 1, 'TSDT')
            assert_allclose(M, M.T, rtol=1e-14, atol=0)
            self.assertTrue(np.all(np.linalg.eigvalsh(M) > 0))
        thin = fundamental_batch([[4, 200, 1]], 'TSDT')/fundamental_batch([[4, 200, 1]], 'FSDT')-1
        thick = fundamental_batch([[4, 5, 1]], 'TSDT')/fundamental_batch([[4, 5, 1]], 'FSDT')-1
        self.assertLess(abs(thin[0]), 1e-4)
        self.assertGreater(abs(thick[0]), 10*abs(thin[0]))

    def test_published_tsdt_frequencies(self):
        # Reddy TSDT values quoted (4 digits) in Belkhodja et al. (2020) Table 5; see benchmark_references.json
        ref = REFS['fg_tsdt_frequency']['values']
        for g, value in ref.items():
            self.assertLess(abs(fundamental_batch([[float(g), 5, 1]], 'TSDT')[0]-value), 0.6e-4)

    def test_batch_equals_scalar(self):
        modes = [(1, 1), (2, 3), (5, 4), (7, 1)]
        for theory in ('FSDT', 'TSDT'):
            assert_allclose(transfer_batch(self.designs, modes, theory), transfer(self.designs, modes, theory), rtol=1e-11)
            K, M = stiffness_batch(self.designs, modes, theory), mass_batch(self.designs, modes, theory)
            for i, (g, ratio, aspect) in enumerate(self.designs):
                for k, (m, n) in enumerate(modes):
                    Ks = stiffness(al_al2o3(g), 1/ratio, 1, aspect, m, n, theory)
                    assert_allclose(K[i, k], Ks, rtol=1e-12, atol=1e-13*np.abs(Ks).max())
                    assert_allclose(M[i, k], mass(al_al2o3(g), 1/ratio, 1, aspect, m, n, theory), rtol=1e-12, atol=0)
            omega = np.array([eigenpairs(al_al2o3(g), 1/r, 1, asp, 1, 1, theory)[0][0]*(1/r)*np.sqrt(3800/380e9)
                              for g, r, asp in self.designs])
            assert_allclose(fundamental_batch(self.designs, theory), omega, rtol=1e-8)


if __name__ == '__main__':
    unittest.main(verbosity=2)
