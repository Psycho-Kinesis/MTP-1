import sys
from pathlib import Path
import tempfile
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fgplate.pilot import (sample_designs, design_loads, transformed_features, design_id, load_key,
                           quantize, GENERATOR_VERSION, LEGACY_VERSION)
from fgplate.baseline import ScalarBaseline


class PilotTests(unittest.TestCase):
    def test_sampling_reproducibility_ranges_and_unique_groups(self):
        x, ids, split = sample_designs()
        again = sample_designs()
        for a, b in zip((x, ids, split), again): np.testing.assert_array_equal(a, b)
        self.assertEqual(len(set(ids)), 200)
        self.assertTrue(np.all((x >= [0, 5, .5]) & (x <= [8, 100, 2])))
        groups = [set(ids[split == name]) for name in ('train', 'validation', 'test')]
        self.assertEqual([len(g) for g in groups], [140, 30, 30])
        self.assertFalse(groups[0] & groups[1] or groups[1] & groups[2] or groups[0] & groups[2])
        case_ids, case_splits = np.repeat(ids, 3), np.repeat(split, 3)
        for design_id in ids: self.assertEqual(len(set(case_splits[case_ids == design_id])), 1)
        # Latin hypercube stratification is in the declared transformed coordinates.
        t = transformed_features(x)
        u = (t-[0, np.log(5), .5])/np.array([np.log(9), np.log(20), 1.5])
        for j in range(3): self.assertEqual(len(set(np.floor(200*u[:, j]).astype(int))), 200)

    def test_ids_and_loads_survive_last_bit_parameter_differences(self):
        # pilot-1.1: a 1-ulp change in the raw samples (another platform/library build)
        # must not change quantized parameters, design IDs or mixture loads.
        x, ids, _ = sample_designs()
        raw, _, _ = sample_designs(version=LEGACY_VERSION)
        perturbed = quantize(np.nextafter(raw, np.inf))
        np.testing.assert_array_equal(perturbed, x)
        self.assertEqual([design_id(*row) for row in perturbed], ids.tolist())
        # Loads are keyed by sample index, not by the floating-point identity.
        self.assertEqual(load_key(GENERATOR_VERSION, 7, ids[7]), 'sample-7')
        self.assertEqual(design_loads(load_key(GENERATOR_VERSION, 7, 'anything'))[0][2],
                         design_loads(load_key(GENERATOR_VERSION, 7, ids[7]))[0][2])
        # The legacy scheme is preserved exactly for auditing the delivered pilot_200,
        # and it is the scheme that is sensitive to last-bit differences.
        legacy_ids = [design_id(*row, version=LEGACY_VERSION) for row in raw]
        self.assertEqual(load_key(LEGACY_VERSION, 7, legacy_ids[7]), legacy_ids[7])
        changed = [design_id(*row, version=LEGACY_VERSION) for row in np.nextafter(raw, np.inf)]
        self.assertTrue(all(a != b for a, b in zip(legacy_ids, changed)))
        with self.assertRaises(ValueError): sample_designs(version='pilot-9')

    def test_load_identity_normalization_and_signed_even_modes(self):
        a, norm = design_loads('design-a')
        b, same_norm = design_loads('design-a')
        self.assertEqual(a, b); self.assertEqual(norm, same_norm)
        self.assertEqual(a[0], {(1, 1): 1.})
        self.assertEqual(len(a[1]), 625)
        self.assertEqual(len(a[2]), 25)
        self.assertAlmostEqual(sum(v*v for v in a[2].values()), 1.)
        self.assertNotEqual(a[2], design_loads('design-b')[0][2])
        self.assertIn((2, 3), a[2])

    def test_model_scaling_uses_fit_subset_and_serialization(self):
        x, ids, split = sample_designs(40)
        train = split == 'train'
        t = transformed_features(x)
        y = np.exp(np.column_stack((.5+.1*t[:, 0]-.2*t[:, 1], 2+.4*t[:, 2])))
        for kind in ('polynomial', 'rbf'):
            model = ScalarBaseline(kind, degree=1, regularization=0.).fit(x[train], y[train])
            np.testing.assert_allclose(model.mean, t[train].mean(axis=0))
            np.testing.assert_allclose(model.scale, t[train].std(axis=0))
            np.testing.assert_allclose(model.predict(x[~train]), y[~train], rtol=1e-10)
            with tempfile.TemporaryDirectory() as temp:
                path = Path(temp)/'model.npz'
                model.save(path)
                np.testing.assert_allclose(ScalarBaseline.load(path).predict(x), model.predict(x), rtol=1e-14)


if __name__ == '__main__': unittest.main()
