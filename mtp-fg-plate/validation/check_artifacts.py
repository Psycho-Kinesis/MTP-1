"""Audit the saved pilot without refitting or tuning against its test set."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fgplate import NavierFSDTPlate, al_al2o3
from fgplate.navier import BendingSolution
from fgplate.pilot import (FIELDS, LEGACY_VERSION, design_id, design_loads, load_key, quantize,
                           sample_designs)
from fgplate.baseline import ScalarBaseline, error_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', nargs='?', type=Path, default=ROOT/'results'/'pilot_200')
    out = parser.parse_args().directory
    meta = json.loads((out/'manifest.json').read_text())
    checksums = json.loads((out/'checksums.json').read_text())
    for filename, checksum in checksums.items():
        assert hashlib.sha256((out/filename).read_bytes()).hexdigest() == checksum, filename
    # Generating source: either the current tree, or (after later development) the
    # archived snapshot provenance/<generator_version>/ must match the recorded hashes.
    drifted = []
    for filename, checksum in meta['source_sha256'].items():
        if hashlib.sha256((ROOT/filename).read_bytes()).hexdigest() != checksum:
            snapshot = ROOT/'provenance'/meta['generator_version']/filename
            assert snapshot.exists() and hashlib.sha256(snapshot.read_bytes()).hexdigest() == checksum, filename
            drifted.append(filename)
    note = f' (changed since generation; verified against provenance snapshot: {", ".join(drifted)})' if drifted else ''
    print('PASS: saved artifacts and generation source checksums' + note)
    with np.load(out/'dataset.npz', allow_pickle=False) as z:
        # The stored archive is the source of truth. Regenerated samples may differ in the
        # last bit across platforms/library builds, so they are compared with a tolerance;
        # identities and loads are then re-derived from the STORED parameters, which is
        # platform independent.
        version = meta['generator_version']
        params, ids, split = z['parameters'], z['design_ids'], z['split']
        regenerated, _, regenerated_split = sample_designs(meta['designs'], meta['seed'], version)
        np.testing.assert_allclose(params, regenerated, rtol=1e-12, atol=0)
        np.testing.assert_array_equal(split, regenerated_split)
        if version != LEGACY_VERSION:
            np.testing.assert_array_equal(params, quantize(params))
        np.testing.assert_array_equal(ids, [design_id(*row, version=version) for row in params])
        assert len(set(ids.tolist())) == meta['designs']
        modes = [tuple(pair) for pair in z['mode_indices']]
        for i, identifier in enumerate(ids):
            loads = design_loads(load_key(version, i, identifier), meta['seed'])[0]
            for j, load in enumerate(loads):
                expected = np.zeros(len(modes))
                for pair, q in load.items(): expected[modes.index(pair)] = q
                np.testing.assert_allclose(z['pressure_coefficients'][i, j], expected, rtol=1e-13, atol=1e-15)
        split_json = json.loads((out/'split.json').read_text())
        for name in ('train', 'validation', 'test'):
            assert ids[split == name].tolist() == split_json[name]
        assert z['fields'].shape == (meta['designs'], 3, 5, 41, 41)
        for name in z.files:
            if z[name].dtype.kind in 'fi': assert np.isfinite(z[name]).all(), name
        print(f'PASS: shapes, finite labels, {version} design identities and loads re-derived from stored parameters, disjoint splits')
        # Audit all three families for designs from each partition; fields are
        # reconstructed from the archive and independently resolved from loads.
        sampled = [np.flatnonzero(split == name)[0] for name in ('train', 'validation', 'test')]
        rng = np.random.default_rng(314159)
        grid_x, grid_y = rng.integers(0, 41, size=(2, 17))
        for i in sampled:
            g, slenderness, aspect = params[i]
            plate = NavierFSDTPlate(1, aspect, 1/slenderness, al_al2o3(g))
            for j in range(3):
                active = z['pressure_coefficients'][i, j] != 0
                modes = [tuple(p) for p in z['mode_indices'][active]]
                loads = dict(zip(modes, z['pressure_coefficients'][i, j, active]))
                coeffs = dict(zip(modes, z['solution_coefficients'][i, j, active]))
                stored = BendingSolution(plate, coeffs)
                direct = plate.bending_modal(loads, method='matrix')
                a = stored.fields(grid_x/40, aspect*grid_y/40)
                b = direct.fields(grid_x/40, aspect*grid_y/40)
                for k, name in enumerate(FIELDS):
                    floor = max(np.max(abs(z['fields'][i, j, k]))*1e-12, 1e-30)
                    np.testing.assert_allclose(a[name], z['fields'][i, j, k, grid_y, grid_x], atol=floor, rtol=1e-10)
                    np.testing.assert_allclose(a[name], b[name], atol=floor, rtol=1e-9)
        print('PASS: archived coefficients and all five fields match fresh matrix solves')
        with np.load(out/'scalar_predictions.npz', allow_pickle=False) as saved:
            test = split == 'test'
            prediction = ScalarBaseline.load(out/'scalar_model.npz').predict(params[test])
            np.testing.assert_array_equal(saved['test_ids'], ids[test])
            np.testing.assert_allclose(saved['test_truth'], z['scalar_targets'][test], rtol=0, atol=0)
            np.testing.assert_allclose(prediction, saved['test_prediction'], rtol=1e-13)
            metrics = error_metrics(saved['test_truth'], prediction)
            for key, value in metrics.items():
                np.testing.assert_allclose(value, meta['scalar_baseline']['test'][key], rtol=1e-12)
        print('PASS: saved model predictions and reported held-out metrics')


if __name__ == '__main__': main()
