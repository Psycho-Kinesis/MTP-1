// Tests the browser math (demo/fg_plate_math.js) against Python reference values in demo/fg_plate_lab_data.json.
// Run from the project root:  node demo/test_demo_math.js
const fs = require('fs');
const path = require('path');
const M = require('./fg_plate_math.js');
const data = JSON.parse(fs.readFileSync(path.join(__dirname, 'fg_plate_lab_data.json'), 'utf8'));
const T = data.tests, S = M.makeSurrogate(data.model);
let worst = {}, failed = 0;
function rel(a, b) { return Math.abs(a / b - 1); }
function track(name, e, tol) { worst[name] = Math.max(worst[name] || 0, e); if (e > tol) { failed++; console.log('FAIL', name, e); } }
const odd25 = M.oddModes(25), odd15 = M.oddModes(15);
T.designs.forEach(([g, r, lam], i) => {
  const Qu = M.loadCoefficients({ type: 'uniform' }, odd25);
  for (const th of ['TSDT', 'FSDT']) {
    const Hw = M.solverHw(g, r, lam, odd25, th);
    track(`${th} centre deflection (uniform)`, rel(M.centreValue(Hw, Qu, odd25), T[`${th.toLowerCase()}_centre_uniform_odd25`][i]), 1e-9);
    track(`${th} fundamental frequency`, rel(M.solverOmegaHat(g, r, lam, th), T[`${th.toLowerCase()}_omega_hat`][i]), 1e-8);
  }
  const Hm = M.solverHw(g, r, lam, T.tsdt_H_w_modes, 'TSDT');
  T.tsdt_H_w_modes.forEach((_, k) => track('TSDT H~_w per mode', rel(Hm[k], T.tsdt_H_w[i][k]), 1e-9));
  const Q15 = M.loadCoefficients({ type: 'uniform' }, odd15);
  for (const arm of ['P', 'C', 'D']) {
    const Hs = S.Hw(arm, g, r, lam, odd15);
    track(`surrogate ${arm} centre deflection`, rel(M.centreValue(Hs, Q15, odd15), T[`surrogate_${arm}_centre_uniform`][i]), 1e-8);
    track(`surrogate ${arm} frequency`, rel(S.omegaHat(arm, g, r, lam), T[`surrogate_${arm}_omega_hat`][i]), 1e-8);
  }
});
// field maximum at the centre for uniform load
const modes = M.squareModes(15), Q = M.loadCoefficients({ type: 'uniform' }, modes), Hw = M.solverHw(2, 10, 1.5, modes, 'TSDT');
const Z = M.field(Hw, Q, modes, 40);
track('uniform field max equals centre', rel(M.maxAbs(Z), Math.abs(M.centreValue(Hw, Q, modes))), 1e-12);
// patch coefficients: a very wide centred patch approaches the uniform load coefficient of mode (1,1)
const Qp = M.loadCoefficients({ type: 'patch', x0: 0.5, y0: 0.5, sigma: 50 }, [[1, 1]]);
track('wide patch -> uniform (1,1)', rel(Qp[0], 16 / Math.PI ** 2), 1e-3);
for (const [k, v] of Object.entries(worst)) console.log(`${k.padEnd(40)} max rel diff ${v.toExponential(2)}`);
console.log(failed ? `${failed} FAILED` : 'all demo math checks passed');
process.exit(failed ? 1 : 0);
