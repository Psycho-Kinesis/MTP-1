/* FG Plate Lab: browser implementation of the project's mechanics and frozen surrogates.
 * Mirrors fgplate/material.py (exact power-law moments), fgplate/tsdt.py (FSDT/TSDT Navier stiffness, consistent
 * mass, fundamental flexural frequency) and the H3 surrogates (experiments/h3_common.py), seed 11.
 * Tested against Python reference values by demo/test_demo_math.js.
 * Units: SI. Design parameters: g (gradient index), r = a/h, lam = b/a.
 */
(function (root) {
  'use strict';
  const EC = 380e9, EM = 70e9, RHOC = 3800, RHOM = 2702, NU = 0.3, KS = 5 / 6;
  const W = 2;

  function comb(n, k) { let c = 1; for (let i = 1; i <= k; i++) c = c * (n - k + i) / i; return c; }

  // exact thickness moments int P(z) z^k dz, k = 0..6 (fgplate.material.FGMaterial.moment)
  function moment(Pc, Pm, n, k, h) {
    const d = Pc - Pm;
    if (k === 0) return h * (Pm + d / (n + 1));
    if (k === 1) return h * h * d / (n + 1) * (n / (n + 2)) / 2;
    if (k === 2) return h ** 3 * (Pm / 12 + d * (0.25 / (n + 1) - 1 / (n + 2) / (n + 3)));
    let total = 0;
    for (let j = 0; j <= k; j++) total += comb(k, j) * (-0.5) ** (k - j) * ((Pc - Pm) / (n + j + 1) + Pm / (j + 1));
    return h ** (k + 1) * total;
  }
  function moments(g, h) {
    const E = [], R = [];
    for (let k = 0; k <= 6; k++) { E.push(moment(EC, EM, g, k, h)); R.push(moment(RHOC, RHOM, g, k, h)); }
    return { E, R };
  }

  // sparse coefficient lists [unknown, z-power, value] of the u, v, w amplitudes and strains of one Navier mode
  function parts(alpha, beta, c1) {
    const u = [[0, 0, 1], [3, 1, 1], [3, 3, -c1], [W, 3, -c1 * alpha]];
    const v = [[1, 0, 1], [4, 1, 1], [4, 3, -c1], [W, 3, -c1 * beta]];
    const w = [[W, 0, 1]];
    const scale = (list, s) => list.map(([i, q, x]) => [i, q, s * x]);
    return {
      u, v, w, ex: scale(u, -alpha), ey: scale(v, -beta), gxy: scale(u, beta).concat(scale(v, alpha)),
      gxz: [[3, 0, 1], [W, 0, alpha], [3, 2, -3 * c1], [W, 2, -3 * c1 * alpha]],
      gyz: [[4, 0, 1], [W, 0, beta], [4, 2, -3 * c1], [W, 2, -3 * c1 * beta]],
    };
  }
  // out += s * int P(z) a(z) b(z)^T dz
  function quadAdd(out, a, b, mom, s) {
    for (const [i, p, x] of a) { if (x === 0) continue; for (const [j, q, y] of b) if (y !== 0) out[i * 5 + j] += s * mom[p + q] * x * y; }
  }
  function symmetrize(K) {
    for (let i = 0; i < 5; i++) for (let j = i + 1; j < 5; j++) { const m = 0.5 * (K[i * 5 + j] + K[j * 5 + i]); K[i * 5 + j] = K[j * 5 + i] = m; }
    return K;
  }
  // 5x5 stiffness (and optionally mass) of half-wave pair (m, n), plate a = 1 (normalized), SI moduli
  function modeMatrices(mom, h, lam, m, n, theory, wantMass) {
    const alpha = m * Math.PI, beta = n * Math.PI / lam;
    const c1 = theory === 'TSDT' ? 4 / (3 * h * h) : 0;
    const sf = theory === 'TSDT' ? 1 : KS;
    const P = parts(alpha, beta, c1);
    const K = new Float64Array(25);
    const c11 = 1 / (1 - NU * NU), c66 = 1 / (2 * (1 + NU));
    quadAdd(K, P.ex, P.ex, mom.E, c11); quadAdd(K, P.ey, P.ey, mom.E, c11);
    quadAdd(K, P.ex, P.ey, mom.E, NU * c11); quadAdd(K, P.ey, P.ex, mom.E, NU * c11);
    quadAdd(K, P.gxy, P.gxy, mom.E, c66);
    quadAdd(K, P.gxz, P.gxz, mom.E, sf * c66); quadAdd(K, P.gyz, P.gyz, mom.E, sf * c66);
    symmetrize(K);
    if (!wantMass) return { K };
    const M = new Float64Array(25);
    quadAdd(M, P.u, P.u, mom.R, 1); quadAdd(M, P.v, P.v, mom.R, 1); quadAdd(M, P.w, P.w, mom.R, 1);
    return { K, M: symmetrize(M) };
  }
  function solve5(K, b) {
    const A = Array.from({ length: 5 }, (_, i) => [K[i * 5], K[i * 5 + 1], K[i * 5 + 2], K[i * 5 + 3], K[i * 5 + 4], b[i]]);
    for (let c = 0; c < 5; c++) {
      let piv = c; for (let r = c + 1; r < 5; r++) if (Math.abs(A[r][c]) > Math.abs(A[piv][c])) piv = r;
      [A[c], A[piv]] = [A[piv], A[c]];
      for (let r = c + 1; r < 5; r++) { const f = A[r][c] / A[c][c]; for (let k = c; k < 6; k++) A[r][k] -= f * A[c][k]; }
    }
    const x = new Float64Array(5);
    for (let r = 4; r >= 0; r--) { let s = A[r][5]; for (let k = r + 1; k < 5; k++) s -= A[r][k] * x[k]; x[r] = s / A[r][r]; }
    return x;
  }

  // exact dimensionless deflection transfer H~_w = W Ec h^3 (unit modal load, a = 1) for each mode
  function solverHw(g, r, lam, modes, theory) {
    const h = 1 / r, mom = moments(g, h), out = new Float64Array(modes.length), e = [0, 0, 1, 0, 0];
    for (let k = 0; k < modes.length; k++) {
      const { K } = modeMatrices(mom, h, lam, modes[k][0], modes[k][1], theory, false);
      out[k] = solve5(K, e)[W] * EC * h ** 3;
    }
    return out;
  }

  // symmetric 5x5 eigen (cyclic Jacobi): returns {values, vectors (columns)}
  function jacobi5(A0) {
    const A = Float64Array.from(A0), V = new Float64Array(25);
    for (let i = 0; i < 5; i++) V[i * 5 + i] = 1;
    for (let sweep = 0; sweep < 60; sweep++) {
      let off = 0; for (let i = 0; i < 5; i++) for (let j = i + 1; j < 5; j++) off += A[i * 5 + j] ** 2;
      let diag = 0; for (let i = 0; i < 5; i++) diag += A[i * 5 + i] ** 2;
      if (off <= 1e-30 * diag) break;
      for (let p = 0; p < 5; p++) for (let q = p + 1; q < 5; q++) {
        const apq = A[p * 5 + q]; if (apq === 0) continue;
        const app = A[p * 5 + p], aqq = A[q * 5 + q];
        const theta = (aqq - app) / (2 * apq);
        const t = Math.sign(theta || 1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
        const c = 1 / Math.sqrt(t * t + 1), s = t * c;
        for (let k = 0; k < 5; k++) {
          const akp = A[k * 5 + p], akq = A[k * 5 + q];
          A[k * 5 + p] = c * akp - s * akq; A[k * 5 + q] = s * akp + c * akq;
        }
        for (let k = 0; k < 5; k++) {
          const apk = A[p * 5 + k], aqk = A[q * 5 + k];
          A[p * 5 + k] = c * apk - s * aqk; A[q * 5 + k] = s * apk + c * aqk;
        }
        for (let k = 0; k < 5; k++) {
          const vkp = V[k * 5 + p], vkq = V[k * 5 + q];
          V[k * 5 + p] = c * vkp - s * vkq; V[k * 5 + q] = s * vkp + c * vkq;
        }
      }
    }
    return { values: [0, 1, 2, 3, 4].map(i => A[i * 5 + i]), vectors: V };
  }
  // dimensionless fundamental flexural frequency omega_hat = omega_11 h sqrt(rho_c/Ec) (fgplate.tsdt.fundamental_batch)
  function solverOmegaHat(g, r, lam, theory) {
    const h = 1 / r, mom = moments(g, h);
    const { K, M } = modeMatrices(mom, h, lam, 1, 1, theory, true);
    const L = new Float64Array(25);                      // Cholesky M = L L^T
    for (let i = 0; i < 5; i++) for (let j = 0; j <= i; j++) {
      let s = M[i * 5 + j]; for (let k = 0; k < j; k++) s -= L[i * 5 + k] * L[j * 5 + k];
      L[i * 5 + j] = i === j ? Math.sqrt(s) : s / L[j * 5 + j];
    }
    const Li = new Float64Array(25);                     // L^{-1}
    for (let c = 0; c < 5; c++) for (let i = 0; i < 5; i++) {
      let s = i === c ? 1 : 0; for (let k = 0; k < i; k++) s -= L[i * 5 + k] * Li[k * 5 + c];
      Li[i * 5 + c] = s / L[i * 5 + i];
    }
    const T = new Float64Array(25), A = new Float64Array(25);   // A = Li K Li^T
    for (let i = 0; i < 5; i++) for (let j = 0; j < 5; j++) { let s = 0; for (let k = 0; k < 5; k++) s += Li[i * 5 + k] * K[k * 5 + j]; T[i * 5 + j] = s; }
    for (let i = 0; i < 5; i++) for (let j = 0; j < 5; j++) { let s = 0; for (let k = 0; k < 5; k++) s += T[i * 5 + k] * Li[j * 5 + k]; A[i * 5 + j] = s; }
    const { values, vectors } = jacobi5(A);
    let best = -1, bestPart = -1;
    for (let c = 0; c < 5; c++) {                        // x = Li^T y; participation I0 x_W^2
      let xw = 0; for (let k = 0; k < 5; k++) xw += Li[k * 5 + W] * vectors[k * 5 + c];
      const part = mom.R[0] * xw * xw;
      if (part > bestPart) { bestPart = part; best = c; }
    }
    return Math.sqrt(values[best]) * h * Math.sqrt(RHOC / EC);
  }

  // ---------------------------------------------------------------- frozen surrogates
  function legendreAll(x, d) {
    const P = [1, x];
    for (let k = 1; k < d; k++) P.push(((2 * k + 1) * x * P[k] - k * P[k - 1]) / (k + 1));
    return P;
  }
  function coords(g, r, lam, domain) {
    const t = [Math.log1p(g), Math.log(r), lam];
    return t.map((v, j) => 2 * (v - domain[j][0]) / (domain[j][1] - domain[j][0]) - 1);
  }
  function features(x, exps, degree) {
    const L = x.map(v => legendreAll(v, Math.max(degree, 1)));
    return exps.map(p => L[0][p[0]] * L[1][p[1]] * L[2][p[2]]);
  }
  function dot(a, b) { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; }
  function kh2(m, n, r, lam) { return Math.PI ** 2 * (m * m + (n / lam) ** 2) / (r * r); }

  function makeSurrogate(model) {
    const modes = model.modes, index = new Map(modes.map((mn, k) => [mn[0] + ',' + mn[1], k]));
    function Hw(arm, g, r, lam, modeList) {
      const x = coords(g, r, lam, model.domain);
      const list = modeList || modes;
      const out = new Float64Array(list.length);
      if (arm === 'D') {
        const F = features(x, model.D.exponents, model.D.degree);
        list.forEach((mn, i) => { out[i] = dot(F, model.D.theta_w[index.get(mn[0] + ',' + mn[1])]); });
        return out;
      }
      const FP = features(x, model.P.exponents, model.P.degree);
      const FC = arm === 'C' ? features(x, model.C.exponents, model.C.degree) : null;
      list.forEach((mn, i) => {
        const k = index.get(mn[0] + ',' + mn[1]);
        const k4 = ((mn[0] * Math.PI) ** 2 + (mn[1] * Math.PI / lam) ** 2) ** 2;
        const hp = dot(FP, model.P.theta_w[k]) / k4;
        out[i] = arm === 'C' ? hp + hp * kh2(mn[0], mn[1], r, lam) * dot(FC, model.C.theta_w[k]) : hp;
      });
      return out;
    }
    function omegaHat(arm, g, r, lam) {
      const x = coords(g, r, lam, model.domain);
      if (arm === 'D') return Math.exp(dot(features(x, model.freq_D.exponents, model.freq_D.degree), model.freq_D.theta));
      const base = solverOmegaHat(g, r, lam, 'FSDT');
      if (arm === 'P') return base;
      return base * Math.sqrt(1 + kh2(1, 1, r, lam) * dot(features(x, model.freq_C.exponents, model.freq_C.degree), model.freq_C.theta));
    }
    return { modes, Hw, omegaHat };
  }

  // ---------------------------------------------------------------- loads and fields
  function oddModes(k) { const out = []; for (let m = 1; m <= k; m += 2) for (let n = 1; n <= k; n += 2) out.push([m, n]); return out; }
  function squareModes(k) { const out = []; for (let m = 1; m <= k; m++) for (let n = 1; n <= k; n++) out.push([m, n]); return out; }
  function patchIntegrals(c, sigma, kmax, pts = 1201) {         // int_0^1 sin(m pi t) exp(-(t-c)^2/2s^2) dt (Simpson)
    const I = new Float64Array(kmax + 1), hstep = 1 / (pts - 1);
    for (let i = 0; i < pts; i++) {
      const t = i * hstep, wgt = (i === 0 || i === pts - 1) ? 1 : (i % 2 ? 4 : 2);
      const e = Math.exp(-((t - c) ** 2) / (2 * sigma * sigma)) * wgt * hstep / 3;
      for (let m = 1; m <= kmax; m++) I[m] += Math.sin(m * Math.PI * t) * e;
    }
    return I;
  }
  // load coefficients Q_mn for unit peak pressure: {type: 'uniform'|'sine'|'patch', x0, y0, sigma}
  function loadCoefficients(load, modes) {
    if (load.type === 'sine') return modes.map(([m, n]) => (m === 1 && n === 1 ? 1 : 0));
    if (load.type === 'uniform') return modes.map(([m, n]) => (m % 2 && n % 2 ? 16 / (Math.PI ** 2 * m * n) : 0));
    const kmax = Math.max(...modes.map(mn => Math.max(mn[0], mn[1])));
    const Ix = patchIntegrals(load.x0, load.sigma, kmax), Iy = patchIntegrals(load.y0, load.sigma, kmax);
    return modes.map(([m, n]) => 4 * Ix[m] * Iy[n]);
  }
  // deflection field (dimensionless w Ec h^3/(q0 a^4)) on an (N+1) x (N+1) grid of xi, eta in [0, 1]
  function field(Hw, Q, modes, N) {
    const kmax = Math.max(...modes.map(mn => Math.max(mn[0], mn[1])));
    const S = []; for (let m = 0; m <= kmax; m++) { const row = new Float64Array(N + 1); for (let i = 0; i <= N; i++) row[i] = Math.sin(m * Math.PI * i / N); S.push(row); }
    const Z = Array.from({ length: N + 1 }, () => new Float64Array(N + 1));
    modes.forEach(([m, n], k) => {
      const c = Q[k] * Hw[k]; if (c === 0) return;
      const Sm = S[m], Sn = S[n];
      for (let j = 0; j <= N; j++) { const cj = c * Sn[j]; if (cj === 0) continue; const Zj = Z[j]; for (let i = 0; i <= N; i++) Zj[i] += cj * Sm[i]; }
    });
    return Z;                                    // Z[j][i]: eta_j (y), xi_i (x)
  }
  function maxAbs(Z) { let m = 0; for (const row of Z) for (const v of row) if (Math.abs(v) > m) m = Math.abs(v); return m; }
  function centreValue(Hw, Q, modes) {
    let s = 0; modes.forEach(([m, n], k) => { s += Q[k] * Hw[k] * Math.sin(m * Math.PI / 2) * Math.sin(n * Math.PI / 2); }); return s;
  }

  // ---------------------------------------------------------------- design problem (TID / H3)
  const physical = {
    deflection: (wHat, p, r) => p.q0 * p.a * wHat * r ** 3 / EC,                      // q0 a^4 w/(Ec h^3), h = a/r
    frequency: (omegaHat, p, r) => omegaHat * r / (2 * Math.PI * p.a) * Math.sqrt(EC / RHOC),
    mass: (h, g) => h * (RHOM + (RHOC - RHOM) / (g + 1)),
    ceramic: (h, g) => h / (g + 1),
  };
  const G_MAX = 8, R_MIN = 5, R_MAX = 100;
  // For each g the constraints are monotone in thickness: the thinnest safe plate r_max(g) (largest a/h) is found by
  // bisection; it is also the lightest and uses the least ceramic for that g. The optimum is then a 1-D search in g.
  function makeProblem(evaluate, p) {
    let count = 0;
    const lam = p.aspect, cache = new Map();
    function phi(g, r) {                         // >= 0 when deflection and frequency limits hold
      const ev = evaluate(g, r, lam); count++;
      return Math.min(1 - physical.deflection(ev.wHat, p, r) / p.wLim, physical.frequency(ev.omegaHat, p, r) / p.fMin - 1);
    }
    function rMax(g) {
      const key = g.toPrecision(15);
      if (cache.has(key)) return cache.get(key);
      let out;
      if (phi(g, R_MAX) >= 0) out = R_MAX;
      else if (phi(g, R_MIN) < 0) out = null;
      else {
        let lo = Math.log(R_MIN), hi = Math.log(R_MAX);           // phi(lo) >= 0 > phi(hi)
        for (let it = 0; it < 44; it++) { const mid = 0.5 * (lo + hi); if (phi(g, Math.exp(mid)) >= 0) lo = mid; else hi = mid; }
        out = Math.exp(lo);
      }
      cache.set(key, out);
      return out;
    }
    function point(g) {                          // lightest design for this g, with its feasibility under the ceramic limit
      const r = rMax(g);
      if (r === null) return { g, feasible: false, reason: 'limits' };
      const h = p.a / r, cer = physical.ceramic(h, g), mass = physical.mass(h, g);
      const feasible = !(p.ceramicMax > 0) || cer <= p.ceramicMax * (1 + 1e-12);
      return { g, r, h, cer, mass, feasible, reason: feasible ? '' : 'ceramic' };
    }
    return { point, rMax, get count() { return count; } };
  }
  function goldenMin(f, a, b, iters = 40) {
    const gr = (Math.sqrt(5) - 1) / 2;
    let c = b - gr * (b - a), d = a + gr * (b - a), fc = f(c), fd = f(d);
    for (let i = 0; i < iters; i++) {
      if (fc < fd) { b = d; d = c; fd = fc; c = b - gr * (b - a); fc = f(c); }
      else { a = c; c = d; fc = fd; d = a + gr * (b - a); fd = f(d); }
    }
    return fc < fd ? c : d;
  }
  // minimum areal mass subject to the limits; evaluate(g, r, lam) -> {wHat (centre, uniform), omegaHat}
  function optimize(evaluate, p, opts = {}) {
    const P = makeProblem(evaluate, p), n = opts.grid || 41;
    const U = Math.log1p(G_MAX), gOf = u => Math.expm1(u);
    const pts = [];
    for (let i = 0; i < n; i++) pts.push(P.point(gOf(U * i / (n - 1))));
    let ib = -1;
    pts.forEach((q, i) => { if (q.feasible && (ib < 0 || q.mass < pts[ib].mass)) ib = i; });
    if (ib < 0) return { feasible: false, count: P.count, profile: pts };
    const cands = [pts[ib]];
    const massAt = u => { const q = P.point(gOf(u)); return q.feasible ? q.mass : Infinity; };
    for (const j of [ib - 1, ib + 1]) {
      if (j < 0 || j >= n) continue;
      const ui = U * ib / (n - 1), uj = U * j / (n - 1);
      if (pts[j].feasible) cands.push(P.point(gOf(goldenMin(massAt, Math.min(ui, uj), Math.max(ui, uj)))));
      else {                                     // feasibility boundary between i and j (ceramic limit or limits)
        let a = ui, b = uj;
        for (let it = 0; it < 44; it++) { const mid = 0.5 * (a + b); if (P.point(gOf(mid)).feasible) a = mid; else b = mid; }
        cands.push(P.point(gOf(a)));
        cands.push(P.point(gOf(goldenMin(massAt, Math.min(ui, a), Math.max(ui, a)))));
      }
    }
    let best = null;
    for (const q of cands) if (q.feasible && (!best || q.mass < best.mass)) best = q;
    return { feasible: true, design: best, count: P.count, profile: pts };
  }
  // lightest design for every g (for the trade-off curve): mass and ceramic use along r_max(g)
  function profile(evaluate, p, n = 81) {
    const P = makeProblem(evaluate, Object.assign({}, p, { ceramicMax: 0 })), U = Math.log1p(G_MAX), out = [];
    for (let i = 0; i < n; i++) out.push(P.point(Math.expm1(U * i / (n - 1))));
    return { points: out, count: P.count };
  }
  // dense map for plotting: phi-feasibility (limits only) and mass over (g, a/h)
  function designMap(evaluate, p, nu = 41, nv = 41) {
    const lam = p.aspect, U = Math.log1p(G_MAX), V0 = Math.log(R_MIN), V1 = Math.log(R_MAX);
    const gs = [], rs = [], ok = [], mass = [];
    for (let j = 0; j < nv; j++) rs.push(Math.exp(V0 + (V1 - V0) * j / (nv - 1)));
    for (let i = 0; i < nu; i++) gs.push(Math.expm1(U * i / (nu - 1)));
    for (let j = 0; j < nv; j++) {
      const rowOk = [], rowM = [];
      for (let i = 0; i < nu; i++) {
        const g = gs[i], r = rs[j], h = p.a / r, ev = evaluate(g, r, lam);
        const w = physical.deflection(ev.wHat, p, r), f = physical.frequency(ev.omegaHat, p, r);
        const cerOk = !(p.ceramicMax > 0) || physical.ceramic(h, g) <= p.ceramicMax;
        rowOk.push(w <= p.wLim && f >= p.fMin && cerOk ? 1 : 0); rowM.push(physical.mass(h, g));
      }
      ok.push(rowOk); mass.push(rowM);
    }
    return { gs, rs, ok, mass };
  }

  const api = { EC, EM, RHOC, RHOM, NU, moment, moments, modeMatrices, solve5, solverHw, solverOmegaHat, jacobi5,
    makeSurrogate, oddModes, squareModes, loadCoefficients, field, maxAbs, centreValue, physical, optimize, profile, designMap, kh2 };
  if (typeof module !== 'undefined' && module.exports) module.exports = api; else root.FGMath = api;
})(typeof window !== 'undefined' ? window : globalThis);
