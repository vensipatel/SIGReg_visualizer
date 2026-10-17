// sigreg.js
// Pure-JavaScript implementations of Weak and Strong SIGReg.
// Operates on Float64Array matrices in row-major layout, shape [n, d].
// Designed for browser-side visualization: every transformation also produces
// a per-iteration trajectory so the UI can animate points from "before" to
// "after" while showing the SIGReg loss dropping.
//
// Notation matches the original PyTorch snippets in kreasof-ai/sigreg:
//   - Strong SIGReg: forces Empirical Characteristic Function -> Gaussian CF
//     across all moments (LeJEPA Algorithm 1).
//   - Weak SIGReg: forces Covariance -> Identity (2nd moment only).

// ---------- math helpers ----------
function randn() {
  // Box-Muller, deterministic per-call.
  let u = 0, v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

// Center a matrix and return centered copy + per-column mean.
function meanCenter(X, n, d) {
  const mean = new Float64Array(d);
  for (let i = 0; i < n; i++) {
    const base = i * d;
    for (let j = 0; j < d; j++) mean[j] += X[base + j];
  }
  for (let j = 0; j < d; j++) mean[j] /= n;
  const Xc = new Float64Array(n * d);
  for (let i = 0; i < n; i++) {
    const base = i * d;
    for (let j = 0; j < d; j++) Xc[base + j] = X[base + j] - mean[j];
  }
  return { Xc, mean };
}

// Covariance of an already-centered matrix.
function covariance(Xc, n, d, out) {
  const cov = out || new Float64Array(d * d);
  for (let i = 0; i < d; i++) {
    for (let j = i; j < d; j++) {
      let s = 0;
      for (let k = 0; k < n; k++) s += Xc[k * d + i] * Xc[k * d + j];
      s /= (n - 1);
      cov[i * d + j] = s;
      cov[j * d + i] = s;
    }
  }
  return cov;
}

// Classical Jacobi rotations for symmetric eigendecomposition. Stable for the
// small-to-medium d we encounter here (d up to ~768).
function symJacobi(M, maxIter, tol) {
  maxIter = maxIter || 500;
  tol = tol || 1e-12;
  const d = Math.round(Math.sqrt(M.length));
  const A = Float64Array.from(M);
  const V = new Float64Array(d * d);
  for (let i = 0; i < d; i++) V[i * d + i] = 1;

  for (let iter = 0; iter < maxIter; iter++) {
    let maxOff = 0, p = 0, q = 0;
    for (let i = 0; i < d; i++) {
      for (let j = i + 1; j < d; j++) {
        const a = Math.abs(A[i * d + j]);
        if (a > maxOff) { maxOff = a; p = i; q = j; }
      }
    }
    if (maxOff < tol) break;

    const app = A[p * d + p];
    const aqq = A[q * d + q];
    const apq = A[p * d + q];
    const theta = (Math.abs(app - aqq) < 1e-30)
      ? (apq >= 0 ? Math.PI / 4 : -Math.PI / 4)
      : 0.5 * Math.atan2(2 * apq, app - aqq);
    const c = Math.cos(theta), s = Math.sin(theta);

    A[p * d + p] = c * c * app + 2 * c * s * apq + s * s * aqq;
    A[q * d + q] = s * s * app - 2 * c * s * apq + c * c * aqq;
    A[p * d + q] = 0; A[q * d + p] = 0;
    for (let i = 0; i < d; i++) {
      if (i !== p && i !== q) {
        const aip = A[i * d + p], aiq = A[i * d + q];
        const newAip = c * aip + s * aiq;
        const newAiq = -s * aip + c * aiq;
        A[i * d + p] = newAip; A[p * d + i] = newAip;
        A[i * d + q] = newAiq; A[q * d + i] = newAiq;
      }
      const vip = V[i * d + p], viq = V[i * d + q];
      V[i * d + p] = c * vip + s * viq;
      V[i * d + q] = -s * vip + c * viq;
    }
  }

  const eigvals = new Float64Array(d);
  for (let i = 0; i < d; i++) eigvals[i] = A[i * d + i];
  return { eigvals, V };
}

// Compute analytical whitening matrix W such that (X_c @ W) has cov ~ I.
// Uses eigendecomp cov = V diag(S) V^T -> W = V diag(1/sqrt(max(S,eps))) V^T.
function whiteningMatrix(Xc, n, d) {
  const cov = covariance(Xc, n, d);
  const { eigvals, V } = symJacobi(cov);
  const eps = 1e-6;
  const invSqrt = new Float64Array(d);
  for (let k = 0; k < d; k++) invSqrt[k] = 1 / Math.sqrt(Math.max(eigvals[k], eps));

  // W = V @ diag(invSqrt) @ V^T
  const W = new Float64Array(d * d);
  // First M = V @ diag(invSqrt), then W = M @ V^T
  const M = new Float64Array(d * d);
  for (let i = 0; i < d; i++) {
    for (let j = 0; j < d; j++) {
      M[i * d + j] = V[i * d + j] * invSqrt[j];
    }
  }
  for (let i = 0; i < d; i++) {
    for (let j = 0; j < d; j++) {
      let s = 0;
      for (let k = 0; k < d; k++) s += M[i * d + k] * V[j * d + k];
      W[i * d + j] = s;
    }
  }
  return W;
}

function applyLinear(X, n, d, W) {
  const Y = new Float64Array(n * d);
  for (let i = 0; i < n; i++) {
    const base = i * d;
    for (let j = 0; j < d; j++) {
      let s = 0;
      for (let k = 0; k < d; k++) s += X[base + k] * W[k * d + j];
      Y[base + j] = s;
    }
  }
  return Y;
}

// ---------- Weak SIGReg (closed-form whitening) ----------
// Returns { Y, trajectory, losses, info }.
// trajectory[0] is the centered input; trajectory[1] is the whitened output.
export function weakSigreg(X, n, d) {
  const { Xc, mean } = meanCenter(X, n, d);
  const W = whiteningMatrix(Xc, n, d);
  const Y = applyLinear(Xc, n, d, W);

  // Verify how close we got to identity.
  const covAfter = covariance(Y, n, d);
  const { eigvals: eAfter } = symJacobi(covAfter);
  let frob = 0;
  for (let i = 0; i < d; i++) {
    for (let j = 0; j < d; j++) {
      const e = (i === j) ? covAfter[i * d + j] - 1 : covAfter[i * d + j];
      frob += e * e;
    }
  }
  frob = Math.sqrt(frob);
  const sortedAfter = Array.from(eAfter).sort((a, b) => b - a);
  return {
    Y,
    trajectory: [Float64Array.from(Xc), Y],
    losses: [weakLoss(Xc, n, d), frob],
    info: {
      mean,
      kind: 'weak',
      eigenvaluesAfter: sortedAfter,
      isotropyAfter: sortedAfter[sortedAfter.length - 1] / (sortedAfter[0] + 1e-10),
    },
  };
}

function weakLoss(Xc, n, d) {
  const cov = covariance(Xc, n, d);
  let s = 0;
  for (let i = 0; i < d; i++) {
    for (let j = 0; j < d; j++) {
      const e = (i === j) ? cov[i * d + j] - 1 : cov[i * d + j];
      s += e * e;
    }
  }
  return Math.sqrt(s);
}

// ---------- Strong SIGReg (iterative gradient descent on CF loss) ----------
// Minimises sum_{s,k} |ECF_{s,k} - exp(-t_k^2/2)|^2 * exp(-t_k^2/2)
// with respect to X. Mirrors LeJEPA Algorithm 1.
// We initialize from the whitened (Weak-SIGReg) state so the optimizer spends
// its iterations on higher moments (kurtosis, etc.) instead of redundantly
// shrinking the 2nd moment. The trajectory still includes the original.
export function strongSigreg(X, n, d, opts) {
  opts = opts || {};
  const sketchDim = opts.sketchDim != null ? opts.sketchDim : 32;
  const nIter = opts.nIter != null ? opts.nIter : 30;
  const lr = opts.lr != null ? opts.lr : 0.4;
  const nPoints = opts.nPoints != null ? opts.nPoints : 17;
  const initFromWhiten = opts.initFromWhiten !== false;

  // Sketching matrix A: [d, sketchDim], Gaussian random with normalized cols.
  // Following the reference: A = randn(C, S) / ||A||_col.
  const A = new Float64Array(d * sketchDim);
  for (let i = 0; i < d * sketchDim; i++) A[i] = randn();
  for (let s = 0; s < sketchDim; s++) {
    let nrm = 0;
    for (let i = 0; i < d; i++) nrm += A[i * sketchDim + s] * A[i * sketchDim + s];
    nrm = Math.sqrt(nrm) + 1e-8;
    for (let i = 0; i < d; i++) A[i * sketchDim + s] /= nrm;
  }

  // Integration points t ~ linspace(-5, 5, nPoints); gaussian CF = exp(-t^2/2).
  const t = new Float64Array(nPoints);
  const gaussCF = new Float64Array(nPoints);
  for (let k = 0; k < nPoints; k++) {
    t[k] = -5 + (10 * k) / (nPoints - 1);
    gaussCF[k] = Math.exp(-0.5 * t[k] * t[k]);
  }

  // Working buffer: start from centered X (or from whitened X for faster
  // convergence on higher moments).
  const { Xc: X0, mean } = meanCenter(X, n, d);
  let Xcur;
  if (initFromWhiten) {
    const W = whiteningMatrix(X0, n, d);
    Xcur = applyLinear(X0, n, d, W);
  } else {
    Xcur = Float64Array.from(X0);
  }

  const trajectory = [Float64Array.from(X0)];  // include original at index 0
  const losses = [strongLoss(X0, A, n, d, sketchDim, t, gaussCF, nPoints)];
  const covEigsTrace = [eigenvalueTrace(X0, n, d)];
  // Also include the whitened initialization so the curve shows both stages.
  trajectory.push(Float64Array.from(Xcur));
  losses.push(strongLoss(Xcur, A, n, d, sketchDim, t, gaussCF, nPoints));
  covEigsTrace.push(eigenvalueTrace(Xcur, n, d));

  for (let it = 0; it < nIter; it++) {
    Xcur = sigregGradientStep(Xcur, A, n, d, sketchDim, t, gaussCF, nPoints, lr);
    // Re-center each step so the mean stays at zero (translation doesn't
    // change the CF; we don't want drift to inflate the loss).
    const { Xc } = meanCenter(Xcur, n, d);
    Xcur = Xc;
    trajectory.push(Float64Array.from(Xcur));
    losses.push(strongLoss(Xcur, A, n, d, sketchDim, t, gaussCF, nPoints));
    covEigsTrace.push(eigenvalueTrace(Xcur, n, d));
  }

  // Final analysis of the converged state.
  const covAfter = covariance(Xcur, n, d);
  const { eigvals: eAfter } = symJacobi(covAfter);
  const sortedAfter = Array.from(eAfter).sort((a, b) => b - a);

  return {
    Y: Xcur,
    trajectory,
    losses,
    info: {
      mean,
      kind: 'strong',
      sketchDim,
      nIter,
      eigenvaluesAfter: sortedAfter,
      isotropyAfter: sortedAfter[sortedAfter.length - 1] / (sortedAfter[0] + 1e-10),
      covEigsTrace,
    },
  };
}

function strongLoss(X, A, n, d, sketchDim, t, gaussCF, nPoints) {
  // proj = X @ A: [n, sketchDim]
  const proj = matmul(X, A, n, d, sketchDim);
  // ecf[s, k] = (1/n) * sum_i exp(i * proj[i, s] * t[k])
  let loss = 0;
  for (let s = 0; s < sketchDim; s++) {
    for (let k = 0; k < nPoints; k++) {
      let cr = 0, ci = 0;
      for (let i = 0; i < n; i++) {
        const z = proj[i * sketchDim + s] * t[k];
        cr += Math.cos(z);
        ci += Math.sin(z);
      }
      const er = cr / n - gaussCF[k];
      const ei = ci / n;
      loss += (er * er + ei * ei) * gaussCF[k];
    }
  }
  return loss;
}

function matmul(X, A, n, d, sketchDim) {
  const P = new Float64Array(n * sketchDim);
  for (let i = 0; i < n; i++) {
    const xb = i * d;
    for (let s = 0; s < sketchDim; s++) {
      let sum = 0;
      for (let c = 0; c < d; c++) sum += X[xb + c] * A[c * sketchDim + s];
      P[i * sketchDim + s] = sum;
    }
  }
  return P;
}

function sigregGradientStep(X, A, n, d, sketchDim, t, gaussCF, nPoints, lr) {
  const proj = matmul(X, A, n, d, sketchDim);
  // Precompute ecf and its per-(s,k) parts once.
  // ecfRe[s,k] = (1/n) sum_i cos(z), ecfIm[s,k] = (1/n) sum_i sin(z)
  const ecfRe = new Float64Array(sketchDim * nPoints);
  const ecfIm = new Float64Array(sketchDim * nPoints);
  for (let s = 0; s < sketchDim; s++) {
    for (let k = 0; k < nPoints; k++) {
      let cr = 0, ci = 0;
      for (let i = 0; i < n; i++) {
        const z = proj[i * sketchDim + s] * t[k];
        cr += Math.cos(z); ci += Math.sin(z);
      }
      ecfRe[s * nPoints + k] = cr / n;
      ecfIm[s * nPoints + k] = ci / n;
    }
  }

  // gradProj[j, s] = (2/n) sum_k t[k] * (-(Re(ecf)-gauss) * sin(z) + Im(ecf) * cos(z)) * gauss[k]
  const gradProj = new Float64Array(n * sketchDim);
  for (let j = 0; j < n; j++) {
    for (let s = 0; s < sketchDim; s++) {
      let g = 0;
      for (let k = 0; k < nPoints; k++) {
        const z = proj[j * sketchDim + s] * t[k];
        const er = ecfRe[s * nPoints + k] - gaussCF[k];
        const ei = ecfIm[s * nPoints + k];
        const w = gaussCF[k];
        g += t[k] * (-er * Math.sin(z) + ei * Math.cos(z)) * w;
      }
      gradProj[j * sketchDim + s] = (2 / n) * g;
    }
  }

  // gradX[j, c] = sum_s gradProj[j, s] * A[c, s]
  const gradX = new Float64Array(n * d);
  for (let j = 0; j < n; j++) {
    for (let c = 0; c < d; c++) {
      let s = 0;
      const cb = c * sketchDim;
      for (let sk = 0; sk < sketchDim; sk++) s += gradProj[j * sketchDim + sk] * A[cb + sk];
      gradX[j * d + c] = s;
    }
  }

  // X <- X - lr * gradX
  const Xnew = new Float64Array(n * d);
  for (let i = 0; i < n * d; i++) Xnew[i] = X[i] - lr * gradX[i];
  return Xnew;
}

function eigenvalueTrace(X, n, d) {
  const cov = covariance(X, n, d);
  const { eigvals } = symJacobi(cov);
  return Array.from(eigvals).sort((a, b) => b - a);
}

// ---------- Identity (no transform) ----------
export function identityTransform(X, n, d) {
  const Xc = meanCenter(X, n, d).Xc;
  return {
    Y: Float64Array.from(Xc),
    trajectory: [Float64Array.from(Xc)],
    losses: [0],
    info: { kind: 'identity', eigenvaluesAfter: eigenvalueTrace(Xc, n, d) },
  };
}

// ---------- Diagnostics on a centered matrix ----------
export function analyzeIsotropy(Xc, n, d) {
  const cov = covariance(Xc, n, d);
  const { eigvals } = symJacobi(cov);
  const sorted = Array.from(eigvals).sort((a, b) => b - a);
  let diag = 0, off = 0;
  for (let i = 0; i < d; i++) {
    for (let j = 0; j < d; j++) {
      const v = cov[i * d + j];
      if (i === j) diag += v * v; else off += v * v;
    }
  }
  return {
    eigenvalues: sorted,
    isotropy: sorted[sorted.length - 1] / (sorted[0] + 1e-10),
    offDiagonalFraction: off / (diag + off + 1e-10),
    trace: diag,
  };
}