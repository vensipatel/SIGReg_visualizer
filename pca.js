// pca.js - PCA via power iteration with deflation. Browser + Node compatible.
// Input: Float64Array (or Float32Array) of shape [n, d] with ._n and ._d
// Output: { points, components, mean, varianceExplained, rank, n, d }

function readInput(vectors) {
  if (vectors instanceof Float64Array || vectors instanceof Float32Array) {
    if (!vectors._n || !vectors._d) {
      throw new Error('pcaReduce: TypedArray needs ._n and ._d metadata.');
    }
    return {
      n: vectors._n,
      d: vectors._d,
      X: vectors instanceof Float64Array ? vectors : Float64Array.from(vectors),
    };
  }
  throw new Error('pcaReduce: unsupported input. Pass a TypedArray with ._n/.__d.');
}

function powerIterationTopK(A0, d, k) {
  const A = new Float64Array(A0);
  const eigvals = new Float64Array(k);
  const V = new Float64Array(d * k);
  const maxIter = 40;
  const tol = 1e-8;
  const v = new Float64Array(d);
  const y = new Float64Array(d);

  for (let j = 0; j < k; j++) {
    const seed = (j * 0.63255813 + 0.12345) % 1;
    for (let i = 0; i < d; i++) v[i] = ((i + 1) * seed + (i * 0.211)) % 1;

    let prevLambda = 0, lambda = 0;
    for (let iter = 0; iter < maxIter; iter++) {
      for (let i = 0; i < d; i++) {
        let s = 0;
        const rb = i * d;
        for (let l = 0; l < d; l++) s += A[rb + l] * v[l];
        y[i] = s;
      }
      lambda = 0;
      for (let i = 0; i < d; i++) lambda += v[i] * y[i];
      let norm = 0;
      for (let i = 0; i < d; i++) norm += y[i] * y[i];
      norm = Math.sqrt(norm) || 1;
      for (let i = 0; i < d; i++) v[i] = y[i] / norm;
      if (iter > 5 && Math.abs(lambda - prevLambda) < tol * (Math.abs(lambda) + 1)) break;
    }

    // Gram-Schmidt against previous eigenvectors
    for (let jp = 0; jp < j; jp++) {
      const vBase = jp * d;
      let dot = 0;
      for (let i = 0; i < d; i++) dot += v[i] * V[vBase + i];
      for (let i = 0; i < d; i++) v[i] -= dot * V[vBase + i];
    }
    let norm = 0;
    for (let i = 0; i < d; i++) norm += v[i] * v[i];
    norm = Math.sqrt(norm) || 1;
    for (let i = 0; i < d; i++) v[i] /= norm;

    for (let i = 0; i < d; i++) {
      let s = 0;
      const rb = i * d;
      for (let l = 0; l < d; l++) s += A[rb + l] * v[l];
      y[i] = s;
    }
    lambda = 0;
    for (let i = 0; i < d; i++) lambda += v[i] * y[i];

    const vBase = j * d;
    for (let i = 0; i < d; i++) V[vBase + i] = v[i];
    eigvals[j] = lambda;

    for (let i = 0; i < d; i++) {
      const vi = v[i];
      for (let l = 0; l < d; l++) A[i * d + l] -= lambda * vi * v[l];
    }
  }
  return { eigvals, V };
}

export function pcaReduce(vectors, k) {
  k = k || 2;
  const { n, d, X } = readInput(vectors);
  if (n < 2) throw new Error('pcaReduce: need at least 2 rows');

  const mean = new Float64Array(d);
  for (let i = 0; i < n; i++) {
    const base = i * d;
    for (let j = 0; j < d; j++) mean[j] += X[base + j];
  }
  for (let j = 0; j < d; j++) mean[j] /= n;
  for (let i = 0; i < n; i++) {
    const base = i * d;
    for (let j = 0; j < d; j++) X[base + j] -= mean[j];
  }

  let eigvals, components;
  if (n <= d) {
    const m = n;
    const Csmall = new Float64Array(m * m);
    const d4 = d & ~3;
    for (let i = 0; i < m; i++) {
      const xiBase = i * d;
      for (let j = i; j < m; j++) {
        const xjBase = j * d;
        let s0 = 0, s1 = 0, s2 = 0, s3 = 0;
        for (let l = 0; l < d4; l += 4) {
          s0 += X[xiBase + l] * X[xjBase + l];
          s1 += X[xiBase + l + 1] * X[xjBase + l + 1];
          s2 += X[xiBase + l + 2] * X[xjBase + l + 2];
          s3 += X[xiBase + l + 3] * X[xjBase + l + 3];
        }
        let s = s0 + s1 + s2 + s3;
        for (let l = d4; l < d; l++) s += X[xiBase + l] * X[xjBase + l];
        s /= (n - 1);
        Csmall[i * m + j] = s;
        Csmall[j * m + i] = s;
      }
    }
    const rank = Math.min(k, m);
    const { eigvals: eSmall, V: Vsmall } = powerIterationTopK(Csmall, m, rank);
    components = new Float32Array(rank * d);
    for (let j = 0; j < rank; j++) {
      const sigma = Math.sqrt(Math.max(eSmall[j] * (n - 1), 0));
      if (sigma < 1e-12) continue;
      const invSigma = 1 / sigma;
      const cBase = j * d;
      const m4 = m & ~3;
      for (let l = 0; l < d; l++) {
        let s0 = 0, s1 = 0, s2 = 0, s3 = 0;
        for (let r = 0; r < m4; r += 4) {
          s0 += X[r * d + l] * Vsmall[j * m + r];
          s1 += X[(r + 1) * d + l] * Vsmall[j * m + r + 1];
          s2 += X[(r + 2) * d + l] * Vsmall[j * m + r + 2];
          s3 += X[(r + 3) * d + l] * Vsmall[j * m + r + 3];
        }
        let s = s0 + s1 + s2 + s3;
        for (let r = m4; r < m; r++) s += X[r * d + l] * Vsmall[j * m + r];
        components[cBase + l] = s * invSigma;
      }
    }
    eigvals = eSmall;
  } else {
    const C = new Float64Array(d * d);
    const n4 = n & ~3;
    for (let i = 0; i < d; i++) {
      for (let j = i; j < d; j++) {
        let s0 = 0, s1 = 0, s2 = 0, s3 = 0;
        for (let r = 0; r < n4; r += 4) {
          s0 += X[r * d + i] * X[r * d + j];
          s1 += X[(r + 1) * d + i] * X[(r + 1) * d + j];
          s2 += X[(r + 2) * d + i] * X[(r + 2) * d + j];
          s3 += X[(r + 3) * d + i] * X[(r + 3) * d + j];
        }
        let s = s0 + s1 + s2 + s3;
        for (let r = n4; r < n; r++) s += X[r * d + i] * X[r * d + j];
        s /= (n - 1);
        C[i * d + j] = s;
        C[j * d + i] = s;
      }
    }
    const rank = Math.min(k, d);
    const { eigvals: eLarge, V: Vlarge } = powerIterationTopK(C, d, rank);
    components = new Float32Array(rank * d);
    for (let j = 0; j < rank; j++) {
      for (let i = 0; i < d; i++) components[j * d + i] = Vlarge[j * d + i];
    }
    eigvals = eLarge;
  }

  const rank = Math.min(k, n);
  const points = new Float32Array(n * rank);
  const d4 = d & ~3;
  for (let i = 0; i < n; i++) {
    const xBase = i * d, yBase = i * rank;
    for (let j = 0; j < rank; j++) {
      const cBase = j * d;
      let s0 = 0, s1 = 0, s2 = 0, s3 = 0;
      for (let l = 0; l < d4; l += 4) {
        s0 += X[xBase + l] * components[cBase + l];
        s1 += X[xBase + l + 1] * components[cBase + l + 1];
        s2 += X[xBase + l + 2] * components[cBase + l + 2];
        s3 += X[xBase + l + 3] * components[cBase + l + 3];
      }
      let s = s0 + s1 + s2 + s3;
      for (let l = d4; l < d; l++) s += X[xBase + l] * components[cBase + l];
      points[yBase + j] = s;
    }
  }

  let tr = 0;
  for (let i = 0; i < n * d; i++) tr += X[i] * X[i];
  const totalVar = tr / (n - 1) || 1;
  const varianceExplained = new Array(rank);
  for (let j = 0; j < rank; j++) {
    varianceExplained[j] = Math.max(eigvals[j], 0) / totalVar;
  }

  return {
    points,
    components,
    mean: new Float32Array(mean),
    varianceExplained,
    rank,
    n,
    d,
  };
}

export function pcaProject(newVectors, pca) {
  let m, d, X;
  if (newVectors instanceof Float64Array || newVectors instanceof Float32Array) {
    m = newVectors._n;
    d = newVectors._d;
    X = newVectors instanceof Float64Array ? new Float64Array(newVectors) : Float64Array.from(newVectors);
  } else {
    throw new Error('pcaProject: unsupported input.');
  }
  if (d !== pca.d) throw new Error(`pcaProject: dim mismatch ${d} vs ${pca.d}`);
  const rank = pca.rank;
  const mean = pca.mean;
  const components = pca.components;
  const d4 = d & ~3;
  const points = new Float32Array(m * rank);
  for (let i = 0; i < m; i++) {
    const xBase = i * d, yBase = i * rank;
    for (let j = 0; j < d4; j += 4) {
      X[xBase + j]     -= mean[j];
      X[xBase + j + 1] -= mean[j + 1];
      X[xBase + j + 2] -= mean[j + 2];
      X[xBase + j + 3] -= mean[j + 3];
    }
    for (let j = d4; j < d; j++) X[xBase + j] -= mean[j];
    for (let j = 0; j < rank; j++) {
      const cBase = j * d;
      let s0 = 0, s1 = 0, s2 = 0, s3 = 0;
      for (let l = 0; l < d4; l += 4) {
        s0 += X[xBase + l] * components[cBase + l];
        s1 += X[xBase + l + 1] * components[cBase + l + 1];
        s2 += X[xBase + l + 2] * components[cBase + l + 2];
        s3 += X[xBase + l + 3] * components[cBase + l + 3];
      }
      let s = s0 + s1 + s2 + s3;
      for (let l = d4; l < d; l++) s += X[xBase + l] * components[cBase + l];
      points[yBase + j] = s;
    }
  }
  return points;
}