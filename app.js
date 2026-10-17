// app.js — Dual-vector-space visualizer for SIGReg.
// Left card = text-to-vector (PCA projection of raw embeddings).
// Right card = SIGReg-transformed embeddings projected into the same PCA basis.
// "Play animation" morphs the right card through the strong-SIGReg iteration
// trajectory so users see points converging toward isotropy in real time.

import { getExtractor, embed } from './embed.js';
import { pcaReduce, pcaProject } from './pca.js';
import { weakSigreg, strongSigreg, identityTransform, analyzeIsotropy } from './sigreg.js';

// ---------- DOM ----------
const $ = (id) => document.getElementById(id);
const $model = $('model');
const $maxlen = $('maxlen');
const $modeSegBtns = document.querySelectorAll('#mode-seg .seg-btn');
const $sketch = $('sketch');
const $iters = $('iters');
const $dimSegBtns = document.querySelectorAll('#dim-seg .seg-btn');
const $play = $('play-anim');
const $status = $('status');

const $batchText = $('batch-text');
const $addBatch = $('add-batch');
const $clearDataset = $('clear-dataset');
const $loadExample = $('load-example');
const $fileInput = $('file-input');
const $datasetSummary = $('dataset-summary');

const $newText = $('new-text');
const $addNew = $('add-new');
const $newList = $('new-list');

const $canvasL = $('chart-left');
const $canvasR = $('chart-right');
const $threeL = $('three-left');
const $threeR = $('three-right');
const $tipL = $('tooltip-left');
const $tipR = $('tooltip-right');

const $leftMeta = $('left-meta');
const $rightMeta = $('right-meta');
const $modeBadge = $('mode-badge');
const $lossBadge = $('loss-badge');

const $statStrong = $('stat-strong');
const $statWeak = $('stat-weak');
const $statIso = $('stat-iso');
const $statOff = $('stat-off');
const $eigBars = $('eig-bars');
const $lossCanvas = $('loss-chart');

const $cardL = $('card-left');
const $cardR = $('card-right');

function maxLength() {
  const v = $maxlen.value;
  if (v === 'auto') return null;
  const n = parseInt(v, 10);
  return Number.isFinite(n) ? n : null;
}

// ---------- state ----------
const COLORS = {
  blue: '#7aa2ff',
  blueEdge: '#0f1115',
  orange: '#f7b955',
  orangeEdge: '#0f1115',
  pink: '#ff6b9b',
  axis: '#3a4257',
  grid: '#252b3a',
  gridStrong: '#2d3344',
};

const dataset = {
  items: [],          // { id, label }
  vectors: null,      // Float32Array(n*d) with ._n, ._d
  pca: null,          // shared PCA basis (computed from ORIGINAL)
  mean: null,         // empirical mean of ORIGINAL
  originalProj: null, // original projected via shared PCA, Float32Array(n*k)
  sigreg: null,       // { mode, result: { Y, trajectory, losses, info } }
  finalProj: null,    // final trajectory point projected via shared PCA
  originalXc: null,   // centered original as Float64Array (for analysis)
};

const newPointData = []; // { id, label, origProj, finalProj, inTrajectory }

let mode = 'weak';      // weak | strong | off
let dim = 2;            // 2 | 3
let anim = null;        // { startTs, frame, totalFrames, pointsAtT } | null
let isPlaying = false;

const state = {
  showLabels: false,   // always-on text labels next to every point
};

const EXAMPLE_DATASET = [
  'a photo of a red rose',
  'a bouquet of tulips',
  'a sunflower in a field',
  'a bowl of fresh pasta',
  'a slice of pepperoni pizza',
  'a cup of black coffee',
  'a thunderstorm in the mountains',
  'rain pattering on a window',
  'ocean waves crashing on rocks',
  'a violin playing a soft melody',
  'a piano concerto in a concert hall',
  'a guitar solo around a campfire',
  'a tabby cat sleeping on a couch',
  'a golden retriever playing fetch',
  'a hummingbird hovering at a flower',
  'a basketball game in a stadium',
  'a tennis match on a clay court',
  'a chess match in a quiet park',
  'a marathon runner crossing the finish line',
  'a programmer debugging a tricky bug',
  'an artist painting a sunset',
  'a chef plating a gourmet dish',
  'a scientist peering into a microscope',
  'a teacher writing on a chalkboard',
  'a child building a sandcastle',
];

// ---------- status helpers ----------
function setStatus(msg, cls = '') {
  $status.textContent = msg;
  $status.className = 'status ' + cls;
}

function randomId() { return Math.random().toString(36).slice(2, 10); }

function truncate(s, n) { return s.length > n ? s.slice(0, n - 1) + '…' : s; }

// ---------- extractors ----------
let activeModelPromise = null;
async function loadExtractor(modelId) {
  if (activeModelPromise) return activeModelPromise;
  setStatus(`Loading model ${modelId}...`, 'busy');
  activeModelPromise = getExtractor(modelId)
    .then((e) => { setStatus(`Model ${modelId} loaded.`, 'ok'); return e; })
    .catch((err) => { setStatus(`Model load failed: ${err.message}`, 'error'); activeModelPromise = null; throw err; });
  return activeModelPromise;
}

function parseLines(text) {
  return text.split(/\r?\n/).map((s) => s.trim()).filter(Boolean);
}

// ---------- embedding pipeline ----------
async function embedTexts(extractor, texts) {
  if (!texts.length) return null;
  return await embed(extractor, texts, { max_length: maxLength() });
}

// Shared PCA basis, computed from the ORIGINAL (mean-centered) vectors so both
// views use the same coordinate system and animations stay meaningful.
async function rebuildPipeline() {
  if (!dataset.items.length) {
    dataset.vectors = null;
    dataset.pca = null;
    dataset.mean = null;
    dataset.originalProj = null;
    dataset.originalXc = null;
    dataset.sigreg = null;
    dataset.finalProj = null;
    $play.disabled = true;
    draw();
    return;
  }
  $datasetSummary.textContent = `${dataset.items.length} items`;
  const extractor = await loadExtractor($model.value);
  setStatus(`Embedding ${dataset.items.length} items...`, 'busy');
  const labels = dataset.items.map((i) => i.label);
  const vectors = await embedTexts(extractor, labels);
  if (!vectors) return;
  dataset.vectors = vectors;

  const k = dim;
  setStatus(`Computing shared ${k}D PCA basis...`, 'busy');
  const pca = pcaReduce(vectors, k);
  dataset.pca = pca;
  dataset.mean = pca.mean;
  dataset.originalProj = pca.points;

  // Save centered original as Float64 for SIGReg input + analysis
  const n = vectors._n, d = vectors._d;
  const Xc = new Float64Array(n * d);
  for (let i = 0; i < n * d; i++) {
    Xc[i] = vectors[i] - dataset.mean[i % d];
  }
  dataset.originalXc = Xc;

  await runSigreg();
  await reprojectNewPoints();
  $play.disabled = !(dataset.sigreg && dataset.sigreg.trajectory && dataset.sigreg.trajectory.length > 1);
  draw();
  updateStats();
  updateLossTrail();
  setStatus('Done.', 'ok');
}

async function runSigreg() {
  if (!dataset.originalXc) { dataset.sigreg = null; return; }
  const n = dataset.originalXc.length / dataset.vectors._d;
  const d = dataset.vectors._d;

  setStatus(`Running ${mode} SIGReg...`, 'busy');
  let result;
  if (mode === 'weak') {
    result = weakSigreg(dataset.originalXc, n, d);
  } else if (mode === 'strong') {
    result = strongSigreg(dataset.originalXc, n, d, {
      sketchDim: parseInt($sketch.value, 10) || 32,
      nIter: parseInt($iters.value, 10) || 30,
      lr: 0.4,
    });
  } else {
    result = identityTransform(dataset.originalXc, n, d);
  }
  dataset.sigreg = result;

  // Project final Y into shared PCA basis
  const finalY = new Float64Array(result.Y);
  finalY._n = n; finalY._d = d;
  const finalProj = pcaProject(finalY, dataset.pca);
  dataset.finalProj = finalProj;

  // Project each trajectory step (we just need the final one for the static view;
  // animation re-projects on the fly).
  if (!newPointData.length) return;
  // Recompute new-point projections against new SIGReg result.
  await reprojectNewPoints();
}

// Re-embed any new points the user added and project them through both bases.
async function reprojectNewPoints() {
  if (!newPointData.length || !dataset.extractor || !dataset.pca || !dataset.sigreg) return;
  // Force a fresh embed (caller passes labels)
  const extractor = await loadExtractor($model.value);
  const labels = newPointData.map((p) => p.label);
  const vectors = await embedTexts(extractor, labels);
  if (!vectors) return;
  const k = dataset.pca.rank;
  // Original projection (through shared basis)
  const origProj = pcaProject(vectors, dataset.pca);
  // Centered original (Float64) — input to SIGReg
  const n = vectors._n, d = vectors._d;
  const Xc = new Float64Array(n * d);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < d; j++) Xc[i * d + j] = vectors[i * d + j] - dataset.mean[j];
  }
  // Apply SIGReg in lockstep. For animation simplicity, re-run on the union of
  // dataset + new points; for the static view we only need the final Y for new points.
  // To avoid redoing everything, we just run a few iterations of strongSigreg on the
  // new points alone using a fresh sketch (only for the new points' final positions).
  let newY;
  if (mode === 'weak') {
    newY = weakSigreg(Xc, n, d).Y;
  } else if (mode === 'strong') {
    newY = strongSigreg(Xc, n, d, {
      sketchDim: parseInt($sketch.value, 10) || 32,
      nIter: parseInt($iters.value, 10) || 30,
      lr: 0.4,
    }).Y;
  } else {
    newY = Xc;
  }
  const newY32 = new Float64Array(newY);
  newY32._n = n; newY32._d = d;
  const newProj = pcaProject(newY32, dataset.pca);
  for (let i = 0; i < newPointData.length; i++) {
    newPointData[i].origProj = { x: origProj[i * k], y: origProj[i * k + 1], z: k === 3 ? origProj[i * k + 2] : 0 };
    newPointData[i].finalProj = { x: newProj[i * k], y: newProj[i * k + 1], z: k === 3 ? newProj[i * k + 2] : 0 };
  }
}

async function addBatch() {
  const lines = parseLines($batchText.value);
  if (!lines.length) { setStatus('Nothing to add.', 'error'); return; }
  for (const label of lines) dataset.items.push({ id: randomId(), label });
  $batchText.value = '';
  await rebuildPipeline();
  renderNewChips();
}

async function addNewPoint() {
  const label = $newText.value.trim();
  if (!label) { setStatus('Enter a description for the new point.', 'error'); return; }
  if (!dataset.pca) { setStatus('Add a dataset first to compute PCA.', 'error'); return; }
  if (!dataset.extractor) await loadExtractor($model.value);
  setStatus('Embedding new point...', 'busy');
  const extractor = await loadExtractor($model.value);
  const vectors = await embedTexts(extractor, [label]);
  if (!vectors) return;
  const k = dataset.pca.rank;
  const origProj = pcaProject(vectors, dataset.pca);
  const n = vectors._n, d = vectors._d;
  const Xc = new Float64Array(n * d);
  for (let i = 0; i < n; i++) for (let j = 0; j < d; j++) Xc[i * d + j] = vectors[i * d + j] - dataset.mean[j];
  let newY;
  if (mode === 'weak') newY = weakSigreg(Xc, n, d).Y;
  else if (mode === 'strong') newY = strongSigreg(Xc, n, d, { sketchDim: parseInt($sketch.value, 10) || 32, nIter: parseInt($iters.value, 10) || 30, lr: 0.4 }).Y;
  else newY = Xc;
  const newY32 = new Float64Array(newY);
  newY32._n = n; newY32._d = d;
  const newProj = pcaProject(newY32, dataset.pca);
  const point = {
    id: randomId(),
    label,
    origProj: { x: origProj[0], y: origProj[1], z: k === 3 ? origProj[2] : 0 },
    finalProj: { x: newProj[0], y: newProj[1], z: k === 3 ? newProj[2] : 0 },
  };
  newPointData.push(point);
  $newText.value = '';
  renderNewChips();
  draw();
  setStatus(`Point added (orig ${point.origProj.x.toFixed(2)}, ${point.origProj.y.toFixed(2)} → after ${point.finalProj.x.toFixed(2)}, ${point.finalProj.y.toFixed(2)}).`, 'ok');
}

function clearAll() {
  dataset.items = [];
  dataset.vectors = null;
  dataset.pca = null;
  dataset.mean = null;
  dataset.originalProj = null;
  dataset.originalXc = null;
  dataset.sigreg = null;
  dataset.finalProj = null;
  newPointData.length = 0;
  renderNewChips();
  $datasetSummary.textContent = '0 items';
  draw();
  updateStats();
  updateLossTrail();
  setStatus('Cleared.', '');
}

function removeNewPoint(id) {
  const idx = newPointData.findIndex((p) => p.id === id);
  if (idx >= 0) newPointData.splice(idx, 1);
  renderNewChips();
  draw();
}

function renderNewChips() {
  $newList.innerHTML = '';
  for (const p of newPointData) {
    const chip = document.createElement('span');
    chip.className = 'chip new';
    chip.textContent = truncate(p.label, 28);
    const x = document.createElement('span');
    x.className = 'x'; x.textContent = '×'; x.title = 'Remove';
    x.onclick = () => removeNewPoint(p.id);
    chip.appendChild(x);
    $newList.appendChild(chip);
  }
}

async function loadExample() {
  dataset.items = EXAMPLE_DATASET.map((label) => ({ id: randomId(), label }));
  $batchText.value = '';
  await rebuildPipeline();
  renderNewChips();
}

// ---------- Drawing: 2D ----------
const ctxL = $canvasL.getContext('2d');
const ctxR = $canvasR.getContext('2d');

function setupCanvas(canvas) {
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth;
  const h = canvas.clientHeight || Math.round(w * 0.6);
  canvas.width = Math.round(w * dpr);
  canvas.height = Math.round(h * dpr);
  canvas.style.height = h + 'px';
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { w, h };
}

function collect2D(side) {
  const pts = [];
  const k = dataset.pca ? dataset.pca.rank : 2;
  if (side === 'left') {
    if (dataset.originalProj) {
      for (let i = 0; i < dataset.items.length; i++) {
        pts.push([dataset.originalProj[i * k], dataset.originalProj[i * k + 1]]);
      }
    }
  } else if (side === 'right') {
    // Right card spans both the "before" and the "after" positions so the
    // faded-before dots and the after dots + their connecting arrows are all
    // visible without clipping.
    if (dataset.originalProj) {
      for (let i = 0; i < dataset.items.length; i++) {
        pts.push([dataset.originalProj[i * k], dataset.originalProj[i * k + 1]]);
      }
    }
    if (dataset.finalProj) {
      for (let i = 0; i < dataset.items.length; i++) {
        pts.push([dataset.finalProj[i * k], dataset.finalProj[i * k + 1]]);
      }
    }
  }
  for (const p of newPointData) {
    if (side === 'left') {
      if (p.origProj) pts.push([p.origProj.x, p.origProj.y]);
    } else {
      if (p.origProj) pts.push([p.origProj.x, p.origProj.y]);
      if (p.finalProj) pts.push([p.finalProj.x, p.finalProj.y]);
    }
  }
  if (!pts.length) return { minX: -1, maxX: 1, minY: -1, maxY: 1 };
  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
  for (const [x, y] of pts) {
    if (x < minX) minX = x; if (x > maxX) maxX = x;
    if (y < minY) minY = y; if (y > maxY) maxY = y;
  }
  const padX = (maxX - minX) * 0.08 || 1;
  const padY = (maxY - minY) * 0.08 || 1;
  return { minX: minX - padX, maxX: maxX + padX, minY: minY - padY, maxY: maxY + padY };
}

function drawGrid(ctx, w, h, sx, sy, b, accent) {
  ctx.strokeStyle = COLORS.grid;
  ctx.lineWidth = 1;
  for (let i = 0; i <= 10; i++) {
    const x = (i / 10) * w;
    const y = (i / 10) * h;
    ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke();
  }
  ctx.strokeStyle = COLORS.axis;
  if (b.minX < 0 && b.maxX > 0) {
    const x0 = sx(0);
    ctx.beginPath(); ctx.moveTo(x0, 0); ctx.lineTo(x0, h); ctx.stroke();
  }
  if (b.minY < 0 && b.maxY > 0) {
    const y0 = sy(0);
    ctx.beginPath(); ctx.moveTo(0, y0); ctx.lineTo(w, y0); ctx.stroke();
  }
}

// Color a displacement arrow by magnitude: short = cool purple, long = hot pink.
// Returns an {r,g,b,a} object plus a CSS string.
function displacementColor(dist, refDist) {
  const norm = Math.min(1, dist / (refDist || 1));
  // lerp purple (138, 99, 240) -> pink (255, 107, 155)
  const r = Math.round(138 + (255 - 138) * norm);
  const g = Math.round(99 + (107 - 99) * norm);
  const b = Math.round(240 + (155 - 240) * norm);
  const a = 0.55 + 0.25 * norm;
  return { r, g, b, a, css: `rgba(${r}, ${g}, ${b}, ${a.toFixed(2)})` };
}

function drawArrow(ctx, ox, oy, tx, ty, color, headLen) {
  ctx.strokeStyle = color;
  ctx.lineWidth = 1.8;
  ctx.beginPath();
  ctx.moveTo(ox, oy);
  ctx.lineTo(tx, ty);
  ctx.stroke();
  const dx = tx - ox, dy = ty - oy;
  const dist = Math.sqrt(dx * dx + dy * dy);
  if (dist < 4 || !headLen) return;
  const angle = Math.atan2(dy, dx);
  const h = Math.min(headLen, dist * 0.45);
  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.moveTo(tx, ty);
  ctx.lineTo(tx - h * Math.cos(angle - Math.PI / 7), ty - h * Math.sin(angle - Math.PI / 7));
  ctx.lineTo(tx - h * Math.cos(angle + Math.PI / 7), ty - h * Math.sin(angle + Math.PI / 7));
  ctx.closePath();
  ctx.fill();
}

function draw2DOne(canvas, ctx, side) {
  const { w, h } = setupCanvas(canvas);
  ctx.clearRect(0, 0, w, h);
  const b = collect2D(side);
  const sx = (v) => ((v - b.minX) / (b.maxX - b.minX)) * w;
  const sy = (v) => h - ((v - b.minY) / (b.maxY - b.minY)) * h;
  drawGrid(ctx, w, h, sx, sy, b, side);

  const k = dataset.pca ? dataset.pca.rank : 2;
  const isRight = side === 'right';
  const dataProj = isRight ? dataset.finalProj : dataset.originalProj;
  const positions = [];

  // ---- Step 1: faded "before" dots in the RIGHT card ----
  // These mark where each point sat in the shared PCA basis *before* SIGReg.
  // They stay stationary during animation so the user can watch the orange
  // dot move away from its origin.
  if (isRight && dataset.originalProj) {
    ctx.fillStyle = 'rgba(122, 162, 255, 0.18)';
    ctx.strokeStyle = 'rgba(122, 162, 255, 0.55)';
    ctx.lineWidth = 1.2;
    for (let i = 0; i < dataset.items.length; i++) {
      const px = sx(dataset.originalProj[i * k]);
      const py = sy(dataset.originalProj[i * k + 1]);
      ctx.beginPath(); ctx.arc(px, py, 4.5, 0, Math.PI * 2); ctx.fill();
      ctx.beginPath(); ctx.arc(px, py, 4.5, 0, Math.PI * 2); ctx.stroke();
      positions.push({ x: px, y: py, label: dataset.items[i].label, kind: 'before', idx: i, side });
    }
  }

  // ---- Step 2: displacement arrows (only on right) ----
  if (isRight && dataset.originalProj && dataProj) {
    // Pick a reference distance = 25% of canvas diagonal so colour ramp is
    // consistent across window sizes.
    const refDist = Math.sqrt(w * w + h * h) * 0.25;
    for (let i = 0; i < dataset.items.length; i++) {
      const ox = sx(dataset.originalProj[i * k]);
      const oy = sy(dataset.originalProj[i * k + 1]);
      const tx = sx(dataProj[i * k]);
      const ty = sy(dataProj[i * k + 1]);
      const dist = Math.hypot(tx - ox, ty - oy);
      if (dist < 2) continue;
      const { css } = displacementColor(dist, refDist);
      drawArrow(ctx, ox, oy, tx, ty, css, 9);
    }
  }

  // ---- Step 3: main "after" dots ----
  if (dataProj) {
    const fill = isRight ? COLORS.orange : COLORS.blue;
    const stroke = isRight ? COLORS.orangeEdge : COLORS.blueEdge;
    ctx.fillStyle = fill;
    for (let i = 0; i < dataset.items.length; i++) {
      const px = sx(dataProj[i * k]);
      const py = sy(dataProj[i * k + 1]);
      ctx.beginPath(); ctx.arc(px, py, 5.5, 0, Math.PI * 2); ctx.fill();
    }
    ctx.strokeStyle = stroke;
    ctx.lineWidth = 1.5;
    for (let i = 0; i < dataset.items.length; i++) {
      const px = sx(dataProj[i * k]);
      const py = sy(dataProj[i * k + 1]);
      ctx.beginPath(); ctx.arc(px, py, 5.5, 0, Math.PI * 2); ctx.stroke();
    }
    for (let i = 0; i < dataset.items.length; i++) {
      const px = sx(dataProj[i * k]);
      const py = sy(dataProj[i * k + 1]);
      positions.push({ x: px, y: py, label: dataset.items[i].label, kind: 'after', idx: i, side });
    }
  }

  // ---- Step 4: new (pink) points with their own displacement arrow on right ----
  for (const p of newPointData) {
    const o = isRight ? p.finalProj : p.origProj;
    if (!o) continue;
    const px = sx(o.x), py = sy(o.y);

    if (isRight && p.origProj) {
      const ox = sx(p.origProj.x);
      const oy = sy(p.origProj.y);
      // Faded before dot
      ctx.fillStyle = 'rgba(255, 107, 155, 0.20)';
      ctx.strokeStyle = 'rgba(255, 107, 155, 0.65)';
      ctx.lineWidth = 1.4;
      ctx.beginPath(); ctx.arc(ox, oy, 4.5, 0, Math.PI * 2); ctx.fill();
      ctx.beginPath(); ctx.arc(ox, oy, 4.5, 0, Math.PI * 2); ctx.stroke();
      // Displacement arrow (always solid pink)
      const refDist = Math.sqrt(w * w + h * h) * 0.25;
      const dist = Math.hypot(px - ox, py - oy);
      const { css } = displacementColor(dist, refDist);
      drawArrow(ctx, ox, oy, px, py, css, 11);
      positions.push({ x: ox, y: oy, label: p.label, kind: 'before', idx: -1, side });
    }

    positions.push({ x: px, y: py, label: p.label, kind: 'new', idx: -1, side });

    // After dot with halo
    ctx.save();
    ctx.globalAlpha = 0.22;
    ctx.fillStyle = COLORS.pink;
    ctx.beginPath(); ctx.arc(px, py, 12, 0, Math.PI * 2); ctx.fill();
    ctx.restore();
    ctx.fillStyle = COLORS.pink;
    ctx.beginPath(); ctx.arc(px, py, 7, 0, Math.PI * 2); ctx.fill();
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = 2;
    ctx.stroke();
  }

  // ---- Step 5: optional always-on labels ----
  if (state.showLabels) {
    ctx.font = '600 11px ui-sans-serif, system-ui, sans-serif';
    ctx.textBaseline = 'middle';
    ctx.textAlign = 'left';
    for (let i = 0; i < dataset.items.length; i++) {
      if (!dataProj) continue;
      const px = sx(dataProj[i * k]);
      const py = sy(dataProj[i * k + 1]);
      const txt = truncate(dataset.items[i].label, 14);
      const tw = ctx.measureText(txt).width;
      const bx = px + 9, by = py - 9, bw = tw + 8, bh = 16;
      ctx.fillStyle = isRight ? 'rgba(36, 26, 8, 0.78)' : 'rgba(15, 17, 21, 0.78)';
      ctx.fillRect(bx, by, bw, bh);
      ctx.fillStyle = isRight ? COLORS.orange : COLORS.blue;
      ctx.fillText(txt, bx + 4, by + bh / 2 + 0.5);
    }
  }

  canvas._positions = positions;
}

function draw2D() {
  draw2DOne($canvasL, ctxL, 'left');
  draw2DOne($canvasR, ctxR, 'right');
}

// ---------- Drawing: 3D ----------
const threeStates = { left: null, right: null };

async function ensureThree(slot) {
  if (threeStates[slot]) return threeStates[slot];
  const THREE = await import('three');
  const { OrbitControls } = await import('three/addons/controls/OrbitControls.js');
  const { CSS2DRenderer } = await import('three/addons/renderers/CSS2DRenderer.js');
  const container = slot === 'left' ? $threeL : $threeR;

  const scene = new THREE.Scene();
  scene.background = null;
  const w = container.clientWidth || 600;
  const h = container.clientHeight || 360;

  const camera = new THREE.PerspectiveCamera(50, w / h, 0.01, 1000);
  camera.position.set(2.4, 2.0, 2.4);

  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setPixelRatio(window.devicePixelRatio || 1);
  renderer.setSize(w, h);
  container.appendChild(renderer.domElement);

  // HTML labels overlaid on the WebGL canvas
  const labelRenderer = new CSS2DRenderer();
  labelRenderer.setSize(w, h);
  labelRenderer.domElement.style.position = 'absolute';
  labelRenderer.domElement.style.top = '0';
  labelRenderer.domElement.style.left = '0';
  labelRenderer.domElement.style.width = '100%';
  labelRenderer.domElement.style.height = '100%';
  labelRenderer.domElement.style.pointerEvents = 'none';
  container.appendChild(labelRenderer.domElement);

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;

  // Lighting (so spheres look 3D, not flat)
  scene.add(new THREE.AmbientLight(0xffffff, 0.45));
  const key = new THREE.DirectionalLight(0xffffff, 0.85);
  key.position.set(4, 6, 4);
  scene.add(key);
  const rim = new THREE.DirectionalLight(0x88aaff, 0.4);
  rim.position.set(-3, -2, -4);
  scene.add(rim);

  // Reference frame: axes + ground grid
  scene.add(new THREE.AxesHelper(1.0));
  scene.add(new THREE.GridHelper(2.0, 10, 0x2a3142, 0x1d2230));

  let hint = container.querySelector('.three-hint');
  if (!hint) {
    hint = document.createElement('div');
    hint.className = 'three-hint';
    hint.textContent = 'drag to rotate · scroll to zoom · right-drag to pan';
    container.appendChild(hint);
  }

  // Reusable sphere geometries (one per size class)
  const sphereGeoSmall = new THREE.SphereGeometry(0.022, 16, 12);
  const sphereGeoLarge = new THREE.SphereGeometry(0.032, 18, 14);

  // Three groups keep the scene tidy and easy to dispose
  const afterGroup = new THREE.Group();   // main colored spheres
  const beforeGroup = new THREE.Group();  // faded "start" spheres (right card)
  const arrowsGroup = new THREE.Group();  // displacement arrows (right card)
  const labelsGroup = new THREE.Group();  // CSS2D labels
  scene.add(afterGroup, beforeGroup, arrowsGroup, labelsGroup);

  const onResize = () => {
    const nw = container.clientWidth, nh = container.clientHeight;
    if (!nw || !nh) return;
    camera.aspect = nw / nh; camera.updateProjectionMatrix();
    renderer.setSize(nw, nh);
    labelRenderer.setSize(nw, nh);
  };
  window.addEventListener('resize', onResize);

  const animate = () => {
    controls.update();
    renderer.render(scene, camera);
    labelRenderer.render(scene, camera);
    requestAnimationFrame(animate);
  };
  animate();

  threeStates[slot] = {
    THREE, scene, camera, renderer, labelRenderer, controls,
    afterGroup, beforeGroup, arrowsGroup, labelsGroup,
    sphereGeoSmall, sphereGeoLarge,
  };
  return threeStates[slot];
}

function disposeGroup(group) {
  while (group.children.length) {
    const c = group.children.pop();
    if (c.geometry && c.userData._disposable) c.geometry.dispose();
    if (c.material) {
      if (Array.isArray(c.material)) c.material.forEach((m) => m.dispose());
      else c.material.dispose();
    }
  }
}

function collect3D(side) {
  const pts = [];
  const k = dataset.pca ? dataset.pca.rank : 3;
  if (side === 'left') {
    if (dataset.originalProj) {
      for (let i = 0; i < dataset.items.length; i++) {
        pts.push([dataset.originalProj[i * k], dataset.originalProj[i * k + 1], dataset.originalProj[i * k + 2]]);
      }
    }
  } else if (side === 'right') {
    if (dataset.originalProj) {
      for (let i = 0; i < dataset.items.length; i++) {
        pts.push([dataset.originalProj[i * k], dataset.originalProj[i * k + 1], dataset.originalProj[i * k + 2]]);
      }
    }
    if (dataset.finalProj) {
      for (let i = 0; i < dataset.items.length; i++) {
        pts.push([dataset.finalProj[i * k], dataset.finalProj[i * k + 1], dataset.finalProj[i * k + 2]]);
      }
    }
  }
  for (const p of newPointData) {
    if (side === 'left') {
      if (p.origProj) pts.push([p.origProj.x, p.origProj.y, p.origProj.z]);
    } else {
      if (p.origProj) pts.push([p.origProj.x, p.origProj.y, p.origProj.z]);
      if (p.finalProj) pts.push([p.finalProj.x, p.finalProj.y, p.finalProj.z]);
    }
  }
  if (!pts.length) return { cx: 0, cy: 0, cz: 0, range: 1 };
  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity, minZ = Infinity, maxZ = -Infinity;
  for (const [x, y, z] of pts) {
    if (x < minX) minX = x; if (x > maxX) maxX = x;
    if (y < minY) minY = y; if (y > maxY) maxY = y;
    if (z < minZ) minZ = z; if (z > maxZ) maxZ = z;
  }
  return { cx: (minX + maxX) / 2, cy: (minY + maxY) / 2, cz: (minZ + maxZ) / 2, range: Math.max(maxX - minX, maxY - minY, maxZ - minZ) || 1 };
}

async function draw3DOne(slot) {
  const t = await ensureThree(slot);
  const { THREE, afterGroup, beforeGroup, arrowsGroup, labelsGroup,
          camera, controls, sphereGeoSmall, sphereGeoLarge } = t;
  disposeGroup(afterGroup);
  disposeGroup(beforeGroup);
  disposeGroup(arrowsGroup);
  disposeGroup(labelsGroup);

  const b = collect3D(slot);
  const norm = (v) => v / b.range;
  const cx = b.cx, cy = b.cy, cz = b.cz;
  const k = dataset.pca ? dataset.pca.rank : 3;
  const isRight = slot === 'right';
  const dataProj = isRight ? dataset.finalProj : dataset.originalProj;

  const HEX_BLUE = 0x7aa2ff;
  const HEX_ORANGE = 0xf7b955;
  const HEX_PINK = 0xff6b9b;

  // -- Faded "before" spheres (right card only) ---------------------------
  if (isRight && dataset.originalProj) {
    for (let i = 0; i < dataset.items.length; i++) {
      const mat = new THREE.MeshPhongMaterial({
        color: HEX_BLUE, transparent: true, opacity: 0.32,
        emissive: HEX_BLUE, emissiveIntensity: 0.15,
        shininess: 30,
      });
      const m = new THREE.Mesh(sphereGeoSmall, mat);
      m.position.set(
        norm(dataset.originalProj[i * k]     - cx),
        norm(dataset.originalProj[i * k + 1] - cy),
        norm(dataset.originalProj[i * k + 2] - cz),
      );
      beforeGroup.add(m);
    }
  }

  // -- Displacement arrows (right card only) -------------------------------
  if (isRight && dataset.originalProj && dataProj) {
    for (let i = 0; i < dataset.items.length; i++) {
      const ox = norm(dataset.originalProj[i * k]     - cx);
      const oy = norm(dataset.originalProj[i * k + 1] - cy);
      const oz = norm(dataset.originalProj[i * k + 2] - cz);
      const tx = norm(dataProj[i * k]     - cx);
      const ty = norm(dataProj[i * k + 1] - cy);
      const tz = norm(dataProj[i * k + 2] - cz);
      const dx = tx - ox, dy = ty - oy, dz = tz - oz;
      const len = Math.sqrt(dx * dx + dy * dy + dz * dz);
      if (len < 0.015) continue;
      const dir = new THREE.Vector3(dx, dy, dz).normalize();
      const origin = new THREE.Vector3(ox, oy, oz);
      // Color by magnitude: short=purple, long=pink
      const refLen = b.range * 0.25;
      const norm2 = Math.min(1, len / (refLen || 1));
      const r = Math.round((138 + (255 - 138) * norm2) * 1);
      const g = Math.round((99 + (107 - 99) * norm2) * 1);
      const bl = Math.round((240 + (155 - 240) * norm2) * 1);
      const color = (r << 16) | (g << 8) | bl;
      const headLen = Math.min(0.06, len * 0.3);
      const arrow = new THREE.ArrowHelper(dir, origin, len, color, headLen, headLen * 0.7);
      arrowsGroup.add(arrow);
    }
  }

  // -- Main "after" spheres ------------------------------------------------
  if (dataProj) {
    for (let i = 0; i < dataset.items.length; i++) {
      const colorHex = isRight ? HEX_ORANGE : HEX_BLUE;
      const mat = new THREE.MeshPhongMaterial({
        color: colorHex,
        emissive: colorHex, emissiveIntensity: 0.25,
        shininess: 80,
      });
      const m = new THREE.Mesh(sphereGeoSmall, mat);
      m.position.set(
        norm(dataProj[i * k]     - cx),
        norm(dataProj[i * k + 1] - cy),
        norm(dataProj[i * k + 2] - cz),
      );
      m.userData.itemIndex = i;
      m.userData.side = slot;
      afterGroup.add(m);

      if (state.showLabels) {
        const div = document.createElement('div');
        div.className = 'three-label';
        div.textContent = truncate(dataset.items[i].label, 14);
        div.style.color = isRight ? COLORS.orange : COLORS.blue;
        const obj = new CSS2DObject(div);
        obj.position.copy(m.position);
        labelsGroup.add(obj);
      }
    }
  }

  // -- New (pink) user points with their own displacement arrows ----------
  if (newPointData.length) {
    for (let i = 0; i < newPointData.length; i++) {
      const o = newPointData[i].origProj;
      const a = newPointData[i].finalProj;
      const target = isRight ? a : o;
      if (!target) continue;
      const mat = new THREE.MeshPhongMaterial({
        color: HEX_PINK, emissive: HEX_PINK, emissiveIntensity: 0.3, shininess: 90,
      });
      const m = new THREE.Mesh(sphereGeoLarge, mat);
      m.position.set(norm(target.x - cx), norm(target.y - cy), norm(target.z - cz));
      afterGroup.add(m);

      if (isRight && o && a) {
        const ox = norm(o.x - cx), oy = norm(o.y - cy), oz = norm(o.z - cz);
        const tx = m.position.x, ty = m.position.y, tz = m.position.z;
        const dx = tx - ox, dy = ty - oy, dz = tz - oz;
        const len = Math.sqrt(dx * dx + dy * dy + dz * dz);
        if (len > 0.01) {
          const dir = new THREE.Vector3(dx, dy, dz).normalize();
          const arrow = new THREE.ArrowHelper(dir, new THREE.Vector3(ox, oy, oz), len, HEX_PINK, Math.min(0.06, len * 0.35), 0.04);
          arrowsGroup.add(arrow);
        }
        // Faded pink before-sphere
        const bmat = new THREE.MeshPhongMaterial({
          color: HEX_PINK, transparent: true, opacity: 0.35,
          emissive: HEX_PINK, emissiveIntensity: 0.1, shininess: 30,
        });
        const bm = new THREE.Mesh(sphereGeoSmall, bmat);
        bm.position.set(ox, oy, oz);
        beforeGroup.add(bm);
      }

      if (state.showLabels) {
        const div = document.createElement('div');
        div.className = 'three-label pink';
        div.textContent = truncate(newPointData[i].label, 14);
        const obj = new CSS2DObject(div);
        obj.position.copy(m.position);
        labelsGroup.add(obj);
      }
    }
  }

  const target = new THREE.Vector3(0, 0, 0);
  const d = b.range * 2.4;
  camera.position.set(d, d * 0.8, d);
  camera.lookAt(target);
  controls.target.copy(target); controls.update();
}

async function draw3D() {
  await draw3DOne('left');
  await draw3DOne('right');
}

// ---------- Tooltip ----------
function bindTooltip(canvas, tip) {
  canvas.addEventListener('mousemove', (e) => {
    if (dim !== 2) return;
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left, my = e.clientY - rect.top;
    const positions = canvas._positions || [];
    let best = null, bestD = 14;
    for (const p of positions) {
      const dx = mx - p.x, dy = my - p.y;
      const d = Math.sqrt(dx * dx + dy * dy);
      if (d < bestD) { bestD = d; best = p; }
    }
    if (best) {
      tip.hidden = false;
      let prefix = '';
      if (best.kind === 'before') prefix = '○ start  ';
      else if (best.kind === 'after') prefix = '● after SIGReg  ';
      else if (best.kind === 'new') prefix = '★ ';
      tip.textContent = prefix + best.label;
      tip.style.left = (mx + 12) + 'px';
      tip.style.top = (my + 12) + 'px';
    } else {
      tip.hidden = true;
    }
  });
  canvas.addEventListener('mouseleave', () => { tip.hidden = true; });
}
bindTooltip($canvasL, $tipL);
bindTooltip($canvasR, $tipR);

// ---------- Dispatch ----------
function draw() {
  if (dim === 2) draw2D();
  else draw3D();
  updateMetas();
}

function updateMetas() {
  if (!dataset.pca) {
    $leftMeta.textContent = '—';
    $rightMeta.textContent = '—';
    return;
  }
  const ve = dataset.pca.varianceExplained;
  const veStr = ve.map((v, i) => `PC${i + 1} ${(v * 100).toFixed(1)}%`).join(' · ');
  $leftMeta.textContent = veStr;
  const count = dataset.sigreg && dataset.sigreg.trajectory ? dataset.sigreg.trajectory.length - 1 : 0;
  const modeLabel = mode === 'weak' ? 'Weak' : mode === 'strong' ? 'Strong' : 'Off';
  $rightMeta.textContent = `${modeLabel} SIGReg · ${count} iter${count === 1 ? '' : 's'}`;
  $modeBadge.textContent = `${modeLabel} SIGReg`;
  $modeBadge.style.color = mode === 'weak' ? COLORS.blue : mode === 'strong' ? COLORS.orange : COLORS.muted || 'var(--muted)';
  if (dataset.sigreg && dataset.sigreg.losses && dataset.sigreg.losses.length) {
    const last = dataset.sigreg.losses[dataset.sigreg.losses.length - 1];
    $lossBadge.textContent = `loss ${last.toFixed(3)}`;
  } else {
    $lossBadge.textContent = '—';
  }
}

// ---------- Stats panel ----------
function updateStats() {
  if (!dataset.originalXc) {
    $statStrong.textContent = '—'; $statWeak.textContent = '—';
    $statIso.textContent = '—'; $statOff.textContent = '—';
    $eigBars.innerHTML = '';
    return;
  }
  const n = dataset.vectors._n, d = dataset.vectors._d;
  const before = analyzeIsotropy(dataset.originalXc, n, d);
  $statIso.textContent = before.isotropy.toFixed(4);
  $statOff.textContent = (before.offDiagonalFraction * 100).toFixed(2) + '%';

  let weakLoss = null, strongLoss = null, eigAfter = before.eigenvalues;
  if (dataset.sigreg) {
    if (mode === 'weak') {
      weakLoss = dataset.sigreg.losses[dataset.sigreg.losses.length - 1];
      strongLoss = computeStrongLossFor(dataset.sigreg.Y, n, d);
      eigAfter = dataset.sigreg.info.eigenvaluesAfter;
    } else if (mode === 'strong') {
      strongLoss = dataset.sigreg.losses[dataset.sigreg.losses.length - 1];
      weakLoss = computeWeakLossFor(dataset.sigreg.Y, n, d);
      eigAfter = dataset.sigreg.info.eigenvaluesAfter;
    } else {
      weakLoss = computeWeakLossFor(dataset.sigreg.Y, n, d);
      strongLoss = computeStrongLossFor(dataset.sigreg.Y, n, d);
      eigAfter = before.eigenvalues;
    }
  }
  $statWeak.textContent = weakLoss != null ? weakLoss.toFixed(4) : '—';
  $statStrong.textContent = strongLoss != null ? strongLoss.toFixed(4) : '—';

  // Eigenvalue bars: top 8 sorted desc. Color green if all ~1 (target Identity)
  const top = eigAfter.slice(0, 8);
  const maxV = Math.max(...top, 1e-6);
  const isIdentity = top.every((v) => Math.abs(v - 1) < 0.15);
  $eigBars.innerHTML = '';
  top.forEach((v) => {
    const bar = document.createElement('div');
    bar.className = 'eig-bar' + (isIdentity ? ' id' : '');
    bar.style.height = (Math.max(2, (v / maxV) * 60)) + 'px';
    bar.title = v.toFixed(4);
    $eigBars.appendChild(bar);
  });
}

// Standalone helpers for displaying the "other" loss even when in a different mode.
function computeWeakLossFor(Xc, n, d) {
  const cov = new Float64Array(d * d);
  for (let i = 0; i < d; i++) for (let j = i; j < d; j++) {
    let s = 0; for (let k = 0; k < n; k++) s += Xc[k * d + i] * Xc[k * d + j];
    s /= (n - 1); cov[i * d + j] = s; cov[j * d + i] = s;
  }
  let s = 0;
  for (let i = 0; i < d; i++) for (let j = 0; j < d; j++) {
    const e = (i === j) ? cov[i * d + j] - 1 : cov[i * d + j];
    s += e * e;
  }
  return Math.sqrt(s);
}

function computeStrongLossFor(Xc, n, d) {
  const sketchDim = Math.min(32, d);
  const A = new Float64Array(d * sketchDim);
  for (let i = 0; i < A.length; i++) A[i] = (Math.random() + Math.random() + Math.random() + Math.random() + Math.random() + Math.random() - 3) / Math.sqrt(0.5);
  for (let s = 0; s < sketchDim; s++) {
    let nrm = 0; for (let i = 0; i < d; i++) nrm += A[i * sketchDim + s] * A[i * sketchDim + s];
    nrm = Math.sqrt(nrm) + 1e-8; for (let i = 0; i < d; i++) A[i * sketchDim + s] /= nrm;
  }
  const t = new Float64Array(17); const gaussCF = new Float64Array(17);
  for (let k = 0; k < 17; k++) { t[k] = -5 + (10 * k) / 16; gaussCF[k] = Math.exp(-0.5 * t[k] * t[k]); }
  const proj = new Float64Array(n * sketchDim);
  for (let i = 0; i < n; i++) for (let s = 0; s < sketchDim; s++) {
    let sum = 0; for (let c = 0; c < d; c++) sum += Xc[i * d + c] * A[c * sketchDim + s];
    proj[i * sketchDim + s] = sum;
  }
  let loss = 0;
  for (let s = 0; s < sketchDim; s++) for (let k = 0; k < 17; k++) {
    let cr = 0, ci = 0; for (let i = 0; i < n; i++) { const z = proj[i * sketchDim + s] * t[k]; cr += Math.cos(z); ci += Math.sin(z); }
    const er = cr / n - gaussCF[k]; const ei = ci / n;
    loss += (er * er + ei * ei) * gaussCF[k];
  }
  return loss;
}

// ---------- Loss trail (canvas) ----------
function updateLossTrail() {
  const ctx = $lossCanvas.getContext('2d');
  const dpr = window.devicePixelRatio || 1;
  const w = $lossCanvas.clientWidth;
  const h = $lossCanvas.clientHeight || 160;
  $lossCanvas.width = Math.round(w * dpr);
  $lossCanvas.height = Math.round(h * dpr);
  $lossCanvas.style.height = h + 'px';
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);

  ctx.fillStyle = COLORS.surface2 || '#1d2230';
  ctx.fillRect(0, 0, w, h);

  const losses = dataset.sigreg ? dataset.sigreg.losses : [];
  if (losses.length < 2) {
    ctx.fillStyle = COLORS.muted || '#8b94a7';
    ctx.font = '12px ui-sans-serif';
    ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    ctx.fillText(mode === 'weak' ? 'Weak SIGReg has a closed-form solution — single step.' : 'Run SIGReg to see the loss curve.', w / 2, h / 2);
    return;
  }
  let maxL = 0, minL = Infinity;
  for (const l of losses) { if (l > maxL) maxL = l; if (l < minL) minL = l; }
  if (maxL === minL) maxL = minL + 1;
  const sx = (i) => (i / (losses.length - 1)) * w;
  const sy = (v) => h - 8 - ((v - minL) / (maxL - minL)) * (h - 16);

  // grid
  ctx.strokeStyle = COLORS.grid;
  ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const y = (i / 4) * h;
    ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke();
  }

  // line
  ctx.strokeStyle = COLORS.orange;
  ctx.lineWidth = 2;
  ctx.beginPath();
  for (let i = 0; i < losses.length; i++) {
    const x = sx(i), y = sy(losses[i]);
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  }
  ctx.stroke();

  // dots
  ctx.fillStyle = COLORS.orange;
  for (let i = 0; i < losses.length; i++) {
    const x = sx(i), y = sy(losses[i]);
    ctx.beginPath(); ctx.arc(x, y, 2.5, 0, Math.PI * 2); ctx.fill();
  }

  ctx.fillStyle = COLORS.muted || '#8b94a7';
  ctx.font = '11px ui-sans-serif';
  ctx.textAlign = 'left'; ctx.textBaseline = 'top';
  ctx.fillText(`start ${losses[0].toFixed(3)}`, 8, 6);
  ctx.textAlign = 'right'; ctx.textBaseline = 'bottom';
  ctx.fillText(`end ${losses[losses.length - 1].toFixed(3)}`, w - 8, h - 4);
  ctx.textAlign = 'center'; ctx.textBaseline = 'top';
  ctx.fillText(`iter ${losses.length - 1}`, w / 2, 6);
}

// ---------- Animation (Strong SIGReg morph) ----------
async function startPlay() {
  if (!dataset.sigreg || !dataset.sigreg.trajectory || dataset.sigreg.trajectory.length < 2) return;
  if (isPlaying) return;
  isPlaying = true;
  $cardR.classList.add('playing');
  $play.textContent = 'Stop';

  const traj = dataset.sigreg.trajectory;
  const losses = dataset.sigreg.losses || [];
  const n = traj[0].length / dataset.vectors._d;
  const d = dataset.vectors._d;

  const durationMs = 4000;
  const t0 = performance.now();

  const tick = async () => {
    if (!isPlaying) return;
    const elapsed = performance.now() - t0;
    const u = Math.min(1, elapsed / durationMs);
    const seg = u * (traj.length - 1);
    const idx = Math.min(traj.length - 2, Math.floor(seg));
    const frac = seg - idx;

    // Project the interpolated X through shared PCA basis.
    const X0 = traj[idx], X1 = traj[idx + 1];
    const interp = new Float64Array(n * d);
    for (let i = 0; i < n * d; i++) interp[i] = X0[i] * (1 - frac) + X1[i] * frac;
    interp._n = n; interp._d = d;
    const proj = pcaProject(interp, dataset.pca);

    // Replace finalProj temporarily, render, then restore.
    const saved = dataset.finalProj;
    dataset.finalProj = proj;
    if (dim === 2) draw2D();
    else await draw3DOne('right');
    dataset.finalProj = saved;

    // Update loss indicator
    const liveLoss = losses[idx] * (1 - frac) + losses[idx + 1] * frac;
    $lossBadge.textContent = `loss ${liveLoss.toFixed(3)}`;

    if (u < 1) {
      requestAnimationFrame(tick);
    } else {
      isPlaying = false;
      $cardR.classList.remove('playing');
      $play.textContent = 'Play animation';
      draw();
      updateStats();
      updateLossTrail();
    }
  };
  requestAnimationFrame(tick);
}

function stopPlay() {
  isPlaying = false;
  $cardR.classList.remove('playing');
  $play.textContent = 'Play animation';
  draw();
  updateStats();
  updateLossTrail();
}

// ---------- Wiring ----------
$addBatch.addEventListener('click', () => addBatch());
$addNew.addEventListener('click', () => addNewPoint());
$clearDataset.addEventListener('click', () => clearAll());
$loadExample.addEventListener('click', () => loadExample());
$play.addEventListener('click', () => { if (isPlaying) stopPlay(); else startPlay(); });

$modeSegBtns.forEach((btn) => {
  btn.addEventListener('click', async () => {
    mode = btn.dataset.mode;
    $modeSegBtns.forEach((b) => b.classList.toggle('active', b === btn));
    if (dataset.originalXc) await rebuildPipeline();
    updateStats();
    updateLossTrail();
  });
});

$dimSegBtns.forEach((btn) => {
  btn.addEventListener('click', async () => {
    const d = parseInt(btn.dataset.dim, 10);
    if (d === dim) return;
    dim = d;
    $dimSegBtns.forEach((b) => b.classList.toggle('active', b === btn));
    if (dim === 2) {
      $threeL.hidden = true; $threeR.hidden = true;
      $threeL.style.display = ''; $threeR.style.display = '';
      $canvasL.hidden = false; $canvasR.hidden = false;
    } else {
      $canvasL.hidden = true; $canvasR.hidden = true;
      $threeL.hidden = false; $threeR.hidden = false;
      $threeL.style.display = 'block'; $threeR.style.display = 'block';
    }
    if (dataset.items.length) {
      await rebuildPipeline();
    } else {
      draw();
    }
  });
});

const $labelsSegBtns = document.querySelectorAll('#labels-seg .seg-btn');
$labelsSegBtns.forEach((btn) => {
  btn.addEventListener('click', () => {
    state.showLabels = btn.dataset.show === 'true';
    $labelsSegBtns.forEach((b) => b.classList.toggle('active', b === btn));
    draw();
  });
});

$sketch.addEventListener('change', async () => { if (dataset.originalXc && mode === 'strong') await rebuildPipeline(); });
$iters.addEventListener('change', async () => { if (dataset.originalXc && mode === 'strong') await rebuildPipeline(); });

$model.addEventListener('change', async () => {
  // need to reset extractor
  await rebuildPipeline();
});

$maxlen.addEventListener('change', async () => {
  if (dataset.items.length) await rebuildPipeline();
});

$batchText.addEventListener('keydown', (e) => {
  if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') { e.preventDefault(); addBatch(); }
});
$newText.addEventListener('keydown', (e) => {
  if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') { e.preventDefault(); addNewPoint(); }
});

$fileInput.addEventListener('change', async (e) => {
  const files = Array.from(e.target.files || []);
  if (!files.length) return;
  setStatus(`Reading ${files.length} file(s)...`, 'busy');
  const all = [];
  for (const f of files) {
    const text = await f.text();
    for (const line of text.split(/\r?\n/)) {
      const t = line.trim();
      if (t) all.push(t);
    }
  }
  if (!all.length) { setStatus('No text found in files.', 'error'); return; }
  for (const label of all) dataset.items.push({ id: randomId(), label });
  $fileInput.value = '';
  await rebuildPipeline();
  renderNewChips();
});

// initial
draw();
updateStats();
updateLossTrail();
setStatus('Ready. Add some data or click "Try example".', '');

// expose for debugging
window.__vis = { dataset, newPointData, mode, dim };