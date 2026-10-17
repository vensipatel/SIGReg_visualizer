// embed.js — Transformers.js wrapper. Browser + Node compatible.
// Returns Float32Array of shape [n, d] with ._n and ._d metadata.
import { env, pipeline, Tensor } from '@huggingface/transformers';

env.allowRemoteModels = true;
env.allowLocalModels = false;
env.useBrowserCache = true;
if (typeof window === 'undefined') {
  env.useBrowserCache = false;
}

const cache = new Map();
const EMBED_OPTS = Object.freeze({ pooling: 'mean', normalize: false });
const TOKEN_CACHE_LIMIT = 8192;
const TOKEN_CACHE = new WeakMap();

function getTokenCache(extractor) {
  let c = TOKEN_CACHE.get(extractor);
  if (!c) { c = new Map(); TOKEN_CACHE.set(extractor, c); }
  return c;
}

export async function getExtractor(modelId, opts = {}) {
  if (cache.has(modelId)) return cache.get(modelId);
  const extractor = await pipeline('feature-extraction', modelId, { quantized: opts.quantized ?? true });
  cache.set(modelId, extractor);
  return extractor;
}

async function tokenizeBatched(extractor, texts, maxLength) {
  const tcache = getTokenCache(extractor);
  if (tcache.size > TOKEN_CACHE_LIMIT) {
    const toRemove = tcache.size - TOKEN_CACHE_LIMIT;
    let i = 0;
    for (const k of tcache.keys()) { if (i++ >= toRemove) break; tcache.delete(k); }
  }
  const idsList = new Array(texts.length);
  const maskList = new Array(texts.length);
  const misses = [];
  const missIdx = [];
  for (let i = 0; i < texts.length; i++) {
    const t = texts[i];
    const hit = tcache.get(t);
    if (hit && (maxLength == null || hit.ids.length <= maxLength)) {
      idsList[i] = hit.ids;
      maskList[i] = hit.mask;
    } else {
      missIdx.push(i);
      misses.push(t);
    }
  }
  if (misses.length) {
    const tokOpts = { padding: true, truncation: true };
    if (maxLength != null) tokOpts.max_length = maxLength;
    const out = await extractor.tokenizer(misses, tokOpts);
    const idsData = out.input_ids.data;
    const maskData = out.attention_mask.data;
    const B = misses.length;
    const S = out.input_ids.dims[1];
    for (let k = 0; k < B; k++) {
      const base = k * S;
      let realEnd = S;
      for (let j = S - 1; j >= 0; j--) {
        if (maskData[base + j]) { realEnd = j + 1; break; }
      }
      const ids = idsData.slice(base, base + realEnd);
      const mask = maskData.slice(base, base + realEnd);
      idsList[missIdx[k]] = ids;
      maskList[missIdx[k]] = mask;
      tcache.set(misses[k], { ids, mask });
    }
  }
  return { idsList, maskList };
}

const POOL_OUT = { data: null };
function meanPoolFast(lh, idsList, maskList) {
  const n = idsList.length;
  const [, allS, D] = lh.dims;
  const data = lh.data;
  if (!POOL_OUT.data || POOL_OUT.data.length !== n * D) {
    POOL_OUT.data = new Float32Array(n * D);
  }
  const out = POOL_OUT.data;
  out.fill(0);
  const D4 = D & ~3;
  for (let i = 0; i < n; i++) {
    const mask = maskList[i];
    const S = mask.length;
    let count = 0;
    for (let j = 0; j < S; j++) if (mask[j]) count++;
    if (count === 0) continue;
    const rowBase = i * allS * D;
    const dst = i * D;
    for (let j = 0; j < S; j++) {
      if (!mask[j]) continue;
      const offset = rowBase + j * D;
      for (let k = 0; k < D4; k += 4) {
        out[dst + k]     += data[offset + k];
        out[dst + k + 1] += data[offset + k + 1];
        out[dst + k + 2] += data[offset + k + 2];
        out[dst + k + 3] += data[offset + k + 3];
      }
      for (let k = D4; k < D; k++) out[dst + k] += data[offset + k];
    }
    const inv = 1 / count;
    for (let k = 0; k < D4; k += 4) {
      out[dst + k]     *= inv;
      out[dst + k + 1] *= inv;
      out[dst + k + 2] *= inv;
      out[dst + k + 3] *= inv;
    }
    for (let k = D4; k < D; k++) out[dst + k] *= inv;
  }
  return out;
}

export async function embed(extractor, texts, opts = {}) {
  const maxLength = opts.max_length ?? null;
  const n = texts.length;
  if (n === 0) { const e = new Float32Array(0); e._n = 0; e._d = 0; return e; }
  const { idsList, maskList } = await tokenizeBatched(extractor, texts, maxLength);

  let S = 0;
  for (let i = 0; i < n; i++) if (idsList[i].length > S) S = idsList[i].length;

  const ids = new BigInt64Array(n * S);
  const mask = new BigInt64Array(n * S);
  for (let i = 0; i < n; i++) {
    const a = idsList[i], b = maskList[i];
    const s = a.length;
    for (let j = 0; j < s; j++) {
      ids[i * S + j] = a[j];
      mask[i * S + j] = b[j];
    }
  }

  const inputs = {
    input_ids: new Tensor('int64', ids, [n, S]),
    attention_mask: new Tensor('int64', mask, [n, S]),
  };
  const result = await extractor.model(inputs);
  const lh = result.last_hidden_state ?? result.logits ?? result.token_embeddings;
  const out = meanPoolFast(lh, idsList, maskList);
  out._n = n;
  out._d = lh.dims[lh.dims.length - 1];
  return out;
}