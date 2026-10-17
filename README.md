# vis_SIGReg

Side-by-side visualizer for **[SIGReg](https://github.com/kreasof-ai/sigreg)**
(Sketched Isotropic Gaussian Regularization, from LeJEPA).

- **Left card** — text → transformer embedding → PCA projection (the "before").
- **Right card** — the same embeddings after a **SIGReg** transformation, projected
  into the **same PCA basis** so the points stay in a shared coordinate space.
- Click **Play animation** to morph the right card through the strong-SIGReg
  iteration trajectory and watch points converge toward isotropy.
- A live loss-vs-iteration plot and isotropy / eigenvalue diagnostics are
  updated in real time.

This project is a sibling of `E:\vis` (the single-space vector explorer) and
reuses its visual conventions (dark theme, transformers.js embeddings, PCA
projection, 2D + 3D rendering).

## Run

```bash
# from F:\projects\vis_SIGReg
npm start
# then open http://localhost:5174/
```

`server.js` is a tiny static server with the right MIME types for ES modules
and cross-origin isolation headers (so `transformers.js` WASM works in
Chrome / Edge / Firefox).

No build step. Everything ships as plain ES modules.

## Files

| File          | Purpose                                                    |
| ------------- | ---------------------------------------------------------- |
| `index.html`  | Split-screen layout with controls, two cards, stats, loss plot |
| `style.css`   | Dark theme, dual grid, animated highlighting               |
| `app.js`      | State, event wiring, 2D/3D drawing, animation, stats        |
| `embed.js`    | `@huggingface/transformers` wrapper (mean-pool, tokenizer cache) |
| `pca.js`      | Power-iteration PCA (top-k eigenpairs + projection helper) |
| `sigreg.js`   | Weak + Strong SIGReg pure-JS implementations               |
| `server.js`   | Tiny static HTTP server                                     |

## SIGReg math

This repo implements both variants from the kreasof-ai/sigreg README in pure JS.

**Weak SIGReg** (`weakSigreg`):
- Center the embeddings.
- Closed-form whitening via eigendecomp of the covariance matrix.
- Forces `Cov(X) ≈ I` (matches 2nd moment only).

**Strong SIGReg** (`strongSigreg`):
- Builds the LeJEPA Algorithm 1 sketch `A ∈ R^{C × S}` with normalized Gaussian columns.
- Iterates gradient descent on the empirical characteristic function loss
  `L = Σ_{s,k} |ECF_{s,k} − exp(−t_k²/2)|² · exp(−t_k²/2)`, where
  `ECF_{s,k} = (1/n) Σ_j exp(i · proj_{j,s} · t_k)`.
- Every iteration, points get re-centered so the mean stays at zero.
- The trajectory is returned so the UI can play back the convergence animation.
- For the right-card display and the loss-vs-iter plot we also project each
  trajectory step through the shared PCA basis computed from the original.

## Controls

- **Embedding model** — Xenova/all-MiniLM-L6-v2 (fast, 384d) / MPNet (768d) / BGE-small.
- **Context** — token cap on each input (shorter = faster).
- **Mode** — Weak / Strong / Off.
- **Sketch dim** — number of random projections for Strong SIGReg (default 32).
- **Iters** — gradient descent iterations for Strong SIGReg (default 40).
- **Dim** — 2D canvas or 3D three.js scene.
- **Play animation** — replays the Strong SIGReg trajectory on the right card.

## Caveats

- Strong-SIGReg gradient descent is implemented in pure JS — for `d ≈ 384` and
  `n ≈ 25` it completes a 40-iter run in a few hundred ms on a modern laptop.
- For very large `n` or `d`, run a smaller experiment first to keep iterations snappy.
- The visualizer uses *post-hoc* SIGReg — apply the loss to the already-embedded
  vectors. In real training, SIGReg is the *gradient signal* that nudges the
  encoder weights so the *embedding distribution* becomes isotropic. We expose
  the same objective so the user can see what "isotropic Gaussian" actually
  looks like in a 2D scatter.