# impact-split documentation

`impact-split` is an EDA-first tree method for additive KPIs (profit, revenue, hours), built for business materiality rather than average purity.

Standard variance-based trees can overvalue tiny but "pure" segments and undervalue large-volume segments that drive real totals. `impact-split` addresses this with four ideas, each derived on [The Math](math.md) page:

1. **Centered-excess routing**: categories route by `D_cat = S_cat − n_cat·ȳ_node`, how far they deviate from the node's expected share, so effect is never confounded with raw volume.
2. **A two-bar sieve**: a category routes to an outer branch only if its excess clears **both** a materiality bar (`delta_pct` of node excess volume) **and** a significance bar (a `noise_z·σ̂·√n` noise floor), so neither sub-material nor noise-only categories get routed.
3. **Gain metric**: `Gain = |S_P|/k_P + |S_N|/k_N` favors large outer-branch totals while dividing by the per-branch category count `k` to penalize high-cardinality slicing.
4. **Dual-pool stop**: branch-level positive and negative mass are each checked against their own global pool (`V_global_P`, `V_global_N`), so a large positive and a large negative cannot cancel into apparent immateriality.

This creates a tree that stays focused on business-impactful structure instead of chasing mathematically neat but operationally small patterns.

## Read Paths

- **The derivations behind every formula and constant:** [The Math](math.md)
- **How the algorithm was arrived at (including refuted variants and the missed floor bar):** [Story](story.md)
- **Environment setup + first run:** [Getting Started](getting-started.md)
- **Concept, real output, guarantees, and validation:** the repository [`README.md`](../../README.md)
- **Worked example and real-data trace:** the `notebooks/` folder in the repository (`1.0-…-explainer.ipynb` and `2.0-…-supermarket-kaggle-trace.ipynb`).

