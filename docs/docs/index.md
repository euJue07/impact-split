# impact-split documentation

`impact-split` is an EDA-first tree method for additive KPIs (profit, revenue, hours), built for business materiality rather than average purity.

Standard variance-based trees can overvalue tiny but "pure" segments and undervalue large-volume segments that drive real totals. `impact-split` addresses this with three formulas:

1. **Local sieve**: `delta = V_node * delta_pct` to route categories into Positive / Neutral / Negative relative to local volume.
2. **Gain metric**: `Gain(X_i) = |S_P|/k_P + |S_N|/k_N` to favor large outer-branch totals while penalizing high-cardinality slicing.
3. **Dual materiality stop**: branch-level positive/negative mass is checked against separate global pools (`V_global_P`, `V_global_N`) to stop globally irrelevant splits.

This creates a tree that stays focused on business-impactful structure instead of chasing mathematically neat but operationally small patterns.

## Read Paths

- **Concept + full origin story:** [README](../../README.md)
- **Mathematical deep dive + synthetic worked example:** [notebooks/1.0-jde-impact-split-explainer.ipynb](../../notebooks/1.0-jde-impact-split-explainer.ipynb)
- **Practical trace diagnostics on real data:** [notebooks/2.0-jde-supermarket-kaggle-trace.ipynb](../../notebooks/2.0-jde-supermarket-kaggle-trace.ipynb)
- **Environment setup + first run:** [getting-started.md](getting-started.md)

