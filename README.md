# impact-split

[![CI](https://github.com/euJue07/impact-split/actions/workflows/ci.yml/badge.svg)](https://github.com/euJue07/impact-split/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/euJue07/impact-split/blob/main/LICENSE)

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" alt="Cookiecutter Data Science project template" />
</a>

**Documentation:** [docs/getting-started](https://github.com/euJue07/impact-split/blob/main/docs/docs/getting-started.md) · **Repository:** [github.com/euJue07/impact-split](https://github.com/euJue07/impact-split) · **Issues:** [github.com/euJue07/impact-split/issues](https://github.com/euJue07/impact-split/issues)

Contributions and security reports: [CONTRIBUTING.md](CONTRIBUTING.md) · [SECURITY.md](SECURITY.md)

A tree-based approach to determine the driver of a KPI. The installable Python package is **`impact_split`** (`import impact_split` / `from impact_split import ImpactSplitter`).

## What is impact-split?

`impact-split` is an ML-driven EDA approach for additive KPIs (extensive metrics), such as:

- Total Revenue
- Total Hours Watched
- Total Profit/Loss

Unlike standard decision trees that optimize for variance reduction (often favoring "pure" average-based segments), impact-split prioritizes **material business impact** by focusing on segment-level totals instead of average purity alone.

## Repository layout

| Path | Purpose |
|------|---------|
| [`impact_split/`](impact_split/) | Library source: [`splitter.py`](impact_split/splitter.py) (`ImpactSplitter`) plus [`viz/`](impact_split/viz/) renderers — [`data.py`](impact_split/viz/data.py) (`to_dict` payload builder), [`text.py`](impact_split/viz/text.py) (`summary()`), [`static.py`](impact_split/viz/static.py) (tornado + icicle matplotlib figures), [`html.py`](impact_split/viz/html.py) (self-contained HTML report) — and [`py.typed`](impact_split/py.typed) |
| [`tests/`](tests/) | Pytest suite |
| [`docs/`](docs/) | MkDocs site ([local build](docs/README.md)) |
| [`notebooks/`](notebooks/) | Explainer and trace walkthrough notebooks |
| [`pyproject.toml`](pyproject.toml) | Package metadata and tool configuration |

## Core Idea

The algorithm builds a ternary tree (Positive / Neutral / Negative) over categorical or pre-binned features and uses:

- centered-excess routing (`D_cat = S_cat - n_cat * mean(y_node)`) so categories are judged by how far they deviate from the node's expected share, never by raw volume,
- a two-part local sieve: a volume-relative threshold (`delta_pct`) **and** a per-category noise floor (`noise_z`) so neither sub-material nor statistically insignificant categories get routed,
- a gain metric that emphasizes outer-branch impact while penalizing high-cardinality noise,
- a global stopping threshold (`min_global_impact_pct`) to stop splitting low-materiality nodes,
- an interaction-order depth cap (`max_depth` counts distinct-feature transitions; same-feature refinements are free),
- guardrails that skip candidate splits which do not partition rows (a feature is constant on the current slice, or Act I routes every category to the same branch), avoiding redundant depth without new information.

## Story Behind the Math

Most decision-tree algorithms were designed to minimize variance and isolate segments with similar average values. In business work, averages can hide what matters most: total impact.

Example: a tiny segment with 2 churn events at -$5,000 each can look "purer" than a segment with 10,000 churn events at -$40 each. But the second segment carries far more total business weight.

`impact-split` was designed to solve this exact mismatch by optimizing for additive totals.

Notation (used across all acts): $y_i$ is the row-level target value for row $i$; $V_{node}=\sum_{i \in node}|y_i|$ is node absolute volume; $S_{cat}=\sum_{i \in cat} y_i$ is the raw sum for a category inside the current node; $n_{cat}$ is that category's row count; $S_P, S_N$ are the current node's positive/negative outer-branch sums; and $k_P, k_N$ are the number of categories routed to each outer branch.

### Act I: The Local Sieve (centered excess + noise floor)

**Problem:** forcing every category into binary good/bad branches hides the baseline. For additive KPIs, we need Positive, Negative, and Neutral branches — and the routing signal must not confound *effect* with *volume*: on one-sided KPIs (revenue-like targets with a positive base), every large category has a large raw sum regardless of whether anything interesting happens inside it.

**Routing signal — centered category excess:**

```math
D_{cat} = S_{cat} - n_{cat}\cdot \bar{y}_{node}
```

where $S_{cat}$ is the category-level sum within the node and $n_{cat}$ is the category row count. On zero-centered targets (profit/loss) $\bar{y}_{node}\approx 0$, so $D_{cat}\approx S_{cat}$; on one-sided targets the centering removes the volume artifact.

**Threshold — a category routes to P (or N) only if it clears BOTH bars:**

```math
\tau_{cat} = \max\Big(\underbrace{V^{c}_{node} \times \mathrm{delta\_pct}}_{\text{materiality}},\ \underbrace{z \cdot \hat{\sigma}_f \cdot \sqrt{n_{cat}}}_{\text{significance}}\Big)
```

where $V^{c}_{node}=\sum|y_i-\bar{y}_{node}|$ is the node's excess volume, $z$ is `noise_z` (default 3.0), and $\hat{\sigma}_f = 1.4826\cdot\mathrm{MAD}$ of the candidate feature's within-category residuals. Route P if $D_{cat} > \tau_{cat}$, N if $D_{cat} < -\tau_{cat}$, else Neutral.

**Why it works:** the materiality bar scales with local volume, so sensitivity adapts by depth — and because it is only 1% of node excess volume, effects trapped inside a large neutral catch-all remain detectable. The significance bar is the category's null band: under pure within-category noise, $D_{cat}$ wanders like $\sigma\sqrt{n_{cat}}$, so noise alone cannot clear it — deep nodes stop fragmenting when only noise is left.

### Act II: The Gain Metric (Category-Averaged Impact Divergence)

**Problem:** after routing categories to Positive/Negative/Neutral, we still need to choose the best splitting feature.

**Evolution:**

- Start from sign-separation intuition (split positive and negative mass so they do not cancel).
- Penalize high-cardinality slicing to avoid overfitting.
- Focus on outer branches because this EDA method is built to find extremes.

**Final formula:**

```math
Gain(X_i) = \frac{|S_P|}{k_P} + \frac{|S_N|}{k_N}
```

Where $S_P, S_N$ are outer-branch sums and $k_P, k_N$ are the number of categories assigned to each branch.

**Why it works:** It balances volume and density, rewarding features that isolate large positive/negative totals with fewer actionable categories; without dividing by $k$, high-cardinality fields like Customer ID or ZIP Code can win by shattering rows into many tiny, low-actionability slices.

### Act III: The Global Kill Switch (Dual Materiality)

**Problem:** as the tree deepens, local thresholds shrink, so eventually even tiny noise can look meaningful.

Standard stopping rules like max depth or min samples are not tied to financial materiality. `impact-split` stops when a branch is globally irrelevant for both positive and negative pools.

**Global theoretical maximums:**

Here, each $y_i$ is an individual row-level target value in the full dataset.

```math
V_{global\_P} = \sum_{y_i > 0} y_i \quad \text{and} \quad V_{global\_N} = \sum_{y_i < 0} |y_i|
```

**Stopping rule:**

```math
\text{Stop if: } \left( \frac{S_P}{V_{global\_P}} \le \theta_{stop} \right) \text{ AND } \left( \frac{S_N}{V_{global\_N}} \le \theta_{stop} \right)
```

**Why it works:** positive and negative impacts are graded against their own global pools, avoiding net-sum distortions and preserving business materiality.

### Act IV: Post-fit Segment Consolidation

**Problem:** the tree can only cut, never regroup. When it splits on one
segment's features, any *other* coherent segment gets tiled across the branches
— dozens of fragments that all behave identically.

After fitting, terminal segments are merged iteratively when (a) their
conditions are identical except on **one** feature's category set — so the
union is still a single readable `feature=cat1,cat2` conjunction (conditions
that grow to cover a feature's full universe are dropped) — and (b) their means
are statistically compatible:

```math
|\bar{y}_1 - \bar{y}_2| \le z \cdot \hat{\sigma} \cdot \sqrt{1/n_1 + 1/n_2}
```

where $\hat{\sigma}$ is the pooled robust (MAD) scale of within-segment
residuals and $z$ is the same `noise_z`. The tree itself is untouched (plots
and traces show the full structure); merging disjoint row sets preserves exact
sum conservation, and consolidation is null-safe — it can only reduce the
segment count (Kaggle suite: −26%). Disable with `consolidate=False`.

### Implementation notes

- Centered-excess routing is the *only* routing mode (since the 2026-07 robustness loop; raw-sum routing was removed because it split on volume rather than effect for one-sided KPIs).
- `max_depth` caps **interaction order** — the number of distinct-feature transitions along a path. Consecutive splits that refine the same feature's category pool are free and remain legal even at the cap; they narrow a segment rather than add an interaction term.
- Defaults (`delta_pct=0.01`, `min_global_impact_pct=0.01`, `max_depth=5`, `noise_z=3.0`, `consolidate=True`) were fixed by benchmark loops over an 8-case synthetic battery and 10 semi-synthetic Kaggle datasets (see `reports/validation-report-v3.md`); the same configuration is used for every dataset — no per-dataset tuning.

## Quick Start

### Install

```bash
pip install impact-split
```

(PyPI, once published — see [CHANGELOG](CHANGELOG.md)). To work on the library itself, use an editable dev install instead:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

### Reproduce The Explainer Notebook

To reproduce `notebooks/1.0-jde-impact-split-explainer.ipynb` from a clean start:

1. Activate the project environment above.
2. Open the notebook and run **Kernel -> Restart & Run All**.
3. Use the notebook's `repro_fingerprint` output to compare reruns.

The explainer notebook is deterministic by design: it uses a seeded RNG (`np.random.default_rng(42)`) and does not require external data/API calls.

### Build & Package Validation

```bash
python -m build --no-isolation
python -m twine check dist/*
```

This creates both wheel and sdist artifacts under `dist/` and validates long-description metadata before publishing.

### Basic Usage

```python
from impact_split import ImpactSplitter

model = ImpactSplitter().fit(X, y)   # X: DataFrame or int-encoded ndarray; y: additive target

print(model)                          # designed text summary (ledger + top segments)
model.plot_segments()                 # tornado: ranked segment impacts (matplotlib)
model.plot_tree()                     # icicle: where impact concentrates in the tree
model.to_html("report.html")          # self-contained interactive report (offline-safe)

segments = model.get_impact_segments()   # DataFrame: path, total_sum, n_samples, node_id, mean, pool_share
payload = model.to_dict()                # JSON-safe dict for custom renderers
```

Constructor knobs (all optional, defaults shown) and `fit()` options:

```python
model = ImpactSplitter(
    delta_pct=0.01,          # materiality: share of node excess volume a category must move
    min_global_impact_pct=0.01,
    max_depth=5,             # interaction-order cap (same-feature refinements are free)
    noise_z=3.0,             # significance: per-category noise floor in sigma units
    consolidate=True,        # post-fit merge of statistically-equal sibling segments
    numeric_binning_strategy="quantiles",  # "quantiles" or "interval"
    numeric_n_bins=10,                     # number of bins for float columns
)
model.fit(X, y, trace=True)  # optional: populate model.fit_trace_
```

If you want the motivation behind each formula (not just usage), read the Story Behind the Math section above, then the explainer notebook linked below.

## Outputs

`impact-split` ships five ways to read a fitted model — a text ledger, two matplotlib figures, a self-contained HTML report, and the raw JSON-safe payload behind all of them. Each is also available as a standalone function (`impact_split.viz.text.render_summary`, `impact_split.viz.static.plot_segments` / `plot_icicle`, `impact_split.viz.html.render_html`) if you want to build a custom renderer from `model.to_dict()`.

- **`print(model)` / `model.summary()`** — a designed text report: a ledger header (global positive/negative pools, split/stop counts) followed by the top segments ranked by absolute impact. Cheapest way to sanity-check a fit in a terminal or log.
- **`model.plot_segments()`** — a tornado chart of the consolidated terminal segments, one horizontal bar per segment, sorted by absolute impact. Bars are additive and diverge from a zero baseline: blue for positive segments, orange for negative, so the reader sees at a glance which segments help and which hurt.

  ![Segment tornado](reports/figures/segments-tornado.png)

- **`model.plot_tree()`** — an impact icicle showing where impact concentrates as the tree splits. Cell width is proportional to Σ|y| within its parent (so wide cells carry more absolute volume), and color encodes the mean excess of that node (diverging: positive vs. negative), making both *where* the impact lives and *which direction* it points visible in one figure.

  ![Impact icicle](reports/figures/impact-icicle.png)

- **`model.to_html(path)`** — a self-contained interactive HTML report combining both figures plus a sortable segment table, with linked highlighting (hovering a segment row highlights it in the icicle and vice versa). No CDN or network calls — every asset is inlined, so the file is safe to open offline or email as an attachment.
- **`model.to_dict()`** — the JSON-safe `meta` / `tree` / `segments` payload every renderer above is built from. Use it to feed a custom dashboard or notebook widget without re-deriving anything from the fitted tree.

### Fit trace (optional)

Pass `trace=True` or `verbose=True` to `fit()` to record one pre-order step per visited node in `model.fit_trace_` (`verbose` is an alias for `trace`; there is no extra logging). Each step includes raw and centered diagnostics (`delta_raw`, `delta_centered_excess`, `V_node`, `V_node_centered`), `routing_mode` (always `centered_excess`), per-category thresholds (`tau`, the max of the materiality and noise-floor bars), `delta_pct`, `s_node_p`, `s_node_n`, `total_sum`, global materiality ratios, per-feature candidate gains, category tables, `chosen_feature_index` when splitting, and `stop_reason` when a leaf is created (`materiality`, `max_depth`, `identical_rows`, or `no_split`). When `X` is a DataFrame, trace rows also include `chosen_feature_name`, `routing_labels`, and per-row `category_label` in category tables where applicable.

## Output

`model.get_impact_segments()` returns terminal segments sorted by absolute impact, with columns:

- `path` — rule path for the segment,
- `total_sum` — sum of `y` in the segment,
- `n_samples` — row count,
- `node_id` — tree node identifier (or `merged(...)` for a consolidated segment),
- `mean` — `total_sum / n_samples`,
- `pool_share` — `|total_sum|` as a share of the segment's own-sign global pool (`V_global_P` or `V_global_N`).

## Assumptions and Limitations

- `fit(X, y)` accepts:
  - `X`: `np.ndarray` with shape `(n_samples, n_features)` and non-negative integer label-encoded categories, or a `pandas.DataFrame`:
    - float columns are converted to bin IDs via `numeric_binning_strategy` and `numeric_n_bins`,
    - other columns are factorized as categorical codes.
  - `y`: `np.ndarray` or `pandas.Series` with shape `(n_samples,)` and float-coercible additive target values.
- For NumPy `X`, inputs should be categorical or discretized before fitting (label-encoded into integer bins).
- Learned bin edges for float columns are stored in `model.numeric_bin_edges_` keyed by feature index.
- Ternary recursion can still grow quickly with depth.
- This is primarily an EDA summarization tool, not a cross-validation-first predictive workflow.

## Learn More

- Documentation is MkDocs source under [`docs/`](docs/) (not yet hosted) — build locally per [`docs/README.md`](docs/README.md)
- Full mathematical walkthrough and toy example (documented synthetic DGP: planted category-interaction effects plus noise; fit uses observed outcome only):
  - [`notebooks/1.0-jde-impact-split-explainer.ipynb`](notebooks/1.0-jde-impact-split-explainer.ipynb)
- Kaggle Sample Supermarket data, `kagglehub` download, and step-by-step trace tables:
  - [`notebooks/2.0-jde-supermarket-kaggle-trace.ipynb`](notebooks/2.0-jde-supermarket-kaggle-trace.ipynb) (requires [Kaggle API credentials](https://github.com/Kaggle/kagglehub#authentication) for `kagglehub`)
- Setup and navigation (source for the docs site):
  - [`docs/docs/getting-started.md`](docs/docs/getting-started.md)