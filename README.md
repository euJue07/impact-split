# impact-split

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

A tree based approach to determine the driver of a KPI

## Impact Split

`impact-split` is an ML-driven EDA approach for additive KPIs (extensive metrics), such as:

- Total Revenue
- Total Hours Watched
- Total Profit/Loss

Unlike standard decision trees that optimize for variance reduction (often favoring "pure" average-based segments), Impact Split prioritizes **material business impact** by focusing on segment-level totals.

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         impact_split and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
└── impact_split   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes impact_split a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

--------

## Overview

`impact-split` is an ML-driven EDA approach for additive KPIs (extensive metrics), such as:

- Total Revenue
- Total Hours Watched
- Total Profit/Loss

Traditional trees optimize for average purity. `impact-split` instead prioritizes **material business impact** by ranking segments based on total contribution.

## Core Idea

The algorithm builds a ternary tree (Positive / Neutral / Negative) over categorical or pre-binned features and uses:

- a local threshold (`delta_pct`) to separate strong positive and negative category impacts from neutral ones,
- a gain metric that emphasizes outer-branch impact while penalizing high-cardinality noise,
- a global stopping threshold (`min_global_impact_pct`) to stop splitting low-materiality nodes.

## Story Behind the Math

Most decision-tree algorithms were designed to minimize variance and isolate segments with similar average values. In business work, averages can hide what matters most: total impact.

Example: a tiny segment with 2 churn events at -$5,000 each can look "purer" than a segment with 10,000 churn events at -$40 each. But the second segment carries far more total business weight.

`impact-split` was designed to solve this exact mismatch by optimizing for additive totals.

### Act I: The Local Sieve (`delta`)

**Problem:** forcing every category into binary good/bad branches hides the baseline. For additive KPIs, we need Positive, Negative, and Neutral branches.

**Formula:**

$$
\delta = V_{node} \times \text{delta\_pct}
$$

Where $V_{node}$ is the absolute sum of target values inside the current node.

**Why it works:** Neutral boundaries scale with local volume, so sensitivity adapts by depth. High-volume nodes ignore small noise; lower-volume nodes detect finer impacts.

### Act II: The Gain Metric (Category-Averaged Impact Divergence)

**Problem:** after routing categories to Positive/Negative/Neutral, we still need to choose the best splitting feature.

**Evolution:**

- Start from sign-separation intuition (split positive and negative mass so they do not cancel).
- Penalize high-cardinality slicing to avoid overfitting.
- Focus on outer branches because this EDA method is built to find extremes.

**Final formula:**

$$
Gain(X_i) = \frac{|S_P|}{k_P} + \frac{|S_N|}{k_N}
$$

Where $S_P, S_N$ are outer-branch sums and $k_P, k_N$ are the number of categories assigned to each branch.

**Why it works:** It balances volume and density, rewarding features that isolate large positive/negative totals with fewer actionable categories.

### Act III: The Global Kill Switch (Dual Materiality)

**Problem:** as the tree deepens, local thresholds shrink, so eventually even tiny noise can look meaningful.

Standard stopping rules like max depth or min samples are not tied to financial materiality. `impact-split` stops when a branch is globally irrelevant for both positive and negative pools.

**Global theoretical maximums:**

$$
V_{global\_P} = \sum_{y_i > 0} y_i \quad \text{and} \quad V_{global\_N} = \sum_{y_i < 0} |y_i|
$$

**Stopping rule:**

$$
\text{Stop if: } \left( \frac{S_{node\_P}}{V_{global\_P}} \le \theta_{stop} \right) \text{ AND } \left( \frac{S_{node\_N}}{V_{global\_N}} \le \theta_{stop} \right)
$$

**Why it works:** positive and negative impacts are graded against their own global pools, avoiding net-sum distortions and preserving business materiality.

### Implementation note

Current implementation applies $\delta = V_{node} \times \text{delta\_pct}$ consistently at every depth, including the root.

## Quick Start

### Install

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
# For contributor tooling (lint/test/build/check): 
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

model = ImpactSplitter(
    delta_pct=0.05,
    min_global_impact_pct=0.01,
    max_depth=5,
)

# X: 2D numpy ndarray of integer label-encoded categories (0, 1, 2, ...)
#     or a pandas DataFrame of categorical columns (factorized internally).
# y: 1D numpy ndarray or pandas Series with additive target (e.g., profit/loss)
model.fit(X, y, trace=True)  # optional: populate model.fit_trace_

model.plot_tree(figsize=(16, 10))  # returns a matplotlib Figure; pass show=False to save without displaying
segments = model.get_impact_segments()
print(segments.head())
```

### Interactive Force Graph (Notebook + HTML Export)

You can render a D3 force graph in notebooks and export the same interactive chart to standalone HTML:

```python
from impact_split import interactive_force_graph

nodes = [
    {"id": "root", "label": "Root", "group": "all", "tooltip": "Global node"},
    {"id": "segA", "label": "Segment A", "group": "positive"},
    {"id": "segB", "label": "Segment B", "group": "negative"},
]
links = [
    {"source": "root", "target": "segA", "value": 2},
    {"source": "root", "target": "segB", "value": 1},
]

def on_select(event: dict) -> None:
    print("Selection event:", event)

graph = interactive_force_graph(
    nodes=nodes,
    links=links,
    filter_keys=["group"],
    options={"width": 860, "height": 520, "charge_strength": -120},
    on_selection=on_select,
)

graph.show()                      # notebook rendering
graph.save_html("reports/force_graph.html")  # standalone interactive export
```

Interaction support includes drag (with simulation reheating), zoom/pan, hover tooltips, click-select highlighting, Python-side filter controls, and click events sent back to the Python callback payload.

If you want the motivation behind each formula (not just usage), read the Story Behind the Math section above, then the explainer notebook linked below.

### Fit trace (optional)

Pass `trace=True` or `verbose=True` to `fit()` to record one pre-order step per visited node in `model.fit_trace_` (`verbose` is an alias for `trace`; there is no extra logging). Every node uses `delta = V_node * delta_pct` for category assignment in split scoring/routing. Each step includes `delta`, `delta_pct`, `V_node`, `s_node_p`, `s_node_n`, `total_sum`, global materiality ratios, per-feature candidate gains, category tables, `chosen_feature_index` when splitting, and `stop_reason` when a leaf is created (`materiality`, `max_depth`, `identical_rows`, or `no_split`). When `X` is a DataFrame, trace rows also include `chosen_feature_name`, `routing_labels`, and per-row `category_label` in category tables where applicable.

## Output

`model.get_impact_segments()` returns terminal segments sorted by absolute impact, with columns such as:

- `path` — rule path for the segment,
- `total_sum` — sum of `y` in the segment,
- `n_samples` — row count,
- `node_id` — tree node identifier.

## Assumptions and Limitations

- `fit(X, y)` accepts:
  - `X`: `np.ndarray` with shape `(n_samples, n_features)` and non-negative integer label-encoded categories, or a `pandas.DataFrame` (each column is factorized to integer codes; see `feature_names_in_` and `category_maps_` after fitting).
  - `y`: `np.ndarray` or `pandas.Series` with shape `(n_samples,)` and float-coercible additive target values.
- For NumPy `X`, inputs should be categorical or discretized before fitting (label-encoded into integer bins). DataFrame columns are treated as categorical.
- Ternary recursion can still grow quickly with depth.
- This is primarily an EDA summarization tool, not a cross-validation-first predictive workflow.

## Learn More

- Full mathematical walkthrough and toy example (documented synthetic DGP: planted category-interaction effects plus noise; fit uses observed outcome only):
  - [`notebooks/1.0-jde-impact-split-explainer.ipynb`](notebooks/1.0-jde-impact-split-explainer.ipynb)
- Kaggle Sample Supermarket data, `kagglehub` download, and step-by-step trace tables:
  - [`notebooks/2.0-jde-supermarket-kaggle-trace.ipynb`](notebooks/2.0-jde-supermarket-kaggle-trace.ipynb) (requires [Kaggle API credentials](https://github.com/Kaggle/kagglehub#authentication) for `kagglehub`)
- Setup and navigation:
  - [`docs/docs/getting-started.md`](docs/docs/getting-started.md)