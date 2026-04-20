# Getting Started

Use this page to set up a local environment and run the first Impact Split workflow.

## Prerequisites

- Python `3.13.x` (project requires `~=3.13.0`)
- `pip`

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
# Optional contributor toolchain (lint, tests, packaging checks):
python -m pip install -e ".[dev]"
```

## Reproducible Notebook Run

For a deterministic run of `notebooks/1.0-jde-impact-split-explainer.ipynb`:

1. Activate the environment from Installation.
2. Open the notebook and execute **Kernel -> Restart & Run All**.
3. Compare the printed `repro_fingerprint` dictionary across reruns.

This notebook uses synthetic data only, with randomness seeded via `np.random.default_rng(42)`, so reruns should match.

## Why This Algorithm Is Different

`impact-split` is designed for additive business KPIs where totals matter more than average purity. It uses a local sieve (`delta = V_node * delta_pct`) to create Positive/Neutral/Negative branches, a gain metric that rewards concentrated outer-branch impact, and a dual-materiality stopping rule that halts branches with globally irrelevant positive and negative mass. Candidate features that would send 100% of rows down a single branch (or are constant on the current slice) are skipped so the tree does not repeat the same split with identical data.

For the full three-act origin story and formulas, read:

- `README.md` ("Story Behind the Math")
- `notebooks/1.0-jde-impact-split-explainer.ipynb`

## First Run

1. Prepare `X` as a 2D `numpy.ndarray` of integer label-encoded categories, or a `pandas.DataFrame` (each column is factorized internally; missing values are not allowed).
2. Prepare `y` as a 1D `numpy.ndarray` or `pandas.Series` with additive target values (for example, profit/loss).
3. Fit and inspect:

```python
from impact_split import ImpactSplitter

model = ImpactSplitter(
    delta_pct=0.05,
    min_global_impact_pct=0.01,
    max_depth=5,
)

model.fit(X, y, trace=True)  # or verbose=True (alias); inspect model.fit_trace_
model.plot_tree(figsize=(16, 10))
segments = model.get_impact_segments()
```

After fitting with a `DataFrame`, `model.feature_names_in_` and `model.category_maps_` hold column names and code-to-value maps for each feature.

`plot_tree` returns a matplotlib `Figure`. Pass `show=False` if you want to save without calling `plt.show()`.

!!! note "`plot_tree` layout, crowded trees, and export"

    - Node labels show the segment (all data or `feature=categories`) on every node; internal nodes add which feature is split on, plus **n** and **Σy** stats.
    - For crowded trees: widen `figsize`, tune `level_gap` and `sibling_gap`, or set `compact_stats=True`. Layout uses measured label widths (iterative); `node_label_max_chars` trims long lines before layout.
    - Optional `max_leaf_width` (in data coordinates) tightens per-line truncation via a binary search on character budget; default `None` means no width budget. Raise `layout_max_iterations` only if needed.
    - Optional `node_facecolor="impact"` (magnitude of **Σy**) or `"n"` (sample count) adds a colorbar and contrasting label text.
    - PDF/SVG export: `fig = model.plot_tree(figsize=(16, 10), show=False); fig.savefig("reports/figures/tree.pdf")`.

## Interactive Chart Workflow

`impact_split` also includes a notebook-first interactive D3 force graph that can be exported to standalone HTML.

```python
from impact_split import interactive_force_graph

nodes = [
    {"id": "root", "label": "Root", "group": "all"},
    {"id": "p1", "label": "Positive branch", "group": "positive"},
    {"id": "n1", "label": "Negative branch", "group": "negative"},
]
links = [
    {"source": "root", "target": "p1", "value": 3},
    {"source": "root", "target": "n1", "value": 2},
]

def on_selection(event: dict) -> None:
    print(event)

graph = interactive_force_graph(
    nodes=nodes,
    links=links,
    filter_keys=["group"],
    on_selection=on_selection,
)
graph.show()
graph.save_html("reports/figures/impact_force_graph.html")
```

The callback receives a payload with:
- `event_type` (`"node_click"`)
- `selected_node_id` (string or `None`)
- `filters` (active filter dictionary)

### Kaggle example notebook

To load [Sample Supermarket](https://www.kaggle.com/datasets/bravehart101/sample-supermarket-dataset) with `kagglehub` and print each algorithm step, run:

- `notebooks/2.0-jde-supermarket-kaggle-trace.ipynb`

Configure Kaggle credentials first ([kagglehub authentication](https://github.com/Kaggle/kagglehub#authentication)).

The notebook passes those columns as a string `pandas.DataFrame` to `fit` (the splitter factorizes internally), fits with `trace=True`, prints a per-node summary (`delta`, `V_node`, `s_node_p` / `s_node_n`, `stop_reason`, `global_ratios`), and adds EDA that compares `delta` to per-category sums and sweeps `delta_pct`—useful when the tree stops at the root with `no_split`.

## Packaging Validation

Run these commands before releasing:

```bash
python -m build --no-isolation
python -m twine check dist/*
```

Optional smoke-install check from the wheel:

```bash
python3 -m venv .venv-smoke
source .venv-smoke/bin/activate
python -m pip install dist/*.whl
python -c "from impact_split import ImpactSplitter; print(ImpactSplitter.__name__)"
```

## Where to Read Next

- Project overview: `README.md`
- Full deep dive with equations and worked example (synthetic DGP: structural `y_expected` from planted category interactions, noise; the tree is fit on observed outcome only): `notebooks/1.0-jde-impact-split-explainer.ipynb`
- Practical trace walkthrough on Kaggle data: `notebooks/2.0-jde-supermarket-kaggle-trace.ipynb`
