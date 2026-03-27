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

## Why This Algorithm Is Different

`impact-split` is designed for additive business KPIs where totals matter more than average purity. It uses a local sieve (`delta = V_node * delta_pct`) to create Positive/Neutral/Negative branches, a gain metric that rewards concentrated outer-branch impact, and a dual-materiality stopping rule that halts branches with globally irrelevant positive and negative mass.

For the full three-act origin story and formulas, read:

- `README.md` ("Story Behind the Math")
- `notebooks/1.0-jde-impact-split-explainer.ipynb`

## First Run

1. Prepare `X` as a 2D `numpy.ndarray` of integer label-encoded categories.
2. Prepare `y` as a 1D `numpy.ndarray` with additive target values (for example, profit/loss).
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

### Kaggle example notebook

To load [Sample Supermarket](https://www.kaggle.com/datasets/bravehart101/sample-supermarket-dataset) with `kagglehub` and print each algorithm step, run:

- `notebooks/2.0-jde-supermarket-kaggle-trace.ipynb`

Configure Kaggle credentials first ([kagglehub authentication](https://github.com/Kaggle/kagglehub#authentication)).

The notebook label-encodes the chosen string columns into integer `numpy` arrays (required by `fit`), fits with `trace=True`, prints a per-node summary (`delta`, `V_node`, `s_node_p` / `s_node_n`, `stop_reason`, `global_ratios`), and adds EDA that compares `delta` to per-category sums and sweeps `delta_pct`—useful when the tree stops at the root with `no_split`.

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
