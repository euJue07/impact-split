# Getting Started

Use this page to set up a local environment and run the first Impact Split workflow.

## Prerequisites

- Python `3.13.x` (project requires `~=3.13.0`)
- `pip`

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## First Run

1. Prepare `X` as categorical or pre-binned features.
2. Prepare `y` as an additive target (for example, profit/loss).
3. Fit and inspect:

```python
from impact_split import ImpactSplitter

model = ImpactSplitter(
    delta_pct=0.05,
    min_global_impact_pct=0.01,
    max_depth=5,
)

model.fit(X, y, trace=True)  # optional: inspect model.fit_trace_
model.plot_tree(figsize=(16, 10))
segments = model.get_impact_segments()
```

### Kaggle example notebook

To load [Sample Supermarket](https://www.kaggle.com/datasets/bravehart101/sample-supermarket-dataset) with `kagglehub` and print each algorithm step, run:

- `notebooks/2.0-jde-supermarket-kaggle-trace.ipynb`

Configure Kaggle credentials first ([kagglehub authentication](https://github.com/Kaggle/kagglehub#authentication)).

## Where to Read Next

- Project overview: `README.md`
- Full deep dive with equations and worked example: `notebooks/1.0-jde-impact-split-explainer.ipynb`
