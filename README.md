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
├── setup.cfg          <- Configuration file for flake8
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

## Quick Start

### Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Basic Usage

```python
from impact_split import ImpactSplitter

model = ImpactSplitter(
    delta_pct=0.05,
    min_global_impact_pct=0.01,
    max_depth=5,
    neutral_root=True,  # default: root uses assignment delta 0; set False for legacy scaled root
)

# X: pandas DataFrame of categorical or pre-binned features
# y: pandas Series with additive target (e.g., profit/loss)
model.fit(X, y, trace=True)  # optional: populate model.fit_trace_

model.plot_tree(figsize=(16, 10))
segments = model.get_impact_segments()
print(segments.head())
```

### Fit trace (optional)

Pass `trace=True` or `verbose=True` to `fit()` to record one pre-order step per visited node in `model.fit_trace_` (`verbose` is an alias for `trace`; there is no extra logging). With the default `neutral_root=True`, the **root** uses assignment thresholds `delta=0` (categories split by the sign of `S_cat`); deeper nodes use `delta = V_node * delta_pct`. Each step includes `delta_nominal` (always `V_node * delta_pct`), assignment `delta` / `delta_neg` / `neutral_band` (used for `S_cat`), `delta_pct`, `V_node`, `s_node_p`, `s_node_n`, `total_sum`, global materiality ratios, per-feature candidate gains, category tables, `chosen_feature` when splitting, and `stop_reason` when a leaf is created (`materiality`, `max_depth`, `no_split`, or `empty_children`).

## Output

`model.get_impact_segments()` returns terminal segments sorted by absolute impact, with columns such as:

- `path` — rule path for the segment,
- `total_sum` — sum of `y` in the segment,
- `n_samples` — row count,
- `node_id` — tree node identifier.

## Assumptions and Limitations

- Inputs should be categorical or discretized before fitting.
- Ternary recursion can still grow quickly with depth.
- This is primarily an EDA summarization tool, not a cross-validation-first predictive workflow.

## Learn More

- Full mathematical walkthrough and toy example:
  - [`notebooks/1.0-jde-impact-split-explainer.ipynb`](notebooks/1.0-jde-impact-split-explainer.ipynb)
- Kaggle Sample Supermarket data, `kagglehub` download, and step-by-step trace tables:
  - [`notebooks/2.0-jde-supermarket-kaggle-trace.ipynb`](notebooks/2.0-jde-supermarket-kaggle-trace.ipynb) (requires [Kaggle API credentials](https://github.com/Kaggle/kagglehub#authentication) for `kagglehub`)
- Setup and navigation:
  - [`docs/docs/getting-started.md`](docs/docs/getting-started.md)