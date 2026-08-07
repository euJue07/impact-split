# Getting Started

Use this page to set up a local environment and run the first Impact Split workflow.

## Prerequisites

- Python 3.10+ (3.10–3.13 tested in CI)
- `pip`

## Installation

```bash
pip install impact-split
```

(PyPI, once published — see [CHANGELOG](../../CHANGELOG.md)). To work on the library itself, use an editable dev install instead:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

## Reproducible Notebook Run

For a deterministic run of `notebooks/1.0-jde-impact-split-explainer.ipynb`:

1. Activate the environment from Installation.
2. Open the notebook and execute **Kernel -> Restart & Run All**.
3. Compare the printed `repro_fingerprint` dictionary across reruns.

This notebook uses synthetic data only, with randomness seeded via `np.random.default_rng(42)`, so reruns should match.

## Why This Algorithm Is Different

`impact-split` is designed for additive business KPIs where totals matter more than average purity. It uses a local sieve (`delta = V_node * delta_pct`) to create Positive/Neutral/Negative branches, a centered-excess fallback (`D_cat = S_cat - n_cat * mean(y_node)`) when raw routing cannot partition a node, a gain metric that rewards concentrated outer-branch impact, and a dual-materiality stopping rule that halts branches with globally irrelevant positive and negative mass. Candidate features that would still send 100% of rows down a single branch (or are constant on the current slice) are skipped so the tree does not repeat the same split with identical data.

For the full three-act origin story and formulas, read:

- `README.md` ("Story Behind the Math")
- `notebooks/1.0-jde-impact-split-explainer.ipynb`

## First Run

1. Prepare `X` as a 2D `numpy.ndarray` of integer label-encoded categories, or a `pandas.DataFrame` (each column is factorized internally; missing values are not allowed).
2. Prepare `y` as a 1D `numpy.ndarray` or `pandas.Series` with additive target values (for example, profit/loss).
3. Fit and inspect:

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

After fitting with a `DataFrame`, `model.feature_names_in_` and `model.category_maps_` hold column names and code-to-value maps for each feature. Pass `trace=True` (or the `verbose=True` alias) to `fit()` to populate `model.fit_trace_` with one pre-order step per visited node.

## Fitting from a star or snowflake schema

If your features live in dimension tables rather than one flat frame, describe the schema
and let `flatten` denormalize it:

```python
from impact_split import ImpactSplitter, Join, SchemaSpec, flatten

tables = {
    "fact_sales": fact_df,        # one row per sale, carries the additive target
    "dim_customer": customer_df,  # one row per customer
    "dim_geo": geo_df,            # one row per geography (snowflaked off dim_customer)
}

spec = SchemaSpec(
    fact="fact_sales",
    target="amount",
    features=("channel",),                     # fact columns to keep as features
    joins=(
        Join(table="dim_customer", left="customer_id", right="customer_id",
             columns=("tier", "segment")),
        Join(table="dim_geo", left="geo_id", right="geo_id",
             parent="dim_customer", columns=("country",)),
    ),
)

result = flatten(tables, spec)
model = ImpactSplitter().fit(result.X, result.y)
print(model)
```

Segment paths name the source table, so a driver reads as
`dim_customer.tier=gold / dim_geo.country=PH`.

`result.provenance` is the audit trail: which tables joined in which order, and how many
rows (and how much `y`) failed to match each dimension.

**The restriction:** every join must be many-to-one. If a dimension's key is not unique,
`flatten` raises `SchemaError` naming the table, the key, and an example duplicate — it
will not silently duplicate rows. One-to-many relationships are out of scope.

## Outputs

Five ways to read a fitted model, all built from the same `model.to_dict()` payload:

- **`print(model)` / `model.summary()`** — text ledger (global pools, split/stop counts) plus the top segments ranked by absolute impact.
- **`model.plot_segments()`** — tornado chart of consolidated segments: additive bars diverging at zero, blue positive / orange negative.
- **`model.plot_tree()`** — impact icicle: cell width ∝ Σ|y| within its parent, color = mean excess (diverging), so magnitude and direction are both visible at a glance. Returns a matplotlib `Figure`; pass `show=False` to save without displaying (`fig = model.plot_tree(show=False); fig.savefig("tree.svg")`).
- **`model.to_html(path)`** — self-contained interactive report (both figures + a sortable segment table with linked highlighting); no CDN, safe to open offline or email.
- **`model.to_dict()`** — the JSON-safe `meta` / `tree` / `segments` payload behind every renderer above, for custom dashboards.

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
