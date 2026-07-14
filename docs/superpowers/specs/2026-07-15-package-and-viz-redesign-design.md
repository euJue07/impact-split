# impact-split v0.1.0 — official package + interpretability-first outputs

**Date:** 2026-07-15 · **Status:** approved · **Owner:** Juedi Eugenio

## Goal

Turn `projects/impact-split` into a release-ready, PyPI-publishable library
(`pip install impact-split`), and replace the three existing outputs — the
box-and-arrow `plot_tree`, the generic CDN-dependent `interactive_force_graph`,
and the undesigned `segments.head()` print experience — with a coordinated,
interpretability-first output suite driven by a single model serializer.

User decisions (2026-07-15): PyPI release-ready (user runs the final upload) ·
segment tornado + impact icicle tree · self-contained HTML (no CDN) · CCDS
scaffold removed · old plots fully removed (no `style="boxes"` compat shim).

## Non-goals / invariants

- **Zero algorithm changes.** The sacred set (ternary P/Neu/N tree, exact sum
  conservation, readable `feature=cat1,cat2` paths) and all fitting code are
  untouched.
- `get_impact_segments()` keeps its existing columns and ordering (benchmark
  scoring depends on them). Additive columns are allowed: `mean`, `pool_share`.
- Benchmarks must score identically before and after.
- No per-dataset configuration anywhere; nothing here affects defaults.

## Part 1 — Packaging

### Changes

- **Delete CCDS scaffold** from the installable package: `config.py`,
  `features.py`, `dataset.py`, `modeling/`. (Verified 2026-07-15: nothing else
  in the repo — library, tests, benchmarks, notebooks — imports them; they only
  import each other.) Remove the lazy `config` re-export from `__init__.py`.
- **Dependencies** become exactly: `numpy`, `pandas`, `matplotlib`.
  Drop `loguru`, `typer`, `tqdm`, `python-dotenv`.
- **`requires-python`**: `>=3.10` (no 3.11+ syntax present; verified by grep,
  enforced by CI matrix). Classifiers list 3.10–3.13.
- **Version** `0.0.1 → 0.1.0`; add `CHANGELOG.md` (Keep-a-Changelog format)
  recording the breaking removals and the new output suite.
- **CI**: extend the existing GitHub Actions workflow to a 3.10/3.11/3.12/3.13
  test matrix; add a release job that on `v*` tags runs `python -m build` +
  `twine check` and uploads artifacts to the GitHub release. The actual PyPI
  upload (or trusted-publishing wiring) is done by the user.
- **README**: rewrite Quick Start around `pip install impact-split`; replace
  the force-graph section with the new output suite (summary / tornado /
  icicle / HTML report); keep the Story Behind the Math acts as-is.
- **`__init__.py` exports**: `ImpactSplitter` plus the new top-level renderer
  names (see Part 2). `InteractiveForceGraph` / `interactive_force_graph` are
  removed.

### Acceptance

- `python -m build --no-isolation` and `twine check dist/*` pass.
- Fresh-venv `pip install dist/*.whl` → `from impact_split import
  ImpactSplitter` works with only numpy/pandas/matplotlib pulled in.
- `pytest` green; benchmark suite reproduces mean 0.9617 / floor 0.8154.

## Part 2 — Output architecture: one payload, four renderers

### 2.1 Serializer — `model.to_dict()`

Public, documented, stable-shape dict built after `fit()`; the single source
for every renderer (and for third-party custom renderers):

```
{
  "meta":     {package version, fit params, n_rows, n_features,
               feature_names (or indices), total_sum, pos_pool, neg_pool,
               n_nodes, n_leaves, physical_depth, interaction_depth},
  "tree":     [ {id, parent_id, branch (P/Neu/N/root), depth,
                 condition (last fragment, human-readable), split_feature|null,
                 n, total_sum, pos_sum, neg_sum, abs_volume (Σ|y|),
                 is_leaf, segment_id|null} ... ],   # pre-order
  "segments": [ {segment_id, path, node_ids, n, total_sum, mean,
                 pool_share (signed fraction of Σy⁺ or Σy⁻)} ... ]  # by |Σy| desc
}
```

JSON-safe: numpy scalars → Python floats/ints; NaN/inf never emitted
(guarded; degenerate values become null). Raises the existing
`RuntimeError("Call fit() before …")` pre-fit — same pattern for all
renderers below.

### 2.2 New subpackage `impact_split/viz/`

```
viz/
  __init__.py   # re-exports
  data.py       # payload builder (to_dict internals) + shared formatting
                # helpers (human number format, path wrapping, color mapping)
  text.py       # summary()
  static.py     # plot_segments(), plot_tree()  (matplotlib)
  html.py       # to_html()  (template + inline vanilla JS/SVG)
```

`splitter.py` delegates thin methods (`to_dict`, `summary`, `plot_segments`,
`plot_tree`, `to_html`) to `viz`; the ~400-line box-and-arrow plotting block
and its helpers leave `splitter.py`. `plots.py` (force graph) is deleted.

### 2.3 Text — `model.summary()` and `print(model)`

`summary()` returns a `str` (never prints); `__repr__` delegates to a compact
one-liner pre-fit and to `summary()` post-fit. Layout:

```
ImpactSplitter — fit summary
============================
rows 12,345 · features 6 · params delta_pct=0.01 noise_z=3.0 max_depth=5 consolidate=True
total Σy  +1,234,568   (Σy⁺ +2,000,123 · Σy⁻ −765,555)
tree      23 nodes · 14 leaves · depth 3 (interaction order 2)
segments  8 after consolidation (14 leaves merged) · conservation exact ✓

Top segments by |impact|
 #  path                                        Σy        n     pool share
 1  region=NA & product=A,B               +523,411    4,102    26.2% of Σy⁺
 2  channel=online & tier=basic           −301,877    9,880    39.4% of Σy⁻
 ...
 8  (baseline: everything else)            +12,004   14,207     0.6% of Σy⁺
```

Rules: thousands separators, explicit sign on Σy, right-aligned numerics,
paths truncated with `…` to a configurable width (default fits 100 cols),
`top` parameter (default 10) with a rolled-up remainder line so the
conservation story stays complete. The conservation line is computed, not
assumed: `Σ segments == total_sum` to float tolerance, reported `exact ✓` /
`MISMATCH ✗`.

### 2.4 Static figures — `viz/static.py`

**`plot_segments(top=15, figsize=None, show=True) -> Figure` — the tornado.**
Consolidated segments sorted by |Σy| desc, horizontal bars diverging from a
zero line. Positive = colorblind-safe blue (#3B7EA1 family), negative =
orange/vermillion (#D55E00 family) — never green/red. Y labels = readable
rule paths (wrapped, not truncated where possible); at each bar end the value
`+523,411` and inside/beside it `n=4,102 · 26% of Σy⁺`. Segments beyond `top`
roll into a hatched "(+k more segments)" bar so the chart still sums to the
total. Footer annotation: `sum of segments = total Σy = +1,234,568 ✓`.

**`plot_tree(figsize=None, show=True, min_label_width=…) -> Figure` — the icicle.**
Root strip at top spanning full width; each level below tiles its parent.
**Width ∝ Σ|y| (abs_volume)** — additive, so children exactly tile parents and
visual area is trustworthy. **Color = diverging impact scale**: sign and
intensity from the node's centered mean excess (blue positive / gray neutral /
orange negative, same hues as the tornado). Cell text (condition fragment +
`Σy`) drawn only when the cell is wide enough to fit it; otherwise blank —
detail lives in the HTML version and `summary()`. Leaves get a thin outline
tinted by their consolidated `segment_id` so consolidation is visible without
clutter. A small legend explains width/color encodings. Both figures return a
matplotlib `Figure`; `show=False` supports `fig.savefig(...)` (documented for
PDF/SVG export).

### 2.5 Interactive HTML — `model.to_html(path=None) -> str | Path`

One fully self-contained file: inline CSS, inline vanilla-JS, SVG rendering,
data embedded as one JSON blob — **zero external requests** (no CDN, no
fonts, no images). Safe to email or open offline.

Layout (single page, light theme, system font stack):
1. **Header ledger** — the `summary()` numbers as stat tiles (rows, total Σy,
   Σy⁺/Σy⁻, nodes/leaves/segments, params, conservation check).
2. **Icicle** (SVG) — same encodings as the static one, plus: hover tooltip
   (full cumulative path, n, Σy, Σy⁺/Σy⁻, share of parent and of global
   pools), click a cell to zoom that subtree to full width (breadcrumb +
   reset), leaf cells carry their `segment_id`.
3. **Tornado** (SVG) — same as static, hover shows the segment tooltip.
4. **Segment table** — sortable by any column (path, Σy, n, mean, pool share),
   default |Σy| desc.
Linked highlighting: hovering a table row or tornado bar highlights that
segment's leaf cells in the icicle, and vice versa.

Implementation: `html.py` holds a Python template (f-string/`string.Template`)
plus one `<script>` of hand-written JS (~300–400 lines). JSON embedding
escapes `</script>` (`<\/`) and forbids NaN (`json.dumps(allow_nan=False)`
over the sanitized payload). `path=None` returns the HTML string (usable with
`IPython.display.HTML` in notebooks); with a path it writes UTF-8 and returns
the `Path`.

### 2.6 Removals

- `plots.py` (`InteractiveForceGraph`, `interactive_force_graph`) — deleted.
- Box-and-arrow `plot_tree` and its layout helpers
  (`_format_plot_node_label`, `_estimate_plot_label_bbox_units`, etc.) —
  deleted from `splitter.py`.
- `tests/test_interactive_plots.py` — replaced (see Part 3).
- Notebooks 1.0/2.0 cells calling the old plots — updated to the new calls and
  re-executed; the validation notebook is unaffected (scores only).
- Docs `getting-started.md` and README sections referencing removed APIs —
  rewritten.

## Part 3 — Testing

New `tests/test_viz.py` (+ keep existing 34-test suite green minus the
replaced force-graph tests):

- **Payload**: to_dict conservation (Σ segment sums == total; Σ leaf sums ==
  total), pre-order tree integrity (every parent_id exists, leaves ↔
  segment_id mapping consistent), JSON round-trip via
  `json.dumps(..., allow_nan=False)` succeeds.
- **Text**: golden-substring checks on `summary()` (header ledger lines, sign
  formatting, rolled-up remainder row when `top` < segment count),
  `RuntimeError` pre-fit, `repr` pre/post fit.
- **Static**: both plots return `Figure` without display (`show=False`,
  `Agg` backend); icicle rectangles per depth level sum to root width
  (additivity); no exception on a root-only (no-split) fit.
- **HTML**: output contains the embedded JSON, contains **no** `http://` /
  `https://` / `src=` external reference, `</script>` never appears inside the
  data blob, file round-trips through `path` write mode.
- **Regression**: `get_impact_segments()` existing columns unchanged; a fit on
  the existing synthetic fixture produces identical segment sums pre/post
  refactor.
- **Packaging smoke** (CI): build wheel, install into a clean venv, import,
  fit a tiny dataset, call all four renderers.

## Part 4 — Delivery

- Work happens in the `projects/impact-split` repo on a feature branch
  (`package-and-viz`), merged `--no-ff` to main, pushed to
  `euJue07/impact-split` (git https; gh API 404s on this account — known).
- Never run ai-os `project git-init` on this folder. ai-os side: update the
  PROJECT manifest + KB after ship, via `engine/aios.py` flows only.
- Commit sequence (one concern per commit): scaffold removal + metadata →
  `viz/data.py` + `to_dict` → text → static → html → tests/notebooks/docs →
  README/CHANGELOG/CI.
- Definition of done: acceptance checks in Part 1 + full test suite + a
  generated `reports/sample-report.html` and tornado/icicle PNGs from the
  explainer dataset, benchmarks reproduced, pushed.
