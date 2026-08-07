# Changelog

All notable changes to this project are documented here.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versioning: [SemVer](https://semver.org/).

## [0.3.0] - Unreleased

### Added
- **`ImpactSplitter.ensemble_report(X, y, ...)`** (new public method, populates
  `model.ensemble_`): a Random-Forest-inspired forest of perturbed refits that
  *annotates* the single fitted tree without ever replacing it — no prediction
  averaging. Two blocks: row-bootstrap replicates give each segment a
  **stability** score and a 5–95% bootstrap **CI** on Σy; feature-subsampled
  replicates surface **shadow segments** — material regions that only appear
  once dominant features are forced out. Shadow candidates must additionally
  clear the same materiality + noise-floor sieve on the *full* data (a
  root-sieve significance gate added as a spec amendment during Task 6, to
  keep bootstrap-only noise from posing as a shadow finding). A third summary,
  availability-weighted **ensemble feature importance**, is derived from the
  same replicate pool. New module `impact_split/ensemble.py`.
- Payload: `to_dict()` gains a single `"ensemble"` key carrying the report
  (absent when no report has been run).
- `summary()` gains stability/CI columns on the segment ledger and a shadow
  drivers section.
- `plot_segments()` and `plot_tree()` gain CI whiskers and stability
  annotations.
- `to_html()` gains stability/CI columns, CI whiskers, and Shadow drivers /
  Ensemble importance sections.
- `benchmarks/ensemble_filter.py` — offline diagnostic scoring stability-
  filtered ledgers against the synthetic battery. Result: filtering made no
  measurable difference (mean impact-F1 0.9836 → 0.9835, floor 0.8846 flat
  across stability thresholds) — no default changed as a result.
- **`impact_split.schema`** (new module; `flatten`, `SchemaSpec`, `Join`,
  `FlattenResult`, `SchemaError` exported from the package root): star/snowflake
  denormalization into the flat frame `fit()` already accepts. `SchemaSpec` + `Join`
  are the declarative contract — fact table, target, join chain, feature selection.
  Joins are **many-to-one only** and execute as a reindex against a unique index, so a
  fan-out cannot occur silently: duplicate or null dimension keys raise `SchemaError`
  naming the table and key, plus an example duplicate value when the violation is
  non-uniqueness. Fact rows whose foreign key does not resolve are kept under an
  `<unmatched>` category rather than dropped, so row count and
  `sum(y)` are conserved exactly. Dimension columns are table-qualified
  (`dim_customer.region`); fact columns keep their names. Accepts
  `{table_name: DataFrame}`; a SQLAlchemy adapter and schema introspection are planned
  for a later release. No new dependencies.

### Guarantees
- Byte-identical `to_dict()` / `summary()` / plot output to v0.2.x when
  `ensemble_report()` has not been run. `to_html()` renders identically (same
  figures, same table) but is *not* byte-identical — the template now ships
  inert ensemble CSS/JS, active only once a report is present.
- `ensemble_report()` never averages predictions or alters the fitted tree —
  it only measures and annotates.
- Deterministic under a fixed `seed`.

### Changed (breaking)
- **Wrong-*type* arguments now raise `TypeError` instead of `ValueError`.**
  Affects five contracts: `fit()`'s `X` and `y` when neither is an accepted
  container, and `ImpactSplitter(...)`'s `numeric_n_bins` (non-integer),
  `consolidate`, and `lookahead` (non-bool). Wrong *values* are unchanged and
  still raise `ValueError` — `numeric_binning_strategy="bad"`,
  `numeric_n_bins=1`, `noise_z=-1`, and every shape/content check in `fit()`.
  `TypeError` is not a subclass of `ValueError`, so `except ValueError` around
  a constructor or `fit()` no longer catches the type case; catch
  `(TypeError, ValueError)` to accept both.
- `ensemble_report()`'s `n_replicates` and `shadow_replicates` follow the same
  rule: non-integer → `TypeError`, out-of-range → `ValueError`. ruff's `TRY004`
  does *not* flag these (the `isinstance` test is `or`-ed with the range test,
  so the branch is not purely type-related) — they were changed to keep the
  convention whole, not because a linter asked. Their validation had no test
  coverage at all; it does now.
- Consequently `numeric_n_bins` no longer raises the same message for two
  different faults: a non-integer says "must be an integer, got a non-integer"
  (`TypeError`), a value below the floor says "must be >= 2" (`ValueError`).
  `n_replicates` / `shadow_replicates` split the same way.
- `schema.SchemaError` still subclasses `ValueError` and is unaffected — a
  schema violation is a wrong value, not a wrong type.

### Internal
- `_TreeNode` gains `split_gain` (internal; feeds ensemble importance).
- `TRY004` is now explicitly selected in ruff's config, so the type/value
  convention above is enforced rather than depending on ruff's default rule
  set, which varies by version.

## [0.2.0] - Unreleased

### Added
- **Pairwise lookahead rescue** (`lookahead=True`, new constructor param): when a
  material node's marginal category tables all net to ~0 (XOR-style interaction
  cancellation), a cross-feature pass re-runs the two-bar sieve on
  crossed category pairs and converts the winning pair into an ordinary
  single-feature split. Fires only where v0.1.0 silently gave up
  (`stop_reason="no_split"` with materiality triggers on), so happy-path fits are
  unchanged. Trace entries record `routing_mode="lookahead_rescue"` plus a
  `rescue` sub-dict (pair, gain, partition, pairs evaluated/skipped). The
  cross-cell significance bar is multiplicity-corrected (z_eff = noise_z +
  √(2 ln K), K = present cross-cells) and singleton cells never count as
  evidence — verified against the full Kaggle suite (byte-identical to the
  pre-rescue baseline where no true interaction exists).
- **Churn flag**: segments and leaf nodes whose positive AND negative gross flows
  each clear `min_global_impact_pct` against their global pools are marked
  `is_churn`. Segments now carry `pos_sum` / `neg_sum`; the payload meta gains
  `n_churn_segments`; `meta.params` gains `lookahead`.
- Churn-aware rendering: tornado hatched gross-range band with
  `net … (gross +…/−…)` labels, icicle dashed churn outline with `net ⇄ ±mass`
  labels, `summary()` gross column + footnote, HTML report churn tile, band,
  ⇄ marker, and Σy⁺/Σy⁻ table columns.
- Benchmarks: three lookahead cases (`xor_pure`, `xor_embedded`,
  `churn_irreducible`) scored separately from the headline battery
  (`python -m benchmarks.run --lookahead --tag <tag>`); pytest bars in
  `tests/test_lookahead_benchmarks.py`.

### Changed (output ordering)
- Segment ranking key is now `max(|Σy|, churn_mass)` where
  `churn_mass = min(Σy⁺, Σy⁻)` for churn segments and 0 otherwise: churn
  segments surface by their offsetting mass instead of sinking on their ~0 net.
  Orderings without churn segments are unchanged.

## [0.1.0] - Unreleased

### Added
- `ImpactSplitter.to_dict()` — JSON-safe payload (meta / tree / segments) feeding every renderer.
- `ImpactSplitter.summary()` and informative `print(model)` — text report with ledger and ranked segment table.
- `ImpactSplitter.plot_segments()` — tornado chart of consolidated segments (matplotlib).
- `ImpactSplitter.plot_tree()` — impact icicle (width ∝ Σ|y|, diverging impact color), replacing the box-and-arrow tree.
- `ImpactSplitter.to_html()` — self-contained interactive HTML report (no CDN, works offline).
- `get_impact_segments()` gains `mean` and `pool_share` columns (existing columns unchanged).

### Removed (breaking)
- CCDS template scaffolding (`impact_split.config`, `features`, `dataset`, `modeling`) and the
  `loguru` / `typer` / `tqdm` / `python-dotenv` dependencies.
- `interactive_force_graph` / `InteractiveForceGraph` (generic, CDN-dependent, not model-bound).
- Old box-and-arrow `plot_tree` and its layout keyword arguments.

### Changed
- `requires-python` widened from `~=3.13.0` to `>=3.10`; CI tests 3.10–3.13.
