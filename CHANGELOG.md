# Changelog

All notable changes to this project are documented here.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versioning: [SemVer](https://semver.org/).

## [0.2.0] - Unreleased

### Added
- **Pairwise lookahead rescue** (`lookahead=True`, new constructor param): when a
  material node's marginal category tables all net to ~0 (XOR-style interaction
  cancellation), a cross-feature pass re-runs the unchanged two-bar sieve on
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
