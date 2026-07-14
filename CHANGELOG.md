# Changelog

All notable changes to this project are documented here.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versioning: [SemVer](https://semver.org/).

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
