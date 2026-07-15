# Lookahead rescue + churn visibility — design spec

- **Date:** 2026-07-15
- **Target version:** 0.2.0
- **Status:** approved (brainstorm 2026-07-15)

## Problem

Offsetting contributors can net to ~0 and disappear. Example: rows worth
+10,000 and −9,999 that share every marginal category. Two distinct failure
modes exist in v0.1.0:

1. **Finder gap (silent).** When the sign of y depends on a *combination* of
   features (XOR-style interaction), every marginal category table nets to ~0,
   every candidate gain is 0, and the node leafs out with
   `stop_reason="no_split"` — even though `positive_trigger` and
   `negative_trigger` both fired. The mass is reported as one ~0 segment.
   Marginal cancellation alone is *not* a miss: the neutral branch is recursed,
   so any single feature that separates the signs will be found. Only
   interaction cancellation escapes.
2. **Display gap.** Nodes and segments that net to ~0 render as invisible or
   inert: the tornado bar has ~0 length, the icicle cell is wide (width is
   already gross Σ|y| via `abs_volume`) but colored neutral gray and labeled
   with the net (`+1`). Gross flows (`pos_sum`/`neg_sum`) are computed and
   stored but never surfaced.

## Decisions already made

- **Single tree stays canon.** Segments remain feature-defined subpopulations
  of one conservation-exact tree. Sign-pool ("gross flow") companion trees are
  rejected: their segments are sign-conditioned objects and select on the
  target.
- **Finder first, then display.** The viz can only show what the tree found;
  residual (irreducible) netted mass is then made visually honest.
- **Rejected alternatives:**
  - *Pairs as first-class candidates at every node* — quadratic fit cost
    everywhere, less readable trees, regression risk on the earned
    0.962/0.815 benchmark, all to fix a rare failure mode.
  - *Post-fit neutral-zone re-mining* — hits the same marginal-cancellation
    wall unless it also uses pairs; it is the rescue relocated post-hoc with
    worse ledger integration.

## Design

### 1. Detection — when the rescue fires

The rescue pass runs only when a node hits today's silent-failure signature:

- materiality triggers fired (`positive_trigger or negative_trigger`), AND
- the best marginal gain is 0 (`best_decision is None or gain == 0.0`,
  today's `no_split` leaf), AND
- the node is **not** at the interaction cap (`at_interaction_cap` is False;
  a rescued split adds an interaction term, which the cap forbids), AND
- `lookahead=True` (new constructor param, default True, same precedent as
  `consolidate`; validated as bool).

Happy-path fits are byte-identical: the rescue never runs unless the tree was
about to give up on a material node.

### 2. Rescue pass — cross-feature sieve on existing machinery

For each unordered feature pair (f, g), f ≠ g:

1. Build the crossed feature `h = codes_f * (max_g + 1) + codes_g` and compute
   cross-category signed sums / counts with one `bincount` (weights =
   `y_centered`, the node's centered-excess signal).
2. Run the **unchanged** sieve on the cross-categories: same
   `delta_centered`, same MAD noise floor (`tau = max(delta, noise_z * sigma *
   sqrt(n_cat))` with sigma from within-cross-category residuals), same
   category-averaged gain `|S_P|/k_P + |S_N|/k_N`.
3. Skip pairs whose crossed cardinality exceeds a hard safety bound
   (constant, e.g. 10,000 present cross-categories) to bound memory.

Pick the best pair by gain. If no pair clears, the node leafs out exactly as
today (`no_split`), and churn flagging (§3) covers visibility.

**Converting the winning pair into a tree split.** The pair (f, g) is
realized as a normal single-feature split on f, so `conditions`,
consolidation, path rendering, and exact conservation are untouched:

- Build the signed-sum matrix `M[a, c]` over f-categories × g-categories
  (from the same bincount).
- Anchor on f's max-norm row; partition f's categories by the **sign of their
  profile's dot product with the anchor row** — agreeing categories route to
  the "positive" branch, opposing to "negative"; categories with near-zero
  row norm route to neutral.
- Guard: the induced split must be non-degenerate (both signed groups
  non-empty and not all rows in one branch); otherwise leaf out.
- The children then split on g through the ordinary marginal sieve (inside
  each group, g's marginals no longer cancel), so signed segments emerge one
  level deeper with plain readable single-feature paths.

The rescue node's branch keys ("positive"/"negative") are structural, not
sign claims — its own branch totals are ~0 by construction; the children
carry the story. The trace entry records `routing_mode="lookahead_rescue"`
plus a `rescue` sub-dict (pairs evaluated, best pair, partition). Depth
accounting is the normal mechanism (the rescue split is an ordinary feature
transition).

Cost: O(F²) pairs × O(n) bincounts, paid only at would-be no_split material
nodes — rare by definition.

Known limitation (accepted): the pairwise rescue cannot see 3-way-or-higher
interactions whose all pairwise margins cancel; churn flagging (§3) still
surfaces them.

### 3. Residual churn — flag what cannot be split

Some offsetting mass is genuinely irreducible (identical feature rows with
±y, or higher-order interactions). It must not vanish:

- `_consolidate_segments` computes per-segment `pos_sum` / `neg_sum` from the
  row masks it already holds (before masks are dropped). Single-leaf segments
  get the same fields.
- A segment (or leaf node) is **churn** when both gross flows independently
  clear materiality: `pos_sum / pos_pool > min_global_impact_pct` AND
  `neg_sum / neg_pool > min_global_impact_pct`.
- Payload additions: `pos_sum`, `neg_sum`, `is_churn` on segments; `is_churn`
  on leaf nodes (nodes already carry `pos_sum`/`neg_sum`);
  `n_churn_segments` in `meta`.

### 4. Display — gross flows where material

- **Segment ranking (breaking output change, documented):** payload sort key
  becomes `max(|total_sum|, churn_mass)` where `churn_mass =
  min(pos_sum, neg_sum)` for churn segments and 0 otherwise. All existing
  orderings are preserved except churn segments surface instead of sinking.
- **Tornado (`plot_segments`):** churn segments render a hatched gross-range
  band from −neg_sum to +pos_sum behind the net bar, labeled like
  `net +1 (gross +10,000 / −9,999)`; footer note clarifies gross bands are
  not additive.
- **Icicle (`plot_icicle`):** churn leaves get the gross annotation in the
  label (`+1 ⇄ ±10K` format) and a dashed edge, parallel to the merged-leaf
  dark-edge convention.
- **`summary()` (text.py) and HTML report (html.py):** same payload-driven
  treatment — churn marker plus gross columns/footnote. Exact rendering
  resolved at plan time after reading both renderers.

### 5. Benchmarks & tests

New DGP cases in the synthetic battery (scored with the existing impact-F1
harness against planted segments):

1. **Pure 2-feature XOR** — rescue must fire at the root and recover the four
   planted signed segments.
2. **Embedded XOR** — a legitimate marginal split first, XOR cancellation at
   an interior node; rescue must fire mid-tree.
3. **Irreducible churn** — offsetting rows with identical feature values;
   the tree must NOT split them, and the segment must be flagged churn.

Hard bar: all three new cases pass AND the full existing suite (8-case
synthetic battery + 10-dataset Kaggle semi-synthetic) holds 0.962/0.815 at
fixed defaults. The rescue's trigger condition makes regressions structurally
unlikely; verified, not assumed.

Unit tests: rescue trigger gating (fires only on no_split + material +
not-at-cap), profile partition correctness on 2×2 XOR, degenerate-split
guards, `lookahead=False` disables the pass, churn flag math, segment gross
sums, conservation exactness with rescue splits present.

### 6. Versioning & docs

- **v0.2.0**: new `lookahead` param, new payload fields, ranking promotion
  rule. API is additive; output ordering change is documented in CHANGELOG.
- README's "three acts" gains the rescue as an explicit extension of the
  sieve act; churn flag documented alongside the output suite.
