# Plan Brief: Relational (star/snowflake) schema input for impact-split

Status: planned, not implemented. Target release `0.4.0`.
Branch: `claude/algorithm-relational-db-support-f9e2dd`.

This is a forward-looking plan, not a record of work done. It is deliberately
outside `docs/docs/` (the MkDocs content root) because it is not user
documentation.

## Goal & success criteria

Widen input coverage from a single flat DataFrame/ndarray to a **star or
snowflake schema**: one fact table carrying the additive target `y` at the
observation grain, plus dimension tables joined many-to-one (chained,
arbitrary depth).

Done when a user can hand over a set of related tables plus a schema spec and
get a flat frame that fits through the existing `ImpactSplitter` unchanged —
with exact sum conservation preserved end to end, and violations of the schema
restriction failing loudly rather than silently corrupting totals.

Success is proven, not measured: the flattened frame must be **identical** to
the equivalent hand-built flat frame, which makes score invariance a theorem
rather than a benchmark result.

## Data sources & grain

- **Grain:** one fact row = one observation = one row of `X`/`y`. Unchanged
  from today.
- **Restriction:** every join in the chain must be **many-to-one** (unique key
  in the parent). Fan-out is out of scope entirely — no one-to-many, no
  fact→children, no aggregation across grains.
- **Snowflake:** allowed, on the grounds that a chain of many-to-one joins is
  itself many-to-one. Each hop is validated independently.
- **Schema source:** an **explicit user-authored spec** is the contract (fact
  table, `y` column, join list with keys, feature column selection). DB
  introspection exists only as a *helper that proposes a spec for review* —
  never as an implicit source of truth. This keeps the core testable with no
  live DB and works on warehouses with undeclared FKs.

Known quality issues handled explicitly: duplicate parent keys (hard failure)
and NULL/orphan FKs (sentinel category). Both are specified under Scope.

## Scope

**In scope (v0.4.0 — core flatten only):**

- Spec object describing fact table, target, join chain, and selected feature
  columns.
- `flatten(tables, spec) -> (DataFrame, provenance)` operating on
  `{table_name: DataFrame}`, joining in pandas.
- Zero new required dependencies.
- Full validation + test suite.
- User then calls `ImpactSplitter().fit(X, y)` as today.

**In scope (v0.5.0 — deferred, but designed for now):**

- Optional `[sql]` extra: **SQLAlchemy Core** adapter taking an engine, pushing
  the join down as one `SELECT`, returning the flat frame into the same code
  path. CI tests against in-memory SQLite — no service, no credentials.
- Introspection helper proposing a spec from PK/FK metadata.

**Out of scope (explicitly, not deferred):**

- Any one-to-many / multi-grain modeling (the Wrobel-1997 direction cited at
  `docs/docs/math.md`).
- SQL push-down of the split search itself; out-of-core / scale work.
- Any change to `ImpactSplitter`, `_prepare_X_y`, the tree logic, or
  `docs/docs/math.md`.

**Edge cases handled:**

| Case | Behavior |
|---|---|
| Duplicate keys in a parent (fan-out) | **Hard fail** pre-join, naming the offending table, key, and an example duplicate |
| Any fan-out from another cause | **Hard fail** post-join via `Σy_flat == Σy_fact` assertion |
| NULL FK or orphan key | **Left join**, dimension columns get an explicit `<unmatched>` sentinel category — conservation exact, and the unattributable pool becomes a routable segment the tree can surface |
| Column-name collisions across dimensions | Impossible by construction — **all joined columns are table-qualified** (`dim_customer.region`), deterministically, regardless of whether a collision exists |

## Validation

Since `fit()` is untouched, proving the flattened frame is identical to the
source frame makes score invariance a theorem — no re-scoring needed.

- **Round-trip frame equality**: normalize each existing benchmark dataset into
  a star/snowflake schema, flatten it back, assert the frame is identical to
  the original. Runs in CI in seconds.
- **Targeted unit tests**: fan-out rejection, the `Σy` assertion, orphan
  sentinel, multi-hop snowflake, and (at 0.5.0) pandas-vs-SQL adapter parity.

Each new claim earns a row in the README "What is guaranteed" table, linked to
the test that would fail if it broke.

## Dependencies & stakeholders

Solo project, no external sign-off. Self-imposed gates from existing repo
culture:

- No new required runtime deps in v0.4.0; SQLAlchemy lands only under an
  optional extra.
- `bandit` will scrutinize the v0.5.0 adapter: table/column names come from a
  user spec and reach SQL, so **identifier quoting and parameter binding are
  non-negotiable**. This is a design requirement, not a review finding to
  handle later.

## Output & delivery

- New module in `impact_split/` exposing the spec type and `flatten`.
- Returns `(DataFrame, provenance)`. Provenance carries the audit trail: tables
  joined, hop order, row counts per hop, unmatched counts per dimension,
  uniqueness checks passed. `ImpactSplitter` never sees it — table attribution
  reaches the reports for free via qualified column names.
- Docs: `README.md` (assumptions/limitations + guarantees table) and
  `docs/docs/getting-started.md`. `math.md` untouched — the algorithm does not
  change.
- Release `0.4.0`, with `CHANGELOG.md` stating plainly that the restriction is
  many-to-one only.

## Risks & assumptions

1. **Naming honesty.** v0.4.0 has no database in it — it accepts DataFrames.
   Calling it "relational database support" in the release notes would
   overstate it. Ship it as *multi-table / star-schema input*, and reserve the
   database framing for 0.5.0 when the adapter lands.
2. **Ledger width regression.** Qualified names (`dim_customer.region=West`)
   are longer, and the text ledger already truncates paths at ~45 chars.
   Expect to widen the column or switch to middle-ellipsis. Left open at
   implementation time.
3. **Sentinel collision.** `<unmatched>` must not collide with a genuine
   category value in the data. Needs a collision check with escalation, not a
   bare string constant. Left open at implementation time.
4. **Correlated sentinels.** One orphan FK marks *every* column of that
   dimension `<unmatched>` on the same rows, producing several
   perfectly-correlated features. The tree may split on any of them
   arbitrarily. Not incorrect, but it can make paths look redundant — worth a
   test that documents the observed behavior rather than pretending it doesn't
   happen.
5. **Untested assumption:** that the round-trip property test is genuinely
   non-trivial. If the normalization step is written as the exact inverse of
   `flatten`, the test proves the two functions are mutual inverses and nothing
   about correctness. The normalizer must be written independently — ideally,
   hand-authored star fixtures for a subset rather than a mechanical inverse.
6. **Deferred-design risk:** the v0.4.0 spec object is public API, and the
   v0.5.0 introspection helper has to *produce* one. If the spec shape turns
   out wrong for introspection, you would be changing a shipped public type.
   Sketch the introspection output shape before freezing the spec, even though
   it is not built yet.

## Skipped / unresolved

None — all seven planning branches were resolved. Two implementation-time
decisions were deliberately left open (ledger truncation strategy,
sentinel-collision escalation rule); both are noted under Risks.
