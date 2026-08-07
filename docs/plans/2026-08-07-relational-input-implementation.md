# Star/Snowflake Relational Input Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `impact_split.schema.flatten()`, which denormalizes a star or snowflake schema into the flat frame `ImpactSplitter.fit()` already accepts, preserving exact sum conservation and failing loudly on any schema violation.

**Architecture:** A single new module `impact_split/schema.py` holding three frozen dataclasses (`Join`, `SchemaSpec`, `FlattenResult`), a structural validator, and a joiner. Joins execute as `dim.set_index(key).reindex(parent_keys)` rather than `pd.merge` — reindexing against a unique index is *structurally incapable* of fan-out and preserves parent row order by construction, so conservation is enforced by the mechanism, not just by an assertion afterwards. `ImpactSplitter`, `_prepare_X_y`, and the tree logic are not touched.

**Tech Stack:** Python ≥3.10, pandas ≥2.1, numpy ≥1.26. No new dependencies. pytest, ruff, mypy.

**Source spec:** `docs/plans/relational-input.md` (committed at `4c98adf`).

## Global Constraints

- **No new runtime dependencies.** v0.4.0 uses only numpy + pandas, both already required.
- **Do not modify** `impact_split/splitter.py`, `impact_split/ensemble.py`, or `impact_split/viz/*`. If you believe you need to, stop and raise it.
- **Do not modify** `docs/docs/math.md` — the algorithm does not change.
- Line length 99 (`ruff`, configured in `pyproject.toml`).
- Every module starts with a one-line docstring, then `from __future__ import annotations`.
- Imports follow ruff isort with `force-sort-within-sections = true` and `known-first-party = ["impact_split"]`.
- Type annotations on every public function; `mypy` runs over `packages = ["impact_split"]`.
- Run `make lint` and `make test` before each commit. Both must pass.
- Commit messages use Conventional Commits (`feat:`, `test:`, `docs:`), matching existing history.

## Spec refinements decided during planning

These extend the source brief. They are decisions, not open questions — but flag them to the user if any looks wrong.

1. **Return type is `FlattenResult`, not a 2-tuple.** The brief said `-> (DataFrame, provenance)`, but the caller also needs `y`. A 3-tuple is worse than a named object.
2. **Fact-table feature columns keep bare names; only dimension columns are qualified.** So a zero-join spec reproduces the source table exactly. Fact feature names containing `.` are rejected, which keeps the two namespaces disjoint.
3. **A dimension's join key is not selectable as a feature.** After `reindex`, the key becomes the index and its value for an orphan row is the unresolvable key itself, which conflicts with the sentinel rule. Users who want the key as a feature select the parent's FK column instead — it lives in the parent table and is selectable there.
4. **Each table may be joined at most once.** Two joins naming the same table would produce colliding qualified column names and an ambiguous alignment.
5. **A dimension with unmatched rows may not have float feature columns.** Inserting a string sentinel into a float column silently flips it from binned-numeric to categorical inside `_prepare_X_y` — a semantic change the user never asked for. Raise and tell them to pre-bin. This matches the existing posture in README "Assumptions and limitations" (*discretize continuous features before fitting*) and introduces no tuned constant.

## File Structure

| File | Responsibility |
|---|---|
| `impact_split/schema.py` (create) | Spec dataclasses, structural validation, join execution, provenance. The whole feature. ~320 lines — smaller than `splitter.py`, consistent with the repo's flat-module style. |
| `impact_split/__init__.py` (modify) | Export `Join`, `SchemaSpec`, `FlattenResult`, `flatten`. |
| `tests/test_schema.py` (create) | Spec validation and join behavior: happy path, fan-out, orphans, snowflake, provenance. |
| `tests/test_schema_roundtrip.py` (create) | Round-trip frame equality over the synthetic benchmark battery, plus an independently-written normalizer. |
| `README.md` (modify) | Guarantees-table row, assumptions/limitations entry. |
| `docs/docs/getting-started.md` (modify) | Usage section. |
| `CHANGELOG.md` (modify) | 0.4.0 entry. |

---

### Task 1: Spec dataclasses and structural validation

Pure spec-graph checking. No pandas joins yet — everything here is answerable from table/column names and the join graph alone.

**Files:**
- Create: `impact_split/schema.py`
- Test: `tests/test_schema.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `Join(table: str, left: str, right: str, columns: tuple[str, ...] | None = None, parent: str | None = None)` — frozen dataclass. `parent=None` means the fact table.
  - `SchemaSpec(fact: str, target: str, features: tuple[str, ...] = (), joins: tuple[Join, ...] = ())` — frozen dataclass.
  - `SchemaError(ValueError)` — every validation failure raised by this module.
  - `validate_spec(tables: dict[str, pd.DataFrame], spec: SchemaSpec) -> list[Join]` — raises `SchemaError` on any violation, otherwise returns the joins in topological order (parents before children).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_schema.py`:

```python
"""Schema spec validation and star/snowflake flattening."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from impact_split.schema import Join, SchemaError, SchemaSpec, validate_spec


def _star() -> tuple[dict[str, pd.DataFrame], SchemaSpec]:
    """A minimal valid star: 4 fact rows, one dimension, one feature each side."""
    tables = {
        "fact": pd.DataFrame(
            {"cust_key": [1, 2, 1, 3], "channel": ["a", "b", "a", "b"], "y": [10.0, -2.0, 3.0, 5.0]}
        ),
        "dim_cust": pd.DataFrame({"cust_key": [1, 2, 3], "region": ["W", "E", "W"]}),
    }
    spec = SchemaSpec(
        fact="fact",
        target="y",
        features=("channel",),
        joins=(Join(table="dim_cust", left="cust_key", right="cust_key"),),
    )
    return tables, spec


def test_validate_spec_accepts_a_valid_star():
    tables, spec = _star()
    assert [j.table for j in validate_spec(tables, spec)] == ["dim_cust"]


def test_validate_spec_orders_snowflake_joins_parents_first():
    tables = {
        "fact": pd.DataFrame({"cust_key": [1], "y": [1.0]}),
        "dim_cust": pd.DataFrame({"cust_key": [1], "geo_key": [7], "name": ["x"]}),
        "dim_geo": pd.DataFrame({"geo_key": [7], "country": ["PH"]}),
    }
    # Deliberately declared child-first to prove the topological sort works.
    spec = SchemaSpec(
        fact="fact",
        target="y",
        joins=(
            Join(table="dim_geo", left="geo_key", right="geo_key", parent="dim_cust"),
            Join(table="dim_cust", left="cust_key", right="cust_key"),
        ),
    )
    assert [j.table for j in validate_spec(tables, spec)] == ["dim_cust", "dim_geo"]


def test_validate_spec_rejects_missing_fact_table():
    tables, spec = _star()
    with pytest.raises(SchemaError, match="fact table 'nope' not found"):
        validate_spec(tables, SchemaSpec(fact="nope", target="y"))


def test_validate_spec_rejects_missing_target_column():
    tables, spec = _star()
    with pytest.raises(SchemaError, match="target column 'missing' not found in 'fact'"):
        validate_spec(tables, SchemaSpec(fact="fact", target="missing"))


def test_validate_spec_rejects_missing_feature_column():
    tables, _ = _star()
    spec = SchemaSpec(fact="fact", target="y", features=("nope",))
    with pytest.raises(SchemaError, match="feature column 'nope' not found in 'fact'"):
        validate_spec(tables, spec)


def test_validate_spec_rejects_dotted_fact_feature_name():
    tables = {"fact": pd.DataFrame({"a.b": [1], "y": [1.0]})}
    spec = SchemaSpec(fact="fact", target="y", features=("a.b",))
    with pytest.raises(SchemaError, match="must not contain '.'"):
        validate_spec(tables, spec)


def test_validate_spec_rejects_unreachable_parent():
    tables, _ = _star()
    spec = SchemaSpec(
        fact="fact",
        target="y",
        joins=(Join(table="dim_cust", left="cust_key", right="cust_key", parent="ghost"),),
    )
    with pytest.raises(SchemaError, match="unreachable from the fact table"):
        validate_spec(tables, spec)


def test_validate_spec_rejects_duplicate_join_target():
    tables, _ = _star()
    spec = SchemaSpec(
        fact="fact",
        target="y",
        joins=(
            Join(table="dim_cust", left="cust_key", right="cust_key"),
            Join(table="dim_cust", left="cust_key", right="cust_key"),
        ),
    )
    with pytest.raises(SchemaError, match="joined more than once"):
        validate_spec(tables, spec)


def test_validate_spec_rejects_selecting_the_join_key_as_a_feature():
    tables, _ = _star()
    spec = SchemaSpec(
        fact="fact",
        target="y",
        joins=(
            Join(table="dim_cust", left="cust_key", right="cust_key", columns=("cust_key",)),
        ),
    )
    with pytest.raises(SchemaError, match="join key 'cust_key' cannot be selected"):
        validate_spec(tables, spec)


def test_validate_spec_rejects_non_numeric_target():
    tables = {"fact": pd.DataFrame({"y": ["a", "b"]})}
    with pytest.raises(SchemaError, match="target column 'y' must be numeric"):
        validate_spec(tables, SchemaSpec(fact="fact", target="y"))
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
python -m pytest tests/test_schema.py -v
```

Expected: collection error — `ModuleNotFoundError: No module named 'impact_split.schema'`.

- [ ] **Step 3: Write the implementation**

Create `impact_split/schema.py`:

```python
"""Star/snowflake schema denormalization into the flat frame ImpactSplitter accepts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype


class SchemaError(ValueError):
    """A schema spec or its data violates the star/snowflake restriction."""


@dataclass(frozen=True)
class Join:
    """One many-to-one hop from a parent table to a dimension table.

    ``left`` is the foreign-key column in the parent; ``right`` is the key
    column in ``table``, which must be unique and non-null. ``columns``
    selects which of the dimension's columns become features (``None`` means
    every column except ``right``). ``parent`` names the table this hop
    departs from; ``None`` means the fact table.
    """

    table: str
    left: str
    right: str
    columns: tuple[str, ...] | None = None
    parent: str | None = None


@dataclass(frozen=True)
class SchemaSpec:
    """The declarative contract: what to join, and what becomes a feature."""

    fact: str
    target: str
    features: tuple[str, ...] = ()
    joins: tuple[Join, ...] = ()


def validate_spec(tables: dict[str, pd.DataFrame], spec: SchemaSpec) -> list[Join]:
    """Check the spec against the supplied tables; return joins parents-first.

    Raises :class:`SchemaError` naming the exact offending table or column.
    This checks structure only — key uniqueness is a property of the data and
    is enforced during the join itself.
    """
    if spec.fact not in tables:
        raise SchemaError(f"fact table {spec.fact!r} not found in tables.")
    fact = tables[spec.fact]

    if spec.target not in fact.columns:
        raise SchemaError(f"target column {spec.target!r} not found in {spec.fact!r}.")
    if not is_numeric_dtype(fact[spec.target]):
        raise SchemaError(f"target column {spec.target!r} must be numeric.")

    for name in spec.features:
        if name not in fact.columns:
            raise SchemaError(f"feature column {name!r} not found in {spec.fact!r}.")
        if "." in name:
            raise SchemaError(
                f"fact feature column {name!r} must not contain '.' — the dot is reserved "
                "for table-qualified dimension columns."
            )

    seen: list[str] = []
    for join in spec.joins:
        if join.table in seen or join.table == spec.fact:
            raise SchemaError(f"table {join.table!r} is joined more than once.")
        seen.append(join.table)
        if join.table not in tables:
            raise SchemaError(f"dimension table {join.table!r} not found in tables.")
        dim = tables[join.table]
        if join.right not in dim.columns:
            raise SchemaError(f"join key {join.right!r} not found in {join.table!r}.")
        for name in join.columns or ():
            if name not in dim.columns:
                raise SchemaError(f"column {name!r} not found in {join.table!r}.")
            if name == join.right:
                raise SchemaError(
                    f"join key {join.right!r} cannot be selected as a feature of "
                    f"{join.table!r}; select the parent's foreign-key column instead."
                )

    return _topological_order(tables, spec)


def _topological_order(tables: dict[str, pd.DataFrame], spec: SchemaSpec) -> list[Join]:
    """Order joins so every parent is resolved before its children."""
    resolved = {spec.fact}
    pending = list(spec.joins)
    ordered: list[Join] = []

    while pending:
        progressed = False
        for join in list(pending):
            parent = join.parent or spec.fact
            if parent not in resolved:
                continue
            if join.left not in tables[parent].columns:
                raise SchemaError(f"foreign key {join.left!r} not found in {parent!r}.")
            ordered.append(join)
            resolved.add(join.table)
            pending.remove(join)
            progressed = True
        if not progressed:
            names = ", ".join(repr(j.table) for j in pending)
            raise SchemaError(
                f"joins are unreachable from the fact table (cycle or unknown parent): {names}."
            )

    return ordered
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
python -m pytest tests/test_schema.py -v
```

Expected: 10 passed.

- [ ] **Step 5: Lint and commit**

```bash
make lint && python -m pytest tests/test_schema.py -q
```

```bash
git add impact_split/schema.py tests/test_schema.py
git commit -m "feat(schema): spec dataclasses and structural validation"
```

---

### Task 2: Zero-join flatten and the conservation assertion

The degenerate case — a "star" with no dimensions — must reproduce the source table exactly. Getting this right first pins down the return type and the conservation check before joins complicate anything.

**Files:**
- Modify: `impact_split/schema.py`
- Test: `tests/test_schema.py`

**Interfaces:**
- Consumes: `Join`, `SchemaSpec`, `SchemaError`, `validate_spec` from Task 1.
- Produces:
  - `FlattenResult(X: pd.DataFrame, y: np.ndarray, provenance: dict[str, Any])` — frozen dataclass.
  - `flatten(tables: dict[str, pd.DataFrame], spec: SchemaSpec) -> FlattenResult`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_schema.py`:

```python
from impact_split.schema import FlattenResult, flatten  # noqa: E402  (append-to-file ordering)


def test_flatten_with_no_joins_reproduces_the_fact_table():
    tables = {
        "fact": pd.DataFrame(
            {"channel": ["a", "b", "a"], "size": ["S", "M", "L"], "y": [1.0, -2.0, 3.5]}
        )
    }
    spec = SchemaSpec(fact="fact", target="y", features=("channel", "size"))

    result = flatten(tables, spec)

    assert isinstance(result, FlattenResult)
    pd.testing.assert_frame_equal(result.X, tables["fact"][["channel", "size"]])
    assert np.array_equal(result.y, np.array([1.0, -2.0, 3.5]))


def test_flatten_conserves_the_target_sum_exactly():
    tables = {"fact": pd.DataFrame({"channel": ["a", "b"], "y": [0.1, 0.2]})}
    spec = SchemaSpec(fact="fact", target="y", features=("channel",))

    result = flatten(tables, spec)

    assert result.y.sum() == tables["fact"]["y"].to_numpy(dtype=float).sum()
    assert result.provenance["target_sum"] == float(tables["fact"]["y"].sum())


def test_flatten_rejects_a_target_containing_nulls():
    tables = {"fact": pd.DataFrame({"channel": ["a", "b"], "y": [1.0, np.nan]})}
    spec = SchemaSpec(fact="fact", target="y", features=("channel",))
    with pytest.raises(SchemaError, match="target column 'y' contains missing values"):
        flatten(tables, spec)


def test_flatten_rejects_a_fact_feature_containing_nulls():
    tables = {"fact": pd.DataFrame({"channel": ["a", None], "y": [1.0, 2.0]})}
    spec = SchemaSpec(fact="fact", target="y", features=("channel",))
    with pytest.raises(SchemaError, match="fact feature column 'channel' contains missing values"):
        flatten(tables, spec)


def test_flatten_requires_at_least_one_feature():
    tables = {"fact": pd.DataFrame({"y": [1.0]})}
    with pytest.raises(SchemaError, match="spec selects no feature columns"):
        flatten(tables, SchemaSpec(fact="fact", target="y"))
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
python -m pytest tests/test_schema.py -k flatten -v
```

Expected: `ImportError: cannot import name 'FlattenResult'`.

- [ ] **Step 3: Write the implementation**

Add to `impact_split/schema.py`, after `SchemaSpec`:

```python
@dataclass(frozen=True)
class FlattenResult:
    """The denormalized frame, the target, and the audit trail of how it was built."""

    X: pd.DataFrame
    y: np.ndarray
    provenance: dict[str, Any] = field(default_factory=dict)
```

Add at the end of the module:

```python
def flatten(tables: dict[str, pd.DataFrame], spec: SchemaSpec) -> FlattenResult:
    """Denormalize a star or snowflake schema into a flat frame plus target.

    Every join must be many-to-one: the dimension's key must be unique and
    non-null. Fact rows whose foreign key does not resolve are kept, with the
    dimension's columns set to an unmatched sentinel, so the row count and
    ``sum(y)`` of the fact table are preserved exactly.
    """
    validate_spec(tables, spec)
    fact = tables[spec.fact]

    target = fact[spec.target]
    if target.isna().any():
        raise SchemaError(f"target column {spec.target!r} contains missing values.")
    y = np.asarray(target, dtype=float)

    columns: dict[str, pd.Series] = {}
    provenance: dict[str, Any] = {
        "fact": spec.fact,
        "n_rows": int(len(fact)),
        "target": spec.target,
        "target_sum": float(y.sum()),
        "joins": [],
        "columns": {},
        "sentinel": None,
    }

    for name in spec.features:
        series = fact[name]
        if series.isna().any():
            raise SchemaError(f"fact feature column {name!r} contains missing values.")
        columns[name] = series.reset_index(drop=True)
        provenance["columns"][name] = {"table": spec.fact, "column": name}

    if not columns:
        raise SchemaError("spec selects no feature columns.")

    X = pd.DataFrame(columns)
    if len(X) != len(fact):
        raise SchemaError(
            f"row count changed during flattening: {len(fact)} fact rows produced {len(X)}."
        )
    return FlattenResult(X=X, y=y, provenance=provenance)
```

`validate_spec` is called for its side effect (raising) here; Task 3 binds its return value to drive the join loop. Do not add an unused-variable suppression.

- [ ] **Step 4: Run the tests to verify they pass**

```bash
python -m pytest tests/test_schema.py -v
```

Expected: 15 passed.

- [ ] **Step 5: Lint and commit**

```bash
make lint && python -m pytest tests/test_schema.py -q
```

```bash
git add impact_split/schema.py tests/test_schema.py
git commit -m "feat(schema): zero-join flatten with exact target conservation"
```

---

### Task 3: Single many-to-one join with qualified naming

**Files:**
- Modify: `impact_split/schema.py`
- Test: `tests/test_schema.py`

**Interfaces:**
- Consumes: everything from Tasks 1–2.
- Produces: `_align_dimension(parent_keys: pd.Series, dim: pd.DataFrame, join: Join) -> pd.DataFrame` — the dimension's selected columns, reindexed to the parent's row order, unqualified. Missing keys yield NaN rows (handled in Task 5).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_schema.py`:

```python
def test_flatten_joins_a_dimension_and_qualifies_its_columns():
    tables, spec = _star()

    result = flatten(tables, spec)

    assert list(result.X.columns) == ["channel", "dim_cust.region"]
    assert list(result.X["dim_cust.region"]) == ["W", "E", "W", "W"]
    assert list(result.X["channel"]) == ["a", "b", "a", "b"]


def test_flatten_preserves_fact_row_order_under_a_join():
    tables = {
        # Keys deliberately out of dimension order.
        "fact": pd.DataFrame({"k": [3, 1, 2], "y": [1.0, 2.0, 3.0]}),
        "dim": pd.DataFrame({"k": [1, 2, 3], "label": ["one", "two", "three"]}),
    }
    spec = SchemaSpec(fact="fact", target="y", joins=(Join(table="dim", left="k", right="k"),))

    result = flatten(tables, spec)

    assert list(result.X["dim.label"]) == ["three", "one", "two"]
    assert np.array_equal(result.y, np.array([1.0, 2.0, 3.0]))


def test_flatten_selects_only_the_requested_dimension_columns():
    tables = {
        "fact": pd.DataFrame({"k": [1], "y": [1.0]}),
        "dim": pd.DataFrame({"k": [1], "keep": ["x"], "drop": ["z"]}),
    }
    spec = SchemaSpec(
        fact="fact",
        target="y",
        joins=(Join(table="dim", left="k", right="k", columns=("keep",)),),
    )

    result = flatten(tables, spec)

    assert list(result.X.columns) == ["dim.keep"]


def test_flatten_defaults_to_every_dimension_column_except_the_key():
    tables = {
        "fact": pd.DataFrame({"k": [1], "y": [1.0]}),
        "dim": pd.DataFrame({"k": [1], "a": ["x"], "b": ["z"]}),
    }
    spec = SchemaSpec(fact="fact", target="y", joins=(Join(table="dim", left="k", right="k"),))

    result = flatten(tables, spec)

    assert list(result.X.columns) == ["dim.a", "dim.b"]


def test_flatten_rejects_nulls_in_a_dimension_feature_column():
    tables = {
        "fact": pd.DataFrame({"k": [1], "y": [1.0]}),
        "dim": pd.DataFrame({"k": [1], "a": [None]}),
    }
    spec = SchemaSpec(fact="fact", target="y", joins=(Join(table="dim", left="k", right="k"),))
    with pytest.raises(SchemaError, match="dimension column 'dim.a' contains missing values"):
        flatten(tables, spec)
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
python -m pytest tests/test_schema.py -k "joins_a_dimension or row_order or requested_dimension or every_dimension or dimension_feature" -v
```

Expected: FAIL — `KeyError: 'dim_cust.region'`, because `flatten` ignores `ordered`.

- [ ] **Step 3: Write the implementation**

Add to `impact_split/schema.py`:

```python
def _selected_columns(dim: pd.DataFrame, join: Join) -> tuple[str, ...]:
    """Dimension columns that become features: the declared set, or all but the key."""
    if join.columns is not None:
        return tuple(join.columns)
    return tuple(c for c in dim.columns if c != join.right)


def _align_dimension(parent_keys: pd.Series, dim: pd.DataFrame, join: Join) -> pd.DataFrame:
    """Reindex ``dim`` onto the parent's row order, one dimension row per parent row.

    Reindexing against a unique index cannot fan out and cannot reorder, so
    the returned frame has exactly ``len(parent_keys)`` rows in the parent's
    order. Unresolvable keys produce all-NaN rows.
    """
    selected = _selected_columns(dim, join)
    indexed = dim.set_index(join.right)[list(selected)]
    aligned = indexed.reindex(pd.Index(parent_keys))
    aligned.index = pd.RangeIndex(len(aligned))
    return aligned
```

Change the first line of `flatten` to bind the return value:

```python
    ordered = validate_spec(tables, spec)
```

Then insert this between the fact-feature loop and the `if not columns` guard:

```python
    aligned_tables: dict[str, pd.DataFrame] = {spec.fact: fact.reset_index(drop=True)}

    for join in ordered:
        parent_name = join.parent or spec.fact
        dim = tables[join.table]
        # Align every dimension column once: a chained hop needs this frame to
        # find its own foreign key, which the user's feature selection may omit.
        full = _align_dimension(
            aligned_tables[parent_name][join.left],
            dim,
            Join(table=join.table, left=join.left, right=join.right, columns=tuple(dim.columns)),
        )
        aligned_tables[join.table] = full

        for name in _selected_columns(dim, join):
            qualified = f"{join.table}.{name}"
            if full[name].isna().any():
                raise SchemaError(f"dimension column {qualified!r} contains missing values.")
            columns[qualified] = full[name]
            provenance["columns"][qualified] = {"table": join.table, "column": name}

        provenance["joins"].append(
            {"table": join.table, "parent": parent_name, "left": join.left, "right": join.right}
        )
```

The explicit `columns=tuple(dim.columns)` argument bypasses `_selected_columns`, so the key column survives in `full` for child hops to key off. The feature loop then takes only the user's selection from that same frame — one alignment per join, not two.

Note: the NaN check above currently conflates "the dimension row itself has a null" with "the key did not resolve". Task 5 separates them.

- [ ] **Step 4: Run the tests to verify they pass**

```bash
python -m pytest tests/test_schema.py -v
```

Expected: 20 passed.

- [ ] **Step 5: Lint and commit**

```bash
make lint && python -m pytest tests/test_schema.py -q
```

```bash
git add impact_split/schema.py tests/test_schema.py
git commit -m "feat(schema): single many-to-one join with table-qualified columns"
```

---

### Task 4: Fan-out detection

The conservation-critical task. A duplicate dimension key must fail with a message naming the table, the key, and an example — never silently duplicate `y`.

**Files:**
- Modify: `impact_split/schema.py`
- Test: `tests/test_schema.py`

**Interfaces:**
- Consumes: everything from Tasks 1–3.
- Produces: `_assert_unique_key(dim: pd.DataFrame, join: Join) -> None` — raises `SchemaError` on duplicate or null keys.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_schema.py`:

```python
def test_flatten_rejects_a_dimension_with_duplicate_keys():
    tables = {
        "fact": pd.DataFrame({"k": [1, 2], "y": [1.0, 2.0]}),
        "dim": pd.DataFrame({"k": [1, 1, 2], "label": ["a", "b", "c"]}),
    }
    spec = SchemaSpec(fact="fact", target="y", joins=(Join(table="dim", left="k", right="k"),))

    with pytest.raises(SchemaError) as excinfo:
        flatten(tables, spec)

    message = str(excinfo.value)
    assert "dim" in message
    assert "'k'" in message
    assert "not unique" in message
    assert "1" in message  # the offending key value is named


def test_flatten_rejects_a_dimension_key_containing_nulls():
    tables = {
        "fact": pd.DataFrame({"k": [1], "y": [1.0]}),
        "dim": pd.DataFrame({"k": [None], "label": ["a"]}),
    }
    spec = SchemaSpec(fact="fact", target="y", joins=(Join(table="dim", left="k", right="k"),))
    with pytest.raises(SchemaError, match="join key 'k' in 'dim' contains missing values"):
        flatten(tables, spec)


def test_flatten_rejects_duplicate_keys_deep_in_a_snowflake():
    tables = {
        "fact": pd.DataFrame({"ck": [1], "y": [1.0]}),
        "dim_cust": pd.DataFrame({"ck": [1], "gk": [9], "name": ["n"]}),
        "dim_geo": pd.DataFrame({"gk": [9, 9], "country": ["PH", "US"]}),
    }
    spec = SchemaSpec(
        fact="fact",
        target="y",
        joins=(
            Join(table="dim_cust", left="ck", right="ck"),
            Join(table="dim_geo", left="gk", right="gk", parent="dim_cust"),
        ),
    )
    with pytest.raises(SchemaError, match="join key 'gk' in 'dim_geo' is not unique"):
        flatten(tables, spec)


def test_flatten_asserts_row_count_and_sum_survive_the_join():
    """Belt-and-braces: the mechanism cannot fan out, but the assertion still runs."""
    tables, spec = _star()
    result = flatten(tables, spec)
    assert len(result.X) == len(tables["fact"])
    assert result.y.sum() == tables["fact"]["y"].sum()
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
python -m pytest tests/test_schema.py -k "duplicate_keys or key_containing_nulls or snowflake" -v
```

Expected: FAIL — pandas raises its own `ValueError` about reindexing on a duplicate axis, not a `SchemaError` naming the table.

- [ ] **Step 3: Write the implementation**

Add to `impact_split/schema.py`:

```python
def _assert_unique_key(dim: pd.DataFrame, join: Join) -> None:
    """Enforce the many-to-one restriction: the dimension key is unique and non-null."""
    key = dim[join.right]
    if key.isna().any():
        raise SchemaError(f"join key {join.right!r} in {join.table!r} contains missing values.")
    duplicated = key[key.duplicated(keep=False)]
    if not duplicated.empty:
        example = duplicated.iloc[0]
        n_dupes = int(duplicated.nunique())
        raise SchemaError(
            f"join key {join.right!r} in {join.table!r} is not unique — "
            f"{n_dupes} duplicated key value(s), e.g. {example!r}. impact-split supports "
            "many-to-one joins only; a fan-out join would duplicate rows and break sum "
            "conservation."
        )
```

In `flatten`, call it as the first statement of the `for join in ordered:` body:

```python
    for join in ordered:
        parent_name = join.parent or spec.fact
        dim = tables[join.table]
        _assert_unique_key(dim, join)
```

And add the post-join guard immediately before `return FlattenResult(...)`, replacing the existing row-count check:

```python
    if len(X) != len(fact):
        raise SchemaError(
            f"row count changed during flattening: {len(fact)} fact rows produced {len(X)}."
        )
    if float(y.sum()) != provenance["target_sum"]:
        raise SchemaError("sum(y) changed during flattening — conservation violated.")
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
python -m pytest tests/test_schema.py -v
```

Expected: 24 passed.

- [ ] **Step 5: Lint and commit**

```bash
make lint && python -m pytest tests/test_schema.py -q
```

```bash
git add impact_split/schema.py tests/test_schema.py
git commit -m "feat(schema): reject fan-out joins with a named-key error"
```

---

### Task 5: Orphan foreign keys and the unmatched sentinel

**Files:**
- Modify: `impact_split/schema.py`
- Test: `tests/test_schema.py`

**Interfaces:**
- Consumes: everything from Tasks 1–4.
- Produces: `_resolve_sentinel(frames: list[pd.DataFrame]) -> str` — returns `"<unmatched>"`, escalating to `"<unmatched_1>"`, `"<unmatched_2>"`, … if that literal already occurs in any selected dimension column.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_schema.py`:

```python
def test_flatten_keeps_orphan_rows_with_an_unmatched_sentinel():
    tables = {
        "fact": pd.DataFrame({"k": [1, 99], "y": [10.0, -4.0]}),
        "dim": pd.DataFrame({"k": [1], "region": ["W"]}),
    }
    spec = SchemaSpec(fact="fact", target="y", joins=(Join(table="dim", left="k", right="k"),))

    result = flatten(tables, spec)

    assert list(result.X["dim.region"]) == ["W", "<unmatched>"]
    assert result.y.sum() == 6.0  # conservation: the orphan row is not dropped


def test_flatten_treats_a_null_foreign_key_as_unmatched():
    tables = {
        "fact": pd.DataFrame({"k": [1.0, np.nan], "y": [1.0, 2.0]}),
        "dim": pd.DataFrame({"k": [1.0], "region": ["W"]}),
    }
    spec = SchemaSpec(fact="fact", target="y", joins=(Join(table="dim", left="k", right="k"),))

    result = flatten(tables, spec)

    assert list(result.X["dim.region"]) == ["W", "<unmatched>"]


def test_flatten_records_unmatched_counts_and_mass_in_provenance():
    tables = {
        "fact": pd.DataFrame({"k": [1, 99, 98], "y": [10.0, -4.0, -6.0]}),
        "dim": pd.DataFrame({"k": [1], "region": ["W"]}),
    }
    spec = SchemaSpec(fact="fact", target="y", joins=(Join(table="dim", left="k", right="k"),))

    result = flatten(tables, spec)

    entry = result.provenance["joins"][0]
    assert entry["n_unmatched"] == 2
    assert entry["unmatched_sum"] == -10.0
    assert result.provenance["sentinel"] == "<unmatched>"


def test_flatten_escalates_the_sentinel_when_it_collides_with_real_data():
    tables = {
        "fact": pd.DataFrame({"k": [1, 99], "y": [1.0, 2.0]}),
        "dim": pd.DataFrame({"k": [1], "region": ["<unmatched>"]}),
    }
    spec = SchemaSpec(fact="fact", target="y", joins=(Join(table="dim", left="k", right="k"),))

    result = flatten(tables, spec)

    assert list(result.X["dim.region"]) == ["<unmatched>", "<unmatched_1>"]
    assert result.provenance["sentinel"] == "<unmatched_1>"


def test_flatten_rejects_unmatched_rows_against_a_float_dimension_column():
    tables = {
        "fact": pd.DataFrame({"k": [1, 99], "y": [1.0, 2.0]}),
        "dim": pd.DataFrame({"k": [1], "score": [0.5]}),
    }
    spec = SchemaSpec(fact="fact", target="y", joins=(Join(table="dim", left="k", right="k"),))

    with pytest.raises(SchemaError, match="float column 'dim.score'"):
        flatten(tables, spec)


def test_flatten_allows_float_dimension_columns_when_everything_matches():
    tables = {
        "fact": pd.DataFrame({"k": [1, 1], "y": [1.0, 2.0]}),
        "dim": pd.DataFrame({"k": [1], "score": [0.5]}),
    }
    spec = SchemaSpec(fact="fact", target="y", joins=(Join(table="dim", left="k", right="k"),))

    result = flatten(tables, spec)

    assert list(result.X["dim.score"]) == [0.5, 0.5]
    assert result.X["dim.score"].dtype == float
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
python -m pytest tests/test_schema.py -k "orphan or unmatched or sentinel or float_dimension" -v
```

Expected: FAIL — the Task 3 null check raises `dimension column 'dim.region' contains missing values` instead of substituting a sentinel.

- [ ] **Step 3: Write the implementation**

Add to `impact_split/schema.py`, importing `is_float_dtype`:

```python
from pandas.api.types import is_float_dtype, is_numeric_dtype
```

```python
_SENTINEL_BASE = "<unmatched>"


def _resolve_sentinel(frames: list[pd.DataFrame]) -> str:
    """Pick an unmatched marker that does not already occur in the data."""
    present: set[Any] = set()
    for frame in frames:
        for name in frame.columns:
            present.update(frame[name].dropna().unique().tolist())
    if _SENTINEL_BASE not in present:
        return _SENTINEL_BASE
    suffix = 1
    while f"<unmatched_{suffix}>" in present:
        suffix += 1
    return f"<unmatched_{suffix}>"
```

Restructure `flatten`'s join loop so alignment happens first, the sentinel is resolved once against every aligned dimension frame, and substitution happens afterwards. Replace the whole `for join in ordered:` block with:

```python
    aligned_tables: dict[str, pd.DataFrame] = {spec.fact: fact.reset_index(drop=True)}
    aligned_features: dict[str, tuple[Join, pd.DataFrame]] = {}
    unmatched_masks: dict[str, np.ndarray] = {}

    for join in ordered:
        parent_name = join.parent or spec.fact
        dim = tables[join.table]
        _assert_unique_key(dim, join)

        parent_keys = aligned_tables[parent_name][join.left]
        full = _align_dimension(
            parent_keys,
            dim,
            Join(table=join.table, left=join.left, right=join.right, columns=tuple(dim.columns)),
        )
        # A key resolves iff its row came back; every dimension row is non-null
        # in the key column (enforced above), so the key column is the witness.
        unmatched = np.asarray(full[join.right].isna())
        aligned_tables[join.table] = full

        selected = _selected_columns(dim, join)
        aligned_features[join.table] = (join, full[list(selected)])
        unmatched_masks[join.table] = unmatched

        provenance["joins"].append(
            {
                "table": join.table,
                "parent": parent_name,
                "left": join.left,
                "right": join.right,
                "n_unmatched": int(unmatched.sum()),
                "unmatched_sum": float(y[unmatched].sum()),
            }
        )

    sentinel = _resolve_sentinel([frame for _, frame in aligned_features.values()])
    provenance["sentinel"] = sentinel

    for table, (join, frame) in aligned_features.items():
        unmatched = unmatched_masks[table]
        source = tables[table]
        for name in frame.columns:
            qualified = f"{table}.{name}"
            series = frame[name]
            if unmatched.any():
                if is_float_dtype(source[name]):
                    raise SchemaError(
                        f"{int(unmatched.sum())} fact row(s) do not match {table!r}, but "
                        f"float column {qualified!r} cannot carry the {sentinel!r} marker "
                        "without silently becoming categorical. Pre-bin this column into "
                        "labels, or repair the foreign key."
                    )
                series = series.astype(object)
                series[unmatched] = sentinel
            if series.isna().any():
                raise SchemaError(f"dimension column {qualified!r} contains missing values.")
            columns[qualified] = series.reset_index(drop=True)
            provenance["columns"][qualified] = {"table": table, "column": name}
```

Two details that matter:

- `unmatched_sum` is computed against `y`, which is already aligned to fact rows — that is why the mask is a plain boolean array over the fact grain, not a pandas index.
- `_resolve_sentinel` runs over the *aligned* frames, not the source dimensions, so it only considers values that actually reach the output.

Delete the now-dead null check that Task 3 placed inside the join loop.

- [ ] **Step 4: Run the tests to verify they pass**

```bash
python -m pytest tests/test_schema.py -v
```

Expected: 30 passed.

- [ ] **Step 5: Lint and commit**

```bash
make lint && python -m pytest tests/test_schema.py -q
```

```bash
git add impact_split/schema.py tests/test_schema.py
git commit -m "feat(schema): keep orphan rows via a collision-safe unmatched sentinel"
```

---

### Task 6: Snowflake chains end to end

Tasks 1–5 built multi-hop support incrementally. This task proves it works as a whole and pins the behavior that a snowflake preserves conservation across every hop, including when an orphan appears mid-chain.

**Files:**
- Test: `tests/test_schema.py` (test-only task — if implementation changes are needed, the earlier tasks were wrong)

**Interfaces:**
- Consumes: everything from Tasks 1–5. Produces nothing new.

- [ ] **Step 1: Write the tests**

Append to `tests/test_schema.py`:

```python
def _snowflake() -> tuple[dict[str, pd.DataFrame], SchemaSpec]:
    tables = {
        "fact": pd.DataFrame(
            {"ck": [1, 2, 3, 1], "channel": ["a", "b", "a", "b"], "y": [5.0, -3.0, 2.0, 1.0]}
        ),
        "dim_cust": pd.DataFrame(
            {"ck": [1, 2, 3], "gk": [10, 20, 10], "tier": ["gold", "silver", "gold"]}
        ),
        "dim_geo": pd.DataFrame({"gk": [10, 20], "country": ["PH", "US"]}),
    }
    spec = SchemaSpec(
        fact="fact",
        target="y",
        features=("channel",),
        joins=(
            Join(table="dim_cust", left="ck", right="ck", columns=("tier",)),
            Join(table="dim_geo", left="gk", right="gk", parent="dim_cust"),
        ),
    )
    return tables, spec


def test_snowflake_resolves_two_hops():
    tables, spec = _snowflake()

    result = flatten(tables, spec)

    assert list(result.X.columns) == ["channel", "dim_cust.tier", "dim_geo.country"]
    assert list(result.X["dim_cust.tier"]) == ["gold", "silver", "gold", "gold"]
    assert list(result.X["dim_geo.country"]) == ["PH", "US", "PH", "PH"]


def test_snowflake_conserves_rows_and_sum():
    tables, spec = _snowflake()

    result = flatten(tables, spec)

    assert len(result.X) == 4
    assert result.y.sum() == pytest.approx(5.0)
    assert result.provenance["target_sum"] == pytest.approx(5.0)


def test_snowflake_propagates_an_unmatched_first_hop_to_the_second():
    """An orphan at hop 1 has no geo key, so hop 2 must also read as unmatched."""
    tables, spec = _snowflake()
    tables["fact"] = pd.DataFrame({"ck": [1, 404], "channel": ["a", "b"], "y": [5.0, -3.0]})

    result = flatten(tables, spec)

    assert list(result.X["dim_cust.tier"]) == ["gold", "<unmatched>"]
    assert list(result.X["dim_geo.country"]) == ["PH", "<unmatched>"]
    assert result.y.sum() == pytest.approx(2.0)


def test_snowflake_three_hops():
    tables = {
        "fact": pd.DataFrame({"ak": [1], "y": [1.0]}),
        "dim_a": pd.DataFrame({"ak": [1], "bk": [2], "av": ["A"]}),
        "dim_b": pd.DataFrame({"bk": [2], "ck": [3], "bv": ["B"]}),
        "dim_c": pd.DataFrame({"ck": [3], "cv": ["C"]}),
    }
    spec = SchemaSpec(
        fact="fact",
        target="y",
        joins=(
            Join(table="dim_a", left="ak", right="ak", columns=("av",)),
            Join(table="dim_b", left="bk", right="bk", parent="dim_a", columns=("bv",)),
            Join(table="dim_c", left="ck", right="ck", parent="dim_b", columns=("cv",)),
        ),
    )

    result = flatten(tables, spec)

    assert list(result.X.columns) == ["dim_a.av", "dim_b.bv", "dim_c.cv"]
    assert result.X.iloc[0].tolist() == ["A", "B", "C"]


def test_flattened_snowflake_fits_and_conserves_through_impact_splitter():
    """The point of the whole feature: the output drops straight into fit()."""
    from impact_split import ImpactSplitter

    rng = np.random.default_rng(0)
    n = 600
    tables = {
        "fact": pd.DataFrame(
            {
                "ck": rng.integers(1, 4, size=n),
                "channel": rng.choice(["online", "partner"], size=n),
                "y": rng.normal(0.0, 10.0, size=n),
            }
        ),
        "dim_cust": pd.DataFrame(
            {"ck": [1, 2, 3], "gk": [10, 20, 10], "tier": ["gold", "silver", "gold"]}
        ),
        "dim_geo": pd.DataFrame({"gk": [10, 20], "country": ["PH", "US"]}),
    }
    spec = SchemaSpec(
        fact="fact",
        target="y",
        features=("channel",),
        joins=(
            Join(table="dim_cust", left="ck", right="ck", columns=("tier",)),
            Join(table="dim_geo", left="gk", right="gk", parent="dim_cust"),
        ),
    )

    result = flatten(tables, spec)
    model = ImpactSplitter().fit(result.X, result.y)
    segments = model.get_impact_segments()

    assert segments["n_samples"].sum() == n
    assert segments["total_sum"].sum() == pytest.approx(result.y.sum(), abs=1e-6)
```

- [ ] **Step 2: Run the tests**

```bash
python -m pytest tests/test_schema.py -k snowflake -v
```

Expected: PASS. If `test_snowflake_propagates_an_unmatched_first_hop_to_the_second` fails, the bug is in Task 5's `_align_dimension` chain — an unmatched parent row yields a NaN foreign key, which must itself fail to resolve at the next hop. Fix it in `schema.py`, do not weaken the test.

- [ ] **Step 3: Lint and commit**

```bash
make lint && python -m pytest tests/test_schema.py -q
```

```bash
git add tests/test_schema.py impact_split/schema.py
git commit -m "test(schema): snowflake chains conserve rows and sum end to end"
```

---

### Task 7: Round-trip frame equality over the benchmark battery

The evidence that earns a row in the README guarantees table. The normalizer here is **hand-written and independent of `flatten`** — it assigns surrogate keys by order of first appearance, which `flatten` knows nothing about. Without that independence the test would only prove the two functions are mutual inverses.

**Files:**
- Create: `tests/test_schema_roundtrip.py`

**Interfaces:**
- Consumes: `Join`, `SchemaSpec`, `flatten` from Tasks 1–5; `benchmarks.dgp.CASE_FACTORIES` (a `dict[str, Callable[[int], BenchDataset]]` — `BenchDataset` has `.X: pd.DataFrame` and `.y: np.ndarray`).
- Produces: nothing consumed by later tasks.

- [ ] **Step 1: Write the test**

Create `tests/test_schema_roundtrip.py`:

```python
"""Round-trip property: normalizing a flat frame and flattening it back is lossless.

Because ``ImpactSplitter.fit`` is untouched by the schema feature, an identical
frame implies an identical fit implies identical scores. That makes score
invariance a theorem rather than a benchmark measurement — so this file, not a
re-run of the battery, is the evidence that relational input changes nothing.

The normalizer below is written independently of ``flatten``: it invents
surrogate keys ordered by first appearance, a convention ``flatten`` has no
knowledge of. A passing round trip is therefore not a tautology about mutual
inverses.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from benchmarks.dgp import CASE_FACTORIES
from impact_split.schema import Join, SchemaSpec, flatten


def normalize_to_star(
    X: pd.DataFrame, y: np.ndarray
) -> tuple[dict[str, pd.DataFrame], SchemaSpec]:
    """Split a flat frame into a fact table plus one dimension per feature column."""
    fact = pd.DataFrame(index=pd.RangeIndex(len(X)))
    tables: dict[str, pd.DataFrame] = {}
    joins: list[Join] = []

    for column in X.columns:
        uniques = pd.unique(X[column])
        code_of = {value: code for code, value in enumerate(uniques)}
        key = f"{column}_key"
        fact[key] = [code_of[value] for value in X[column]]
        tables[f"dim_{column}"] = pd.DataFrame(
            {
                key: np.arange(len(uniques), dtype=np.int64),
                column: pd.Series(uniques, dtype=X[column].dtype),
            }
        )
        joins.append(Join(table=f"dim_{column}", left=key, right=key, columns=(column,)))

    fact["y"] = y
    tables["fact"] = fact
    return tables, SchemaSpec(fact="fact", target="y", joins=tuple(joins))


@pytest.mark.parametrize("case", sorted(CASE_FACTORIES))
def test_normalize_then_flatten_reproduces_the_original_frame(case: str):
    dataset = CASE_FACTORIES[case](42)
    tables, spec = normalize_to_star(dataset.X, dataset.y)

    result = flatten(tables, spec)

    unqualify = {f"dim_{c}.{c}": c for c in dataset.X.columns}
    got = result.X.rename(columns=unqualify)[list(dataset.X.columns)]
    expected = dataset.X.reset_index(drop=True)

    pd.testing.assert_frame_equal(got, expected, check_dtype=True)
    assert np.array_equal(result.y, dataset.y)


@pytest.mark.parametrize("case", sorted(CASE_FACTORIES))
def test_qualified_names_do_not_change_the_fitted_tree(case: str):
    """Renaming a column must not move a split.

    This is the non-tautological half. The frame test above proves the *values*
    survive the round trip; this proves the qualified names impact-split now sees
    (``dim_region.region`` rather than ``region``) do not perturb the tree, which
    they could if any tie-break or ordering depended on the column label.
    """
    from impact_split import ImpactSplitter

    dataset = CASE_FACTORIES[case](42)
    tables, spec = normalize_to_star(dataset.X, dataset.y)
    result = flatten(tables, spec)

    direct = ImpactSplitter().fit(dataset.X, dataset.y).get_impact_segments()
    qualified = ImpactSplitter().fit(result.X, result.y).get_impact_segments()

    assert len(qualified) == len(direct)
    np.testing.assert_allclose(
        sorted(qualified["total_sum"]), sorted(direct["total_sum"]), rtol=0, atol=1e-9
    )
    assert sorted(qualified["n_samples"]) == sorted(direct["n_samples"])
    assert qualified["total_sum"].sum() == pytest.approx(dataset.y.sum(), abs=1e-6)
```

- [ ] **Step 2: Run the test to verify it fails or passes for the right reason**

```bash
python -m pytest tests/test_schema_roundtrip.py -v
```

Expected: PASS for every case. If `check_dtype=True` fails, that is a **real dtype-preservation bug in `_align_dimension`**, not a test to loosen — the benchmark frames use pandas `StringDtype`, and a reindex that silently downgrades it to `object` would change how `_prepare_X_y` factorizes. Fix `schema.py`.

If `test_qualified_names_do_not_change_the_fitted_tree` fails while the frame test passes, a split moved purely because a column was renamed — investigate ordering or tie-breaking in `splitter.py` that depends on the column label, and report it rather than weakening the test.

- [ ] **Step 3: Lint and commit**

```bash
make lint && python -m pytest tests/test_schema_roundtrip.py -q
```

```bash
git add tests/test_schema_roundtrip.py impact_split/schema.py
git commit -m "test(schema): round-trip frame equality over the synthetic battery"
```

---

### Task 8: Public exports

**Files:**
- Modify: `impact_split/__init__.py`
- Test: `tests/test_schema.py`

**Interfaces:**
- Consumes: everything from Tasks 1–5.
- Produces: `from impact_split import Join, SchemaSpec, FlattenResult, SchemaError, flatten`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_schema.py`:

```python
def test_schema_api_is_exported_from_the_package_root():
    import impact_split

    for name in ("FlattenResult", "Join", "SchemaError", "SchemaSpec", "flatten"):
        assert name in impact_split.__all__
        assert hasattr(impact_split, name)
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
python -m pytest tests/test_schema.py -k exported -v
```

Expected: FAIL — `assert 'FlattenResult' in [...]`.

- [ ] **Step 3: Write the implementation**

Replace `impact_split/__init__.py` with:

```python
from impact_split.schema import FlattenResult, Join, SchemaError, SchemaSpec, flatten
from impact_split.splitter import ImpactSplitter
from impact_split.viz.html import render_html
from impact_split.viz.static import plot_icicle, plot_segments
from impact_split.viz.text import render_summary

__all__ = [
    "FlattenResult",
    "ImpactSplitter",
    "Join",
    "SchemaError",
    "SchemaSpec",
    "flatten",
    "plot_icicle",
    "plot_segments",
    "render_html",
    "render_summary",
]
```

- [ ] **Step 4: Run the full suite to verify nothing regressed**

```bash
python -m pytest tests -q
```

Expected: all tests pass, including the pre-existing suite.

- [ ] **Step 5: Lint and commit**

```bash
make lint && python -m pytest tests -q
```

```bash
git add impact_split/__init__.py tests/test_schema.py
git commit -m "feat(schema): export the schema API from the package root"
```

---

### Task 9: Documentation

**Files:**
- Modify: `README.md`
- Modify: `docs/docs/getting-started.md`
- Modify: `CHANGELOG.md`

**Interfaces:**
- Consumes: the public API from Task 8. Produces nothing.

Read `docs/docs/getting-started.md` before editing to match its existing heading level and voice. Note that `tests/test_docs_links.py` checks every relative markdown link in `README.md`, `CHANGELOG.md`, `docs/*.md`, `docs/docs/*.md`, and `reports/*.md` — any path you write must exist.

- [ ] **Step 1: Add the guarantees-table row**

In `README.md`, in the "What is guaranteed" table (starts around line 130), add:

```markdown
| **Relational input is lossless** | Many-to-one joins are reindex-aligned, so rows cannot duplicate; orphan keys are kept under a sentinel rather than dropped | `tests/test_schema_roundtrip.py::test_normalize_then_flatten_reproduces_the_original_frame`, `tests/test_schema_roundtrip.py::test_qualified_names_do_not_change_the_fitted_tree`, `tests/test_schema.py::test_flatten_rejects_a_dimension_with_duplicate_keys` |
```

- [ ] **Step 2: Extend the assumptions section**

In `README.md`, under "Assumptions and limitations", after the `fit(X, y) accepts:` bullet block, add:

```markdown
- `flatten(tables, spec)` additionally accepts a **star or snowflake schema** — one fact
  table at the observation grain plus dimension tables joined many-to-one — and returns the
  flat frame `fit()` expects. The restriction is strict: every join key must be unique and
  non-null in its dimension, and a fan-out join is rejected rather than silently duplicating
  rows. Fact rows whose foreign key does not resolve are kept under an `<unmatched>` category,
  so `sum(y)` is preserved exactly. Dimension columns are table-qualified
  (`dim_customer.region`); fact columns keep their names. One-to-many relationships
  (fact → line items) are **not** supported.
```

- [ ] **Step 3: Add a getting-started section**

In `docs/docs/getting-started.md`, add a section:

````markdown
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
````

- [ ] **Step 4: Add the CHANGELOG entry**

In `CHANGELOG.md`, add under a new `0.4.0` heading (match the existing entry format):

```markdown
### Added

- `impact_split.schema` — star/snowflake denormalization via `flatten(tables, spec)`,
  with `SchemaSpec` and `Join` as the declarative contract. Many-to-one joins only:
  duplicate or null dimension keys raise `SchemaError`, and orphan foreign keys are kept
  under an `<unmatched>` category so `sum(y)` is conserved exactly. Dimension columns are
  table-qualified in the output frame; fact columns keep their names.
  Accepts `{name: DataFrame}` — a SQLAlchemy adapter and schema introspection are planned
  for 0.5.0. No new dependencies.
```

- [ ] **Step 5: Verify docs links and full suite, then commit**

```bash
python -m pytest tests -q
```

Expected: all pass, including `tests/test_docs_links.py`.

```bash
make lint && make test
```

```bash
git add README.md docs/docs/getting-started.md CHANGELOG.md
git commit -m "docs: star/snowflake relational input"
```

---

## Self-Review

**Spec coverage** — every decision in `docs/plans/relational-input.md` maps to a task:

| Spec decision | Task |
|---|---|
| Star + snowflake, many-to-one only | 1 (graph), 3 (single hop), 6 (chains) |
| Explicit spec is the contract | 1 |
| Hard fail on fan-out, naming the join | 4 |
| Post-join `Σy` assertion | 4 |
| Left join, `<unmatched>` sentinel for orphans | 5 |
| Table-qualified dimension columns | 3 |
| `flatten()` standalone; `ImpactSplitter` untouched | 2–5 (no splitter edits anywhere) |
| DataFrame-dict core, no new deps | 2 |
| Provenance payload | 2 (skeleton), 5 (unmatched counts/mass) |
| Round-trip frame equality | 7 |
| Unit tests: fan-out, Σy, orphan, multi-hop | 4, 4, 5, 6 |
| README guarantees row + docs | 9 |
| Sentinel-collision escalation (risk 3) | 5 |
| Correlated-sentinel behavior documented (risk 4) | 6 (`test_snowflake_propagates_an_unmatched_first_hop_to_the_second` pins it) |
| Independent normalizer (risk 5) | 7 (module docstring states why) |

**Deferred to 0.5.0, no task here:** SQLAlchemy adapter, schema introspection helper. Correct — the source spec scopes them out of this release.

**Not addressed by this plan:** risk 2, the ledger path-width regression from qualified names. Qualified names are longer and the text ledger truncates at ~45 chars, but changing truncation touches `impact_split/viz/text.py`, which this plan forbids. Raise it as a separate change once real snowflake output exists to judge against — deciding the truncation strategy before seeing a real path would be guesswork.

**Placeholder scan:** no TBDs, no "add error handling", no "similar to Task N". Every code step carries runnable code.

**Type consistency:** `Join`, `SchemaSpec`, `FlattenResult`, `SchemaError`, `validate_spec`, `flatten`, `_align_dimension`, `_selected_columns`, `_assert_unique_key`, `_resolve_sentinel` are spelled identically in every task. `FlattenResult` exposes `.X`, `.y`, `.provenance` throughout — no task uses `.frame`.
