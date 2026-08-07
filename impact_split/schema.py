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


@dataclass(frozen=True)
class FlattenResult:
    """The denormalized frame, the target, and the audit trail of how it was built."""

    X: pd.DataFrame
    y: np.ndarray
    provenance: dict[str, Any] = field(default_factory=dict)


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
    # drop=False keeps the key available as a column as well as the index, so an
    # explicit columns= request that names it still resolves — and Task 5 reads it
    # as the witness for "did this key resolve?".
    indexed = dim.set_index(join.right, drop=False)[list(selected)]
    aligned = indexed.reindex(pd.Index(parent_keys))
    aligned.index = pd.RangeIndex(len(aligned))
    return aligned


def flatten(tables: dict[str, pd.DataFrame], spec: SchemaSpec) -> FlattenResult:
    """Denormalize a star or snowflake schema into a flat frame plus target.

    Every join must be many-to-one: the dimension's key must be unique and
    non-null. Fact rows whose foreign key does not resolve are kept, with the
    dimension's columns set to an unmatched sentinel, so the row count and
    ``sum(y)`` of the fact table are preserved exactly.
    """
    ordered = validate_spec(tables, spec)
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

    if not columns:
        raise SchemaError("spec selects no feature columns.")

    X = pd.DataFrame(columns)
    if len(X) != len(fact):
        raise SchemaError(
            f"row count changed during flattening: {len(fact)} fact rows produced {len(X)}."
        )
    return FlattenResult(X=X, y=y, provenance=provenance)
