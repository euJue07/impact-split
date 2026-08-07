"""Star/snowflake schema denormalization into the flat frame ImpactSplitter accepts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from pandas.api.types import is_float_dtype, is_numeric_dtype


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
    aligned_features: dict[str, tuple[Join, pd.DataFrame]] = {}
    unmatched_masks: dict[str, np.ndarray] = {}

    for join in ordered:
        parent_name = join.parent or spec.fact
        dim = tables[join.table]
        _assert_unique_key(dim, join)
        # Align every dimension column once: a chained hop needs this frame to
        # find its own foreign key, which the user's feature selection may omit.
        parent_keys = aligned_tables[parent_name][join.left]
        full = _align_dimension(
            parent_keys,
            dim,
            Join(table=join.table, left=join.left, right=join.right, columns=tuple(dim.columns)),
        )

        if len(full) != len(fact):
            raise SchemaError(
                f"joining {join.table!r} changed the row count: {len(fact)} fact rows "
                f"produced {len(full)}. This should be unreachable — reindexing against a "
                "unique index cannot fan out — so treat it as a bug in impact-split."
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

    for table, (_join, frame) in aligned_features.items():
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

    if not columns:
        raise SchemaError("spec selects no feature columns.")

    X = pd.DataFrame(columns)
    # sum(y) needs no guard: y is read from the fact table once and never passes
    # through a join, so only a row-count change can break the row-to-target
    # correspondence. Both length checks above and below are that guard.
    if len(X) != len(fact):
        raise SchemaError(
            f"row count changed during flattening: {len(fact)} fact rows produced {len(X)}."
        )
    return FlattenResult(X=X, y=y, provenance=provenance)
