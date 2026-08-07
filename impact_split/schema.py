"""Star/snowflake schema denormalization into the flat frame ImpactSplitter accepts."""

from __future__ import annotations

from dataclasses import dataclass

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
