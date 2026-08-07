"""Schema spec validation and star/snowflake flattening."""

from __future__ import annotations

import pandas as pd
import pytest

from impact_split.schema import Join, SchemaError, SchemaSpec, validate_spec


def _star() -> tuple[dict[str, pd.DataFrame], SchemaSpec]:
    """A minimal valid star: 4 fact rows, one dimension, one feature each side."""
    tables = {
        "fact": pd.DataFrame(
            {
                "cust_key": [1, 2, 1, 3],
                "channel": ["a", "b", "a", "b"],
                "y": [10.0, -2.0, 3.0, 5.0],
            }
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
        joins=(Join(table="dim_cust", left="cust_key", right="cust_key", columns=("cust_key",)),),
    )
    with pytest.raises(SchemaError, match="join key 'cust_key' cannot be selected"):
        validate_spec(tables, spec)


def test_validate_spec_rejects_non_numeric_target():
    tables = {"fact": pd.DataFrame({"y": ["a", "b"]})}
    with pytest.raises(SchemaError, match="target column 'y' must be numeric"):
        validate_spec(tables, SchemaSpec(fact="fact", target="y"))
