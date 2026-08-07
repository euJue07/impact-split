"""Schema spec validation and star/snowflake flattening."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from impact_split.schema import (
    FlattenResult,
    Join,
    SchemaError,
    SchemaSpec,
    flatten,
    validate_spec,
)


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
