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


def test_validate_spec_rejects_a_dimension_table_not_found():
    tables, _ = _star()
    spec = SchemaSpec(
        fact="fact",
        target="y",
        joins=(Join(table="ghost", left="cust_key", right="cust_key"),),
    )
    with pytest.raises(SchemaError, match="dimension table 'ghost' not found"):
        validate_spec(tables, spec)


def test_validate_spec_rejects_a_join_key_not_found_in_the_dimension():
    tables, _ = _star()
    spec = SchemaSpec(
        fact="fact",
        target="y",
        joins=(Join(table="dim_cust", left="cust_key", right="nope"),),
    )
    with pytest.raises(SchemaError, match="join key 'nope' not found in 'dim_cust'"):
        validate_spec(tables, spec)


def test_validate_spec_rejects_a_column_not_found_in_the_dimension():
    tables, _ = _star()
    spec = SchemaSpec(
        fact="fact",
        target="y",
        joins=(Join(table="dim_cust", left="cust_key", right="cust_key", columns=("nope",)),),
    )
    with pytest.raises(SchemaError, match="column 'nope' not found in 'dim_cust'"):
        validate_spec(tables, spec)


def test_validate_spec_rejects_a_foreign_key_not_found_in_the_parent():
    tables, _ = _star()
    spec = SchemaSpec(
        fact="fact",
        target="y",
        joins=(Join(table="dim_cust", left="nope", right="cust_key"),),
    )
    with pytest.raises(SchemaError, match="foreign key 'nope' not found in 'fact'"):
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


def test_validate_spec_rejects_the_fact_table_joined_to_itself():
    """Joining the fact to itself is a different mistake from a role-playing
    dimension and must not share that message."""
    tables, _ = _star()
    spec = SchemaSpec(
        fact="fact",
        target="y",
        joins=(Join(table="fact", left="cust_key", right="cust_key"),),
    )
    with pytest.raises(SchemaError, match="is the fact table and cannot also be joined"):
        validate_spec(tables, spec)


def test_validate_spec_names_role_playing_dimensions_in_the_duplicate_join_message():
    """A role-playing dimension (the same table joined twice under different
    FKs, e.g. ship_geo / bill_geo) is a legitimate star-schema pattern that
    still hits the once-per-table restriction; the message should say so."""
    tables, _ = _star()
    spec = SchemaSpec(
        fact="fact",
        target="y",
        joins=(
            Join(table="dim_cust", left="cust_key", right="cust_key"),
            Join(table="dim_cust", left="cust_key", right="cust_key"),
        ),
    )
    with pytest.raises(SchemaError, match="role-playing dimension"):
        validate_spec(tables, spec)


def test_role_playing_dimension_works_when_listed_under_two_names():
    """The pattern the message above points at must actually work, or that
    message is sending people somewhere broken. The same DataFrame object goes
    under two keys in `tables` -- no copy -- and each is joined once. Both roles
    must resolve independently and stay distinguishable by their qualifier.
    """
    geo = pd.DataFrame({"geo_id": [1, 2], "region": ["North", "South"]})
    fact = pd.DataFrame({"ship_geo": [1, 2, 1], "bill_geo": [2, 2, 1], "y": [10.0, 20.0, 30.0]})
    tables = {"fact": fact, "geo_ship": geo, "geo_bill": geo}
    assert tables["geo_ship"] is tables["geo_bill"], "the point is one object, two keys"

    spec = SchemaSpec(
        fact="fact",
        target="y",
        joins=(
            Join(table="geo_ship", left="ship_geo", right="geo_id", columns=("region",)),
            Join(table="geo_bill", left="bill_geo", right="geo_id", columns=("region",)),
        ),
        features=(),
    )
    result = flatten(tables, spec)

    assert list(result.X.columns) == ["geo_ship.region", "geo_bill.region"]
    # Row 0 ships North but bills South -- the roles must not collapse together.
    assert result.X["geo_ship.region"].tolist() == ["North", "South", "North"]
    assert result.X["geo_bill.region"].tolist() == ["South", "South", "North"]
    assert float(result.y.sum()) == float(fact["y"].sum())


def test_validate_spec_rejects_a_dimension_with_duplicate_columns():
    tables = {
        "fact": pd.DataFrame({"k": [1], "y": [1.0]}),
        "dim": pd.DataFrame([[1, "x", "z"]], columns=["k", "v", "v"]),
    }
    spec = SchemaSpec(fact="fact", target="y", joins=(Join(table="dim", left="k", right="k"),))
    with pytest.raises(SchemaError, match="duplicate column name 'v'"):
        validate_spec(tables, spec)


def test_validate_spec_rejects_duplicate_feature_columns():
    tables, _ = _star()
    spec = SchemaSpec(fact="fact", target="y", features=("channel", "channel"))
    with pytest.raises(SchemaError, match="selected more than once"):
        validate_spec(tables, spec)


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


def test_flatten_rejects_a_non_finite_target():
    tables = {"fact": pd.DataFrame({"channel": ["a", "b"], "y": [1.0, np.inf]})}
    spec = SchemaSpec(fact="fact", target="y", features=("channel",))
    with pytest.raises(SchemaError, match="non-finite"):
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


def test_flatten_snowflake_child_keys_off_the_parents_own_key_column():
    # dim_b hops off dim_a's "k" column, which is also dim_a's own join key —
    # the internal alignment frame must keep that key column available.
    tables = {
        "fact": pd.DataFrame({"k": [1, 2], "y": [1.0, 2.0]}),
        "dim_a": pd.DataFrame({"k": [1, 2], "a_val": ["x", "y"]}),
        "dim_b": pd.DataFrame({"k": [1, 2], "b_val": ["p", "q"]}),
    }
    spec = SchemaSpec(
        fact="fact",
        target="y",
        joins=(
            Join(table="dim_a", left="k", right="k"),
            Join(table="dim_b", left="k", right="k", parent="dim_a"),
        ),
    )

    result = flatten(tables, spec)

    assert list(result.X["dim_a.a_val"]) == ["x", "y"]
    assert list(result.X["dim_b.b_val"]) == ["p", "q"]


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
    """Belt-and-braces: the mechanism cannot fan out, but the guard still runs."""
    tables, spec = _star()
    result = flatten(tables, spec)
    assert len(result.X) == len(tables["fact"])
    assert result.y.sum() == tables["fact"]["y"].sum()


def test_align_dimension_preserves_extension_dtypes():
    """dtype preservation is load-bearing: Task 7 round-trips StringDtype frames.

    Without a permanent test here, a future 'simplification' of _align_dimension
    (to pd.merge, say) would silently downgrade StringDtype to object and only
    surface as a confusing Task 7 failure.
    """
    tables = {
        "fact": pd.DataFrame({"k": [1, 2], "y": [1.0, 2.0]}),
        "dim": pd.DataFrame(
            {"k": [1, 2], "label": pd.Series(["a", "b"], dtype="string"), "n": [10, 20]}
        ),
    }
    spec = SchemaSpec(fact="fact", target="y", joins=(Join(table="dim", left="k", right="k"),))

    result = flatten(tables, spec)

    assert result.X["dim.label"].dtype == pd.StringDtype()
    assert result.X["dim.n"].dtype == np.int64


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


def test_flatten_rejects_a_totally_mismatched_join_key_dtype():
    """int64 fact keys against str dimension keys resolve nothing; without this
    guard the join would silently succeed with every row unmatched."""
    tables = {
        "fact": pd.DataFrame({"cust": np.arange(5, dtype=np.int64), "y": [1.0] * 5}),
        "dim": pd.DataFrame({"cust": [str(i) for i in range(5)], "region": ["W"] * 5}),
    }
    spec = SchemaSpec(
        fact="fact", target="y", joins=(Join(table="dim", left="cust", right="cust"),)
    )

    with pytest.raises(SchemaError) as excinfo:
        flatten(tables, spec)

    message = str(excinfo.value)
    assert "matched none of 5" in message
    assert "int64" in message
    assert str(tables["dim"]["cust"].dtype) in message


def test_flatten_allows_an_empty_fact_table_even_with_a_dtype_mismatched_dimension():
    """The zero-match guard must not fire on a legitimately empty fact table —
    np.array([]).all() is True, so len(fact) > 0 is required alongside it."""
    tables = {
        "fact": pd.DataFrame(
            {"cust": pd.array([], dtype="int64"), "y": pd.array([], dtype="float64")}
        ),
        "dim": pd.DataFrame({"cust": ["0", "1"], "region": ["W", "E"]}),
    }
    spec = SchemaSpec(
        fact="fact", target="y", joins=(Join(table="dim", left="cust", right="cust"),)
    )

    result = flatten(tables, spec)

    assert len(result.X) == 0


def test_flatten_keeps_integer_looking_labels_when_a_sibling_row_is_unmatched():
    """reindex() promotes int64 -> float64 to carry NaN; without pre-casting to
    object, the surviving values would freeze as stringified floats ('7.0')."""
    tables = {
        "fact": pd.DataFrame({"k": [1, 2, 99], "y": [1.0, 2.0, 3.0]}),
        "dim": pd.DataFrame({"k": [1, 2], "store_no": [7, 8]}),
    }
    spec = SchemaSpec(fact="fact", target="y", joins=(Join(table="dim", left="k", right="k"),))

    result = flatten(tables, spec)

    assert list(result.X["dim.store_no"]) == [7, 8, "<unmatched>"]
    assert isinstance(result.X["dim.store_no"].iloc[0], (int, np.integer))


def test_flatten_skips_sentinel_resolution_when_everything_matches():
    """_resolve_sentinel's .dropna().unique() scan raises on unhashable cells;
    it must not run at all when nothing is unmatched."""
    tables = {
        "fact": pd.DataFrame({"k": [1, 2], "y": [1.0, 2.0]}),
        "dim": pd.DataFrame({"k": [1, 2], "tags": [["a"], ["b"]]}),
    }
    spec = SchemaSpec(fact="fact", target="y", joins=(Join(table="dim", left="k", right="k"),))

    result = flatten(tables, spec)

    assert result.provenance["sentinel"] is None
    assert list(result.X["dim.tags"]) == [["a"], ["b"]]


def test_unmatched_dimension_columns_are_identical_across_a_multi_column_dimension():
    """One orphan FK marks every selected column of that dimension identically,
    producing perfectly-correlated features — the risk-4 behavior from the
    implementation plan's Self-Review, distinct from the cross-hop propagation
    pinned by test_snowflake_propagates_an_unmatched_first_hop_to_the_second."""
    tables = {
        "fact": pd.DataFrame({"k": [1, 99], "y": [1.0, 2.0]}),
        "dim": pd.DataFrame({"k": [1], "region": ["W"], "tier": ["gold"], "channel": ["online"]}),
    }
    spec = SchemaSpec(fact="fact", target="y", joins=(Join(table="dim", left="k", right="k"),))

    result = flatten(tables, spec)

    row = result.X.iloc[1]
    assert row["dim.region"] == row["dim.tier"] == row["dim.channel"] == "<unmatched>"


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


def test_schema_api_is_exported_from_the_package_root():
    import impact_split

    for name in ("FlattenResult", "Join", "SchemaError", "SchemaSpec", "flatten"):
        assert name in impact_split.__all__
        assert hasattr(impact_split, name)
