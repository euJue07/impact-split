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

from benchmarks.dgp import CASE_FACTORIES
import numpy as np
import pandas as pd
import pytest

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
