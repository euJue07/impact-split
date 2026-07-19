"""Forest annotation layer: matching, replicates, stability/CI, importance, shadows."""
import numpy as np
import pytest

from impact_split.ensemble import greedy_match, jaccard, mask_from_conditions
from impact_split.splitter import ImpactSplitter


def _m(*idx, n=6):
    out = np.zeros(n, dtype=bool)
    out[list(idx)] = True
    return out


def test_jaccard_basics():
    assert jaccard(_m(0, 1), _m(0, 1)) == 1.0
    assert jaccard(_m(0, 1), _m(2, 3)) == 0.0
    assert jaccard(_m(), _m()) == 0.0  # empty union -> 0, no ZeroDivisionError
    assert jaccard(_m(0, 1, 2), _m(1, 2, 3)) == pytest.approx(2 / 4)


def test_mask_from_conditions_conjunction_and_col_map():
    X = np.array([[0, 5], [0, 6], [1, 5], [2, 6]], dtype=np.int64)
    cond = {0: frozenset({0}), 1: frozenset({5})}
    np.testing.assert_array_equal(
        mask_from_conditions(cond, X), np.array([True, False, False, False])
    )
    # replicate fit on column subset [1]: its feature 0 is original feature 1
    np.testing.assert_array_equal(
        mask_from_conditions({0: frozenset({6})}, X, col_map=np.array([1])),
        np.array([False, True, False, True]),
    )


def test_greedy_match_sign_gate_and_one_to_one():
    ref = [_m(0, 1, 2), _m(3, 4)]
    rep = [_m(0, 1, 2), _m(0, 1), _m(3, 4)]
    # rep 0 and rep 1 both overlap ref 0; only the better one may claim it
    out = greedy_match(ref, [1, -1], rep, [1, 1, -1], threshold=0.5)
    assert (0, 0, 1.0) in out
    assert not any(r == 0 and p == 1 for r, p, _ in out)
    # rep 2 has jaccard 1.0 with ref 1 but only matches because signs agree
    assert (1, 2, 1.0) in out
    # flip rep 2's sign -> no match for ref 1
    out2 = greedy_match(ref, [1, -1], rep, [1, 1, 1], threshold=0.5)
    assert not any(r == 1 for r, _, _ in out2)


def _planted_model(seed=0, n=2000):
    rng = np.random.default_rng(seed)
    a = rng.integers(0, 3, n)
    b = rng.integers(0, 4, n)
    y = np.where(a == 0, 100.0, 0.0) + rng.normal(0, 3, n)
    X = np.column_stack([a, b]).astype(np.int64)
    return ImpactSplitter().fit(X, y), X, y


def test_fit_replicate_deterministic_and_remappable():
    from impact_split.ensemble import fit_replicate, mask_from_conditions

    model, X, y = _planted_model()
    cols = np.array([1, 0])  # deliberately permuted subset
    rep1 = fit_replicate(model, np.random.default_rng(42), cols)
    rep2 = fit_replicate(model, np.random.default_rng(42), cols)
    assert [s["conditions"] for s in rep1.segments_] == [
        s["conditions"] for s in rep2.segments_
    ]
    # remapped masks are full-length and the planted a==0 segment reappears
    masks = [
        mask_from_conditions(s["conditions"], X, col_map=cols)
        for s in rep1.segments_
    ]
    assert all(m.shape == (X.shape[0],) for m in masks)
    planted = X[:, 0] == 0
    from impact_split.ensemble import jaccard

    assert max(jaccard(planted, m) for m in masks) > 0.8
