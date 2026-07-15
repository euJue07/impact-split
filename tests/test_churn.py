"""Tests for segment gross flows and the churn flag (v0.2.0)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from impact_split import ImpactSplitter


def churn_frame(n: int = 400) -> tuple[pd.DataFrame, pd.Series]:
    """Identical feature rows carrying offsetting +100 / -99: irreducible churn."""
    rng = np.random.default_rng(5)
    X = pd.DataFrame({"a": ["x"] * n})
    y = np.where(np.arange(n) % 2 == 0, 100.0, -99.0) + rng.normal(0, 0.1, n)
    return X, pd.Series(y)


def test_single_churn_segment_carries_gross_flows_and_flag() -> None:
    X, y = churn_frame()
    model = ImpactSplitter().fit(X, y)
    (seg,) = model.segments_
    y_arr = y.to_numpy()
    assert seg["pos_sum"] == pytest.approx(float(y_arr[y_arr > 0].sum()))
    assert seg["neg_sum"] == pytest.approx(float(np.abs(y_arr[y_arr < 0]).sum()))
    assert seg["is_churn"] is True


def test_gross_flows_on_all_segments_and_flag_matches_rule() -> None:
    # Shattered-rule fixture from test_consolidation: exercises both the
    # consolidated path and (with consolidate=False) the plain-leaf path.
    rng = np.random.default_rng(3)
    n = 1200
    f0 = (rng.random(n) < 0.3).astype(np.int64)
    f1 = (rng.random(n) < 0.5).astype(np.int64)
    y = 5.0 * (f0 == 1) + 10.0 * ((f0 == 0) & (f1 == 1)) + rng.normal(0, 0.5, n)
    X = np.column_stack([f0, f1])
    pos_pool = float(y[y > 0].sum())
    neg_pool = float(np.abs(y[y < 0]).sum())
    for consolidate in (True, False):
        model = ImpactSplitter(consolidate=consolidate).fit(X, y)
        for seg in model.segments_:
            assert seg["pos_sum"] >= 0.0 and seg["neg_sum"] >= 0.0
            assert seg["pos_sum"] - seg["neg_sum"] == pytest.approx(
                seg["total_sum"], abs=1e-6
            )
            expected = (
                seg["pos_sum"] / pos_pool > model.min_global_impact_pct
                and seg["neg_sum"] / neg_pool > model.min_global_impact_pct
            )
            assert seg["is_churn"] is expected


def test_gross_flows_conserve_pools_exactly() -> None:
    X, y = churn_frame()
    model = ImpactSplitter().fit(X, y)
    y_arr = y.to_numpy()
    assert sum(s["pos_sum"] for s in model.segments_) == pytest.approx(
        float(y_arr[y_arr > 0].sum())
    )
    assert sum(s["neg_sum"] for s in model.segments_) == pytest.approx(
        float(np.abs(y_arr[y_arr < 0]).sum())
    )


def test_fit_only_keys_still_dropped() -> None:
    X, y = churn_frame()
    model = ImpactSplitter().fit(X, y)
    for seg in model.segments_:
        assert "mask" not in seg and "mean" not in seg
