"""Tests for ImpactSplitter and fit trace."""

import numpy as np
import pandas as pd
import pytest

from impact_split import ImpactSplitter


def test_fit_requires_matching_length() -> None:
    X = pd.DataFrame({"a": [1, 2]})
    y = pd.Series([1.0])
    with pytest.raises(ValueError, match="same number of rows"):
        ImpactSplitter().fit(X, y)


def test_trace_records_split_and_leaf_steps() -> None:
    """Two clear opposite regions so root splits on `region`; trace has split + child leaves."""
    X = pd.DataFrame(
        {
            "region": ["A", "A", "B", "B"],
            "noise": ["x", "y", "x", "y"],
        }
    )
    y = pd.Series([50.0, 50.0, -50.0, -50.0])
    m = ImpactSplitter(
        delta_pct=0.05,
        min_global_impact_pct=0.001,
        max_depth=3,
    )
    m.fit(X, y, trace=True)
    assert len(m.fit_trace_) >= 1
    root = m.fit_trace_[0]
    assert root["depth"] == 0
    assert root["materiality_pass"] is True
    assert root["action"] == "split"
    assert root["chosen_feature"] == "region"
    segments = m.get_impact_segments()
    assert len(segments) >= 2
    assert np.isclose(segments["total_sum"].sum(), y.sum(), rtol=1e-6)


def test_no_split_when_gain_zero() -> None:
    """Single category per feature: no split; one leaf segment."""
    X = pd.DataFrame({"only": ["a"] * 10})
    y = pd.Series([1.0] * 10)
    m = ImpactSplitter(delta_pct=0.01, min_global_impact_pct=0.001, max_depth=5)
    m.fit(X, y, trace=True)
    root = m.fit_trace_[0]
    assert root["action"] == "leaf"
    assert root["stop_reason"] == "no_split"
    seg = m.get_impact_segments()
    assert len(seg) == 1
    assert seg.iloc[0]["total_sum"] == pytest.approx(10.0)


def test_max_depth_zero_stops_at_root() -> None:
    """With max_depth=0, root is a leaf before any split."""
    X = pd.DataFrame({"region": ["A", "B"] * 50})
    y = pd.Series([0.01, -0.01] * 50)
    m = ImpactSplitter(
        delta_pct=0.01,
        min_global_impact_pct=0.001,
        max_depth=0,
    )
    m.fit(X, y, trace=True)
    assert m.fit_trace_[0]["action"] == "leaf"
    assert m.fit_trace_[0]["stop_reason"] == "max_depth"
