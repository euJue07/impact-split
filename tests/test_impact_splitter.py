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
    assert root["delta_pct"] == pytest.approx(0.05)
    assert root["delta"] == pytest.approx(0.0)
    assert root["delta_nominal"] == pytest.approx(10.0)
    assert root["delta_neg"] == pytest.approx(0.0)
    assert root["neutral_band"]["low"] == pytest.approx(0.0)
    assert root["neutral_band"]["high"] == pytest.approx(0.0)
    assert root["s_node_p"] == pytest.approx(100.0)
    assert root["s_node_n"] == pytest.approx(100.0)
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


def test_root_scaled_delta_when_neutral_root_false() -> None:
    """With scaled delta at root, category sums can fall in neutral band; neutral_root splits by sign."""
    X = pd.DataFrame({"region": ["A", "A", "B", "B"]})
    y = pd.Series([2.5, 2.5, -2.5, -2.5])
    m_legacy = ImpactSplitter(
        delta_pct=0.5,
        min_global_impact_pct=0.001,
        max_depth=2,
        neutral_root=False,
    )
    m_legacy.fit(X, y, trace=True)
    assert m_legacy.fit_trace_[0]["action"] == "leaf"
    assert m_legacy.fit_trace_[0]["stop_reason"] == "no_split"
    assert m_legacy.fit_trace_[0]["delta"] == pytest.approx(m_legacy.fit_trace_[0]["delta_nominal"])

    m_neutral = ImpactSplitter(
        delta_pct=0.5,
        min_global_impact_pct=0.001,
        max_depth=2,
        neutral_root=True,
    )
    m_neutral.fit(X, y, trace=True)
    assert m_neutral.fit_trace_[0]["action"] == "split"
    assert m_neutral.fit_trace_[0]["chosen_feature"] == "region"


def test_child_uses_scaled_assignment_delta() -> None:
    """After root (assignment_delta=0), children use delta = V_node * delta_pct."""
    X = pd.DataFrame({"region": ["A", "A", "B", "B"], "other": ["x", "y", "x", "y"]})
    y = pd.Series([50.0, 50.0, -50.0, -50.0])
    m = ImpactSplitter(
        delta_pct=0.05,
        min_global_impact_pct=0.001,
        max_depth=3,
        neutral_root=True,
    )
    m.fit(X, y, trace=True)
    children = [s for s in m.fit_trace_ if s["depth"] == 1]
    assert len(children) >= 1
    for child in children:
        assert child["delta_nominal"] == pytest.approx(child["V_node"] * 0.05)
        assert child["delta"] == pytest.approx(child["delta_nominal"])


def test_verbose_enables_trace() -> None:
    """verbose=True is an alias for trace=True (same fit_trace_)."""
    X = pd.DataFrame({"region": ["A", "B"]})
    y = pd.Series([1.0, -1.0])
    m = ImpactSplitter(delta_pct=0.01, min_global_impact_pct=0.001, max_depth=1)
    m.fit(X, y, verbose=True)
    assert len(m.fit_trace_) >= 1
    assert "delta_neg" in m.fit_trace_[0]


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
