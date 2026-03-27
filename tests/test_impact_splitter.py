"""Tests for ImpactSplitter with NumPy / pandas inputs and spec-aligned recursion."""

import numpy as np
import pandas as pd
import pytest

from impact_split import ImpactSplitter


def test_fit_rejects_invalid_inputs() -> None:
    model = ImpactSplitter()

    with pytest.raises(ValueError, match="X must be a numpy.ndarray or pandas.DataFrame"):
        model.fit([[0], [1]], np.array([1.0, -1.0]))

    with pytest.raises(ValueError, match="y must be a numpy.ndarray or pandas.Series"):
        model.fit(np.array([[0], [1]], dtype=np.int64), [1.0, -1.0])


def test_fit_rejects_dataframe_with_missing() -> None:
    model = ImpactSplitter()
    x = pd.DataFrame({"a": ["x", None]})
    y = np.array([1.0, -1.0])
    with pytest.raises(ValueError, match="missing values"):
        model.fit(x, y)


def test_fit_validates_shapes_and_integer_encoding() -> None:
    model = ImpactSplitter()
    x_good = np.array([[0, 1], [1, 0]], dtype=np.int64)
    y_good = np.array([1.0, -1.0], dtype=float)

    with pytest.raises(ValueError, match="2D numpy.ndarray"):
        model.fit(np.array([0, 1], dtype=np.int64), y_good)
    with pytest.raises(ValueError, match="integer label-encoded"):
        model.fit(np.array([[0.0], [1.0]], dtype=float), y_good)
    with pytest.raises(ValueError, match="non-negative integers"):
        model.fit(np.array([[0], [-1]], dtype=np.int64), y_good)
    with pytest.raises(ValueError, match="1D numpy.ndarray"):
        model.fit(x_good, np.array([[1.0], [-1.0]], dtype=float))
    with pytest.raises(ValueError, match="same number of rows"):
        model.fit(x_good, np.array([1.0], dtype=float))


def test_trace_records_split_and_conserves_total_sum() -> None:
    x = np.array(
        [
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1],
        ],
        dtype=np.int64,
    )
    y = np.array([50.0, 50.0, -50.0, -50.0], dtype=float)
    model = ImpactSplitter(delta_pct=0.05, min_global_impact_pct=0.001, max_depth=3)

    model.fit(x, y, trace=True)

    assert len(model.fit_trace_) >= 1
    root = model.fit_trace_[0]
    assert root["depth"] == 0
    assert root["action"] == "split"
    assert root["chosen_feature_index"] == 0
    assert root["delta"] == pytest.approx(np.abs(y).sum() * 0.05)
    assert root["s_node_p"] == pytest.approx(100.0)
    assert root["s_node_n"] == pytest.approx(100.0)

    segments = model.get_impact_segments()
    assert len(segments) >= 2
    assert np.isclose(float(segments["total_sum"].sum()), float(y.sum()), rtol=1e-6)


def test_one_sided_gain_can_split() -> None:
    x = np.array([[0], [0], [1], [1]], dtype=np.int64)
    y = np.array([8.0, 7.0, 0.0, 0.0], dtype=float)
    model = ImpactSplitter(delta_pct=0.1, min_global_impact_pct=0.001, max_depth=2)

    model.fit(x, y, trace=True)
    root = model.fit_trace_[0]

    assert root["action"] == "split"
    assert root["chosen_feature_index"] == 0
    assert root["candidate_gains"][0]["gain_P"] > 0.0
    assert root["candidate_gains"][0]["gain_N"] == pytest.approx(0.0)


def test_materiality_uses_strict_greater_than() -> None:
    x = np.array([[0], [1], [0], [1]], dtype=np.int64)
    y = np.array([2.0, -2.0, 2.0, -2.0], dtype=float)
    model = ImpactSplitter(delta_pct=0.1, min_global_impact_pct=1.0, max_depth=3)

    model.fit(x, y, trace=True)
    root = model.fit_trace_[0]

    assert root["action"] == "leaf"
    assert root["stop_reason"] == "materiality"
    assert root["positive_trigger"] is False
    assert root["negative_trigger"] is False


def test_identical_rows_stop_condition() -> None:
    x = np.array([[1, 2], [1, 2], [1, 2]], dtype=np.int64)
    y = np.array([10.0, -5.0, 3.0], dtype=float)
    model = ImpactSplitter(delta_pct=0.01, min_global_impact_pct=0.001, max_depth=5)

    model.fit(x, y, trace=True)

    assert model.fit_trace_[0]["action"] == "leaf"
    assert model.fit_trace_[0]["stop_reason"] == "identical_rows"


def test_no_split_when_all_category_sums_within_delta() -> None:
    x = np.array([[0], [1], [2], [0], [1], [2]], dtype=np.int64)
    y = np.array([1.0, -1.0, 0.5, -0.5, 0.1, -0.1], dtype=float)
    model = ImpactSplitter(delta_pct=0.9, min_global_impact_pct=0.001, max_depth=3)

    model.fit(x, y, trace=True)

    assert model.fit_trace_[0]["action"] == "leaf"
    assert model.fit_trace_[0]["stop_reason"] == "no_split"


def test_max_depth_zero_stops_at_root() -> None:
    x = np.array([[0], [1], [0], [1]], dtype=np.int64)
    y = np.array([1.0, -1.0, 1.0, -1.0], dtype=float)
    model = ImpactSplitter(delta_pct=0.01, min_global_impact_pct=0.001, max_depth=0)

    model.fit(x, y, trace=True)

    assert model.fit_trace_[0]["action"] == "leaf"
    assert model.fit_trace_[0]["stop_reason"] == "max_depth"


def test_dataframe_fit_stores_maps_and_paths_use_column_names() -> None:
    x = pd.DataFrame(
        {
            "region": ["East", "East", "West", "West"],
            "other": [0, 1, 0, 1],
        },
    )
    y = pd.Series([50.0, 50.0, -50.0, -50.0], dtype=float)
    model = ImpactSplitter(delta_pct=0.05, min_global_impact_pct=0.001, max_depth=3)

    model.fit(x, y, trace=True)

    assert model.feature_names_in_ == ["region", "other"]
    assert model.category_maps_ is not None
    assert list(model.category_maps_[0]) == ["East", "West"]

    root = model.fit_trace_[0]
    assert root["action"] == "split"
    assert root["chosen_feature_name"] == "region"
    assert root["routing_labels"]["positive"] == ["East"]
    assert root["routing_labels"]["negative"] == ["West"]

    segments = model.get_impact_segments()
    assert segments["path"].str.contains("region=").any()
    assert not segments["path"].str.contains("f0=").any()

    cat_row = root["category_tables"][0][0]
    assert cat_row["category_label"] == "East"


def test_format_split_node_label_shows_decoded_categories() -> None:
    x = pd.DataFrame({"dim": ["A", "B", "A", "B"]})
    y = np.array([10.0, -10.0, 10.0, -10.0], dtype=float)
    model = ImpactSplitter(delta_pct=0.01, min_global_impact_pct=0.001, max_depth=1)
    model.fit(x, y)

    assert model._tree is not None
    assert not model._tree.is_leaf
    label = model._format_split_node_label(model._tree)
    assert "dim" in label
    assert "A" in label
    assert "B" in label
