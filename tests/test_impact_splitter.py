"""Tests for ImpactSplitter with NumPy / pandas inputs and spec-aligned recursion."""

import io

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
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
    assert root["routing_mode"] == "raw"
    assert root["candidate_gains"][0]["gain_P"] > 0.0
    assert root["candidate_gains"][0]["gain_N"] == pytest.approx(0.0)


def test_centered_excess_fallback_splits_one_sided_target() -> None:
    x = np.array([[0], [0], [1], [1]], dtype=np.int64)
    y = np.array([10.0, 10.0, 1.0, 1.0], dtype=float)
    model = ImpactSplitter(delta_pct=0.05, min_global_impact_pct=0.001, max_depth=2)

    model.fit(x, y, trace=True)
    root = model.fit_trace_[0]

    assert root["action"] == "split"
    assert root["routing_mode"] == "centered_excess"
    assert root["chosen_feature_index"] == 0
    assert root["delta_centered_excess"] > 0.0
    assert root["candidate_gains_by_mode"]["raw"] == []
    assert root["candidate_gains_by_mode"]["centered_excess"]
    assert root["category_tables_by_mode"]["centered_excess"][0][0]["D_cat"] != 0.0

    segments = model.get_impact_segments()
    assert len(segments) == 2
    assert np.isclose(float(segments["total_sum"].sum()), float(y.sum()), rtol=1e-6)


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


def test_noop_routing_skips_feature_prefers_partitioning_column() -> None:
    """Feature 0 routes every category to P (no row partition); must not be chosen."""
    x = np.array(
        [
            [0, 0],
            [1, 0],
            [0, 1],
            [1, 1],
        ],
        dtype=np.int64,
    )
    y = np.array([100.0, 100.0, -50.0, -50.0], dtype=float)
    model = ImpactSplitter(delta_pct=0.05, min_global_impact_pct=0.001, max_depth=3)

    model.fit(x, y, trace=True)

    root = model.fit_trace_[0]
    assert root["action"] == "split"
    assert root["chosen_feature_index"] == 1


def test_constant_feature_skipped_child_prefers_other_column() -> None:
    """After routing on col1, each child slice has a single col1 value (constant); split uses col0."""
    x = np.array(
        [
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1],
        ],
        dtype=np.int64,
    )
    y = np.array([100.0, -50.0, -20.0, 10.0], dtype=float)
    model = ImpactSplitter(delta_pct=0.05, min_global_impact_pct=0.001, max_depth=3)

    model.fit(x, y, trace=True)

    assert model.fit_trace_[0]["chosen_feature_index"] == 1
    depth1_splits = [
        e for e in model.fit_trace_ if e["depth"] == 1 and e["action"] == "split"
    ]
    assert depth1_splits
    assert all(e["chosen_feature_index"] == 0 for e in depth1_splits)


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


def test_plot_node_label_segment_on_every_node_split_on_internal_only() -> None:
    x = pd.DataFrame({"dim": ["A", "B", "A", "B"]})
    y = np.array([10.0, -10.0, 10.0, -10.0], dtype=float)
    model = ImpactSplitter(delta_pct=0.01, min_global_impact_pct=0.001, max_depth=2)
    model.fit(x, y)

    assert model._tree is not None
    assert not model._tree.is_leaf
    root_label = model._format_plot_node_label(model._tree)
    assert root_label.startswith("all data\n")
    assert "split on dim" in root_label
    assert "Σy" in root_label and "Σy⁺" in root_label and "Σy⁻" in root_label
    assert "n=" in root_label
    assert model._tree.children
    for ch in model._tree.children.values():
        ch_label = model._format_plot_node_label(ch)
        assert "dim=" in ch_label
        if ch.is_leaf:
            assert "split on" not in ch_label
        else:
            assert "split on" in ch_label


def test_format_plot_node_label_compact_omits_signed_sums() -> None:
    x = pd.DataFrame({"dim": ["A", "B", "A", "B"]})
    y = np.array([10.0, -10.0, 10.0, -10.0], dtype=float)
    model = ImpactSplitter(delta_pct=0.01, min_global_impact_pct=0.001, max_depth=2)
    model.fit(x, y)
    assert model._tree is not None
    full = model._format_plot_node_label(model._tree, compact_stats=False)
    compact = model._format_plot_node_label(model._tree, compact_stats=True)
    assert "Σy⁺" in full and "Σy⁻" in full
    assert "Σy⁺" not in compact and "Σy⁻" not in compact
    assert "n=" in compact and "Σy=" in compact


def test_plot_tree_smoke() -> None:
    x = np.array([[0], [1], [0], [1]], dtype=np.int64)
    y = np.array([1.0, -1.0, 1.0, -1.0], dtype=float)
    model = ImpactSplitter(delta_pct=0.01, min_global_impact_pct=0.001, max_depth=2)
    model.fit(x, y)

    fig = model.plot_tree(show=False, fontsize=6.0, edge_label_fontsize=5.0)
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    assert buf.tell() > 0
    plt.close(fig)


def test_plot_tree_label_truncation_smoke() -> None:
    x = np.array([[0], [1], [0], [1]], dtype=np.int64)
    y = np.array([1.0, -1.0, 1.0, -1.0], dtype=float)
    model = ImpactSplitter(delta_pct=0.01, min_global_impact_pct=0.001, max_depth=2)
    model.fit(x, y)

    fig = model.plot_tree(show=False, node_label_max_chars=20)
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    assert buf.tell() > 0
    plt.close(fig)


def test_plot_tree_max_leaf_width_budget_truncates_labels() -> None:
    long_name = "feature_" + "x" * 60
    x = pd.DataFrame({long_name: [0, 1, 0, 1]})
    y = np.array([1.0, -1.0, 1.0, -1.0], dtype=float)
    model = ImpactSplitter(delta_pct=0.01, min_global_impact_pct=0.001, max_depth=2)
    model.fit(x, y)

    fig_full = model.plot_tree(show=False, node_label_max_chars=None)
    texts_full = [t.get_text() for t in fig_full.axes[0].texts if "\n" in t.get_text()]
    plt.close(fig_full)
    assert texts_full
    max_len_full = max(len(s) for s in texts_full)

    fig_budget = model.plot_tree(show=False, node_label_max_chars=200, max_leaf_width=1.0)
    texts_budget = [t.get_text() for t in fig_budget.axes[0].texts if "\n" in t.get_text()]
    plt.close(fig_budget)
    assert texts_budget
    assert any("..." in txt for txt in texts_budget)
    assert max(len(s) for s in texts_budget) < max_len_full * 0.85


def test_plot_tree_impact_encoding_sets_contrasting_text_color() -> None:
    x = np.array([[0], [1], [0], [1]], dtype=np.int64)
    y = np.array([10.0, -10.0, 10.0, -10.0], dtype=float)
    model = ImpactSplitter(delta_pct=0.01, min_global_impact_pct=0.001, max_depth=2)
    model.fit(x, y)
    fig = model.plot_tree(show=False, node_facecolor="impact", compact_stats=True)
    node_texts = [t for t in fig.axes[0].texts if "\n" in t.get_text()]
    plt.close(fig)
    assert node_texts
    colors = {t.get_color() for t in node_texts}
    assert len(colors) >= 2


def test_plot_tree_layout_and_facecolor_smoke() -> None:
    x = np.array([[0], [1], [0], [1]], dtype=np.int64)
    y = np.array([1.0, -1.0, 1.0, -1.0], dtype=float)
    model = ImpactSplitter(delta_pct=0.01, min_global_impact_pct=0.001, max_depth=2)
    model.fit(x, y)

    fig = model.plot_tree(
        show=False,
        compact_stats=True,
        level_gap=1.4,
        sibling_gap=0.2,
        min_leaf_width=1.1,
        edge_label_pos=0.25,
        edge_label_bbox=False,
        node_facecolor="impact",
    )
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    assert buf.tell() > 0
    plt.close(fig)

    fig_n = model.plot_tree(show=False, node_facecolor="n")
    buf_n = io.BytesIO()
    fig_n.savefig(buf_n, format="png")
    assert buf_n.tell() > 0
    plt.close(fig_n)
