"""Tests for static figure renderers (tornado + icicle)."""

import matplotlib

matplotlib.use("Agg")

from matplotlib.figure import Figure
import numpy as np
import pandas as pd
import pytest

from impact_split import ImpactSplitter
from tests.test_viz_data import _fitted, churn_mix_fitted, fitted


def test_plot_segments_requires_fit() -> None:
    with pytest.raises(RuntimeError, match="fit\\(\\)"):
        ImpactSplitter().plot_segments()


def test_plot_segments_returns_figure() -> None:
    fig = fitted().plot_segments(show=False)
    assert isinstance(fig, Figure)


def test_plot_segments_rolls_up_remainder() -> None:
    model = fitted()
    fig = model.plot_segments(top=1, show=False)
    labels = [t.get_text() for t in fig.axes[0].get_yticklabels()]
    assert any("more segments" in label for label in labels)


def test_plot_segments_annotation_text_stays_within_xlim() -> None:
    """Regression for the xlim-fit helper (_fit_xlim_to_annotations).

    The `fitted()` fixture naturally produces small-magnitude segments (e.g.
    total_sum ~ -10.5 and ~ -3.6, dwarfed by the ~896 max bar) whose "n=...
    x% of Sy" note text is wide relative to the bar's own data-unit span —
    the long-note-vs-small-bar scenario the xlim-fit loop must converge on.
    Mirrors the implementation: draw the canvas, grab the renderer via
    fig.canvas.get_renderer(), and transform each annotation Text's window
    extent into data coordinates.
    """
    model = fitted()
    fig = model.plot_segments(top=15, show=False)
    ax = fig.axes[0]

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inv = ax.transData.inverted()
    lo, hi = ax.get_xlim()
    tol = 1e-6 * (hi - lo)

    assert ax.texts, "expected annotation texts on the tornado chart"
    for text in ax.texts:
        data_bbox = text.get_window_extent(renderer=renderer).transformed(inv)
        assert data_bbox.x0 >= lo - tol
        assert data_bbox.x1 <= hi + tol


def test_plot_segments_root_only_model() -> None:
    # all-zero target -> materiality leaf at the root; must not raise
    X = pd.DataFrame({"a": ["x", "y"] * 10})
    y = pd.Series(np.zeros(20))
    model = ImpactSplitter().fit(X, y)
    fig = model.plot_segments(show=False)
    assert isinstance(fig, Figure)


def test_icicle_layout_children_tile_parent_exactly() -> None:
    from impact_split.viz.static import layout_icicle

    payload = fitted().to_dict()
    rects = {r["id"]: r for r in layout_icicle(payload)}
    kids_of: dict[str, list[str]] = {}
    for node in payload["tree"]:
        if node["parent_id"] is not None:
            kids_of.setdefault(node["parent_id"], []).append(node["id"])
    root_id = payload["tree"][0]["id"]
    assert rects[root_id]["x0"] == 0.0 and rects[root_id]["width"] == pytest.approx(1.0)
    for parent_id, kid_ids in kids_of.items():
        parent = rects[parent_id]
        assert sum(rects[k]["width"] for k in kid_ids) == pytest.approx(
            parent["width"], abs=1e-9
        )
        assert min(rects[k]["x0"] for k in kid_ids) == pytest.approx(parent["x0"], abs=1e-9)


def test_plot_tree_returns_figure_and_requires_fit() -> None:
    with pytest.raises(RuntimeError, match="fit\\(\\)"):
        ImpactSplitter().plot_tree()
    fig = fitted().plot_tree(show=False)
    assert isinstance(fig, Figure)


def test_plot_tree_root_only_model() -> None:
    X = pd.DataFrame({"a": ["x", "y"] * 10})
    y = pd.Series(np.zeros(20))
    fig = ImpactSplitter().fit(X, y).plot_tree(show=False)
    assert isinstance(fig, Figure)


def test_plot_segments_churn_band_and_gross_label() -> None:
    fig = churn_mix_fitted().plot_segments(show=False)
    ax = fig.axes[0]
    texts = [t.get_text() for t in ax.texts]
    assert any(t.startswith("net ") and "gross +" in t for t in texts)
    # the hatched gross band is a real patch (no rolled-up bar in this fixture)
    hatched = [p for p in ax.patches if p.get_hatch()]
    assert hatched


def test_plot_tree_churn_leaf_dashed_outline() -> None:
    fig = churn_mix_fitted().plot_tree(show=False)
    ax = fig.axes[0]
    dashed = [p for p in ax.patches if p.get_linestyle() not in ("solid", "-")]
    assert dashed


def test_tornado_draws_ci_whiskers_with_ensemble() -> None:
    model, X, y = _fitted()
    model.ensemble_report(X, y, n_replicates=12, shadow_replicates=0, seed=3)
    fig = model.plot_segments(show=False)
    texts = [t.get_text() for ax in fig.axes for t in ax.texts]
    assert any("stab" in t for t in texts)


def test_icicle_leaf_stability_with_ensemble() -> None:
    model, X, y = _fitted()
    model.ensemble_report(X, y, n_replicates=12, shadow_replicates=0, seed=3)
    fig = model.plot_tree(show=False)
    texts = [t.get_text() for ax in fig.axes for t in ax.texts]
    assert any("⟳" in t for t in texts)
