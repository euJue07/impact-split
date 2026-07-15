"""Tests for static figure renderers (tornado + icicle)."""

import matplotlib

matplotlib.use("Agg")

from matplotlib.figure import Figure
import numpy as np
import pandas as pd
import pytest

from impact_split import ImpactSplitter
from tests.test_viz_data import fitted


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


def test_plot_segments_root_only_model() -> None:
    # all-zero target -> materiality leaf at the root; must not raise
    X = pd.DataFrame({"a": ["x", "y"] * 10})
    y = pd.Series(np.zeros(20))
    model = ImpactSplitter().fit(X, y)
    fig = model.plot_segments(show=False)
    assert isinstance(fig, Figure)
