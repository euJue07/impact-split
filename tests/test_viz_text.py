"""Tests for the text renderer (summary / repr)."""

import numpy as np
import pandas as pd
import pytest

from impact_split import ImpactSplitter
from tests.test_viz_data import churn_mix_fitted, fitted


def test_summary_requires_fit() -> None:
    with pytest.raises(RuntimeError, match="fit\\(\\)"):
        ImpactSplitter().summary()


def test_summary_ledger_and_table() -> None:
    model = fitted()
    text = model.summary()
    assert "ImpactSplitter — fit summary" in text
    assert "total Σy" in text and "Σy⁺" in text and "Σy⁻" in text
    assert "conservation exact ✓" in text
    assert "Top segments by |impact|" in text
    # every displayed segment row shows a pool-share annotation
    assert "of Σy⁺" in text or "of Σy⁻" in text


def test_summary_rolls_up_remainder() -> None:
    model = fitted()
    n_segments = model.to_dict()["meta"]["n_segments"]
    if n_segments < 2:
        pytest.skip("fixture produced a single segment")
    text = model.summary(top=1)
    assert f"(+{n_segments - 1} more segments)" in text


def test_repr_pre_and_post_fit() -> None:
    model = ImpactSplitter()
    assert repr(model).startswith("ImpactSplitter(delta_pct=")
    fitted_model = fitted()
    assert "fit summary" in repr(fitted_model)


def test_summary_flags_churn_segments() -> None:
    text = churn_mix_fitted().summary()
    assert "lookahead=True" in text
    assert "churn ⇄" in text          # segments ledger line
    assert "gross ⇄" in text          # table column header
    assert " / -" in text             # gross column rendered for the churn row
    assert "offsetting mass" in text  # footnote


def test_summary_without_churn_has_no_footnote() -> None:
    # Strictly non-negative target: the negative pool is 0, so no segment can
    # ever flag churn. (Do NOT use fitted() here — its symmetric noise gives
    # the catch-all segments material gross flows in BOTH directions, which
    # correctly flags them as churn under the dual-pool rule.)
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"a": rng.choice(["x", "y"], size=200)})
    y = pd.Series(np.abs(rng.normal(0.0, 1.0, 200)) + (X["a"] == "x") * 5.0)
    text = ImpactSplitter().fit(X, y).summary()
    assert "offsetting mass" not in text
    assert "churn ⇄" not in text
