"""Tests for the text renderer (summary / repr)."""

import pytest

from impact_split import ImpactSplitter
from tests.test_viz_data import fitted


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
