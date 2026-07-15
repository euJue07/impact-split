"""Tests for the pairwise lookahead rescue (v0.2.0)."""

from __future__ import annotations

import pytest

from impact_split import ImpactSplitter


def test_lookahead_constructor_validation() -> None:
    with pytest.raises(ValueError, match="lookahead"):
        ImpactSplitter(lookahead="yes")  # type: ignore[arg-type]


def test_lookahead_default_true_and_in_repr() -> None:
    model = ImpactSplitter()
    assert model.lookahead is True
    assert "lookahead=True" in repr(model)
