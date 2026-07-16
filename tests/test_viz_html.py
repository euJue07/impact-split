"""Tests for the self-contained HTML report."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from impact_split import ImpactSplitter
from tests.test_viz_data import churn_mix_fitted, fitted


def test_to_html_requires_fit() -> None:
    with pytest.raises(RuntimeError, match="fit\\(\\)"):
        ImpactSplitter().to_html()


def test_html_is_fully_self_contained() -> None:
    html_out = fitted().to_html()
    assert isinstance(html_out, str)
    for token in ("http://", "https://", "src=", "url(", "@import"):
        assert token not in html_out, f"external reference found: {token}"
    assert '"segments"' in html_out  # payload embedded


def test_html_escapes_script_closers_in_data() -> None:
    rng = np.random.default_rng(3)
    X = pd.DataFrame({"attack": rng.choice(["</script>", "safe"], size=200)})
    y = pd.Series(rng.normal(0, 1, size=200) + (X["attack"] == "</script>") * 9.0)
    html_out = ImpactSplitter().fit(X, y).to_html()
    # the only literal closing tag is the template's own single script block
    assert html_out.count("</script>") == 1


def test_html_write_mode_roundtrip(tmp_path: Path) -> None:
    out = fitted().to_html(tmp_path / "report.html")
    assert isinstance(out, Path)
    text = out.read_text(encoding="utf-8")
    assert text.startswith("<!doctype html>") and '"segments"' in text


def test_html_marks_churn() -> None:
    html_out = churn_mix_fitted().to_html()
    assert "churn segments" in html_out   # ledger tile
    assert "lookahead=" in html_out       # params line
    assert "tband" in html_out            # gross-band CSS + tornado renderer
    assert "stroke-dasharray" in html_out # icicle churn outline
    assert "Σy⁺" in html_out              # gross table columns
    # Data-dependent: only holds when churn exists in the output
    assert '"is_churn": true' in html_out
    # Verify the assertion is data-dependent by confirming it does NOT appear
    # in a churn-free model (strictly non-negative y, so negative pool is 0).
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"a": rng.choice(["x", "y"], size=200)})
    y = pd.Series(np.abs(rng.normal(0.0, 1.0, 200)) + (X["a"] == "x") * 5.0)
    html_no_churn = ImpactSplitter().fit(X, y).to_html()
    assert '"is_churn": true' not in html_no_churn
