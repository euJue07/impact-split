"""Tests for the self-contained HTML report."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from impact_split import ImpactSplitter
from tests.test_viz_data import fitted


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
