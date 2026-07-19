"""The README's showcase assets must be regenerable, not hand-typed."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def make_showcase():
    spec = importlib.util.spec_from_file_location(
        "make_showcase", REPO_ROOT / "reports" / "make_showcase.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["make_showcase"] = module
    spec.loader.exec_module(module)
    return module


def test_demo_dataset_is_deterministic(make_showcase) -> None:
    x_first, y_first = make_showcase.build_demo()
    x_second, y_second = make_showcase.build_demo()
    assert x_first.equals(x_second)
    assert (y_first == y_second).all()


def test_summary_text_is_regenerable_and_nonempty(make_showcase) -> None:
    text = make_showcase.render_summary_text()
    # Tightened in Step 6 against the real generated ledger: both header
    # tokens below appear verbatim in reports/showcase/summary.txt.
    assert "ImpactSplitter — fit summary" in text
    assert "Top segments by |impact|" in text
    assert len(text.splitlines()) > 5


def test_segments_markdown_has_documented_columns(make_showcase) -> None:
    table = make_showcase.render_segments_markdown()
    for column in ("path", "total_sum", "n_samples", "mean", "pool_share"):
        assert column in table
