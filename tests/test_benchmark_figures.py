"""The validation figures must be reproducible from committed results JSON."""

from __future__ import annotations

from pathlib import Path

from benchmarks import figures


def test_load_scored_drops_null_and_merges_halves() -> None:
    rows = figures.load_scored("cycle4")
    assert len(rows) == 51
    assert all(row["case"] != "null" for row in rows)
    assert all("cart_impact_f1" in row for row in rows)


def test_current_tag_matches_published_headline() -> None:
    rows = figures.load_scored(figures.CURRENT_TAG)
    mean = sum(r["impact_f1"] for r in rows) / len(rows)
    assert len(rows) == 51
    assert round(mean, 4) == 0.9646
    assert round(min(r["impact_f1"] for r in rows), 4) == 0.8154


def test_all_three_figures_render(tmp_path: Path) -> None:
    written = [
        figures.plot_vs_cart(tmp_path),
        figures.plot_distribution(tmp_path),
        figures.plot_progression(tmp_path),
    ]
    assert {p.name for p in written} == {
        "validation-vs-cart.png",
        "validation-distribution.png",
        "story-progression.png",
    }
    for path in written:
        assert path.exists() and path.stat().st_size > 5_000
