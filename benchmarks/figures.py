"""Regenerate the validation figures from committed benchmark results.

Reads ``benchmarks/results/*.json`` only — no refit, no network, no Kaggle
credentials. Rendering is idempotent given the same inputs.

Usage:  python -m benchmarks.figures
"""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
from statistics import mean
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402  (must follow the backend selection)

from impact_split.viz.data import (  # noqa: E402
    NEGATIVE_COLOR,
    NEUTRAL_STROKE,
    POSITIVE_COLOR,
)

RESULTS_DIR = Path(__file__).parent / "results"
FIGURES_DIR = Path(__file__).resolve().parents[1] / "reports" / "figures"

#: The run in which CART was scored alongside impact-split (v0.1.0 floor loop).
CART_TAG = "cycle4"
#: The suite at current shipped defaults (v0.2.0 lookahead + v0.3.0 annotations).
CURRENT_TAG = "v020-post3"
#: Pre-registered floor bar from the 2026-07-15 floor loop.
FLOOR_BAR = 0.85

#: Development cycles in order, with the label shown on the progression chart.
PROGRESSION: list[tuple[str, str]] = [
    ("cycle0", "cycle 0\nraw-sum draft"),
    ("cycle1", "cycle 1\ncentered excess"),
    ("cycle2", "cycle 2\nnoise floor"),
    ("cycle3", "cycle 3\ndelta + interaction cap"),
    ("cycle4", "cycle 4\nconsolidation"),
    ("v020-post3", "v0.2.0\nlookahead"),
]
#: Scored but never shipped — the refuted relaxed-merge variant.
REFUTED_TAG = "cycle5"

_INK = "#26261f"
_MUTED_TEXT = "#5f5f5c"
_GRID = "#e6e6e2"


def load_scored(tag: str) -> list[dict[str, Any]]:
    """Merged synthetic+kaggle rows for ``tag``, excluding the unscored null case."""
    rows: list[dict[str, Any]] = []
    for half in ("synthetic", "kaggle"):
        path = RESULTS_DIR / f"{tag}-{half}.json"
        rows.extend(json.loads(path.read_text(encoding="utf-8"))["results"])
    return [row for row in rows if row["case"] != "null"]


def _by_case(rows: list[dict[str, Any]], key: str) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[row["case"]].append(row[key])
    return {case: mean(values) for case, values in grouped.items()}


def _style(ax: plt.Axes) -> None:
    ax.grid(axis="x", color=_GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#d8d8d4")
    ax.tick_params(labelsize=8, colors=_MUTED_TEXT)


def _save(fig: plt.Figure, outdir: Path, name: str) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_vs_cart(outdir: Path = FIGURES_DIR) -> Path:
    """Per-case impact-F1: impact-split vs a depth-matched CART reference."""
    rows = load_scored(CART_TAG)
    ours = _by_case(rows, "impact_f1")
    cart = _by_case(rows, "cart_impact_f1")
    cases = sorted(ours, key=lambda c: ours[c] - cart[c])

    y = range(len(cases))
    fig, ax = plt.subplots(figsize=(8.2, 0.42 * len(cases) + 1.8))
    ax.barh(
        [i + 0.19 for i in y],
        [ours[c] for c in cases],
        height=0.36,
        color=POSITIVE_COLOR,
        label="impact-split",
        zorder=2,
    )
    ax.barh(
        [i - 0.19 for i in y],
        [cart[c] for c in cases],
        height=0.36,
        color=NEUTRAL_STROKE,
        label="CART (variance)",
        zorder=2,
    )
    for i, case in enumerate(cases):
        delta = ours[case] - cart[case]
        ax.text(
            max(ours[case], cart[case]) + 0.012,
            i,
            f"{delta:+.3f}",
            va="center",
            fontsize=7.5,
            color=POSITIVE_COLOR if delta > 0 else NEGATIVE_COLOR,
        )
    ax.set_yticks(list(y))
    ax.set_yticklabels([c.replace("kaggle_", "") for c in cases], fontsize=8)
    ax.set_xlim(0, 1.08)
    ax.set_xlabel("impact-weighted F1 (mean over 3 seeds)", fontsize=8, color=_MUTED_TEXT)
    # Anchored outside the axes (top-right margin) so it never sits on top of a
    # bar — every row, including the noise_2x loss, must stay fully visible.
    ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.005, 1.0))
    _style(ax)
    ax.set_title(
        "Planted-rule recovery vs CART — one fixed configuration, no per-dataset tuning",
        fontsize=10,
        color=_INK,
        loc="left",
        pad=12,
    )
    fig.text(
        0.01,
        0.005,
        f"Bars are per-case means over 3 seeds ({CART_TAG}). Across all 51 scored "
        "dataset-seeds impact-split wins 37. "
        "Source: benchmarks/results/ — regenerate with python -m benchmarks.figures",
        fontsize=7,
        color=_MUTED_TEXT,
    )
    return _save(fig, outdir, "validation-vs-cart.png")


def plot_distribution(outdir: Path = FIGURES_DIR) -> Path:
    """Every scored dataset-seed against the pre-registered 0.85 floor bar."""
    rows = load_scored(CURRENT_TAG)
    scores = sorted(row["impact_f1"] for row in rows)
    below = [s for s in scores if s < FLOOR_BAR]

    fig, ax = plt.subplots(figsize=(8.2, 3.4))
    ax.scatter(
        range(len(scores)),
        scores,
        s=26,
        zorder=3,
        color=[POSITIVE_COLOR if s >= FLOOR_BAR else NEGATIVE_COLOR for s in scores],
    )
    ax.axhline(FLOOR_BAR, color=NEGATIVE_COLOR, linewidth=1.1, linestyle="--", zorder=2)
    # Placed past the ascending elbow (all points here are >=0.95, well clear of
    # the floor line) instead of at the crowded left edge, where the label used
    # to render through the x=3 point (score ~0.861).
    ax.text(
        20,
        FLOOR_BAR + 0.006,
        f"pre-registered floor bar {FLOOR_BAR:.2f}",
        fontsize=7.5,
        color=NEGATIVE_COLOR,
    )
    ax.set_xlabel("dataset-seed (sorted)", fontsize=8, color=_MUTED_TEXT)
    ax.set_ylabel("impact-weighted F1", fontsize=8, color=_MUTED_TEXT)
    ax.set_ylim(min(scores) - 0.04, 1.02)
    _style(ax)
    ax.grid(axis="y", color=_GRID, linewidth=0.8, zorder=0)
    ax.set_title(
        f"{len(scores) - len(below)} of {len(scores)} dataset-seeds clear the bar "
        f"— the {len(below)} that do not are explained, not hidden",
        fontsize=10,
        color=_INK,
        loc="left",
        pad=12,
    )
    fig.text(
        0.01,
        -0.05,
        "Source: benchmarks/results/ — see reports/validation-report-v3.md §4 for the three cases.",
        fontsize=7,
        color=_MUTED_TEXT,
    )
    return _save(fig, outdir, "validation-distribution.png")


def plot_progression(outdir: Path = FIGURES_DIR) -> Path:
    """Suite mean and floor across development cycles, including a refuted variant."""
    labels, means, floors = [], [], []
    for tag, label in PROGRESSION:
        scores = [row["impact_f1"] for row in load_scored(tag)]
        labels.append(label)
        means.append(mean(scores))
        floors.append(min(scores))

    refuted = [row["impact_f1"] for row in load_scored(REFUTED_TAG)]
    x = range(len(labels))

    fig, ax = plt.subplots(figsize=(8.2, 3.8))
    ax.plot(
        x, means, marker="o", color=POSITIVE_COLOR, linewidth=1.8, label="suite mean", zorder=3
    )
    ax.plot(
        x, floors, marker="o", color=NEGATIVE_COLOR, linewidth=1.8, label="suite floor", zorder=3
    )
    ax.axhline(FLOOR_BAR, color=NEUTRAL_STROKE, linewidth=1.0, linestyle="--", zorder=1)
    ax.text(0.02, FLOOR_BAR + 0.015, "floor bar 0.85", fontsize=7.5, color=_MUTED_TEXT)

    # The refuted relaxed-merge variant sat between cycle4 and v0.2.0 and lost ground.
    refuted_x = len(labels) - 1.5
    ax.scatter([refuted_x], [min(refuted)], marker="x", s=60, color=NEGATIVE_COLOR, zorder=4)
    ax.annotate(
        f"refuted variant\nfloor {min(refuted):.3f}",
        xy=(refuted_x, min(refuted)),
        xytext=(refuted_x - 1.1, min(refuted) - 0.13),
        fontsize=7.5,
        color=_MUTED_TEXT,
        arrowprops={"arrowstyle": "->", "color": NEUTRAL_STROKE, "linewidth": 0.8},
    )

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_ylim(0, 1.06)
    ax.set_ylabel("impact-weighted F1 (51 dataset-seeds)", fontsize=8, color=_MUTED_TEXT)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    _style(ax)
    ax.grid(axis="y", color=_GRID, linewidth=0.8, zorder=0)
    ax.set_title(
        "What each loop bought — and what it cost",
        fontsize=10,
        color=_INK,
        loc="left",
        pad=12,
    )
    return _save(fig, outdir, "story-progression.png")


def main() -> None:
    for path in (plot_vs_cart(), plot_distribution(), plot_progression()):
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
