"""Static matplotlib renderers: segment tornado and impact icicle."""

from __future__ import annotations

import textwrap
from typing import Any

from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

from impact_split.viz.data import (
    NEGATIVE_COLOR,
    NEUTRAL_FILL,
    NEUTRAL_STROKE,
    POSITIVE_COLOR,
    fmt_num,
    fmt_pct,
)

_MUTED_TEXT = "#5f5f5c"
_INK = "#26261f"
_ROLLED_FILL = "#c9c9c5"
_GRID = "#e6e6e2"

_DIVERGING_CMAP = LinearSegmentedColormap.from_list(
    "impact_diverging", [NEGATIVE_COLOR, NEUTRAL_FILL, POSITIVE_COLOR]
)


def _text_color_for(rgb: tuple[float, float, float]) -> str:
    """Black or white ink for readable text on the given face color."""

    def lin(c: float) -> float:
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

    luminance = 0.2126 * lin(rgb[0]) + 0.7152 * lin(rgb[1]) + 0.0722 * lin(rgb[2])
    return "white" if luminance < 0.45 else _INK


def plot_segments(
    payload: dict[str, Any],
    *,
    top: int = 15,
    figsize: tuple[float, float] | None = None,
    show: bool = True,
) -> Figure:
    """Tornado chart: consolidated segments diverging at zero, largest |Σy| on top."""
    meta = payload["meta"]
    segments = payload["segments"]
    bars: list[dict[str, Any]] = [{**s, "_rolled": False} for s in segments[:top]]
    rest = segments[top:]
    if rest:
        bars.append(
            {
                "path": f"(+{len(rest)} more segments)",
                "total_sum": sum(float(s["total_sum"] or 0.0) for s in rest),
                "n": sum(int(s["n"]) for s in rest),
                "pool_share": None,
                "_rolled": True,
            }
        )

    if figsize is None:
        figsize = (11.0, max(3.2, 0.62 * len(bars) + 1.8))
    fig, ax = plt.subplots(figsize=figsize)

    values = [float(b["total_sum"] or 0.0) for b in bars]
    max_abs = max((abs(v) for v in values), default=1.0) or 1.0
    pad = 0.015 * max_abs
    labels: list[str] = []
    for i, (b, value) in enumerate(zip(bars, values, strict=True)):
        y = len(bars) - 1 - i
        if b["_rolled"]:
            face, hatch = _ROLLED_FILL, "///"
        else:
            face, hatch = (POSITIVE_COLOR if value >= 0 else NEGATIVE_COLOR), None
        ax.barh(y, value, height=0.72, color=face, hatch=hatch, edgecolor="white", linewidth=0.8)
        labels.append(textwrap.fill(str(b["path"]), width=38))
        note = f"n={b['n']:,}"
        if b["pool_share"] is not None:
            note += f" · {fmt_pct(b['pool_share'])} of {'Σy⁺' if value >= 0 else 'Σy⁻'}"
        offset = pad if value >= 0 else -pad
        ha = "left" if value >= 0 else "right"
        ax.text(
            value + offset,
            y + 0.16,
            fmt_num(value, sign=True),
            va="center",
            ha=ha,
            fontsize=9,
            fontweight="bold",
            color=_INK,
        )
        ax.text(value + offset, y - 0.2, note, va="center", ha=ha, fontsize=7.5, color=_MUTED_TEXT)

    ax.set_yticks([len(bars) - 1 - i for i in range(len(bars))], labels=labels, fontsize=8)
    ax.set_xlim(
        min(0.0, min(values, default=0.0)) - 0.24 * max_abs,
        max(0.0, max(values, default=0.0)) + 0.24 * max_abs,
    )
    ax.axvline(0.0, color=NEUTRAL_STROKE, linewidth=1.0, zorder=0)
    ax.grid(axis="x", color=_GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#d8d8d4")
    ax.tick_params(axis="x", labelsize=8, colors=_MUTED_TEXT)
    ax.tick_params(axis="y", length=0)
    ax.set_title(
        f"Impact segments — {meta['n_segments']} segments · "
        f"total Σy {fmt_num(meta['total_sum'], sign=True)}",
        loc="left",
        fontsize=11,
        color=_INK,
    )
    conservation = "exact ✓" if meta["conservation_exact"] else "MISMATCH ✗"
    fig.text(
        0.01,
        0.01,
        f"bars are additive: sum of all segments = total Σy ({conservation}) · "
        f"blue = positive impact · orange = negative",
        fontsize=7.5,
        color=_MUTED_TEXT,
    )
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    if show:
        plt.show()
    return fig
