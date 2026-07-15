"""Static matplotlib renderers: segment tornado and impact icicle."""

from __future__ import annotations

import textwrap
from typing import Any, cast

from matplotlib.colors import LinearSegmentedColormap, Normalize
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


def _fit_xlim_to_annotations(fig: Figure, ax: Any, texts: list[Any]) -> None:
    """Expand xlim so every annotation Text sits fully inside the axes.

    The caller's initial ``set_xlim`` is a heuristic guess based on the bar
    value range; the "n=... x% of Sy" note text can still overflow it when
    it's long relative to the value range (e.g. a small bar with a wide
    note). Measure the actual rendered text extents and grow xlim to
    contain them. Widening xlim actually makes a fixed-pixel-width text span
    *more* data units, not fewer — but the loop still converges: each pass's
    overshoot (the amount a text's data-unit footprint exceeds the current
    range) shrinks geometrically, because the text's pixel width is a small,
    fixed fraction of the axes' pixel width and each new xlim is only grown
    just enough to cover the previous overshoot, not open-endedly.
    """
    if not texts:
        return
    try:
        for _ in range(5):
            fig.canvas.draw()
            renderer = cast(Any, fig.canvas).get_renderer()
            inv = ax.transData.inverted()
            lo, hi = ax.get_xlim()
            new_lo, new_hi = lo, hi
            for text in texts:
                data_bbox = text.get_window_extent(renderer=renderer).transformed(inv)
                new_lo = min(new_lo, data_bbox.x0)
                new_hi = max(new_hi, data_bbox.x1)
            if new_lo >= lo and new_hi <= hi:
                return
            margin = 0.02 * (new_hi - new_lo)
            ax.set_xlim(new_lo - margin, new_hi + margin)
    except AttributeError:
        return


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
    annotation_texts = []
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
        annotation_texts.append(
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
        )
        annotation_texts.append(
            ax.text(
                value + offset, y - 0.2, note, va="center", ha=ha, fontsize=7.5, color=_MUTED_TEXT
            )
        )

    ax.set_yticks([len(bars) - 1 - i for i in range(len(bars))], labels=labels, fontsize=8)
    ax.set_xlim(
        min(0.0, min(values, default=0.0)) - 0.24 * max_abs,
        max(0.0, max(values, default=0.0)) + 0.24 * max_abs,
    )
    _fit_xlim_to_annotations(fig, ax, annotation_texts)
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


def layout_icicle(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Rect per node in root-relative [0, 1] coordinates; children tile their parent exactly.

    Widths are proportional to ``abs_volume`` (Σ|y|); a sibling group whose volumes
    are all zero falls back to row counts so the layout stays total.
    """
    nodes = payload["tree"]
    children: dict[str | None, list[dict[str, Any]]] = {}
    for node in nodes:
        children.setdefault(node["parent_id"], []).append(node)
    root = children[None][0]
    rects: list[dict[str, Any]] = []

    def place(node: dict[str, Any], x0: float, width: float) -> None:
        rects.append(
            {"id": node["id"], "x0": x0, "width": width, "depth": node["depth"], "node": node}
        )
        kids = children.get(node["id"], [])
        if not kids:
            return
        weights = [float(k["abs_volume"] or 0.0) for k in kids]
        total = sum(weights)
        if total <= 0:
            weights = [float(k["n"]) for k in kids]
            total = sum(weights) or 1.0
        cursor = x0
        for kid, w in zip(kids, weights, strict=True):
            kid_width = width * (w / total)
            place(kid, cursor, kid_width)
            cursor += kid_width

    place(root, 0.0, 1.0)
    return rects


def plot_icicle(
    payload: dict[str, Any],
    *,
    figsize: tuple[float, float] | None = None,
    show: bool = True,
) -> Figure:
    """Impact icicle: cell width ∝ Σ|y|, diverging color = mean excess vs overall mean."""
    from matplotlib.patches import Rectangle

    meta = payload["meta"]
    rects = layout_icicle(payload)
    depth_max = max(r["depth"] for r in rects)
    if figsize is None:
        figsize = (12.0, 1.15 * (depth_max + 1) + 1.9)
    fig, ax = plt.subplots(figsize=figsize)

    root_node = rects[0]["node"]
    root_mean = (root_node["total_sum"] or 0.0) / root_node["n"] if root_node["n"] else 0.0
    for r in rects:
        node = r["node"]
        mean = (node["total_sum"] or 0.0) / node["n"] if node["n"] else 0.0
        r["excess"] = mean - root_mean
    vmax = max((abs(r["excess"]) for r in rects), default=1.0) or 1.0
    norm = Normalize(vmin=-vmax, vmax=vmax)

    seg_leaf_count: dict[str, int] = {}
    for s in payload["segments"]:
        seg_leaf_count[s["segment_id"]] = len(s["node_ids"])

    fig_width_px = figsize[0] * 72.0
    for r in rects:
        node = r["node"]
        face = _DIVERGING_CMAP(norm(r["excess"]))[:3]
        merged_leaf = (
            node["is_leaf"]
            and node["segment_id"] is not None
            and seg_leaf_count.get(node["segment_id"], 1) > 1
        )
        ax.add_patch(
            Rectangle(
                (r["x0"], -r["depth"] - 0.94),
                r["width"],
                0.88,
                facecolor=face,
                edgecolor="#3a3a36" if merged_leaf else "white",
                linewidth=1.8 if merged_leaf else 1.1,
            )
        )
        label = f"{node['condition']}\n{fmt_num(node['total_sum'], sign=True)}"
        longest = max(len(line) for line in label.split("\n"))
        if r["width"] * fig_width_px * 0.86 >= longest * 5.0 and r["width"] >= 0.03:
            ax.text(
                r["x0"] + r["width"] / 2,
                -r["depth"] - 0.5,
                label,
                ha="center",
                va="center",
                fontsize=7.5,
                color=_text_color_for(face),
            )

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(-depth_max - 1.0, 0.0)
    ax.set_xticks([])
    ax.set_yticks(
        [-d - 0.5 for d in range(depth_max + 1)],
        labels=["root" if d == 0 else f"depth {d}" for d in range(depth_max + 1)],
        fontsize=8,
    )
    ax.tick_params(axis="y", length=0, colors=_MUTED_TEXT)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(
        f"Impact tree — width ∝ Σ|y| · total Σy {fmt_num(meta['total_sum'], sign=True)}",
        loc="left",
        fontsize=11,
        color=_INK,
    )
    mappable = plt.cm.ScalarMappable(cmap=_DIVERGING_CMAP, norm=norm)
    mappable.set_array([])
    cbar = fig.colorbar(mappable, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label("segment mean − overall mean", fontsize=8, color=_MUTED_TEXT)
    cbar.ax.tick_params(labelsize=7, colors=_MUTED_TEXT)
    fig.text(
        0.01,
        0.01,
        "each row tiles the one above (children are their parent's rows) · "
        "dark-outlined leaves merged into one consolidated segment",
        fontsize=7.5,
        color=_MUTED_TEXT,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    if show:
        plt.show()
    return fig
