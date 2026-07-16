"""Text renderer: a designed fit summary for terminals, logs, and notebooks."""

from __future__ import annotations

from typing import Any

from impact_split.viz.data import fmt_num, fmt_pct


def render_summary(payload: dict[str, Any], *, top: int = 10, path_width: int = 44) -> str:
    """Ledger header + ranked segment table; always ends with a complete total story."""
    meta = payload["meta"]
    segments = payload["segments"]
    p = meta["params"]

    conservation = "exact ✓" if meta["conservation_exact"] else "MISMATCH ✗"
    merged_note = (
        f" ({meta['n_leaves']} leaves merged)"
        if p["consolidate"] and meta["n_segments"] < meta["n_leaves"]
        else ""
    )
    lines = [
        "ImpactSplitter — fit summary",
        "============================",
        (
            f"rows {meta['n_rows']:,} · features {meta['n_features']} · "
            f"params delta_pct={p['delta_pct']} noise_z={p['noise_z']} "
            f"max_depth={p['max_depth']} consolidate={p['consolidate']} "
            f"lookahead={p['lookahead']}"
        ),
        (
            f"total Σy {fmt_num(meta['total_sum'], sign=True)}   "
            f"(Σy⁺ {fmt_num(meta['pos_pool'])} · Σy⁻ -{fmt_num(meta['neg_pool'])})"
        ),
        (
            f"tree      {meta['n_nodes']} nodes · {meta['n_leaves']} leaves · "
            f"depth {meta['physical_depth']} "
            f"(interaction order {meta['interaction_depth']})"
        ),
    ]

    churn_note = (
        f" · {meta['n_churn_segments']} churn ⇄" if meta["n_churn_segments"] else ""
    )
    lines.extend([
        f"segments  {meta['n_segments']}{merged_note}{churn_note} · "
        f"conservation {conservation}",
        "",
        "Top segments by |impact|",
        f" {'#':>2}  {'path':<{path_width}}  {'Σy':>14}  {'n':>9}  {'pool share':>16}  {'gross ⇄':>22}",
    ])

    shown = segments[:top]
    rest = segments[top:]
    for i, seg in enumerate(shown, start=1):
        path = str(seg["path"])
        if len(path) > path_width:
            path = path[: path_width - 1] + "…"
        if seg["pool_share"] is not None:
            pool_label = "Σy⁺" if (seg["total_sum"] or 0.0) >= 0 else "Σy⁻"
            share = f"{fmt_pct(seg['pool_share'])} of {pool_label}"
        else:
            share = "—"
        gross = (
            f"+{fmt_num(seg['pos_sum'])} / -{fmt_num(seg['neg_sum'])}"
            if seg["is_churn"]
            else ""
        )
        lines.append(
            f" {i:>2}  {path:<{path_width}}  {fmt_num(seg['total_sum'], sign=True):>14}"
            f"  {seg['n']:>9,}  {share:>16}  {gross:>22}"
        )
    if rest:
        rest_total = sum(float(s["total_sum"] or 0.0) for s in rest)
        rest_n = sum(int(s["n"]) for s in rest)
        label = f"(+{len(rest)} more segments)"
        lines.append(
            f" {'…':>2}  {label:<{path_width}}  {fmt_num(rest_total, sign=True):>14}"
            f"  {rest_n:>9,}  {'':>16}  {'':>22}"
        )

    if meta["n_churn_segments"]:
        lines.append("")
        lines.append(
            " ⇄ churn segment: positive and negative flows are both material — "
            "the net hides offsetting mass (gross column shows both)."
        )

    return "\n".join(lines)
