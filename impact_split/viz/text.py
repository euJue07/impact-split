"""Text renderer: a designed fit summary for terminals, logs, and notebooks."""

from __future__ import annotations

from typing import Any

from impact_split.viz.data import fmt_num, fmt_pct


def render_summary(payload: dict[str, Any], *, top: int = 10, path_width: int = 44) -> str:
    """Ledger header + ranked segment table; always ends with a complete total story."""
    meta = payload["meta"]
    segments = payload["segments"]
    p = meta["params"]
    ens = payload.get("ensemble")

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
    ens_header = f"  {'stability':>10}  {'Σy 5–95%':>28}" if ens else ""
    lines.extend([
        f"segments  {meta['n_segments']}{merged_note}{churn_note} · "
        f"conservation {conservation}",
        "",
        "Top segments by |impact|",
        f" {'#':>2}  {'path':<{path_width}}  {'Σy':>14}  {'n':>9}  {'pool share':>16}  "
        f"{'gross ⇄':>22}{ens_header}",
    ])

    ens_missing_extra = f"  {'—':>10}  {'—':>28}"

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
        row = (
            f" {i:>2}  {path:<{path_width}}  {fmt_num(seg['total_sum'], sign=True):>14}"
            f"  {seg['n']:>9,}  {share:>16}  {gross:>22}"
        )
        row_extra = ""
        if ens:
            st = ens["segments"].get(str(seg.get("segment_id")))
            if st is None:
                row_extra = ens_missing_extra
            else:
                stab = fmt_pct(st["stability"]) + (" †" if st["fragile"] else "")
                ci = (
                    f"[{fmt_num(st['ci_low'], sign=True)}, {fmt_num(st['ci_high'], sign=True)}]"
                    if st["ci_low"] is not None
                    else "—"
                )
                row_extra = f"  {stab:>10}  {ci:>28}"
        lines.append(row + row_extra)
    if rest:
        rest_total = sum(float(s["total_sum"] or 0.0) for s in rest)
        rest_n = sum(int(s["n"]) for s in rest)
        label = f"(+{len(rest)} more segments)"
        rest_extra = ens_missing_extra if ens else ""
        lines.append(
            f" {'…':>2}  {label:<{path_width}}  {fmt_num(rest_total, sign=True):>14}"
            f"  {rest_n:>9,}  {'':>16}  {'':>22}{rest_extra}"
        )

    if meta["n_churn_segments"]:
        lines.append("")
        lines.append(
            " ⇄ churn segment: positive and negative flows are both material — "
            "the net hides offsetting mass (gross column shows both)."
        )

    if ens:
        if any(s["fragile"] for s in ens["segments"].values()):
            lines.append("")
            lines.append(
                " † fragile: segment re-emerged in <50% of bootstrap refits."
            )
        if ens["shadows"]:
            lines.append("")
            lines.append("Shadow drivers (material regions the main tree does not report)")
            for sh in ens["shadows"][:top]:
                lines.append(
                    f"  · {sh['path']}  Σy≈{fmt_num(sh['mean_impact'], sign=True)}"
                    f" · recurrence {fmt_pct(sh['recurrence'])}"
                    f" · via {', '.join(sh['features'])}"
                )

    return "\n".join(lines)
