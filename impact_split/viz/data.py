"""Payload builder: one JSON-safe dict that feeds every renderer."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from impact_split.splitter import ImpactSplitter, _TreeNode

# Diverging pair validated against the dataviz six-checks (light surface):
# blue = positive, orange = negative, neutral gray midpoint. Never green/red.
POSITIVE_COLOR = "#0173B2"
NEGATIVE_COLOR = "#C6660A"
NEUTRAL_FILL = "#F2F2F0"
NEUTRAL_STROKE = "#949494"


def safe_float(value: Any) -> float | None:
    """Finite float or None — NaN/inf never reach a renderer or JSON."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def fmt_num(value: float | None, *, sign: bool = False) -> str:
    """Human number: thousands separators; decimals only when small."""
    if value is None:
        return "—"
    prefix = "+" if sign and value > 0 else ""
    if abs(value) >= 1000:
        return f"{prefix}{value:,.0f}"
    if abs(value) >= 1:
        return f"{prefix}{value:,.2f}"
    return f"{prefix}{value:.4g}"


def fmt_pct(value: float | None) -> str:
    return "—" if value is None else f"{100.0 * value:.1f}%"


def _package_version() -> str:
    try:
        from importlib.metadata import version

        return version("impact_split")
    except Exception:  # pragma: no cover - metadata absent in odd envs
        return "unknown"


def build_payload(model: ImpactSplitter) -> dict[str, Any]:
    """Serialize a fitted model into the shared ``meta``/``tree``/``segments`` payload."""
    tree = model._tree
    if tree is None:
        raise RuntimeError("Call fit() before to_dict() or any renderer.")

    pos_pool = float(model._v_global_p)
    neg_pool = float(model._v_global_n)

    min_pct = float(model.min_global_impact_pct)

    def churn_mass(seg: dict[str, Any]) -> float:
        if not seg["is_churn"]:
            return 0.0
        return min(float(seg["pos_sum"]), float(seg["neg_sum"]))

    # Churn segments rank by their offsetting mass, not their (misleading) net.
    seg_sorted = sorted(
        model.segments_,
        key=lambda s: -max(abs(float(s["total_sum"])), churn_mass(s)),
    )
    segments: list[dict[str, Any]] = []
    node_to_segment: dict[str, str] = {}
    for rank, seg in enumerate(seg_sorted):
        seg_id = f"s{rank}"
        total = float(seg["total_sum"])
        n = int(seg["n_samples"])
        pool = pos_pool if total >= 0 else neg_pool
        segments.append(
            {
                "segment_id": seg_id,
                "path": seg["path"],
                "node_ids": list(seg["node_ids"]),
                "n": n,
                "total_sum": safe_float(total),
                "mean": safe_float(total / n) if n else None,
                "pool_share": safe_float(abs(total) / pool) if pool > 0 else None,
                "pos_sum": safe_float(seg["pos_sum"]),
                "neg_sum": safe_float(seg["neg_sum"]),
                "is_churn": bool(seg["is_churn"]),
            }
        )
        for nid in seg["node_ids"]:
            node_to_segment[nid] = seg_id

    nodes: list[dict[str, Any]] = []
    stats = {"leaves": 0, "depth": 0, "inter_depth": 0}

    def condition_of(node: _TreeNode) -> str:
        if not node.path or node.path == "root":
            return "all data"
        parts = [p.strip() for p in node.path.split(" / ") if p.strip()]
        return parts[-1] if parts else "all data"

    def walk(
        node: _TreeNode,
        parent_id: str | None,
        branch: str,
        inter_depth: int,
        parent_feature: int | None,
    ) -> None:
        is_leaf = node.is_leaf or not node.children
        stats["depth"] = max(stats["depth"], node.depth)
        stats["inter_depth"] = max(stats["inter_depth"], inter_depth)
        if is_leaf:
            stats["leaves"] += 1
        split_feature = (
            None
            if is_leaf or node.feature_index is None
            else model._feature_display_name(node.feature_index)
        )
        node_pos = float(node.s_node_p)
        node_neg = float(node.s_node_n)
        node_churn = bool(
            is_leaf
            and pos_pool > 0
            and neg_pool > 0
            and node_pos / pos_pool > min_pct
            and node_neg / neg_pool > min_pct
        )
        nodes.append(
            {
                "id": node.node_id,
                "parent_id": parent_id,
                "branch": branch,
                "depth": node.depth,
                "condition": condition_of(node),
                "split_feature": split_feature,
                "n": int(node.n_samples),
                "total_sum": safe_float(node.total_sum),
                "pos_sum": safe_float(node.s_node_p),
                "neg_sum": safe_float(node.s_node_n),
                "abs_volume": safe_float(node.s_node_p + node.s_node_n),
                "is_leaf": is_leaf,
                "is_churn": node_churn,
                "segment_id": node_to_segment.get(node.node_id),
            }
        )
        if is_leaf:
            return
        child_inter = inter_depth + (1 if node.feature_index != parent_feature else 0)
        for key in ("positive", "neutral", "negative"):
            child = (node.children or {}).get(key)
            if child is not None:
                walk(child, node.node_id, key, child_inter, node.feature_index)

    walk(tree, None, "root", 0, None)

    total_sum = float(tree.total_sum)
    seg_total = sum(float(s["total_sum"] or 0.0) for s in segments)
    n_features = int(model._X.shape[1]) if model._X is not None else 0
    feature_names = (
        list(model.feature_names_in_)
        if model.feature_names_in_ is not None
        else [f"f{i}" for i in range(n_features)]
    )

    payload: dict[str, Any] = {
        "meta": {
            "package_version": _package_version(),
            "params": {
                "delta_pct": safe_float(model.delta_pct),
                "min_global_impact_pct": safe_float(model.min_global_impact_pct),
                "max_depth": int(model.max_depth),
                "noise_z": safe_float(model.noise_z),
                "consolidate": bool(model.consolidate),
                "lookahead": bool(model.lookahead),
            },
            "n_rows": int(tree.n_samples),
            "n_features": n_features,
            "feature_names": feature_names,
            "total_sum": safe_float(total_sum),
            "pos_pool": safe_float(pos_pool),
            "neg_pool": safe_float(neg_pool),
            "n_nodes": len(nodes),
            "n_leaves": stats["leaves"],
            "physical_depth": stats["depth"],
            "interaction_depth": stats["inter_depth"],
            "n_segments": len(segments),
            "n_churn_segments": sum(1 for s in segments if s["is_churn"]),
            "conservation_exact": abs(seg_total - total_sum) <= 1e-9 * max(1.0, abs(total_sum)),
        },
        "tree": nodes,
        "segments": segments,
    }

    ens = getattr(model, "ensemble_", None)
    if ens is not None:
        order = {id(s): i for i, s in enumerate(model.segments_)}
        per_seg: dict[str, Any] = {}
        for rank, seg in enumerate(seg_sorted):
            st = ens["segments"][order[id(seg)]]
            per_seg[f"s{rank}"] = {
                "stability": safe_float(st["stability"]),
                "n_matched": int(st["n_matched"]),
                "ci_low": safe_float(st["ci_low"]),
                "ci_high": safe_float(st["ci_high"]),
                "fragile": bool(st["fragile"]),
            }
        payload["ensemble"] = {
            "config": dict(ens["config"]),
            "segments": per_seg,
            "importance": [dict(r) for r in ens["importance"]],
            "shadows": [dict(s) for s in ens["shadows"]],
        }

    return payload
