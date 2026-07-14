"""Impact-weighted F1 scoring for planted-rule recovery.

Per planted rule R with per-row contribution e_R and matched row-set M
(a union of at most ``max_union`` terminal segments):

- recall  = |sum(e_R over M)| / |sum(e_R over all rows)|
- precision (raw) = |sum(e_R over M)| / sum(|y_obs| over M)
- precision is normalized by the precision of the *true* rule mask itself
  (so perfect recovery scores 1.0 at any noise level), capped at 1.

Row-Jaccard against the best single segment is kept as a shape diagnostic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .dgp import BenchDataset


@dataclass
class RuleScore:
    rule: str
    f1: float
    recall: float
    precision: float
    n_segments_used: int
    best_jaccard: float
    matched_paths: list[str]


@dataclass
class DatasetScore:
    case: str
    seed: int
    impact_f1: float  # mean over rules; None-like -1 for null case
    floor_rule_f1: float
    rule_scores: list[RuleScore]
    n_terminal_segments: int
    null_pass: bool | None  # only set for null case
    conservation_ok: bool
    extras: dict[str, Any]


def leaf_masks_from_model(model: Any, X_codes: np.ndarray) -> list[tuple[str, np.ndarray]]:
    """Recompute terminal-segment row masks by walking the fitted tree.

    Avoids parsing ``path`` strings (which truncate at 8 labels for
    high-cardinality features). Returns ``(path, bool_mask)`` per leaf.
    """
    tree = model._tree
    if tree is None:
        raise RuntimeError("model must be fitted")
    n = X_codes.shape[0]
    leaves: list[tuple[str, np.ndarray]] = []

    def rec(node: Any, mask: np.ndarray) -> None:
        if node.is_leaf or not node.children:
            leaves.append((node.path, mask))
            return
        col = X_codes[:, node.feature_index]
        routing = node.routing
        pos = np.isin(col, routing["positive"]) & mask
        neg = np.isin(col, routing["negative"]) & mask
        neu = mask & ~(np.isin(col, routing["positive"]) | np.isin(col, routing["negative"]))
        for key, m in (("positive", pos), ("negative", neg), ("neutral", neu)):
            ch = node.children.get(key)
            if ch is not None:
                rec(ch, m)

    rec(tree, np.ones(n, dtype=bool))
    return leaves


def encode_with_model_maps(model: Any, X_df: Any) -> np.ndarray:
    """Encode a DataFrame into the integer codes the fitted model uses."""
    maps = model.category_maps_
    cols = []
    for i, name in enumerate(X_df.columns):
        lookup = {v: c for c, v in enumerate(maps[i])}
        cols.append(X_df[name].map(lookup).to_numpy(dtype=np.int64))
    return np.column_stack(cols)


def _f1(p: float, r: float) -> float:
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def score_dataset(
    ds: BenchDataset,
    leaves: list[tuple[str, np.ndarray]],
    *,
    max_union: int = 3,
    null_materiality_pct: float = 0.05,
) -> DatasetScore:
    """Score one fitted tree's terminal segments against the planted rules."""
    y = ds.y
    abs_y = np.abs(y)
    total_sum = float(y.sum())
    seg_sum = float(sum(y[m].sum() for _, m in leaves))
    conservation_ok = bool(np.isclose(total_sum, seg_sum, rtol=1e-6, atol=1e-6))

    if not ds.rules:  # null case: no split should survive
        # Materiality of any non-root partition: does the tree isolate a segment
        # whose |total| exceeds null_materiality_pct of global |y| volume?
        v_total = float(abs_y.sum())
        worst = 0.0
        if len(leaves) > 1:
            worst = max(abs(float(y[m].sum())) / v_total for _, m in leaves)
        null_pass = len(leaves) == 1 or worst < null_materiality_pct
        return DatasetScore(
            ds.case,
            ds.seed,
            impact_f1=float("nan"),
            floor_rule_f1=float("nan"),
            rule_scores=[],
            n_terminal_segments=len(leaves),
            null_pass=null_pass,
            conservation_ok=conservation_ok,
            extras={"worst_null_segment_share": worst},
        )

    rule_scores: list[RuleScore] = []
    for rule in ds.rules:
        contrib = rule.contrib
        total_contrib = abs(float(contrib.sum()))
        if total_contrib == 0:
            continue
        ideal_mass = float(abs_y[rule.mask].sum())
        precision_ideal = (
            abs(float(contrib[rule.mask].sum())) / ideal_mass if ideal_mass > 0 else 1.0
        )

        # Rank segments by how much of this rule's contribution they hold.
        ranked = sorted(
            range(len(leaves)),
            key=lambda i: -abs(float(contrib[leaves[i][1]].sum())),
        )

        best = (0.0, 0.0, 0.0, [])  # f1, recall, precision, indices
        m_union = np.zeros(len(y), dtype=bool)
        chosen: list[int] = []
        for idx in ranked[: max_union * 2]:  # small candidate pool
            trial = m_union | leaves[idx][1]
            captured = abs(float(contrib[trial].sum()))
            recall = min(1.0, captured / total_contrib)
            mass = float(abs_y[trial].sum())
            prec_raw = captured / mass if mass > 0 else 0.0
            precision = min(1.0, prec_raw / precision_ideal) if precision_ideal > 0 else 0.0
            f1 = _f1(precision, recall)
            if f1 > best[0]:
                m_union = trial
                chosen = chosen + [idx]
                best = (f1, recall, precision, list(chosen))
            if len(chosen) >= max_union:
                break

        # Shape diagnostic: best single-segment row-Jaccard.
        best_j = 0.0
        for _, m in leaves:
            inter = int((rule.mask & m).sum())
            union = int((rule.mask | m).sum())
            j = inter / union if union else 0.0
            best_j = max(best_j, j)

        rule_scores.append(
            RuleScore(
                rule=rule.label,
                f1=best[0],
                recall=best[1],
                precision=best[2],
                n_segments_used=len(best[3]),
                best_jaccard=best_j,
                matched_paths=[leaves[i][0] for i in best[3]],
            )
        )

    f1s = [rs.f1 for rs in rule_scores]
    return DatasetScore(
        ds.case,
        ds.seed,
        impact_f1=float(np.mean(f1s)) if f1s else 0.0,
        floor_rule_f1=float(np.min(f1s)) if f1s else 0.0,
        rule_scores=rule_scores,
        n_terminal_segments=len(leaves),
        null_pass=None,
        conservation_ok=conservation_ok,
        extras={},
    )
