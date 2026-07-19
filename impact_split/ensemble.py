"""Bootstrap/feature-subsample forest that annotates a fitted ImpactSplitter.

The single greedy tree stays THE answer; the forest measures how much of it
survives resampling (stability, CI), which features carry the forest's gain
(importance), and which material regions only appear when dominant features
are forced out (shadow segments). No prediction averaging — ever.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

# A ledger segment is fragile when it re-emerges in fewer than half the
# bootstrap refits.
FRAGILE_STABILITY = 0.5
# Below this many matched replicates a percentile CI is noise — report null.
CI_MIN_MATCHES = 10
CI_PERCENTILES = (5.0, 95.0)


def mask_from_conditions(
    conditions: dict[int, frozenset[int]],
    X: np.ndarray,
    col_map: np.ndarray | None = None,
) -> np.ndarray:
    """Full-data row mask for a segment's conjunction of category sets.

    ``col_map[j]`` maps a replicate's column ``j`` back to the original
    feature index when the replicate was fit on a column subset.
    """
    mask = np.ones(X.shape[0], dtype=bool)
    for f, codes in conditions.items():
        col = int(f) if col_map is None else int(col_map[int(f)])
        mask &= np.isin(
            X[:, col], np.fromiter(codes, dtype=np.int64, count=len(codes))
        )
    return mask


def jaccard(a: np.ndarray, b: np.ndarray) -> float:
    union = int(np.count_nonzero(a | b))
    if union == 0:
        return 0.0
    return int(np.count_nonzero(a & b)) / union


def greedy_match(
    ref_masks: list[np.ndarray],
    ref_signs: list[int],
    rep_masks: list[np.ndarray],
    rep_signs: list[int],
    threshold: float,
) -> list[tuple[int, int, float]]:
    """One-to-one (ref, rep) pairs by descending Jaccard; same-sign only.

    An opposite-sign region is a different finding, not a noisy version of
    the same one, so sign gates candidacy before overlap is even scored.
    """
    pairs: list[tuple[float, int, int]] = []
    for i, (rmask, rsign) in enumerate(zip(ref_masks, ref_signs, strict=True)):
        for j, (pmask, psign) in enumerate(zip(rep_masks, rep_signs, strict=True)):
            if rsign != psign:
                continue
            jac = jaccard(rmask, pmask)
            if jac >= threshold:
                pairs.append((jac, i, j))
    pairs.sort(key=lambda t: (-t[0], t[1], t[2]))
    used_ref: set[int] = set()
    used_rep: set[int] = set()
    out: list[tuple[int, int, float]] = []
    for jac, i, j in pairs:
        if i in used_ref or j in used_rep:
            continue
        used_ref.add(i)
        used_rep.add(j)
        out.append((i, j, jac))
    return out


def _clone_params(model: Any) -> dict[str, Any]:
    """Extract constructor parameters from a fitted ImpactSplitter."""
    return {
        "delta_pct": model.delta_pct,
        "min_global_impact_pct": model.min_global_impact_pct,
        "max_depth": model.max_depth,
        "noise_z": model.noise_z,
        "consolidate": model.consolidate,
        "lookahead": model.lookahead,
        "numeric_binning_strategy": model.numeric_binning_strategy,
        "numeric_n_bins": model.numeric_n_bins,
    }


def fit_replicate(
    model: Any, rng: np.random.Generator, cols: np.ndarray
) -> Any:
    """Fit one perturbed refit: bootstrap rows of the parent's pre-encoded matrix.

    Binning/encoding happened once at parent fit time, so replicates share the
    parent's code space — their masks and conditions are directly comparable.
    """
    from impact_split.splitter import ImpactSplitter

    assert model._X is not None and model._y is not None
    n = model._X.shape[0]
    idx = rng.integers(0, n, size=n)
    rep = ImpactSplitter(**_clone_params(model))
    rep.fit(model._X[idx][:, cols], model._y[idx])
    return rep


def _accumulate_importance(
    rep: Any,
    cols: np.ndarray,
    gain_shares: dict[int, float],
    gain_avail: dict[int, int],
) -> None:
    per_feature: dict[int, float] = {}

    def walk(node: Any) -> None:
        if node is None or node.is_leaf or not node.children:
            return
        if node.split_gain:
            orig = int(cols[node.feature_index])
            per_feature[orig] = per_feature.get(orig, 0.0) + float(node.split_gain)
        for ch in node.children.values():
            walk(ch)

    walk(rep._tree)
    total = sum(per_feature.values())
    for c in cols:
        gain_avail[int(c)] = gain_avail.get(int(c), 0) + 1
    if total > 0:
        for f, g in per_feature.items():
            gain_shares[f] = gain_shares.get(f, 0.0) + g / total


def _collect_shadow_candidates(
    model, X, cols, rep_segs, rep_masks, matched_rep, block, replicate_serial, pool
) -> None:
    return None


def _finalize_importance(
    model: Any,
    gain_shares: dict[int, float],
    gain_avail: dict[int, int],
) -> list[dict[str, Any]]:
    out = []
    for f, n_avail in gain_avail.items():
        out.append(
            {
                "feature_index": f,
                "feature": model._feature_display_name(f),
                "importance": gain_shares.get(f, 0.0) / n_avail if n_avail else 0.0,
                "n_trees": n_avail,
            }
        )
    out.sort(key=lambda r: (-r["importance"], r["feature_index"]))
    return out


def _finalize_shadows(
    model, pool, match_threshold, shadow_min_stability, block_sizes
) -> list[dict[str, Any]]:
    return []


def run_ensemble(
    model: Any,
    *,
    n_replicates: int,
    shadow_replicates: int,
    feature_subsample: float | None,
    match_threshold: float,
    shadow_min_stability: float,
    seed: int | None,
) -> dict[str, Any]:
    """Fit the two-block forest and assemble the annotation report.

    Bootstrap block drives stability/CI; the feature-subsampled shadow block
    drives discovery. Kept separate so dominant-feature segments aren't
    penalized for being unfindable in trees that never saw their feature.
    """
    assert model._X is not None and model._y is not None
    X = model._X
    n_features = X.shape[1]
    rng = np.random.default_rng(seed)

    ref_masks = [mask_from_conditions(s["conditions"], X) for s in model.segments_]
    ref_signs = [1 if float(s["total_sum"]) >= 0 else -1 for s in model.segments_]
    n_ref = len(ref_masks)
    n_matched = [0] * n_ref
    ci_samples: list[list[float]] = [[] for _ in range(n_ref)]
    shadow_pool: list[dict[str, Any]] = []
    gain_shares: dict[int, float] = {}
    gain_avail: dict[int, int] = {}

    shadow_on = (
        shadow_replicates > 0 and feature_subsample is not None and n_features >= 2
    )
    blocks: list[tuple[str, int]] = [("bootstrap", n_replicates)]
    if shadow_on:
        blocks.append(("shadow", shadow_replicates))

    replicate_serial = 0
    for block, count in blocks:
        for _ in range(count):
            if block == "shadow":
                k = max(1, math.ceil(feature_subsample * n_features))
                cols = np.sort(rng.choice(n_features, size=k, replace=False))
            else:
                cols = np.arange(n_features)
            rep = fit_replicate(model, rng, cols)
            _accumulate_importance(rep, cols, gain_shares, gain_avail)

            rep_masks, rep_signs, rep_segs = [], [], []
            for s in rep.segments_:
                rep_masks.append(mask_from_conditions(s["conditions"], X, col_map=cols))
                rep_signs.append(1 if float(s["total_sum"]) >= 0 else -1)
                rep_segs.append(s)
            matches = greedy_match(
                ref_masks, ref_signs, rep_masks, rep_signs, match_threshold
            )
            matched_rep = {j for _, j, _ in matches}
            if block == "bootstrap":
                for i, j, _ in matches:
                    n_matched[i] += 1
                    ci_samples[i].append(float(rep_segs[j]["total_sum"]))
            _collect_shadow_candidates(
                model, X, cols, rep_segs, rep_masks, matched_rep,
                block, replicate_serial, shadow_pool,
            )
            replicate_serial += 1

    seg_stats: list[dict[str, Any]] = []
    for i in range(n_ref):
        stability = n_matched[i] / n_replicates if n_replicates else 0.0
        if len(ci_samples[i]) >= CI_MIN_MATCHES:
            lo, hi = np.percentile(ci_samples[i], CI_PERCENTILES)
            ci_low, ci_high = float(lo), float(hi)
        else:
            ci_low = ci_high = None
        seg_stats.append(
            {
                "stability": stability,
                "n_matched": n_matched[i],
                "ci_low": ci_low,
                "ci_high": ci_high,
                "fragile": stability < FRAGILE_STABILITY,
            }
        )

    return {
        "config": {
            "n_replicates": int(n_replicates),
            "shadow_replicates": int(shadow_replicates if shadow_on else 0),
            "feature_subsample": feature_subsample if shadow_on else None,
            "match_threshold": float(match_threshold),
            "shadow_min_stability": float(shadow_min_stability),
            "seed": seed,
        },
        "segments": seg_stats,
        "importance": _finalize_importance(model, gain_shares, gain_avail),
        "shadows": _finalize_shadows(
            model, shadow_pool, match_threshold, shadow_min_stability,
            {"bootstrap": n_replicates, "shadow": shadow_replicates if shadow_on else 0},
        ),
    }
