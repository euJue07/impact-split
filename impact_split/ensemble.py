"""Bootstrap/feature-subsample forest that annotates a fitted ImpactSplitter.

The single greedy tree stays THE answer; the forest measures how much of it
survives resampling (stability, CI), which features carry the forest's gain
(importance), and which material regions only appear when dominant features
are forced out (shadow segments). No prediction averaging — ever.
"""

from __future__ import annotations

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
