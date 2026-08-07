"""Tests for post-fit segment consolidation (floor loop cycle 1).

Setup shared by most tests: rule A is ``f0=1 -> +5``; rule B is
``f1=1 & f0=0 -> +10``. The tree splits f1 at the root (rule B dominates), which
shatters rule A's mass across both f1 branches into two leaves with equal means.
Consolidation must reassemble them; the f0=0 leaves (means ~10 and ~0) must not
merge.
"""

from __future__ import annotations

import numpy as np
import pytest

from impact_split.splitter import ImpactSplitter


def _shattered_rule_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(3)
    n = 1200
    f0 = (rng.random(n) < 0.3).astype(np.int64)
    f1 = (rng.random(n) < 0.5).astype(np.int64)
    y = 5.0 * (f0 == 1) + 10.0 * ((f0 == 0) & (f1 == 1)) + rng.normal(0, 0.5, n)
    return np.column_stack([f0, f1]), y, f0


def test_consolidation_merges_shattered_rule_fragments() -> None:
    X, y, f0 = _shattered_rule_data()
    model = ImpactSplitter().fit(X, y)  # consolidate defaults to True
    segs = model.get_impact_segments()
    # 4 leaves -> 3 segments: the two f0=1 fragments merge across f1.
    assert len(segs) == 3
    merged = segs[segs["n_samples"] == int((f0 == 1).sum())]
    assert len(merged) == 1
    # The merged segment is exactly the planted rule's row set.
    assert np.isclose(float(merged["total_sum"].iloc[0]), float(y[f0 == 1].sum()))
    # Its path is the rule's conjunction; the vacuous f1 condition is dropped.
    assert "f0=1" in str(merged["path"].iloc[0])
    assert "f1" not in str(merged["path"].iloc[0])


def test_consolidate_false_preserves_leaf_segments() -> None:
    X, y, _ = _shattered_rule_data()
    model = ImpactSplitter(consolidate=False).fit(X, y)
    segs = model.get_impact_segments()
    assert len(segs) == 4
    assert all("root" in p for p in segs["path"])


def test_incompatible_means_are_not_merged() -> None:
    # Fully additive effects: all four leaves have distinct means; nothing merges.
    rng = np.random.default_rng(3)
    n = 1200
    f0 = (rng.random(n) < 0.3).astype(np.int64)
    f1 = (rng.random(n) < 0.5).astype(np.int64)
    y = 5.0 * (f0 == 1) + 10.0 * (f1 == 1) + rng.normal(0, 0.5, n)
    X = np.column_stack([f0, f1])
    model = ImpactSplitter().fit(X, y)
    segs = model.get_impact_segments()
    assert len(segs) == 4


def test_consolidation_preserves_conservation_and_partition() -> None:
    X, y, _ = _shattered_rule_data()
    model = ImpactSplitter().fit(X, y)
    segs = model.get_impact_segments()
    assert np.isclose(float(segs["total_sum"].sum()), float(y.sum()))
    assert int(segs["n_samples"].sum()) == len(y)
    # Segment masks over the training encoding must tile the rows exactly.
    from benchmarks.scoring import leaf_masks_from_model

    masks = leaf_masks_from_model(model, X)
    assert len(masks) == 3
    total = np.zeros(len(y), dtype=int)
    for _, m in masks:
        total += m.astype(int)
    assert (total == 1).all()


def test_consolidation_is_null_safe() -> None:
    # Pure noise: the tree abstains (single root leaf); consolidation is a no-op.
    rng = np.random.default_rng(0)
    X = rng.integers(0, 4, size=(2000, 3)).astype(np.int64)
    y = rng.normal(0, 1, 2000)
    model = ImpactSplitter().fit(X, y)
    segs = model.get_impact_segments()
    assert len(segs) == 1


def test_consolidate_constructor_validation() -> None:
    with pytest.raises(TypeError):
        ImpactSplitter(consolidate="yes")  # type: ignore[arg-type]
