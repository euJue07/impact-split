"""Tests for the pairwise lookahead rescue (v0.2.0)."""

from __future__ import annotations

import numpy as np
import pytest

from impact_split import ImpactSplitter


def test_lookahead_constructor_validation() -> None:
    with pytest.raises(TypeError, match="lookahead"):
        ImpactSplitter(lookahead="yes")  # type: ignore[arg-type]


def test_lookahead_default_true_and_in_repr() -> None:
    model = ImpactSplitter()
    assert model.lookahead is True
    assert "lookahead=True" in repr(model)


def _xor_arrays(n: int = 2000, seed: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Pure 2-feature XOR: every marginal category nets ~0; only the cross sees signal."""
    rng = np.random.default_rng(seed)
    f0 = rng.integers(0, 2, size=n).astype(np.int64)
    f1 = rng.integers(0, 2, size=n).astype(np.int64)
    y = np.where((f0 ^ f1) == 1, 100.0, -100.0) + rng.normal(0, 1.0, n)
    return np.column_stack([f0, f1]), y


def test_rescue_fires_at_root_on_pure_xor() -> None:
    X, y = _xor_arrays()
    model = ImpactSplitter().fit(X, y, trace=True)
    root = model.fit_trace_[0]
    assert root["routing_mode"] == "lookahead_rescue"
    assert root["rescue"]["pair"] == [0, 1]
    payload = model.to_dict()
    assert payload["meta"]["n_segments"] == 4
    means = sorted(s["mean"] for s in payload["segments"])
    assert means[0] == pytest.approx(-100.0, abs=2.0)
    assert means[1] == pytest.approx(-100.0, abs=2.0)
    assert means[2] == pytest.approx(100.0, abs=2.0)
    assert means[3] == pytest.approx(100.0, abs=2.0)
    assert payload["meta"]["conservation_exact"] is True


def test_lookahead_false_reproduces_v010_silent_miss() -> None:
    X, y = _xor_arrays()
    model = ImpactSplitter(lookahead=False).fit(X, y, trace=True)
    root = model.fit_trace_[0]
    assert root["action"] == "leaf"
    assert root["stop_reason"] == "no_split"
    assert len(model.segments_) == 1


def test_rescue_respects_interaction_cap() -> None:
    # f0 is a legit marginal driver; the XOR pair sits one level down. With
    # max_depth=1 the children are at the cap, so the rescue must NOT fire
    # (a rescued split would add a forbidden interaction term).
    rng = np.random.default_rng(1)
    n = 4000
    f0 = rng.integers(0, 2, size=n).astype(np.int64)
    f1 = rng.integers(0, 2, size=n).astype(np.int64)
    f2 = rng.integers(0, 2, size=n).astype(np.int64)
    y = 50.0 * f0 + np.where((f1 ^ f2) == 1, 100.0, -100.0) + rng.normal(0, 1.0, n)
    X = np.column_stack([f0, f1, f2])
    model = ImpactSplitter(max_depth=1).fit(X, y, trace=True)
    assert not any(t.get("routing_mode") == "lookahead_rescue" for t in model.fit_trace_)


def test_rescue_leaves_irreducible_churn_alone() -> None:
    # Offsetting ±y independent of every feature: marginals AND crosses all
    # net ~0, so the rescue finds nothing and the node leafs out unchanged.
    rng = np.random.default_rng(2)
    n = 4000
    X = rng.integers(0, 2, size=(n, 2)).astype(np.int64)
    y = np.where(rng.random(n) < 0.5, 100.0, -99.0) + rng.normal(0, 1.0, n)
    model = ImpactSplitter().fit(X, y, trace=True)
    root = model.fit_trace_[0]
    assert root["routing_mode"] is None
    assert root["stop_reason"] == "no_split"
    assert len(model.segments_) == 1


def test_rescue_skips_pairs_over_cardinality_bound() -> None:
    # 150x150 categories -> crossed bincount would allocate 22,500 cells
    # (> _LOOKAHEAD_MAX_CROSS), so the only pair is skipped and the XOR is
    # (acceptably) missed: clean no_split leaf, no crash, no memory blowup.
    rng = np.random.default_rng(3)
    n = 6000
    f0 = rng.integers(0, 150, size=n).astype(np.int64)
    f1 = rng.integers(0, 150, size=n).astype(np.int64)
    y = np.where(((f0 % 2) ^ (f1 % 2)) == 1, 100.0, -100.0) + rng.normal(0, 1.0, n)
    X = np.column_stack([f0, f1])
    model = ImpactSplitter().fit(X, y, trace=True)
    root = model.fit_trace_[0]
    assert root["stop_reason"] == "no_split"
    assert len(model.segments_) == 1


def test_rescue_multiplicity_floor_suppresses_chance_cells() -> None:
    # High-cardinality cross (10x10 = 100 present cells) with NO real
    # cross-feature structure: y is pure standard-normal noise, plus a
    # planted bump of +1.0 on the ~18-20 rows of exactly one (f, g) cell.
    # That single cell's excess clears the per-cell 3-sigma floor by chance
    # multiplicity (K~100 simultaneous comparisons), which is exactly the
    # Kaggle Genre x Publisher regression mechanism (task-8-report.md). The
    # multiplicity-corrected z_eff = noise_z + sqrt(2*ln(K)) must not be
    # fooled: the node must leaf out at the root with no rescue firing.
    rng = np.random.default_rng(2)
    n = 2000
    ncat = 10
    f0 = rng.integers(0, ncat, size=n).astype(np.int64)
    f1 = rng.integers(0, ncat, size=n).astype(np.int64)
    y = rng.normal(0, 1.0, n)
    bump_mask = (f0 == 0) & (f1 == 0)
    y = y.copy()
    y[bump_mask] += 1.0
    X = np.column_stack([f0, f1])
    model = ImpactSplitter().fit(X, y, trace=True)
    root = model.fit_trace_[0]
    assert root["routing_mode"] is None
    assert root["stop_reason"] == "no_split"
    assert len(model.segments_) == 1


def test_rescue_ignores_singleton_cells() -> None:
    # Small-n mechanism (kaggle_ibm_hr seed 7, node_40 in task-8-report.md):
    # a tiny material node whose cross-cells are all singletons. One row per
    # (f0, f1) cell, huge |y| so both materiality triggers fire, marginals
    # net exactly 0 (XOR layout) -> the rescue is consulted. Every cell is a
    # single observation with no within-cell noise estimate, so none may
    # count as sieve-clearing evidence: the node must leaf out. (Verified to
    # falsely fire under the multiplicity-corrected floor alone, because at
    # K=4 present cells z_eff is barely above noise_z and the MAD sigma of
    # singleton cells is 0.)
    rng = np.random.default_rng(0)
    f0 = np.array([0, 0, 1, 1], dtype=np.int64)
    f1 = np.array([0, 1, 0, 1], dtype=np.int64)
    y = np.where((f0 ^ f1) == 1, 100.0, -100.0) + rng.normal(0, 0.5, 4)
    X = np.column_stack([f0, f1])
    model = ImpactSplitter().fit(X, y, trace=True)
    root = model.fit_trace_[0]
    assert root["routing_mode"] is None
    assert root["stop_reason"] == "no_split"
    assert len(model.segments_) == 1


def test_lookahead_partition_xor_correctness_and_degenerate_guards() -> None:
    # White-box: the profile partition on a crafted 2x2 XOR cross-table, plus
    # the two degenerate exits (same-sign rows; fewer than 2 signal rows).
    model = ImpactSplitter()
    col = np.array([0, 0, 1, 1], dtype=np.int64)
    xor_profile = np.array([[10.0, -10.0], [-10.0, 10.0]])
    d = model._lookahead_partition(0, col, xor_profile, np.array([0, 1]), 1.0, 4)
    assert d is not None
    assert d.mode == "lookahead_rescue"
    assert d.pos_categories.tolist() == [0]
    assert d.neg_categories.tolist() == [1]
    assert d.neu_categories.tolist() == []
    # Same-sign rows: no opposing group -> degenerate -> None.
    one_sided = np.array([[10.0, 0.0], [8.0, 0.0]])
    assert model._lookahead_partition(0, col, one_sided, np.array([0, 1]), 1.0, 4) is None
    # Only one row carries a sieve-clearing cell -> cannot partition -> None.
    assert model._lookahead_partition(0, col, xor_profile, np.array([0]), 1.0, 4) is None
