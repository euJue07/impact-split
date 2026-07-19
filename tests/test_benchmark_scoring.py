"""Tests for the benchmark scoring metric (capped + uncapped-union diagnostic)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from benchmarks.dgp import BenchDataset, Rule
from benchmarks.scoring import score_dataset


def _dataset_with_fragmented_rule(n_fragments: int) -> tuple[BenchDataset, list]:
    """One planted rule whose mass is tiled across ``n_fragments`` clean leaves."""
    n = 100 * n_fragments
    rng = np.random.default_rng(0)
    mask = np.zeros(n, dtype=bool)
    mask[: 50 * n_fragments] = True  # rule covers the first half of every fragment
    contrib = np.where(mask, 4.0, 0.0)
    y = contrib + rng.normal(0, 0.01, n)
    X = pd.DataFrame({"f": ["a"] * n})
    rule = Rule("planted", mask, contrib, 4.0)
    ds = BenchDataset("frag", 0, X, y, [rule], 0.01, {})
    # Leaves: each fragment split into its rule half and its empty half.
    leaves = []
    for i in range(n_fragments):
        lo = 100 * i
        m_rule = np.zeros(n, dtype=bool)
        m_rule[lo : lo + 50] = True
        m_rest = np.zeros(n, dtype=bool)
        m_rest[lo + 50 : lo + 100] = True
        leaves.append((f"leaf_rule_{i}", m_rule))
        leaves.append((f"leaf_rest_{i}", m_rest))
    # Rebuild mask/contrib so the rule half is scattered, not contiguous.
    rule_mask = np.zeros(n, dtype=bool)
    for i in range(n_fragments):
        rule_mask[100 * i : 100 * i + 50] = True
    contrib = np.where(rule_mask, 4.0, 0.0)
    y = contrib + rng.normal(0, 0.01, n)
    ds = BenchDataset("frag", 0, X, y, [Rule("planted", rule_mask, contrib, 4.0)], 0.01, {})
    return ds, leaves


def test_uncapped_recovers_fragmentation_beyond_cap() -> None:
    ds, leaves = _dataset_with_fragmented_rule(n_fragments=8)
    score = score_dataset(ds, leaves)
    rs = score.rule_scores[0]
    # Capped union stops at 3 segments: recall ~3/8 of the mass.
    assert rs.n_segments_used <= 3
    assert rs.f1 < 0.7
    # Uncapped union collects all 8 fragments and recovers the rule.
    assert rs.uncapped_n_segments == 8
    assert rs.uncapped_f1 > 0.99
    assert rs.uncapped_f1 >= rs.f1


def test_uncapped_equals_capped_when_rule_fits_in_cap() -> None:
    ds, leaves = _dataset_with_fragmented_rule(n_fragments=2)
    score = score_dataset(ds, leaves)
    rs = score.rule_scores[0]
    assert rs.n_segments_used == 2
    assert rs.uncapped_n_segments == 2
    assert np.isclose(rs.uncapped_f1, rs.f1)


def test_ensemble_filter_scores_one_case():
    from benchmarks.dgp import case_baseline
    from benchmarks.ensemble_filter import score_case_at_taus

    ds = case_baseline(0)
    rows = score_case_at_taus(ds, n_replicates=5, seed=0)
    assert set(rows) == {0.0, 0.3, 0.5, 0.7}
    assert all(0.0 <= v <= 1.0 for v in rows.values())
