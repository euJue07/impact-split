"""Benchmark bars for the v0.2.0 lookahead/churn cases.

These thresholds are the spec's hard bar. If one fails, the design has a real
problem — investigate; never lower a threshold to pass.
"""

from __future__ import annotations

from benchmarks.battery import fit_and_score
from benchmarks.dgp import SEEDS, case_churn, case_xor, case_xor_embedded

from impact_split import ImpactSplitter


def test_xor_pure_recovered_across_seeds() -> None:
    for seed in SEEDS:
        score = fit_and_score(case_xor(seed))
        assert score.conservation_ok
        assert score.impact_f1 >= 0.90, f"seed {seed}: {score.impact_f1:.4f}"


def test_xor_pure_missed_without_lookahead() -> None:
    # Documents the v0.1.0 silent failure the rescue exists to fix. Not one of
    # the spec's named hard bars. Threshold recalibrated from the brief's <0.5:
    # a "pure" (fully-covered, symmetric two-outcome) XOR case is mathematically
    # pinned to F1 = harmonic_mean(recall=1, precision=0.5) = 0.667 when no split
    # occurs (the single leaf's captured mass is always exactly half its total
    # mass), regardless of amp/n — no DGP tuning reaches <0.5 here. 0.70 still
    # documents the rescue's effect: well below the >=0.90 bar it clears with
    # lookahead on.
    score = fit_and_score(case_xor(42), {"lookahead": False})
    assert score.impact_f1 < 0.70


def test_xor_embedded_recovered_across_seeds() -> None:
    for seed in SEEDS:
        score = fit_and_score(case_xor_embedded(seed))
        assert score.conservation_ok
        assert score.impact_f1 >= 0.80, f"seed {seed}: {score.impact_f1:.4f}"


def test_churn_case_not_split_and_flagged() -> None:
    for seed in SEEDS:
        ds = case_churn(seed)
        model = ImpactSplitter().fit(ds.X, ds.y)
        payload = model.to_dict()
        assert payload["meta"]["n_segments"] == 1
        assert payload["segments"][0]["is_churn"] is True
        assert payload["meta"]["n_churn_segments"] == 1
