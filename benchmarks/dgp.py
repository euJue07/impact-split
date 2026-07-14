"""Synthetic data-generating processes for the robustness battery.

Each case factory returns a :class:`BenchDataset` with per-row effect
contributions per planted rule, so scoring can attribute impact exactly
even when rule masks overlap.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class Rule:
    """A planted effect: label + per-row contribution to y_expected."""

    label: str
    mask: np.ndarray  # bool, rows the rule touches
    contrib: np.ndarray  # float, per-row contribution (mask * increment)
    increment: float


@dataclass
class BenchDataset:
    """One benchmark dataset: covariates, observed target, planted ground truth."""

    case: str
    seed: int
    X: pd.DataFrame
    y: np.ndarray  # observed = sum(contribs) + noise
    rules: list[Rule]
    noise_sigma: float
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def y_expected(self) -> np.ndarray:
        out = np.zeros(len(self.y))
        for r in self.rules:
            out += r.contrib
        return out


def _mk_rules(X: pd.DataFrame, specs: list[tuple[str, pd.Series, float]]) -> list[Rule]:
    rules = []
    for label, mask, inc in specs:
        m = mask.to_numpy()
        rules.append(Rule(label, m, np.where(m, inc, 0.0), inc))
    return rules


def _assemble(
    case: str,
    seed: int,
    X: pd.DataFrame,
    rules: list[Rule],
    sigma: float,
    rng: np.random.Generator,
    **meta: Any,
) -> BenchDataset:
    y_expected = np.zeros(len(X))
    for r in rules:
        y_expected += r.contrib
    y = y_expected + rng.normal(0, sigma, len(X))
    return BenchDataset(case, seed, X, y, rules, sigma, dict(meta))


def _baseline_covariates(rng: np.random.Generator, n: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "region": rng.choice(
                ["NCR", "Luzon", "Visayas", "Mindanao"], size=n, p=[0.35, 0.3, 0.2, 0.15]
            ),
            "channel": rng.choice(["Direct", "Partner", "Online"], size=n, p=[0.25, 0.35, 0.4]),
            "product": rng.choice(["A", "B", "C"], size=n, p=[0.4, 0.35, 0.25]),
        }
    )


def _baseline_specs(X: pd.DataFrame) -> list[tuple[str, pd.Series, float]]:
    return [
        ("NCR x Direct", (X["region"] == "NCR") & (X["channel"] == "Direct"), 120.0),
        (
            "Mindanao x Partner x {A,B}",
            (X["region"] == "Mindanao") & (X["channel"] == "Partner") & X["product"].isin(["A", "B"]),
            -95.0,
        ),
        (
            "Luzon x Online x C",
            (X["region"] == "Luzon") & (X["channel"] == "Online") & (X["product"] == "C"),
            35.0,
        ),
        ("Luzon x Partner", (X["region"] == "Luzon") & (X["channel"] == "Partner"), 60.0),
        ("Visayas x Online", (X["region"] == "Visayas") & (X["channel"] == "Online"), -45.0),
        (
            "Luzon x Online x A",
            (X["region"] == "Luzon") & (X["channel"] == "Online") & (X["product"] == "A"),
            50.0,
        ),
    ]


def case_baseline(seed: int, *, sigma: float = 22.0, n: int = 5000) -> BenchDataset:
    """Case 1 — the explainer DGP (regression guard; seed 42 = published notebook)."""
    rng = np.random.default_rng(seed)
    X = _baseline_covariates(rng, n)
    rules = _mk_rules(X, _baseline_specs(X))
    return _assemble("baseline", seed, X, rules, sigma, rng)


def case_one_sided(seed: int, *, n: int = 5000) -> BenchDataset:
    """Case 2 — all effects the same sign (stresses the centered-excess fallback)."""
    rng = np.random.default_rng(seed)
    X = _baseline_covariates(rng, n)
    specs = [
        ("NCR x Direct", (X["region"] == "NCR") & (X["channel"] == "Direct"), 120.0),
        (
            "Mindanao x Partner x {A,B}",
            (X["region"] == "Mindanao") & (X["channel"] == "Partner") & X["product"].isin(["A", "B"]),
            95.0,
        ),
        ("Luzon x Partner", (X["region"] == "Luzon") & (X["channel"] == "Partner"), 60.0),
        ("Visayas x Online", (X["region"] == "Visayas") & (X["channel"] == "Online"), 45.0),
        (
            "Luzon x Online x C",
            (X["region"] == "Luzon") & (X["channel"] == "Online") & (X["product"] == "C"),
            35.0,
        ),
    ]
    return _assemble("one_sided", seed, X, _mk_rules(X, specs), 22.0, rng)


def case_high_cardinality(seed: int, *, n: int = 5000, nuisance_levels: int = 50) -> BenchDataset:
    """Case 3 — baseline rules + a 50-level nuisance feature with no true effect."""
    rng = np.random.default_rng(seed)
    X = _baseline_covariates(rng, n)
    X["store_id"] = rng.choice([f"S{i:02d}" for i in range(nuisance_levels)], size=n)
    rules = _mk_rules(X, _baseline_specs(X))
    return _assemble("high_cardinality", seed, X, rules, 22.0, rng, nuisance_levels=nuisance_levels)


def case_deep_interactions(seed: int, *, n: int = 8000) -> BenchDataset:
    """Case 4 — 3- and 4-way planted rules over five features."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        {
            "region": rng.choice(
                ["NCR", "Luzon", "Visayas", "Mindanao"], size=n, p=[0.35, 0.3, 0.2, 0.15]
            ),
            "channel": rng.choice(["Direct", "Partner", "Online"], size=n, p=[0.25, 0.35, 0.4]),
            "product": rng.choice(["A", "B", "C"], size=n, p=[0.4, 0.35, 0.25]),
            "tier": rng.choice(["Gold", "Silver", "Bronze"], size=n, p=[0.2, 0.35, 0.45]),
            "acct": rng.choice(["SME", "Corp", "Gov", "Retail"], size=n, p=[0.3, 0.25, 0.15, 0.3]),
        }
    )
    specs = [
        (
            "NCR x Direct x Gold",
            (X["region"] == "NCR") & (X["channel"] == "Direct") & (X["tier"] == "Gold"),
            150.0,
        ),
        (
            "Luzon x Partner x A x Corp",
            (X["region"] == "Luzon")
            & (X["channel"] == "Partner")
            & (X["product"] == "A")
            & (X["acct"] == "Corp"),
            -200.0,
        ),
        (
            "Visayas x Online x Silver",
            (X["region"] == "Visayas") & (X["channel"] == "Online") & (X["tier"] == "Silver"),
            90.0,
        ),
        (
            "Online x C x Gov x Bronze",
            (X["channel"] == "Online")
            & (X["product"] == "C")
            & (X["acct"] == "Gov")
            & (X["tier"] == "Bronze"),
            -180.0,
        ),
    ]
    return _assemble("deep_interactions", seed, X, _mk_rules(X, specs), 22.0, rng)


def case_noise(seed: int, *, sigma: float = 44.0, n: int = 5000) -> BenchDataset:
    """Case 5 — baseline rules under 2x noise (scored); frontier sweep is diagnostic."""
    rng = np.random.default_rng(seed)
    X = _baseline_covariates(rng, n)
    rules = _mk_rules(X, _baseline_specs(X))
    return _assemble("noise_2x", seed, X, rules, sigma, rng)


def case_skewed_volume(seed: int, *, n: int = 8000) -> BenchDataset:
    """Case 6 — one region holds 80% of rows; rules in both the giant and a tiny cell."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        {
            "region": rng.choice(
                ["NCR", "Luzon", "Visayas", "Mindanao"], size=n, p=[0.80, 0.10, 0.07, 0.03]
            ),
            "channel": rng.choice(["Direct", "Partner", "Online"], size=n, p=[0.25, 0.35, 0.4]),
            "product": rng.choice(["A", "B", "C"], size=n, p=[0.4, 0.35, 0.25]),
        }
    )
    specs = [
        ("NCR x Direct", (X["region"] == "NCR") & (X["channel"] == "Direct"), 40.0),
        (
            "Mindanao x Partner",
            (X["region"] == "Mindanao") & (X["channel"] == "Partner"),
            -150.0,
        ),
        (
            "NCR x Online x C",
            (X["region"] == "NCR") & (X["channel"] == "Online") & (X["product"] == "C"),
            -35.0,
        ),
    ]
    return _assemble("skewed_volume", seed, X, _mk_rules(X, specs), 22.0, rng)


def case_overlapping(seed: int, *, n: int = 5000) -> BenchDataset:
    """Case 7 — non-disjoint rule masks (real data won't be pairwise disjoint)."""
    rng = np.random.default_rng(seed)
    X = _baseline_covariates(rng, n)
    specs = [
        ("NCR x Direct", (X["region"] == "NCR") & (X["channel"] == "Direct"), 120.0),
        ("NCR x A", (X["region"] == "NCR") & (X["product"] == "A"), -80.0),
        ("Luzon x Partner", (X["region"] == "Luzon") & (X["channel"] == "Partner"), 60.0),
        ("Partner x B", (X["channel"] == "Partner") & (X["product"] == "B"), -50.0),
    ]
    return _assemble("overlapping", seed, X, _mk_rules(X, specs), 22.0, rng)


def case_null(seed: int, *, n: int = 5000) -> BenchDataset:
    """Case 8 — pure noise; the tree must report nothing material (FP control)."""
    rng = np.random.default_rng(seed)
    X = _baseline_covariates(rng, n)
    return _assemble("null", seed, X, [], 22.0, rng)


CASE_FACTORIES: dict[str, Callable[[int], BenchDataset]] = {
    "baseline": case_baseline,
    "one_sided": case_one_sided,
    "high_cardinality": case_high_cardinality,
    "deep_interactions": case_deep_interactions,
    "noise_2x": case_noise,
    "skewed_volume": case_skewed_volume,
    "overlapping": case_overlapping,
    "null": case_null,
}

SEEDS = [42, 7, 2026]

NOISE_FRONTIER_SIGMAS = [22.0, 44.0, 88.0, 176.0, 352.0]
