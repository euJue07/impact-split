"""Run the synthetic battery (and optional extra datasets) against ImpactSplitter."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from impact_split.splitter import ImpactSplitter

from .dgp import CASE_FACTORIES, NOISE_FRONTIER_SIGMAS, SEEDS, BenchDataset, case_baseline
from .scoring import DatasetScore, encode_with_model_maps, leaf_masks_from_model, score_dataset

RESULTS_DIR = Path(__file__).parent / "results"

DEFAULT_PARAMS: dict[str, Any] = {
    "delta_pct": 0.05,
    "min_global_impact_pct": 0.01,
    "max_depth": 5,
}


def fit_and_score(ds: BenchDataset, params: dict[str, Any] | None = None) -> DatasetScore:
    p = dict(DEFAULT_PARAMS)
    if params:
        p.update(params)
    model = ImpactSplitter(**p)
    model.fit(ds.X, ds.y)
    X_codes = encode_with_model_maps(model, ds.X)
    leaves = leaf_masks_from_model(model, X_codes)
    # Partition sanity: leaf masks must tile the data exactly.
    counts = sum(int(m.sum()) for _, m in leaves)
    if counts != len(ds.y):
        raise AssertionError(f"leaf masks do not partition rows: {counts} != {len(ds.y)}")
    return score_dataset(ds, leaves)


def cart_reference(ds: BenchDataset, max_depth: int = 5) -> DatasetScore:
    """Score a CART (MSE) tree's leaves with the same impact-F1 metric."""
    from sklearn.tree import DecisionTreeRegressor

    X_ohe = pd.get_dummies(ds.X)
    dt = DecisionTreeRegressor(max_depth=max_depth, random_state=42)
    dt.fit(X_ohe, ds.y)
    leaf_ids = dt.apply(X_ohe)
    leaves = [(f"cart_leaf_{lid}", leaf_ids == lid) for lid in np.unique(leaf_ids)]
    return score_dataset(ds, leaves)


def run_battery(
    params: dict[str, Any] | None = None,
    *,
    seeds: list[int] | None = None,
    with_cart: bool = False,
) -> dict[str, Any]:
    """Run all cases x seeds; return an aggregate result dict (JSON-serializable)."""
    seeds = seeds or SEEDS
    results: list[dict[str, Any]] = []
    for case, factory in CASE_FACTORIES.items():
        for seed in seeds:
            ds = factory(seed)
            score = fit_and_score(ds, params)
            row = asdict(score)
            if with_cart and ds.rules:
                cart = cart_reference(ds)
                row["cart_impact_f1"] = cart.impact_f1
                row["cart_n_segments"] = cart.n_terminal_segments
            results.append(row)

    scored = [r for r in results if r["case"] != "null"]
    nulls = [r for r in results if r["case"] == "null"]
    per_case: dict[str, float] = {}
    for case in CASE_FACTORIES:
        vals = [r["impact_f1"] for r in scored if r["case"] == case]
        if vals:
            per_case[case] = float(np.mean(vals))

    summary = {
        "params": {**DEFAULT_PARAMS, **(params or {})},
        "mean_impact_f1": float(np.mean([r["impact_f1"] for r in scored])),
        "floor_dataset_f1": float(np.min([r["impact_f1"] for r in scored])),
        "per_case_mean_f1": per_case,
        "null_pass_rate": float(np.mean([r["null_pass"] for r in nulls])) if nulls else None,
        "conservation_all_ok": bool(all(r["conservation_ok"] for r in results)),
        "mean_n_segments": float(np.mean([r["n_terminal_segments"] for r in scored])),
        "results": results,
    }
    return summary


def noise_frontier(params: dict[str, Any] | None = None, seed: int = 42) -> list[dict[str, Any]]:
    """Diagnostic: baseline rules under escalating noise; where does recovery break?"""
    rows = []
    for sigma in NOISE_FRONTIER_SIGMAS:
        ds = case_baseline(seed, sigma=sigma)
        score = fit_and_score(ds, params)
        rows.append(
            {
                "sigma": sigma,
                "impact_f1": score.impact_f1,
                "floor_rule_f1": score.floor_rule_f1,
                "n_segments": score.n_terminal_segments,
            }
        )
    return rows


def save_results(tag: str, payload: dict[str, Any]) -> Path:
    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / f"{tag}.json"
    out.write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
    return out
