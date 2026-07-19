"""Offline diagnostic: does stability-filtering the ledger beat the baseline?

Scores "reference segments with stability >= tau" against the synthetic
battery. Evidence-gathering for a future gated default (the lookahead-rescue
path) — nothing here changes fit defaults or the Kaggle gate.
"""

from __future__ import annotations

import numpy as np

from impact_split.splitter import ImpactSplitter

from .battery import DEFAULT_PARAMS
from .dgp import CASE_FACTORIES, SEEDS, BenchDataset
from .scoring import encode_with_model_maps, leaf_masks_from_model, score_dataset

TAUS = (0.0, 0.3, 0.5, 0.7)


def score_case_at_taus(
    ds: BenchDataset, *, n_replicates: int = 50, seed: int = 0
) -> dict[float, float]:
    model = ImpactSplitter(**DEFAULT_PARAMS)
    model.fit(ds.X, ds.y)
    model.ensemble_report(
        ds.X, ds.y, n_replicates=n_replicates, shadow_replicates=0, seed=seed
    )
    X_codes = encode_with_model_maps(model, ds.X)
    leaves = leaf_masks_from_model(model, X_codes)
    stats = model.ensemble_["segments"]  # index-aligned with leaves
    out: dict[float, float] = {}
    for tau in TAUS:
        keep = [lm for lm, st in zip(leaves, stats, strict=True) if st["stability"] >= tau]
        if not keep:  # never score an empty ledger; fall back to unfiltered
            keep = leaves
        sc = score_dataset(ds, keep)
        out[tau] = float(sc.impact_f1) if ds.rules else float("nan")
    return out


def main() -> None:
    per_tau: dict[float, list[float]] = {tau: [] for tau in TAUS}
    for case, factory in CASE_FACTORIES.items():
        if case == "null":
            continue
        for seed in SEEDS:
            rows = score_case_at_taus(factory(seed), seed=seed)
            for tau, f1 in rows.items():
                per_tau[tau].append(f1)
    print("tau   mean_impact_f1   floor")
    for tau in TAUS:
        vals = per_tau[tau]
        print(f"{tau:.1f}   {np.mean(vals):.4f}          {np.min(vals):.4f}")


if __name__ == "__main__":
    main()
