"""Phase-0 floor-case diagnostics for the floor loop (KB #101).

The five sub-0.85 dataset-seed cases from cycle 3, re-run individually:
reproduction check against the committed cycle-3 scores, per-rule autopsy with
the uncapped-union diagnostic column, and a hyperparameter sweep restricted to
these cases (any winning config must still be re-scored on the full suite
before it is believed — this module never does that promotion itself).

Usage:
    python -m benchmarks.floor_cases                 # repro check + autopsy
    python -m benchmarks.floor_cases --sweep         # coarse HPO grid
"""

from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from .battery import RESULTS_DIR, fit_and_score
from .kaggle_suite import build_semi_synth

# (case name, KAGGLE_SPECS index, seed) — cycle-3 impact-F1 in comments.
FLOOR_CASES: list[tuple[str, int, int]] = [
    ("kaggle_black_friday", 7, 2026),  # 0.781
    ("kaggle_insurance", 1, 42),  # 0.794
    ("kaggle_ibm_hr", 6, 7),  # 0.810
    ("kaggle_olist", 8, 42),  # 0.815
    ("kaggle_telco_churn", 5, 2026),  # 0.837
]

CYCLE3_REFERENCE = Path(__file__).parent / "results" / "cycle3-kaggle.json"


def load_reference() -> dict[tuple[str, int], float]:
    ref = json.loads(CYCLE3_REFERENCE.read_text(encoding="utf-8"))
    return {(r["case"], r["seed"]): r["impact_f1"] for r in ref["results"]}


def run_floor_cases(params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    rows = []
    for case, idx, seed in FLOOR_CASES:
        ds = build_semi_synth(idx, seed)
        score = fit_and_score(ds, params)
        row = asdict(score)
        row["params"] = params or {}
        rows.append(row)
    return rows


def repro_and_autopsy() -> None:
    ref = load_reference()
    rows = run_floor_cases()
    payload = {"rows": rows, "repro": {}}
    print("=== Phase 0: reproduction check (defaults vs committed cycle-3) ===")
    all_ok = True
    for row in rows:
        key = (row["case"], row["seed"])
        expected = ref[key]
        got = row["impact_f1"]
        ok = bool(np.isclose(got, expected, atol=1e-9))
        all_ok &= ok
        payload["repro"][f"{key[0]}/{key[1]}"] = {"expected": expected, "got": got, "ok": ok}
        print(f"  {row['case']:22s} seed={row['seed']:<5d} got={got:.4f} "
              f"expected={expected:.4f} {'OK' if ok else 'MISMATCH'}")
    print(f"repro: {'DETERMINISTIC' if all_ok else 'MISMATCH — investigate before anything else'}")

    print("\n=== Per-rule autopsy (capped vs uncapped union) ===")
    for row in rows:
        print(f"\n-- {row['case']} seed {row['seed']} "
              f"(F1={row['impact_f1']:.3f}, {row['n_terminal_segments']} segments)")
        for rs in row["rule_scores"]:
            gap = rs["uncapped_f1"] - rs["f1"]
            print(f"   {rs['rule'][:58]:58s} capped={rs['f1']:.3f} "
                  f"(r={rs['recall']:.2f}/p={rs['precision']:.2f}, {rs['n_segments_used']} segs) "
                  f"uncapped={rs['uncapped_f1']:.3f} ({rs['uncapped_n_segments']} segs) "
                  f"gap={gap:+.3f}")
    out = RESULTS_DIR / "floor-repro-autopsy.json"
    out.write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
    print(f"\nsaved -> {out}")


def sweep(grid: dict[str, list[Any]] | None = None) -> None:
    grid = grid or {
        "delta_pct": [0.005, 0.01, 0.02],
        "noise_z": [2.0, 3.0, 4.0],
        "max_depth": [5, 6, 7],
    }
    keys = list(grid)
    combos = list(itertools.product(*(grid[k] for k in keys)))
    print(f"sweep: {len(combos)} configs x {len(FLOOR_CASES)} cases")
    results = []
    for combo in combos:
        params = dict(zip(keys, combo))
        rows = run_floor_cases(params)
        f1s = {r["case"]: r["impact_f1"] for r in rows}
        segs = {r["case"]: r["n_terminal_segments"] for r in rows}
        entry = {"params": params, "f1": f1s, "segments": segs,
                 "min_f1": min(f1s.values()), "mean_f1": float(np.mean(list(f1s.values())))}
        results.append(entry)
        print(f"  {params} -> min={entry['min_f1']:.3f} mean={entry['mean_f1']:.3f}")
    results.sort(key=lambda e: (-e["min_f1"], -e["mean_f1"]))
    out = RESULTS_DIR / "floor-sweep.json"
    out.write_text(json.dumps(results, indent=2, default=float), encoding="utf-8")
    print("\ntop 5 by floor-case min F1:")
    for e in results[:5]:
        print(f"  min={e['min_f1']:.3f} mean={e['mean_f1']:.3f} {e['params']}")
    print(f"saved -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", action="store_true", help="run the coarse HPO grid")
    args = ap.parse_args()
    if args.sweep:
        sweep()
    else:
        repro_and_autopsy()


if __name__ == "__main__":
    main()
