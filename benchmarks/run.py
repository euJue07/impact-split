"""CLI: run the synthetic battery and print/save a summary.

Usage:  python -m benchmarks.run --tag cycle0 [--cart] [--frontier]
"""

from __future__ import annotations

import argparse

from .battery import noise_frontier, run_battery, save_results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="results file tag, e.g. cycle0-synthetic")
    ap.add_argument("--cart", action="store_true", help="include CART reference scores")
    ap.add_argument("--frontier", action="store_true", help="include noise-frontier diagnostic")
    args = ap.parse_args()

    summary = run_battery(with_cart=args.cart)
    if args.frontier:
        summary["noise_frontier"] = noise_frontier()

    path = save_results(args.tag, summary)

    print(f"mean impact-F1 : {summary['mean_impact_f1']:.4f}")
    print(f"floor dataset  : {summary['floor_dataset_f1']:.4f}")
    print(f"null pass rate : {summary['null_pass_rate']}")
    print(f"conservation   : {summary['conservation_all_ok']}")
    print(f"mean #segments : {summary['mean_n_segments']:.1f}")
    print("per-case mean F1:")
    for case, f1 in summary["per_case_mean_f1"].items():
        print(f"  {case:20s} {f1:.4f}")
    if args.frontier:
        print("noise frontier (baseline rules, seed 42):")
        for row in summary["noise_frontier"]:
            print(
                f"  sigma={row['sigma']:>6.1f}  F1={row['impact_f1']:.4f}"
                f"  floor={row['floor_rule_f1']:.4f}  segs={row['n_segments']}"
            )
    print(f"saved -> {path}")


if __name__ == "__main__":
    main()
