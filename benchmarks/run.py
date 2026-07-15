"""CLI: run the synthetic battery and print/save a summary.

Usage:  python -m benchmarks.run --tag cycle0 [--cart] [--frontier]
"""

from __future__ import annotations

import argparse

from .battery import noise_frontier, run_battery, run_kaggle_suite, run_lookahead_cases, save_results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="results file tag, e.g. cycle0-synthetic")
    ap.add_argument("--cart", action="store_true", help="include CART reference scores")
    ap.add_argument("--frontier", action="store_true", help="include noise-frontier diagnostic")
    ap.add_argument("--kaggle", action="store_true", help="run the semi-synthetic Kaggle suite")
    ap.add_argument("--face-validity", action="store_true", help="record real-target groupbys")
    ap.add_argument(
        "--lookahead", action="store_true", help="run the v0.2.0 lookahead/churn cases"
    )
    args = ap.parse_args()

    if args.lookahead:
        summary = run_lookahead_cases()
        path = save_results(args.tag, summary)
        print("per-case mean F1:")
        for case, f1 in summary["per_case_mean_f1"].items():
            print(f"  {case:20s} {f1:.4f}")
        print(f"churn null-pass  : {summary['churn_null_pass_rate']}")
        print(f"conservation     : {summary['conservation_all_ok']}")
        print(f"saved -> {path}")
        return

    if args.kaggle:
        summary = run_kaggle_suite(with_cart=args.cart, face_validity=args.face_validity)
        path = save_results(args.tag, summary)
        print(f"mean impact-F1 : {summary['mean_impact_f1']:.4f}")
        print(f"floor dataset  : {summary['floor_dataset_f1']:.4f}")
        print(f"conservation   : {summary['conservation_all_ok']}")
        print(f"mean #segments : {summary['mean_n_segments']:.1f}")
        print("per-dataset mean F1:")
        for case, f1 in summary["per_dataset_mean_f1"].items():
            print(f"  {case:24s} {f1:.4f}")
        print(f"saved -> {path}")
        return

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
