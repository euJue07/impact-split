"""Oracle ceiling for shadow promotion: could shadows beat the status quo?

Scores each battery case with the partition's terminal segments PLUS every
accepted shadow handed to the unchanged scorer as extra candidates. The
union is greedy over a top-6 candidate window, so with many shadows a broad
shadow can crowd out partition segments (visible as f1_all < f1_partition);
the score is an upper bound on promotion value only in the absence of that
crowding — at high feature_subsample the pool is small and the measurement
is clean.

Decision gate (pre-registered in the 2026-07-19 spec): a config shows real
headroom iff (mean >= 0.9836 + 0.003 or floor >= 0.8846 + 0.010) evaluated
on the LOW-OVERLAP score (shadows with Jaccard >= 0.5 vs any partition
segment removed — gains must be new mass, not fragmentation credit), and
the null case surfaces zero shadows.

Conservation note: the combined candidate list overlaps by construction, so
``DatasetScore.conservation_ok`` is meaningless there and ignored; partition
integrity is asserted by the row-tiling check instead. ``scoring.py`` is
untouched — the impact-F1 metric is byte-identical to the published battery.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from impact_split.ensemble import jaccard, mask_from_conditions
from impact_split.splitter import ImpactSplitter

from .battery import DEFAULT_PARAMS
from .dgp import CASE_FACTORIES, SEEDS, BenchDataset
from .scoring import encode_with_model_maps, leaf_masks_from_model, score_dataset

BASELINE_MEAN = 0.9836
BASELINE_FLOOR = 0.8846
GATE_MEAN_GAIN = 0.003
GATE_FLOOR_GAIN = 0.010
OVERLAP_CAP = 0.5

FEATURE_SUBSAMPLES = (0.4, 0.6, 0.8)
SHADOW_REPLICATES = (50, 100)
N_BOOTSTRAP = 50


@dataclass
class CellResult:
    case: str
    seed: int
    f1_partition: float    # partition only (live re-derivation of the baseline)
    f1_all: float          # partition + all shadows (nan for null)
    f1_low_overlap: float  # partition + shadows with max Jaccard < OVERLAP_CAP
    n_shadows: int
    n_used: int            # shadows the greedy union actually used (all-shadows run)
    n_low_overlap: int     # shadows below the overlap cap
    null_clean: bool | None  # set only for the null case


def ceiling_cell(
    ds: BenchDataset,
    *,
    feature_subsample: float,
    shadow_replicates: int,
    n_replicates: int = N_BOOTSTRAP,
    seed: int = 0,
) -> CellResult:
    model = ImpactSplitter(**DEFAULT_PARAMS)
    model.fit(ds.X, ds.y)
    model.ensemble_report(
        ds.X,
        ds.y,
        n_replicates=n_replicates,
        shadow_replicates=shadow_replicates,
        feature_subsample=feature_subsample,
        seed=seed,
    )
    X_codes = encode_with_model_maps(model, ds.X)
    partition = leaf_masks_from_model(model, X_codes)
    if sum(int(m.sum()) for _, m in partition) != len(ds.y):
        raise AssertionError("partition masks do not tile the rows")
    shadows = model.ensemble_["shadows"]

    if not ds.rules:  # null case: the guardrail IS the result
        return CellResult(
            ds.case, ds.seed, float("nan"), float("nan"), float("nan"),
            len(shadows), 0, 0, null_clean=not shadows,
        )

    sc_part = score_dataset(ds, partition)

    entries = []
    for k, sh in enumerate(shadows):
        m = mask_from_conditions(sh["conditions"], X_codes)
        max_j = max((jaccard(m, pm) for _, pm in partition), default=0.0)
        entries.append((f"shadow[{k}]:{sh['path']}", m, max_j))

    def run(subset: list[tuple[str, np.ndarray, float]]):
        return score_dataset(ds, partition + [(lbl, m) for lbl, m, _ in subset])

    sc_all = run(entries)
    low = [e for e in entries if e[2] < OVERLAP_CAP]
    sc_low = run(low) if len(low) < len(entries) else sc_all
    used = {
        p
        for rs in sc_all.rule_scores
        for p in rs.matched_paths
        if p.startswith("shadow[")
    }
    return CellResult(
        ds.case, ds.seed,
        float(sc_part.impact_f1), float(sc_all.impact_f1), float(sc_low.impact_f1),
        len(shadows), len(used), len(low), null_clean=None,
    )


def main() -> None:
    print(f"published baseline: mean {BASELINE_MEAN:.4f}  floor {BASELINE_FLOOR:.4f}")
    header = (
        "fs    reps  part_mean  all_mean  all_floor  lowJ_mean  lowJ_floor"
        "   d_mean   d_floor  shadows  used  gate"
    )
    print(header)
    results: list[tuple[bool, float, float, int, list[CellResult]]] = []
    for fs in FEATURE_SUBSAMPLES:
        for sr in SHADOW_REPLICATES:
            cells: list[CellResult] = []
            null_clean = True
            for case, factory in CASE_FACTORIES.items():
                for seed in SEEDS:
                    cell = ceiling_cell(
                        factory(seed), feature_subsample=fs,
                        shadow_replicates=sr, seed=seed,
                    )
                    cells.append(cell)
                    if case == "null":
                        null_clean = null_clean and bool(cell.null_clean)
            scored = [c for c in cells if c.null_clean is None]
            part_mean = float(np.mean([c.f1_partition for c in scored]))
            all_mean = float(np.mean([c.f1_all for c in scored]))
            all_floor = float(np.min([c.f1_all for c in scored]))
            low_mean = float(np.mean([c.f1_low_overlap for c in scored]))
            low_floor = float(np.min([c.f1_low_overlap for c in scored]))
            gate = null_clean and (
                low_mean >= BASELINE_MEAN + GATE_MEAN_GAIN
                or low_floor >= BASELINE_FLOOR + GATE_FLOOR_GAIN
            )
            flag = "PASS" if gate else ("NULL-FAIL" if not null_clean else "-")
            print(
                f"{fs:.1f}   {sr:>4}  {part_mean:.4f}     {all_mean:.4f}    "
                f"{all_floor:.4f}     {low_mean:.4f}     {low_floor:.4f}    "
                f"{low_mean - BASELINE_MEAN:+.4f}  {low_floor - BASELINE_FLOOR:+.4f}"
                f"  {sum(c.n_shadows for c in scored):>7}"
                f"  {sum(c.n_used for c in scored):>4}  {flag}"
            )
            results.append((gate, low_mean, fs, sr, cells))
    gate, low_mean, fs, sr, cells = max(results, key=lambda r: (r[0], r[1]))
    label = "best passing config" if gate else "best config (gate NOT passed)"
    print(f"\nper-case breakdown, {label}: fs={fs} reps={sr}")
    print("case               seed  f1_part  f1_all  f1_lowJ  shadows  used")
    for c in cells:
        if c.null_clean is not None:
            continue
        print(
            f"{c.case:<18} {c.seed:>4}  {c.f1_partition:.4f}   {c.f1_all:.4f}"
            f"  {c.f1_low_overlap:.4f}   {c.n_shadows:>6}  {c.n_used:>4}"
        )
    if not any(r[0] for r in results):
        print("\nno config passed the gate — shadow promotion cannot beat the"
              " status quo on this battery (record verdict in reports/cycle-log.md)")


if __name__ == "__main__":
    main()
