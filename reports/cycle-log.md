# Robustness-loop cycle log

Spec: ai-os KB `impact-split-robustness-loop` (2026-07-14). Bar: mean impact-F1 ≥ 0.9
across the full suite AND no dataset below 0.7, with one fixed default configuration
(`delta_pct=0.05, min_global_impact_pct=0.01, max_depth=5`). Primary metric:
impact-weighted F1 (union of ≤3 segments per rule; precision normalized by the true
rule mask's own achievable precision). Row-Jaccard demoted to shape diagnostic.

---

## Cycle 0 — baseline measurement (no formula change)

**Synthetic battery** (8 cases × seeds {42, 7, 2026}; `benchmarks/results/cycle0-synthetic.json`):

| metric | value |
|---|---|
| mean impact-F1 (21 scored datasets) | **0.9735** |
| floor dataset F1 | **0.8174** (noise_2x, seed 2026) |
| null-case pass rate (FP control) | 3/3 |
| sum conservation | exact, all datasets |
| mean terminal segments | 20.8 |

Per-case mean F1: baseline 1.000 · one_sided 1.000 · high_cardinality 0.961 ·
deep_interactions 0.942 · noise_2x 0.939 · skewed_volume 0.986 · overlapping 0.987.

**Reading.** Under the impact-weighted metric the current formula is much stronger than
the old Jaccard view implied (4/6): fragmentation that Jaccard punished carries the
impact mass just fine (e.g. one_sided `Visayas x Online` recovered exactly by a union
of 2 leaves). The genuine weaknesses that remain:

1. **Dilution under noise / distraction** — the recurring failure shape is
   `recall = 1.0, precision ≈ 0.3`: the rule's mass ends up inside one broad segment the
   tree never refines (noise_2x seed 2026: `Mindanao x Partner x {A,B}` prec 0.277;
   high_cardinality seed 7: `Luzon x Online x C` prec 0.345). The δ sieve at those nodes
   is too coarse once σ or a 50-level nuisance inflates V_node.
2. **Noise frontier** (diagnostic, seed 42): F1 = 1.00 @ σ22, 1.00 @ σ44, 0.886 @ σ88,
   0.660 @ σ176, 0.537 @ σ352. Breakdown begins ~σ88 (weakest planted increment 35).

CART (MSE, depth 5) reference on the same metric: comparable on average; CART wins on
noise_2x seed 2026 (0.93 vs 0.82), impact-split wins elsewhere. No regression guard
violated (this is the anchor cycle).

**Verdict.** Synthetic leg already clears the bar (0.97 / 0.82). The bar is over the
*full* suite — Kaggle semi-synthetic leg pending; improvement cycles target whatever
the combined suite exposes, starting from the dilution failure shape.
