# Impact-Split Validation & Benchmark Report — v3 (floor loop)

> Generated at the end of the 2026-07-15 floor loop (
> `reports/cycle-log.md` §Loop 2 has the cycle record, `reports/floor-diagnosis.md`
> the Phase-0 diagnosis; raw scores in `benchmarks/results/cycle4-*.json` — the
> cycle-5 tags are the record of a refuted variant, not shipped). Supersedes
> `validation-report-v2.md`.

---

## 1. Executive summary

The floor loop targeted the five sub-0.85 dataset-seeds left by the robustness
loop (floor 0.781). Phase 0 ruled out hyperparameter fixes (27-config sweep,
best floor-case min 0.806) and located the mechanism: **the splitter could only
cut, never regroup** — 21 of 25 floor-case rules were recovered at uncapped-union
F1 0.87–1.00 with precision ≈ 1.0 but fragmented past the 3-segment union cap.

**Shipped change: post-fit segment consolidation** (`consolidate=True` default).
Terminal segments whose conditions differ on exactly one feature's category set
(the union stays one readable `feature=cat1,cat2` conjunction; vacuous
full-universe conditions are dropped) and whose means are statistically
compatible (two-sample z-test at `noise_z` against the pooled robust
within-segment residual scale) merge iteratively to fixpoint. The tree itself is
untouched — plots, traces and conservation semantics are unchanged; merging
disjoint row sets preserves exact sum conservation by construction, and
consolidation is null-safe (it can only reduce segment counts).

| suite | robustness loop (v2) | floor loop (v3, shipped) |
|---|---|---|
| synthetic battery (8 × 3 seeds) | 0.991 / 0.921 | 0.977 / 0.885 |
| semi-synthetic Kaggle (10 × 3 seeds) | 0.936 / 0.781 | **0.951 / 0.815** |
| full suite (51 scored) | 0.959 / 0.781 | **0.962 / 0.815** |
| mean segments (Kaggle / synthetic) | 29.8 / 14.3 | **22.1 / 11.4** |

Floor cases: black_friday/2026 0.781→0.835 · insurance/42 0.794→0.836 ·
ibm_hr/7 0.810→0.815 · olist/42 0.815→**0.915** · telco_churn/2026 0.837→**0.861**.
Null FP-control 3/3; conservation exact everywhere; pytest 34/34 (8 new tests);
CART (same metric) beaten on both suites (0.943 synthetic / 0.854 Kaggle vs our
0.977 / 0.951). Defaults: `delta_pct=0.01 · min_global_impact_pct=0.01 ·
max_depth=5 · noise_z=3.0 · consolidate=True`.

**The pre-registered floor bar (≥0.85) was NOT met: floor 0.8154.** The loop
closed on the pre-registered *explained-and-accepted* exit for the three
remaining cases (§4) after two relaxed-merge variants were refuted on guard
evidence (§3). 48 of 51 dataset-seeds sit at ≥0.85.

## 2. What consolidation fixes (and what it costs)

Fixed: cross-product fragmentation. When the tree splits on other rules'
features, a single-conjunction rule's mass tiles across sibling branches whose
means do not differ — consolidation reassembles them (olist `customer_state=MG`:
19 fragments → capped F1 0.72→0.82 at dataset level 0.815→0.915).

Recorded cost: noise_2x 1.000→0.885 (all seeds; still ≥0.85). Under 2× noise the
pooled σ̂ widens the z-band and two same-sign planted rules 15 units apart
(A +50 / C +35 at σ=44, statistically indistinguishable at their n) merge. This
is the correct statistical decision on the observable data — the "error" exists
only relative to ground truth no local test can see. The noise frontier
(diagnostic) shows the same at σ≥44.

## 3. Refuted variants (kept as negative results)

1. **Globally-immaterial distinction merging** — also merge when misattributed
   mass < 1% of global excess volume (same-excess-class guard). Broke baseline
   1.000→0.883 and adult_census/airbnb; autopsy showed good and bad merges both
   sit within 5% of any global floor — no separating margin exists.
2. **Equivalence margin** (`|Δm| ≤ 0.1σ̂`) — inert: real fragments differ by
   overlap composition (0.3–3σ); sub-band contrasts never get split apart in the
   first place. The dead zone it would rescue is empirically empty.

## 4. Remaining known weaknesses (explained and accepted)

- **ibm_hr/7 (0.815)** — overlapping rules tile the data into lattice cells with
  real 2–3σ pairwise differences; the truthful partition needs 5–7 segments and
  the ≤3-union under-credits it (uncapped 0.88–1.00). Merging those cells would
  be factually wrong. This is a metric-boundary case for overlapping ground
  truth, not a formula defect.
- **insurance/42 (0.836)** — planted smoker×sex contrast (0.7σ) is at the
  detectability edge even with oracle noise knowledge (τ 37.8 vs D 46.7); n=1.3k.
- **black_friday/2026 (0.835)** — a 1.9%-support order-3 rule is root-shattered
  by two 10%-support rules; even the uncapped union reaches only 0.736. No
  configuration in the sweep recovers it; a fix would need non-greedy/lookahead
  splitting.
- Very-high-noise abstention (σ ≥ 8× weakest effect) unchanged from v2 — by
  design.

## 5. Reproduction

```bash
python -m benchmarks.run --tag <tag> --frontier            # synthetic battery
python -m benchmarks.run --tag <tag> --kaggle --cart       # Kaggle suite
python -m benchmarks.floor_cases                           # floor-case repro + autopsy
python -m benchmarks.floor_cases --sweep                   # Phase-0 HPO grid
```
