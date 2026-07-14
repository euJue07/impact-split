# Floor diagnosis — Phase 0 of the floor loop

> Spec: ai-os KB #101 (`impact-split-floor-loop`, 2026-07-14). Diagnoses the five
> sub-0.85 dataset-seed cases from cycle 3 before any formula change. Raw data:
> `benchmarks/results/floor-repro-autopsy.json`, `benchmarks/results/floor-sweep.json`.
> Tooling: `benchmarks/floor_cases.py`; the uncapped-union diagnostic column was
> added to `benchmarks/scoring.py` (diagnostic only — the primary capped metric is
> unchanged and reproduced cycle-3 scores exactly, see §1).

## 1. Reproduction

All five floor cases reproduce the committed cycle-3 scores to machine precision
under shipped defaults (`delta_pct=0.01, min_global_impact_pct=0.01, max_depth=5,
noise_z=3.0`): black_friday/2026 = 0.7806, insurance/42 = 0.7942, ibm_hr/7 =
0.8096, olist/42 = 0.8147, telco_churn/2026 = 0.8375. Deterministic.

## 2. Per-rule autopsy — capped vs uncapped union

The uncapped-union column (same greedy accept-if-F1-improves, no 3-segment cap)
splits the 25 floor-case rules into two sharply different populations:

**Fragmentation cluster (21 rules).** Capped F1 0.61–0.98 with precision ≈ 1.0;
uncapped F1 **0.87–1.00** using 4–21 segments. The tree *finds* these rules —
their mass sits in clean, high-precision leaves — but scattered past the
3-segment union. Inspection of the matched paths shows fragments typically
differ in **one feature's category set across sibling branches** (e.g. olist
`customer_state=MG` recovered perfectly by 19 leaves that differ in month /
product-category routing). The worst capped scores in this cluster:
olist `product_category_name=papelaria|eletronicos` 0.712→0.995 uncapped,
olist `customer_state=MG` 0.721→1.000, telco `Contract=One year & Partner=No`
0.740→1.000, ibm_hr `Department=HR|Sales & OverTime=Yes` 0.739→1.000.

**Genuine misses (4 rules, 2 mechanisms).**

- *black_friday/2026 root-split shattering* — `Gender=M & Age=18-25 &
  Occupation=0|20` (support 1.9%, inc −2.55): uncapped only **0.736**. The root
  split on `Stay_In_Current_City_Years` (driven by two 10%-support rules)
  shatters this rule orthogonally; its mass lands in 10+ leaves, several heavily
  diluted (raw precision ~0.2), and `Gender` never becomes a split feature.
- *insurance/42 sub-noise-floor contrast* — `smoker=yes & sex=female` (uncapped
  0.682) and `smoker=yes & sex=male` (uncapped 0.802): four of the dataset's
  five rules overlap on smoker slices; within `smoker=yes` the planted sex
  contrast is ~0.7σ against a shared −7 effect (n=274). The tree correctly
  isolates smokers; the sex refinement is statistically marginal at n=1.3k.

## 3. Hyperparameter sweep — outcome (a) ruled out

27 configs (`delta_pct` ∈ {0.005, 0.01, 0.02} × `noise_z` ∈ {2, 3, 4} ×
`max_depth` ∈ {5, 6, 7}), scored on the five floor cases:

| config | floor-case min F1 | floor-case mean F1 |
|---|---|---|
| defaults (0.01 / 3.0 / 5) | 0.781 | 0.807 |
| best: 0.005 / 4.0 / any depth | **0.806** | 0.841 |
| worst: 0.01 / 2.0 / 6–7 | 0.694 | 0.802 |

**No configuration reaches 0.85.** The best config moves the floor +0.025;
`max_depth` is irrelevant (±0.003 across 5/6/7); loosening `noise_z` to 2.0
actively hurts (more fragmentation). The formula, at any setting in this
family, cannot consolidate rule mass into ≤3 segments — because greedy
splitting *has no consolidation operator at all*. Neither new global defaults
nor an auto-tuner can clear the bar. (No full-suite re-score was run: no
candidate cleared the floor-case screen, so there was nothing to promote.)

## 4. Assignment

| case | verdict | evidence |
|---|---|---|
| black_friday/2026 | fragmentation (4 rules) + **one formula gap** (shattered order-3 rule, uncapped 0.736) | §2 |
| insurance/42 | fragmentation (3 rules) + **two marginal-contrast rules** (sub-noise-floor sex split) | §2 |
| ibm_hr/7 | pure fragmentation (uncapped 0.88–1.00) | §2 |
| olist/42 | pure fragmentation (uncapped 0.97–1.00) | §2 |
| telco_churn/2026 | pure fragmentation (uncapped 0.87–1.00) | §2 |

Outcome **(a) config-fixable: NO** (§3). Outcome **(c) pure metric artifact:
NO** — the ≤3-segment cap is doing its job of encoding readability; the tree
genuinely emits 25–72 segments where a readable answer needs consolidated ones.
Verdict: **(b) formula gap, specifically a missing consolidation step.** The
splitter can only cut, never regroup; when planted structure is a single
conjunction whose mask the tree partitions along irrelevant features, nothing
ever reassembles it.

## 5. Decision — Phase 1 proceeds

**Cycle-1 hypothesis: post-fit segment consolidation (merge pass).** After
fitting, iteratively merge terminal segments whose paths differ in exactly one
feature's category set (so the merged path stays in the legal
`feature=cat1,cat2` conjunction grammar) and whose per-row excess means are
statistically compatible (same sign, difference within the existing per-feature
noise floor). This directly reverses noise-irrelevant fragmentation while
provably preserving exact sum conservation (unions of disjoint segments) and
readable paths.

Upper bound if consolidation reaches uncapped-level recovery within 3 segments:
black_friday/2026 → ~0.94, insurance/42 → ~0.88, ibm_hr/7 → ~0.96, olist/42 →
~0.99, telco_churn/2026 → ~0.96 — all five clear 0.85 **without** touching the
two genuine-miss mechanisms (root-split shattering, marginal contrasts), which
remain fallback targets for cycles 2+ only if consolidation alone falls short.

Guards unchanged (KB #101 §Constraints): pytest green, conservation exact,
null 3/3 (merging cannot create splits — null-safe by construction), no passing
dataset below 0.85, mean ≥ 0.955, segments ≤ 1.5× cycle-3 (merging reduces
them), CART beaten.
