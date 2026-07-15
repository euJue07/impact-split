# Robustness-loop cycle log

Spec: robustness loop (2026-07-14). Bar: mean impact-F1 ≥ 0.9
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

**Kaggle leg (cycle 0)** — 10 datasets × seeds {42, 7, 2026}, semi-synthetic (real
covariates, injected rules; half the datasets get a +3σ constant base to mimic
revenue-like one-sided KPIs). `benchmarks/results/cycle0-kaggle.json`:

| metric | value |
|---|---|
| mean impact-F1 | **0.6802** |
| floor dataset | **0.1208** (adult_census seed 42) |
| CART reference mean | 0.854 (beats us) |

Failure autopsy: 58 failing rules — mean recall 0.74, mean precision **0.30**; 74% have
precision < 0.5. Base=3σ datasets score 0.56 vs 0.80 for zero-base. **Mechanism:** with
an all-positive base, the *raw* δ sieve always finds categories over the 5%-of-volume
threshold — the big ones — so the tree splits on volume, not effect, and the
centered-excess fallback (which would see through the base) never fires.

---

## Cycle 1 — centered excess as the only routing signal

**Hypothesis.** Routing categories by raw sums confounds effect with volume whenever the
target has a one-sided base; making the centered excess D_cat = S_cat − n_cat·ȳ_node the
*only* routing signal (threshold δ = delta_pct × Σ|y−ȳ|) removes the volume artifact
while leaving zero-centered behavior unchanged (mean≈0 ⇒ centered ≡ raw).

**Change.** `splitter.py::_build`: dropped the raw-mode phase; the former fallback is now
the single sieve. Test `test_one_sided_gain_can_split` updated (same intent — one-sided
targets still split — new mode label and both-sided gains).

**Scores.**

| suite | cycle 0 | cycle 1 | Δ |
|---|---|---|---|
| synthetic mean / floor | 0.9735 / 0.8174 | **0.9935 / 0.9367** | +0.020 / +0.119 |
| kaggle mean / floor | 0.6802 / 0.1208 | **0.8583 / 0.6039** | +0.178 / +0.483 |

Guards: pytest 26/26 green · conservation exact everywhere · case-1 baseline 1.000 (no
regression) · null 3/3 · segments: synthetic 20.8→17.6, kaggle 29.1→36.6 (1.26×, within
1.5×) · CART now matched (0.858 vs 0.854). Base=3σ vs zero-base gap closed
(0.848 vs 0.868). Noise frontier note: recovery now perfect through σ=88 (was 0.886) but
the tree abstains entirely (1 segment) at σ≥176 where it previously emitted partial
recoveries — a defensible trade (diagnostic only, not scored).

**Verdict: material improvement (+0.178 kaggle mean). Bar not yet met (0.86 < 0.9,
floor 0.60 < 0.7). New dominant failure shape: fragmentation — recall 0.5–0.7 at
precision ≈ 1.0 with the 3-segment union cap exhausted; over-splitting deep in the tree
(ibm_hr: 74 leaves on n=1,470) because 5% of a small node's excess volume is below the
noise band. Next: noise-aware sieve floor.**

---

## Cycle 2 — per-category noise floor in the sieve

**Hypothesis.** Deep-node over-splitting happens because the relative δ has no
statistical meaning: 5% of a small node's excess volume sits below the noise band, so
noise categories keep routing. Under a within-category-noise null, D_cat wanders
~σ√n_cat; requiring |D_cat| > max(δ_rel, z·σ̂_f·√n_cat) (σ̂_f = 1.4826·MAD of the
feature's within-category residuals, z = 3.0 fixed) should stop noise routing without
touching real effects.

**Change.** `noise_z=3.0` parameter + per-category τ in the sieve.
`test_constant_feature_skipped_child_prefers_other_column` re-fixtured (its 4-row toy
had no statistically real split; intent preserved with clean signal).

**Scores.** synthetic 0.9935→0.9925 / floor 0.9367→0.9211 (noise-level churn), kaggle
mean 0.8583→**0.8991**, floor 0.6036 (unchanged, olist). Segments: kaggle 36.6→26.1,
ibm_hr 74→31 leaves. Guards all green (case-1 = 1.000, null 3/3, conservation, pytest).

**Verdict: material (+0.041 kaggle mean). Remaining failure: catastrophic dilution of
1–2 rules per floor dataset — rules trapped in giant neutral catch-alls.**

---

## Cycle 3 — sieve rebalance (δ 0.05→0.01) + interaction-order depth cap

**Hypothesis.** Trace autopsy on olist seed 7: failing rules sit 88–91% inside ONE
leaf with n=90,225 (80% of rows), stop=no_split — a 1%-support rule carries ~4.3% of
that node's excess volume, permanently below a 5% relative bar. With the noise floor
now carrying significance duty, δ_rel can drop to 1% (materiality only). Second
finding: with the finer sieve, the tree pools categories coarsely then re-splits the
same feature deeper (region→channel→region→channel→product), exhausting max_depth=5
before separating 3-way rules — so max_depth should cap *interaction order* (distinct-
feature transitions), with same-feature refinements free and still legal at the cap.

**Change.** `delta_pct` default 0.05→0.01; `_build` tracks `inter_depth`/
`parent_feature`; at the cap only same-feature refinements are evaluated. Plot y-limits
switched to measured physical depth.

**Scores.**

| suite | cycle 2 | cycle 3 | Δ |
|---|---|---|---|
| synthetic mean / floor | 0.9925 / 0.9211 | 0.9906 / 0.9211 | −0.002 / = |
| kaggle mean / floor | 0.8991 / 0.6036 | **0.9361 / 0.7806** | +0.037 / +0.177 |

olist 0.80→0.90 (floor rule 0.01→0.91), black_friday 0.82→0.87, vgsales 0.91→0.99.
Guards: pytest 26/26 · case-1 = **1.000** · null 3/3 · conservation exact · segments
kaggle 29.8 (1.02× cycle-0's 29.1), synthetic 14.3 (0.69×) · CART 0.854 now beaten.

---

## Loop end — bar met at cycle 3

Full suite (51 scored datasets): **mean impact-F1 0.959 ≥ 0.9; floor 0.781 ≥ 0.7;
null control 3/3; one fixed default configuration** (`delta_pct=0.01,
min_global_impact_pct=0.01, max_depth=5, noise_z=3.0`). Loop closed per spec —
no auto-tuning heuristic needed (defaults did not plateau below the bar), sacred set
untouched. Residual weaknesses recorded in `validation-report-v2.md` §5.


---

# Loop 2 — floor loop (2026-07-14)

Bar: floor >= 0.85 (stretch 0.90), mean >= 0.955, defaults-or-auto-tuner only.
Phase 0 (`reports/floor-diagnosis.md`): repro deterministic; 27-config HPO sweep
ruled out config-fixability (best floor-case min 0.806 < 0.85); uncapped-union
diagnostic showed 21/25 floor rules found at F1 0.87-1.00 / precision ~1.0 but
fragmented past the 3-segment cap. Verdict: missing consolidation operator.

## Cycle 1 — post-fit segment consolidation (merge pass)

**Hypothesis.** The splitter can only cut, never regroup; when planted structure is
a single conjunction whose mask the tree partitions along other rules' features,
nothing reassembles it. A post-fit merge pass — merge terminal segments whose
conditions differ on exactly one feature's category set (union stays one readable
conjunction; vacuous full-universe conditions dropped) and whose means pass a
two-sample z-test at `noise_z` against the pooled robust within-segment residual
scale, iterated to fixpoint — should consolidate fragmentation without inventing
structure (null-safe by construction: merging cannot create splits).

**Change.** `splitter.py`: `consolidate=True` constructor default; `segments_`
(post-fit consolidated segments with exact conjunction conditions);
`get_impact_segments()` serves consolidated segments; tree untouched (plots,
trace unchanged). `benchmarks/scoring.py::leaf_masks_from_model` prefers
`segments_`. No existing test needed edits; 8 new tests (consolidation + scoring).

**Scores.**

| suite | cycle 3 | cycle 1 (=cycle4 tags) | delta |
|---|---|---|---|
| synthetic mean / floor | 0.9906 / 0.9211 | 0.9767 / 0.8846 | -0.014 / -0.037 |
| kaggle mean / floor | 0.9361 / 0.7806 | 0.9513 / 0.8154 | +0.015 / +0.035 |
| full suite mean / floor | 0.959 / 0.781 | **0.9617 / 0.8154** | +0.003 / +0.035 |

Floor cases: black_friday/2026 0.781->0.835, insurance/42 0.794->0.836, ibm_hr/7
0.810->0.815, olist/42 0.815->0.915, telco/2026 0.837->0.861. Guards: pytest 34/34,
conservation exact, null 3/3, no passing dataset-seed below 0.85, segments 0.74x
cycle-3 (22.1 kaggle / 11.4 synthetic), CART beaten both suites (0.854/0.943),
case-1 = 1.000.

**Cost (recorded).** noise_2x 1.000->0.885 all seeds (still >= 0.85): under 2x noise
the pooled sigma-hat inflates the merge threshold and consolidation occasionally
joins a small genuinely-different segment. Noise frontier (diagnostic) degraded at
sigma 44-88. Cycle-2 candidate: tighten merge compatibility under noise (mirror the
split sieve: only merge pairs the sieve would refuse to split).

**Verdict: material (+0.035 floor, bar not yet met — 0.815 < 0.85). Remaining
sub-0.85: ibm_hr/7 0.815, black_friday/2026 0.835, insurance/42 0.836.**


## Cycle 2 — REFUTED: relaxed merge compatibility (two variants)

**Hypothesis.** Cycle-1's z-test merge criterion under-merges: (v1) merge also when
the pair's distinction mass n1*n2/(n1+n2)*|m1-m2| is below min_global_impact_pct of
global excess volume (with same-P/Neu/N-class guard); (v2) merge also when
|m1-m2| <= merge_eps*sigma (equivalence margin, fixing the z-test's large-n
inconsistency).

**Evidence against (v1).** Guard battery FAILED: synthetic baseline 1.000->0.883
(case-1 guard), high_cardinality 0.951->0.913, adult_census 0.924->0.853, airbnb
0.996->0.938, kaggle floor 0.718 (`cycle5-*.json`). Autopsy: the baseline wrong
merge (Luzon x Online product A+C, two distinct planted rules, diff 0.68 sigma,
distinction ~1310 vs floor 1250) and the desired ibm_hr lattice merges
(distinction ~22 vs floor ~25) both sit within 5% of their floors — there is no
margin separating good from bad merges with this statistic at any threshold.

**Evidence against (v2).** Inert: floor cases identical to cycle 1 to 4 decimals.
Cross-feature fragments differ by overlap composition (0.3-3 sigma, above any safe
band); sub-band contrasts are never split apart in the first place (the split sieve
needs ~4 sigma/sqrt(n)). The dead zone the margin would rescue is empirically empty.

**Action: reverted to cycle-1 consolidation. No formula change shipped.**

---

## Loop 2 close — explained-and-accepted exit (floor 0.8154 vs 0.85 bar)

Shipped state = cycle 1 (consolidation, z-test compatibility). Full suite
**mean 0.9617 / floor 0.8154** under the frozen primary metric; bar floor 0.85 NOT
met; remaining sub-0.85 residue is 3 dataset-seeds with fully-diagnosed mechanisms
at honest statistical limits:

- **ibm_hr/7 (0.815)** — overlapping planted rules tile the data into lattice cells
  with real 2-3 sigma pairwise differences; the truthful partition needs 5-7
  segments and the frozen <=3-union under-credits it (uncapped-union F1 0.88-1.00,
  dataset mean ~0.93). Merging those cells would be factually wrong (cycle-2 v1
  showed the collateral damage). Metric-boundary case, not a formula defect.
- **insurance/42 (0.836)** — the planted smoker-sex contrast (0.7 sigma between
  overlapping rules) is at the detectability edge even with oracle noise knowledge
  (tau 37.8 vs D 46.7 at true sigma); n=1.3k. Genuinely ambiguous data.
- **black_friday/2026 (0.835)** — Gender x Age x Occupation (support 1.9%) is
  root-shattered by two 10%-support rules; uncapped 0.736 (real dilution); the
  27-config sweep showed no setting recovers it.

Cycles 3-4 unused: both cycle-2 variants were refuted on evidence and the residual
mechanisms bound any locally-computable improvement. Per the pre-registered exits,
these three cases close as **explained and accepted**; everything else
meets the bar (48/51 dataset-seeds >= 0.85, mean +0.003 over cycle 3, segments
-26%, CART beaten on both suites).
