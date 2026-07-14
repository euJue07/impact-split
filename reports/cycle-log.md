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
