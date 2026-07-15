# Impact-Split Validation & Benchmark Report — v2 (robustness loop)

> **SUPERSEDED — see `validation-report-v3.md`** (floor loop). Kept for history.

> Generated at the end of the 2026-07-14 formula-robustness loop
> (`reports/cycle-log.md` has the full cycle-by-cycle record; raw scores live in
> `benchmarks/results/*.json`). Supersedes `validation-report.md` (v1), whose
> Jaccard-based rule-recovery metric is retained only as a shape diagnostic.

---

## 1. Executive summary

Three formula cycles took the algorithm from failing on realistic covariate
structure (impact-F1 0.68, floor 0.12 on the Kaggle suite) to clearing the
pre-registered robustness bar with one fixed default configuration:

| suite | cycle 0 (old formula) | cycle 3 (shipped) |
|---|---|---|
| synthetic battery (8 cases × 3 seeds) | 0.974 mean / 0.817 floor | **0.991 / 0.921** |
| semi-synthetic Kaggle (10 datasets × 3 seeds) | 0.680 mean / 0.121 floor | **0.936 / 0.781** |
| full suite (51 scored datasets) | 0.853 mean / 0.121 floor | **0.959 / 0.781** |

**Bar: mean impact-F1 ≥ 0.9 AND no dataset below 0.7, defaults only — met.**
Null-dataset false-positive control passes 3/3 in every cycle; exact sum
conservation holds on every dataset; the 26-test pytest suite is green; terminal
segment counts stayed within 1.02× of the cycle-0 formula on the Kaggle suite
(and *dropped* ~30% on the synthetic battery). CART (MSE, depth 5) scored with
the identical metric: 0.854 Kaggle mean — the shipped formula now beats it.

## 2. What changed in the formula (and why)

1. **Centered excess is the only routing signal** (cycle 1). Raw category sums
   confound effect with volume: on one-sided KPIs (constant positive base),
   every large category cleared the old raw sieve, the tree split on size, and
   the centered fallback never fired. D_cat = S_cat − n_cat·ȳ_node is now the
   single signal; zero-centered targets behave as before (mean ≈ 0 ⇒ centered ≡
   raw). Kaggle mean +0.18.
2. **Per-category noise floor** (cycle 2). Categories must clear
   max(delta_pct·V_excess, noise_z·σ̂_f·√n_cat), σ̂_f = 1.4826·MAD of the
   feature's within-category residuals. Under a pure-noise null D_cat wanders
   ~σ√n_cat, so noise alone can no longer route — deep-node fragmentation
   stopped (ibm_hr: 74 leaves on n=1,470 → 31). Kaggle mean +0.04.
3. **Sieve rebalance + interaction-order depth** (cycle 3). With the floor
   carrying significance duty, the materiality bar dropped delta_pct 0.05 →
   0.01 — a 1%-support rule trapped in a giant neutral catch-all (80% of rows)
   is detectable again (olist floor rule 0.01 → 0.91). max_depth now counts
   distinct-feature transitions; same-feature refinements are free and remain
   legal at the cap (restores the two 3-way baseline rules that alternating
   region/channel re-splits had squeezed out of the depth budget). Kaggle mean
   +0.037, floor 0.60 → 0.78.

Shipped defaults: `delta_pct=0.01 · min_global_impact_pct=0.01 · max_depth=5 ·
noise_z=3.0`. Sacred set untouched: ternary P/Neu/N tree over categoricals,
exact sum conservation, readable `feature=cat1,cat2` paths.

## 3. Primary metric — impact-weighted F1

Per planted rule R with per-row contribution e_R, matched against a union of at
most 3 terminal segments M (greedy, accept-if-F1-improves):

- recall = |Σ_M e_R| / |Σ e_R| — share of the rule's impact captured;
- precision = (|Σ_M e_R| / Σ_M |y|) normalized by the true mask's own achievable
  precision (so perfect recovery scores 1.0 at any noise level), capped at 1;
- row-Jaccard (v1's metric) is kept per rule as a shape diagnostic only — it is
  count-weighted and punishes impact-preserving fragmentation.

## 4. Test suite

### 4.1 Synthetic battery (8 cases × seeds {42, 7, 2026})

| case | stresses | cycle-3 mean F1 |
|---|---|---|
| baseline | explainer DGP, regression anchor | 1.000 |
| one_sided | all effects same sign | 1.000 |
| high_cardinality | 50-level nuisance feature | 0.948 |
| deep_interactions | 3- and 4-way rules | 1.000 |
| noise_2x | σ doubled | 1.000 |
| skewed_volume | one category = 80% of rows | 1.000 |
| overlapping | non-disjoint rule masks | 0.987 |
| null | pure noise (FP control) | pass 3/3 |

Noise frontier (diagnostic, baseline rules): perfect recovery through σ=88
(weakest planted increment = 35); at σ=176 partial (0.73); at σ=352 the tree
abstains (single root leaf) rather than hallucinate — consistent with the
noise-floor design.

### 4.2 Semi-synthetic Kaggle suite (10 datasets × seeds {42, 7, 2026})

Real covariate structure, injected known effects (5 rules/dataset, order 1–3,
support 1–12%, |increment| 1.5–6σ, mixed signs; half the datasets get a +3σ
constant base to mimic revenue-like KPIs). Real targets used only for
face-validity groupbys, never scored — real data has no segment ground truth.

| dataset | domain | n | max cardinality | cycle-3 F1 |
|---|---|---|---|---|
| superstore | retail | 10k | 49 (State) | 0.949 |
| insurance | health | 1.3k | 6 | 0.927 |
| adult_census | socioeconomic | 33k | 42 (country) | 0.906 |
| vgsales | media | 17k | 579 (Publisher) | 0.991 |
| airbnb_nyc | housing | 49k | 221 (neighbourhood) | 0.991 |
| telco_churn | telecom | 7k | 4 | 0.942 |
| ibm_hr | HR | 1.5k | 9 (JobRole) | 0.881 |
| black_friday | retail-large | 550k | 21 (Occupation) | 0.874 |
| olist | e-commerce | 113k | 74 (category) | 0.905 |
| wine | wine | 130k | 708 (variety) | 0.994 |

## 5. Remaining known weaknesses

- Very-high-noise regimes (σ ≥ 8× the weakest effect) produce abstention, not
  recovery — by design, but callers should know a silent tree ≠ no structure.
- black_friday / ibm_hr (~0.87–0.88): residual fragmentation of low-support
  interaction rules across sibling branches when several rules share a feature;
  the ≤3-segment union recovers most but not all mass.
- The injection DGP assumes effects are constant per rule (homogeneous
  increments). Heterogeneous within-rule effects were not benchmarked.

## 6. Reproduction

```bash
python -m benchmarks.run --tag <tag> --frontier          # synthetic battery
python -m benchmarks.run --tag <tag> --kaggle --cart     # Kaggle suite (needs KAGGLE_API_TOKEN)
```
