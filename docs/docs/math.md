# The math behind impact-split

This page is the derivation layer. The [README](../../README.md) states each
formula and why it exists; this page shows where each one comes from, what it
assumes, and what breaks when those assumptions fail. It is written for a reader
who wants to check the algorithm rather than use it — every section closes with
an **Enforced by:** line naming the tests that hold the property in place, and
the last section states plainly which numbers in the library were *not* derived
at all. Nothing here describes behaviour the code does not have: the derivations
follow `impact_split/splitter.py` and `impact_split/ensemble.py` as shipped.

A note on what "derived" means here. The **shapes** of the formulas — the
$\sqrt{n}$ in the null band, the $1/k$ in the gain, the $\sqrt{2\ln K}$
multiplicity term, the $\sqrt{1/n_1 + 1/n_2}$ in the merge test — follow from
stated assumptions and are argued below. The **levels** (`delta_pct=0.01`,
`min_global_impact_pct=0.01`, `max_depth=5`) do not. Section 9 says so
explicitly.

---

## 1. Notation and setting

| Symbol | Meaning |
| --- | --- |
| $y_i$ | row-level target value for row $i$ |
| $V_{node} = \sum_{i \in node} \lvert y_i \rvert$ | node absolute volume |
| $V^{c}_{node} = \sum_{i \in node} \lvert y_i - \bar{y}_{node} \rvert$ | node **excess** volume (absolute deviation from the node mean) |
| $S_{cat} = \sum_{i \in cat} y_i$ | raw sum of a category within the current node |
| $n_{cat}$ | that category's row count in the node |
| $D_{cat} = S_{cat} - n_{cat}\bar{y}_{node}$ | the category's **centered excess** (the routing signal) |
| $S_P,\ S_N$ | sums of the routed positive / negative outer branches |
| $k_P,\ k_N$ | number of categories routed to each outer branch |
| $V_{global\_P},\ V_{global\_N}$ | global positive / negative pools, defined in §6 |

**Assumption (A1) — additive (extensive) KPI.** The method assumes $y$ is a quantity whose *sum*
over a set of rows is the meaningful aggregate for that set: profit, revenue,
margin, cost, hours, claim amount, churned dollars. Formally, the method needs
the target to be defined for any row subset $A$ as $Y(A) = \sum_{i \in A} y_i$
and needs $Y$ to be the business quantity of interest — so that $Y(A \cup B) =
Y(A) + Y(B)$ for disjoint $A, B$.

This is not a formality; three things in the algorithm depend on it directly:

1. **A category's contribution is its sum.** Routing compares $D_{cat}$ across
   categories, which is only a fair comparison if adding rows adds impact.
2. **Conservation is meaningful.** Every leaf sum adds back to the dataset total
   (§7), which is a statement about the KPI, not just about arithmetic on
   floats — it is only *interpretable* when $y$ is extensive.
3. **The global pools in §6 exist.** $V_{global\_P}$ is "all the positive impact
   there is". For an intensive target this denominator has no meaning.

**Where it does not apply.** Intensive targets — ratios, rates, percentages,
per-unit prices, scores, NPS, conversion rate — are not additive: the sum of a
segment's conversion rates is not the segment's conversion rate. Fitting one
anyway produces arithmetically valid output whose interpretation is wrong. The
correct move for a ratio KPI is to fit on its extensive numerator (e.g. margin
dollars rather than margin percent) and read the ratio afterwards.

The features $X$ are categorical throughout; float columns are binned into
integer categories at fit time (`numeric_binning_strategy`, `numeric_n_bins`),
so everything below applies to the binned codes rather than the raw values.

**Position in the literature.** The method is a member of the tree-structured
subgroup-discovery family. Its ancestry is concrete rather than rhetorical:
recursive partitioning on a target statistic is the CART template (Breiman et
al. 1984); splitting categorical features by grouping levels through
significance tests — rather than exhaustive binary search — is the CHAID move
(Kass 1980), and §7's merge test is the same idea run in reverse; hunting for
row subsets whose *aggregate* target mass is unusually large, rather than
minimizing predictive loss, is the defining objective of bump hunting (Friedman
& Fisher 1999) and of subgroup discovery with an explicit quality function
(Klösgen 1996; Wrobel 1997). What this method adds to that family is the
routing statistic itself: a centered, volume-denominated excess with a
two-bar materiality-and-significance sieve (§§2–3), chosen so that the
reported segments are business-material by construction and not merely
statistically distinguishable.

**Enforced by:** `tests/test_impact_splitter.py::test_trace_records_split_and_conserves_total_sum`,
`tests/test_viz_data.py::test_payload_conservation_and_counts`.

---

## 2. Centered excess — separating effect from volume

The routing signal is not the category's raw sum but its deviation from the
node's expected share:

```math
D_{cat} = S_{cat} - n_{cat}\cdot \bar{y}_{node}
```

**The artifact this removes.** Read $n_{cat}\bar{y}_{node}$ as the sum this
category *would* have if it were unremarkable — that is, if its rows were drawn
from the node's overall level. $D_{cat}$ is what is left after charging the
category for its own size.

**Proposition 2.1 (contrast form).** $D_{cat} = n_{cat}\big(\bar{y}_{cat} -
\bar{y}_{node}\big)$: the centered excess is the category's mean elevation
above the node, weighted by its size. *Proof.* $S_{cat} = n_{cat}\bar{y}_{cat}$
by definition of the mean; substitute. ∎

This identifies the routing signal as a size-weighted contrast of group means
in the one-way layout (Scheffé 1959) — the same object analysis of variance
tests, but accumulated as impact ($n \cdot$ mean-difference has the units of
$y$-mass) rather than normalized into an $F$-ratio.

The failure mode without that subtraction is not subtle on one-sided KPIs. Take
a revenue-like target with a positive base and no real category effect, and let
category $A$ hold 10,000 rows and category $B$ hold 100. Even with identical
per-row behaviour, $S_A \approx 100 \cdot S_B$. Any threshold on the raw sum is
therefore, in effect, a threshold on row count: the big categories clear it, the
small ones do not, and the tree splits on *size* while reporting that it found
*effect*. Depth makes it worse, not better, because each child inherits the same
one-sided base.

A concrete instance is the regression test for this section. With
$y = (10, 10, 1, 1)$ over categories $(0,0,1,1)$ and `delta_pct=0.05`, the raw
sums are $S_0 = 20$ and $S_1 = 2$ against a materiality bar of
$0.05 \times 22 = 1.1$. **Both** clear it, both route positive, no rows are
separated, and the candidate is discarded as a non-partitioning split. Centering
gives $\bar{y} = 5.5$, $D_0 = 20 - 11 = +9$, $D_1 = 2 - 11 = -9$: the same data
now splits cleanly into a high side and a low side.

**Agreement on zero-centered targets.** When $y$ is already centered — profit
and loss, variance-to-budget, week-over-week delta — $\bar{y}_{node} \approx 0$,
so $D_{cat} = S_{cat} - n_{cat}\bar{y}_{node} \approx S_{cat}$ and centered
routing reduces to raw routing. The correction costs nothing where it is not
needed, which is why it is the *only* routing mode as of the 2026-07 robustness
loop; raw-sum routing was removed rather than kept as an option.

**Proposition 2.2 (zero-sum constraint).** $\sum_{cat} D_{cat} = 0$ exactly
within every node. *Proof.* $\sum_{cat} D_{cat} = \sum_{cat} S_{cat} -
\bar{y}_{node}\sum_{cat} n_{cat} = n_{node}\bar{y}_{node} -
n_{node}\bar{y}_{node} = 0$. ∎ A node's positive and negative excesses
always balance. The method therefore reports *relative* over- and
under-performance within each slice, not absolute sign — a category routed
"negative" in a strongly profitable node may still be profitable, just less so
than its peers. The segment ledger reports the true $\Sigma y$, so the reported
numbers are absolute even though the routing decision was relative.

**Enforced by:** `tests/test_impact_splitter.py::test_centered_excess_fallback_splits_one_sided_target`,
`tests/test_impact_splitter.py::test_one_sided_gain_can_split`.

---

## 3. The two-bar threshold

A category routes to P (or N) only if its centered excess clears both a
materiality bar and a significance bar:

```math
\tau_{cat} = \max\Big(\underbrace{V^{c}_{node} \cdot \mathrm{delta\_pct}}_{\text{materiality}},\ \underbrace{z \cdot \hat{\sigma}_f \cdot \sqrt{n_{cat}}}_{\text{significance}}\Big)
```

Route P if $D_{cat} > \tau_{cat}$, N if $D_{cat} < -\tau_{cat}$, else neutral.

### 3.1 Deriving the $\sigma\sqrt{n}$ null band

Consider the null hypothesis for a single category: its rows carry no category
effect, so each row's deviation is noise. Write row $i$'s residual as
$\varepsilon_i = y_i - \bar{y}_{node}$. Then

```math
D_{cat} = \sum_{i \in cat} \varepsilon_i
```

is a **sum** of $n_{cat}$ residuals, not an average.

**Proposition 3.1 (null scale of a category sum).** Assume, under the null of
no category effect: **(A2)** $\mathbb{E}[\varepsilon_i] = 0$; **(A3)** the
$\varepsilon_i$ are mutually independent within the category; **(A4)**
$\mathrm{Var}(\varepsilon_i) = \sigma^2 < \infty$, common across rows. Then

```math
\mathrm{Var}(D_{cat}) = \sum_{i \in cat}\mathrm{Var}(\varepsilon_i) = n_{cat}\,\sigma^2
\qquad\Longrightarrow\qquad
\mathrm{SD}(D_{cat}) = \sigma\sqrt{n_{cat}}.
```

*Proof.* Variance of a sum of independent terms is the sum of variances. ∎

So the natural scale of a *do-nothing* category grows like $\sqrt{n_{cat}}$. A
fixed threshold would be wrong in both directions: too permissive for large
categories (a 10,000-row category wanders far from zero on noise alone) and too
strict for small ones. Setting the bar at $z\sigma\sqrt{n_{cat}}$ makes it a
constant number of null standard deviations regardless of category size, which
is exactly the invariance we want. This is also the reason the bar is *not*
$z\sigma n_{cat}$: a real category effect of size $\mu$ per row contributes
$n_{cat}\mu$ to $D_{cat}$, growing like $n$ while the noise grows like
$\sqrt{n}$ — so genuine effects separate from the band as $n$ increases, which
is the whole point.

**Where the assumptions bite.** Independence (A3) is the load-bearing one. If
rows within a category are positively correlated — repeated measurements on the
same customer, the same store on consecutive days, an unmodelled time trend —
then $\mathrm{Var}(D_{cat}) = n\sigma^2 + \sum_{i \neq j}\mathrm{Cov}$ exceeds
$n\sigma^2$, the true null band is wider than $\sigma\sqrt{n}$, and the sieve
becomes anti-conservative: it will route categories that are only clustered
noise. There is no correction for this in the library. On clustered data the
honest response is to raise `noise_z` or to aggregate to the cluster level
before fitting. Assumption (A4), a common $\sigma$ across the feature's
categories, is likewise not tested; under strong heteroscedasticity the pooled
$\hat{\sigma}_f$ is too large for the quiet categories and too small for the
loud ones.

One structural dependence is always present and is *not* a defect:
$\sum_{cat} D_{cat} = 0$ by construction (§2), so the per-category statistics are
negatively dependent, and the size of that effect is computable. Because
$\bar{y}_{node}$ is the node's own sample mean, $D_{cat}$ can be written as a
linear form in the node's rows with weight $1 - n_{cat}/n_{node}$ inside the
category and $-n_{cat}/n_{node}$ outside it. For i.i.d. rows this gives

**Proposition 3.2 (exact null variance under mean estimation).**

```math
\mathrm{Var}(D_{cat}) = n_{cat}\,\sigma^2\Big(1 - \tfrac{n_{cat}}{n_{node}}\Big)
```

so the shipped bar $\sigma\sqrt{n_{cat}}$ overstates the true null standard
deviation by a factor $\big(1 - n_{cat}/n_{node}\big)^{-1/2}$. That is 5% for a
category holding a tenth of the node, **41% for a category holding half of it**,
and unbounded as $n_{cat} \to n_{node}$. The direction is conservative — the bar
is too high, never too low — but "slight" is the wrong word for a category that
dominates its node, and this is worth knowing when reading a near-miss at a
node with few categories.

> **Exact form (not implemented).** The shipped significance bar uses
> $\hat{\sigma}_f\sqrt{n_{cat}}$; Proposition 3.2 says the exact null standard
> deviation is
>
> ```math
> \mathrm{SD}(D_{cat}) = \sigma\sqrt{n_{cat}\Big(1 - \tfrac{n_{cat}}{n_{node}}\Big)}
> ```
>
> — the same $(1-f)$ finite-population correction that appears in the variance
> of a sample mean under simple random sampling without replacement (Cochran
> 1977), with $f = n_{cat}/n_{node}$ the category's share of its node. An
> algorithm pass that adopts it would multiply the shipped bar by
> $\sqrt{1-f}$, tightening it most exactly where the current bar is most
> conservative: categories that dominate their node. Nothing else in the sieve
> changes.

### 3.2 The robust scale $\hat{\sigma}_f = 1.4826 \cdot \mathrm{MAD}$

$\sigma$ is unknown and must be estimated from the data being tested — which
means the estimator must not be inflated by the very effects it is used to
detect. The implementation therefore computes residuals **against each
category's own mean**, for the feature under evaluation:

```math
r_i = y^{c}_i - \bar{y}^{c}_{cat(i)},
\qquad
\hat{\sigma}_f = 1.4826 \cdot \mathrm{median}_i\big(\lvert r_i - \mathrm{median}(r)\rvert\big)
```

Removing the category means first is deliberate: a large genuine
between-category effect leaves $\hat{\sigma}_f$ untouched, because it has been
subtracted out. Had residuals been taken against the node mean, a real effect
would inflate $\hat{\sigma}_f$, raise the bar, and hide itself.

The constant $1.4826 \approx 1/\Phi^{-1}(0.75)$ makes the MAD a consistent
estimator of $\sigma$ **for Gaussian noise** — for $\varepsilon \sim
\mathcal{N}(0,\sigma^2)$ the population MAD is exactly $\sigma\Phi^{-1}(0.75)
\approx 0.6745\sigma$, and the sample MAD converges to it in probability,
$\mathrm{MAD}_n \xrightarrow{\;p\;} 0.6745\sigma$, so dividing by that constant
gives back $\sigma$ in the limit (Rousseeuw & Croux 1993). (Consistency is the
right claim here, not unbiasedness: at finite $n$, $\mathbb{E}[\mathrm{MAD}_n]$
is not exactly the population value.) The MAD is chosen over the sample
standard deviation because it has a breakdown point of 50% (Hampel 1974;
Huber 1981): on a KPI where a handful of rows carry outsized values — which is
the norm for revenue and claims — a few extremes would inflate an SD-based bar
enough to suppress every real finding. The price is honest to state: under non-Gaussian
noise the 1.4826 calibration no longer holds exactly, so `noise_z=3.0` is "three
robust scale units", not a certified 3-sigma Gaussian tail probability.

The MAD's known cost is efficiency: at the Gaussian its asymptotic relative
efficiency is only about 37%, and Rousseeuw & Croux (1993) construct
higher-efficiency robust scales ($S_n$, $Q_n$) with the same 50% breakdown.
Those are the natural upgrades if the significance bar ever proves too noisy
at small $n$; the library ships the MAD for its simplicity and its
zero-configuration robustness, not because it is optimal.

Two edge cases follow directly from the definition and are visible in the code
path. If more than half the residuals are identical, $\mathrm{MAD} = 0$, the
significance bar collapses to zero, and only the materiality bar remains — this
is the exact mechanism that makes the singleton guard in §4 necessary. And when
`noise_z=0` the significance bar is disabled entirely by construction and
$\tau_{cat} = V^{c}_{node}\cdot\mathrm{delta\_pct}$.

### 3.3 Why `max()` and not a sum or a product

The two bars encode two **separately necessary** conditions:

- *materiality* — the effect must be big enough to be worth a stakeholder's
  attention relative to the volume in front of them;
- *significance* — the effect must be big enough that noise alone would not
  produce it.

A finding that fails either one should not be routed. Formally this is an
**intersection–union test** (Berger 1982): the null hypothesis is a *union* —
"the effect is immaterial **or** it is within noise" — and the alternative is
the *intersection* of "material" and "significant". Berger's result is that if
each component condition is tested at level $\alpha$, the rule *reject only if
every component rejects* has level at most $\alpha$ overall — an IUT needs
**no** multiplicity correction, unlike the union-of-rejections situation in
§4. For two one-sided thresholds on the same statistic the intersection is
$\{D > a\} \cap \{D > b\} = \{D > \max(a,b)\}$ — so `max()` *is* the
conjunction, not an approximation of it, and inherits the component level.
Whichever bar is binding at that node does the work:
in a large shallow node the materiality bar dominates; in a deep, small node 1%
of a small excess volume falls below the noise band and the significance bar
takes over, which is what stops deep fragmentation.

A sum $a + b$ would be strictly more conservative than either condition requires
and would reject findings that are both material and significant. A product
$a \cdot b$ is worse than conservative — it is dimensionally incoherent: $a$ has
units of $y$ and $b$ has units of $y$, so $a\cdot b$ has units of $y^2$ and
cannot be compared to $D_{cat}$ at all. Rescaling the whole KPI by a constant
$c$ multiplies $D_{cat}$, $a$ and $b$ each by $c$ but the product by $c^2$,
which would make routing depend on whether the analyst measured in dollars or
thousands of dollars. `max()` is scale-equivariant; the product is not.

**Enforced by:** `tests/test_impact_splitter.py::test_no_split_when_all_category_sums_within_delta`
(materiality bar alone can veto every category and force `stop_reason="no_split"`),
`tests/test_consolidation.py::test_consolidation_is_null_safe` (2,000 rows of pure
Gaussian noise over 3 four-level features: the sieve abstains entirely and returns
a single root segment).

**The null-safe test shows the significance bar is load-bearing, not decorative.**
Its numbers are worth reading, because on this data the materiality bar alone
would have split on noise. With $y \sim \mathcal{N}(0,1)$ over 2,000 rows,
$V^{c}_{root} = \sum_i \lvert y_i - \bar{y}\rvert \approx 2000\cdot\mathbb{E}\lvert
Z\rvert \approx 1{,}600$, so the materiality bar is $0.01 \times 1600 \approx 16$.
Each of the 12 categories (3 features × 4 levels) holds about 500 rows out of
2,000, so $f = n_{cat}/n_{node} \approx 1/4$ and, by §3.1, a do-nothing
category's excess wanders on the *corrected* null scale
$\sigma\sqrt{500\,(1-\tfrac14)} = \sigma\sqrt{375} \approx 19.4$ — still
*larger than the materiality bar* of 16. (The shipped significance bar
conservatively uses the uncorrected $\sqrt{n_{cat}} = \sqrt{500} \approx 22.4$,
per §3.1.) On the test's actual seed 5 of the 12 categories exceed 16 in
absolute excess, and one feature carries both a $+20.1$ and a $-24.5$ category:
a materiality-only sieve would route them to opposite branches and return a
split of pure noise. What forces abstention is the significance bar,
$\tau = 3\hat{\sigma}_f\sqrt{500} \approx 68$, which no category comes close to. So this test is direct evidence for the $\sigma\sqrt{n}$
term specifically, not merely an end-to-end null check.

*Scope note:* what the test does not do is vary $n_{cat}$, so the $\sqrt{n_{cat}}$
*scaling* — as opposed to the presence of a $\sigma\sqrt{n}$-sized bar at one
node size — is still not isolated by any unit test. The end-to-end evidence that
the two-bar sieve controls false positives across configurations is the benchmark
null case (`benchmarks/dgp.py::case_null`), recorded as "Null FP-control 3/3" in
[`reports/validation-report-v3.md`](../../reports/validation-report-v3.md) §1.

---

## 4. Multiplicity correction on the lookahead rescue

The pairwise lookahead rescue (v0.2.0) fires only at a node that is material but
where every marginal category table nets to ≈ 0 — the XOR signature, where the
sign of $y$ depends on a *combination* of features and each feature's own table
cancels. It re-runs the same two-bar sieve over the crossed categories of a
feature pair $(f, g)$.

**Why the marginal bar cannot be reused unchanged.** The marginal sieve
evaluates a feature's categories; the rescue evaluates $K$ crossed cells
simultaneously, and $K$ can reach the hundreds or thousands on real
high-cardinality pairs. A per-cell bar calibrated to a single comparison is the
classic multiple-comparisons error: with $K$ independent tests each at
false-positive rate $\alpha$, the chance that *at least one* fires is
$1 - (1-\alpha)^K \approx K\alpha$ for small $\alpha$. At $K = 100$ and a
nominal per-cell $\alpha$, the family-wise rate is roughly 100× the per-cell
rate — so a rescue that "finds an interaction" is, at large $K$, mostly finding
the maximum of a lot of noise.

**The correction.** The relevant statistic is not any one cell but the *maximum*
over cells. For $K$ standard normals $Z_1,\dots,Z_K$, the classical Gaussian
maximal inequality gives (Boucheron, Lugosi & Massart 2013)

```math
\mathbb{E}\Big[\max_{1\le j\le K} Z_j\Big] \le \sqrt{2\ln K}
```

which follows from the standard MGF argument: for any $t>0$, Jensen gives
$e^{t\,\mathbb{E}[\max_j Z_j]} \le \mathbb{E}[e^{t\max_j Z_j}] =
\mathbb{E}[\max_j e^{tZ_j}] \le \sum_j \mathbb{E}[e^{tZ_j}] = K e^{t^2/2}$, so
$\mathbb{E}[\max_j Z_j] \le \frac{\ln K}{t} + \frac{t}{2}$, minimised at
$t = \sqrt{2\ln K}$ to give $\sqrt{2\ln K}$. In words: searching $K$ cells buys
you roughly $\sqrt{2\ln K}$ standard deviations of apparent signal for free.

Note what that argument does *not* use. The step $\mathbb{E}[\max_j e^{tZ_j}]
\le \sum_j \mathbb{E}[e^{tZ_j}]$ is a bound on a maximum by a sum, and
$\mathbb{E}[e^{tZ_j}] = e^{t^2/2}$ is a statement about each $Z_j$ on its own.
Only standard-normal **marginals** are required; the bound holds under arbitrary
dependence among the $Z_j$. This matters here, because the cross-cells are
manifestly dependent — they partition the node's rows and the centering of §2
constrains their sums — and that dependence costs the bound nothing.

The per-cell $z$ must therefore carry that term:

```math
z_{eff} = \mathrm{noise\_z} + \sqrt{2\ln K},
\qquad
\tau_{cell} = \max\big(V^{c}_{node}\cdot\mathrm{delta\_pct},\ z_{eff}\cdot\hat{\sigma}\cdot\sqrt{n_{cell}}\big)
```

where $K$ is the number of **present** cross-cells (empty cells are not tests
and do not count against the budget). The additive form is deliberate and
conservative: $\sqrt{2\ln K}$ upper-bounds the expectation of the null maximum,
and `noise_z` is then retained as a buffer *on top of* that bound rather than
being replaced by it.
The growth is gentle, which is the desired behaviour — a genuine 2×2 XOR pays
$\sqrt{2\ln 4} \approx 1.67$, while a 100-cell cross pays
$\sqrt{2\ln 100} \approx 3.03$ and a 500-cell cross $\approx 3.53$. Small honest
interactions are barely taxed; wide fishing expeditions are taxed heavily.

**Assumptions, stated.** Two genuine gaps remain, and cross-cell dependence is
not one of them.

1. **Non-normality.** The bound is stated for standard-normal marginals, and the
   cell statistics here are not exactly normal — the scale is MAD-based (§3.2),
   so the standardisation is only approximate on non-Gaussian or heavy-tailed
   residuals.
2. **Expectation, not tail.** $\sqrt{2\ln K}$ upper-bounds the *expected* value
   of the null maximum. It therefore bounds the location of the null max from
   above, not a quantile of it, and does not by itself control family-wise error
   at any stated $\alpha$. A tail version of the same argument would carry
   roughly $\sqrt{2\ln(K/\alpha)}$ instead.

> **Exact form (not implemented).** Both gaps close at once with a
> tail-controlled per-cell threshold. Fix a family-wise error rate $\alpha$
> and test each of the $K$ cells two-sided at level $\alpha/K$ (Bonferroni;
> Dunn 1961):
>
> ```math
> z_{exact}(K,\alpha) = \Phi^{-1}\!\Big(1 - \frac{\alpha}{2K}\Big)
> ```
>
> By the union bound this controls the probability that *any* null cell
> clears the bar at $\alpha$, under **arbitrary dependence** among cells —
> the same dependence-freeness the shipped bound enjoys. (Šidák's
> $\Phi^{-1}\big(\tfrac{1}{2}(1 + (1-\alpha)^{1/K})\big)$ is marginally
> sharper but needs independence or positive dependence (Šidák 1967);
> at these $\alpha$ the two agree to two decimals, so Bonferroni is the
> right default.) Taking $\alpha = 2(1-\Phi(3)) \approx 0.0027$ — the
> two-sided tail the marginal sieve's `noise_z = 3` implies — gives:
>
> | $K$ | shipped $3 + \sqrt{2\ln K}$ | exact $z_{exact}$ |
> | --- | --- | --- |
> | 4 | 4.67 | 3.40 |
> | 100 | 6.03 | 4.20 |
> | 500 | 6.53 | 4.55 |
>
> The shipped additive form is therefore uniformly *stricter* than exact
> family-wise control at the sieve's own implied $\alpha$ — by 1.3 to 2
> robust-scale units. That is the right direction for a rescue path (a missed
> interaction, never a false one, per the paragraph below), but it quantifies
> what the conservatism costs: an algorithm pass that adopts $z_{exact}$
> would admit genuinely smaller interactions at large $K$ while holding an
> explicit, stated error rate — turning §4's "calibrated heuristic" into a
> guarantee. The caveat that survives the swap is non-normality: both forms
> price Gaussian tails, and the MAD standardisation (§3.2) is approximate on
> heavy-tailed residuals.

So $z_{eff}$ is a calibrated heuristic in the right functional form, not an exact
family-wise error guarantee. Gap 2 is the reason the form is additive rather
than a replacement, as noted above: `noise_z` rides on top of $\sqrt{2\ln K}$
because the multiplicity term supplies only an upper bound on that centre and
something has to stand in for the missing quantile. And the correction is used only to *raise* a bar that
would otherwise be uncorrected, at a node where the tree was about to give up —
so an over-strict bar costs a missed interaction, never a false one.

**Why singleton cells are excluded.** A cell with $n_{cell} = 1$ contributes one
observation. It supplies no within-cell noise estimate, and its "excess" is a
single residual, which is unfalsifiable evidence for an interaction. Worse, the
two mechanisms compound: when many cells are singletons the residuals against
cell means are all zero, so $\mathrm{MAD} = 0$ and $\hat{\sigma} = 0$ (§3.2),
collapsing the significance bar to nothing — and at small $K$ the multiplicity
term is too small to compensate. The implementation therefore requires
$n_{cell} \ge 2$ for a cell to count as sieve-clearing evidence. This is
belt-and-braces by design: the regression test for it documents that the case
*did* falsely fire under the multiplicity correction alone.

Two further guards are structural rather than statistical: pairs whose crossed
cardinality exceeds a hard memory bound are skipped outright, and 3-way or
higher cancellation whose every pairwise margin also cancels remains out of
reach — that mass is surfaced by the churn flag instead of being silently lost.

**Enforced by:** `tests/test_lookahead.py::test_rescue_multiplicity_floor_suppresses_chance_cells`
(a 10×10 cross over pure noise with one cell bumped by chance: the corrected bar
refuses it and the node leafs out),
`tests/test_lookahead.py::test_rescue_ignores_singleton_cells` (a 2×2 XOR of
singleton cells with huge $\lvert y\rvert$: no cell may count, node leafs out).

---

## 5. The gain metric

Among the features that survive routing, the split is chosen by

```math
Gain(X_i) = \frac{\lvert S_P\rvert}{k_P} + \frac{\lvert S_N\rvert}{k_N}
```

Read it as **average separated impact per actionable category**: the total
signed mass each outer branch pulls out, divided by how many categories it took
to pull it.

### 5.1 Why divide by $k$

Consider the undivided criterion $\lvert S_P\rvert + \lvert S_N\rvert$ and a
high-cardinality nuisance column — Customer ID, ZIP code, transaction ID, store
ID with 50 levels. Such a column has one structural advantage over every real
feature: it can place *each* row's excess into its own category, and a category
holding one row has nothing to average away.

$V^c_{node}$ is the ceiling on that criterion. Each category contributes
$\lvert D_{cat}\rvert = \lvert\sum_{i \in cat} y^c_i\rvert \le \sum_{i \in cat}
\lvert y^c_i\rvert$ by the triangle inequality, and the routed categories are
disjoint, so

**Proposition 5.1 (shattering attains the supremum of the undivided criterion).**

```math
\lvert S_P\rvert + \lvert S_N\rvert \;\le\; \sum_{cat}\lvert D_{cat}\rvert \;\le\; \sum_i \lvert y^c_i \rvert = V^c_{node}
```

with equality exactly when every category is sign-pure internally and every row
is routed to an outer branch — which is what shattering into singletons would
deliver, since a singleton category is trivially sign-pure. *Proof.* Triangle
inequality per category; disjointness of routed categories for the outer
inequality; equality requires every $\lvert D_{cat}\rvert$ to equal
$\sum_{i\in cat}\lvert y^c_i\rvert$ (sign-purity) and all rows routed.
Singletons satisfy both. ∎ So the undivided
criterion does not merely have a slight preference for high cardinality: its
**supremum** is attained by pure shattering, and no genuine feature can beat a
column that indexes the rows. This is the same pathology that information gain
has on ID columns, arriving through a different route — recognized since the
earliest tree induction work (Quinlan 1986, whose gain-ratio divides by an
attribute's own entropy), formally analysed by White & Liu (1994), and still
biting modern ensembles through variable-importance bias toward many-category
predictors (Strobl et al. 2007). $1/k$ is this method's answer to that
disease: like gain ratio, it normalizes the criterion by a function of the
attribute's cardinality — but by the count of categories *actually routed*,
not the count that exists.

**Two defenses, and which one binds depends on node size.** With one row per
category, each category's residual against its own mean is identically zero, so
$\mathrm{MAD} = 0$ and $\hat{\sigma}_f = 0$ — the §3.2 mechanism — which
collapses the significance bar and leaves
$\tau_{cat} = \mathrm{delta\_pct}\cdot V^{c}_{node} > 0$. A singleton row
therefore routes only if $\lvert y^{c}_i\rvert > 0.01\cdot V^{c}_{node}$: one row
carrying a hundredth of the node's entire excess. That bar scales with the
node's size, so it blocks shattering only where the node is large. Writing
$V^{c}_{node} \approx n_{node}\cdot\mathbb{E}\lvert y^c\rvert$, the bar is about
$0.01\,n_{node}$ average-magnitude units, prohibitive once
$n_{node} \gg 1/\mathrm{delta\_pct} \approx 100$ rows at the default — but there
is no `min_samples` guard, and with `max_depth=5` nodes of a few dozen rows are
ordinary. Below that size the materiality bar admits most singletons, they *are*
routed, the shattering candidate *is* scored, and $1/k$ is the operative
defense. A concrete node ($n = 30$, $y = (+1,-1)$ repeated, a unique-ID feature
beside an honest 2-level one) makes it explicit: the ID column routes all 30
rows ($k_P = k_N = 15$) and scores $Gain = 2.0$, while the honest column scores
$Gain = 30.0$ and wins — the ID column loses **because of $1/k$**, exactly the
adjudication that this section justifies. Only when the node is large enough for
the materiality bar to reject the singletons first does $1/k$ not have to
adjudicate. Both defenses are real; which one binds is a function of $n_{node}$.

What $1/k$ removes is the *criterion's* reward for cardinality among whatever
does clear that bar — a nuisance column with moderate cardinality, where cells
are big enough to be material but numerous enough to slice the node finely.
Dividing by $k$ turns each branch's score into a mean over the categories that
branch actually routed, so a column that spreads the same mass over more
categories scores lower. Two levels of comparison, with $k_P$ and $k_N$ read
throughout as the **per-branch** category counts:

- A balanced 2-category feature routes one category each way, $k_P = k_N = 1$
  and $\lvert S_P\rvert = \lvert S_N\rvert = V^c_{node}/2$, so
  $Gain = V^c_{node}/2 + V^c_{node}/2 = V^c_{node}$.
- A 50-level column splitting sign-purely down the middle has $k_P = k_N = 25$
  and the same branch masses, so $Gain = 2\cdot(V^c_{node}/2)/25 = V^c_{node}/25$.

The nuisance column scores $1/25$ of the honest one on identical separated mass.
The penalty is therefore not a tuned regularisation term with a coefficient to
choose but a direct consequence of scoring the average rather than the total.

There is a second, non-statistical reason for the same choice, and it is the
one that matters to the intended user: a split into 50 categories is not
actionable output. Scoring per-category impact selects for findings a person can
carry into a room.

### 5.2 Why only the outer branches enter the sum

The neutral branch is, by construction, the rows the sieve could **not**
distinguish from the node's own level. Crediting it would reward a feature for
the mass it failed to explain. Only $S_P$ and $S_N$ enter the sum, so the gain
measures separation achieved, not volume handled. The neutral branch still
recurses — it is a catch-all to be re-examined at greater depth, not a discard —
which is why the materiality bar is set relative to node excess volume rather
than dataset volume: effects trapped inside a large neutral pool stay detectable
after the pool is re-entered.

A guard sits alongside the metric: a candidate whose routing places *every* row
in one branch is not a split at all — it adds a level of depth without
partitioning anything — and is skipped before it can be scored, along with
features that are constant on the current slice.

**Enforced by:** `tests/test_impact_splitter.py::test_noop_routing_skips_feature_prefers_partitioning_column`
(a feature routing all its categories to one branch is passed over for the
column that actually partitions rows) and
`tests/test_impact_splitter.py::test_constant_feature_skipped_child_prefers_other_column`.

*Scope note:* those two tests establish the **degenerate-split guard**, not the
cardinality penalty. The evidence for the $1/k$ penalty is the benchmark case
`high_cardinality` (`benchmarks/dgp.py::case_high_cardinality`: the baseline
planted rules plus a 50-level nuisance `store_id` with no true effect), whose
scores are in [`reports/validation-report-v3.md`](../../reports/validation-report-v3.md).

---

## 6. The dual-pool stop

Splitting stops at a node that is globally immaterial. The pools are the
dataset's total positive and negative mass:

```math
V_{global\_P} = \sum_{y_i > 0} y_i
\qquad\text{and}\qquad
V_{global\_N} = \sum_{y_i < 0} \lvert y_i\rvert
```

and the node is graded on its own gross flows — $S^{node}_P = \sum_{i \in node,\,
y_i>0} y_i$ and $S^{node}_N = \sum_{i \in node,\, y_i<0}\lvert y_i\rvert$ —
against each pool separately:

```math
\text{Stop if } \left(\frac{S^{node}_P}{V_{global\_P}} \le \theta_{stop}\right)
\text{ \textbf{and} } \left(\frac{S^{node}_N}{V_{global\_N}} \le \theta_{stop}\right)
```

with $\theta_{stop} =$ `min_global_impact_pct`. Splitting continues if **either**
ratio clears the bar.

*Notation caution:* the $S_P$ / $S_N$ of §5 are the routed **outer-branch**
sums used to score a candidate split; the $S^{node}_P$ / $S^{node}_N$ here are
the node's own gross positive and negative flows, computed before any split is
considered. The README uses the shorter $S_P$ / $S_N$ in both places; they are
different quantities.

**Why two pools rather than one net criterion.** Suppose the rule were stated on
the net sum, $\lvert \Sigma_{node}\, y \rvert / V_{global} \le \theta_{stop}$.
Take a node holding $+8\%$ of all positive impact and $-8\%$ of all negative
impact. Its net is $\approx 0$, so a net-sum rule declares it immaterial and
stops — discarding a node that contains **16% of the total impact in the
dataset**, and precisely the node most worth investigating, since something
inside it is driving large flows in both directions. The cancellation is not an
edge case; it is what a node looks like when it contains an unsplit driver with
opposite-signed sub-populations, which is exactly the structure the method
exists to find.

Grading the two pools separately makes the stop rule immune to that
cancellation: a node survives if it is material in *either* direction, so
positive and negative mass can never net each other out of existence. Separate
denominators matter too — a dataset with $\$10\mathrm{M}$ of gains and
$\$200\mathrm{k}$ of losses would, under a single combined denominator, make
every loss-side node look immaterial by construction; against $V_{global\_N}$
the loss-side nodes are graded on the loss book they actually belong to.

The comparison is a **strict** inequality on the continue side (`ratio >
min_global_impact_pct`), so a node sitting exactly at the threshold does not
clear it and leafs out with `stop_reason="materiality"`.

The same two-pool logic reappears at the reporting layer as the **churn flag**:
a terminal segment whose positive *and* negative gross flows each clear
`min_global_impact_pct` against their own pools is marked `is_churn` and
rendered with both flows (`net +1 (gross +10,000 / −9,999)`) rather than as a
single near-zero net. Where the stop rule refuses to *discard* offsetting mass,
the churn flag refuses to *hide* it once no further split can separate it.

**Enforced by:** `tests/test_impact_splitter.py::test_materiality_uses_strict_greater_than`
(a node at exactly the threshold leafs out with `stop_reason="materiality"` and
both triggers false) and `tests/test_viz_data.py::test_payload_conservation_and_counts`
(segment sums and leaf sums each add back to the dataset total, with
`conservation_exact` true — the P/N/neutral branches tile the node, so nothing
is dropped by the stop rule).

---

## 7. Consolidation

The tree can only cut, never regroup. When it splits on one segment's features,
any *other* coherent segment is tiled across the resulting branches into
fragments that all behave identically. Post-fit consolidation repairs this
without touching the tree.

Two terminal segments merge when both conditions hold:

**(a) Structural — the union stays readable.** Their conditions must be
identical except on exactly one feature's category set, so the merged condition
is still a single conjunction of the form `feature=cat1,cat2 & ...`. Conditions
that grow to cover a feature's entire observed universe are vacuous and dropped.
Merging is iterated to fixpoint, which is what lets cross-product fragmentation
collapse: a segment merged along feature $f$ may then become mergeable along
feature $g$.

**(b) Statistical — the means are compatible.** A two-sample z-test against the
pooled robust scale:

```math
\lvert \bar{y}_1 - \bar{y}_2 \rvert \;\le\; z \cdot \hat{\sigma} \cdot \sqrt{\tfrac{1}{n_1} + \tfrac{1}{n_2}}
```

The $\sqrt{1/n_1 + 1/n_2}$ is the standard error of a difference of independent
means: if both segments' rows have per-row variance $\sigma^2$, then
$\mathrm{Var}(\bar{y}_1) = \sigma^2/n_1$, $\mathrm{Var}(\bar{y}_2) =
\sigma^2/n_2$, and by independence of the two disjoint row sets
$\mathrm{Var}(\bar{y}_1 - \bar{y}_2) = \sigma^2(1/n_1 + 1/n_2)$. The bar is $z$
of those standard errors, with $z =$ `noise_z` — the same constant as the sieve,
so a difference too small for the splitter to have acted on is also too small to
keep two segments apart. $\hat{\sigma} = 1.4826\cdot\mathrm{MAD}$ is pooled over
**within-segment** residuals across all segments, which assumes a common
residual scale; under strong heteroscedasticity the test is too permissive for
the quiet segments and too strict for the loud ones.

Merging leaf categories by a significance test is the CHAID primitive (Kass
1980) — there applied during growth, here applied after it, which is what
lets the tree stay greedy while the *report* recovers the coherent segments
the greedy cuts fractured.

Note the direction of the test. It merges on *failure to reject* the null of
equal means, which is not evidence of equality. This is the right default here
— the operation is a readability repair, and the cost of an over-merge (two
genuinely different segments reported as one) is bounded by the fact that the
full tree structure remains available in plots and traces — but it should not be
read as a claim that merged segments are proven identical.

> **Exact form (not implemented) — heteroscedasticity.** The pooled-scale
> assumption drops out by testing with per-segment scales (Welch 1947):
>
> ```math
> \lvert \bar{y}_1 - \bar{y}_2 \rvert \;\le\; z \cdot
> \sqrt{\tfrac{\hat{\sigma}_1^2}{n_1} + \tfrac{\hat{\sigma}_2^2}{n_2}},
> \qquad \hat{\sigma}_j = 1.4826\cdot\mathrm{MAD}(\text{segment } j\text{ residuals})
> ```
>
> which is the shipped test with $\hat{\sigma}^2(1/n_1+1/n_2)$ replaced by
> the Welch standard error. Quiet segments stop inheriting the loud
> segments' scale, at the cost of two scale estimates that are themselves
> noisier on small segments.

> **Exact form (not implemented) — proving mergeability.** Failure to reject
> equality is not evidence of equality; the test that *affirms* "these two
> segments are the same finding" is equivalence testing (TOST; Schuirmann
> 1987). Fix an equivalence margin $\delta$ — the largest mean difference
> that would still count as "the same segment", a materiality decision, not
> a statistical one — and merge only if both one-sided nulls are rejected:
>
> ```math
> \bar{y}_1 - \bar{y}_2 > -\delta + z_{1-\alpha}\,\mathrm{SE}
> \quad\text{and}\quad
> \bar{y}_1 - \bar{y}_2 < \delta - z_{1-\alpha}\,\mathrm{SE}
> ```
>
> equivalently: the $(1-2\alpha)$ confidence interval for
> $\bar{y}_1-\bar{y}_2$ lies inside $(-\delta, +\delta)$. This inverts the
> burden of proof — small segments with wide intervals then *fail* to merge
> rather than merging by default, the opposite of the shipped behaviour and
> the statistically honest one. The open design choice an algorithm pass
> would face is $\delta$; the natural candidate is a materiality-scaled
> margin (per-row excess at `delta_pct`), keeping the method's convention
> that statistical knobs are denominated in business units.

### 7.1 Conservation is structural, not checked

Sum conservation under merging needs no numerical guarantee, because it is true
by definition. Every routing decision assigns each row to exactly one of P, N or
neutral, so the leaf row sets **partition** the dataset and any two distinct
segments have disjoint row masks $M_1 \cap M_2 = \varnothing$. For disjoint sets,

```math
\Sigma_{M_1 \cup M_2}\, y \;=\; \sum_{i \in M_1} y_i + \sum_{i \in M_2} y_i \;=\; \Sigma_{M_1} y + \Sigma_{M_2} y
```

identically — the merged sum *is* the sum of the parts, and the merged count is
the sum of the counts, by the definition of a disjoint union. Each merge is
again a disjoint union of a partition's blocks, so the argument survives
iteration to fixpoint unchanged. The implementation adds `total_sum` and
`n_samples` directly and unions the masks; conservation would only be at risk if
leaf masks could overlap, which is what the partition test verifies.

**Where the float64 caveat actually lives.** The identity above is exact in
$\mathbb{R}$, and that is a real claim, but it is not the claim that the shipped
numbers agree bit-for-bit. A leaf's `total_sum` is `float(y_sub.sum())` over that
leaf's rows (`impact_split/splitter.py`), and floating-point addition is not
associative, so summing the leaf totals re-rounds along a different tree of
partial sums than `np.sum` over all rows does. The two results can differ in the
last bits. Counts are integers and are exact; only the sums carry this.

That is why the library's check is a relative assertion rather than a
reconciliation step: `conservation_exact` is
`abs(seg_total - total_sum) <= 1e-9 * max(1.0, abs(total_sum))`
(`impact_split/viz/data.py`), and the tests cited below likewise assert to
tolerance (`pytest.approx(abs=1e-9·…)`, `np.isclose`, `rtol=1e-6`). The
structural argument is what licenses that choice. Because conservation holds
identically over the reals, the only thing a numerical check can find is
accumulated rounding. For pairwise summation that error is bounded by roughly
$u\cdot\log_2(n)\cdot\sum_i\lvert y_i\rvert$ — machine epsilon $u$ times the
summation depth, but scaled by the *gross* magnitude $\sum\lvert y_i\rvert$, not
the net total. The shipped check normalises by $\max(1,\lvert\text{total\_sum}\rvert)$,
the *net* sum, so the two agree comfortably whenever the net total is an ordinary
fraction of the gross — which is the common case. The one regime where a genuine
partition can trip a 1e-9 relative tolerance is $\lvert\Sigma y\rvert \ll
\sum\lvert y_i\rvert$: the churn structure of §6 and the zero-centered targets of
§2, pushed to the extreme where the net is vanishing relative to the gross. There
the guard is measuring rounding against a near-zero denominator, and a failure
then means the tolerance was too tight for that data, not that the masks overlap.
Away from that corner, a failure does indicate overlapping masks rather than
arithmetic drift; the structural identity is what makes even the corner a
tolerance question rather than a conservation defect.

### 7.2 Null-safety

Consolidation cannot manufacture a finding. It only unions existing segments, so
it can reduce the segment count but never increase it, never move a row between
segments, and never create a segment from rows the tree did not already group.
On data with no structure the tree abstains at the root and returns one segment,
leaving consolidation a no-op. The failure mode it *can* have is over-merging
(condition (b) above), not invention.

**Enforced by:** `tests/test_consolidation.py::test_consolidation_preserves_conservation_and_partition`
(segment sums add back to $\Sigma y$, counts add back to $n$, and the segment
masks tile every row exactly once), `tests/test_consolidation.py::test_consolidation_is_null_safe`
(pure noise → a single segment), `tests/test_consolidation.py::test_incompatible_means_are_not_merged`
(four fully additive effects with distinct means: all four segments survive, so
the z-test is not merging everything in sight).

---

## 8. Ensemble annotations

The single greedy tree stays the answer. `ensemble_report()` fits a forest of
perturbed refits that *measures* that tree and never replaces it — there is no
prediction averaging anywhere in the module. Two blocks run:

- **Bootstrap block** — resample rows with replacement, refit, match the
  replicate's segments to the reference tree's segments. A segment's
  **stability** is the share of bootstrap replicates in which it re-emerged;
  below 0.5 it is flagged `fragile`.
- **Feature-subsampled block** — refit on a random subset of features, so
  dominant features are sometimes forced out. Segments that appear here but
  match nothing in the reference tree are **shadow** candidates.

Matching is one-to-one and greedy by descending Jaccard overlap above
`match_threshold`, and **same-sign only** — an opposite-signed region over the
same rows is a different finding, not a noisy version of the same one, so sign
gates candidacy before overlap is scored at all.

### 8.1 What the CI is conditioned on

Each segment's band is the 5th–95th percentile of its $\Sigma y$ **across the
replicates in which it was matched**, reported only when at least 10 such
matches exist (below that a percentile interval is itself noise, and the library
returns null rather than a number).

This conditioning is the caveat that must travel with the number. The CI is
computed over matched replicates only, so it is a distribution of
"$\Sigma y$ **given that the segment was rediscovered**", not the unconditional
sampling distribution of the segment's impact. For a **fragile** segment the
difference is material: the replicates in which it failed to re-emerge — often
the ones where the effect happened to resample weakly — contribute nothing to
the band, so the band is conditioned on rediscovery and **can understate
uncertainty**. Stability and CI must therefore be read together. A tight band on
a segment with stability 0.3 is not a precise estimate; it is a narrow summary of
the minority of resamples that happened to find the segment at all. A tight band
on a segment with stability 0.95 means what it appears to mean.

### 8.2 Why shadow candidates need the root-sieve gate

A bootstrap resample duplicates some rows and omits others. That reshaping alone
can let a replicate tree clear the two-bar sieve on a region whose excess, on
the *full* data, is indistinguishable from noise — the replicate is fitting its
own resampling artifact. Reporting such a region as a "shadow driver" would be
worse than reporting nothing, because a shadow arrives with the implicit claim
that the main tree *missed* something real.

The gate closes that path by re-testing every candidate on the full dataset with
the same statistic the root would use. With $M$ the candidate's full-data row
mask and $n_M = \lvert M \rvert$, the candidate must satisfy

```math
\Big\lvert \textstyle\sum_{i \in M} y_i - n_M\,\bar{y} \Big\rvert \;>\; \max\big(\mathrm{delta\_pct}\cdot V^{c}_{root},\ \mathrm{noise\_z}\cdot\hat{\sigma}_{full}\cdot\sqrt{n_M}\big)
```

which is the same **two-bar form** as §3 — centered excess against materiality
and a $\sigma\sqrt{n}$ null band — evaluated at the root rather than inside a
replicate, but with a **root-level scale** rather than §3.2's per-feature one.
The difference is worth naming precisely, because it is the one place the page
would otherwise overstate the match. §3.2's $\hat{\sigma}_f$ is the MAD of
residuals taken against each *category's own* mean; the gate's
$\hat{\sigma}_{full}$ is `1.4826 * median(|yc - median(yc)|)`
(`impact_split/ensemble.py`), which centres only on the global median and
removes no group means — precisely the estimator §3.2 argues against for the
splitter.

The direction of that discrepancy is safe here. Leaving genuine between-group
structure in the residuals inflates $\hat{\sigma}_{full}$, which raises the bar,
so the gate errs **strict**: it can refuse a real shadow, not admit a spurious
one. That is the correct way to be wrong for a gate whose whole purpose is
specificity — and it is the opposite of the splitter's situation in §3.2, where
an inflated scale would let a real effect hide itself. A candidate must
additionally be materially large against a global pool (§6) and must recur
across replicates above `shadow_min_stability` before it is reported. Discovery
is thus allowed to be noisy; the *report* is not.

**Enforced by:** `tests/test_ensemble.py::test_run_ensemble_stability_and_ci_on_planted_effect`
(a planted effect scores stability ≥ 0.8, is not fragile, and its CI brackets
the true planted total), `tests/test_ensemble.py::test_shadow_block_recovers_masked_driver`
(two real drivers with `max_depth=1` so the greedy tree can only report the
stronger one; the shadow block recovers the masked second driver with positive
mean impact), `tests/test_ensemble.py::test_no_shadow_on_control_without_secondary_effect`
(same setup with the second feature replaced by pure nuisance: no shadow is
reported — the gate's specificity, and the direct control for §8.2).

---

## 9. The constants that were not derived

Three defaults in this library are **empirical**. They were fixed by benchmark
loops over the validation suite, not obtained from any argument in the sections
above, and this page will not pretend otherwise.

| Constant | Default | Status |
| --- | --- | --- |
| `delta_pct` | `0.01` | **Empirical.** Tuned; no derivation. |
| `min_global_impact_pct` | `0.01` | **Empirical.** Tuned; no derivation. |
| `max_depth` | `5` | **Empirical.** Tuned; no derivation. |

The distinction that *is* defensible is between form and level. The sections
above derive the **functional form** of each rule from stated assumptions — the
$\sqrt{n_{cat}}$ scaling of the null band (§3.1), the $1.4826$ MAD constant
(§3.2, which is a Gaussian-consistency constant and genuinely derived), the
$1/k$ cardinality penalty (§5.1), the $\sqrt{2\ln K}$ multiplicity term (§4),
the $\sqrt{1/n_1 + 1/n_2}$ standard error (§7). None of those derivations picks
the numbers in the table. "1% of node excess volume" is not implied by anything;
it is a level that scored well and was kept.

What backs them is benchmark evidence rather than theory: an 8-case synthetic
battery plus 10 semi-synthetic Kaggle datasets, each with known planted driver
rules, run at three seeds and scored by impact-weighted F1. One fixed
configuration is used for every dataset, with no per-dataset tuning — which is
the property that makes the defaults meaningful as defaults rather than as a
fit to the benchmark. The floor loop's Phase 0 additionally ran a 27-config
hyperparameter sweep, and what it establishes should be stated exactly. No
configuration in the grid fixed the remaining floor cases: the best sweep
configuration reached a floor-case minimum of 0.806, short of the 0.85 bar
([`reports/validation-report-v3.md`](../../reports/validation-report-v3.md) §1).
The conclusion drawn was that the deficit was **structural rather than a
hyperparameter level** — which is what motivated consolidation, and what
consolidation then fixed.

Note what this does *not* show. Against the then-shipped floor of 0.781, that
best grid configuration was about 0.025 *better* on the floor metric, so the
sweep does not establish that the shipped defaults are a local optimum, and this
page will not claim they sit on a plateau rather than a knife edge. The sweep
speaks to the ceiling of hyperparameter tuning on those cases, not to the shape
of the objective around the shipped point. It is, if anything, further evidence
that these are tuning outcomes rather than theoretical optima. Method, per-case
results, the sweep, and the known-weakness analysis:
[`reports/validation-report-v3.md`](../../reports/validation-report-v3.md).

`noise_z = 3.0` sits between the two categories and is worth naming separately.
Its *role* is derived — it is the number of robust scale units in §3.1's null
band, and §3.3 explains why it is the binding bar in deep nodes — but the value
3.0 is a conventional choice, not a calibrated one. Because $\hat{\sigma}$ is a
MAD-based scale under non-Gaussian noise, "3" does not correspond to an exact
tail probability (§3.2). Treat it as a tunable strictness dial with a sensible
default: raise it on clustered or heavy-tailed data where §3.1's independence
assumption is doubtful, lower it only with a reason.

Practical consequence for a reader deciding whether to trust the defaults: they
are the right starting point because they were validated broadly and never
tuned per dataset, and they are the right thing to change first if a fit looks
too eager or too quiet on your data.
