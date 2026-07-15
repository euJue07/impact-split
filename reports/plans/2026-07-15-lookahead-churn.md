# Lookahead Rescue + Churn Visibility (v0.2.0) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Catch XOR-style offsetting contributors that v0.1.0 silently drops (pairwise lookahead rescue), flag irreducible ± churn, and surface gross flows in every renderer.

**Architecture:** The rescue is a new private pass in `ImpactSplitter._build` that fires only at the exact v0.1.0 silent-failure signature (material node, zero marginal gain, not at the interaction cap). It runs the *unchanged* two-bar sieve on crossed category pairs and converts the winner into an ordinary single-feature split, so conditions, consolidation, conservation, and path rendering are untouched. Churn flagging is computed from segment row masks already held during consolidation; renderers read new payload fields (`pos_sum`/`neg_sum`/`is_churn`/`n_churn_segments`).

**Tech Stack:** Python ≥3.10, numpy, pandas, matplotlib, pytest, ruff, mypy. No new dependencies.

**Spec:** `reports/specs/2026-07-15-lookahead-churn-design.md` (approved 2026-07-15). Read it before starting.

## Global Constraints

- Working directory for every command: `C:\Users\juedi.eugenio\Documents\ai-os\projects\impact-split` (its own git repo; commit there, never in ai-os).
- Invoke Python as `python` (not `python3`) on this machine.
- Happy-path fits must be **byte-identical** to v0.1.0: the rescue may only run when `best_decision is None or best_decision.gain == 0.0` at a material, below-cap node with ≥2 features.
- Hard benchmark bar: the existing 8-case synthetic battery and 10-dataset Kaggle suite must **match or exceed** the pre-change baseline recorded in Task 1 (published reference: 0.962 mean / 0.815 floor impact-F1 at fixed defaults). New lookahead cases: `xor_pure` mean F1 ≥ 0.90, `xor_embedded` ≥ 0.80, `churn_irreducible` must not split and must flag. If a bar fails, STOP and investigate — never lower a threshold to pass.
- New constructor param is `lookahead: bool = True`; validated exactly like `consolidate`.
- Cross-cardinality safety bound: module constant `_LOOKAHEAD_MAX_CROSS = 10_000` (allocated bincount size `(max_f+1)*(max_g+1)`).
- Churn rule (segments AND leaf nodes): `pos_sum / pos_pool > min_global_impact_pct` AND `neg_sum / neg_pool > min_global_impact_pct` (both pools must be > 0).
- Segment ranking key (breaking output change, documented in CHANGELOG): descending `max(|total_sum|, churn_mass)` with `churn_mass = min(pos_sum, neg_sum)` if churn else 0.
- Colors: only the existing palette (`POSITIVE_COLOR`/`NEGATIVE_COLOR`/`NEUTRAL_FILL`/`NEUTRAL_STROKE`). Never green/red.
- Keep lines ≤ 99 chars; before each commit run `python -m ruff check .` and fix findings.
- Run `git status` before each commit; stage only files this task touched.

---

### Task 1: Baseline benchmark snapshot + `lookahead` constructor param

**Files:**
- Modify: `impact_split/splitter.py` (constructor ~lines 165-204, `__repr__` ~lines 816-824)
- Test: `tests/test_lookahead.py` (create)

**Interfaces:**
- Consumes: nothing new.
- Produces: `ImpactSplitter(lookahead: bool = True)` and `self.lookahead: bool` — Tasks 2 and 4 read this attribute.

- [ ] **Step 1: Record the pre-change benchmark baseline (BEFORE any code edit)**

Run:
```
python -m benchmarks.run --tag v020-baseline-synthetic
python -m benchmarks.run --tag v020-baseline-kaggle --kaggle
```
Expected: both print `mean impact-F1` / `floor dataset` and save JSON under `benchmarks/results/`. Write the four numbers (synthetic mean+floor, Kaggle mean+floor) into a note; the Task 8 regression compares against them. If the Kaggle run fails (missing local data/creds), record synthetic only and flag it in the final report.

- [ ] **Step 2: Write the failing tests**

Create `tests/test_lookahead.py`:

```python
"""Tests for the pairwise lookahead rescue (v0.2.0)."""

from __future__ import annotations

import pytest

from impact_split import ImpactSplitter


def test_lookahead_constructor_validation() -> None:
    with pytest.raises(ValueError, match="lookahead"):
        ImpactSplitter(lookahead="yes")  # type: ignore[arg-type]


def test_lookahead_default_true_and_in_repr() -> None:
    model = ImpactSplitter()
    assert model.lookahead is True
    assert "lookahead=True" in repr(model)
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `python -m pytest tests/test_lookahead.py -v`
Expected: FAIL — `TypeError: __init__() got an unexpected keyword argument 'lookahead'` and `AttributeError`/assert failure.

- [ ] **Step 4: Implement the param**

In `impact_split/splitter.py`:

(a) Constructor signature — insert after `consolidate: bool = True,`:
```python
        lookahead: bool = True,
```

(b) Validation — directly after the existing `consolidate` check:
```python
        if not isinstance(lookahead, bool):
            raise ValueError("lookahead must be a bool.")
```

(c) Assignment — after `self.consolidate = consolidate`:
```python
        self.lookahead = lookahead
```

(d) `__repr__` unfitted branch: change the final fragment
`f"consolidate={self.consolidate})"` to
`f"consolidate={self.consolidate}, lookahead={self.lookahead})"`.

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_lookahead.py tests/test_viz_text.py -v`
Expected: PASS (test_viz_text's `test_repr_pre_and_post_fit` only checks the `ImpactSplitter(delta_pct=` prefix, so it still passes).

- [ ] **Step 6: Commit**

```bash
git add tests/test_lookahead.py impact_split/splitter.py
git commit -m "feat: add lookahead constructor param (validated bool, default True)"
```

---

### Task 2: Lookahead rescue engine in `_build`

**Files:**
- Modify: `impact_split/splitter.py` (module constant near line 17; `_build` no-split block at lines 531-536; new methods after `_build`, ~line 617)
- Test: `tests/test_lookahead.py`

**Interfaces:**
- Consumes: `self.lookahead` (Task 1); existing `_SplitDecision`, `y_centered`, `delta_centered`, `at_interaction_cap` locals in `_build`.
- Produces: `_lookahead_rescue(x_sub, y_centered, delta_centered) -> tuple[_SplitDecision, dict[str, Any]] | None` and `_lookahead_partition(feature_index, col_vals, profile, sig_rows, gain, n_samples) -> _SplitDecision | None`. Trace entries where the rescue fires carry `routing_mode == "lookahead_rescue"` and a `rescue` dict with keys `pair`, `pair_names`, `split_feature_index`, `gain`, `k_P`, `k_N`, `partition`, `pairs_evaluated`, `pairs_skipped_cardinality`. Task 8's benchmarks rely on the fitted behavior only.

- [ ] **Step 1: Write the failing tests**

In `tests/test_lookahead.py`, add `import numpy as np` to the imports (Task 1 deliberately left it out to keep ruff clean), then append:

```python
def _xor_arrays(n: int = 2000, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Pure 2-feature XOR: every marginal category nets ~0; only the cross sees signal."""
    rng = np.random.default_rng(seed)
    f0 = rng.integers(0, 2, size=n).astype(np.int64)
    f1 = rng.integers(0, 2, size=n).astype(np.int64)
    y = np.where((f0 ^ f1) == 1, 100.0, -100.0) + rng.normal(0, 1.0, n)
    return np.column_stack([f0, f1]), y


def test_rescue_fires_at_root_on_pure_xor() -> None:
    X, y = _xor_arrays()
    model = ImpactSplitter().fit(X, y, trace=True)
    root = model.fit_trace_[0]
    assert root["routing_mode"] == "lookahead_rescue"
    assert root["rescue"]["pair"] == [0, 1]
    payload = model.to_dict()
    assert payload["meta"]["n_segments"] == 4
    means = sorted(s["mean"] for s in payload["segments"])
    assert means[0] == pytest.approx(-100.0, abs=2.0)
    assert means[1] == pytest.approx(-100.0, abs=2.0)
    assert means[2] == pytest.approx(100.0, abs=2.0)
    assert means[3] == pytest.approx(100.0, abs=2.0)
    assert payload["meta"]["conservation_exact"] is True


def test_lookahead_false_reproduces_v010_silent_miss() -> None:
    X, y = _xor_arrays()
    model = ImpactSplitter(lookahead=False).fit(X, y, trace=True)
    root = model.fit_trace_[0]
    assert root["action"] == "leaf"
    assert root["stop_reason"] == "no_split"
    assert len(model.segments_) == 1


def test_rescue_respects_interaction_cap() -> None:
    # f0 is a legit marginal driver; the XOR pair sits one level down. With
    # max_depth=1 the children are at the cap, so the rescue must NOT fire
    # (a rescued split would add a forbidden interaction term).
    rng = np.random.default_rng(1)
    n = 4000
    f0 = rng.integers(0, 2, size=n).astype(np.int64)
    f1 = rng.integers(0, 2, size=n).astype(np.int64)
    f2 = rng.integers(0, 2, size=n).astype(np.int64)
    y = 50.0 * f0 + np.where((f1 ^ f2) == 1, 100.0, -100.0) + rng.normal(0, 1.0, n)
    X = np.column_stack([f0, f1, f2])
    model = ImpactSplitter(max_depth=1).fit(X, y, trace=True)
    assert not any(t.get("routing_mode") == "lookahead_rescue" for t in model.fit_trace_)


def test_rescue_leaves_irreducible_churn_alone() -> None:
    # Offsetting ±y independent of every feature: marginals AND crosses all
    # net ~0, so the rescue finds nothing and the node leafs out unchanged.
    rng = np.random.default_rng(2)
    n = 4000
    X = rng.integers(0, 2, size=(n, 2)).astype(np.int64)
    y = np.where(rng.random(n) < 0.5, 100.0, -99.0) + rng.normal(0, 1.0, n)
    model = ImpactSplitter().fit(X, y, trace=True)
    root = model.fit_trace_[0]
    assert root["routing_mode"] is None
    assert root["stop_reason"] == "no_split"
    assert len(model.segments_) == 1


def test_rescue_skips_pairs_over_cardinality_bound() -> None:
    # 150x150 categories -> crossed bincount would allocate 22,500 cells
    # (> _LOOKAHEAD_MAX_CROSS), so the only pair is skipped and the XOR is
    # (acceptably) missed: clean no_split leaf, no crash, no memory blowup.
    rng = np.random.default_rng(3)
    n = 6000
    f0 = rng.integers(0, 150, size=n).astype(np.int64)
    f1 = rng.integers(0, 150, size=n).astype(np.int64)
    y = np.where(((f0 % 2) ^ (f1 % 2)) == 1, 100.0, -100.0) + rng.normal(0, 1.0, n)
    X = np.column_stack([f0, f1])
    model = ImpactSplitter().fit(X, y, trace=True)
    root = model.fit_trace_[0]
    assert root["stop_reason"] == "no_split"
    assert len(model.segments_) == 1


def test_lookahead_partition_xor_correctness_and_degenerate_guards() -> None:
    # White-box: the profile partition on a crafted 2x2 XOR cross-table, plus
    # the two degenerate exits (same-sign rows; fewer than 2 signal rows).
    model = ImpactSplitter()
    col = np.array([0, 0, 1, 1], dtype=np.int64)
    xor_profile = np.array([[10.0, -10.0], [-10.0, 10.0]])
    d = model._lookahead_partition(0, col, xor_profile, np.array([0, 1]), 1.0, 4)
    assert d is not None
    assert d.mode == "lookahead_rescue"
    assert d.pos_categories.tolist() == [0]
    assert d.neg_categories.tolist() == [1]
    assert d.neu_categories.tolist() == []
    # Same-sign rows: no opposing group -> degenerate -> None.
    one_sided = np.array([[10.0, 0.0], [8.0, 0.0]])
    assert model._lookahead_partition(0, col, one_sided, np.array([0, 1]), 1.0, 4) is None
    # Only one row carries a sieve-clearing cell -> cannot partition -> None.
    assert model._lookahead_partition(0, col, xor_profile, np.array([0]), 1.0, 4) is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_lookahead.py -v`
Expected: the two Task-1 tests PASS; `test_rescue_fires_at_root_on_pure_xor` FAILS (routing_mode is None); `test_lookahead_false_reproduces_v010_silent_miss`, `test_rescue_respects_interaction_cap`, `test_rescue_leaves_irreducible_churn_alone`, `test_rescue_skips_pairs_over_cardinality_bound` may already pass (they assert v0.1.0 behavior) — that is fine; they are regression guards for after the change.

- [ ] **Step 3: Implement the rescue**

In `impact_split/splitter.py`:

(a) Module constant, directly under `_PATH_SEGMENT_MAX_LABELS = 8`:
```python
# Hard memory bound for the lookahead rescue's crossed-category bincount:
# feature pairs whose (max_f + 1) * (max_g + 1) allocation exceeds this are skipped.
_LOOKAHEAD_MAX_CROSS = 10_000
```

(b) Replace the no-split leaf block in `_build` (currently lines 531-536):
```python
        if best_decision is None or best_decision.gain == 0.0:
            trace_entry["action"] = "leaf"
            trace_entry["stop_reason"] = "max_depth" if at_interaction_cap else "no_split"
            if self._trace_enabled:
                self.fit_trace_.append(trace_entry)
            return _TreeNode(True, node_id, depth, n_samples, total_sum, path, s_node_p, s_node_n)
```
with:
```python
        rescue_info: dict[str, Any] | None = None
        if best_decision is None or best_decision.gain == 0.0:
            # Silent-failure signature: a materiality trigger fired (guaranteed —
            # the materiality leaf returned earlier), yet every marginal category
            # table nets ~0. Try the pairwise rescue, unless a rescued split
            # would add an interaction term past the cap.
            if self.lookahead and not at_interaction_cap and x_sub.shape[1] >= 2:
                rescued = self._lookahead_rescue(x_sub, y_centered, delta_centered)
                if rescued is not None:
                    best_decision, rescue_info = rescued

        if best_decision is None or best_decision.gain == 0.0:
            trace_entry["action"] = "leaf"
            trace_entry["stop_reason"] = "max_depth" if at_interaction_cap else "no_split"
            if self._trace_enabled:
                self.fit_trace_.append(trace_entry)
            return _TreeNode(True, node_id, depth, n_samples, total_sum, path, s_node_p, s_node_n)

        if rescue_info is not None:
            trace_entry["rescue"] = rescue_info
```
No other change in `_build`: `trace_entry["routing_mode"] = best_mode` (line ~550) picks up `"lookahead_rescue"` from the decision's `mode`, `category_tables_by_mode.get("lookahead_rescue", {})` correctly yields `{}`, and the child recursion / depth accounting below treats the rescue split as an ordinary single-feature split.

(c) New methods, placed immediately after `_build` (before `_render_conditions_path`):
```python
    def _lookahead_rescue(
        self,
        x_sub: np.ndarray,
        y_centered: np.ndarray,
        delta_centered: float,
    ) -> tuple[_SplitDecision, dict[str, Any]] | None:
        """Cross-feature sieve for XOR-style cancellation.

        Runs only at would-be no_split material nodes. For each feature pair,
        sums ``y_centered`` over crossed categories and applies the unchanged
        two-bar sieve (volume delta + MAD noise floor) plus the unchanged
        category-averaged gain to the cross-cells. The winning pair is realized
        as an ordinary single-feature split via ``_lookahead_partition`` so all
        downstream machinery (conditions, consolidation, conservation) is
        untouched.
        """
        n_samples = int(y_centered.shape[0])
        n_features = int(x_sub.shape[1])
        best: _SplitDecision | None = None
        best_gain = 0.0
        best_info: dict[str, Any] | None = None
        pairs_evaluated = 0
        pairs_skipped = 0

        for f in range(n_features):
            codes_f = x_sub[:, f]
            max_f = int(codes_f.max(initial=0))
            for g in range(f + 1, n_features):
                codes_g = x_sub[:, g]
                max_g = int(codes_g.max(initial=0))
                size = (max_f + 1) * (max_g + 1)
                if size > _LOOKAHEAD_MAX_CROSS:
                    pairs_skipped += 1
                    continue
                h = codes_f * (max_g + 1) + codes_g
                cross_signal = np.bincount(h, weights=y_centered, minlength=size)
                cross_counts = np.bincount(h, minlength=size)
                present = np.flatnonzero(cross_counts)
                if present.size <= 1:
                    continue
                pairs_evaluated += 1

                present_signal = cross_signal[present]
                present_counts = cross_counts[present]
                if self.noise_z > 0:
                    cross_means = np.zeros(size, dtype=float)
                    nz = cross_counts > 0
                    cross_means[nz] = cross_signal[nz] / cross_counts[nz]
                    resid = y_centered - cross_means[h]
                    med = float(np.median(resid))
                    sigma = 1.4826 * float(np.median(np.abs(resid - med)))
                    tau = np.maximum(
                        delta_centered, self.noise_z * sigma * np.sqrt(present_counts)
                    )
                else:
                    tau = np.full(present.shape[0], delta_centered)

                pos_mask = present_signal > tau
                neg_mask = present_signal < -tau
                k_p = int(pos_mask.sum())
                k_n = int(neg_mask.sum())
                s_p = float(present_signal[pos_mask].sum())
                s_n = float(present_signal[neg_mask].sum())
                gain = (abs(s_p) / k_p if k_p else 0.0) + (abs(s_n) / k_n if k_n else 0.0)
                if gain == 0.0 or gain <= best_gain:
                    continue

                sig_cells = present[pos_mask | neg_mask]
                profile = cross_signal.reshape(max_f + 1, max_g + 1)
                decision = self._lookahead_partition(
                    f, codes_f, profile, sig_cells // (max_g + 1), gain, n_samples
                )
                if decision is None:
                    decision = self._lookahead_partition(
                        g, codes_g, profile.T, sig_cells % (max_g + 1), gain, n_samples
                    )
                if decision is None:
                    continue
                best_gain = gain
                best = decision
                best_info = {
                    "pair": [f, g],
                    "pair_names": [
                        self._feature_display_name(f),
                        self._feature_display_name(g),
                    ],
                    "split_feature_index": decision.feature_index,
                    "gain": gain,
                    "k_P": k_p,
                    "k_N": k_n,
                    "partition": {
                        "positive": decision.pos_categories.tolist(),
                        "negative": decision.neg_categories.tolist(),
                        "neutral": decision.neu_categories.tolist(),
                    },
                }

        if best is None or best_info is None:
            return None
        best_info["pairs_evaluated"] = pairs_evaluated
        best_info["pairs_skipped_cardinality"] = pairs_skipped
        return best, best_info

    def _lookahead_partition(
        self,
        feature_index: int,
        col_vals: np.ndarray,
        profile: np.ndarray,
        sig_rows: np.ndarray,
        gain: float,
        n_samples: int,
    ) -> _SplitDecision | None:
        """Convert a winning cross-table into a single-feature split on ``feature_index``.

        Categories whose profile row holds no sieve-clearing cross-cell carry no
        signal (the spec's "near-zero row norm") and route neutral; the rest are
        partitioned by the sign of their profile row's dot product with the
        max-norm anchor row. Returns None when the induced split is degenerate
        (either signed group empty, or one branch holds every row).
        """
        present_rows = np.flatnonzero(np.bincount(col_vals, minlength=profile.shape[0]))
        carries = np.isin(present_rows, sig_rows)
        active = present_rows[carries]
        if active.size < 2:
            return None
        norms = np.linalg.norm(profile[active], axis=1)
        anchor_row = profile[active[int(np.argmax(norms))]]
        dots = profile[active] @ anchor_row
        pos = active[dots > 0]
        neg = active[dots < 0]
        neu = np.concatenate([present_rows[~carries], active[dots == 0]])
        if pos.size == 0 or neg.size == 0:
            return None
        row_p = np.isin(col_vals, pos)
        row_n = np.isin(col_vals, neg)
        row_u = ~(row_p | row_n)
        if (
            int(row_p.sum()) == n_samples
            or int(row_n.sum()) == n_samples
            or int(row_u.sum()) == n_samples
        ):
            return None
        return _SplitDecision(
            gain=gain,
            feature_index=int(feature_index),
            pos_categories=np.sort(pos).astype(np.int64, copy=False),
            neg_categories=np.sort(neg).astype(np.int64, copy=False),
            neu_categories=np.sort(neu).astype(np.int64, copy=False),
            mode="lookahead_rescue",
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_lookahead.py -v`
Expected: all PASS. `test_rescue_fires_at_root_on_pure_xor` now sees `routing_mode == "lookahead_rescue"`, a root split on f0 (or the symmetric conversion on f1), children splitting the other feature marginally, and 4 segments at ±100 means.

- [ ] **Step 5: Run the full suite to confirm nothing regressed**

Run: `python -m pytest tests/ -q`
Expected: all PASS (happy-path fits never reach the rescue).

- [ ] **Step 6: Lint and commit**

```bash
python -m ruff check .
git add tests/test_lookahead.py impact_split/splitter.py
git commit -m "feat: pairwise lookahead rescue for XOR-cancellation nodes"
```

---

### Task 3: Segment gross flows + churn flag

**Files:**
- Modify: `impact_split/splitter.py` (`_consolidate_segments` lines 664-761; new helper `_finalize_segments`)
- Test: `tests/test_churn.py` (create)

**Interfaces:**
- Consumes: segment dicts with `mask` from `_leaf_segments` / the consolidation loop; `self._v_global_p`, `self._v_global_n`, `self.min_global_impact_pct`.
- Produces: every dict in `model.segments_` gains float keys `pos_sum`, `neg_sum` and bool key `is_churn` (fit-only keys `mask`/`mean` still dropped). Task 4's payload builder reads exactly these three keys.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_churn.py`:

```python
"""Tests for segment gross flows and the churn flag (v0.2.0)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from impact_split import ImpactSplitter


def churn_frame(n: int = 400) -> tuple[pd.DataFrame, pd.Series]:
    """Identical feature rows carrying offsetting +100 / -99: irreducible churn."""
    rng = np.random.default_rng(5)
    X = pd.DataFrame({"a": ["x"] * n})
    y = np.where(np.arange(n) % 2 == 0, 100.0, -99.0) + rng.normal(0, 0.1, n)
    return X, pd.Series(y)


def test_single_churn_segment_carries_gross_flows_and_flag() -> None:
    X, y = churn_frame()
    model = ImpactSplitter().fit(X, y)
    (seg,) = model.segments_
    y_arr = y.to_numpy()
    assert seg["pos_sum"] == pytest.approx(float(y_arr[y_arr > 0].sum()))
    assert seg["neg_sum"] == pytest.approx(float(np.abs(y_arr[y_arr < 0]).sum()))
    assert seg["is_churn"] is True


def test_gross_flows_on_all_segments_and_flag_matches_rule() -> None:
    # Shattered-rule fixture from test_consolidation: exercises both the
    # consolidated path and (with consolidate=False) the plain-leaf path.
    rng = np.random.default_rng(3)
    n = 1200
    f0 = (rng.random(n) < 0.3).astype(np.int64)
    f1 = (rng.random(n) < 0.5).astype(np.int64)
    y = 5.0 * (f0 == 1) + 10.0 * ((f0 == 0) & (f1 == 1)) + rng.normal(0, 0.5, n)
    X = np.column_stack([f0, f1])
    pos_pool = float(y[y > 0].sum())
    neg_pool = float(np.abs(y[y < 0]).sum())
    for consolidate in (True, False):
        model = ImpactSplitter(consolidate=consolidate).fit(X, y)
        for seg in model.segments_:
            assert seg["pos_sum"] >= 0.0 and seg["neg_sum"] >= 0.0
            assert seg["pos_sum"] - seg["neg_sum"] == pytest.approx(
                seg["total_sum"], abs=1e-6
            )
            expected = (
                seg["pos_sum"] / pos_pool > model.min_global_impact_pct
                and seg["neg_sum"] / neg_pool > model.min_global_impact_pct
            )
            assert seg["is_churn"] is expected


def test_gross_flows_conserve_pools_exactly() -> None:
    X, y = churn_frame()
    model = ImpactSplitter().fit(X, y)
    y_arr = y.to_numpy()
    assert sum(s["pos_sum"] for s in model.segments_) == pytest.approx(
        float(y_arr[y_arr > 0].sum())
    )
    assert sum(s["neg_sum"] for s in model.segments_) == pytest.approx(
        float(np.abs(y_arr[y_arr < 0]).sum())
    )


def test_fit_only_keys_still_dropped() -> None:
    X, y = churn_frame()
    model = ImpactSplitter().fit(X, y)
    for seg in model.segments_:
        assert "mask" not in seg and "mean" not in seg
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_churn.py -v`
Expected: FAIL with `KeyError: 'pos_sum'`.

- [ ] **Step 3: Implement `_finalize_segments`**

In `impact_split/splitter.py`, add this method directly before `_consolidate_segments`:

```python
    def _finalize_segments(self, segs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Attach gross flows + churn flag from row masks, then drop fit-only fields.

        A segment is churn when its positive AND negative gross flows each clear
        ``min_global_impact_pct`` against their global pools — the net then hides
        offsetting material mass.
        """
        assert self._y is not None
        y = self._y
        for s in segs:
            ym = y[s["mask"]]
            pos = float(ym[ym > 0].sum())
            neg = float(np.abs(ym[ym < 0]).sum())
            s["pos_sum"] = pos
            s["neg_sum"] = neg
            ratio_p = pos / self._v_global_p if self._v_global_p > 0 else 0.0
            ratio_n = neg / self._v_global_n if self._v_global_n > 0 else 0.0
            s["is_churn"] = bool(
                ratio_p > self.min_global_impact_pct and ratio_n > self.min_global_impact_pct
            )
            s.pop("mask", None)
            s.pop("mean", None)
        return segs
```

Then rewire both exits of `_consolidate_segments`:

(a) Replace the early-return block
```python
        if not self.consolidate or len(segs) <= 1:
            for s in segs:
                s.pop("mask", None)
            return segs
```
with:
```python
        if not self.consolidate or len(segs) <= 1:
            return self._finalize_segments(segs)
```

(b) Replace the final block
```python
        for s in segs:
            s.pop("mask", None)
            s.pop("mean", None)
        return segs
```
with:
```python
        return self._finalize_segments(segs)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_churn.py tests/test_consolidation.py -v`
Expected: all PASS (consolidation behavior itself is unchanged; only the finalize step gained fields).

- [ ] **Step 5: Commit**

```bash
python -m ruff check .
git add tests/test_churn.py impact_split/splitter.py
git commit -m "feat: per-segment gross flows and dual-materiality churn flag"
```

---

### Task 4: Payload — new fields, churn-aware ranking, shared churn fixture

**Files:**
- Modify: `impact_split/viz/data.py`
- Modify: `tests/test_viz_data.py` (update sort assertion; add fixtures + tests)

**Interfaces:**
- Consumes: `seg["pos_sum"]`, `seg["neg_sum"]`, `seg["is_churn"]` (Task 3); `model.lookahead` (Task 1); existing node fields `s_node_p`/`s_node_n`.
- Produces (consumed by Tasks 5-7): payload `segments[*]` gain `pos_sum: float`, `neg_sum: float`, `is_churn: bool`; payload `tree[*]` gain `is_churn: bool` (true only on leaves); `meta` gains `n_churn_segments: int`; `meta.params` gains `lookahead: bool`; segment order = descending `max(|total_sum|, min(pos_sum, neg_sum) if is_churn else 0)`. Also test helpers `churn_mix_frame()` / `churn_mix_fitted()` in `tests/test_viz_data.py`, imported by the viz tests in Tasks 5-7.

- [ ] **Step 1: Update the existing sort assertion and add failing tests**

In `tests/test_viz_data.py`, replace the last two lines of `test_payload_tree_integrity`:
```python
    # segments sorted by |impact| descending
    mags = [abs(s["total_sum"]) for s in payload["segments"]]
    assert mags == sorted(mags, reverse=True)
```
with:
```python
    # segments sorted by max(|impact|, churn mass) descending
    keys = [
        max(
            abs(s["total_sum"]),
            min(s["pos_sum"], s["neg_sum"]) if s["is_churn"] else 0.0,
        )
        for s in payload["segments"]
    ]
    assert keys == sorted(keys, reverse=True)
```

Append to `tests/test_viz_data.py`:

```python
def churn_mix_frame() -> tuple[pd.DataFrame, pd.Series]:
    """One clean +200 segment (a=z) plus one ±(100/-99) churn segment (a=x)."""
    rng = np.random.default_rng(5)
    n = 1200
    a = np.where(rng.random(n) < 0.5, "x", "z")
    y = np.zeros(n)
    xmask = a == "x"
    y[xmask] = np.where(np.arange(int(xmask.sum())) % 2 == 0, 100.0, -99.0)
    y[~xmask] = 200.0
    y += rng.normal(0, 0.5, n)
    return pd.DataFrame({"a": a}), pd.Series(y)


def churn_mix_fitted() -> ImpactSplitter:
    X, y = churn_mix_frame()
    return ImpactSplitter().fit(X, y)


def test_payload_segment_gross_flows_and_churn() -> None:
    payload = churn_mix_fitted().to_dict()
    assert payload["meta"]["params"]["lookahead"] is True
    churn = [s for s in payload["segments"] if s["is_churn"]]
    assert len(churn) == 1
    assert payload["meta"]["n_churn_segments"] == 1
    seg = churn[0]
    assert seg["pos_sum"] > 0 and seg["neg_sum"] > 0
    assert seg["pos_sum"] - seg["neg_sum"] == pytest.approx(seg["total_sum"], abs=1e-6)
    churn_leaves = [n for n in payload["tree"] if n["is_churn"]]
    assert churn_leaves
    assert all(n["is_leaf"] for n in churn_leaves)


def test_payload_churn_fields_json_safe() -> None:
    payload = churn_mix_fitted().to_dict()
    parsed = json.loads(json.dumps(payload, allow_nan=False))
    seg = next(s for s in parsed["segments"] if s["is_churn"])
    assert isinstance(seg["is_churn"], bool)
    assert isinstance(parsed["meta"]["n_churn_segments"], int)
```

Note: `churn_mix_frame` is designed so the root splits `a` into P={z} / N={x} (x's centered excess is strongly negative), the x child leafs out as `identical_rows`, and the x segment nets ~+300 with gross flows ~±30,000 — a churn leaf inside a real tree, used by every renderer test.

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_viz_data.py -v`
Expected: the two new tests FAIL with `KeyError: 'is_churn'` / missing `lookahead`; `test_payload_tree_integrity` FAILS with `KeyError: 'pos_sum'` on segments.

- [ ] **Step 3: Implement the payload changes**

In `impact_split/viz/data.py`, inside `build_payload`:

(a) Replace
```python
    seg_sorted = sorted(model.segments_, key=lambda s: -abs(float(s["total_sum"])))
```
with:
```python
    min_pct = float(model.min_global_impact_pct)

    def churn_mass(seg: dict[str, Any]) -> float:
        if not seg["is_churn"]:
            return 0.0
        return min(float(seg["pos_sum"]), float(seg["neg_sum"]))

    # Churn segments rank by their offsetting mass, not their (misleading) net.
    seg_sorted = sorted(
        model.segments_,
        key=lambda s: -max(abs(float(s["total_sum"])), churn_mass(s)),
    )
```

(b) In the segment dict append (after `"pool_share": ...`):
```python
                "pos_sum": safe_float(seg["pos_sum"]),
                "neg_sum": safe_float(seg["neg_sum"]),
                "is_churn": bool(seg["is_churn"]),
```

(c) In `walk`, before `nodes.append(...)`:
```python
        node_pos = float(node.s_node_p)
        node_neg = float(node.s_node_n)
        node_churn = bool(
            is_leaf
            and pos_pool > 0
            and neg_pool > 0
            and node_pos / pos_pool > min_pct
            and node_neg / neg_pool > min_pct
        )
```
and in the node dict, after `"is_leaf": is_leaf,`:
```python
                "is_churn": node_churn,
```

(d) In `meta.params`, after `"consolidate": bool(model.consolidate),`:
```python
                "lookahead": bool(model.lookahead),
```

(e) In `meta`, after `"n_segments": len(segments),`:
```python
            "n_churn_segments": sum(1 for s in segments if s["is_churn"]),
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_viz_data.py tests/ -q`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
python -m ruff check .
git add tests/test_viz_data.py impact_split/viz/data.py
git commit -m "feat: payload gross flows, churn flag, and churn-aware segment ranking"
```

---

### Task 5: Static renderers — tornado gross band + icicle dashed churn leaves

**Files:**
- Modify: `impact_split/viz/static.py` (`plot_segments` lines 76-172, `plot_icicle` lines 210-304)
- Test: `tests/test_viz_static.py`

**Interfaces:**
- Consumes: payload fields from Task 4; `churn_mix_fitted` from `tests/test_viz_data.py`.
- Produces: no API change (same function signatures); churn segments render a hatched gross band `-neg_sum → +pos_sum` behind the net bar with label `net {…} (gross +{…} / −{…})`; churn leaves in the icicle get a dashed dark outline and a `{net} ⇄ ±{churn_mass}` second label line.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_viz_static.py`:

```python
from tests.test_viz_data import churn_mix_fitted


def test_plot_segments_churn_band_and_gross_label() -> None:
    fig = churn_mix_fitted().plot_segments(show=False)
    ax = fig.axes[0]
    texts = [t.get_text() for t in ax.texts]
    assert any(t.startswith("net ") and "gross +" in t for t in texts)
    # the hatched gross band is a real patch (no rolled-up bar in this fixture)
    hatched = [p for p in ax.patches if p.get_hatch()]
    assert hatched


def test_plot_tree_churn_leaf_dashed_outline() -> None:
    fig = churn_mix_fitted().plot_tree(show=False)
    ax = fig.axes[0]
    dashed = [p for p in ax.patches if p.get_linestyle() not in ("solid", "-")]
    assert dashed
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_viz_static.py -v`
Expected: the two new tests FAIL (no `net` label, no hatch, no dashed patch); existing tests PASS.

- [ ] **Step 3: Implement `plot_segments` changes**

In `impact_split/viz/static.py`:

(a) The rolled-up bar dict (line ~89) gains `"is_churn": False, "pos_sum": 0.0, "neg_sum": 0.0` alongside its existing keys.

(b) Replace
```python
    values = [float(b["total_sum"] or 0.0) for b in bars]
    max_abs = max((abs(v) for v in values), default=1.0) or 1.0
```
with:
```python
    values = [float(b["total_sum"] or 0.0) for b in bars]
    extents = list(values)
    for b in bars:
        if b.get("is_churn"):
            extents.append(float(b["pos_sum"]))
            extents.append(-float(b["neg_sum"]))
    max_abs = max((abs(v) for v in extents), default=1.0) or 1.0
```
and update the initial `ax.set_xlim(...)` call to use `extents` instead of `values` in both `min(...)` and `max(...)`.

(c) In the bar loop, replace
```python
        ax.barh(y, value, height=0.72, color=face, hatch=hatch, edgecolor="white", linewidth=0.8)
```
with:
```python
        is_churn = bool(b.get("is_churn"))
        if is_churn:
            gross_left = -float(b["neg_sum"])
            gross_width = float(b["pos_sum"]) + float(b["neg_sum"])
            ax.barh(
                y,
                gross_width,
                left=gross_left,
                height=0.72,
                facecolor="none",
                hatch="////",
                edgecolor=NEUTRAL_STROKE,
                linewidth=0.9,
                zorder=1,
            )
        ax.barh(
            y, value, height=0.72, color=face, hatch=hatch,
            edgecolor="white", linewidth=0.8, zorder=2,
        )
```

(d) Replace the annotation anchor/label logic
```python
        offset = pad if value >= 0 else -pad
        ha = "left" if value >= 0 else "right"
```
with:
```python
        anchor = float(b["pos_sum"]) if is_churn else value
        offset = pad if (value >= 0 or is_churn) else -pad
        ha = "left" if (value >= 0 or is_churn) else "right"
        if is_churn:
            value_label = (
                f"net {fmt_num(value, sign=True)} (gross +{fmt_num(b['pos_sum'])}"
                f" / −{fmt_num(b['neg_sum'])})"
            )
        else:
            value_label = fmt_num(value, sign=True)
```
and in the two `ax.text(...)` calls change the x argument from `value + offset` to `anchor + offset`, and the first text's content from `fmt_num(value, sign=True)` to `value_label`.

(e) Replace the footer `fig.text(...)` body with:
```python
    footer = (
        f"bars are additive: sum of all segments = total Σy ({conservation}) · "
        f"blue = positive impact · orange = negative"
    )
    if any(b.get("is_churn") for b in bars):
        footer += " · hatched band = churn segment's gross ±flows (band is not additive)"
    fig.text(0.01, 0.01, footer, fontsize=7.5, color=_MUTED_TEXT)
```

- [ ] **Step 4: Implement `plot_icicle` changes**

(a) In the rect loop, after the `merged_leaf = (...)` assignment, add:
```python
        churn_leaf = bool(node["is_leaf"] and node.get("is_churn"))
        edge_dark = merged_leaf or churn_leaf
```
and change the `Rectangle(...)` kwargs
```python
                edgecolor="#3a3a36" if merged_leaf else "white",
                linewidth=1.8 if merged_leaf else 1.1,
```
to:
```python
                edgecolor="#3a3a36" if edge_dark else "white",
                linewidth=1.8 if edge_dark else 1.1,
                linestyle=(0, (3, 2)) if churn_leaf else "solid",
```

(b) Replace
```python
        label = f"{node['condition']}\n{fmt_num(node['total_sum'], sign=True)}"
```
with:
```python
        if churn_leaf:
            mass = min(float(node["pos_sum"] or 0.0), float(node["neg_sum"] or 0.0))
            value_line = f"{fmt_num(node['total_sum'], sign=True)} ⇄ ±{fmt_num(mass)}"
        else:
            value_line = fmt_num(node["total_sum"], sign=True)
        label = f"{node['condition']}\n{value_line}"
```

(c) Extend the footer `fig.text` string from
```python
        "each row tiles the one above (children are their parent's rows) · "
        "dark-outlined leaves merged into one consolidated segment",
```
to:
```python
        "each row tiles the one above (children are their parent's rows) · "
        "dark-outlined leaves merged into one consolidated segment · "
        "dashed outline = churn leaf (net ⇄ ±offsetting mass)",
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_viz_static.py -v`
Expected: all PASS, including the pre-existing `test_plot_segments_annotation_text_stays_within_xlim` (the xlim-fit loop absorbs the longer churn label).

- [ ] **Step 6: Commit**

```bash
python -m ruff check .
git add tests/test_viz_static.py impact_split/viz/static.py
git commit -m "feat: churn-aware tornado gross band and icicle dashed outlines"
```

---

### Task 6: Text summary — churn marker, gross column, footnote

**Files:**
- Modify: `impact_split/viz/text.py` (`render_summary`)
- Test: `tests/test_viz_text.py`

**Interfaces:**
- Consumes: payload fields from Task 4; `churn_mix_fitted` from `tests/test_viz_data.py`.
- Produces: `render_summary` output gains a `gross ⇄` column (blank for non-churn rows), `lookahead=` in the params line, a `· N churn ⇄` note in the segments line, and a footnote when churn is present. Signature unchanged.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_viz_text.py`:

```python
from tests.test_viz_data import churn_mix_fitted


def test_summary_flags_churn_segments() -> None:
    text = churn_mix_fitted().summary()
    assert "lookahead=True" in text
    assert "churn ⇄" in text          # segments ledger line
    assert "gross ⇄" in text          # table column header
    assert " / -" in text             # gross column rendered for the churn row
    assert "offsetting mass" in text  # footnote


def test_summary_without_churn_has_no_footnote() -> None:
    # Strictly non-negative target: the negative pool is 0, so no segment can
    # ever flag churn. (Do NOT use fitted() here — its symmetric noise gives
    # the catch-all segments material gross flows in BOTH directions, which
    # correctly flags them as churn under the dual-pool rule.)
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"a": rng.choice(["x", "y"], size=200)})
    y = pd.Series(np.abs(rng.normal(0.0, 1.0, 200)) + (X["a"] == "x") * 5.0)
    text = ImpactSplitter().fit(X, y).summary()
    assert "offsetting mass" not in text
    assert "churn ⇄" not in text
```

Also add `import numpy as np` and `import pandas as pd` to the imports of `tests/test_viz_text.py` (they are not there yet).

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_viz_text.py -v`
Expected: `test_summary_flags_churn_segments` FAILS; the rest PASS.

- [ ] **Step 3: Implement**

In `impact_split/viz/text.py`, inside `render_summary`:

(a) Params line: change
```python
            f"params delta_pct={p['delta_pct']} noise_z={p['noise_z']} "
            f"max_depth={p['max_depth']} consolidate={p['consolidate']}"
```
to:
```python
            f"params delta_pct={p['delta_pct']} noise_z={p['noise_z']} "
            f"max_depth={p['max_depth']} consolidate={p['consolidate']} "
            f"lookahead={p['lookahead']}"
```

(b) Before the `lines = [` assignment add:
```python
    churn_note = (
        f" · {meta['n_churn_segments']} churn ⇄" if meta["n_churn_segments"] else ""
    )
```
and change the segments line to:
```python
        f"segments  {meta['n_segments']}{merged_note}{churn_note} · "
        f"conservation {conservation}",
```

(c) Table header: change
```python
        f" {'#':>2}  {'path':<{path_width}}  {'Σy':>14}  {'n':>9}  {'pool share':>16}",
```
to:
```python
        f" {'#':>2}  {'path':<{path_width}}  {'Σy':>14}  {'n':>9}"
        f"  {'pool share':>16}  {'gross ⇄':>22}",
```

(d) Row rendering: before the `lines.append(...)` inside the `for i, seg` loop add:
```python
        gross = (
            f"+{fmt_num(seg['pos_sum'])} / -{fmt_num(seg['neg_sum'])}"
            if seg["is_churn"]
            else ""
        )
```
and change the appended row string to:
```python
            f" {i:>2}  {path:<{path_width}}  {fmt_num(seg['total_sum'], sign=True):>14}"
            f"  {seg['n']:>9,}  {share:>16}  {gross:>22}"
```

(e) Rolled-up remainder row: append `f"  {'':>22}"` to its f-string.

(f) After the `if rest:` block, before `return`, add:
```python
    if meta["n_churn_segments"]:
        lines.append("")
        lines.append(
            " ⇄ churn segment: positive and negative flows are both material — "
            "the net hides offsetting mass (gross column shows both)."
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_viz_text.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
python -m ruff check .
git add tests/test_viz_text.py impact_split/viz/text.py
git commit -m "feat: churn marker, gross column, and footnote in summary()"
```

---

### Task 7: HTML report — churn tile, gross band, table columns, dashed icicle

**Files:**
- Modify: `impact_split/viz/html.py` (`render_html` + `_TEMPLATE`)
- Test: `tests/test_viz_html.py`

**Interfaces:**
- Consumes: payload fields from Task 4; `churn_mix_fitted` from `tests/test_viz_data.py`.
- Produces: HTML report shows churn everywhere the other renderers do. Report stays fully self-contained (no external references — the existing token test must keep passing).

- [ ] **Step 1: Write the failing test**

Append to `tests/test_viz_html.py`:

```python
from tests.test_viz_data import churn_mix_fitted


def test_html_marks_churn() -> None:
    html_out = churn_mix_fitted().to_html()
    assert "churn segments" in html_out   # ledger tile
    assert "lookahead=" in html_out       # params line
    assert "tband" in html_out            # gross-band CSS + tornado renderer
    assert "stroke-dasharray" in html_out # icicle churn outline
    assert "Σy⁺" in html_out              # gross table columns
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_viz_html.py -v`
Expected: `test_html_marks_churn` FAILS on `churn segments` / `tband`.

- [ ] **Step 3: Implement**

In `impact_split/viz/html.py`:

(a) `render_html` tiles list — append after the `("conservation", conservation),` entry:
```python
        ("churn segments", str(meta["n_churn_segments"])),
```

(b) `params_line` — change to:
```python
    params_line = escape(
        f"delta_pct={p['delta_pct']} · noise_z={p['noise_z']} · "
        f"max_depth={p['max_depth']} · consolidate={p['consolidate']} · "
        f"lookahead={p['lookahead']} · impact_split v{meta['package_version']}"
    )
```

(c) CSS — after the `.tbar.rolled { ... }` rule add:
```css
.tband { position:absolute; top:1px; height:14px; border:1px dashed #949494;
         border-radius:4px; background:repeating-linear-gradient(45deg, transparent,
         transparent 3px, rgba(148,148,148,0.35) 3px, rgba(148,148,148,0.35) 6px); }
.churn-mark { color:var(--muted); font-weight:600; }
```

(d) Tornado hint (`<p class="hint">` above `#tornado`) — append inside the paragraph:
` Churn segments (⇄) also show a hatched band spanning their gross ±flows — the band is not additive.`

(e) Icicle hint — append inside its paragraph:
` Dashed-outlined leaves are churn (offsetting ±flows both material).`

(f) `renderTornado` rows — change the `shown.map` callback to:
```js
  var rows = shown.map(function (s) {
    return { path: s.path, v: s.total_sum || 0, n: s.n, share: s.pool_share,
             seg: s.segment_id, rolled: false, churn: !!s.is_churn,
             pos: s.pos_sum || 0, neg: s.neg_sum || 0 };
  });
```
and the rolled-up push gains `churn: false, pos: 0, neg: 0`.

(g) Extents — replace
```js
  rows.forEach(function (r) { lo = Math.min(lo, r.v); hi = Math.max(hi, r.v); });
```
with:
```js
  rows.forEach(function (r) {
    lo = Math.min(lo, r.v); hi = Math.max(hi, r.v);
    if (r.churn) { lo = Math.min(lo, -r.neg); hi = Math.max(hi, r.pos); }
  });
```

(h) Row markup — inside the `rows.forEach` build, before the `out +=` statement add:
```js
    var band = "";
    if (r.churn) {
      var bLeft = (-r.neg - lo) / range * 100;
      var bWidth = (r.pos + r.neg) / range * 100;
      band = '<div class="tband" style="left:' + bLeft + "%;width:" + bWidth + '%"></div>';
    }
    if (r.churn) {
      note += " · gross +" + fmtMag(r.pos) + "/−" + fmtMag(r.neg);
    }
```
(move the existing `var note = ...` line above this block), insert `band` right after the `tzero` div:
```js
      '<div class="ttrack"><div class="tzero" style="left:' + zeroPct + '%"></div>' + band +
```
and change the value cell to `'<div class="tval">' + (r.churn ? "net " : "") + fmt(r.v) + "</div>"`.

(i) `nodeTip` — after the `Σy = ...` line push, add:
```js
  if (n.is_churn) {
    lines.push("churn ⇄ net hides ±" +
      fmtMag(Math.min(n.pos_sum || 0, n.neg_sum || 0)) + " offsetting mass");
  }
```

(j) `renderIcicle` — after `var merged = seg && seg.node_ids.length > 1;` add:
```js
    var churn = n.is_leaf && n.is_churn;
    var dark = merged || churn;
```
and change the rect attributes from
```js
      ' fill="' + col.css + '" stroke="' + (merged ? "#3a3a36" : "#ffffff") + '"' +
      ' stroke-width="' + (merged ? 2 : 1) + '" rx="3"></rect>';
```
to:
```js
      ' fill="' + col.css + '" stroke="' + (dark ? "#3a3a36" : "#ffffff") + '"' +
      ' stroke-width="' + (dark ? 2 : 1) + '"' +
      (churn ? ' stroke-dasharray="6,3"' : "") + ' rx="3"></rect>';
```

(k) Segment table — extend `COLS`:
```js
var COLS = [
  ["#", "rank"], ["path", "path"], ["Σy", "total_sum"], ["Σy⁺", "pos_sum"],
  ["Σy⁻", "neg_sum"], ["n", "n"], ["mean", "mean"], ["pool share", "pool_share"],
  ["leaves", "leaves"]
];
```
`tableRows` map gains `pos_sum: s.pos_sum || 0, neg_sum: s.neg_sum || 0, churn: !!s.is_churn,`. In the row template, change the path cell to:
```js
      '<td class="path">' + chip +
      (r.churn ? '<span class="churn-mark" title="churn: offsetting flows both material">⇄ </span>' : "") +
      esc(r.path) + "</td>" +
```
and insert after the `Σy` cell:
```js
      "<td>" + fmtMag(r.pos_sum) + "</td>" +
      "<td>" + (r.neg_sum ? "−" : "") + fmtMag(r.neg_sum) + "</td>" +
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_viz_html.py -v`
Expected: all PASS, including `test_html_is_fully_self_contained` (no new external references) and `test_html_escapes_script_closers_in_data`.

- [ ] **Step 5: Visual spot-check**

Run:
```
python -c "from tests.test_viz_data import churn_mix_fitted; churn_mix_fitted().to_html('reports/_churn_spotcheck.html')"
```
Open `reports/_churn_spotcheck.html` in a browser (or confirm it renders via the file size + a grep for `tband`). Verify: churn tile shows 1, the x-segment row shows the ⇄ marker and hatched band, the icicle x-leaf is dash-outlined. Then delete the file:
```
python -c "import os; os.remove('reports/_churn_spotcheck.html')"
```

- [ ] **Step 6: Commit**

```bash
python -m ruff check .
git add tests/test_viz_html.py impact_split/viz/html.py
git commit -m "feat: churn treatment in the interactive HTML report"
```

---

### Task 8: Benchmark cases, CLI flag, and full-suite regression

**Files:**
- Modify: `benchmarks/dgp.py` (three new factories + registry)
- Modify: `benchmarks/battery.py` (`run_lookahead_cases`)
- Modify: `benchmarks/run.py` (`--lookahead` flag)
- Test: `tests/test_lookahead_benchmarks.py` (create)

**Interfaces:**
- Consumes: `ImpactSplitter` behavior from Tasks 2-4; existing `_assemble`, `_mk_rules`, `BenchDataset`, `fit_and_score`, `DEFAULT_PARAMS`, `SEEDS`, `save_results`.
- Produces: `LOOKAHEAD_CASE_FACTORIES: dict[str, Callable[[int], BenchDataset]]` in `benchmarks/dgp.py`; `run_lookahead_cases(params=None, *, seeds=None) -> dict` in `benchmarks/battery.py`; `python -m benchmarks.run --lookahead --tag <tag>` CLI path. These are scored **separately** from the headline battery so the published 0.962/0.815 aggregates keep their meaning.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_lookahead_benchmarks.py`:

```python
"""Benchmark bars for the v0.2.0 lookahead/churn cases.

These thresholds are the spec's hard bar. If one fails, the design has a real
problem — investigate; never lower a threshold to pass.
"""

from __future__ import annotations

from benchmarks.battery import fit_and_score
from benchmarks.dgp import SEEDS, case_churn, case_xor, case_xor_embedded
from impact_split import ImpactSplitter


def test_xor_pure_recovered_across_seeds() -> None:
    for seed in SEEDS:
        score = fit_and_score(case_xor(seed))
        assert score.conservation_ok
        assert score.impact_f1 >= 0.90, f"seed {seed}: {score.impact_f1:.4f}"


def test_xor_pure_missed_without_lookahead() -> None:
    # Documents the v0.1.0 silent failure the rescue exists to fix.
    score = fit_and_score(case_xor(42), {"lookahead": False})
    assert score.impact_f1 < 0.5


def test_xor_embedded_recovered_across_seeds() -> None:
    for seed in SEEDS:
        score = fit_and_score(case_xor_embedded(seed))
        assert score.conservation_ok
        assert score.impact_f1 >= 0.80, f"seed {seed}: {score.impact_f1:.4f}"


def test_churn_case_not_split_and_flagged() -> None:
    for seed in SEEDS:
        ds = case_churn(seed)
        model = ImpactSplitter().fit(ds.X, ds.y)
        payload = model.to_dict()
        assert payload["meta"]["n_segments"] == 1
        assert payload["segments"][0]["is_churn"] is True
        assert payload["meta"]["n_churn_segments"] == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_lookahead_benchmarks.py -v`
Expected: FAIL with `ImportError: cannot import name 'case_xor'`.

- [ ] **Step 3: Add the DGP factories**

In `benchmarks/dgp.py`, after `case_null` and before `CASE_FACTORIES`:

```python
def case_xor(seed: int, *, n: int = 5000, amp: float = 120.0) -> BenchDataset:
    """Lookahead case 1 — pure 2-feature XOR: every marginal nets ~0; the rescue
    must fire at the root to recover the four signed cells."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        {
            "promo": rng.choice(["yes", "no"], size=n),
            "daytype": rng.choice(["weekday", "weekend"], size=n),
        }
    )
    xor = (X["promo"] == "yes") ^ (X["daytype"] == "weekend")
    specs = [
        ("promo XOR weekend (+)", xor, amp),
        ("promo XOR weekend (-)", ~xor, -amp),
    ]
    return _assemble("xor_pure", seed, X, _mk_rules(X, specs), 22.0, rng)


def case_xor_embedded(seed: int, *, n: int = 8000) -> BenchDataset:
    """Lookahead case 2 — a clean marginal rule plus an XOR pocket confined to
    one region; the rescue must fire at an interior node, not the root."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        {
            "region": rng.choice(
                ["NCR", "Luzon", "Visayas", "Mindanao"], size=n, p=[0.35, 0.3, 0.2, 0.15]
            ),
            "promo": rng.choice(["yes", "no"], size=n),
            "daytype": rng.choice(["weekday", "weekend"], size=n),
        }
    )
    xor = (X["promo"] == "yes") ^ (X["daytype"] == "weekend")
    ncr = X["region"] == "NCR"
    specs = [
        ("Visayas uplift", X["region"] == "Visayas", 80.0),
        ("NCR x XOR (+)", ncr & xor, 150.0),
        ("NCR x XOR (-)", ncr & ~xor, -150.0),
    ]
    return _assemble("xor_embedded", seed, X, _mk_rules(X, specs), 22.0, rng)


def case_churn(seed: int, *, n: int = 4000, amp: float = 100.0) -> BenchDataset:
    """Lookahead case 3 — irreducible ±churn independent of every feature: the
    tree must NOT split (scored via the null-case machinery) and the single
    segment must be flagged churn (asserted in tests)."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        {
            "region": rng.choice(["NCR", "Luzon"], size=n),
            "channel": rng.choice(["Direct", "Online"], size=n),
        }
    )
    base = np.where(rng.random(n) < 0.5, amp, -amp + 1.0)  # +100 / -99: nets ~0
    y = base + rng.normal(0, 2.0, n)
    return BenchDataset("churn_irreducible", seed, X, y, [], 2.0, {"amp": amp})
```

After `CASE_FACTORIES` (leave it untouched), add:

```python
# v0.2.0 lookahead/churn cases — scored separately from the headline battery so
# the published mean/floor aggregates keep their meaning.
LOOKAHEAD_CASE_FACTORIES: dict[str, Callable[[int], BenchDataset]] = {
    "xor_pure": case_xor,
    "xor_embedded": case_xor_embedded,
    "churn_irreducible": case_churn,
}
```

- [ ] **Step 4: Add `run_lookahead_cases` to `benchmarks/battery.py`**

Extend the `.dgp` import line with `LOOKAHEAD_CASE_FACTORIES`, then add after `run_battery`:

```python
def run_lookahead_cases(
    params: dict[str, Any] | None = None,
    *,
    seeds: list[int] | None = None,
) -> dict[str, Any]:
    """Score the v0.2.0 lookahead/churn cases (separate from headline aggregates)."""
    seeds = seeds or SEEDS
    results: list[dict[str, Any]] = []
    for case, factory in LOOKAHEAD_CASE_FACTORIES.items():
        for seed in seeds:
            ds = factory(seed)
            score = fit_and_score(ds, params)
            results.append(asdict(score))

    scored = [r for r in results if r["case"] != "churn_irreducible"]
    churn = [r for r in results if r["case"] == "churn_irreducible"]
    per_case: dict[str, float] = {}
    for case in LOOKAHEAD_CASE_FACTORIES:
        vals = [r["impact_f1"] for r in scored if r["case"] == case]
        if vals:
            per_case[case] = float(np.mean(vals))
    return {
        "params": {**DEFAULT_PARAMS, **(params or {})},
        "per_case_mean_f1": per_case,
        "churn_null_pass_rate": (
            float(np.mean([r["null_pass"] for r in churn])) if churn else None
        ),
        "conservation_all_ok": bool(all(r["conservation_ok"] for r in results)),
        "results": results,
    }
```

- [ ] **Step 5: Add the CLI flag to `benchmarks/run.py`**

Extend the import to include `run_lookahead_cases`, add the argument:
```python
    ap.add_argument(
        "--lookahead", action="store_true", help="run the v0.2.0 lookahead/churn cases"
    )
```
and insert before the `if args.kaggle:` block:
```python
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
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `python -m pytest tests/test_lookahead_benchmarks.py -v`
Expected: all PASS. If an F1 bar fails, debug the rescue (fit the failing seed with `trace=True`, inspect the root/interior trace entries) before touching thresholds.

- [ ] **Step 7: Full-suite regression against the Task 1 baseline**

Run:
```
python -m benchmarks.run --tag v020-post-synthetic
python -m benchmarks.run --tag v020-post-kaggle --kaggle
python -m benchmarks.run --tag v020-lookahead --lookahead
python -m pytest tests/ -q
```
Expected: synthetic and Kaggle mean/floor **match or exceed** the Task 1 baseline (reference bar 0.962/0.815); all tests pass. Improvements are possible (the rescue may fire on previously-abandoned material nodes); any decrease is a stop-and-investigate.

- [ ] **Step 8: Churn-prevalence spot-check (record, do not "fix")**

Run:
```
python -c "from benchmarks.dgp import CASE_FACTORIES; from impact_split import ImpactSplitter; [print(n, (lambda m: (m['n_segments'], m['n_churn_segments']))(ImpactSplitter().fit(f(42).X, f(42).y).to_dict()['meta'])) for n, f in CASE_FACTORIES.items()]"
```
Record segment vs churn counts per case in the final report to the user. Known design consequence to watch: on one-sided cases with symmetric noise (e.g. `one_sided`), the per-sign-pool churn rule can flag noise-driven negative flows as churn. This matches the approved spec — report the observed counts as input for a possible v0.2.1 tightening; do NOT change the rule in this plan.

- [ ] **Step 9: Commit**

```bash
python -m ruff check .
git add tests/test_lookahead_benchmarks.py benchmarks/dgp.py benchmarks/battery.py benchmarks/run.py
git commit -m "feat: lookahead/churn benchmark cases and --lookahead CLI"
git add benchmarks/results/
git commit -m "chore: v0.2.0 benchmark results (baseline + post + lookahead)"
```
(If `benchmarks/results/` is gitignored, skip the second commit and keep the numbers in the final report instead.)

---

### Task 9: Version 0.2.0, CHANGELOG, README

**Files:**
- Modify: `pyproject.toml` (line 7)
- Modify: `CHANGELOG.md`
- Modify: `README.md` (Core Idea bullets ~line 38-45; new subsection after Act I ~line 78; implementation-notes defaults line ~147; constructor snippet ~line 234; `plot_segments` bullet ~line 248)

**Interfaces:**
- Consumes: everything shipped in Tasks 1-8.
- Produces: released-shape v0.2.0 metadata and docs. No code changes.

- [ ] **Step 1: Bump the version**

In `pyproject.toml` change `version = "0.1.0"` to `version = "0.2.0"`.

- [ ] **Step 2: Add the CHANGELOG section**

Insert above `## [0.1.0] - Unreleased` in `CHANGELOG.md`:

```markdown
## [0.2.0] - Unreleased

### Added
- **Pairwise lookahead rescue** (`lookahead=True`, new constructor param): when a
  material node's marginal category tables all net to ~0 (XOR-style interaction
  cancellation), a cross-feature pass re-runs the unchanged two-bar sieve on
  crossed category pairs and converts the winning pair into an ordinary
  single-feature split. Fires only where v0.1.0 silently gave up
  (`stop_reason="no_split"` with materiality triggers on), so happy-path fits are
  unchanged. Trace entries record `routing_mode="lookahead_rescue"` plus a
  `rescue` sub-dict (pair, gain, partition, pairs evaluated/skipped).
- **Churn flag**: segments and leaf nodes whose positive AND negative gross flows
  each clear `min_global_impact_pct` against their global pools are marked
  `is_churn`. Segments now carry `pos_sum` / `neg_sum`; the payload meta gains
  `n_churn_segments`; `meta.params` gains `lookahead`.
- Churn-aware rendering: tornado hatched gross-range band with
  `net … (gross +…/−…)` labels, icicle dashed churn outline with `net ⇄ ±mass`
  labels, `summary()` gross column + footnote, HTML report churn tile, band,
  ⇄ marker, and Σy⁺/Σy⁻ table columns.
- Benchmarks: three lookahead cases (`xor_pure`, `xor_embedded`,
  `churn_irreducible`) scored separately from the headline battery
  (`python -m benchmarks.run --lookahead --tag <tag>`); pytest bars in
  `tests/test_lookahead_benchmarks.py`.

### Changed (output ordering)
- Segment ranking key is now `max(|Σy|, churn_mass)` where
  `churn_mass = min(Σy⁺, Σy⁻)` for churn segments and 0 otherwise: churn
  segments surface by their offsetting mass instead of sinking on their ~0 net.
  Orderings without churn segments are unchanged.
```

- [ ] **Step 3: README — Core Idea bullet**

After the two-part-sieve bullet (line ~41) insert:

```markdown
- a pairwise lookahead rescue that catches XOR-style interaction cancellation (offsetting contributors whose every marginal table nets to ~0) by re-running the same sieve on crossed category pairs — plus a churn flag for offsetting mass that genuinely cannot be split,
```

- [ ] **Step 4: README — Act I extension subsection**

Insert after the Act I "Why it works" paragraph (line ~77), before `### Act II`:

```markdown
#### Act I extension: the pairwise lookahead rescue (v0.2.0)

**Problem:** when the *sign* of $y$ depends on a combination of features
(XOR-style interaction), every marginal category table nets to ~0 and the sieve
finds nothing — even though the node's positive and negative flows are both
material. In v0.1.0 the node silently leafed out (`stop_reason="no_split"`),
e.g. +10,000 and −9,999 reported as one ~0 segment.

**Rescue:** exactly at that signature (materiality triggers fired, best marginal
gain 0, interaction cap not reached, `lookahead=True`), the same sieve re-runs
over **crossed categories** of each feature pair $(f, g)$ — same $\tau$
(materiality + noise floor), same gain metric. The winning pair is realized as
an ordinary single-feature split: $f$'s categories are partitioned by the sign
of their cross-profile row's dot product with the max-norm anchor row
(categories carrying no sieve-clearing cell stay neutral), and the children then
split on $g$ through the normal marginal sieve. Happy-path fits are unchanged —
the rescue only runs where the tree was about to give up. Pairs whose crossed
cardinality exceeds a safety bound are skipped, and 3-way-or-higher cancellation
whose pairwise margins all cancel remains out of reach — the churn flag below
still surfaces it.

**Churn flag:** offsetting mass that no split can separate (identical feature
rows with ±y, higher-order interactions) is flagged instead of vanishing: a
segment whose positive and negative gross flows *each* clear
`min_global_impact_pct` against their global pools is marked `is_churn`, carries
`pos_sum`/`neg_sum`, ranks by `max(|Σy|, min(Σy⁺, Σy⁻))`, and every renderer
shows the gross flows (`net +1 (gross +10,000 / −9,999)`).
```

- [ ] **Step 5: README — defaults, constructor snippet, output suite**

(a) Implementation-notes defaults line (~147): change `consolidate=True)` to
`consolidate=True, lookahead=True)` and append to the sentence: `; lookahead=True was added in v0.2.0 and validated on dedicated XOR/churn cases without moving the headline suite.`

(b) Constructor snippet (~line 234): after the `consolidate=True,` line add:
```python
    lookahead=True,          # pairwise rescue for XOR-style marginal cancellation
```

(c) `plot_segments` bullet (~line 248): append the sentence:
`Churn segments (both gross flows material) additionally show a hatched band spanning −Σy⁻ to +Σy⁺ with a net + gross label.`

- [ ] **Step 6: Verify docs and full suite**

Run:
```
python -m pytest tests/ -q
python -m ruff check .
```
Expected: all PASS, no lint findings. Skim README rendering for broken math fences.

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml CHANGELOG.md README.md
git commit -m "docs: v0.2.0 — lookahead rescue + churn visibility"
```

- [ ] **Step 8: Push (established precedent: JuediEugenio pushes as collaborator on euJue07/impact-split)**

```bash
git push
```
If the push is rejected or anything about the remote state looks unexpected, stop and report instead of forcing.

---

## Verification summary (what "done" means)

1. `python -m pytest tests/ -q` — green.
2. `python -m benchmarks.run --tag v020-post-synthetic` and `--kaggle` — mean/floor ≥ Task 1 baseline (reference 0.962/0.815).
3. `python -m benchmarks.run --lookahead --tag v020-lookahead` — `xor_pure` ≥ 0.90, `xor_embedded` ≥ 0.80, churn null-pass 1.0, conservation all ok.
4. `python -m ruff check .` — clean.
5. Report to the user: baseline vs post benchmark numbers, lookahead-case scores, and the Task 8 churn-prevalence counts per battery case (spec-adjustment input for v0.2.1).
