# impact-split v0.1.0 Package + Viz Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `impact_split` a PyPI-release-ready library and replace its three outputs (box-and-arrow tree, force graph, raw DataFrame print) with a coordinated suite — `summary()`, tornado, icicle, self-contained HTML — all driven by one `model.to_dict()` payload.

**Architecture:** A new `impact_split/viz/` subpackage: `data.py` builds a JSON-safe payload from the fitted model; `text.py`, `static.py`, `html.py` render that payload. `splitter.py` gains thin delegate methods and loses ~450 lines of plot layout. Spec: `docs/superpowers/specs/2026-07-15-package-and-viz-redesign-design.md`.

**Tech Stack:** Python ≥3.10, numpy/pandas/matplotlib only, flit build, pytest, vanilla JS + SVG-via-innerHTML in the HTML export (no d3, no CDN).

## Global Constraints

- Runtime dependencies exactly: `numpy`, `pandas`, `matplotlib`. Nothing else.
- `requires-python = ">=3.10"`; no 3.11+ syntax (no `Self`, no `tomllib`, no `except*`).
- **Zero algorithm changes.** Never touch `fit`, `_build`, `_consolidate_segments`, `_leaf_segments`, `_prepare_X_y`.
- `get_impact_segments()` existing columns keep name, order, and values: `path`, `total_sum`, `n_samples`, `node_id`. New columns append after.
- Colors (validated with dataviz six-checks, light surface): positive `#0173B2`, negative `#C6660A`, diverging midpoint `#F2F2F0`, neutral stroke `#949494`. Positive is ALWAYS blue, negative ALWAYS orange, in every renderer.
- HTML export: zero external requests — no `http://`, `https://`, `src=`, `url(`, `@import` anywhere in the output.
- Every renderer raises `RuntimeError` mentioning `fit()` when called pre-fit.
- All work in `projects/impact-split` on branch `package-and-viz`. Run commands from the repo root. Python is `python` (not `python3`).
- Repo-wide gates that must stay green after every task: `python -m pytest -q`, `python -m ruff format --check .`, `python -m ruff check .`, `python -m mypy impact_split`.

---

### Task 0: Branch

- [ ] **Step 0.1:**

```bash
cd projects/impact-split
git checkout -b package-and-viz
```

---

### Task 1: Packaging cleanup — remove CCDS scaffold, trim metadata

**Files:**
- Delete: `impact_split/config.py`, `impact_split/features.py`, `impact_split/dataset.py`, `impact_split/modeling/` (whole dir)
- Modify: `impact_split/__init__.py`, `pyproject.toml`
- Create: `CHANGELOG.md`

**Interfaces:**
- Produces: `impact_split.__init__` no longer exports `config`; package deps = numpy/pandas/matplotlib (+ the still-present `plots.py` needs no extra deps — its ipykernel/ipywidgets imports are guarded try/except and stay until Task 6 deletes the file).

- [ ] **Step 1.1: Delete scaffold**

```bash
git rm impact_split/config.py impact_split/features.py impact_split/dataset.py -r impact_split/modeling
```

- [ ] **Step 1.2: Replace `impact_split/__init__.py` entirely with:**

```python
from impact_split.plots import InteractiveForceGraph, interactive_force_graph
from impact_split.splitter import ImpactSplitter

__all__ = ["ImpactSplitter", "InteractiveForceGraph", "interactive_force_graph"]
```

- [ ] **Step 1.3: Edit `pyproject.toml`**

Set `version = "0.1.0"`; `requires-python = ">=3.10"`; dependencies:

```toml
dependencies = [
    "matplotlib>=3.8",
    "numpy>=1.26",
    "pandas>=2.1",
]
```

Classifiers: replace the single `3.13` entry with `3.10`, `3.11`, `3.12`, `3.13` lines (keep the bare `Programming Language :: Python :: 3`). In `[tool.mypy]` set `python_version = "3.10"` and trim the overrides module list to `["matplotlib.*", "ipykernel.*", "IPython.*", "ipywidgets.*"]` (loguru/tqdm/typer gone).

- [ ] **Step 1.4: Create `CHANGELOG.md`**

```markdown
# Changelog

All notable changes to this project are documented here.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versioning: [SemVer](https://semver.org/).

## [0.1.0] - Unreleased

### Added
- `ImpactSplitter.to_dict()` — JSON-safe payload (meta / tree / segments) feeding every renderer.
- `ImpactSplitter.summary()` and informative `print(model)` — text report with ledger and ranked segment table.
- `ImpactSplitter.plot_segments()` — tornado chart of consolidated segments (matplotlib).
- `ImpactSplitter.plot_tree()` — impact icicle (width ∝ Σ|y|, diverging impact color), replacing the box-and-arrow tree.
- `ImpactSplitter.to_html()` — self-contained interactive HTML report (no CDN, works offline).
- `get_impact_segments()` gains `mean` and `pool_share` columns (existing columns unchanged).

### Removed (breaking)
- CCDS template scaffolding (`impact_split.config`, `features`, `dataset`, `modeling`) and the
  `loguru` / `typer` / `tqdm` / `python-dotenv` dependencies.
- `interactive_force_graph` / `InteractiveForceGraph` (generic, CDN-dependent, not model-bound).
- Old box-and-arrow `plot_tree` and its layout keyword arguments.

### Changed
- `requires-python` widened from `~=3.13.0` to `>=3.10`; CI tests 3.10–3.13.
```

- [ ] **Step 1.5: Verify**

```bash
python -m pytest -q          # expected: 34 passed (nothing imported the scaffold)
python -m pip install -e . --quiet && python -c "import impact_split; print(impact_split.ImpactSplitter)"
python -m ruff check . && python -m ruff format --check . && python -m mypy impact_split
```

- [ ] **Step 1.6: Commit**

```bash
git add -A && git commit -m "chore(pkg): remove CCDS scaffold, trim deps to numpy/pandas/matplotlib, widen to py>=3.10, v0.1.0"
```

---

### Task 2: `viz/data.py` — payload builder + `to_dict()` + segment columns

**Files:**
- Create: `impact_split/viz/__init__.py`, `impact_split/viz/data.py`, `tests/test_viz_data.py`
- Modify: `impact_split/splitter.py` (add `to_dict`; extend `get_impact_segments`)

**Interfaces:**
- Produces: `build_payload(model) -> dict` with keys `meta`/`tree`/`segments` (shapes below); helpers `safe_float(v) -> float|None`, `fmt_num(v, *, sign=False) -> str`, `fmt_pct(v) -> str`; color constants `POSITIVE_COLOR`, `NEGATIVE_COLOR`, `NEUTRAL_FILL`, `NEUTRAL_STROKE`. `ImpactSplitter.to_dict()` delegates to `build_payload`. Later tasks consume the payload dict only — never the model internals.

- [ ] **Step 2.1: Write failing tests — `tests/test_viz_data.py`:**

```python
"""Tests for the shared renderer payload (to_dict) and segment columns."""

import json

import numpy as np
import pandas as pd
import pytest

from impact_split import ImpactSplitter


def demo_frame() -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(7)
    n = 400
    region = rng.choice(["north", "south", "east"], size=n)
    product = rng.choice(["basic", "plus"], size=n)
    y = rng.normal(0.0, 1.0, size=n)
    y[(region == "north") & (product == "plus")] += 8.0
    y[region == "east"] -= 6.0
    return pd.DataFrame({"region": region, "product": product}), pd.Series(y)


def fitted() -> ImpactSplitter:
    X, y = demo_frame()
    return ImpactSplitter().fit(X, y)


def test_to_dict_requires_fit() -> None:
    with pytest.raises(RuntimeError, match="fit\\(\\)"):
        ImpactSplitter().to_dict()


def test_payload_conservation_and_counts() -> None:
    model = fitted()
    payload = model.to_dict()
    total = payload["meta"]["total_sum"]
    seg_sum = sum(s["total_sum"] for s in payload["segments"])
    leaf_sum = sum(n["total_sum"] for n in payload["tree"] if n["is_leaf"])
    assert seg_sum == pytest.approx(total, abs=1e-9 * max(1.0, abs(total)))
    assert leaf_sum == pytest.approx(total, abs=1e-9 * max(1.0, abs(total)))
    assert payload["meta"]["conservation_exact"] is True
    assert payload["meta"]["n_nodes"] == len(payload["tree"])
    assert payload["meta"]["n_leaves"] == sum(1 for n in payload["tree"] if n["is_leaf"])
    assert payload["meta"]["n_segments"] == len(payload["segments"])


def test_payload_tree_integrity() -> None:
    payload = fitted().to_dict()
    ids = [n["id"] for n in payload["tree"]]
    assert len(ids) == len(set(ids))
    id_set = set(ids)
    root = payload["tree"][0]
    assert root["parent_id"] is None and root["branch"] == "root"
    assert root["condition"] == "all data"
    for n in payload["tree"][1:]:
        assert n["parent_id"] in id_set
        assert n["branch"] in {"positive", "neutral", "negative"}
    leaf_ids = {n["id"] for n in payload["tree"] if n["is_leaf"]}
    for n in payload["tree"]:
        assert (n["segment_id"] is not None) == n["is_leaf"]
    for s in payload["segments"]:
        assert set(s["node_ids"]) <= leaf_ids
    # segments sorted by |impact| descending
    mags = [abs(s["total_sum"]) for s in payload["segments"]]
    assert mags == sorted(mags, reverse=True)


def test_payload_json_safe() -> None:
    payload = fitted().to_dict()
    text = json.dumps(payload, allow_nan=False)
    assert json.loads(text)["meta"]["n_rows"] == 400


def test_get_impact_segments_gains_columns_after_existing() -> None:
    df = fitted().get_impact_segments()
    assert list(df.columns) == ["path", "total_sum", "n_samples", "node_id", "mean", "pool_share"]
    assert (df["mean"] == df["total_sum"] / df["n_samples"]).all()
    assert (df["pool_share"].dropna() >= 0).all()
```

- [ ] **Step 2.2: Run — expect FAIL** (`AttributeError: ... no attribute 'to_dict'`):

```bash
python -m pytest tests/test_viz_data.py -q
```

- [ ] **Step 2.3: Create `impact_split/viz/__init__.py`:**

```python
"""Renderers for fitted ImpactSplitter payloads (text, static figures, HTML)."""
```

- [ ] **Step 2.4: Create `impact_split/viz/data.py`:**

```python
"""Payload builder: one JSON-safe dict that feeds every renderer."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from impact_split.splitter import ImpactSplitter, _TreeNode

# Diverging pair validated against the dataviz six-checks (light surface):
# blue = positive, orange = negative, neutral gray midpoint. Never green/red.
POSITIVE_COLOR = "#0173B2"
NEGATIVE_COLOR = "#C6660A"
NEUTRAL_FILL = "#F2F2F0"
NEUTRAL_STROKE = "#949494"


def safe_float(value: Any) -> float | None:
    """Finite float or None — NaN/inf never reach a renderer or JSON."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def fmt_num(value: float | None, *, sign: bool = False) -> str:
    """Human number: thousands separators; decimals only when small."""
    if value is None:
        return "—"
    prefix = "+" if sign and value > 0 else ""
    if abs(value) >= 1000:
        return f"{prefix}{value:,.0f}"
    if abs(value) >= 1:
        return f"{prefix}{value:,.2f}"
    return f"{prefix}{value:.4g}"


def fmt_pct(value: float | None) -> str:
    return "—" if value is None else f"{100.0 * value:.1f}%"


def _package_version() -> str:
    try:
        from importlib.metadata import version

        return version("impact_split")
    except Exception:  # pragma: no cover - metadata absent in odd envs
        return "unknown"


def build_payload(model: ImpactSplitter) -> dict[str, Any]:
    """Serialize a fitted model into the shared ``meta``/``tree``/``segments`` payload."""
    tree = model._tree
    if tree is None:
        raise RuntimeError("Call fit() before to_dict() or any renderer.")

    pos_pool = float(model._v_global_p)
    neg_pool = float(model._v_global_n)

    seg_sorted = sorted(model.segments_, key=lambda s: -abs(float(s["total_sum"])))
    segments: list[dict[str, Any]] = []
    node_to_segment: dict[str, str] = {}
    for rank, seg in enumerate(seg_sorted):
        seg_id = f"s{rank}"
        total = float(seg["total_sum"])
        n = int(seg["n_samples"])
        pool = pos_pool if total >= 0 else neg_pool
        segments.append(
            {
                "segment_id": seg_id,
                "path": seg["path"],
                "node_ids": list(seg["node_ids"]),
                "n": n,
                "total_sum": safe_float(total),
                "mean": safe_float(total / n) if n else None,
                "pool_share": safe_float(abs(total) / pool) if pool > 0 else None,
            }
        )
        for nid in seg["node_ids"]:
            node_to_segment[nid] = seg_id

    nodes: list[dict[str, Any]] = []
    stats = {"leaves": 0, "depth": 0, "inter_depth": 0}

    def condition_of(node: _TreeNode) -> str:
        if not node.path or node.path == "root":
            return "all data"
        parts = [p.strip() for p in node.path.split(" / ") if p.strip()]
        return parts[-1] if parts else "all data"

    def walk(
        node: _TreeNode,
        parent_id: str | None,
        branch: str,
        inter_depth: int,
        parent_feature: int | None,
    ) -> None:
        is_leaf = node.is_leaf or not node.children
        stats["depth"] = max(stats["depth"], node.depth)
        stats["inter_depth"] = max(stats["inter_depth"], inter_depth)
        if is_leaf:
            stats["leaves"] += 1
        split_feature = (
            None
            if is_leaf or node.feature_index is None
            else model._feature_display_name(node.feature_index)
        )
        nodes.append(
            {
                "id": node.node_id,
                "parent_id": parent_id,
                "branch": branch,
                "depth": node.depth,
                "condition": condition_of(node),
                "split_feature": split_feature,
                "n": int(node.n_samples),
                "total_sum": safe_float(node.total_sum),
                "pos_sum": safe_float(node.s_node_p),
                "neg_sum": safe_float(node.s_node_n),
                "abs_volume": safe_float(node.s_node_p + node.s_node_n),
                "is_leaf": is_leaf,
                "segment_id": node_to_segment.get(node.node_id),
            }
        )
        if is_leaf:
            return
        child_inter = inter_depth + (1 if node.feature_index != parent_feature else 0)
        for key in ("positive", "neutral", "negative"):
            child = (node.children or {}).get(key)
            if child is not None:
                walk(child, node.node_id, key, child_inter, node.feature_index)

    walk(tree, None, "root", 0, None)

    total_sum = float(tree.total_sum)
    seg_total = sum(float(s["total_sum"] or 0.0) for s in segments)
    n_features = int(model._X.shape[1]) if model._X is not None else 0
    feature_names = (
        list(model.feature_names_in_)
        if model.feature_names_in_ is not None
        else [f"f{i}" for i in range(n_features)]
    )

    return {
        "meta": {
            "package_version": _package_version(),
            "params": {
                "delta_pct": model.delta_pct,
                "min_global_impact_pct": model.min_global_impact_pct,
                "max_depth": model.max_depth,
                "noise_z": model.noise_z,
                "consolidate": model.consolidate,
            },
            "n_rows": int(tree.n_samples),
            "n_features": n_features,
            "feature_names": feature_names,
            "total_sum": safe_float(total_sum),
            "pos_pool": safe_float(pos_pool),
            "neg_pool": safe_float(neg_pool),
            "n_nodes": len(nodes),
            "n_leaves": stats["leaves"],
            "physical_depth": stats["depth"],
            "interaction_depth": stats["inter_depth"],
            "n_segments": len(segments),
            "conservation_exact": abs(seg_total - total_sum)
            <= 1e-9 * max(1.0, abs(total_sum)),
        },
        "tree": nodes,
        "segments": segments,
    }
```

- [ ] **Step 2.5: Wire into `splitter.py`.** Add after `get_impact_segments` (imports at use-site to keep splitter import light):

```python
    def to_dict(self) -> dict[str, Any]:
        """JSON-safe payload (``meta`` / ``tree`` / ``segments``) feeding every renderer.

        Stable shape for third-party renderers; see ``impact_split.viz.data``.
        """
        from impact_split.viz.data import build_payload

        return build_payload(self)
```

Then, inside `get_impact_segments`, replace the `rows = [...]` comprehension so each row also carries `mean` and `pool_share` (existing keys stay first; `_v_global_p`/`_v_global_n` are already set post-fit):

```python
        rows = []
        for s in self.segments_:
            total = float(s["total_sum"])
            n = int(s["n_samples"])
            pool = self._v_global_p if total >= 0 else self._v_global_n
            rows.append(
                {
                    "path": s["path"],
                    "total_sum": s["total_sum"],
                    "n_samples": s["n_samples"],
                    "node_id": (
                        s["node_ids"][0]
                        if len(s["node_ids"]) == 1
                        else "merged(" + "+".join(s["node_ids"]) + ")"
                    ),
                    "mean": total / n if n else float("nan"),
                    "pool_share": abs(total) / pool if pool > 0 else float("nan"),
                }
            )
```

- [ ] **Step 2.6: Run — expect PASS; full suite + lint + mypy green:**

```bash
python -m pytest tests/test_viz_data.py -q && python -m pytest -q
python -m ruff format . && python -m ruff check . && python -m mypy impact_split
```

- [ ] **Step 2.7: Commit**

```bash
git add impact_split/viz tests/test_viz_data.py impact_split/splitter.py
git commit -m "feat(viz): to_dict payload builder; get_impact_segments gains mean/pool_share"
```

---

### Task 3: `viz/text.py` — `summary()` and `print(model)`

**Files:**
- Create: `impact_split/viz/text.py`, `tests/test_viz_text.py`
- Modify: `impact_split/splitter.py` (add `summary`, `__repr__`)

**Interfaces:**
- Consumes: payload from Task 2.
- Produces: `render_summary(payload, *, top=10, path_width=44) -> str`; `ImpactSplitter.summary(*, top=10) -> str`; `ImpactSplitter.__repr__`.

- [ ] **Step 3.1: Write failing tests — `tests/test_viz_text.py`:**

```python
"""Tests for the text renderer (summary / repr)."""

import pytest

from impact_split import ImpactSplitter
from tests.test_viz_data import fitted


def test_summary_requires_fit() -> None:
    with pytest.raises(RuntimeError, match="fit\\(\\)"):
        ImpactSplitter().summary()


def test_summary_ledger_and_table() -> None:
    model = fitted()
    text = model.summary()
    assert "ImpactSplitter — fit summary" in text
    assert "total Σy" in text and "Σy⁺" in text and "Σy⁻" in text
    assert "conservation exact ✓" in text
    assert "Top segments by |impact|" in text
    # every displayed segment row shows a pool-share annotation
    assert "of Σy⁺" in text or "of Σy⁻" in text


def test_summary_rolls_up_remainder() -> None:
    model = fitted()
    n_segments = model.to_dict()["meta"]["n_segments"]
    if n_segments < 2:
        pytest.skip("fixture produced a single segment")
    text = model.summary(top=1)
    assert f"(+{n_segments - 1} more segments)" in text


def test_repr_pre_and_post_fit() -> None:
    model = ImpactSplitter()
    assert repr(model).startswith("ImpactSplitter(delta_pct=")
    fitted_model = fitted()
    assert "fit summary" in repr(fitted_model)
```

- [ ] **Step 3.2: Run — expect FAIL:** `python -m pytest tests/test_viz_text.py -q`

- [ ] **Step 3.3: Create `impact_split/viz/text.py`:**

```python
"""Text renderer: a designed fit summary for terminals, logs, and notebooks."""

from __future__ import annotations

from typing import Any

from impact_split.viz.data import fmt_num, fmt_pct


def render_summary(payload: dict[str, Any], *, top: int = 10, path_width: int = 44) -> str:
    """Ledger header + ranked segment table; always ends with a complete total story."""
    meta = payload["meta"]
    segments = payload["segments"]
    p = meta["params"]

    conservation = "exact ✓" if meta["conservation_exact"] else "MISMATCH ✗"
    merged_note = (
        f" ({meta['n_leaves']} leaves merged)"
        if p["consolidate"] and meta["n_segments"] < meta["n_leaves"]
        else ""
    )
    lines = [
        "ImpactSplitter — fit summary",
        "============================",
        (
            f"rows {meta['n_rows']:,} · features {meta['n_features']} · "
            f"params delta_pct={p['delta_pct']} noise_z={p['noise_z']} "
            f"max_depth={p['max_depth']} consolidate={p['consolidate']}"
        ),
        (
            f"total Σy {fmt_num(meta['total_sum'], sign=True)}   "
            f"(Σy⁺ {fmt_num(meta['pos_pool'])} · Σy⁻ -{fmt_num(meta['neg_pool'])})"
        ),
        (
            f"tree      {meta['n_nodes']} nodes · {meta['n_leaves']} leaves · "
            f"depth {meta['physical_depth']} "
            f"(interaction order {meta['interaction_depth']})"
        ),
        f"segments  {meta['n_segments']}{merged_note} · conservation {conservation}",
        "",
        "Top segments by |impact|",
        f" {'#':>2}  {'path':<{path_width}}  {'Σy':>14}  {'n':>9}  {'pool share':>16}",
    ]

    shown = segments[:top]
    rest = segments[top:]
    for i, seg in enumerate(shown, start=1):
        path = str(seg["path"])
        if len(path) > path_width:
            path = path[: path_width - 1] + "…"
        if seg["pool_share"] is not None:
            pool_label = "Σy⁺" if (seg["total_sum"] or 0.0) >= 0 else "Σy⁻"
            share = f"{fmt_pct(seg['pool_share'])} of {pool_label}"
        else:
            share = "—"
        lines.append(
            f" {i:>2}  {path:<{path_width}}  {fmt_num(seg['total_sum'], sign=True):>14}"
            f"  {seg['n']:>9,}  {share:>16}"
        )
    if rest:
        rest_total = sum(float(s["total_sum"] or 0.0) for s in rest)
        rest_n = sum(int(s["n"]) for s in rest)
        label = f"(+{len(rest)} more segments)"
        lines.append(
            f" {'…':>2}  {label:<{path_width}}  {fmt_num(rest_total, sign=True):>14}"
            f"  {rest_n:>9,}  {'':>16}"
        )
    return "\n".join(lines)
```

- [ ] **Step 3.4: Add to `ImpactSplitter` in `splitter.py`:**

```python
    def summary(self, *, top: int = 10) -> str:
        """Designed text report: ledger header + top segments table (returns, never prints)."""
        from impact_split.viz.text import render_summary

        return render_summary(self.to_dict(), top=top)

    def __repr__(self) -> str:
        if self._tree is None:
            return (
                f"ImpactSplitter(delta_pct={self.delta_pct}, "
                f"min_global_impact_pct={self.min_global_impact_pct}, "
                f"max_depth={self.max_depth}, noise_z={self.noise_z}, "
                f"consolidate={self.consolidate})"
            )
        return self.summary()
```

- [ ] **Step 3.5: Run — expect PASS; gates green:**

```bash
python -m pytest tests/test_viz_text.py -q && python -m pytest -q
python -m ruff format . && python -m ruff check . && python -m mypy impact_split
```

- [ ] **Step 3.6: Commit**

```bash
git add impact_split/viz/text.py tests/test_viz_text.py impact_split/splitter.py
git commit -m "feat(viz): summary() text report and informative repr"
```

---

### Task 4: `viz/static.py` — segment tornado (`plot_segments`)

**Files:**
- Create: `impact_split/viz/static.py`, `tests/test_viz_static.py`
- Modify: `impact_split/splitter.py` (add `plot_segments` delegate)

**Interfaces:**
- Consumes: payload.
- Produces: `plot_segments(payload, *, top=15, figsize=None, show=True) -> Figure`; module-level `_text_color_for(rgb) -> str` and `_DIVERGING_CMAP` (reused by Task 5); `ImpactSplitter.plot_segments(*, top=15, figsize=None, show=True) -> Figure`.

- [ ] **Step 4.1: Write failing tests — `tests/test_viz_static.py`:**

```python
"""Tests for static figure renderers (tornado + icicle)."""

import matplotlib

matplotlib.use("Agg")

from matplotlib.figure import Figure
import numpy as np
import pandas as pd
import pytest

from impact_split import ImpactSplitter
from tests.test_viz_data import fitted


def test_plot_segments_requires_fit() -> None:
    with pytest.raises(RuntimeError, match="fit\\(\\)"):
        ImpactSplitter().plot_segments()


def test_plot_segments_returns_figure() -> None:
    fig = fitted().plot_segments(show=False)
    assert isinstance(fig, Figure)


def test_plot_segments_rolls_up_remainder() -> None:
    model = fitted()
    fig = model.plot_segments(top=1, show=False)
    labels = [t.get_text() for t in fig.axes[0].get_yticklabels()]
    assert any("more segments" in label for label in labels)


def test_plot_segments_root_only_model() -> None:
    # all-zero target -> materiality leaf at the root; must not raise
    X = pd.DataFrame({"a": ["x", "y"] * 10})
    y = pd.Series(np.zeros(20))
    model = ImpactSplitter().fit(X, y)
    fig = model.plot_segments(show=False)
    assert isinstance(fig, Figure)
```

- [ ] **Step 4.2: Run — expect FAIL:** `python -m pytest tests/test_viz_static.py -q`

- [ ] **Step 4.3: Create `impact_split/viz/static.py`:**

```python
"""Static matplotlib renderers: segment tornado and impact icicle."""

from __future__ import annotations

import textwrap
from typing import Any

from matplotlib.colors import LinearSegmentedColormap, Normalize, to_rgb
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

from impact_split.viz.data import (
    NEGATIVE_COLOR,
    NEUTRAL_FILL,
    NEUTRAL_STROKE,
    POSITIVE_COLOR,
    fmt_num,
    fmt_pct,
)

_MUTED_TEXT = "#5f5f5c"
_INK = "#26261f"
_ROLLED_FILL = "#c9c9c5"
_GRID = "#e6e6e2"

_DIVERGING_CMAP = LinearSegmentedColormap.from_list(
    "impact_diverging", [NEGATIVE_COLOR, NEUTRAL_FILL, POSITIVE_COLOR]
)


def _text_color_for(rgb: tuple[float, float, float]) -> str:
    """Black or white ink for readable text on the given face color."""

    def lin(c: float) -> float:
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

    luminance = 0.2126 * lin(rgb[0]) + 0.7152 * lin(rgb[1]) + 0.0722 * lin(rgb[2])
    return "white" if luminance < 0.45 else _INK


def plot_segments(
    payload: dict[str, Any],
    *,
    top: int = 15,
    figsize: tuple[float, float] | None = None,
    show: bool = True,
) -> Figure:
    """Tornado chart: consolidated segments diverging at zero, largest |Σy| on top."""
    meta = payload["meta"]
    segments = payload["segments"]
    bars: list[dict[str, Any]] = [{**s, "_rolled": False} for s in segments[:top]]
    rest = segments[top:]
    if rest:
        bars.append(
            {
                "path": f"(+{len(rest)} more segments)",
                "total_sum": sum(float(s["total_sum"] or 0.0) for s in rest),
                "n": sum(int(s["n"]) for s in rest),
                "pool_share": None,
                "_rolled": True,
            }
        )

    if figsize is None:
        figsize = (11.0, max(3.2, 0.62 * len(bars) + 1.8))
    fig, ax = plt.subplots(figsize=figsize)

    values = [float(b["total_sum"] or 0.0) for b in bars]
    max_abs = max((abs(v) for v in values), default=1.0) or 1.0
    pad = 0.015 * max_abs
    labels: list[str] = []
    for i, (b, value) in enumerate(zip(bars, values)):
        y = len(bars) - 1 - i
        if b["_rolled"]:
            face, hatch = _ROLLED_FILL, "///"
        else:
            face, hatch = (POSITIVE_COLOR if value >= 0 else NEGATIVE_COLOR), None
        ax.barh(
            y, value, height=0.72, color=face, hatch=hatch, edgecolor="white", linewidth=0.8
        )
        labels.append(textwrap.fill(str(b["path"]), width=38))
        note = f"n={b['n']:,}"
        if b["pool_share"] is not None:
            note += f" · {fmt_pct(b['pool_share'])} of {'Σy⁺' if value >= 0 else 'Σy⁻'}"
        offset = pad if value >= 0 else -pad
        ha = "left" if value >= 0 else "right"
        ax.text(
            value + offset, y + 0.16, fmt_num(value, sign=True),
            va="center", ha=ha, fontsize=9, fontweight="bold", color=_INK,
        )
        ax.text(value + offset, y - 0.2, note, va="center", ha=ha, fontsize=7.5, color=_MUTED_TEXT)

    ax.set_yticks([len(bars) - 1 - i for i in range(len(bars))], labels=labels, fontsize=8)
    ax.set_xlim(min(0.0, min(values, default=0.0)) - 0.24 * max_abs,
                max(0.0, max(values, default=0.0)) + 0.24 * max_abs)
    ax.axvline(0.0, color=NEUTRAL_STROKE, linewidth=1.0, zorder=0)
    ax.grid(axis="x", color=_GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#d8d8d4")
    ax.tick_params(axis="x", labelsize=8, colors=_MUTED_TEXT)
    ax.tick_params(axis="y", length=0)
    ax.set_title(
        f"Impact segments — {meta['n_segments']} segments · "
        f"total Σy {fmt_num(meta['total_sum'], sign=True)}",
        loc="left", fontsize=11, color=_INK,
    )
    conservation = "exact ✓" if meta["conservation_exact"] else "MISMATCH ✗"
    fig.text(
        0.01, 0.01,
        f"bars are additive: sum of all segments = total Σy ({conservation}) · "
        f"blue = positive impact · orange = negative",
        fontsize=7.5, color=_MUTED_TEXT,
    )
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    if show:
        plt.show()
    return fig
```

(`Normalize` and `to_rgb` are used by Task 5's icicle in this same module; if ruff flags them unused at this point, add them in Task 5 instead.)

- [ ] **Step 4.4: Add delegate to `ImpactSplitter`:**

```python
    def plot_segments(
        self,
        *,
        top: int = 15,
        figsize: tuple[float, float] | None = None,
        show: bool = True,
    ) -> Figure:
        """Tornado chart of consolidated segments (the stakeholder deliverable view)."""
        from impact_split.viz.static import plot_segments

        return plot_segments(self.to_dict(), top=top, figsize=figsize, show=show)
```

- [ ] **Step 4.5: Run — expect PASS; gates green:**

```bash
python -m pytest tests/test_viz_static.py -q && python -m pytest -q
python -m ruff format . && python -m ruff check . && python -m mypy impact_split
```

- [ ] **Step 4.6: Commit**

```bash
git add impact_split/viz/static.py tests/test_viz_static.py impact_split/splitter.py
git commit -m "feat(viz): segment tornado chart (plot_segments)"
```

---

### Task 5: Icicle `plot_tree` — replace the box-and-arrow tree

**Files:**
- Modify: `impact_split/viz/static.py` (add `layout_icicle`, `plot_icicle`), `impact_split/splitter.py` (replace `plot_tree`; delete old plot block), `tests/test_impact_splitter.py` (delete old plot tests), `tests/test_viz_static.py` (add icicle tests)

**Interfaces:**
- Consumes: payload; `_DIVERGING_CMAP`, `_text_color_for` from Task 4.
- Produces: `layout_icicle(payload) -> list[dict]` (each: `id`, `x0`, `width`, `depth`, `node`) and `plot_icicle(payload, *, figsize=None, show=True) -> Figure`; `ImpactSplitter.plot_tree(figsize=None, *, show=True) -> Figure`.

- [ ] **Step 5.1: Add failing tests to `tests/test_viz_static.py`:**

```python
def test_icicle_layout_children_tile_parent_exactly() -> None:
    from impact_split.viz.static import layout_icicle

    payload = fitted().to_dict()
    rects = {r["id"]: r for r in layout_icicle(payload)}
    kids_of: dict[str, list[str]] = {}
    for node in payload["tree"]:
        if node["parent_id"] is not None:
            kids_of.setdefault(node["parent_id"], []).append(node["id"])
    root_id = payload["tree"][0]["id"]
    assert rects[root_id]["x0"] == 0.0 and rects[root_id]["width"] == pytest.approx(1.0)
    for parent_id, kid_ids in kids_of.items():
        parent = rects[parent_id]
        assert sum(rects[k]["width"] for k in kid_ids) == pytest.approx(
            parent["width"], abs=1e-9
        )
        assert min(rects[k]["x0"] for k in kid_ids) == pytest.approx(parent["x0"], abs=1e-9)


def test_plot_tree_returns_figure_and_requires_fit() -> None:
    with pytest.raises(RuntimeError, match="fit\\(\\)"):
        ImpactSplitter().plot_tree()
    fig = fitted().plot_tree(show=False)
    assert isinstance(fig, Figure)


def test_plot_tree_root_only_model() -> None:
    X = pd.DataFrame({"a": ["x", "y"] * 10})
    y = pd.Series(np.zeros(20))
    fig = ImpactSplitter().fit(X, y).plot_tree(show=False)
    assert isinstance(fig, Figure)
```

- [ ] **Step 5.2: Run — expect FAIL:** `python -m pytest tests/test_viz_static.py -q`

- [ ] **Step 5.3: Append to `impact_split/viz/static.py`:**

```python
def layout_icicle(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Rect per node in root-relative [0, 1] coordinates; children tile their parent exactly.

    Widths are proportional to ``abs_volume`` (Σ|y|); a sibling group whose volumes
    are all zero falls back to row counts so the layout stays total.
    """
    nodes = payload["tree"]
    children: dict[str | None, list[dict[str, Any]]] = {}
    for node in nodes:
        children.setdefault(node["parent_id"], []).append(node)
    root = children[None][0]
    rects: list[dict[str, Any]] = []

    def place(node: dict[str, Any], x0: float, width: float) -> None:
        rects.append(
            {"id": node["id"], "x0": x0, "width": width, "depth": node["depth"], "node": node}
        )
        kids = children.get(node["id"], [])
        if not kids:
            return
        weights = [float(k["abs_volume"] or 0.0) for k in kids]
        total = sum(weights)
        if total <= 0:
            weights = [float(k["n"]) for k in kids]
            total = sum(weights) or 1.0
        cursor = x0
        for kid, w in zip(kids, weights):
            kid_width = width * (w / total)
            place(kid, cursor, kid_width)
            cursor += kid_width

    place(root, 0.0, 1.0)
    return rects


def plot_icicle(
    payload: dict[str, Any],
    *,
    figsize: tuple[float, float] | None = None,
    show: bool = True,
) -> Figure:
    """Impact icicle: cell width ∝ Σ|y|, diverging color = mean excess vs overall mean."""
    from matplotlib.patches import Rectangle

    meta = payload["meta"]
    rects = layout_icicle(payload)
    depth_max = max(r["depth"] for r in rects)
    if figsize is None:
        figsize = (12.0, 1.15 * (depth_max + 1) + 1.9)
    fig, ax = plt.subplots(figsize=figsize)

    root_node = rects[0]["node"]
    root_mean = (root_node["total_sum"] or 0.0) / root_node["n"] if root_node["n"] else 0.0
    for r in rects:
        node = r["node"]
        mean = (node["total_sum"] or 0.0) / node["n"] if node["n"] else 0.0
        r["excess"] = mean - root_mean
    vmax = max((abs(r["excess"]) for r in rects), default=1.0) or 1.0
    norm = Normalize(vmin=-vmax, vmax=vmax)

    seg_leaf_count: dict[str, int] = {}
    for s in payload["segments"]:
        seg_leaf_count[s["segment_id"]] = len(s["node_ids"])

    fig_width_px = figsize[0] * 72.0
    for r in rects:
        node = r["node"]
        face = _DIVERGING_CMAP(norm(r["excess"]))[:3]
        merged_leaf = (
            node["is_leaf"]
            and node["segment_id"] is not None
            and seg_leaf_count.get(node["segment_id"], 1) > 1
        )
        ax.add_patch(
            Rectangle(
                (r["x0"], -r["depth"] - 0.94),
                r["width"],
                0.88,
                facecolor=face,
                edgecolor="#3a3a36" if merged_leaf else "white",
                linewidth=1.8 if merged_leaf else 1.1,
            )
        )
        label = f"{node['condition']}\n{fmt_num(node['total_sum'], sign=True)}"
        longest = max(len(line) for line in label.split("\n"))
        if r["width"] * fig_width_px * 0.86 >= longest * 5.0 and r["width"] >= 0.03:
            ax.text(
                r["x0"] + r["width"] / 2,
                -r["depth"] - 0.5,
                label,
                ha="center",
                va="center",
                fontsize=7.5,
                color=_text_color_for(face),
            )

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(-depth_max - 1.0, 0.0)
    ax.set_xticks([])
    ax.set_yticks(
        [-d - 0.5 for d in range(depth_max + 1)],
        labels=["root" if d == 0 else f"depth {d}" for d in range(depth_max + 1)],
        fontsize=8,
    )
    ax.tick_params(axis="y", length=0, colors=_MUTED_TEXT)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(
        f"Impact tree — width ∝ Σ|y| · total Σy {fmt_num(meta['total_sum'], sign=True)}",
        loc="left", fontsize=11, color=_INK,
    )
    mappable = plt.cm.ScalarMappable(cmap=_DIVERGING_CMAP, norm=norm)
    mappable.set_array([])
    cbar = fig.colorbar(mappable, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label("segment mean − overall mean", fontsize=8, color=_MUTED_TEXT)
    cbar.ax.tick_params(labelsize=7, colors=_MUTED_TEXT)
    fig.text(
        0.01, 0.01,
        "each row tiles the one above (children are their parent's rows) · "
        "dark-outlined leaves merged into one consolidated segment",
        fontsize=7.5, color=_MUTED_TEXT,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    if show:
        plt.show()
    return fig
```

Add `Normalize` to the `matplotlib.colors` import if not present (drop `to_rgb` if unused).

- [ ] **Step 5.4: Replace the old `plot_tree` in `splitter.py`.** Delete the entire old method (from `def plot_tree(` to the end of the file) and these now-orphaned module members: `_canvas_renderer`, `_BRANCH_EDGE_SHORT`, `_DEFAULT_BRANCH_EDGE_COLORS`, `_VERTICAL_LABEL_MARGIN`, `_iter_tree_nodes`, and methods `_format_plot_node_label`, `_last_path_segment_for_plot`, `_estimate_plot_label_bbox_units`, `_relative_luminance_srgb`, `_text_color_for_face_rgb`, `_measure_text_bbox_size_data`, `_measure_text_bbox_width_data`. Remove now-unused imports (`Normalize`, `matplotlib.pyplot as plt`, `Callable`; move `Figure` into a `TYPE_CHECKING` block — annotations are strings via `from __future__ import annotations`). Grep first to confirm nothing else references each deleted name:

```bash
grep -n "_iter_tree_nodes\|_canvas_renderer\|_format_plot_node_label\|_measure_text_bbox\|_estimate_plot_label\|_relative_luminance\|_text_color_for_face\|_last_path_segment_for_plot" -r impact_split tests benchmarks
```

Then add the new method where the old one was:

```python
    def plot_tree(
        self,
        figsize: tuple[float, float] | None = None,
        *,
        show: bool = True,
    ) -> Figure:
        """Impact icicle: cell width ∝ Σ|y| within its parent, color = impact direction.

        Save with ``fig = model.plot_tree(show=False); fig.savefig("tree.svg")``.
        """
        from impact_split.viz.static import plot_icicle

        return plot_icicle(self.to_dict(), figsize=figsize, show=show)
```

Note the pre-fit path: `to_dict()` already raises the required `RuntimeError`.

- [ ] **Step 5.5: Delete the old plot tests** in `tests/test_impact_splitter.py`: `test_plot_tree_smoke`, `test_plot_tree_label_truncation_smoke`, `test_plot_tree_max_leaf_width_budget_truncates_labels`, `test_plot_tree_impact_encoding_sets_contrasting_text_color`, `test_plot_tree_layout_and_facecolor_smoke` (and any other test whose body uses removed kwargs like `node_facecolor`/`max_leaf_width`; remove now-unused imports such as `io`/`plt` if ruff flags them).

- [ ] **Step 5.6: Run — expect PASS; gates green:**

```bash
python -m pytest -q
python -m ruff format . && python -m ruff check . && python -m mypy impact_split
```

- [ ] **Step 5.7: Commit**

```bash
git add -A
git commit -m "feat(viz)!: icicle plot_tree replaces box-and-arrow tree (-450 lines of layout)"
```

---

### Task 6: `viz/html.py` — self-contained interactive report; delete force graph

**Files:**
- Create: `impact_split/viz/html.py`, `tests/test_viz_html.py`
- Delete: `impact_split/plots.py`, `tests/test_interactive_plots.py`
- Modify: `impact_split/splitter.py` (add `to_html`), `impact_split/__init__.py`, `pyproject.toml` (mypy overrides)

**Interfaces:**
- Consumes: payload; `fmt_num`/`fmt_pct` from `viz.data`.
- Produces: `render_html(payload, *, title="impact-split report") -> str`; `ImpactSplitter.to_html(path=None, *, title=...) -> str | Path`. Final public API: `impact_split.__init__` exports `ImpactSplitter`, `plot_icicle`, `plot_segments`, `render_html`, `render_summary`.

- [ ] **Step 6.1: Write failing tests — `tests/test_viz_html.py`:**

```python
"""Tests for the self-contained HTML report."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from impact_split import ImpactSplitter
from tests.test_viz_data import fitted


def test_to_html_requires_fit() -> None:
    with pytest.raises(RuntimeError, match="fit\\(\\)"):
        ImpactSplitter().to_html()


def test_html_is_fully_self_contained() -> None:
    html_out = fitted().to_html()
    assert isinstance(html_out, str)
    for token in ("http://", "https://", "src=", "url(", "@import"):
        assert token not in html_out, f"external reference found: {token}"
    assert '"segments"' in html_out  # payload embedded


def test_html_escapes_script_closers_in_data() -> None:
    rng = np.random.default_rng(3)
    X = pd.DataFrame({"attack": rng.choice(["</script>", "safe"], size=200)})
    y = pd.Series(rng.normal(0, 1, size=200) + (X["attack"] == "</script>") * 9.0)
    html_out = ImpactSplitter().fit(X, y).to_html()
    # the only literal closing tag is the template's own single script block
    assert html_out.count("</script>") == 1


def test_html_write_mode_roundtrip(tmp_path: Path) -> None:
    out = fitted().to_html(tmp_path / "report.html")
    assert isinstance(out, Path)
    text = out.read_text(encoding="utf-8")
    assert text.startswith("<!doctype html>") and '"segments"' in text
```

- [ ] **Step 6.2: Run — expect FAIL:** `python -m pytest tests/test_viz_html.py -q`

- [ ] **Step 6.3: Create `impact_split/viz/html.py`** (complete file; the template uses `string.Template` so CSS/JS braces are safe — the JS deliberately avoids backtick template literals and `$`):

```python
"""Self-contained interactive HTML report: inline CSS + vanilla JS + SVG, no CDN."""

from __future__ import annotations

from html import escape
import json
from string import Template
from typing import Any

from impact_split.viz.data import fmt_num


def render_html(payload: dict[str, Any], *, title: str = "impact-split report") -> str:
    """One offline-safe HTML file: ledger tiles, zoomable icicle, tornado, sortable table."""
    data_json = json.dumps(payload, allow_nan=False).replace("</", "<\\/")
    meta = payload["meta"]
    p = meta["params"]
    conservation = "exact ✓" if meta["conservation_exact"] else "MISMATCH ✗"
    tiles = [
        ("rows", f"{meta['n_rows']:,}"),
        ("total Σy", fmt_num(meta["total_sum"], sign=True)),
        ("Σy⁺ pool", fmt_num(meta["pos_pool"])),
        ("Σy⁻ pool", "-" + fmt_num(meta["neg_pool"])),
        ("nodes / leaves", f"{meta['n_nodes']} / {meta['n_leaves']}"),
        ("segments", f"{meta['n_segments']}"),
        ("conservation", conservation),
    ]
    tiles_html = "".join(
        '<div class="tile"><div class="tile-label">'
        + escape(label)
        + '</div><div class="tile-value">'
        + escape(value)
        + "</div></div>"
        for label, value in tiles
    )
    params_line = escape(
        f"delta_pct={p['delta_pct']} · noise_z={p['noise_z']} · "
        f"max_depth={p['max_depth']} · consolidate={p['consolidate']} · "
        f"impact_split v{meta['package_version']}"
    )
    return _TEMPLATE.substitute(
        title=escape(title), tiles=tiles_html, params=params_line, data=data_json
    )


_TEMPLATE = Template(
    """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>$title</title>
<style>
:root { --pos:#0173B2; --neg:#C6660A; --ink:#26261f; --muted:#6b6b66; --line:#e4e4df;
        --surface:#fcfcfb; --card:#ffffff; }
* { box-sizing:border-box; }
body { margin:0; padding:24px 28px 60px; background:var(--surface); color:var(--ink);
       font:14px/1.5 system-ui, "Segoe UI", Roboto, sans-serif; }
h1 { font-size:20px; margin:0 0 2px; }
h2 { font-size:15px; margin:34px 0 4px; }
.params { color:var(--muted); font-size:12px; margin-bottom:14px; }
.hint { color:var(--muted); font-size:12px; margin:0 0 10px; }
.tiles { display:flex; flex-wrap:wrap; gap:10px; }
.tile { background:var(--card); border:1px solid var(--line); border-radius:8px;
        padding:8px 14px; min-width:110px; }
.tile-label { font-size:11px; color:var(--muted); }
.tile-value { font-size:16px; font-weight:600; font-variant-numeric:tabular-nums; }
#breadcrumb { font-size:12px; margin:6px 0 8px; min-height:18px; }
.crumb { color:var(--pos); cursor:pointer; }
.crumb:hover { text-decoration:underline; }
.crumb-sep { color:var(--muted); }
#icicle { width:100%; display:block; background:var(--card); border:1px solid var(--line);
          border-radius:8px; }
#icicle rect { cursor:pointer; }
#icicle rect.hl { stroke:#1a1a18; stroke-width:3; }
#icicle text { pointer-events:none; font-size:11px; }
#tornado { background:var(--card); border:1px solid var(--line); border-radius:8px;
           padding:10px 14px; }
.trow { display:grid; grid-template-columns:minmax(180px, 30%) 1fr 110px 200px;
        gap:10px; align-items:center; padding:3px 0; }
.trow.hl { background:#f3f0e9; }
.tpath { font-size:12px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
.ttrack { position:relative; height:16px; }
.tzero { position:absolute; top:-2px; bottom:-2px; width:1px; background:#949494; }
.tbar { position:absolute; top:1px; height:14px; border-radius:4px; }
.tbar.pos { background:var(--pos); }
.tbar.neg { background:var(--neg); }
.tbar.rolled { background:#c9c9c5; }
.tval { font-variant-numeric:tabular-nums; font-size:12px; font-weight:600; text-align:right; }
.tnote { color:var(--muted); font-size:11px; }
table { border-collapse:collapse; width:100%; background:var(--card);
        border:1px solid var(--line); border-radius:8px; font-size:12.5px; }
th, td { padding:6px 10px; text-align:right; border-bottom:1px solid var(--line);
         font-variant-numeric:tabular-nums; }
th { cursor:pointer; user-select:none; color:var(--muted); font-weight:600; }
th:hover { color:var(--ink); }
td.path, th.path { text-align:left; }
tr.hl td { background:#f3f0e9; }
.chip { display:inline-block; width:9px; height:9px; border-radius:2px; margin-right:6px; }
#tooltip { display:none; position:fixed; z-index:10; max-width:420px; background:#26261f;
           color:#fcfcfb; border-radius:6px; padding:8px 10px; font-size:12px;
           pointer-events:none; white-space:pre-line; }
</style>
</head>
<body>
<h1>$title</h1>
<div class="params">$params</div>
<div class="tiles">$tiles</div>

<h2>Impact tree — where the impact concentrates</h2>
<p class="hint">Cell width ∝ Σ|y| · blue = above overall mean, orange = below ·
click a cell to zoom into that subtree · dark-outlined leaves were merged into one
consolidated segment · hover for the full rule path.</p>
<div id="breadcrumb"></div>
<svg id="icicle"></svg>

<h2>Segments ranked by |impact|</h2>
<p class="hint">Each bar is a consolidated segment's total Σy. Bars are additive —
together they reconstruct the total exactly. Hover to locate the segment in the tree.</p>
<div id="tornado"></div>

<h2>All segments</h2>
<table id="segtable"><thead></thead><tbody></tbody></table>

<div id="tooltip"></div>
<script>
var DATA = $data;

var byId = {}; DATA.tree.forEach(function (n) { byId[n.id] = n; });
var childrenOf = {};
DATA.tree.forEach(function (n) {
  if (n.parent_id !== null) {
    (childrenOf[n.parent_id] = childrenOf[n.parent_id] || []).push(n);
  }
});
var ROOT = DATA.tree[0];
var segById = {}; DATA.segments.forEach(function (s) { segById[s.segment_id] = s; });

function esc(s) {
  return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;")
    .replace(/>/g, "&gt;").replace(/"/g, "&quot;");
}
function fmt(v) {
  if (v === null || v === undefined) return "—";
  var sign = v > 0 ? "+" : "";
  if (Math.abs(v) >= 1000) return sign + Math.round(v).toLocaleString("en-US");
  return sign + v.toLocaleString("en-US", { maximumFractionDigits: 2 });
}
function pct(v) { return v == null ? "—" : (100 * v).toFixed(1) + "%"; }

var POS_RGB = [1, 115, 178], NEG_RGB = [198, 102, 10], MID_RGB = [242, 242, 240];
function mix(a, b, t) {
  return [Math.round(a[0] + (b[0] - a[0]) * t),
          Math.round(a[1] + (b[1] - a[1]) * t),
          Math.round(a[2] + (b[2] - a[2]) * t)];
}
function divergingColor(t) {
  t = Math.max(-1, Math.min(1, t));
  var rgb = t >= 0 ? mix(MID_RGB, POS_RGB, t) : mix(MID_RGB, NEG_RGB, -t);
  return { css: "rgb(" + rgb.join(",") + ")", rgb: rgb };
}
function inkFor(rgb) {
  function lin(c) { c /= 255; return c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4); }
  var L = 0.2126 * lin(rgb[0]) + 0.7152 * lin(rgb[1]) + 0.0722 * lin(rgb[2]);
  return L < 0.45 ? "#ffffff" : "#26261f";
}

var rootMean = ROOT.n ? (ROOT.total_sum || 0) / ROOT.n : 0;
var VMAX = 0;
DATA.tree.forEach(function (n) {
  var e = Math.abs((n.n ? (n.total_sum || 0) / n.n : 0) - rootMean);
  if (e > VMAX) VMAX = e;
});
if (VMAX === 0) VMAX = 1;

var tip = document.getElementById("tooltip");
function showTip(ev, text) {
  tip.textContent = text;
  tip.style.display = "block";
  tip.style.left = Math.min(ev.clientX + 14, window.innerWidth - 440) + "px";
  tip.style.top = (ev.clientY + 14) + "px";
}
function hideTip() { tip.style.display = "none"; }

function cumPath(n) {
  var parts = [], cur = n;
  while (cur && cur.parent_id !== null) { parts.unshift(cur.condition); cur = byId[cur.parent_id]; }
  return parts.length ? parts.join("  /  ") : "all data";
}
function nodeTip(n) {
  var mean = n.n ? (n.total_sum || 0) / n.n : 0;
  var lines = [cumPath(n),
    "n = " + n.n.toLocaleString("en-US") + "   mean = " + fmt(mean),
    "Σy = " + fmt(n.total_sum) + "   Σy⁺ = " + fmt(n.pos_sum) + "   Σy⁻ = -" + fmt(n.neg_sum)];
  if (n.split_feature) lines.push("splits on: " + n.split_feature);
  if (n.segment_id) {
    var s = segById[n.segment_id];
    lines.push("segment #" + (DATA.segments.indexOf(s) + 1) +
      (s.node_ids.length > 1 ? " (merged from " + s.node_ids.length + " leaves)" : ""));
  }
  return lines.join("\\n");
}

function highlightSegment(segId, on) {
  document.querySelectorAll('[data-seg="' + segId + '"]').forEach(function (el) {
    el.classList.toggle("hl", on);
  });
}
function hookSegHover(el, segId) {
  el.addEventListener("mouseenter", function () { highlightSegment(segId, true); });
  el.addEventListener("mouseleave", function () { highlightSegment(segId, false); });
}

var currentRoot = ROOT.id;
var W = 1000, ROW_H = 46, PAD = 2;

function renderIcicle() {
  var rects = [];
  function place(node, x0, w, level) {
    rects.push({ node: node, x0: x0, w: w, level: level });
    var kids = childrenOf[node.id] || [];
    if (!kids.length) return;
    var ws = kids.map(function (k) { return k.abs_volume || 0; });
    var tot = ws.reduce(function (a, b) { return a + b; }, 0);
    if (tot <= 0) {
      ws = kids.map(function (k) { return k.n; });
      tot = ws.reduce(function (a, b) { return a + b; }, 0) || 1;
    }
    var cx = x0;
    kids.forEach(function (k, i) {
      var kw = w * ws[i] / tot;
      place(k, cx, kw, level + 1);
      cx += kw;
    });
  }
  place(byId[currentRoot], 0, W, 0);

  var depth = 0;
  rects.forEach(function (r) { if (r.level > depth) depth = r.level; });
  var H = (depth + 1) * ROW_H;
  var svg = document.getElementById("icicle");
  svg.setAttribute("viewBox", "0 0 " + W + " " + H);
  svg.style.height = Math.min(H, 560) + "px";

  var out = "";
  rects.forEach(function (r) {
    var n = r.node;
    var mean = n.n ? (n.total_sum || 0) / n.n : 0;
    var col = divergingColor((mean - rootMean) / VMAX);
    var seg = n.segment_id ? segById[n.segment_id] : null;
    var merged = seg && seg.node_ids.length > 1;
    out += '<rect data-node="' + n.id + '"' +
      (n.segment_id ? ' data-seg="' + n.segment_id + '"' : "") +
      ' x="' + (r.x0 + PAD / 2) + '" y="' + (r.level * ROW_H + PAD / 2) + '"' +
      ' width="' + Math.max(0.6, r.w - PAD) + '" height="' + (ROW_H - PAD) + '"' +
      ' fill="' + col.css + '" stroke="' + (merged ? "#3a3a36" : "#ffffff") + '"' +
      ' stroke-width="' + (merged ? 2 : 1) + '" rx="3"></rect>';
    if (r.w > 76) {
      var maxChars = Math.floor(r.w / 6.4);
      var label = n.condition.length > maxChars
        ? n.condition.slice(0, maxChars - 1) + "…" : n.condition;
      var ink = inkFor(col.rgb);
      out += '<text x="' + (r.x0 + r.w / 2) + '" y="' + (r.level * ROW_H + ROW_H / 2 - 3) +
        '" text-anchor="middle" fill="' + ink + '">' + esc(label) + "</text>";
      out += '<text x="' + (r.x0 + r.w / 2) + '" y="' + (r.level * ROW_H + ROW_H / 2 + 12) +
        '" text-anchor="middle" fill="' + ink + '" opacity="0.85">' +
        esc(fmt(n.total_sum)) + "</text>";
    }
  });
  svg.innerHTML = out;

  svg.querySelectorAll("rect").forEach(function (el) {
    var n = byId[el.getAttribute("data-node")];
    el.addEventListener("mousemove", function (ev) { showTip(ev, nodeTip(n)); });
    el.addEventListener("mouseleave", hideTip);
    el.addEventListener("click", function () {
      currentRoot = n.id; renderIcicle(); renderBreadcrumb();
    });
    var segId = el.getAttribute("data-seg");
    if (segId) hookSegHover(el, segId);
  });
}

function renderBreadcrumb() {
  var parts = [], cur = byId[currentRoot];
  while (cur) {
    parts.unshift(cur);
    cur = cur.parent_id !== null ? byId[cur.parent_id] : null;
  }
  document.getElementById("breadcrumb").innerHTML = parts.map(function (n) {
    return '<span class="crumb" data-node="' + n.id + '">' +
      esc(n.parent_id === null ? "all data" : n.condition) + "</span>";
  }).join(' <span class="crumb-sep">›</span> ');
  document.querySelectorAll("#breadcrumb .crumb").forEach(function (el) {
    el.addEventListener("click", function () {
      currentRoot = el.getAttribute("data-node"); renderIcicle(); renderBreadcrumb();
    });
  });
}

function renderTornado() {
  var TOP = 15;
  var shown = DATA.segments.slice(0, TOP);
  var rest = DATA.segments.slice(TOP);
  var rows = shown.map(function (s) {
    return { path: s.path, v: s.total_sum || 0, n: s.n, share: s.pool_share,
             seg: s.segment_id, rolled: false };
  });
  if (rest.length) {
    rows.push({
      path: "(+" + rest.length + " more segments)",
      v: rest.reduce(function (a, s) { return a + (s.total_sum || 0); }, 0),
      n: rest.reduce(function (a, s) { return a + s.n; }, 0),
      share: null, seg: null, rolled: true
    });
  }
  var lo = 0, hi = 0;
  rows.forEach(function (r) { lo = Math.min(lo, r.v); hi = Math.max(hi, r.v); });
  var range = (hi - lo) || 1;
  var zeroPct = (0 - lo) / range * 100;
  var out = "";
  rows.forEach(function (r) {
    var leftPct = (Math.min(r.v, 0) - lo) / range * 100;
    var widthPct = Math.max(Math.abs(r.v) / range * 100, 0.4);
    var cls = r.rolled ? "rolled" : (r.v >= 0 ? "pos" : "neg");
    var note = "n=" + r.n.toLocaleString("en-US") +
      (r.share != null ? " · " + pct(r.share) + " of " + (r.v >= 0 ? "Σy⁺" : "Σy⁻") : "");
    out += '<div class="trow"' + (r.seg ? ' data-seg="' + r.seg + '"' : "") + ">" +
      '<div class="tpath" title="' + esc(r.path) + '">' + esc(r.path) + "</div>" +
      '<div class="ttrack"><div class="tzero" style="left:' + zeroPct + '%"></div>' +
      '<div class="tbar ' + cls + '" style="left:' + leftPct + "%;width:" + widthPct +
      '%"></div></div>' +
      '<div class="tval">' + fmt(r.v) + "</div>" +
      '<div class="tnote">' + note + "</div></div>";
  });
  var host = document.getElementById("tornado");
  host.innerHTML = out;
  host.querySelectorAll(".trow[data-seg]").forEach(function (el) {
    hookSegHover(el, el.getAttribute("data-seg"));
  });
}

var tableRows = DATA.segments.map(function (s, i) {
  return { rank: i + 1, path: s.path, total_sum: s.total_sum || 0, n: s.n,
           mean: s.mean, pool_share: s.pool_share, leaves: s.node_ids.length,
           seg: s.segment_id };
});
var sortKey = "rank", sortDir = 1;
var COLS = [
  ["#", "rank"], ["path", "path"], ["Σy", "total_sum"], ["n", "n"],
  ["mean", "mean"], ["pool share", "pool_share"], ["leaves", "leaves"]
];
function renderTable() {
  var thead = document.querySelector("#segtable thead");
  var tbody = document.querySelector("#segtable tbody");
  thead.innerHTML = "<tr>" + COLS.map(function (c) {
    var mark = c[1] === sortKey ? (sortDir > 0 ? " ▲" : " ▼") : "";
    return '<th class="' + (c[1] === "path" ? "path" : "") + '" data-key="' + c[1] + '">' +
      esc(c[0]) + mark + "</th>";
  }).join("") + "</tr>";
  var rows = tableRows.slice().sort(function (a, b) {
    var av = a[sortKey], bv = b[sortKey];
    if (av == null) return 1;
    if (bv == null) return -1;
    if (typeof av === "string") return sortDir * av.localeCompare(bv);
    return sortDir * (av - bv);
  });
  tbody.innerHTML = rows.map(function (r) {
    var chip = '<span class="chip" style="background:' +
      (r.total_sum >= 0 ? "var(--pos)" : "var(--neg)") + '"></span>';
    return '<tr data-seg="' + r.seg + '">' +
      "<td>" + r.rank + "</td>" +
      '<td class="path">' + chip + esc(r.path) + "</td>" +
      "<td>" + fmt(r.total_sum) + "</td>" +
      "<td>" + r.n.toLocaleString("en-US") + "</td>" +
      "<td>" + fmt(r.mean) + "</td>" +
      "<td>" + (r.pool_share != null
        ? pct(r.pool_share) + " of " + (r.total_sum >= 0 ? "Σy⁺" : "Σy⁻") : "—") + "</td>" +
      "<td>" + r.leaves + "</td></tr>";
  }).join("");
  thead.querySelectorAll("th").forEach(function (th) {
    th.addEventListener("click", function () {
      var key = th.getAttribute("data-key");
      if (key === sortKey) { sortDir = -sortDir; } else { sortKey = key; sortDir = key === "path" ? 1 : -1; }
      renderTable();
    });
  });
  tbody.querySelectorAll("tr[data-seg]").forEach(function (tr) {
    hookSegHover(tr, tr.getAttribute("data-seg"));
  });
}

renderIcicle();
renderBreadcrumb();
renderTornado();
renderTable();
</script>
</body>
</html>
"""
)
```

- [ ] **Step 6.4: Add `to_html` to `ImpactSplitter`** (also add `from pathlib import Path` to splitter imports):

```python
    def to_html(
        self,
        path: str | Path | None = None,
        *,
        title: str = "impact-split report",
    ) -> str | Path:
        """Self-contained interactive HTML report (no CDN; safe to open offline or email).

        Returns the HTML string when ``path`` is None (usable with
        ``IPython.display.HTML``); otherwise writes UTF-8 and returns the ``Path``.
        """
        from impact_split.viz.html import render_html

        html_text = render_html(self.to_dict(), title=title)
        if path is None:
            return html_text
        out = Path(path)
        out.write_text(html_text, encoding="utf-8")
        return out
```

- [ ] **Step 6.5: Delete the force graph and finalize exports:**

```bash
git rm impact_split/plots.py tests/test_interactive_plots.py
```

Replace `impact_split/__init__.py` with:

```python
from impact_split.splitter import ImpactSplitter
from impact_split.viz.html import render_html
from impact_split.viz.static import plot_icicle, plot_segments
from impact_split.viz.text import render_summary

__all__ = ["ImpactSplitter", "plot_icicle", "plot_segments", "render_html", "render_summary"]
```

In `pyproject.toml` trim the mypy overrides list to `["matplotlib.*"]` (ipykernel/IPython/ipywidgets imports left with `plots.py`).

- [ ] **Step 6.6: Run — expect PASS; gates green:**

```bash
python -m pytest -q
python -m ruff format . && python -m ruff check . && python -m mypy impact_split && python -m bandit -r impact_split -ll
```

- [ ] **Step 6.7: Commit**

```bash
git add -A
git commit -m "feat(viz)!: self-contained interactive HTML report; remove force graph"
```

---

### Task 7: Docs, README, notebooks, sample artifacts

**Files:**
- Modify: `README.md`, `docs/docs/getting-started.md`, `notebooks/1.0-jde-impact-split-explainer.ipynb`, `notebooks/2.0-jde-supermarket-kaggle-trace.ipynb`
- Create: `reports/sample-report.html`, `reports/figures/segments-tornado.png`, `reports/figures/impact-icicle.png`

- [ ] **Step 7.1: README rewrite.** Replace the Quick Start / force-graph / Output sections (keep Story Behind the Math and Act I–IV untouched):
  - Install: `pip install impact-split` (PyPI, once published) + dev-install block.
  - Basic usage becomes:

```python
from impact_split import ImpactSplitter

model = ImpactSplitter().fit(X, y)   # X: DataFrame or int-encoded ndarray; y: additive target

print(model)                          # designed text summary (ledger + top segments)
model.plot_segments()                 # tornado: ranked segment impacts (matplotlib)
model.plot_tree()                     # icicle: where impact concentrates in the tree
model.to_html("report.html")          # self-contained interactive report (offline-safe)

segments = model.get_impact_segments()   # DataFrame: path, total_sum, n_samples, node_id, mean, pool_share
payload = model.to_dict()                # JSON-safe dict for custom renderers
```

  - New "Outputs" section: one paragraph per renderer explaining the encodings (tornado bars additive and diverging at zero, blue positive / orange negative; icicle width ∝ Σ|y|, color = mean excess; HTML report = both + sortable table + linked highlighting, zero external requests). Embed the two PNGs from Step 7.3.
  - Update the Repository-layout table row for `impact_split/` (mention `viz/`, drop `plots.py`/config/features/dataset/modeling).
- [ ] **Step 7.2: `docs/docs/getting-started.md`:** replace the `plot_tree` kwargs note and the force-graph section with the new five-output tour (same content as README, shortened).
- [ ] **Step 7.3: Generate sample artifacts** with a scratch script against the explainer DGP (seeded synthetic; commit the outputs):

```bash
python - <<'PY'
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
from impact_split import ImpactSplitter

rng = np.random.default_rng(42)
n = 6000
region = rng.choice(["north", "south", "east", "west"], size=n, p=[0.3, 0.3, 0.2, 0.2])
product = rng.choice(["basic", "plus", "premium"], size=n, p=[0.5, 0.3, 0.2])
channel = rng.choice(["online", "retail", "partner"], size=n, p=[0.5, 0.3, 0.2])
y = rng.normal(0, 40, size=n)
y[(region == "north") & (product == "premium")] += 260
y[(channel == "partner") & (region == "west")] -= 200
y[product == "plus"] += 60
X = pd.DataFrame({"region": region, "product": product, "channel": channel})
m = ImpactSplitter().fit(X, pd.Series(y))
print(m.summary())
m.plot_segments(show=False).savefig("reports/figures/segments-tornado.png", dpi=160)
m.plot_tree(show=False).savefig("reports/figures/impact-icicle.png", dpi=160)
m.to_html("reports/sample-report.html", title="impact-split sample report")
PY
```

Open `reports/sample-report.html` in a browser (or the Browser pane) and eyeball: icicle zoom works, tooltips show, table sorts, no console errors. Look at both PNGs for label collisions.
- [ ] **Step 7.4: Notebooks.** In 1.0 and 2.0, update every cell using removed APIs: `plot_tree(...)` old kwargs → `model.plot_tree()` / `model.plot_segments()`; force-graph cells → `model.to_html(...)` demo; add a `print(model)` cell after each fit. Re-execute 1.0 end-to-end (`jupyter nbconvert --to notebook --execute --inplace notebooks/1.0-...ipynb` — install jupyter into the venv if missing, it is not a package dependency). Re-execute 2.0 the same way (kagglehub cache exists; set `KAGGLE_API_TOKEN` from ai-os `.env` if needed).
- [ ] **Step 7.5: Commit**

```bash
git add -A
git commit -m "docs: README/getting-started for v0.1.0 outputs; notebooks on new API; sample report artifacts"
```

---

### Task 8: CI matrix + release workflow + build validation

**Files:**
- Modify: `.github/workflows/ci.yml`
- Create: `.github/workflows/release.yml`

- [ ] **Step 8.1: CI matrix.** In `ci.yml` replace the single `validate` job's Python setup with a matrix (lint/type/build stay on 3.13 only to keep signal clean):

```yaml
jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      fail-fast: false
      matrix:
        python-version: ["3.10", "3.11", "3.12", "3.13"]
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
      - name: Install package with dev extras
        run: |
          python -m pip install --upgrade pip
          python -m pip install -e ".[dev]"
      - name: Run tests
        run: python -m pytest
  validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.13"
      - name: Install package with dev extras
        run: |
          python -m pip install --upgrade pip
          python -m pip install -e ".[dev]"
      - name: Ruff format
        run: python -m ruff format --check .
      - name: Ruff lint
        run: python -m ruff check .
      - name: Mypy
        run: python -m mypy impact_split
      - name: Bandit
        run: python -m bandit -r impact_split -ll
      - name: Build distributions
        run: python -m build --no-isolation
      - name: Validate package metadata
        run: python -m twine check dist/*
```

- [ ] **Step 8.2: Create `.github/workflows/release.yml`:**

```yaml
name: Release

on:
  push:
    tags: ["v*"]

jobs:
  build:
    runs-on: ubuntu-latest
    permissions:
      contents: write
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.13"
      - name: Build distributions
        run: |
          python -m pip install --upgrade pip build twine
          python -m build
          python -m twine check dist/*
      - name: Attach artifacts to GitHub release
        uses: softprops/action-gh-release@v2
        with:
          files: dist/*
      # PyPI publishing is a manual owner step:
      #   either `python -m twine upload dist/*` with your PyPI API token,
      #   or wire PyPI Trusted Publishing to this workflow and append
      #   `pypa/gh-action-pypi-publish@release/v1` here.
```

- [ ] **Step 8.3: Local build + fresh-venv wheel smoke:**

```bash
python -m build --no-isolation && python -m twine check dist/*
python -m venv .venv-smoke
.venv-smoke/Scripts/python -m pip install --quiet dist/*.whl
.venv-smoke/Scripts/python -c "
import numpy as np, pandas as pd
from impact_split import ImpactSplitter
X = pd.DataFrame({'a': ['x', 'y'] * 50, 'b': ['p', 'q', 'r', 'p'] * 25})
y = pd.Series(np.r_[np.ones(50) * 3, -np.ones(50) * 2])
m = ImpactSplitter().fit(X, y)
print(m.summary()[:200])
m.plot_segments(show=False); m.plot_tree(show=False)
assert '<!doctype html>' in m.to_html()
print('wheel smoke OK')
"
rm -rf .venv-smoke
```

Expected: `wheel smoke OK` and only numpy/pandas/matplotlib (+ their own deps) installed.
- [ ] **Step 8.4: Commit**

```bash
git add .github && git commit -m "ci: 3.10-3.13 test matrix; tag-triggered release build workflow"
```

---

### Task 9: Benchmark reproduction, merge, push

- [ ] **Step 9.1: Reproduce the frozen suite** (algorithm untouched, so scores must match `reports/validation-report-v3.md` exactly):

```bash
python -m benchmarks.run --tag postpkg-synth
python -m benchmarks.run --tag postpkg-kaggle --kaggle
```

Expected: Kaggle mean impact-F1 `0.9617`, floor `0.8154`, conservation True; synthetic numbers identical to the v3 report. Any drift = a bug in this refactor — stop and diagnose (`get_impact_segments` values must be byte-identical for the first four columns).
- [ ] **Step 9.2: Full gate sweep:** `python -m pytest -q && python -m ruff format --check . && python -m ruff check . && python -m mypy impact_split && python -m bandit -r impact_split -ll`
- [ ] **Step 9.3: Merge and push** (finishing-a-development-branch skill applies):

```bash
git checkout main && git merge --no-ff package-and-viz -m "Merge package-and-viz: v0.1.0 release-ready packaging + interpretability output suite"
git push origin main
```

- [ ] **Step 9.4: Hand the release to the user.** Remaining owner-only steps (do NOT attempt): create PyPI account/token or Trusted Publishing, `git tag v0.1.0 && git push origin v0.1.0`, `python -m twine upload dist/*`.
- [ ] **Step 9.5: ai-os bookkeeping** (from `C:/Users/juedi.eugenio/Documents/ai-os`): update `projects/impact-split.PROJECT.md` (status/version/output-suite note), run `python engine/aios.py maintain`, commit `aios.db.sql` + manifest per golden rules. Never run `project git-init` on this folder.

---

## Self-Review Notes

- Spec coverage: Part 1 → Tasks 1, 8; Part 2.1 → Task 2; 2.2 structure → Tasks 2–6; 2.3 → Task 3; 2.4 → Tasks 4–5; 2.5 → Task 6; 2.6 removals → Tasks 5–6; Part 3 testing → Tasks 2–6, 8.3; Part 4 delivery → Tasks 0, 9.
- Type consistency: renderers consume `payload: dict[str, Any]`; model methods delegate via `self.to_dict()`; `fmt_num`/`fmt_pct`/color constants live in `viz.data` and are imported by `text`/`static`/`html`.
- Known judgment calls an implementer may exercise: exact matplotlib padding numbers, JS clipping constants (6.4 px/char), and label-fit heuristics — visual polish is verified by eyeballing Step 7.3's artifacts, not by tests.
