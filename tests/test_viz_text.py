"""Tests for the text renderer (summary / repr)."""

import numpy as np
import pandas as pd
import pytest
from tests.test_viz_data import _fitted, churn_mix_fitted, fitted

from impact_split import ImpactSplitter
from impact_split.viz.text import render_summary


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


def test_summary_flags_churn_segments() -> None:
    text = churn_mix_fitted().summary()
    assert "lookahead=True" in text
    assert "churn ⇄" in text  # segments ledger line
    assert "gross ⇄" in text  # table column header
    assert " / -" in text  # gross column rendered for the churn row
    assert "offsetting mass" in text  # footnote


def test_summary_unchanged_without_ensemble_and_annotated_with() -> None:
    model, X, y = _fitted()
    before = model.summary()
    model.ensemble_report(X, y, n_replicates=12, shadow_replicates=0, seed=3)
    after = model.summary()
    assert "stability" in after and "stability" not in before
    assert "Σy 5–95%" in after
    # shadow section only when shadows exist
    if model.ensemble_["shadows"]:
        assert "Shadow drivers" in after


def test_render_summary_ensemble_formatting_branches() -> None:
    """Synthetic payload exercising every ensemble-annotation branch directly
    (stable+CI, fragile+no-CI, missing-from-map, shadow section) without
    depending on shadow_replicates>0 producing real shadows in a live run."""
    model = fitted()
    payload = model.to_dict()
    seg_ids = [s["segment_id"] for s in payload["segments"]]
    assert len(seg_ids) >= 3, "fixture must produce >=3 segments for this test"
    s0, s1 = seg_ids[0], seg_ids[1]

    payload["ensemble"] = {
        "config": {},
        "segments": {
            s0: {
                "stability": 0.9,
                "n_matched": 9,
                "ci_low": 1.5,
                "ci_high": 3.5,
                "fragile": False,
            },
            s1: {
                "stability": 0.3,
                "n_matched": 3,
                "ci_low": None,
                "ci_high": None,
                "fragile": True,
            },
            # s2 intentionally omitted -> exercises the "segment absent from
            # the ensemble map" em-dash fallback branch.
        },
        "importance": [],
        "shadows": [
            {
                "path": "shadow-test / synthetic=1",
                "features": ["a", "b"],
                "recurrence": 0.4,
                "mean_impact": 123.0,
                "n_members": 3,
                "block": "shadow",
            }
        ],
    }

    text = render_summary(payload)
    lines = text.splitlines()
    header_idx = lines.index("Top segments by |impact|")
    # header_idx+1 is the column-header row; data rows follow in payload order.
    row_lines = lines[header_idx + 2 : header_idx + 2 + len(seg_ids)]

    # s0: stable, real CI -> formatted stability + bracketed CI, no dagger.
    row0 = row_lines[0]
    assert "90.0%" in row0
    assert "[+1.50, +3.50]" in row0
    assert "†" not in row0
    assert row0.count("—") == 0

    # s1: fragile, no CI -> dagger marker + a single em-dash for the CI column.
    row1 = row_lines[1]
    assert "30.0% †" in row1
    assert row1.count("—") == 1

    # s2: absent from the ensemble map -> em-dash fallback in BOTH columns.
    row2 = row_lines[2]
    assert row2.count("—") == 2

    # fragile footnote appears (only s1 is fragile).
    assert " † fragile: segment re-emerged in <50% of bootstrap refits." in text

    # shadow drivers section.
    assert "Shadow drivers" in text
    shadow_line = next(line for line in lines if "shadow-test / synthetic=1" in line)
    assert "Σy≈+123.00" in shadow_line
    assert "recurrence 40.0%" in shadow_line
    assert "via a, b" in shadow_line


def _segment_with_path(path: str) -> dict:
    return {
        "segment_id": "s0",
        "path": path,
        "node_ids": ["node_0"],
        "n": 10,
        "total_sum": 1.0,
        "mean": 0.1,
        "pool_share": 0.5,
        "pos_sum": 1.0,
        "neg_sum": 0.0,
        "is_churn": False,
    }


def test_render_summary_default_width_does_not_collide_two_qualified_paths() -> None:
    """Regression for the pre-fix 44-char default: two segments differing only
    after the old cutoff used to render as identical rows."""
    payload = fitted().to_dict()
    path_a = "root / dim_customer_master.customer_segment=A"
    path_b = "root / dim_customer_master.customer_segment=B"
    assert path_a[:43] == path_b[:43]  # would have collided under the old default
    payload["segments"] = [_segment_with_path(path_a), _segment_with_path(path_b)]

    text = render_summary(payload)

    rows = [line for line in text.splitlines() if "customer_segment=" in line]
    assert len(rows) == 2
    assert rows[0] != rows[1]
    assert "customer_segment=A" in rows[0]
    assert "customer_segment=B" in rows[1]


def test_render_summary_default_width_fits_a_realistic_two_hop_qualified_path() -> None:
    """A two-hop snowflake path with realistic table/column names must render
    in full, not lose its last hop to truncation."""
    path = "root / dim_customer_master.customer_segment=A / dim_geo_region.region_name=Europe"
    payload = fitted().to_dict()
    payload["segments"] = [_segment_with_path(path)]

    text = render_summary(payload)

    assert path in text
    assert "…" not in text


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
