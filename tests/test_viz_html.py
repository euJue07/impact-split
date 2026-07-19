"""Tests for the self-contained HTML report."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from impact_split import ImpactSplitter
from tests.test_viz_data import churn_mix_fitted, demo_frame, fitted


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


def test_html_marks_churn() -> None:
    html_out = churn_mix_fitted().to_html()
    assert "churn segments" in html_out   # ledger tile
    assert "lookahead=" in html_out       # params line
    assert "tband" in html_out            # gross-band CSS + tornado renderer
    assert "stroke-dasharray" in html_out # icicle churn outline
    assert "Σy⁺" in html_out              # gross table columns
    # Data-dependent: only holds when churn exists in the output
    assert '"is_churn": true' in html_out
    # Verify the assertion is data-dependent by confirming it does NOT appear
    # in a churn-free model (strictly non-negative y, so negative pool is 0).
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"a": rng.choice(["x", "y"], size=200)})
    y = pd.Series(np.abs(rng.normal(0.0, 1.0, 200)) + (X["a"] == "x") * 5.0)
    html_no_churn = ImpactSplitter().fit(X, y).to_html()
    assert '"is_churn": true' not in html_no_churn


def test_html_unchanged_without_ensemble_and_annotated_with() -> None:
    # The report is one static template whose embedded JS renders everything from
    # the payload JSON, so JS-source strings like "stability" or "Shadow drivers"
    # are always present in the script text -- they're runtime-gated on
    # payload.ensemble, not physically absent. The genuinely data-dependent,
    # testable surface (same convention as test_html_marks_churn's is_churn
    # check) is the embedded JSON: no "ensemble" key at all until ensemble_report
    # runs, then the full config/segments/importance/shadows block appears.
    # Build X/y directly (fitted()'s own convention) rather than reaching into
    # the model's private _X/_y so ensemble_report gets the exact training data.
    X, y = demo_frame()
    model = ImpactSplitter().fit(X, y)
    before = model.to_html()
    assert '"ensemble"' not in before
    model.ensemble_report(X, y, n_replicates=12, shadow_replicates=0, seed=3)
    after = model.to_html()
    assert '"ensemble"' in after
    assert '"stability":' in after
    assert '"shadows": []' in after  # shadow_replicates=0 -> no shadows found
    # the annotation JS itself (column headers, whisker CSS, section builders) is
    # part of the template regardless of data -- confirm it actually shipped
    for marker in ("stability", "Σy 5", "twhisk", "Shadow drivers", "Ensemble importance"):
        assert marker in after


def test_html_ensemble_importance_and_whiskers() -> None:
    """Data-dependent assertions on the embedded JSON, not JS/CSS source text.

    "Shadow drivers"/"Ensemble importance"/"twhisk"/"n_trees" (as bare strings)
    are baked unconditionally into _TEMPLATE and would pass even if
    ensemble_report were never called -- so every assertion here targets a
    JSON fragment that can only appear once real ensemble data is serialized.
    """
    from tests.test_ensemble import _masked_driver_data

    model, X, y = _masked_driver_data()
    baseline = model.to_html()
    assert '"ensemble"' not in baseline

    model.ensemble_report(
        X, y, n_replicates=10, shadow_replicates=30, feature_subsample=0.5,
        match_threshold=0.5, shadow_min_stability=0.2, seed=13,
    )
    assert model.ensemble_["shadows"], "fixture must produce real shadows"
    assert model.ensemble_["importance"], "fixture must produce real importance rows"
    html_out = model.to_html()

    # Real shadows serialize actual field values -- an empty shadows list would
    # instead produce the literal "shadows": [] seen in the sibling test above.
    assert '"shadows": []' not in html_out
    assert '"block": "shadow"' in html_out
    assert '"recurrence":' in html_out

    # Segment-level stability/CI annotations landed in the payload.
    assert '"stability":' in html_out
    assert '"ci_low":' in html_out

    # "feature_index" is a JSON-only key -- the template's JS never reads it
    # (only .feature/.importance/.n_trees), so it can only come from a real
    # serialized importance row.
    assert '"feature_index":' in html_out

    # "n_trees" itself is ambiguous (it's also the always-present <th> label),
    # but real importance rows add one more occurrence per row on top of that
    # baseline -- a strict count increase is genuinely data-dependent.
    assert html_out.count("n_trees") > baseline.count("n_trees")
