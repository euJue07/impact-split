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
    # segments sorted by max(|impact|, churn mass) descending
    keys = [
        max(
            abs(s["total_sum"]),
            min(s["pos_sum"], s["neg_sum"]) if s["is_churn"] else 0.0,
        )
        for s in payload["segments"]
    ]
    assert keys == sorted(keys, reverse=True)


def test_payload_json_safe() -> None:
    payload = fitted().to_dict()
    text = json.dumps(payload, allow_nan=False)
    assert json.loads(text)["meta"]["n_rows"] == 400


def test_get_impact_segments_gains_columns_after_existing() -> None:
    df = fitted().get_impact_segments()
    assert list(df.columns) == ["path", "total_sum", "n_samples", "node_id", "mean", "pool_share"]
    assert (df["mean"] == df["total_sum"] / df["n_samples"]).all()
    assert (df["pool_share"].dropna() >= 0).all()


def test_payload_params_json_safe_with_numpy_scalars() -> None:
    """numpy scalar constructor args must not break json.dumps(..., allow_nan=False)."""
    X, y = demo_frame()
    model = ImpactSplitter(
        max_depth=np.int64(5),
        delta_pct=np.float64(0.01),
        min_global_impact_pct=np.float64(0.01),
        noise_z=np.float64(3.0),
    ).fit(X, y)
    text = json.dumps(model.to_dict(), allow_nan=False)
    params = json.loads(text)["meta"]["params"]
    assert params["max_depth"] == 5
    assert isinstance(params["max_depth"], int)
    assert params["delta_pct"] == pytest.approx(0.01)
    assert params["consolidate"] is True


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
