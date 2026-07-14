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
