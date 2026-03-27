"""Tests for interactive force graph serialization and export."""

from pathlib import Path

import pytest

from impact_split.plots import (
    _validate_selection_event,
    interactive_force_graph,
)


def _sample_graph() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    nodes = [
        {"id": "A", "label": "Node A", "group": "g1"},
        {"id": "B", "label": "Node B", "group": "g2"},
    ]
    links = [{"source": "A", "target": "B", "value": 1}]
    return nodes, links


def test_selection_event_payload_schema_accepts_valid_payload() -> None:
    payload = {
        "event_type": "node_click",
        "selected_node_id": "A",
        "filters": {"group": "g1"},
    }
    validated = _validate_selection_event(payload)
    assert validated["event_type"] == "node_click"
    assert validated["selected_node_id"] == "A"
    assert validated["filters"] == {"group": "g1"}


def test_selection_event_payload_schema_rejects_invalid_payload() -> None:
    with pytest.raises(ValueError, match="missing keys"):
        _validate_selection_event({"event_type": "node_click"})


def test_interactive_force_graph_state_and_export_contains_bootstrap(tmp_path: Path) -> None:
    nodes, links = _sample_graph()
    graph = interactive_force_graph(
        nodes=nodes,
        links=links,
        filter_keys=["group"],
        options={"width": 640, "height": 420},
    )

    state = graph._serialize_state({"group": "g1"})
    assert state["nodes"] == nodes
    assert state["links"] == links
    assert state["filter_keys"] == ["group"]
    assert state["active_filters"] == {"group": "g1"}
    assert state["options"]["width"] == 640

    html_str = graph.to_html({"group": "g2"})
    assert "<script src=" in html_str
    assert "d3.min.js" in html_str
    assert "event_type" in html_str
    assert "selected_node_id" in html_str

    out_path = tmp_path / "graph.html"
    written_path = graph.save_html(out_path, {"group": "g2"})
    assert written_path == out_path
    assert out_path.exists()
    assert "impact-split-force-" in out_path.read_text(encoding="utf-8")
