"""Interactive plotting helpers for impact_split."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import html
import json
from pathlib import Path
from typing import Any
from uuid import uuid4

SelectionCallback = Callable[[dict[str, Any]], None]


@dataclass(frozen=True)
class GraphSpec:
    """Validated graph payload for frontend rendering."""

    nodes: list[dict[str, Any]]
    links: list[dict[str, Any]]


def _validate_graph_spec(nodes: list[dict[str, Any]], links: list[dict[str, Any]]) -> GraphSpec:
    node_ids = set()
    for node in nodes:
        if "id" not in node:
            raise ValueError("Each node must include an 'id' field.")
        node_id = str(node["id"])
        if node_id in node_ids:
            raise ValueError(f"Duplicate node id found: {node_id}")
        node_ids.add(node_id)

    for link in links:
        if "source" not in link or "target" not in link:
            raise ValueError("Each link must include 'source' and 'target' fields.")
        if str(link["source"]) not in node_ids:
            raise ValueError(f"Link source does not exist in nodes: {link['source']}")
        if str(link["target"]) not in node_ids:
            raise ValueError(f"Link target does not exist in nodes: {link['target']}")
    return GraphSpec(nodes=nodes, links=links)


def _validate_selection_event(payload: dict[str, Any]) -> dict[str, Any]:
    """Validate callback payload from frontend selection events."""
    required_keys = {"event_type", "selected_node_id", "filters"}
    missing_keys = required_keys - set(payload)
    if missing_keys:
        raise ValueError(f"Selection event missing keys: {sorted(missing_keys)}")

    if payload["event_type"] != "node_click":
        raise ValueError("Selection event 'event_type' must be 'node_click'.")

    selected_node_id = payload["selected_node_id"]
    if selected_node_id is not None and not isinstance(selected_node_id, str):
        raise ValueError("Selection event 'selected_node_id' must be a string or null.")

    filters = payload["filters"]
    if not isinstance(filters, dict):
        raise ValueError("Selection event 'filters' must be a dictionary.")

    return {
        "event_type": "node_click",
        "selected_node_id": selected_node_id,
        "filters": filters,
    }


def _build_render_script(
    *,
    container_id: str,
    graph_state_json: str,
    comm_target_name: str | None,
) -> str:
    callback_snippet = "const callbackEnabled = false;"
    if comm_target_name is not None:
        callback_snippet = f"""
const callbackEnabled = true;
const commTargetName = {json.dumps(comm_target_name)};
let comm = null;
if (window.Jupyter && window.Jupyter.notebook && window.Jupyter.notebook.kernel) {{
  comm = window.Jupyter.notebook.kernel.comm_manager.new_comm(commTargetName, {{}});
}}
"""

    return f"""
(function() {{
  const container = document.getElementById({json.dumps(container_id)});
  if (!container) return;
  const state = {graph_state_json};
  {callback_snippet}
  const width = state.options.width || 900;
  const height = state.options.height || 560;
  let drewAnything = false;
  const tooltip = document.createElement("div");
  tooltip.style.position = "absolute";
  tooltip.style.pointerEvents = "none";
  tooltip.style.opacity = "0";
  tooltip.style.background = "rgba(20, 20, 20, 0.85)";
  tooltip.style.color = "#fff";
  tooltip.style.padding = "6px 8px";
  tooltip.style.borderRadius = "4px";
  tooltip.style.fontSize = "12px";
  container.style.position = "relative";
  container.innerHTML = "";
  container.appendChild(tooltip);

  const selectedFilters = Object.assign({{}}, state.active_filters || {{}});
  const filterKeys = state.filter_keys || [];

  function filterNodes(nodes) {{
    if (!filterKeys.length) return nodes;
    return nodes.filter((n) => {{
      return filterKeys.every((key) => {{
        const selected = selectedFilters[key];
        return selected === "__all__" || String(n[key]) === selected;
      }});
    }});
  }}

  function filterLinks(links, nodeSet) {{
    return links.filter((l) => nodeSet.has(String(l.source)) && nodeSet.has(String(l.target)));
  }}

  const nodes = filterNodes(state.nodes).map((n) => Object.assign({{}}, n));
  const nodeSet = new Set(nodes.map((n) => String(n.id)));
  const links = filterLinks(state.links, nodeSet).map((l) => Object.assign({{}}, l));

  const svg = d3.select(container)
    .append("svg")
    .attr("viewBox", [0, 0, width, height])
    .attr("width", width)
    .attr("height", height)
    .style("max-width", "100%")
    .style("height", "auto");

  const g = svg.append("g");
  drewAnything = true;
  const zoom = d3.zoom().scaleExtent([0.3, 6]).on("zoom", (event) => g.attr("transform", event.transform));
  svg.call(zoom);

  const link = g.append("g")
    .attr("stroke", "#999")
    .attr("stroke-opacity", 0.65)
    .selectAll("line")
    .data(links)
    .join("line")
    .attr("stroke-width", (d) => Math.max(1, Math.sqrt(d.value || 1)));

  const node = g.append("g")
    .attr("stroke", "#fff")
    .attr("stroke-width", 1.3)
    .selectAll("circle")
    .data(nodes)
    .join("circle")
    .attr("r", (d) => d.radius || 7)
    .attr("fill", (d) => d.color || "#4C78A8")
    .style("cursor", "pointer");

  const labels = g.append("g")
    .selectAll("text")
    .data(nodes)
    .join("text")
    .text((d) => d.label || d.id)
    .attr("font-size", 10)
    .attr("dx", 10)
    .attr("dy", 3)
    .attr("fill", "#222");

  function sendSelection(nodeId) {{
    if (!callbackEnabled || !comm) return;
    const payload = {{
      event_type: "node_click",
      selected_node_id: nodeId,
      filters: selectedFilters,
    }};
    comm.send(payload);
  }}

  node.on("mouseover", function(event, d) {{
    tooltip.style.opacity = "1";
    tooltip.textContent = d.tooltip || d.label || d.id;
  }})
  .on("mousemove", function(event) {{
    const rect = container.getBoundingClientRect();
    tooltip.style.left = (event.clientX - rect.left + 10) + "px";
    tooltip.style.top = (event.clientY - rect.top + 10) + "px";
  }})
  .on("mouseout", function() {{
    tooltip.style.opacity = "0";
  }})
  .on("click", function(event, d) {{
    node.attr("stroke", "#fff").attr("stroke-width", 1.3);
    d3.select(this).attr("stroke", "#111").attr("stroke-width", 3);
    sendSelection(String(d.id));
  }});

  const simulation = d3.forceSimulation(nodes)
    .force("link", d3.forceLink(links).id((d) => d.id).distance(65))
    .force("charge", d3.forceManyBody().strength(state.options.charge_strength || -110))
    .force("x", d3.forceX(width / 2).strength(0.08))
    .force("y", d3.forceY(height / 2).strength(0.08))
    .force("collide", d3.forceCollide().radius((d) => (d.radius || 7) + 2));

  function drag(simulation) {{
    function dragstarted(event, d) {{
      if (!event.active) simulation.alphaTarget(0.25).restart();
      d.fx = d.x;
      d.fy = d.y;
    }}
    function dragged(event, d) {{
      d.fx = event.x;
      d.fy = event.y;
    }}
    function dragended(event, d) {{
      if (!event.active) simulation.alphaTarget(0);
      d.fx = null;
      d.fy = null;
    }}
    return d3.drag().on("start", dragstarted).on("drag", dragged).on("end", dragended);
  }}

  node.call(drag(simulation));

  simulation.on("tick", () => {{
    link
      .attr("x1", (d) => d.source.x)
      .attr("y1", (d) => d.source.y)
      .attr("x2", (d) => d.target.x)
      .attr("y2", (d) => d.target.y);
    node.attr("cx", (d) => d.x).attr("cy", (d) => d.y);
    labels.attr("x", (d) => d.x).attr("y", (d) => d.y);
  }});

  setTimeout(() => {{
    if (!drewAnything) {{
      container.innerHTML = "<div style='padding:8px;color:#b00020;'>Graph did not initialize. Check browser console and D3 loading.</div>";
    }}
  }}, 1200);
}})();
"""


class InteractiveForceGraph:
    """Notebook + standalone-HTML hybrid force-graph renderer."""

    def __init__(
        self,
        *,
        nodes: list[dict[str, Any]],
        links: list[dict[str, Any]],
        filter_keys: list[str] | None = None,
        options: dict[str, Any] | None = None,
        on_selection: SelectionCallback | None = None,
        d3_url: str = "https://cdn.jsdelivr.net/npm/d3@7/dist/d3.min.js",
    ) -> None:
        spec = _validate_graph_spec(nodes=nodes, links=links)
        self.nodes = spec.nodes
        self.links = spec.links
        self.filter_keys = filter_keys or []
        self.options = options or {}
        self.on_selection = on_selection
        self.d3_url = d3_url

    def _serialize_state(self, active_filters: dict[str, str] | None = None) -> dict[str, Any]:
        resolved_filters = {}
        active_filters = active_filters or {}
        for key in self.filter_keys:
            resolved_filters[key] = active_filters.get(key, "__all__")
        return {
            "nodes": self.nodes,
            "links": self.links,
            "filter_keys": self.filter_keys,
            "active_filters": resolved_filters,
            "options": self.options,
        }

    def _register_callback_comm(self) -> str | None:
        on_selection = self.on_selection
        if on_selection is None:
            return None
        try:
            from ipykernel.comm import Comm
            from IPython import get_ipython
        except Exception:
            return None

        ipy = get_ipython()
        if ipy is None or getattr(ipy, "kernel", None) is None:
            return None

        target_name = f"impact_split_select_{uuid4().hex}"

        def _target(comm: Comm, _open_msg: dict[str, Any]) -> None:
            @comm.on_msg
            def _recv(msg: dict[str, Any]) -> None:
                payload = msg.get("content", {}).get("data", {})
                validated = _validate_selection_event(payload)
                on_selection(validated)

        ipy.kernel.comm_manager.register_target(target_name, _target)
        return target_name

    def to_html(self, active_filters: dict[str, str] | None = None) -> str:
        container_id = f"impact-split-force-{uuid4().hex}"
        state_json = json.dumps(self._serialize_state(active_filters))
        render_script = _build_render_script(
            container_id=container_id,
            graph_state_json=state_json,
            comm_target_name=None,
        )
        return (
            f'<div id="{html.escape(container_id)}"></div>\n'
            f'<script src="{html.escape(self.d3_url)}"></script>\n'
            f"<script>{render_script}</script>"
        )

    def save_html(
        self,
        output_path: str | Path,
        active_filters: dict[str, str] | None = None,
    ) -> Path:
        out = Path(output_path)
        out.write_text(self.to_html(active_filters=active_filters), encoding="utf-8")
        return out

    def show(self) -> Any:
        try:
            from IPython.display import HTML, display
        except Exception:
            return self.to_html()

        try:
            import ipywidgets as widgets
        except Exception:
            return display(HTML(self.to_html()))

        control_widgets: dict[str, Any] = {}
        active_filters: dict[str, str] = {}
        for key in self.filter_keys:
            values = sorted({str(node.get(key, "")) for node in self.nodes})
            options = ["__all__"] + values
            dropdown = widgets.Dropdown(
                options=options,
                value="__all__",
                description=f"{key}:",
            )
            control_widgets[key] = dropdown
            active_filters[key] = "__all__"

        graph_output = widgets.Output()

        def _render_graph() -> None:
            with graph_output:
                graph_output.clear_output(wait=True)
                # VS Code/Jupyter frontends are more consistent with iframe srcdoc rendering
                # than injecting separate Javascript display payloads in output areas.
                html_payload = self.to_html(active_filters=active_filters)
                escaped_payload = html.escape(html_payload, quote=True)
                iframe_height = int(self.options.get("height", 560)) + 40
                display(
                    HTML(
                        f'<iframe srcdoc="{escaped_payload}" '
                        f'style="width:100%;height:{iframe_height}px;border:0;"></iframe>'
                    )
                )

        def _on_filter_change(change: dict[str, Any], key: str) -> None:
            active_filters[key] = change["new"]
            _render_graph()

        for key, widget in control_widgets.items():
            widget.observe(lambda change, k=key: _on_filter_change(change, k), names="value")

        _render_graph()
        controls_box = widgets.VBox(list(control_widgets.values()))
        return display(widgets.VBox([controls_box, graph_output]))


def interactive_force_graph(
    *,
    nodes: list[dict[str, Any]],
    links: list[dict[str, Any]],
    filter_keys: list[str] | None = None,
    options: dict[str, Any] | None = None,
    on_selection: SelectionCallback | None = None,
    d3_url: str = "https://cdn.jsdelivr.net/npm/d3@7/dist/d3.min.js",
) -> InteractiveForceGraph:
    """Create an interactive force graph renderer."""
    return InteractiveForceGraph(
        nodes=nodes,
        links=links,
        filter_keys=filter_keys,
        options=options,
        on_selection=on_selection,
        d3_url=d3_url,
    )
