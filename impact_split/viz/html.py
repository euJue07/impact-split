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
        ("churn segments", str(meta["n_churn_segments"])),
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
        f"lookahead={p['lookahead']} · impact_split v{meta['package_version']}"
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
.tband { position:absolute; top:1px; height:14px; border:1px dashed #949494;
         border-radius:4px; background:repeating-linear-gradient(45deg, transparent,
         transparent 3px, rgba(148,148,148,0.35) 3px, rgba(148,148,148,0.35) 6px); }
.twhisk { position:absolute; top:7.25px; height:1.5px; background:#1a1a17; }
.twhisk-cap { position:absolute; top:4px; width:1.5px; height:8px; background:#1a1a17; }
.churn-mark { color:var(--muted); font-weight:600; }
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
consolidated segment · hover for the full rule path. Dashed-outlined leaves are churn
(offsetting ±flows both material).</p>
<div id="breadcrumb"></div>
<svg id="icicle"></svg>

<h2>Segments ranked by |impact|</h2>
<p class="hint">Each bar is a consolidated segment's total Σy. Bars are additive —
together they reconstruct the total exactly. Hover to locate the segment in the tree.
Churn segments (⇄) also show a hatched band spanning their gross ±flows — the band is
not additive.</p>
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
var ens = DATA.ensemble || null;

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
function fmtMag(v) {
  if (v === null || v === undefined) return "—";
  var a = Math.abs(v);
  if (a >= 1000) return Math.round(a).toLocaleString("en-US");
  return a.toLocaleString("en-US", { maximumFractionDigits: 2 });
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
    "Σy = " + fmt(n.total_sum) + "   Σy⁺ = " + fmt(n.pos_sum) + "   Σy⁻ = " +
      (n.neg_sum ? "-" : "") + fmtMag(n.neg_sum)];
  if (n.is_churn) {
    lines.push("churn ⇄ net hides ±" +
      fmtMag(Math.min(n.pos_sum || 0, n.neg_sum || 0)) + " offsetting mass");
  }
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
    var churn = n.is_leaf && n.is_churn;
    var dark = merged || churn;
    out += '<rect data-node="' + n.id + '"' +
      (n.segment_id ? ' data-seg="' + n.segment_id + '"' : "") +
      ' x="' + (r.x0 + PAD / 2) + '" y="' + (r.level * ROW_H + PAD / 2) + '"' +
      ' width="' + Math.max(0.6, r.w - PAD) + '" height="' + (ROW_H - PAD) + '"' +
      ' fill="' + col.css + '" stroke="' + (dark ? "#3a3a36" : "#ffffff") + '"' +
      ' stroke-width="' + (dark ? 2 : 1) + '"' +
      (churn ? ' stroke-dasharray="6,3"' : "") + ' rx="3"></rect>';
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
             seg: s.segment_id, rolled: false, churn: !!s.is_churn,
             pos: s.pos_sum || 0, neg: s.neg_sum || 0 };
  });
  if (rest.length) {
    rows.push({
      path: "(+" + rest.length + " more segments)",
      v: rest.reduce(function (a, s) { return a + (s.total_sum || 0); }, 0),
      n: rest.reduce(function (a, s) { return a + s.n; }, 0),
      share: null, seg: null, rolled: true, churn: false, pos: 0, neg: 0
    });
  }
  var lo = 0, hi = 0;
  rows.forEach(function (r) {
    lo = Math.min(lo, r.v); hi = Math.max(hi, r.v);
    if (r.churn) { lo = Math.min(lo, -r.neg); hi = Math.max(hi, r.pos); }
    if (ens && r.seg) {
      var stDom = ens.segments[r.seg];
      if (stDom && stDom.ci_low !== null) {
        lo = Math.min(lo, stDom.ci_low); hi = Math.max(hi, stDom.ci_high);
      }
    }
  });
  var range = (hi - lo) || 1;
  var zeroPct = (0 - lo) / range * 100;
  var out = "";
  rows.forEach(function (r) {
    var leftPct = (Math.min(r.v, 0) - lo) / range * 100;
    var widthPct = Math.max(Math.abs(r.v) / range * 100, 0.4);
    var cls = r.rolled ? "rolled" : (r.v >= 0 ? "pos" : "neg");
    var note = "n=" + r.n.toLocaleString("en-US") +
      (r.share != null ? " · " + pct(r.share) + " of " + (r.v >= 0 ? "Σy⁺" : "Σy⁻") : "");
    var band = "";
    if (r.churn) {
      var bLeft = (-r.neg - lo) / range * 100;
      var bWidth = (r.pos + r.neg) / range * 100;
      band = '<div class="tband" style="left:' + bLeft + "%;width:" + bWidth + '%"></div>';
    }
    if (r.churn) {
      note += " · gross +" + fmtMag(r.pos) + "/−" + fmtMag(r.neg);
    }
    var whisker = "";
    if (ens && r.seg) {
      var st = ens.segments[r.seg];
      if (st && st.ci_low !== null) {
        var wLeft = (st.ci_low - lo) / range * 100;
        var wRight = (st.ci_high - lo) / range * 100;
        whisker = '<div class="twhisk" style="left:' + wLeft + "%;width:" +
          (wRight - wLeft) + '%"></div>' +
          '<div class="twhisk-cap" style="left:' + wLeft + '%"></div>' +
          '<div class="twhisk-cap" style="left:' + wRight + '%"></div>';
        note += " · CI [" + fmt(st.ci_low) + ", " + fmt(st.ci_high) + "]";
      }
    }
    out += '<div class="trow"' + (r.seg ? ' data-seg="' + r.seg + '"' : "") + ">" +
      '<div class="tpath" title="' + esc(r.path) + '">' + esc(r.path) + "</div>" +
      '<div class="ttrack"><div class="tzero" style="left:' + zeroPct + '%"></div>' + band +
      '<div class="tbar ' + cls + '" style="left:' + leftPct + "%;width:" + widthPct +
      '%"></div>' + whisker + '</div>' +
      '<div class="tval">' + (r.churn ? "net " : "") + fmt(r.v) + "</div>" +
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
           seg: s.segment_id, pos_sum: s.pos_sum || 0, neg_sum: s.neg_sum || 0,
           churn: !!s.is_churn };
});
var sortKey = "rank", sortDir = 1;
var COLS = [
  ["#", "rank"], ["path", "path"], ["Σy", "total_sum"], ["Σy⁺", "pos_sum"],
  ["Σy⁻", "neg_sum"], ["n", "n"], ["mean", "mean"], ["pool share", "pool_share"],
  ["leaves", "leaves"]
];
function renderTable() {
  var thead = document.querySelector("#segtable thead");
  var tbody = document.querySelector("#segtable tbody");
  var ensHead = ens ? "<th>stability</th><th>Σy 5–95%</th>" : "";
  thead.innerHTML = "<tr>" + COLS.map(function (c) {
    var mark = c[1] === sortKey ? (sortDir > 0 ? " ▲" : " ▼") : "";
    return '<th class="' + (c[1] === "path" ? "path" : "") + '" data-key="' + c[1] + '">' +
      esc(c[0]) + mark + "</th>";
  }).join("") + ensHead + "</tr>";
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
    var ensCells = "";
    if (ens) {
      var st = ens.segments[r.seg];
      var stab = st ? pct(st.stability) + (st.fragile ? " †" : "") : "—";
      var ci = st && st.ci_low !== null
        ? "[" + fmt(st.ci_low) + ", " + fmt(st.ci_high) + "]" : "—";
      ensCells = "<td>" + esc(stab) + "</td><td>" + esc(ci) + "</td>";
    }
    return '<tr data-seg="' + r.seg + '">' +
      "<td>" + r.rank + "</td>" +
      '<td class="path">' + chip +
      (r.churn ? '<span class="churn-mark" title="churn: offsetting flows both material">⇄ </span>' : "") +
      esc(r.path) + "</td>" +
      "<td>" + fmt(r.total_sum) + "</td>" +
      "<td>" + fmtMag(r.pos_sum) + "</td>" +
      "<td>" + (r.neg_sum ? "−" : "") + fmtMag(r.neg_sum) + "</td>" +
      "<td>" + r.n.toLocaleString("en-US") + "</td>" +
      "<td>" + fmt(r.mean) + "</td>" +
      "<td>" + (r.pool_share != null
        ? pct(r.pool_share) + " of " + (r.total_sum >= 0 ? "Σy⁺" : "Σy⁻") : "—") + "</td>" +
      "<td>" + r.leaves + "</td>" + ensCells + "</tr>";
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

function renderEnsembleExtras() {
  if (!ens) return;
  var out = "";
  var anyFragile = Object.keys(ens.segments).some(function (k) {
    return ens.segments[k].fragile;
  });
  if (anyFragile) {
    out += '<p class="hint">† fragile: segment re-emerged in less than 50% of ' +
      "bootstrap refits.</p>";
  }
  if (ens.shadows.length) {
    out += "<h2>Shadow drivers</h2>" +
      '<p class="hint">material regions the main tree does not report — found by ' +
      "feature-subsampled refits</p>" +
      '<table><thead><tr><th class="path">path</th><th>Σy</th><th>recurrence</th>' +
      '<th class="path">features</th><th class="path">block</th></tr></thead><tbody>' +
      ens.shadows.map(function (sh) {
        return "<tr>" +
          '<td class="path">' + esc(sh.path) + "</td>" +
          "<td>" + fmt(sh.mean_impact) + "</td>" +
          "<td>" + pct(sh.recurrence) + "</td>" +
          '<td class="path">' + esc(sh.features.join(", ")) + "</td>" +
          '<td class="path">' + esc(sh.block) + "</td></tr>";
      }).join("") +
      "</tbody></table>";
  }
  if (ens.importance.length) {
    out += "<h2>Ensemble importance</h2>" +
      '<table><thead><tr><th class="path">feature</th><th>importance</th>' +
      "<th>n_trees</th></tr></thead><tbody>" +
      ens.importance.map(function (r) {
        return "<tr>" +
          '<td class="path">' + esc(r.feature) + "</td>" +
          "<td>" + r.importance.toFixed(4) + "</td>" +
          "<td>" + r.n_trees.toLocaleString("en-US") + "</td></tr>";
      }).join("") +
      "</tbody></table>";
  }
  if (out) {
    document.getElementById("tooltip").insertAdjacentHTML("beforebegin", out);
  }
}

renderIcicle();
renderBreadcrumb();
renderTornado();
renderTable();
renderEnsembleExtras();
</script>
</body>
</html>
"""
)
