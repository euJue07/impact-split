"""Impact-driven ternary tree for additive KPIs over categorical features."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any, cast

from matplotlib.colors import Normalize
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_float_dtype

# Max decoded category labels per feature in stored segment paths (fit time).
_PATH_SEGMENT_MAX_LABELS = 8

# Short edge annotations (distinct from first-letter truncation: neutral was also "N").
_BRANCH_EDGE_SHORT: dict[str, str] = {
    "positive": "P",
    "negative": "Neg",
    "neutral": "Neu",
}


def _canvas_renderer(fig: Figure) -> Any:
    """Return Agg canvas renderer; base ``FigureCanvasBase`` stubs omit ``get_renderer``."""
    return cast(Any, fig.canvas).get_renderer()


# Color-blind–friendly edge colors; redundant with text labels for accessibility.
_DEFAULT_BRANCH_EDGE_COLORS: dict[str, str] = {
    "positive": "#0173B2",
    "negative": "#DE8F05",
    "neutral": "#949494",
}

# Minimum vertical gap between tree levels vs tallest node label (data coordinates).
_VERTICAL_LABEL_MARGIN = 1.28


def _prepare_X_y(
    X: np.ndarray | pd.DataFrame,
    y: np.ndarray | pd.Series,
    *,
    numeric_binning_strategy: str,
    numeric_n_bins: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    list[str] | None,
    tuple[np.ndarray, ...] | None,
    dict[int, np.ndarray],
]:
    """Validate inputs and return integer matrix, target, and fitted DataFrame metadata."""
    if isinstance(y, pd.Series):
        y_arr = np.asarray(y, dtype=float)
    elif isinstance(y, np.ndarray):
        y_arr = y.astype(float, copy=False)
    else:
        raise ValueError("y must be a numpy.ndarray or pandas.Series.")

    if y_arr.ndim != 1:
        raise ValueError("y must be a 1D numpy.ndarray or pandas.Series.")

    feature_names: list[str] | None = None
    category_maps: tuple[np.ndarray, ...] | None = None
    numeric_bin_edges: dict[int, np.ndarray] = {}

    if isinstance(X, pd.DataFrame):
        if X.ndim != 2:
            raise ValueError("X must be a 2D pandas.DataFrame.")
        if X.shape[0] != y_arr.shape[0]:
            raise ValueError("X and y must have the same number of rows.")
        cols: list[np.ndarray] = []
        maps: list[np.ndarray] = []
        feature_names = [str(c) for c in X.columns]
        for feature_index, name in enumerate(X.columns):
            s = X[name]
            if s.isna().any():
                raise ValueError("X must not contain missing values.")
            if is_float_dtype(s) and not is_bool_dtype(s):
                values = np.asarray(s, dtype=float)
                codes, edges, labels = _bin_numeric_values(
                    values,
                    strategy=numeric_binning_strategy,
                    n_bins=numeric_n_bins,
                )
                cols.append(codes.astype(np.int64, copy=False))
                maps.append(labels)
                numeric_bin_edges[feature_index] = edges
            else:
                codes, uniques = pd.factorize(s, sort=True)
                if np.any(codes < 0):
                    raise ValueError(
                        "X categories must be non-negative integer codes after factorization."
                    )
                cols.append(codes.astype(np.int64, copy=False))
                maps.append(np.asarray(uniques, dtype=object))
        x_arr = np.column_stack(cols) if cols else np.empty((X.shape[0], 0), dtype=np.int64)
        category_maps = tuple(maps)
    elif isinstance(X, np.ndarray):
        if X.ndim != 2:
            raise ValueError("X must be a 2D numpy.ndarray.")
        if getattr(X.dtype, "kind", None) not in ("i", "u"):
            raise ValueError(
                "X must contain integer label-encoded categories (signed or unsigned int dtype)."
            )
        if X.size and np.any(X < 0):
            raise ValueError("X categories must be non-negative integers.")
        if X.shape[0] != y_arr.shape[0]:
            raise ValueError("X and y must have the same number of rows.")
        x_arr = X.astype(np.int64, copy=False)
    else:
        raise ValueError("X must be a numpy.ndarray or pandas.DataFrame.")

    return x_arr, y_arr, feature_names, category_maps, numeric_bin_edges


def _bin_numeric_values(
    values: np.ndarray,
    *,
    strategy: str,
    n_bins: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bin numeric values into integer categories and return codes, edges, and labels."""
    min_value = float(values.min(initial=0.0))
    max_value = float(values.max(initial=0.0))
    if min_value == max_value:
        single_label = f"[{min_value:.6g}, {max_value:.6g}]"
        return (
            np.zeros(values.shape[0], dtype=np.int64),
            np.array([min_value, max_value], dtype=float),
            np.array([single_label], dtype=object),
        )

    if strategy == "quantiles":
        quantiles = np.linspace(0.0, 1.0, num=n_bins + 1)
        edges = np.quantile(values, quantiles)
        edges = np.unique(edges)
    else:
        edges = np.linspace(min_value, max_value, num=n_bins + 1)

    if edges.size < 2:
        edges = np.array([min_value, max_value], dtype=float)

    codes = np.digitize(values, edges[1:-1], right=False).astype(np.int64, copy=False)
    labels = [
        (
            f"[{edges[i]:.6g}, {edges[i + 1]:.6g})"
            if i < edges.size - 2
            else f"[{edges[i]:.6g}, {edges[i + 1]:.6g}]"
        )
        for i in range(edges.size - 1)
    ]
    return codes, edges.astype(float, copy=False), np.asarray(labels, dtype=object)


@dataclass
class _TreeNode:
    is_leaf: bool
    node_id: str
    depth: int
    n_samples: int
    total_sum: float
    path: str
    s_node_p: float
    s_node_n: float
    feature_index: int | None = None
    routing: dict[str, list[int]] | None = None
    children: dict[str, _TreeNode] | None = None


@dataclass
class _SplitDecision:
    gain: float
    feature_index: int
    pos_categories: np.ndarray
    neg_categories: np.ndarray
    neu_categories: np.ndarray
    mode: str


def _iter_tree_nodes(root: _TreeNode) -> Iterator[_TreeNode]:
    """Depth-first iteration over nodes in stable child order."""
    yield root
    if root.children:
        for ch in root.children.values():
            yield from _iter_tree_nodes(ch)


class ImpactSplitter:
    """Ternary impact tree for additive targets over categorical features (NumPy or pandas)."""

    def __init__(
        self,
        delta_pct: float = 0.05,
        min_global_impact_pct: float = 0.01,
        max_depth: int = 5,
        numeric_binning_strategy: str = "quantiles",
        numeric_n_bins: int = 10,
    ) -> None:
        if numeric_binning_strategy not in {"quantiles", "interval"}:
            raise ValueError(
                "numeric_binning_strategy must be one of {'quantiles', 'interval'}."
            )
        if isinstance(numeric_n_bins, bool) or not isinstance(numeric_n_bins, int):
            raise ValueError("numeric_n_bins must be an integer >= 2.")
        if numeric_n_bins < 2:
            raise ValueError("numeric_n_bins must be an integer >= 2.")

        self.delta_pct = delta_pct
        self.min_global_impact_pct = min_global_impact_pct
        self.max_depth = max_depth
        self.numeric_binning_strategy = numeric_binning_strategy
        self.numeric_n_bins = numeric_n_bins
        self._X: np.ndarray | None = None
        self._y: np.ndarray | None = None
        self._tree: _TreeNode | None = None
        self._v_global_p: float = 0.0
        self._v_global_n: float = 0.0
        self._trace_enabled = False
        self._node_counter = 0
        self.fit_trace_: list[dict[str, Any]] = []
        self.feature_names_in_: list[str] | None = None
        self.category_maps_: tuple[np.ndarray, ...] | None = None
        self.numeric_bin_edges_: dict[int, np.ndarray] = {}

    def fit(
        self,
        X: np.ndarray | pd.DataFrame,
        y: np.ndarray | pd.Series,
        *,
        trace: bool = False,
        verbose: bool = False,
    ) -> ImpactSplitter:
        """Fit the impact tree on categorical features and an additive target.

        Args:
            X: 2D ``numpy.ndarray`` of non-negative integer label-encoded categories
                per column, or a ``pandas.DataFrame`` of categorical columns (factorized
                internally).
            y: 1D ``numpy.ndarray`` or ``pandas.Series`` with additive target values.
            trace: Record per-node trace entries in ``fit_trace_``.
            verbose: Alias for ``trace``.
        """
        trace = trace or verbose

        x_arr, y_arr, feature_names, category_maps, numeric_bin_edges = _prepare_X_y(
            X,
            y,
            numeric_binning_strategy=self.numeric_binning_strategy,
            numeric_n_bins=self.numeric_n_bins,
        )

        self._X = x_arr
        self._y = y_arr
        self.feature_names_in_ = feature_names
        self.category_maps_ = category_maps
        self.numeric_bin_edges_ = numeric_bin_edges
        self._v_global_p = float(y_arr[y_arr > 0].sum())
        self._v_global_n = float(np.abs(y_arr[y_arr < 0]).sum())
        self._tree = None
        self.fit_trace_ = []
        self._trace_enabled = trace
        self._node_counter = 0

        self._tree = self._build(x_arr, y_arr, depth=0, path="root")
        return self

    def _feature_path_key(self, feature_index: int) -> str:
        if self.feature_names_in_ is not None:
            return str(self.feature_names_in_[feature_index])
        return f"f{feature_index}"

    def _decode_codes(self, feature_index: int, codes: list[int]) -> list[Any]:
        if self.category_maps_ is None:
            return list(codes)
        m = self.category_maps_[feature_index]
        return [m[int(c)] for c in codes]

    def _feature_display_name(self, feature_index: int) -> str:
        if self.feature_names_in_ is not None:
            return str(self.feature_names_in_[feature_index])
        return f"f{feature_index}"

    def _format_branch_values(self, feature_index: int, codes: list[int], max_show: int) -> str:
        if not codes:
            return "—"
        if self.category_maps_ is None:
            parts = [str(c) for c in codes]
        else:
            m = self.category_maps_[feature_index]
            parts = [str(m[int(c)]) for c in codes]
        if len(parts) <= max_show:
            return ", ".join(parts)
        head = ", ".join(parts[:max_show])
        return f"{head} (+{len(parts) - max_show} more)"

    def _path_segment_for_branch(self, feature_index: int, codes: list[int]) -> str:
        """One path fragment: ``feature=decoded_categories`` (fit-time segment description)."""
        feat_key = self._feature_path_key(feature_index)
        return f"{feat_key}={self._format_branch_values(feature_index, codes, _PATH_SEGMENT_MAX_LABELS)}"

    @staticmethod
    def _last_path_segment_for_plot(path: str) -> str:
        """Final ``feature=…`` fragment for this node's row subset; empty at root only."""
        if not path or path == "root":
            return ""
        parts = [p.strip() for p in path.split(" / ") if p.strip()]
        return parts[-1] if parts else ""

    def _format_plot_node_label(self, n: _TreeNode, *, compact_stats: bool = False) -> str:
        """Label for ``plot_tree``: segment filter on every node, then local split when not a leaf.

        Every node starts with the **last path segment** (``feature=categories`` for this
        slice), or ``all data`` at the root. Non-leaf nodes add ``split on <feature>``.
        All nodes include ``n`` and Σy / Σy⁺ / Σy⁻ (or one compact stats line when
        ``compact_stats`` is True).
        """
        lines: list[str] = []
        seg = self._last_path_segment_for_plot(n.path)
        lines.append(seg if seg else "all data")

        if not n.is_leaf:
            if n.feature_index is None:
                lines.append("split")
            else:
                fname = self._feature_display_name(n.feature_index)
                lines.append(f"split on {fname}")
        if compact_stats:
            lines.append(
                f"n={n.n_samples}  Σy={n.total_sum:.1f}",
            )
        else:
            lines.extend(
                (
                    f"n={n.n_samples}",
                    f"Σy={n.total_sum:.1f}",
                    f"Σy⁺={n.s_node_p:.1f}",
                    f"Σy⁻={n.s_node_n:.1f}",
                ),
            )
        return "\n".join(lines)

    @staticmethod
    def _estimate_plot_label_bbox_units(
        label: str,
        *,
        fontsize: float,
        min_leaf_width: float,
        min_height: float = 0.12,
    ) -> tuple[float, float]:
        """Heuristic (width, height) for a multi-line label in abstract layout units."""
        lines = label.split("\n") if label else [""]
        longest_line = max(lines, key=len) if lines else ""
        longest = len(longest_line)
        unicode_bump = 1.15 if any(ord(c) > 127 for c in longest_line) else 1.0
        n_lines = max(len(lines), 1)
        scale = fontsize / 7.0
        w = longest * 0.12 * scale * unicode_bump + (n_lines - 1) * 0.18 * scale
        h = n_lines * 0.24 * scale
        return float(max(min_leaf_width, w)), float(max(min_height, h))

    @staticmethod
    def _relative_luminance_srgb(rgb: tuple[float, float, float]) -> float:
        """Relative luminance in [0, 1] for sRGB components in [0, 1]."""

        def _lin(c: float) -> float:
            return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

        r, g, b = (float(rgb[0]), float(rgb[1]), float(rgb[2]))
        return 0.2126 * _lin(r) + 0.7152 * _lin(g) + 0.0722 * _lin(b)

    @staticmethod
    def _text_color_for_face_rgb(rgb: tuple[float, float, float]) -> str:
        return "white" if ImpactSplitter._relative_luminance_srgb(rgb) < 0.45 else "black"

    @staticmethod
    def _measure_text_bbox_size_data(
        ax: Any,
        fig: Figure,
        label: str,
        *,
        fontsize: float,
        bbox_style: dict[str, Any],
        renderer: Any | None = None,
    ) -> tuple[float, float]:
        """Width and height of a node label in data coordinates (matches ``ax`` limits)."""
        try:
            if renderer is None:
                fig.canvas.draw()
                renderer = _canvas_renderer(fig)
        except Exception:
            w, h = ImpactSplitter._estimate_plot_label_bbox_units(
                label,
                fontsize=fontsize,
                min_leaf_width=1.0,
            )
            return w, h
        t = ax.text(
            0.0,
            0.0,
            label,
            fontsize=fontsize,
            bbox=bbox_style,
            ha="center",
            va="center",
        )
        try:
            bbox_disp = t.get_window_extent(renderer=renderer)
            bbox_disp = bbox_disp.expanded(1.05, 1.05)
            pts = bbox_disp.get_points()
            x0, y0 = float(pts[0, 0]), float(pts[0, 1])
            x1, y1 = float(pts[1, 0]), float(pts[1, 1])
            corners = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=float)
            data = ax.transData.inverted().transform(corners)
            xs = data[:, 0]
            ys = data[:, 1]
            return float(max(np.ptp(xs), 1e-6)), float(max(np.ptp(ys), 1e-6))
        finally:
            t.remove()

    @staticmethod
    def _measure_text_bbox_width_data(
        ax: Any,
        fig: Figure,
        label: str,
        *,
        fontsize: float,
        bbox_style: dict[str, Any],
        renderer: Any | None = None,
    ) -> float:
        """Horizontal extent of a node label in data coordinates (matches ``ax`` limits)."""
        w, _ = ImpactSplitter._measure_text_bbox_size_data(
            ax,
            fig,
            label,
            fontsize=fontsize,
            bbox_style=bbox_style,
            renderer=renderer,
        )
        return w

    def _next_node_id(self) -> str:
        node_id = f"node_{self._node_counter}"
        self._node_counter += 1
        return node_id

    @staticmethod
    def _all_rows_identical(x_sub: np.ndarray) -> bool:
        if x_sub.shape[0] <= 1:
            return True
        return bool(np.all(x_sub == x_sub[0]))

    def _build(self, x_sub: np.ndarray, y_sub: np.ndarray, depth: int, path: str) -> _TreeNode:
        n_samples = int(y_sub.shape[0])
        total_sum = float(y_sub.sum())
        node_id = self._next_node_id()

        s_node_p = float(y_sub[y_sub > 0].sum())
        s_node_n = float(np.abs(y_sub[y_sub < 0]).sum())
        ratio_p = s_node_p / self._v_global_p if self._v_global_p > 0 else 0.0
        ratio_n = s_node_n / self._v_global_n if self._v_global_n > 0 else 0.0
        positive_trigger = ratio_p > self.min_global_impact_pct
        negative_trigger = ratio_n > self.min_global_impact_pct

        v_node = float(np.abs(y_sub).sum())
        delta_raw = v_node * self.delta_pct
        y_centered = y_sub - float(y_sub.mean()) if n_samples > 0 else y_sub
        v_node_centered = float(np.abs(y_centered).sum())
        delta_centered = v_node_centered * self.delta_pct

        trace_entry: dict[str, Any] = {
            "node_id": node_id,
            "depth": depth,
            "n_samples": n_samples,
            "V_node": v_node,
            "V_node_centered": v_node_centered,
            "delta_pct": self.delta_pct,
            "delta": delta_raw,
            "delta_raw": delta_raw,
            "delta_centered_excess": delta_centered,
            "s_node_p": s_node_p,
            "s_node_n": s_node_n,
            "total_sum": total_sum,
            "path": path,
            "global_ratios": {
                "pos_ratio": ratio_p,
                "neg_ratio": ratio_n,
                "V_global_P": self._v_global_p,
                "V_global_N": self._v_global_n,
            },
            "positive_trigger": positive_trigger,
            "negative_trigger": negative_trigger,
            "candidate_gains": [],
            "candidate_gains_by_mode": {"raw": [], "centered_excess": []},
            "chosen_feature_index": None,
            "routing_mode": None,
            "category_tables": {},
            "category_tables_by_mode": {},
            "action": "split",
            "stop_reason": None,
        }

        if (not positive_trigger) and (not negative_trigger):
            trace_entry["action"] = "leaf"
            trace_entry["stop_reason"] = "materiality"
            if self._trace_enabled:
                self.fit_trace_.append(trace_entry)
            return _TreeNode(True, node_id, depth, n_samples, total_sum, path, s_node_p, s_node_n)

        if depth == self.max_depth:
            trace_entry["action"] = "leaf"
            trace_entry["stop_reason"] = "max_depth"
            if self._trace_enabled:
                self.fit_trace_.append(trace_entry)
            return _TreeNode(True, node_id, depth, n_samples, total_sum, path, s_node_p, s_node_n)

        if self._all_rows_identical(x_sub):
            trace_entry["action"] = "leaf"
            trace_entry["stop_reason"] = "identical_rows"
            if self._trace_enabled:
                self.fit_trace_.append(trace_entry)
            return _TreeNode(True, node_id, depth, n_samples, total_sum, path, s_node_p, s_node_n)

        def evaluate_split_mode(
            *,
            mode: str,
            signal_values: np.ndarray,
            delta_mode: float,
            include_centered_signal: bool,
        ) -> tuple[list[dict[str, Any]], dict[int, list[dict[str, Any]]], _SplitDecision | None]:
            mode_candidates: list[dict[str, Any]] = []
            mode_tables: dict[int, list[dict[str, Any]]] = {}
            best: _SplitDecision | None = None
            best_gain = 0.0

            for feature_index in range(x_sub.shape[1]):
                col_vals = x_sub[:, feature_index]
                if col_vals.size == 0:
                    continue

                max_cat = int(col_vals.max(initial=0))
                cat_signal_sums = np.bincount(
                    col_vals,
                    weights=signal_values,
                    minlength=max_cat + 1,
                )
                cat_raw_sums = np.bincount(col_vals, weights=y_sub, minlength=max_cat + 1)
                cat_counts = np.bincount(col_vals, minlength=max_cat + 1)
                present_categories = np.flatnonzero(cat_counts)
                if present_categories.size <= 1:
                    continue

                present_signal_sums = cat_signal_sums[present_categories]
                present_raw_sums = cat_raw_sums[present_categories]

                pos_mask = present_signal_sums > delta_mode
                neg_mask = present_signal_sums < -delta_mode
                neu_mask = ~(pos_mask | neg_mask)

                s_p = float(present_signal_sums[pos_mask].sum())
                s_n = float(present_signal_sums[neg_mask].sum())
                k_p = int(pos_mask.sum())
                k_n = int(neg_mask.sum())
                gain_p = abs(s_p) / k_p if k_p > 0 else 0.0
                gain_n = abs(s_n) / k_n if k_n > 0 else 0.0
                total_gain = gain_p + gain_n

                pos_categories = present_categories[pos_mask].astype(np.int64, copy=False)
                neg_categories = present_categories[neg_mask].astype(np.int64, copy=False)
                neu_categories = present_categories[neu_mask].astype(np.int64, copy=False)

                row_p = np.isin(col_vals, pos_categories)
                row_n = np.isin(col_vals, neg_categories)
                row_u = ~(row_p | row_n)
                if (
                    int(row_p.sum()) == n_samples
                    or int(row_n.sum()) == n_samples
                    or int(row_u.sum()) == n_samples
                ):
                    continue

                cat_rows: list[dict[str, Any]] = []
                for cat, raw_val, signal_val in zip(
                    present_categories.tolist(),
                    present_raw_sums.tolist(),
                    present_signal_sums.tolist(),
                    strict=True,
                ):
                    row: dict[str, Any] = {
                        "category": int(cat),
                        "S_cat": float(raw_val),
                        "branch": (
                            "P"
                            if signal_val > delta_mode
                            else ("N" if signal_val < -delta_mode else "neutral")
                        ),
                    }
                    if include_centered_signal:
                        row["D_cat"] = float(signal_val)
                    if self.category_maps_ is not None:
                        row["category_label"] = self.category_maps_[feature_index][int(cat)]
                    cat_rows.append(row)
                mode_tables[feature_index] = cat_rows

                mode_candidates.append(
                    {
                        "feature_index": feature_index,
                        "gain": total_gain,
                        "gain_P": gain_p,
                        "gain_N": gain_n,
                        "k_P": k_p,
                        "k_N": k_n,
                        "mode": mode,
                        "delta_mode": delta_mode,
                    }
                )
                if total_gain > best_gain:
                    best_gain = total_gain
                    best = _SplitDecision(
                        gain=total_gain,
                        feature_index=feature_index,
                        pos_categories=pos_categories,
                        neg_categories=neg_categories,
                        neu_categories=neu_categories,
                        mode=mode,
                    )

            mode_candidates.sort(key=lambda item: -item["gain"])
            return mode_candidates, mode_tables, best

        raw_candidates, raw_tables, best_decision = evaluate_split_mode(
            mode="raw",
            signal_values=y_sub,
            delta_mode=delta_raw,
            include_centered_signal=False,
        )
        trace_entry["candidate_gains_by_mode"]["raw"] = raw_candidates
        trace_entry["category_tables_by_mode"]["raw"] = raw_tables
        trace_entry["candidate_gains"].extend(raw_candidates)

        if best_decision is None:
            centered_candidates, centered_tables, best_decision = evaluate_split_mode(
                mode="centered_excess",
                signal_values=y_centered,
                delta_mode=delta_centered,
                include_centered_signal=True,
            )
            trace_entry["candidate_gains_by_mode"]["centered_excess"] = centered_candidates
            trace_entry["category_tables_by_mode"]["centered_excess"] = centered_tables
            trace_entry["candidate_gains"].extend(centered_candidates)
        else:
            trace_entry["candidate_gains_by_mode"]["centered_excess"] = []
            trace_entry["category_tables_by_mode"]["centered_excess"] = {}

        trace_entry["candidate_gains"].sort(key=lambda item: -item["gain"])

        if best_decision is None or best_decision.gain == 0.0:
            trace_entry["action"] = "leaf"
            trace_entry["stop_reason"] = "no_split"
            if self._trace_enabled:
                self.fit_trace_.append(trace_entry)
            return _TreeNode(True, node_id, depth, n_samples, total_sum, path, s_node_p, s_node_n)

        best_feature_index = best_decision.feature_index
        best_pos_categories = best_decision.pos_categories
        best_neg_categories = best_decision.neg_categories
        best_neu_categories = best_decision.neu_categories
        best_mode = best_decision.mode

        best_col_vals = x_sub[:, best_feature_index]
        mask_p = np.isin(best_col_vals, best_pos_categories)
        mask_n = np.isin(best_col_vals, best_neg_categories)
        mask_u = ~mask_p & ~mask_n

        trace_entry["chosen_feature_index"] = best_feature_index
        trace_entry["routing_mode"] = best_mode
        trace_entry["category_tables"] = trace_entry["category_tables_by_mode"].get(best_mode, {})
        trace_entry["routing"] = {
            "positive": best_pos_categories.tolist(),
            "negative": best_neg_categories.tolist(),
            "neutral": best_neu_categories.tolist(),
        }
        if self.feature_names_in_ is not None and best_feature_index is not None:
            trace_entry["chosen_feature_name"] = self.feature_names_in_[best_feature_index]
            trace_entry["routing_labels"] = {
                "positive": self._decode_codes(best_feature_index, best_pos_categories.tolist()),
                "negative": self._decode_codes(best_feature_index, best_neg_categories.tolist()),
                "neutral": self._decode_codes(best_feature_index, best_neu_categories.tolist()),
            }
        if self._trace_enabled:
            self.fit_trace_.append(trace_entry)

        seg_p = self._path_segment_for_branch(best_feature_index, best_pos_categories.tolist())
        seg_n = self._path_segment_for_branch(best_feature_index, best_neg_categories.tolist())
        seg_u = self._path_segment_for_branch(best_feature_index, best_neu_categories.tolist())
        children: dict[str, _TreeNode] = {}
        if np.any(mask_p):
            children["positive"] = self._build(
                x_sub[mask_p],
                y_sub[mask_p],
                depth + 1,
                f"{path} / {seg_p}",
            )
        if np.any(mask_n):
            children["negative"] = self._build(
                x_sub[mask_n],
                y_sub[mask_n],
                depth + 1,
                f"{path} / {seg_n}",
            )
        if np.any(mask_u):
            children["neutral"] = self._build(
                x_sub[mask_u],
                y_sub[mask_u],
                depth + 1,
                f"{path} / {seg_u}",
            )

        return _TreeNode(
            False,
            node_id,
            depth,
            n_samples,
            total_sum,
            path,
            s_node_p,
            s_node_n,
            feature_index=best_feature_index,
            routing={
                "positive": best_pos_categories.tolist(),
                "negative": best_neg_categories.tolist(),
                "neutral": best_neu_categories.tolist(),
            },
            children=children,
        )

    def get_impact_segments(self) -> pd.DataFrame:
        """Return terminal segments sorted by absolute total impact."""
        if self._tree is None:
            raise RuntimeError("Call fit() before get_impact_segments().")

        rows: list[dict[str, Any]] = []

        def walk(n: _TreeNode) -> None:
            if n.is_leaf:
                rows.append(
                    {
                        "path": n.path,
                        "total_sum": n.total_sum,
                        "n_samples": n.n_samples,
                        "node_id": n.node_id,
                    }
                )
                return
            if n.children:
                for ch in n.children.values():
                    walk(ch)

        walk(self._tree)
        df = pd.DataFrame(rows)
        if df.empty:
            return df
        df = df.assign(abs_impact=df["total_sum"].abs()).sort_values(
            "abs_impact",
            ascending=False,
        )
        return df.drop(columns=["abs_impact"]).reset_index(drop=True)

    def plot_tree(
        self,
        figsize: tuple[float, float] = (16.0, 10.0),
        *,
        fontsize: float = 7.0,
        edge_label_fontsize: float | None = None,
        node_bbox: dict[str, Any] | None = None,
        branch_edge_colors: dict[str, str] | None = None,
        show: bool = True,
        level_gap: float = 1.4,
        min_leaf_width: float = 1.2,
        max_leaf_width: float | None = None,
        sibling_gap: float = 0.35,
        layout_max_iterations: int = 24,
        edge_label_pos: float = 0.22,
        edge_label_bbox: bool = True,
        compact_stats: bool = False,
        node_label_max_chars: int | None = 44,
        node_facecolor: str | None = None,
    ) -> Figure:
        """Plot the fitted tree with matplotlib.

        Every node shows the **segment** for its row subset (``all data`` at the root,
        else the last ``feature=categories`` fragment). Non-leaf nodes also show
        ``split on <feature>``. Row count ``n`` and ``Σy`` / ``Σy⁺`` / ``Σy⁻`` appear
        on every node (unless ``compact_stats`` is True). Branch direction is on the
        edges (``P`` / ``Neg`` / ``Neu``). Full cumulative paths remain in
        ``get_impact_segments()`` ``path`` column.

        **Readability:** subtree layout uses **matplotlib-measured** label widths in data
        coordinates (iterated with ``layout_max_iterations``) so spacing matches what is
        drawn. Increase ``figsize`` (width), ``level_gap`` (minimum vertical step),
        ``min_leaf_width``, or ``sibling_gap``; reduce ``fontsize``, set
        ``node_label_max_chars``, use ``compact_stats=True``, or pass ``max_leaf_width``
        to tighten per-line truncation until labels fit that width budget.

        For **publication or static files**, save before displaying:
        ``fig = model.plot_tree(show=False); fig.savefig("tree.pdf")``
        (vector PDF/SVG avoids raster scaling issues).

        Args:
            figsize: Figure size in inches.
            fontsize: Font size for node text.
            edge_label_fontsize: Font size for P/N edge labels; defaults to ``fontsize``.
            node_bbox: Matplotlib ``bbox`` dict for node boxes; default depends on
                ``node_facecolor`` (rounded boxes; ``wheat`` when encoding is off).
            branch_edge_colors: Override per-branch edge colors (keys ``positive``,
                ``negative``, ``neutral``). ``None`` uses built-in color-blind–friendly hues.
            show: If True, call ``plt.show()`` after drawing.
            level_gap: Minimum vertical distance between tree levels (data coordinates);
                may grow with the tallest measured node label.
            min_leaf_width: Minimum width reserved per leaf in layout data coordinates.
            max_leaf_width: If set, per-line truncation is tightened (binary search on
                character budget) so measured label widths stay within this data-space
                budget where possible; layout always uses true measured widths.
            sibling_gap: Extra horizontal space between sibling subtrees (data coordinates).
            layout_max_iterations: Iterations to converge node widths with measured text
                boxes (matplotlib bbox in data coordinates; avoids overlap).
            edge_label_pos: Interpolation factor from parent toward child (0–1) for edge
                labels; ``0.22`` places text near the parent fork.
            edge_label_bbox: If True, draw a light background behind P/Neu/Neg labels.
            compact_stats: If True, show one stats line per node (``n`` and Σy only).
            node_label_max_chars: Optional per-line cap for node label text. Longer
                lines are truncated with ``...`` before layout/painting.
            node_facecolor: If ``"impact"``, shade node boxes by ``|total_sum|`` (light
                sequential tint, contrasting text). If ``"n"``, encode ``n_samples``.
                ``None`` uses uniform facecolor (``wheat`` when ``node_bbox`` is unset).
        """
        if self._tree is None:
            raise RuntimeError("Call fit() before plot_tree().")
        tree = self._tree

        edge_fs = edge_label_fontsize if edge_label_fontsize is not None else fontsize
        edge_colors = (
            _DEFAULT_BRANCH_EDGE_COLORS if branch_edge_colors is None else branch_edge_colors
        )

        default_wheat = {"boxstyle": "round", "facecolor": "wheat", "alpha": 0.8}
        if node_bbox is not None:
            bbox_style: dict[str, Any] = dict(node_bbox)
        else:
            bbox_style = dict(default_wheat)

        fig, ax = plt.subplots(figsize=figsize)
        ax.set_axis_off()
        ax.set_title("Impact split tree")

        positions: dict[str, tuple[float, float]] = {}
        subtree_width: dict[str, float] = {}

        user_cap = (
            node_label_max_chars
            if node_label_max_chars is not None and node_label_max_chars >= 4
            else None
        )

        def max_raw_line_len() -> int:
            m = 0
            for node in _iter_tree_nodes(tree):
                raw_ln = self._format_plot_node_label(node, compact_stats=compact_stats)
                for line in raw_ln.split("\n"):
                    m = max(m, len(line))
            return max(m, 4)

        def make_label_fn(budget_cap_inner: int | None) -> Callable[[_TreeNode], str]:
            def label_fn(n: _TreeNode) -> str:
                raw = self._format_plot_node_label(n, compact_stats=compact_stats)
                caps = [c for c in (user_cap, budget_cap_inner) if c is not None and c >= 4]
                eff = min(caps) if caps else None
                if eff is None:
                    return raw
                lines = raw.split("\n")
                clipped = [(ln if len(ln) <= eff else f"{ln[: eff - 3]}...") for ln in lines]
                return "\n".join(clipped)

            return label_fn

        def run_horizontal_pass(
            label_fn: Callable[[_TreeNode], str],
        ) -> tuple[float, float]:
            """Return ``(tree_total_width, max_own_label_width)``."""

            def own_label_width_heuristic(n: _TreeNode) -> float:
                w, _ = self._estimate_plot_label_bbox_units(
                    label_fn(n),
                    fontsize=fontsize,
                    min_leaf_width=min_leaf_width,
                )
                return w

            def measure(n: _TreeNode, own_width_fn: Callable[[_TreeNode], float]) -> float:
                own = own_width_fn(n)
                if n.is_leaf or not n.children:
                    subtree_width[n.node_id] = own
                    return own
                child_list = list(n.children.values())
                k = len(child_list)
                child_sum = (
                    sum(measure(c, own_width_fn) for c in child_list) + (k - 1) * sibling_gap
                )
                w = max(own, child_sum)
                subtree_width[n.node_id] = w
                return w

            subtree_width.clear()
            measure(tree, own_label_width_heuristic)
            tw = subtree_width[tree.node_id]
            max_own = 0.0
            width_cache: dict[str, float] = {}
            for _ in range(max(1, layout_max_iterations)):
                ax.set_xlim(-tw / 2 - 0.5, tw / 2 + 0.5)
                ax.set_ylim(-self.max_depth * level_gap - 2, 1)
                fig.canvas.draw()
                loop_renderer = _canvas_renderer(fig)
                width_cache.clear()

                def own_label_width_measured(
                    n: _TreeNode,
                    *,
                    _renderer: Any = loop_renderer,
                ) -> float:
                    if n.node_id not in width_cache:
                        width_cache[n.node_id] = self._measure_text_bbox_width_data(
                            ax,
                            fig,
                            label_fn(n),
                            fontsize=fontsize,
                            bbox_style=bbox_style,
                            renderer=_renderer,
                        )
                    w_m = width_cache[n.node_id]
                    return float(max(min_leaf_width, w_m))

                subtree_width.clear()
                measure(tree, own_label_width_measured)
                tw_new = subtree_width[tree.node_id]
                max_own = max(width_cache.values()) if width_cache else max_own
                if abs(tw_new - tw) <= 1e-2 * max(1.0, tw):
                    tw = tw_new
                    break
                tw = tw_new
            return tw, max_own

        budget_cap_final: int | None = None
        if max_leaf_width is not None:
            hi = max_raw_line_len()
            if user_cap is not None:
                hi = min(hi, user_cap)
            lo, hi_b = 4, hi
            best = 4
            while lo <= hi_b:
                mid = (lo + hi_b) // 2
                _, max_own = run_horizontal_pass(make_label_fn(mid))
                if max_own <= max_leaf_width:
                    best = mid
                    lo = mid + 1
                else:
                    hi_b = mid - 1
            budget_cap_final = best

        label_fn_final = make_label_fn(budget_cap_final)
        tw, _ = run_horizontal_pass(label_fn_final)

        ax.set_xlim(-tw / 2 - 0.5, tw / 2 + 0.5)
        ax.set_ylim(-self.max_depth * level_gap - 2, 1)
        fig.canvas.draw()
        renderer = _canvas_renderer(fig)
        max_h = 0.0
        for node in _iter_tree_nodes(tree):
            _, h = self._measure_text_bbox_size_data(
                ax,
                fig,
                label_fn_final(node),
                fontsize=fontsize,
                bbox_style=bbox_style,
                renderer=renderer,
            )
            max_h = max(max_h, h)
        effective_level_gap = max(level_gap, max_h * _VERTICAL_LABEL_MARGIN)

        def place(n: _TreeNode, x: float, y: float, x_span: float, y_gap: float) -> None:
            positions[n.node_id] = (x, y)
            if n.is_leaf or not n.children:
                return
            child_nodes = list(n.children.values())
            k = len(child_nodes)
            total = subtree_width[n.node_id]
            child_total = (
                sum(subtree_width[ch.node_id] for ch in child_nodes) + (k - 1) * sibling_gap
            )
            pad = max(0.0, (total - child_total) / 2.0)
            cur = x - x_span * (total - 1) / 2 + x_span * pad
            for i, ch in enumerate(child_nodes):
                w_ch = subtree_width[ch.node_id]
                cx = cur + (w_ch - 1) / 2
                place(ch, cx, y - y_gap, x_span, y_gap)
                cur += w_ch
                if i < k - 1:
                    cur += sibling_gap

        tw = subtree_width[tree.node_id]
        place(tree, 0.0, 0.0, 1.0 / max(tw, 1.0), effective_level_gap)

        t_pos = float(min(0.95, max(0.05, edge_label_pos)))

        def draw_edges(n: _TreeNode) -> None:
            if n.is_leaf or not n.children:
                return
            x0, y0 = positions[n.node_id]
            for branch_key, ch in n.children.items():
                x1, y1 = positions[ch.node_id]
                ec = edge_colors.get(branch_key, "#888888")
                short = _BRANCH_EDGE_SHORT.get(branch_key, branch_key[:3])
                ax.plot([x0, x1], [y0, y1], color=ec, linewidth=1, zorder=1)
                lx = x0 + t_pos * (x1 - x0)
                ly = y0 + t_pos * (y1 - y0)
                elab_kwargs: dict[str, Any] = {
                    "fontsize": edge_fs,
                    "ha": "center",
                    "va": "center",
                    "color": ec,
                    "zorder": 3,
                }
                if edge_label_bbox:
                    elab_kwargs["bbox"] = {
                        "boxstyle": "round,pad=0.15",
                        "facecolor": "white",
                        "edgecolor": ec,
                        "alpha": 0.92,
                        "linewidth": 0.6,
                    }
                ax.text(lx, ly, short, **elab_kwargs)
                draw_edges(ch)

        draw_edges(tree)

        node_colors: dict[str, tuple[float, float, float]] | None = None
        cmap = None
        norm: Normalize | None = None
        if node_facecolor == "impact":
            values: list[float] = []

            def collect_vals(n: _TreeNode) -> None:
                values.append(abs(n.total_sum))
                if n.children:
                    for ch in n.children.values():
                        collect_vals(ch)

            collect_vals(tree)
            vmax = max(values) if values else 1.0
            vmin = 0.0
            norm_imp = Normalize(vmin=vmin, vmax=max(vmax, 1e-9))
            cmap_imp = plt.get_cmap("YlOrBr")
            assert cmap_imp is not None

            def face_for_node(n: _TreeNode) -> tuple[float, float, float]:
                return cmap_imp(norm_imp(abs(n.total_sum)))[:3]

            node_colors = {n.node_id: face_for_node(n) for n in _iter_tree_nodes(tree)}
            norm, cmap = norm_imp, cmap_imp
        elif node_facecolor == "n":
            values = []

            def collect_n(n: _TreeNode) -> None:
                values.append(float(n.n_samples))
                if n.children:
                    for ch in n.children.values():
                        collect_n(ch)

            collect_n(tree)
            vmax = max(values) if values else 1.0
            vmin = min(values) if values else 0.0
            norm_n = Normalize(vmin=vmin, vmax=max(vmax, vmin + 1.0))
            cmap_n = plt.get_cmap("YlGnBu")
            assert cmap_n is not None

            def face_for_node_n(n: _TreeNode) -> tuple[float, float, float]:
                return cmap_n(norm_n(float(n.n_samples)))[:3]

            node_colors = {n.node_id: face_for_node_n(n) for n in _iter_tree_nodes(tree)}
            norm, cmap = norm_n, cmap_n

        def label_positions(n: _TreeNode) -> None:
            x, y = positions[n.node_id]
            one_bbox = dict(bbox_style)
            txt_kwargs: dict[str, Any] = {
                "ha": "center",
                "va": "center",
                "fontsize": fontsize,
                "bbox": one_bbox,
                "zorder": 5,
            }
            if node_colors is not None:
                fc = node_colors[n.node_id]
                one_bbox["facecolor"] = fc
                one_bbox.setdefault("edgecolor", "0.35")
                one_bbox.setdefault("linewidth", 0.6)
                txt_kwargs["color"] = self._text_color_for_face_rgb(fc)
            ax.text(x, y, label_fn_final(n), **txt_kwargs)
            if n.children:
                for ch in n.children.values():
                    label_positions(ch)

        label_positions(tree)

        use_cbar = node_facecolor in ("impact", "n") and cmap is not None and norm is not None
        if use_cbar:
            fig.subplots_adjust(left=0.06, right=0.86, top=0.90, bottom=0.06)
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cbar_label = "|Σy|" if node_facecolor == "impact" else "n samples"
            fig.colorbar(sm, ax=ax, shrink=0.35, label=cbar_label, pad=0.02)
        else:
            fig.subplots_adjust(left=0.06, right=0.96, top=0.90, bottom=0.06)

        ax.set_xlim(-tw / 2 - 0.5, tw / 2 + 0.5)
        ax.set_ylim(-self.max_depth * effective_level_gap - 2, 1)
        if show:
            plt.show()
        return fig
