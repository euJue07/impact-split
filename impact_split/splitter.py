"""Impact-driven ternary tree for additive KPIs over categorical features."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Max decoded category labels per feature in stored segment paths (fit time).
_PATH_SEGMENT_MAX_LABELS = 8

# Short edge annotations (distinct from first-letter truncation: neutral was also "N").
_BRANCH_EDGE_SHORT: dict[str, str] = {
    "positive": "P",
    "negative": "Neg",
    "neutral": "Neu",
}

# Color-blind–friendly edge colors; redundant with text labels for accessibility.
_DEFAULT_BRANCH_EDGE_COLORS: dict[str, str] = {
    "positive": "#0173B2",
    "negative": "#DE8F05",
    "neutral": "#949494",
}


def _prepare_X_y(
    X: np.ndarray | pd.DataFrame,
    y: np.ndarray | pd.Series,
) -> tuple[np.ndarray, np.ndarray, list[str] | None, tuple[np.ndarray, ...] | None]:
    """Validate inputs and return integer matrix, float target, and optional pandas metadata."""
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

    if isinstance(X, pd.DataFrame):
        if X.ndim != 2:
            raise ValueError("X must be a 2D pandas.DataFrame.")
        if X.shape[0] != y_arr.shape[0]:
            raise ValueError("X and y must have the same number of rows.")
        cols: list[np.ndarray] = []
        maps: list[np.ndarray] = []
        feature_names = [str(c) for c in X.columns]
        for name in X.columns:
            s = X[name]
            if s.isna().any():
                raise ValueError("X must not contain missing values.")
            codes, uniques = pd.factorize(s, sort=True)
            if np.any(codes < 0):
                raise ValueError("X categories must be non-negative integer codes after factorization.")
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

    return x_arr, y_arr, feature_names, category_maps


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


class ImpactSplitter:
    """Ternary impact tree for additive targets over categorical features (NumPy or pandas)."""

    def __init__(
        self,
        delta_pct: float = 0.05,
        min_global_impact_pct: float = 0.01,
        max_depth: int = 5,
    ) -> None:
        self.delta_pct = delta_pct
        self.min_global_impact_pct = min_global_impact_pct
        self.max_depth = max_depth
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

        x_arr, y_arr, feature_names, category_maps = _prepare_X_y(X, y)

        self._X = x_arr
        self._y = y_arr
        self.feature_names_in_ = feature_names
        self.category_maps_ = category_maps
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

    def _format_plot_node_label(self, n: _TreeNode) -> str:
        """Label for ``plot_tree``: segment filter on every node, then local split when not a leaf.

        Every node starts with the **last path segment** (``feature=categories`` for this
        slice), or ``all data`` at the root. Non-leaf nodes add ``split on <feature>``.
        All nodes include ``n`` and Σy / Σy⁺ / Σy⁻.
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
        lines.extend(
            (
                f"n={n.n_samples}",
                f"Σy={n.total_sum:.1f}",
                f"Σy⁺={n.s_node_p:.1f}",
                f"Σy⁻={n.s_node_n:.1f}",
            ),
        )
        return "\n".join(lines)

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
        delta = v_node * self.delta_pct

        trace_entry: dict[str, Any] = {
            "node_id": node_id,
            "depth": depth,
            "n_samples": n_samples,
            "V_node": v_node,
            "delta_pct": self.delta_pct,
            "delta": delta,
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
            "chosen_feature_index": None,
            "category_tables": {},
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

        best_gain = 0.0
        best_feature_index: int | None = None
        best_pos_categories = np.array([], dtype=np.int64)
        best_neg_categories = np.array([], dtype=np.int64)
        best_neu_categories = np.array([], dtype=np.int64)

        for feature_index in range(x_sub.shape[1]):
            col_vals = x_sub[:, feature_index]
            if col_vals.size == 0:
                continue

            max_cat = int(col_vals.max(initial=0))
            cat_sums = np.bincount(col_vals, weights=y_sub, minlength=max_cat + 1)
            cat_counts = np.bincount(col_vals, minlength=max_cat + 1)
            present_categories = np.flatnonzero(cat_counts)
            if present_categories.size == 0:
                continue
            if present_categories.size <= 1:
                continue

            present_sums = cat_sums[present_categories]
            pos_mask = present_sums > delta
            neg_mask = present_sums < -delta
            neu_mask = ~(pos_mask | neg_mask)

            s_p = float(present_sums[pos_mask].sum())
            s_n = float(present_sums[neg_mask].sum())
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
            for cat, sum_val in zip(present_categories.tolist(), present_sums.tolist()):
                row: dict[str, Any] = {
                    "category": int(cat),
                    "S_cat": float(sum_val),
                    "branch": (
                        "P"
                        if sum_val > delta
                        else ("N" if sum_val < -delta else "neutral")
                    ),
                }
                if self.category_maps_ is not None:
                    row["category_label"] = self.category_maps_[feature_index][int(cat)]
                cat_rows.append(row)
            trace_entry["category_tables"][feature_index] = cat_rows
            trace_entry["candidate_gains"].append(
                {
                    "feature_index": feature_index,
                    "gain": total_gain,
                    "gain_P": gain_p,
                    "gain_N": gain_n,
                    "k_P": k_p,
                    "k_N": k_n,
                }
            )

            if total_gain > best_gain:
                best_gain = total_gain
                best_feature_index = feature_index
                best_pos_categories = pos_categories
                best_neg_categories = neg_categories
                best_neu_categories = neu_categories

        trace_entry["candidate_gains"].sort(key=lambda item: -item["gain"])

        if best_gain == 0.0 or best_feature_index is None:
            trace_entry["action"] = "leaf"
            trace_entry["stop_reason"] = "no_split"
            if self._trace_enabled:
                self.fit_trace_.append(trace_entry)
            return _TreeNode(True, node_id, depth, n_samples, total_sum, path, s_node_p, s_node_n)

        best_col_vals = x_sub[:, best_feature_index]
        mask_p = np.isin(best_col_vals, best_pos_categories)
        mask_n = np.isin(best_col_vals, best_neg_categories)
        mask_u = ~mask_p & ~mask_n

        trace_entry["chosen_feature_index"] = best_feature_index
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
    ) -> Figure:
        """Plot the fitted tree with matplotlib.

        Every node shows the **segment** for its row subset (``all data`` at the root,
        else the last ``feature=categories`` fragment). Non-leaf nodes also show
        ``split on <feature>``. Row count ``n`` and ``Σy`` / ``Σy⁺`` / ``Σy⁻`` appear
        on every node. Branch direction is on the edges (``P`` / ``Neg`` / ``Neu``).
        Full cumulative paths remain in ``get_impact_segments()`` ``path`` column.

        For **deep trees**, increase ``figsize`` (wider for many leaves, taller for
        depth) or reduce ``fontsize``. For **publication or static files**, save
        before displaying: ``fig = model.plot_tree(show=False); fig.savefig("tree.pdf")``
        (vector PDF/SVG avoids raster scaling issues).

        Args:
            figsize: Figure size in inches.
            fontsize: Font size for node text.
            edge_label_fontsize: Font size for P/N edge labels; defaults to ``fontsize``.
            node_bbox: Matplotlib ``bbox`` dict for node boxes; default is rounded wheat.
            branch_edge_colors: Override per-branch edge colors (keys ``positive``,
                ``negative``, ``neutral``). ``None`` uses built-in color-blind–friendly hues.
            show: If True, call ``plt.show()`` after drawing.
        """
        if self._tree is None:
            raise RuntimeError("Call fit() before plot_tree().")

        edge_fs = edge_label_fontsize if edge_label_fontsize is not None else fontsize
        bbox_style = (
            node_bbox
            if node_bbox is not None
            else {"boxstyle": "round", "facecolor": "wheat", "alpha": 0.8}
        )
        edge_colors = _DEFAULT_BRANCH_EDGE_COLORS if branch_edge_colors is None else branch_edge_colors

        fig, ax = plt.subplots(figsize=figsize)
        ax.set_axis_off()
        ax.set_title("Impact split tree")

        positions: dict[str, tuple[float, float]] = {}
        subtree_width: dict[str, float] = {}

        def measure(n: _TreeNode) -> float:
            if n.is_leaf or not n.children:
                w = 1.0
                subtree_width[n.node_id] = w
                return w
            s = sum(measure(c) for c in n.children.values())
            subtree_width[n.node_id] = s
            return s

        measure(self._tree)

        def place(n: _TreeNode, x: float, y: float, x_span: float) -> None:
            positions[n.node_id] = (x, y)
            if n.is_leaf or not n.children:
                return
            child_nodes = list(n.children.values())
            total = subtree_width[n.node_id]
            cur = x - x_span * (total - 1) / 2
            for ch in child_nodes:
                w = subtree_width[ch.node_id]
                cx = cur + (w - 1) / 2
                place(ch, cx, y - 1.2, x_span)
                cur += w

        tw = subtree_width[self._tree.node_id]
        place(self._tree, 0.0, 0.0, 1.0 / max(tw, 1.0))

        def draw_edges(n: _TreeNode) -> None:
            if n.is_leaf or not n.children:
                return
            x0, y0 = positions[n.node_id]
            for branch_key, ch in n.children.items():
                x1, y1 = positions[ch.node_id]
                ec = edge_colors.get(branch_key, "#888888")
                short = _BRANCH_EDGE_SHORT.get(branch_key, branch_key[:3])
                ax.plot([x0, x1], [y0, y1], color=ec, linewidth=1)
                mx, my = (x0 + x1) / 2, (y0 + y1) / 2
                ax.text(mx, my, short, fontsize=edge_fs, ha="center", color="0.2")
                draw_edges(ch)

        draw_edges(self._tree)

        def node_label(n: _TreeNode) -> str:
            return self._format_plot_node_label(n)

        def label_positions(n: _TreeNode) -> None:
            x, y = positions[n.node_id]
            ax.text(
                x,
                y,
                node_label(n),
                ha="center",
                va="center",
                fontsize=fontsize,
                bbox=bbox_style,
            )
            if n.children:
                for ch in n.children.values():
                    label_positions(ch)

        label_positions(self._tree)

        ax.set_xlim(-tw / 2 - 0.5, tw / 2 + 0.5)
        ax.set_ylim(-self.max_depth - 2, 1)
        plt.tight_layout()
        if show:
            plt.show()
        return fig
