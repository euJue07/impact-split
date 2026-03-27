"""Impact-driven ternary tree for additive KPIs over categorical features."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass
class _TreeNode:
    is_leaf: bool
    node_id: str
    depth: int
    n_samples: int
    total_sum: float
    path: str
    feature: str | None = None
    assignment: dict[str, str] | None = None
    children: dict[str, _TreeNode] | None = None


class ImpactSplitter:
    """Ternary impact tree: categories route to positive / negative / neutral by local delta.

    Attributes:
        fit_trace_: Populated after :meth:`fit` when ``trace=True`` or ``verbose=True`` —
            pre-order list of per-node dicts with ``delta_nominal`` (``V_node * delta_pct``),
            assignment ``delta`` / ``delta_neg`` / ``neutral_band``, ``delta_pct``, ``V_node``,
            ``s_node_p``, ``s_node_n``, ``total_sum``, candidate gains, ``chosen_feature``,
            ``stop_reason``, ….
    """

    def __init__(
        self,
        delta_pct: float = 0.05,
        min_global_impact_pct: float = 0.01,
        max_depth: int = 5,
        neutral_root: bool = True,
    ) -> None:
        self.delta_pct = delta_pct
        self.min_global_impact_pct = min_global_impact_pct
        self.max_depth = max_depth
        self.neutral_root = neutral_root
        self._X: pd.DataFrame | None = None
        self._y: np.ndarray | None = None
        self._y_series: pd.Series | None = None
        self._tree: _TreeNode | None = None
        self._v_global_p: float = 0.0
        self._v_global_n: float = 0.0
        self.fit_trace_: list[dict[str, Any]] = []

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        *,
        trace: bool = False,
        verbose: bool = False,
    ) -> ImpactSplitter:
        """Fit the impact tree.

        Set ``trace=True`` or ``verbose=True`` to populate :attr:`fit_trace_`. ``verbose`` is
        an alias for ``trace`` (no extra logging).
        """
        trace = trace or verbose
        if len(X) != len(y):
            raise ValueError("X and y must have the same number of rows.")
        if X.index.tolist() != y.index.tolist():
            X = X.reset_index(drop=True)
            y = y.reset_index(drop=True)
        self._X = X
        self._y_series = y
        y_arr = y.astype(float).to_numpy()
        self._y = y_arr
        pos = y_arr[y_arr > 0].sum()
        neg = (-y_arr[y_arr < 0]).sum()
        self._v_global_p = float(pos)
        self._v_global_n = float(neg)
        self.fit_trace_ = []
        self._trace_enabled = trace
        self._node_counter = 0
        indices = np.arange(len(y_arr), dtype=np.intp)
        self._tree = self._build(indices, depth=0, path="root", used_features=frozenset())
        return self

    def _next_node_id(self) -> str:
        nid = f"node_{self._node_counter}"
        self._node_counter += 1
        return nid

    def _build(
        self,
        indices: np.ndarray,
        depth: int,
        path: str,
        used_features: frozenset[str],
    ) -> _TreeNode:
        y_sub = self._y[indices]
        n_samples = int(len(indices))
        total_sum = float(y_sub.sum())
        v_node = float(np.abs(y_sub).sum())
        delta_nominal = v_node * self.delta_pct
        assignment_delta = (
            0.0 if depth == 0 and self.neutral_root else delta_nominal
        )

        s_node_p = float(y_sub[y_sub > 0].sum())
        s_node_n = float((-y_sub[y_sub < 0]).sum())
        ratio_p = s_node_p / self._v_global_p if self._v_global_p > 0 else 0.0
        ratio_n = s_node_n / self._v_global_n if self._v_global_n > 0 else 0.0
        materiality = (ratio_p >= self.min_global_impact_pct) or (
            ratio_n >= self.min_global_impact_pct
        )

        delta_neg = -assignment_delta
        trace_entry: dict[str, Any] = {
            "node_id": None,
            "depth": depth,
            "n_samples": n_samples,
            "V_node": v_node,
            "delta_pct": self.delta_pct,
            "delta_nominal": delta_nominal,
            "delta": assignment_delta,
            "delta_neg": delta_neg,
            "neutral_band": {"low": delta_neg, "high": assignment_delta},
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
            "materiality_pass": materiality,
            "candidate_gains": [],
            "chosen_feature": None,
            "category_tables": {},
            "action": "split",
            "stop_reason": None,
        }

        node_id = self._next_node_id()
        trace_entry["node_id"] = node_id

        if not materiality:
            trace_entry["action"] = "leaf"
            trace_entry["stop_reason"] = "materiality"
            if self._trace_enabled:
                self.fit_trace_.append(trace_entry)
            return _TreeNode(
                True,
                node_id,
                depth,
                n_samples,
                total_sum,
                path,
            )

        if depth >= self.max_depth:
            trace_entry["action"] = "leaf"
            trace_entry["stop_reason"] = "max_depth"
            if self._trace_enabled:
                self.fit_trace_.append(trace_entry)
            return _TreeNode(
                True,
                node_id,
                depth,
                n_samples,
                total_sum,
                path,
            )

        best_gain = -1.0
        best_feature: str | None = None
        best_assignment: dict[str, str] | None = None
        best_detail: dict[str, Any] = {}

        X_df = self._X
        assert X_df is not None

        for col in X_df.columns:
            if col in used_features:
                continue
            col_vals = X_df.iloc[indices][col].astype(str)
            cats = col_vals.unique()
            if len(cats) < 2:
                continue

            s_by_cat: dict[str, float] = {}
            for c in cats:
                mask = col_vals == c
                idx_local = np.where(mask)[0]
                idx_global = indices[idx_local]
                s_by_cat[str(c)] = float(self._y[idx_global].sum())

            assignment: dict[str, str] = {}
            for c, s_cat in s_by_cat.items():
                if s_cat > assignment_delta:
                    assignment[c] = "P"
                elif s_cat < -assignment_delta:
                    assignment[c] = "N"
                else:
                    assignment[c] = "neutral"

            k_p = sum(1 for v in assignment.values() if v == "P")
            k_n = sum(1 for v in assignment.values() if v == "N")
            if k_p == 0 or k_n == 0:
                gain = 0.0
            else:
                s_p = sum(s_by_cat[c] for c, a in assignment.items() if a == "P")
                s_n = sum(s_by_cat[c] for c, a in assignment.items() if a == "N")
                gain = abs(s_p) / k_p + abs(s_n) / k_n

            cat_rows = [
                {
                    "category": c,
                    "S_cat": s_by_cat[c],
                    "branch": assignment[c],
                }
                for c in sorted(s_by_cat.keys(), key=lambda x: (-abs(s_by_cat[x]), x))
            ]
            trace_entry["category_tables"][col] = cat_rows
            trace_entry["candidate_gains"].append(
                {
                    "feature": col,
                    "gain": gain,
                    "k_P": k_p,
                    "k_N": k_n,
                }
            )

            if gain > best_gain:
                best_gain = gain
                best_feature = col
                best_assignment = assignment
                best_detail = {
                    "k_P": k_p,
                    "k_N": k_n,
                    "gain": gain,
                }

        trace_entry["candidate_gains"].sort(key=lambda x: -x["gain"])

        if best_feature is None or best_gain <= 0 or best_assignment is None:
            trace_entry["action"] = "leaf"
            trace_entry["stop_reason"] = "no_split"
            if self._trace_enabled:
                self.fit_trace_.append(trace_entry)
            return _TreeNode(
                True,
                node_id,
                depth,
                n_samples,
                total_sum,
                path,
            )

        trace_entry["chosen_feature"] = best_feature
        trace_entry["chosen_gain"] = best_detail.get("gain", best_gain)

        col_vals = X_df.iloc[indices][best_feature].astype(str)
        new_used = used_features | {best_feature}
        child_plan: list[tuple[str, np.ndarray, str]] = []
        for branch_label, branch_code in (
            ("positive", "P"),
            ("negative", "N"),
            ("neutral", "neutral"),
        ):
            cat_set = {c for c, a in best_assignment.items() if a == branch_code}
            if not cat_set:
                continue
            mask = col_vals.isin(cat_set)
            child_idx = indices[np.where(mask)[0]]
            if len(child_idx) == 0:
                continue
            child_path = f"{path} / {best_feature}={branch_label}"
            child_plan.append((branch_label, child_idx, child_path))

        if not child_plan:
            trace_entry["action"] = "leaf"
            trace_entry["stop_reason"] = "empty_children"
            if self._trace_enabled:
                self.fit_trace_.append(trace_entry)
            return _TreeNode(
                True,
                node_id,
                depth,
                n_samples,
                total_sum,
                path,
            )

        trace_entry["action"] = "split"
        if self._trace_enabled:
            self.fit_trace_.append(trace_entry)

        children: dict[str, _TreeNode] = {}
        for branch_label, child_idx, child_path in child_plan:
            children[branch_label] = self._build(
                child_idx,
                depth + 1,
                child_path,
                new_used,
            )

        return _TreeNode(
            False,
            node_id,
            depth,
            n_samples,
            total_sum,
            path,
            feature=best_feature,
            assignment=best_assignment,
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

    def plot_tree(self, figsize: tuple[float, float] = (16.0, 10.0)) -> None:
        """Plot the fitted tree with matplotlib."""
        if self._tree is None:
            raise RuntimeError("Call fit() before plot_tree().")

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
            for label, ch in n.children.items():
                x1, y1 = positions[ch.node_id]
                ax.plot([x0, x1], [y0, y1], color="gray", linewidth=1)
                mx, my = (x0 + x1) / 2, (y0 + y1) / 2
                ax.text(mx, my, label[:1].upper(), fontsize=7, ha="center")
                draw_edges(ch)

        draw_edges(self._tree)

        def node_label(n: _TreeNode) -> str:
            if n.is_leaf:
                return f"{n.node_id}\nΣy={n.total_sum:.1f}"
            feat = n.feature or ""
            return f"{n.node_id}\n{feat}"

        def label_positions(n: _TreeNode) -> None:
            x, y = positions[n.node_id]
            ax.text(
                x,
                y,
                node_label(n),
                ha="center",
                va="center",
                fontsize=7,
                bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.8},
            )
            if n.children:
                for ch in n.children.values():
                    label_positions(ch)

        label_positions(self._tree)

        ax.set_xlim(-tw / 2 - 0.5, tw / 2 + 0.5)
        ax.set_ylim(-self.max_depth - 2, 1)
        plt.tight_layout()
        plt.show()
