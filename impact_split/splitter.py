"""Impact-driven ternary tree for additive KPIs over categorical features."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_float_dtype

if TYPE_CHECKING:
    from matplotlib.figure import Figure

# Max decoded category labels per feature in stored segment paths (fit time).
_PATH_SEGMENT_MAX_LABELS = 8

# Hard memory bound for the lookahead rescue's crossed-category bincount:
# feature pairs whose (max_f + 1) * (max_g + 1) allocation exceeds this are skipped.
_LOOKAHEAD_MAX_CROSS = 10_000


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
    split_gain: float | None = None


@dataclass
class _SplitDecision:
    gain: float
    feature_index: int
    pos_categories: np.ndarray
    neg_categories: np.ndarray
    neu_categories: np.ndarray
    mode: str


class ImpactSplitter:
    """Ternary impact tree for additive targets over categorical features (NumPy or pandas)."""

    def __init__(
        self,
        delta_pct: float = 0.01,
        min_global_impact_pct: float = 0.01,
        max_depth: int = 5,
        noise_z: float = 3.0,
        consolidate: bool = True,
        lookahead: bool = True,
        numeric_binning_strategy: str = "quantiles",
        numeric_n_bins: int = 10,
    ) -> None:
        if numeric_binning_strategy not in {"quantiles", "interval"}:
            raise ValueError("numeric_binning_strategy must be one of {'quantiles', 'interval'}.")
        if isinstance(numeric_n_bins, bool) or not isinstance(numeric_n_bins, int):
            raise ValueError("numeric_n_bins must be an integer >= 2.")
        if numeric_n_bins < 2:
            raise ValueError("numeric_n_bins must be an integer >= 2.")
        if noise_z < 0:
            raise ValueError("noise_z must be >= 0 (0 disables the noise floor).")
        if not isinstance(consolidate, bool):
            raise ValueError("consolidate must be a bool.")
        if not isinstance(lookahead, bool):
            raise ValueError("lookahead must be a bool.")

        self.delta_pct = delta_pct
        self.min_global_impact_pct = min_global_impact_pct
        self.max_depth = max_depth
        self.noise_z = noise_z
        self.consolidate = consolidate
        self.lookahead = lookahead
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
        self.segments_: list[dict[str, Any]] = []

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
        self.segments_ = self._consolidate_segments()
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

    def _next_node_id(self) -> str:
        node_id = f"node_{self._node_counter}"
        self._node_counter += 1
        return node_id

    @staticmethod
    def _all_rows_identical(x_sub: np.ndarray) -> bool:
        if x_sub.shape[0] <= 1:
            return True
        return bool(np.all(x_sub == x_sub[0]))

    def _build(
        self,
        x_sub: np.ndarray,
        y_sub: np.ndarray,
        depth: int,
        path: str,
        inter_depth: int = 0,
        parent_feature: int | None = None,
    ) -> _TreeNode:
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

        # max_depth caps the *interaction order* (distinct-feature transitions along
        # the path); consecutive refinements of the same feature are free — they
        # narrow a category pool rather than add an interaction term.
        at_interaction_cap = inter_depth >= self.max_depth
        if at_interaction_cap and parent_feature is None:
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

            # At the interaction cap only same-feature refinements remain legal:
            # they narrow the parent's category pool without adding a new
            # interaction term, so they don't consume depth budget.
            if at_interaction_cap:
                candidate_features: range | list[int] = [cast(int, parent_feature)]
            else:
                candidate_features = range(x_sub.shape[1])

            for feature_index in candidate_features:
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
                present_counts = cat_counts[present_categories]

                # Noise floor: under a within-category-noise null, a category's
                # excess sum wanders ~ sigma * sqrt(n_cat). Categories must clear
                # BOTH the volume-relative delta and z * sigma_f * sqrt(n_cat),
                # where sigma_f is the robust (MAD) scale of this feature's
                # within-category residuals. Stops noise-driven routing in deep
                # nodes where 5% of a small excess volume is below the noise band.
                if self.noise_z > 0:
                    cat_means = np.zeros(max_cat + 1, dtype=float)
                    nz = cat_counts > 0
                    cat_means[nz] = cat_signal_sums[nz] / cat_counts[nz]
                    resid = signal_values - cat_means[col_vals]
                    med = float(np.median(resid))
                    sigma_f = 1.4826 * float(np.median(np.abs(resid - med)))
                    tau = np.maximum(delta_mode, self.noise_z * sigma_f * np.sqrt(present_counts))
                else:
                    tau = np.full(present_categories.shape[0], delta_mode)

                pos_mask = present_signal_sums > tau
                neg_mask = present_signal_sums < -tau
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
                for cat, raw_val, signal_val, tau_val in zip(
                    present_categories.tolist(),
                    present_raw_sums.tolist(),
                    present_signal_sums.tolist(),
                    tau.tolist(),
                    strict=True,
                ):
                    row: dict[str, Any] = {
                        "category": int(cat),
                        "S_cat": float(raw_val),
                        "tau": float(tau_val),
                        "branch": (
                            "P"
                            if signal_val > tau_val
                            else ("N" if signal_val < -tau_val else "neutral")
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

        # Routing signal is always the centered excess D_cat = S_cat - n_cat * mean(node):
        # a category is P/N by how far it deviates from the node's expected share, not by
        # its raw total. On zero-centered targets this equals the raw signal; on one-sided
        # targets (revenue-like, constant base) it stops volume-driven splits where every
        # large category cleared the raw sieve regardless of effect.
        centered_candidates, centered_tables, best_decision = evaluate_split_mode(
            mode="centered_excess",
            signal_values=y_centered,
            delta_mode=delta_centered,
            include_centered_signal=True,
        )
        trace_entry["candidate_gains_by_mode"]["raw"] = []
        trace_entry["category_tables_by_mode"]["raw"] = {}
        trace_entry["candidate_gains_by_mode"]["centered_excess"] = centered_candidates
        trace_entry["category_tables_by_mode"]["centered_excess"] = centered_tables
        trace_entry["candidate_gains"].extend(centered_candidates)

        trace_entry["candidate_gains"].sort(key=lambda item: -item["gain"])

        rescue_info: dict[str, Any] | None = None
        needs_rescue = best_decision is None or best_decision.gain == 0.0
        # Silent-failure signature: a materiality trigger fired (guaranteed —
        # the materiality leaf returned earlier), yet every marginal category
        # table nets ~0. Try the pairwise rescue, unless a rescued split
        # would add an interaction term past the cap.
        if needs_rescue and self.lookahead and not at_interaction_cap and x_sub.shape[1] >= 2:
            rescued = self._lookahead_rescue(x_sub, y_centered, delta_centered)
            if rescued is not None:
                best_decision, rescue_info = rescued

        if best_decision is None or best_decision.gain == 0.0:
            trace_entry["action"] = "leaf"
            trace_entry["stop_reason"] = "max_depth" if at_interaction_cap else "no_split"
            if self._trace_enabled:
                self.fit_trace_.append(trace_entry)
            return _TreeNode(True, node_id, depth, n_samples, total_sum, path, s_node_p, s_node_n)

        if rescue_info is not None:
            trace_entry["rescue"] = rescue_info

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
        child_inter_depth = inter_depth + (1 if best_feature_index != parent_feature else 0)
        children: dict[str, _TreeNode] = {}
        if np.any(mask_p):
            children["positive"] = self._build(
                x_sub[mask_p],
                y_sub[mask_p],
                depth + 1,
                f"{path} / {seg_p}",
                inter_depth=child_inter_depth,
                parent_feature=best_feature_index,
            )
        if np.any(mask_n):
            children["negative"] = self._build(
                x_sub[mask_n],
                y_sub[mask_n],
                depth + 1,
                f"{path} / {seg_n}",
                inter_depth=child_inter_depth,
                parent_feature=best_feature_index,
            )
        if np.any(mask_u):
            children["neutral"] = self._build(
                x_sub[mask_u],
                y_sub[mask_u],
                depth + 1,
                f"{path} / {seg_u}",
                inter_depth=child_inter_depth,
                parent_feature=best_feature_index,
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
            split_gain=float(best_decision.gain),
        )

    def _lookahead_rescue(
        self,
        x_sub: np.ndarray,
        y_centered: np.ndarray,
        delta_centered: float,
    ) -> tuple[_SplitDecision, dict[str, Any]] | None:
        """Cross-feature sieve for XOR-style cancellation.

        Runs only at would-be no_split material nodes. For each feature pair,
        sums ``y_centered`` over crossed categories and applies the unchanged
        two-bar sieve (volume delta + MAD noise floor) plus the unchanged
        category-averaged gain to the cross-cells. The winning pair is realized
        as an ordinary single-feature split via ``_lookahead_partition`` so all
        downstream machinery (conditions, consolidation, conservation) is
        untouched.
        """
        n_samples = int(y_centered.shape[0])
        n_features = int(x_sub.shape[1])
        best: _SplitDecision | None = None
        best_gain = 0.0
        best_info: dict[str, Any] | None = None
        pairs_evaluated = 0
        pairs_skipped = 0

        for f in range(n_features):
            codes_f = x_sub[:, f]
            max_f = int(codes_f.max(initial=0))
            for g in range(f + 1, n_features):
                codes_g = x_sub[:, g]
                max_g = int(codes_g.max(initial=0))
                size = (max_f + 1) * (max_g + 1)
                if size > _LOOKAHEAD_MAX_CROSS:
                    pairs_skipped += 1
                    continue
                h = codes_f * (max_g + 1) + codes_g
                cross_signal = np.bincount(h, weights=y_centered, minlength=size)
                cross_counts = np.bincount(h, minlength=size)
                present = np.flatnonzero(cross_counts)
                if present.size <= 1:
                    continue
                pairs_evaluated += 1

                present_signal = cross_signal[present]
                present_counts = cross_counts[present]
                if self.noise_z > 0:
                    cross_means = np.zeros(size, dtype=float)
                    nz = cross_counts > 0
                    cross_means[nz] = cross_signal[nz] / cross_counts[nz]
                    resid = y_centered - cross_means[h]
                    med = float(np.median(resid))
                    sigma = 1.4826 * float(np.median(np.abs(resid - med)))
                    # Multiplicity correction: the rescue searches K = present.size
                    # crossed cells simultaneously (unlike the marginal sieve's F
                    # single-feature categories), and K can run into the hundreds
                    # or thousands on real high-cardinality features (e.g. Kaggle
                    # Genre x Publisher, task-8-report.md). Under a null with K
                    # cells, the max |cell excess| wanders like sigma*sqrt(n) *
                    # sqrt(2*ln(K)) (extreme-value bound), so the per-cell z must
                    # carry that sqrt(2*ln(K)) term on top of the configured
                    # noise_z buffer. K=4 XOR crosses barely move (sqrt(2*ln4)
                    # ~1.7); K~500-cell crosses get an honest, much higher bar.
                    z_eff = self.noise_z + math.sqrt(2.0 * math.log(present.size))
                    tau = np.maximum(delta_centered, z_eff * sigma * np.sqrt(present_counts))
                else:
                    tau = np.full(present.shape[0], delta_centered)

                # Singleton cells carry no verifiable signal (one observation,
                # no within-cell noise estimate); requiring n_cell >= 2 stops
                # near-singleton cherry-picking that the multiplicity
                # correction can't reach at small K.
                pos_mask = (present_signal > tau) & (present_counts >= 2)
                neg_mask = (present_signal < -tau) & (present_counts >= 2)
                k_p = int(pos_mask.sum())
                k_n = int(neg_mask.sum())
                s_p = float(present_signal[pos_mask].sum())
                s_n = float(present_signal[neg_mask].sum())
                gain = (abs(s_p) / k_p if k_p else 0.0) + (abs(s_n) / k_n if k_n else 0.0)
                if gain == 0.0 or gain <= best_gain:
                    continue

                sig_cells = present[pos_mask | neg_mask]
                profile = cross_signal.reshape(max_f + 1, max_g + 1)
                decision = self._lookahead_partition(
                    f, codes_f, profile, sig_cells // (max_g + 1), gain, n_samples
                )
                if decision is None:
                    decision = self._lookahead_partition(
                        g, codes_g, profile.T, sig_cells % (max_g + 1), gain, n_samples
                    )
                if decision is None:
                    continue
                best_gain = gain
                best = decision
                best_info = {
                    "pair": [f, g],
                    "pair_names": [
                        self._feature_display_name(f),
                        self._feature_display_name(g),
                    ],
                    "split_feature_index": decision.feature_index,
                    "gain": gain,
                    "k_P": k_p,
                    "k_N": k_n,
                    "partition": {
                        "positive": decision.pos_categories.tolist(),
                        "negative": decision.neg_categories.tolist(),
                        "neutral": decision.neu_categories.tolist(),
                    },
                }

        if best is None or best_info is None:
            return None
        best_info["pairs_evaluated"] = pairs_evaluated
        best_info["pairs_skipped_cardinality"] = pairs_skipped
        return best, best_info

    def _lookahead_partition(
        self,
        feature_index: int,
        col_vals: np.ndarray,
        profile: np.ndarray,
        sig_rows: np.ndarray,
        gain: float,
        n_samples: int,
    ) -> _SplitDecision | None:
        """Convert a winning cross-table into a single-feature split on ``feature_index``.

        Categories whose profile row holds no sieve-clearing cross-cell carry no
        signal (the spec's "near-zero row norm") and route neutral; the rest are
        partitioned by the sign of their profile row's dot product with the
        max-norm anchor row. Returns None when the induced split is degenerate
        (either signed group empty, or one branch holds every row).
        """
        present_rows = np.flatnonzero(np.bincount(col_vals, minlength=profile.shape[0]))
        carries = np.isin(present_rows, sig_rows)
        active = present_rows[carries]
        if active.size < 2:
            return None
        norms = np.linalg.norm(profile[active], axis=1)
        anchor_row = profile[active[int(np.argmax(norms))]]
        dots = profile[active] @ anchor_row
        pos = active[dots > 0]
        neg = active[dots < 0]
        neu = np.concatenate([present_rows[~carries], active[dots == 0]])
        if pos.size == 0 or neg.size == 0:
            return None
        row_p = np.isin(col_vals, pos)
        row_n = np.isin(col_vals, neg)
        row_u = ~(row_p | row_n)
        if (
            int(row_p.sum()) == n_samples
            or int(row_n.sum()) == n_samples
            or int(row_u.sum()) == n_samples
        ):
            return None
        return _SplitDecision(
            gain=gain,
            feature_index=int(feature_index),
            pos_categories=np.sort(pos).astype(np.int64, copy=False),
            neg_categories=np.sort(neg).astype(np.int64, copy=False),
            neu_categories=np.sort(neu).astype(np.int64, copy=False),
            mode="lookahead_rescue",
        )

    def _render_conditions_path(self, conditions: dict[int, frozenset[int]]) -> str:
        """Conjunction path (``f=a, b & g=c``) for a consolidated segment."""
        parts = [
            self._path_segment_for_branch(f, sorted(conditions[f])) for f in sorted(conditions)
        ]
        return " & ".join(parts) if parts else "all data"

    def _leaf_segments(self) -> list[dict[str, Any]]:
        """Terminal leaves as segment dicts with exact accumulated conditions + row masks.

        A leaf's row set equals the conjunction of the branch category sets
        accumulated along its path (each routing decision is category
        membership), so ``conditions`` reconstructs the mask exactly.
        """
        assert self._tree is not None and self._X is not None
        X = self._X
        n = X.shape[0]
        out: list[dict[str, Any]] = []

        def rec(node: _TreeNode, cond: dict[int, frozenset[int]], mask: np.ndarray) -> None:
            if node.is_leaf or not node.children:
                out.append(
                    {
                        "path": node.path,
                        "conditions": dict(cond),
                        "node_ids": [node.node_id],
                        "mask": mask,
                        "n_samples": node.n_samples,
                        "total_sum": node.total_sum,
                    }
                )
                return
            f = cast(int, node.feature_index)
            routing = cast(dict[str, list[int]], node.routing)
            col = X[:, f]
            for key, ch in node.children.items():
                codes = frozenset(int(c) for c in routing[key])
                prev = cond.get(f)
                new_cond = dict(cond)
                new_cond[f] = codes if prev is None else (prev & codes)
                child_mask = mask & np.isin(col, list(codes))
                rec(ch, new_cond, child_mask)

        rec(self._tree, {}, np.ones(n, dtype=bool))
        return out

    def _finalize_segments(self, segs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Attach gross flows + churn flag from row masks, then drop fit-only fields.

        A segment is churn when its positive AND negative gross flows each clear
        ``min_global_impact_pct`` against their global pools — the net then hides
        offsetting material mass.
        """
        assert self._y is not None
        y = self._y
        for s in segs:
            ym = y[s["mask"]]
            pos = float(ym[ym > 0].sum())
            neg = float(np.abs(ym[ym < 0]).sum())
            s["pos_sum"] = pos
            s["neg_sum"] = neg
            ratio_p = pos / self._v_global_p if self._v_global_p > 0 else 0.0
            ratio_n = neg / self._v_global_n if self._v_global_n > 0 else 0.0
            s["is_churn"] = bool(
                ratio_p > self.min_global_impact_pct and ratio_n > self.min_global_impact_pct
            )
            s.pop("mask", None)
            s.pop("mean", None)
        return segs

    def _consolidate_segments(self) -> list[dict[str, Any]]:
        """Merge terminal segments that fragmentation split without statistical cause.

        Two segments merge when (a) their conditions are identical except on one
        feature — so the union stays a single readable conjunction — and (b) their
        per-row means are compatible under a two-sample z-test at ``noise_z``
        against the pooled robust within-segment residual scale. Iterated to
        fixpoint (a merged segment may merge again along another feature, which
        collapses cross-product fragmentation). Exact sum conservation holds by
        construction: merges are unions of disjoint row sets.
        """
        assert self._y is not None and self._X is not None
        segs = self._leaf_segments()
        if not self.consolidate or len(segs) <= 1:
            return self._finalize_segments(segs)

        X = self._X
        y = self._y
        # Pooled robust residual scale (within-segment).
        resid = np.empty_like(y)
        for s in segs:
            m = s["mask"]
            resid[m] = y[m] - float(y[m].mean())
        med = float(np.median(resid))
        sigma = 1.4826 * float(np.median(np.abs(resid - med)))

        # Universe per feature: categories present anywhere in the training data.
        universes: dict[int, frozenset[int]] = {}

        def drop_vacuous(cond: dict[int, frozenset[int]]) -> dict[int, frozenset[int]]:
            out = {}
            for f, codes in cond.items():
                if f not in universes:
                    universes[f] = frozenset(int(c) for c in np.unique(X[:, f]))
                if codes < universes[f]:
                    out[f] = codes
            return out

        for s in segs:
            s["conditions"] = drop_vacuous(s["conditions"])
            s["mean"] = s["total_sum"] / s["n_samples"] if s["n_samples"] else 0.0

        def find_merge() -> tuple[int, int, int] | None:
            """Best (i, j, feature) pair passing the compatibility test, or None."""
            best: tuple[int, int, int] | None = None
            best_diff = np.inf
            features = sorted({f for s in segs for f in s["conditions"]})
            for f in features:
                groups: dict[tuple[tuple[int, tuple[int, ...]], ...], list[int]] = {}
                for i, s in enumerate(segs):
                    if f not in s["conditions"]:
                        continue
                    sig = tuple(
                        sorted((g, tuple(sorted(c))) for g, c in s["conditions"].items() if g != f)
                    )
                    groups.setdefault(sig, []).append(i)
                for members in groups.values():
                    for a in range(len(members)):
                        for b in range(a + 1, len(members)):
                            i, j = members[a], members[b]
                            si, sj = segs[i], segs[j]
                            diff = abs(si["mean"] - sj["mean"])
                            thr = (
                                self.noise_z
                                * sigma
                                * float(np.sqrt(1.0 / si["n_samples"] + 1.0 / sj["n_samples"]))
                            )
                            if diff <= thr and diff < best_diff:
                                best = (i, j, f)
                                best_diff = diff
            return best

        while True:
            hit = find_merge()
            if hit is None:
                break
            i, j, f = hit
            si, sj = segs[i], segs[j]
            cond = dict(si["conditions"])
            cond[f] = si["conditions"][f] | sj["conditions"][f]
            merged: dict[str, Any] = {
                "conditions": drop_vacuous(cond),
                "node_ids": si["node_ids"] + sj["node_ids"],
                "mask": si["mask"] | sj["mask"],
                "n_samples": si["n_samples"] + sj["n_samples"],
                "total_sum": si["total_sum"] + sj["total_sum"],
            }
            merged["mean"] = merged["total_sum"] / merged["n_samples"]
            merged["path"] = self._render_conditions_path(merged["conditions"])
            segs = [s for k, s in enumerate(segs) if k not in (i, j)]
            segs.append(merged)

        return self._finalize_segments(segs)

    def get_impact_segments(self) -> pd.DataFrame:
        """Return terminal segments sorted by absolute total impact.

        With ``consolidate=True`` (default) these are the post-fit consolidated
        segments; single-leaf segments keep their tree path, merged segments get
        a conjunction path rendered from their combined conditions.
        """
        if self._tree is None:
            raise RuntimeError("Call fit() before get_impact_segments().")

        rows = []
        for s in self.segments_:
            total = float(s["total_sum"])
            n = int(s["n_samples"])
            pool = self._v_global_p if total >= 0 else self._v_global_n
            rows.append(
                {
                    "path": s["path"],
                    "total_sum": s["total_sum"],
                    "n_samples": s["n_samples"],
                    "node_id": (
                        s["node_ids"][0]
                        if len(s["node_ids"]) == 1
                        else "merged(" + "+".join(s["node_ids"]) + ")"
                    ),
                    "mean": total / n if n else float("nan"),
                    "pool_share": abs(total) / pool if pool > 0 else float("nan"),
                }
            )
        df = pd.DataFrame(rows)
        if df.empty:
            return df
        df = df.assign(abs_impact=df["total_sum"].abs()).sort_values(
            "abs_impact",
            ascending=False,
        )
        return df.drop(columns=["abs_impact"]).reset_index(drop=True)

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe payload (``meta`` / ``tree`` / ``segments``) feeding every renderer.

        Stable shape for third-party renderers; see ``impact_split.viz.data``.
        """
        from impact_split.viz.data import build_payload

        return build_payload(self)

    def summary(self, *, top: int = 10) -> str:
        """Designed text report: ledger header + top segments table (returns, never prints)."""
        from impact_split.viz.text import render_summary

        return render_summary(self.to_dict(), top=top)

    def __repr__(self) -> str:
        if self._tree is None:
            return (
                f"ImpactSplitter(delta_pct={self.delta_pct}, "
                f"min_global_impact_pct={self.min_global_impact_pct}, "
                f"max_depth={self.max_depth}, noise_z={self.noise_z}, "
                f"consolidate={self.consolidate}, lookahead={self.lookahead})"
            )
        return self.summary()

    def plot_segments(
        self,
        *,
        top: int = 15,
        figsize: tuple[float, float] | None = None,
        show: bool = True,
    ) -> Figure:
        """Tornado chart of consolidated segments (the stakeholder deliverable view)."""
        from impact_split.viz.static import plot_segments

        return plot_segments(self.to_dict(), top=top, figsize=figsize, show=show)

    def plot_tree(
        self,
        figsize: tuple[float, float] | None = None,
        *,
        show: bool = True,
    ) -> Figure:
        """Impact icicle: cell width ∝ Σ|y| within its parent, color = impact direction.

        Save with ``fig = model.plot_tree(show=False); fig.savefig("tree.svg")``.
        """
        from impact_split.viz.static import plot_icicle

        return plot_icicle(self.to_dict(), figsize=figsize, show=show)

    def to_html(
        self,
        path: str | Path | None = None,
        *,
        title: str = "impact-split report",
    ) -> str | Path:
        """Self-contained interactive HTML report (no CDN; safe to open offline or email).

        Returns the HTML string when ``path`` is None (usable with
        ``IPython.display.HTML``); otherwise writes UTF-8 and returns the ``Path``.
        """
        from impact_split.viz.html import render_html

        html_text = render_html(self.to_dict(), title=title)
        if path is None:
            return html_text
        out = Path(path)
        out.write_text(html_text, encoding="utf-8")
        return out
