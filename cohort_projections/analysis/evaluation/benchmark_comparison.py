"""Benchmark and Comparison Framework (Module 4).

Provides structured comparison of projection methods against benchmark
alternatives, component-swapping analysis, horizon-blended projections,
and ensemble construction.

References:
    - docs/plans/evaluation-blueprint.md, Module 4
    - config/evaluation_config.yaml
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from .schemas import (
    METRIC_REGISTRY,
    PROJECTION_JOIN_KEYS,
    PROJECTION_RESULT_COLUMNS,
    HorizonBands,
)
from .utils import validate_dataframe

logger = logging.getLogger(__name__)

# Standard benchmark families referenced in the blueprint
BENCHMARK_FAMILIES = [
    "baseline_cohort_component",
    "carry_forward",
    "moving_average_rates",
    "mean_reverting_cc",
    "trend_extrapolation",
    "state_share",
]

# Ordered list form of PROJECTION_RESULT_COLUMNS for column selection
_RESULT_COLUMNS = sorted(PROJECTION_RESULT_COLUMNS)

# Default metrics: extract just the callable from METRIC_REGISTRY entries
_DEFAULT_METRICS: dict[str, Any] = {
    name: fn for name, (fn, _group) in METRIC_REGISTRY.items()
}


class BenchmarkComparisonModule:
    """Compare candidate projection methods against benchmarks.

    Parameters
    ----------
    config : dict
        Evaluation configuration dictionary (loaded from
        ``config/evaluation_config.yaml``).

    Example
    -------
    >>> module = BenchmarkComparisonModule(config)
    >>> comparison = module.compare_all(method_results, baseline_name="m2024")
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.near_term_max: int = config.get("near_term_max_horizon", 5)
        self.long_term_min: int = config.get("long_term_min_horizon", 10)
        self.horizons: list[int] = config.get("horizons", [1, 2, 3, 5, 10, 15, 20])
        self.county_groups: dict[str, list[str]] = config.get("county_groups", {})
        self._hbands = HorizonBands(
            near_max=self.near_term_max, long_min=self.long_term_min
        )

    # ------------------------------------------------------------------
    # Full comparison
    # ------------------------------------------------------------------

    def compare_all(
        self,
        method_results: dict[str, pd.DataFrame],
        baseline_name: str,
    ) -> pd.DataFrame:
        """Compare every method against *baseline_name*.

        Parameters
        ----------
        method_results : dict[str, pd.DataFrame]
            Mapping from method name to a DataFrame with
            ``ProjectionResultRecord`` columns.
        baseline_name : str
            Key in *method_results* used as the reference baseline.

        Returns
        -------
        pd.DataFrame
            One row per (method, metric, geography_group, horizon_band)
            with columns ``method``, ``baseline``, ``metric_name``,
            ``method_value``, ``baseline_value``, ``delta``,
            ``geography_group``, ``horizon_band``.
        """
        if baseline_name not in method_results:
            raise KeyError(
                f"Baseline '{baseline_name}' not found in method_results. "
                f"Available: {sorted(method_results.keys())}"
            )
        baseline_df = method_results[baseline_name]
        validate_dataframe(baseline_df, PROJECTION_RESULT_COLUMNS, f"baseline ({baseline_name})")

        rows: list[dict[str, Any]] = []
        for name, challenger_df in method_results.items():
            if name == baseline_name:
                continue
            pair = self.pairwise_comparison(
                challenger_df, baseline_df, name, baseline_name
            )
            rows.extend(pair.to_dict("records"))

        if not rows:
            return pd.DataFrame(
                columns=[
                    "method",
                    "baseline",
                    "metric_name",
                    "method_value",
                    "baseline_value",
                    "delta",
                    "geography_group",
                    "horizon_band",
                ]
            )
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Pairwise comparison
    # ------------------------------------------------------------------

    def pairwise_comparison(
        self,
        challenger_df: pd.DataFrame,
        baseline_df: pd.DataFrame,
        method_name: str,
        baseline_name: str,
    ) -> pd.DataFrame:
        """Compare a single challenger method against a baseline.

        Parameters
        ----------
        challenger_df : pd.DataFrame
            Projection results for the challenger method.
        baseline_df : pd.DataFrame
            Projection results for the baseline method.
        method_name : str
            Label for the challenger.
        baseline_name : str
            Label for the baseline.

        Returns
        -------
        pd.DataFrame
            Rows with metric deltas by geography group and horizon band.
        """
        validate_dataframe(challenger_df, PROJECTION_RESULT_COLUMNS, method_name)
        validate_dataframe(baseline_df, PROJECTION_RESULT_COLUMNS, baseline_name)

        rows: list[dict[str, Any]] = []

        for horizon_band, horizon_filter in self._horizon_bands().items():
            c_sub = challenger_df[challenger_df["horizon"].isin(horizon_filter)]
            b_sub = baseline_df[baseline_df["horizon"].isin(horizon_filter)]

            for geo_group in ["all", *self.county_groups.keys()]:
                c_geo = self._filter_geography_group(c_sub, geo_group)
                b_geo = self._filter_geography_group(b_sub, geo_group)

                if c_geo.empty or b_geo.empty:
                    continue

                for metric_name, metric_fn in _DEFAULT_METRICS.items():
                    c_val = float(
                        metric_fn(c_geo["projected_value"], c_geo["actual_value"])
                    )
                    b_val = float(
                        metric_fn(b_geo["projected_value"], b_geo["actual_value"])
                    )
                    rows.append(
                        {
                            "method": method_name,
                            "baseline": baseline_name,
                            "metric_name": metric_name,
                            "method_value": c_val,
                            "baseline_value": b_val,
                            "delta": c_val - b_val,
                            "geography_group": geo_group,
                            "horizon_band": horizon_band,
                        }
                    )

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Component-swap analysis
    # ------------------------------------------------------------------

    def component_swap_analysis(
        self,
        swap_results: dict[str, pd.DataFrame],
    ) -> pd.DataFrame:
        """Analyse results from component-swapped runs.

        Parameters
        ----------
        swap_results : dict[str, pd.DataFrame]
            Mapping from swap label (e.g. ``"adv_migration_simple_fert_mort"``)
            to projection-result DataFrames.

        Returns
        -------
        pd.DataFrame
            Comparison table with columns ``swap_label``, ``metric_name``,
            ``value``, ``geography_group``, ``horizon_band``.
        """
        rows: list[dict[str, Any]] = []
        for swap_label, df in swap_results.items():
            validate_dataframe(df, PROJECTION_RESULT_COLUMNS, swap_label)

            for horizon_band, horizon_filter in self._horizon_bands().items():
                sub = df[df["horizon"].isin(horizon_filter)]

                for geo_group in ["all", *self.county_groups.keys()]:
                    geo_sub = self._filter_geography_group(sub, geo_group)
                    if geo_sub.empty:
                        continue

                    for metric_name, metric_fn in _DEFAULT_METRICS.items():
                        val = float(
                            metric_fn(
                                geo_sub["projected_value"], geo_sub["actual_value"]
                            )
                        )
                        rows.append(
                            {
                                "swap_label": swap_label,
                                "metric_name": metric_name,
                                "value": val,
                                "geography_group": geo_group,
                                "horizon_band": horizon_band,
                            }
                        )

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Horizon blending
    # ------------------------------------------------------------------

    def horizon_blend(
        self,
        near_term_df: pd.DataFrame,
        long_term_df: pd.DataFrame,
        blend_horizon: int,
        method_name: str = "blended",
    ) -> pd.DataFrame:
        """Create a blended projection transitioning between two methods.

        For horizons <= *blend_horizon* the near-term method is used;
        beyond that the long-term method is used.  Within the transition
        band (blend_horizon to blend_horizon + transition_width) a linear
        weight is applied.

        Parameters
        ----------
        near_term_df : pd.DataFrame
            Projection results for the near-term method.
        long_term_df : pd.DataFrame
            Projection results for the long-term method.
        blend_horizon : int
            Horizon year at which the transition begins.
        method_name : str
            Label for the blended result.

        Returns
        -------
        pd.DataFrame
            Blended projection result with the same columns as the inputs.
        """
        validate_dataframe(near_term_df, PROJECTION_RESULT_COLUMNS, "near_term")
        validate_dataframe(long_term_df, PROJECTION_RESULT_COLUMNS, "long_term")

        merged = near_term_df.merge(
            long_term_df,
            on=PROJECTION_JOIN_KEYS,
            suffixes=("_near", "_long"),
            how="outer",
        )

        transition_width = max(1, blend_horizon // 2)
        transition_end = blend_horizon + transition_width

        def _weight_near(horizon: int) -> float:
            if horizon <= blend_horizon:
                return 1.0
            if horizon >= transition_end:
                return 0.0
            return float(
                (transition_end - horizon) / (transition_end - blend_horizon)
            )

        w_near = merged["horizon"].apply(_weight_near)
        w_long = 1.0 - w_near

        # Fill missing projected values with zeros to allow outer-join blending
        near_proj = merged.get("projected_value_near", pd.Series(0.0, index=merged.index)).fillna(0.0)
        long_proj = merged.get("projected_value_long", pd.Series(0.0, index=merged.index)).fillna(0.0)

        blended = merged[PROJECTION_JOIN_KEYS].copy()
        blended["projected_value"] = w_near * near_proj + w_long * long_proj

        # Carry actual and base values from whichever side is available
        for col in ("actual_value", "base_value"):
            near_col = merged.get(f"{col}_near")
            long_col = merged.get(f"{col}_long")
            if near_col is not None and long_col is not None:
                blended[col] = near_col.fillna(long_col)
            elif near_col is not None:
                blended[col] = near_col
            elif long_col is not None:
                blended[col] = long_col
            else:
                blended[col] = 0.0

        blended["run_id"] = method_name
        return blended[_RESULT_COLUMNS]

    # ------------------------------------------------------------------
    # Ensemble methods
    # ------------------------------------------------------------------

    def ensemble_average(
        self,
        method_results: dict[str, pd.DataFrame],
        weights: dict[str, float] | None = None,
    ) -> pd.DataFrame:
        """Create an ensemble projection by (optionally weighted) averaging.

        Parameters
        ----------
        method_results : dict[str, pd.DataFrame]
            Mapping from method name to projection-result DataFrames.
        weights : dict[str, float] | None
            Optional mapping from method name to weight.  If ``None``,
            equal weights are used.

        Returns
        -------
        pd.DataFrame
            Ensemble-averaged projection result.
        """
        if not method_results:
            raise ValueError("method_results must not be empty")

        names = list(method_results.keys())
        if weights is None:
            w = {n: 1.0 / len(names) for n in names}
        else:
            total = sum(weights.values())
            if total == 0:
                raise ValueError("Sum of weights must be nonzero")
            w = {n: weights.get(n, 0.0) / total for n in names}

        # Validate all DataFrames
        for name, df in method_results.items():
            validate_dataframe(df, PROJECTION_RESULT_COLUMNS, name)

        # Start from the first method and merge in the rest
        base_df = method_results[names[0]][PROJECTION_JOIN_KEYS + ["actual_value", "base_value"]].copy()
        base_df["projected_value"] = (
            method_results[names[0]]["projected_value"] * w[names[0]]
        )

        for name in names[1:]:
            other = method_results[name]
            merged = base_df.merge(
                other[PROJECTION_JOIN_KEYS + ["projected_value"]],
                on=PROJECTION_JOIN_KEYS,
                how="outer",
                suffixes=("", f"_{name}"),
            )
            proj_col = f"projected_value_{name}"
            merged["projected_value"] = (
                merged["projected_value"].fillna(0.0)
                + merged[proj_col].fillna(0.0) * w[name]
            )
            merged.drop(columns=[proj_col], inplace=True)
            base_df = merged

        base_df["run_id"] = "ensemble"
        # Fill any NaN actual/base from the merge
        base_df["actual_value"] = base_df["actual_value"].fillna(0.0)
        base_df["base_value"] = base_df["base_value"].fillna(0.0)
        return base_df[_RESULT_COLUMNS]

    def ensemble_by_county_type(
        self,
        method_results: dict[str, pd.DataFrame],
        county_groups: dict[str, list[str]],
        selector: dict[str, str],
    ) -> pd.DataFrame:
        """Build a county-type-specific ensemble.

        For each county-type group, selects the method specified by
        *selector* and concatenates the results.

        Parameters
        ----------
        method_results : dict[str, pd.DataFrame]
            Mapping from method name to projection-result DataFrames.
        county_groups : dict[str, list[str]]
            Mapping from group name to list of FIPS codes.
        selector : dict[str, str]
            Mapping from group name (or ``"_default"``) to the method
            name to use for that group.

        Returns
        -------
        pd.DataFrame
            Combined result using the selected method per county group.
        """
        for name, df in method_results.items():
            validate_dataframe(df, PROJECTION_RESULT_COLUMNS, name)

        # Build a set of assigned FIPS codes
        assigned_fips: set[str] = set()
        for fips_list in county_groups.values():
            assigned_fips.update(fips_list)

        parts: list[pd.DataFrame] = []

        for group_name, fips_list in county_groups.items():
            method_name = selector.get(group_name, selector.get("_default", ""))
            if method_name not in method_results:
                logger.warning(
                    "Selector references unknown method '%s' for group '%s'; skipping",
                    method_name,
                    group_name,
                )
                continue
            df = method_results[method_name]
            group_df = df[df["geography"].isin(fips_list)].copy()
            parts.append(group_df)

        # Handle remaining geographies with default method
        default_method = selector.get("_default", "")
        if default_method and default_method in method_results:
            df = method_results[default_method]
            remaining = df[~df["geography"].isin(assigned_fips)].copy()
            parts.append(remaining)

        if not parts:
            return pd.DataFrame(columns=_RESULT_COLUMNS)

        result = pd.concat(parts, ignore_index=True)
        result["run_id"] = "ensemble_county_type"
        return result[_RESULT_COLUMNS]

    # ------------------------------------------------------------------
    # Ranking and improvement
    # ------------------------------------------------------------------

    def rank_methods(
        self,
        comparison_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Rank methods by each dimension in the comparison table.

        Parameters
        ----------
        comparison_df : pd.DataFrame
            Output of :meth:`compare_all`.

        Returns
        -------
        pd.DataFrame
            Columns ``method``, ``metric_name``, ``horizon_band``,
            ``geography_group``, ``method_value``, ``rank``.
            Rank 1 = best (lowest metric value for error metrics,
            closest to zero for bias metrics).
        """
        if comparison_df.empty:
            return pd.DataFrame(
                columns=[
                    "method",
                    "metric_name",
                    "horizon_band",
                    "geography_group",
                    "method_value",
                    "rank",
                ]
            )

        bias_metrics = {"mean_signed_error", "mean_signed_percentage_error"}
        group_cols = ["metric_name", "horizon_band", "geography_group"]

        rows: list[dict[str, Any]] = []
        for group_key, group_df in comparison_df.groupby(group_cols):
            metric_name = group_key[0]  # type: ignore[index]
            horizon_band = group_key[1]  # type: ignore[index]
            geo_group = group_key[2]  # type: ignore[index]

            if metric_name in bias_metrics:
                sort_vals = group_df["method_value"].abs()
            else:
                sort_vals = group_df["method_value"]

            ranks = sort_vals.rank(method="min").astype(int)
            for idx, row in group_df.iterrows():
                rows.append(
                    {
                        "method": row["method"],
                        "metric_name": metric_name,
                        "horizon_band": horizon_band,
                        "geography_group": geo_group,
                        "method_value": row["method_value"],
                        "rank": int(ranks.loc[idx]),  # type: ignore[arg-type]
                    }
                )

        return pd.DataFrame(rows)

    def improvement_over_baseline(
        self,
        comparison_df: pd.DataFrame,
        metric: str = "mape",
    ) -> pd.DataFrame:
        """Compute signed improvement of each method over baseline.

        Positive values mean the method *improved* (lower error) relative
        to the baseline.

        Parameters
        ----------
        comparison_df : pd.DataFrame
            Output of :meth:`compare_all`.
        metric : str
            Metric to report improvement for.

        Returns
        -------
        pd.DataFrame
            Columns ``method``, ``horizon_band``, ``geography_group``,
            ``baseline_value``, ``method_value``, ``improvement``,
            ``improvement_pct``.
        """
        filtered = comparison_df[comparison_df["metric_name"] == metric].copy()
        if filtered.empty:
            return pd.DataFrame(
                columns=[
                    "method",
                    "horizon_band",
                    "geography_group",
                    "baseline_value",
                    "method_value",
                    "improvement",
                    "improvement_pct",
                ]
            )

        # improvement = baseline - method  (positive = challenger is better)
        filtered["improvement"] = filtered["baseline_value"] - filtered["method_value"]
        filtered["improvement_pct"] = np.where(
            filtered["baseline_value"] != 0,
            filtered["improvement"] / filtered["baseline_value"] * 100,
            0.0,
        )

        return filtered[
            [
                "method",
                "horizon_band",
                "geography_group",
                "baseline_value",
                "method_value",
                "improvement",
                "improvement_pct",
            ]
        ].reset_index(drop=True)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _horizon_bands(self) -> dict[str, list[int]]:
        """Return near-term, long-term, and overall horizon groupings."""
        all_h = self.horizons
        return {
            "near_term": [h for h in all_h if h <= self._hbands.near_max],
            "long_term": [h for h in all_h if h >= self._hbands.long_min],
            "all": all_h,
        }

    def _filter_geography_group(
        self, df: pd.DataFrame, group: str
    ) -> pd.DataFrame:
        """Filter *df* by county-group membership."""
        if group == "all":
            return df
        fips_list = self.county_groups.get(group, [])
        if not fips_list:
            return df.iloc[0:0]
        return df[df["geography"].isin(fips_list)]
