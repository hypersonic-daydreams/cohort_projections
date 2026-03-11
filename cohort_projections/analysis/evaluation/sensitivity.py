"""Sensitivity and Robustness module (Module 3 of the Evaluation Blueprint).

This module identifies which assumptions matter and where fragility appears
by re-running projections under perturbed inputs and alternative parameter
settings.  It requires a *projection runner callable* so that projections
can be re-executed with modified configurations.

The callable signature is::

    run_projection_fn(overrides: dict) -> pd.DataFrame

where ``overrides`` is a dict whose keys are parameter names (or the
special keys ``"__perturbation__"`` and ``"__origin_year__"``) and whose
values are the values to use.  The returned DataFrame must contain at
least the columns ``geography``, ``year``, ``horizon``, ``projected_value``,
and ``actual_value``.
"""

from __future__ import annotations

import itertools
import logging
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from .metrics import mae, mape, mean_signed_percentage_error
from .schemas import HorizonBands
from .utils import build_lookup, validate_dataframe

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_REQUIRED_COLS = frozenset(
    {"geography", "year", "horizon", "projected_value", "actual_value"}
)


def _error_metrics(projected: np.ndarray, actual: np.ndarray) -> dict[str, float]:
    """Return a dict of standard error metrics for two arrays."""
    return {
        "mae": mae(projected, actual),
        "mape": mape(projected, actual),
        "bias": mean_signed_percentage_error(projected, actual),
    }


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class SensitivityModule:
    """Sensitivity and robustness analysis for population projections.

    Args:
        run_projection_fn: Callable that accepts a dict of parameter
            overrides and returns a projection-results DataFrame.
        config: The ``sensitivity`` section from ``evaluation_config.yaml``.
    """

    def __init__(
        self,
        run_projection_fn: Callable[[dict[str, Any]], pd.DataFrame],
        config: dict[str, Any],
    ) -> None:
        self.run_projection_fn = run_projection_fn
        self.config = config
        self._perturbation_pcts: dict[str, list[float]] = config.get(
            "perturbation_pct",
            {"base_population": [0.5, 1.0], "births": [1.0], "deaths": [1.0], "migration": [5.0]},
        )
        self._sweep_levels: int = int(config.get("parameter_sweep_levels", 3))
        self._mc_iterations: int = int(config.get("monte_carlo_iterations", 500))
        self._near_term_max: int = int(config.get("near_term_max_horizon", 5))
        self._long_term_min: int = int(config.get("long_term_min_horizon", 10))
        self._horizon_bands = HorizonBands(
            near_max=self._near_term_max, long_min=self._long_term_min
        )

    # ------------------------------------------------------------------
    # Parameter sweep
    # ------------------------------------------------------------------

    def parameter_sweep(
        self,
        param_name: str,
        values: list[Any],
        baseline_results: pd.DataFrame,
    ) -> pd.DataFrame:
        """Sweep a single parameter through *values* and score each run.

        Args:
            param_name: Name of the parameter to vary.
            values: List of values to test.
            baseline_results: Results DataFrame from the baseline run.

        Returns:
            DataFrame with one row per parameter value containing columns:
            ``param_name``, ``param_value``, ``near_term_error``,
            ``long_term_error``, ``bias``, ``realism_score``,
            ``stability_score``.
        """
        validate_dataframe(baseline_results, _REQUIRED_COLS, "baseline_results")

        records: list[dict[str, Any]] = []
        for val in values:
            logger.info("Parameter sweep: %s = %s", param_name, val)
            run_results = self.run_projection_fn({param_name: val})
            validate_dataframe(run_results, _REQUIRED_COLS, f"run({param_name}={val})")

            projected = run_results["projected_value"].to_numpy()
            actual = run_results["actual_value"].to_numpy()

            # Near-term error
            nt = self._horizon_bands.near_term_mask(run_results)
            near_err = mape(projected[nt], actual[nt]) if nt.any() else float("nan")

            # Long-term error
            lt = self._horizon_bands.long_term_mask(run_results)
            long_err = mape(projected[lt], actual[lt]) if lt.any() else float("nan")

            # Bias
            bias = mean_signed_percentage_error(projected, actual)

            # Realism score: 1 - normalised JSD proxy (uses MAE as stand-in)
            realism = max(0.0, 1.0 - mae(projected, actual) / max(actual.mean(), 1.0))

            # Stability: correlation with baseline
            base_proj = baseline_results["projected_value"].to_numpy()
            if len(base_proj) == len(projected) and np.std(base_proj) > 0:
                corr = float(np.corrcoef(base_proj, projected)[0, 1])
                stability = max(0.0, corr)
            else:
                stability = float("nan")

            records.append({
                "param_name": param_name,
                "param_value": val,
                "near_term_error": near_err,
                "long_term_error": long_err,
                "bias": bias,
                "realism_score": realism,
                "stability_score": stability,
            })

        return pd.DataFrame(records)

    # ------------------------------------------------------------------
    # Interaction sweep
    # ------------------------------------------------------------------

    def interaction_sweep(
        self,
        param_pairs: list[tuple[str, str]],
        values_dict: dict[str, list[Any]],
        baseline_results: pd.DataFrame,
    ) -> pd.DataFrame:
        """Factorial test of parameter pairs.

        Args:
            param_pairs: List of ``(param_a, param_b)`` tuples to test.
            values_dict: Mapping of param name to list of values.
            baseline_results: Baseline projection results.

        Returns:
            DataFrame with columns ``param_a``, ``param_b``,
            ``value_a``, ``value_b``, ``near_term_error``,
            ``long_term_error``, ``bias``.
        """
        validate_dataframe(baseline_results, _REQUIRED_COLS, "baseline_results")

        records: list[dict[str, Any]] = []
        for param_a, param_b in param_pairs:
            vals_a = values_dict.get(param_a, [])
            vals_b = values_dict.get(param_b, [])
            for va, vb in itertools.product(vals_a, vals_b):
                logger.info(
                    "Interaction sweep: %s=%s, %s=%s", param_a, va, param_b, vb
                )
                run_results = self.run_projection_fn({param_a: va, param_b: vb})
                validate_dataframe(run_results, _REQUIRED_COLS)

                projected = run_results["projected_value"].to_numpy()
                actual = run_results["actual_value"].to_numpy()

                nt = self._horizon_bands.near_term_mask(run_results)
                lt = self._horizon_bands.long_term_mask(run_results)

                records.append({
                    "param_a": param_a,
                    "param_b": param_b,
                    "value_a": va,
                    "value_b": vb,
                    "near_term_error": mape(projected[nt], actual[nt]) if nt.any() else float("nan"),
                    "long_term_error": mape(projected[lt], actual[lt]) if lt.any() else float("nan"),
                    "bias": mean_signed_percentage_error(projected, actual),
                })

        return pd.DataFrame(records)

    # ------------------------------------------------------------------
    # Perturbation test
    # ------------------------------------------------------------------

    def perturbation_test(
        self,
        baseline_inputs: dict[str, Any],
        perturbation_pcts: dict[str, list[float]] | None,
        baseline_results: pd.DataFrame,
    ) -> pd.DataFrame:
        """Small perturbation stability tests.

        Perturbs each input component by the given percentage amounts
        (both positive and negative) and measures the response.

        Args:
            baseline_inputs: Dict of baseline input component names
                (e.g. ``"base_population"``, ``"births"``).
            perturbation_pcts: Mapping of component name to list of
                perturbation percentages.  If ``None``, uses config
                defaults.
            baseline_results: Baseline projection results.

        Returns:
            DataFrame with columns ``component``, ``perturbation_pct``,
            ``direction``, ``geography``, ``horizon``,
            ``baseline_value``, ``perturbed_value``,
            ``abs_change``, ``sensitivity_index``.
        """
        validate_dataframe(baseline_results, _REQUIRED_COLS, "baseline_results")
        if perturbation_pcts is None:
            perturbation_pcts = self._perturbation_pcts

        records: list[dict[str, Any]] = []
        for component, pcts in perturbation_pcts.items():
            for pct in pcts:
                for direction, sign in [("positive", 1), ("negative", -1)]:
                    override = {
                        "__perturbation__": {
                            "component": component,
                            "pct": pct * sign,
                        }
                    }
                    logger.info(
                        "Perturbation test: %s %+.1f%%", component, pct * sign
                    )
                    run_results = self.run_projection_fn(override)
                    validate_dataframe(run_results, _REQUIRED_COLS)

                    # Match rows by geography and horizon
                    merged = baseline_results.merge(
                        run_results,
                        on=["geography", "horizon"],
                        suffixes=("_base", "_pert"),
                    )

                    for _, row in merged.iterrows():
                        base_val = float(row["projected_value_base"])
                        pert_val = float(row["projected_value_pert"])
                        abs_change = abs(pert_val - base_val)
                        # Sensitivity index: % change in output / % change in input
                        if pct != 0 and base_val != 0:
                            sens_idx = abs((pert_val - base_val) / base_val * 100) / pct
                        else:
                            sens_idx = float("nan")

                        records.append({
                            "component": component,
                            "perturbation_pct": pct * sign,
                            "direction": direction,
                            "geography": row["geography"],
                            "horizon": row["horizon"],
                            "baseline_value": base_val,
                            "perturbed_value": pert_val,
                            "abs_change": abs_change,
                            "sensitivity_index": sens_idx,
                        })

        return pd.DataFrame(records)

    # ------------------------------------------------------------------
    # Monte Carlo uncertainty propagation
    # ------------------------------------------------------------------

    def monte_carlo_propagation(
        self,
        baseline_inputs: dict[str, Any],
        n_iterations: int | None,
        baseline_results: pd.DataFrame,
        seed: int | None = None,
    ) -> pd.DataFrame:
        """Monte Carlo uncertainty propagation.

        Simulates *n_iterations* draws where each input component is
        perturbed by a random normal amount scaled to its configured
        perturbation percentage.

        Args:
            baseline_inputs: Dict of baseline input component names.
            n_iterations: Number of MC draws.  ``None`` uses the config
                default.
            baseline_results: Baseline projection results.
            seed: Optional RNG seed for reproducibility.

        Returns:
            DataFrame with columns ``geography``, ``horizon``,
            ``mean_projected``, ``p05``, ``p25``, ``p50``, ``p75``,
            ``p95``, ``interval_width``,
            ``uncertainty_contribution_<component>``, and
            ``uncertainty_amplification``.
        """
        validate_dataframe(baseline_results, _REQUIRED_COLS, "baseline_results")
        if n_iterations is None:
            n_iterations = self._mc_iterations

        rng = np.random.default_rng(seed)

        # Collect all iteration results keyed by (geography, horizon)
        geo_horizons = (
            baseline_results[["geography", "horizon"]]
            .drop_duplicates()
            .values.tolist()
        )
        # Matrix: rows = iterations, cols = geo-horizon combinations
        all_projections: dict[tuple[str, int], list[float]] = {
            (g, h): [] for g, h in geo_horizons
        }
        components = list(self._perturbation_pcts.keys())

        for _i in range(n_iterations):
            # Draw perturbations for all components simultaneously
            perturbations: dict[str, float] = {}
            for comp in components:
                max_pct = max(self._perturbation_pcts[comp]) if self._perturbation_pcts[comp] else 1.0
                perturbations[comp] = float(rng.normal(0, max_pct))

            override: dict[str, Any] = {
                "__perturbation__": {
                    "components": perturbations,
                }
            }
            run_results = self.run_projection_fn(override)
            validate_dataframe(run_results, _REQUIRED_COLS)

            for _, row in run_results.iterrows():
                key = (str(row["geography"]), int(row["horizon"]))
                if key in all_projections:
                    all_projections[key].append(float(row["projected_value"]))

        # Build summary statistics
        records: list[dict[str, Any]] = []
        for (geo, hor), values in all_projections.items():
            if not values:
                continue
            arr = np.array(values)
            p05, p25, p50, p75, p95 = np.percentile(arr, [5, 25, 50, 75, 95])
            interval_width = p95 - p05

            # Uncertainty amplification: CV of MC draws relative to mean
            mean_val = float(np.mean(arr))
            std_val = float(np.std(arr))
            amplification = std_val / abs(mean_val) if mean_val != 0 else float("nan")

            record: dict[str, Any] = {
                "geography": geo,
                "horizon": hor,
                "mean_projected": mean_val,
                "std_projected": std_val,
                "p05": float(p05),
                "p25": float(p25),
                "p50": float(p50),
                "p75": float(p75),
                "p95": float(p95),
                "interval_width": float(interval_width),
                "uncertainty_amplification": amplification,
            }
            records.append(record)

        return pd.DataFrame(records)

    # ------------------------------------------------------------------
    # Base-year sensitivity
    # ------------------------------------------------------------------

    def base_year_sensitivity(
        self,
        origin_years: list[int],
        baseline_results: pd.DataFrame,
    ) -> pd.DataFrame:
        """Re-run projections from different base years and compare.

        Args:
            origin_years: List of projection origin years to test.
            baseline_results: Baseline projection results.

        Returns:
            DataFrame with columns ``origin_year``, ``geography``,
            ``horizon``, ``projected_value``, ``baseline_projected``,
            ``deviation``, ``deviation_pct``.
        """
        validate_dataframe(baseline_results, _REQUIRED_COLS, "baseline_results")

        records: list[dict[str, Any]] = []
        base_lookup = build_lookup(
            baseline_results, ["geography", "horizon"], "projected_value"
        )

        for origin in origin_years:
            logger.info("Base-year sensitivity: origin=%d", origin)
            run_results = self.run_projection_fn({"__origin_year__": origin})
            validate_dataframe(run_results, _REQUIRED_COLS)

            for _, row in run_results.iterrows():
                key = (row["geography"], row["horizon"])
                baseline_val = float(base_lookup.get(key, float("nan")))
                proj_val = float(row["projected_value"])
                deviation = proj_val - baseline_val
                dev_pct = (
                    deviation / abs(baseline_val) * 100
                    if baseline_val != 0 and not np.isnan(baseline_val)
                    else float("nan")
                )
                records.append({
                    "origin_year": origin,
                    "geography": row["geography"],
                    "horizon": row["horizon"],
                    "projected_value": proj_val,
                    "baseline_projected": baseline_val,
                    "deviation": deviation,
                    "deviation_pct": dev_pct,
                })

        return pd.DataFrame(records)

    # ------------------------------------------------------------------
    # History-window sensitivity
    # ------------------------------------------------------------------

    def history_window_sensitivity(
        self,
        windows: list[dict[str, Any]],
        baseline_results: pd.DataFrame,
    ) -> pd.DataFrame:
        """Test different training-history windows.

        Args:
            windows: List of window specs, each a dict with at least a
                ``"label"`` key and any keys the projection runner needs
                (e.g. ``"start_year"``, ``"end_year"``).
            baseline_results: Baseline projection results.

        Returns:
            DataFrame with columns ``window_label``, ``near_term_error``,
            ``long_term_error``, ``bias``, ``n_rows``.
        """
        validate_dataframe(baseline_results, _REQUIRED_COLS, "baseline_results")

        records: list[dict[str, Any]] = []
        for window in windows:
            label = window.get("label", str(window))
            logger.info("History-window sensitivity: %s", label)
            override = {"__history_window__": window}
            run_results = self.run_projection_fn(override)
            validate_dataframe(run_results, _REQUIRED_COLS)

            projected = run_results["projected_value"].to_numpy()
            actual = run_results["actual_value"].to_numpy()

            nt = self._horizon_bands.near_term_mask(run_results)
            lt = self._horizon_bands.long_term_mask(run_results)

            records.append({
                "window_label": label,
                "near_term_error": mape(projected[nt], actual[nt]) if nt.any() else float("nan"),
                "long_term_error": mape(projected[lt], actual[lt]) if lt.any() else float("nan"),
                "bias": mean_signed_percentage_error(projected, actual),
                "n_rows": len(run_results),
            })

        return pd.DataFrame(records)

    # ------------------------------------------------------------------
    # Shock-year sensitivity
    # ------------------------------------------------------------------

    def shock_year_sensitivity(
        self,
        shock_configs: list[dict[str, Any]],
        baseline_results: pd.DataFrame,
    ) -> pd.DataFrame:
        """Test shock-year exclusion / winsorization strategies.

        Args:
            shock_configs: List of shock config dicts, each containing
                a ``"label"`` key and override keys for the runner (e.g.
                ``"exclude_years"``, ``"winsorize_years"``).
            baseline_results: Baseline projection results.

        Returns:
            DataFrame with columns ``shock_label``, ``near_term_error``,
            ``long_term_error``, ``bias``, ``n_rows``.
        """
        validate_dataframe(baseline_results, _REQUIRED_COLS, "baseline_results")

        records: list[dict[str, Any]] = []
        for shock_cfg in shock_configs:
            label = shock_cfg.get("label", str(shock_cfg))
            logger.info("Shock-year sensitivity: %s", label)
            override = {"__shock_config__": shock_cfg}
            run_results = self.run_projection_fn(override)
            validate_dataframe(run_results, _REQUIRED_COLS)

            projected = run_results["projected_value"].to_numpy()
            actual = run_results["actual_value"].to_numpy()

            nt = self._horizon_bands.near_term_mask(run_results)
            lt = self._horizon_bands.long_term_mask(run_results)

            records.append({
                "shock_label": label,
                "near_term_error": mape(projected[nt], actual[nt]) if nt.any() else float("nan"),
                "long_term_error": mape(projected[lt], actual[lt]) if lt.any() else float("nan"),
                "bias": mean_signed_percentage_error(projected, actual),
                "n_rows": len(run_results),
            })

        return pd.DataFrame(records)

    # ------------------------------------------------------------------
    # Stability index
    # ------------------------------------------------------------------

    def compute_stability_index(
        self,
        perturbation_results: pd.DataFrame,
    ) -> pd.DataFrame:
        """Aggregate perturbation results into a stability ranking.

        Args:
            perturbation_results: Output of ``perturbation_test()``.

        Returns:
            DataFrame with columns ``geography``, ``mean_sensitivity``,
            ``max_sensitivity``, ``stability_rank``,
            ``disproportionate_flag``.  Lower rank = more stable.
        """
        validate_dataframe(
            perturbation_results,
            frozenset({"geography", "sensitivity_index"}),
            "perturbation_results",
        )

        grouped = perturbation_results.groupby("geography")["sensitivity_index"]
        summary = pd.DataFrame({
            "geography": grouped.mean().index,
            "mean_sensitivity": grouped.mean().values,
            "max_sensitivity": grouped.max().values,
        })

        # Rank by mean sensitivity (lower = more stable)
        summary = summary.sort_values("mean_sensitivity").reset_index(drop=True)
        summary["stability_rank"] = range(1, len(summary) + 1)

        # Disproportionate flag: sensitivity index > 2.0 means a 1% input
        # change produces >2% output change
        disproportionate_threshold = float(
            self.config.get("disproportionate_threshold", 2.0)
        )
        summary["disproportionate_flag"] = (
            summary["max_sensitivity"] > disproportionate_threshold
        )

        return summary
