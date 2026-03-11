"""Evaluation orchestrator: ties all five evaluation modules together.

Loads configuration from ``config/evaluation_config.yaml`` and provides
convenience methods for running full or partial evaluation pipelines.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from .data_structures import ScorecardEntry
from .metrics import (
    mae,
    mape,
    mean_signed_error,
    mean_signed_percentage_error,
    median_absolute_percentage_error,
    rmse,
    wape,
)
from .scorecard import ModelScorecard
from .visualization import (
    MATPLOTLIB_AVAILABLE,
    plot_bias_map,
    plot_component_blame,
    plot_county_horizon_heatmap,
    plot_horizon_profile,
    plot_stability_scatter,
    save_evaluation_report,
)

logger = logging.getLogger(__name__)

_DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parents[3] / "config" / "evaluation_config.yaml"
)

# Map config metric names to functions
_METRIC_FUNCS: dict[str, Callable[..., float]] = {
    "mae": mae,
    "rmse": rmse,
    "mape": mape,
    "median_ape": median_absolute_percentage_error,
    "wape": wape,
    "mean_signed_error": mean_signed_error,
    "mean_signed_percentage_error": mean_signed_percentage_error,
}


class EvaluationRunner:
    """Orchestrates the full evaluation pipeline.

    Args:
        config_path: Path to ``evaluation_config.yaml``.  If ``None``,
            uses the default project-relative path.
    """

    def __init__(self, config_path: str | Path | None = None) -> None:
        path = Path(config_path) if config_path else _DEFAULT_CONFIG_PATH
        with open(path) as fh:
            self.config: dict[str, Any] = yaml.safe_load(fh)
        logger.info("Loaded evaluation config from %s", path)

        self.horizons: list[int] = self.config.get("horizons", [1, 2, 3, 5, 10, 15, 20])
        self.county_groups: dict[str, list[str]] = self.config.get("county_groups", {})
        self.near_term_max: int = self.config.get("near_term_max_horizon", 5)
        self.long_term_min: int = self.config.get("long_term_min_horizon", 10)
        self.accuracy_metrics: list[str] = self.config.get(
            "accuracy_metrics",
            ["mae", "rmse", "mape", "mean_signed_error", "mean_signed_percentage_error"],
        )

        self._scorecard = ModelScorecard(self.config)

    # ------------------------------------------------------------------
    # Full evaluation
    # ------------------------------------------------------------------

    def run_full_evaluation(
        self,
        results_df: pd.DataFrame,
        components_df: pd.DataFrame | None = None,
        method_results_dict: dict[str, pd.DataFrame] | None = None,
        projection_runner_fn: Any = None,
        output_dir: str | Path | None = None,
    ) -> dict[str, Any]:
        """Run all five evaluation modules.

        Args:
            results_df: Tidy projection results with ``projected_value``,
                ``actual_value``, ``horizon``, ``geography``, ``run_id``,
                ``model_name``.
            components_df: Component-level data (births/deaths/migration).
                Optional.
            method_results_dict: Mapping ``{model_name: results_df}`` for
                multi-model comparison.  Optional.
            projection_runner_fn: Callable to re-run projections for
                sensitivity analysis.  Optional / reserved for future use.
            output_dir: If provided, save report to this directory.

        Returns:
            Dictionary with keys ``accuracy_diagnostics``,
            ``component_diagnostics``, ``scorecard``, ``comparison``,
            ``figures``.
        """
        out: dict[str, Any] = {}

        # Module 1 -- Accuracy
        accuracy_diag = self._compute_accuracy_diagnostics(results_df)
        out["accuracy_diagnostics"] = accuracy_diag

        # Module 2 -- Structural realism (age JSD if age data available)
        realism_diag = self._compute_realism_diagnostics(results_df)
        out["realism_diagnostics"] = realism_diag

        # Module 3 -- Component decomposition
        component_diag = None
        if components_df is not None and not components_df.empty:
            component_diag = components_df.copy()
        out["component_diagnostics"] = component_diag

        # Module 4 -- Benchmark comparison
        comparison_df = None
        if method_results_dict is not None:
            comparison_df = self.run_comparison(method_results_dict)
        out["comparison"] = comparison_df

        # Scorecard
        entry = self._scorecard.build_scorecard(
            accuracy_diagnostics=accuracy_diag,
            realism_diagnostics=realism_diag,
            run_id=results_df["run_id"].iloc[0] if "run_id" in results_df.columns else "",
            model_name=(
                results_df["model_name"].iloc[0]
                if "model_name" in results_df.columns
                else ""
            ),
        )
        out["scorecard"] = entry

        # Module 5 -- Visualizations
        figures: dict[str, Any] = {}
        if MATPLOTLIB_AVAILABLE:
            figures = self._generate_figures(
                accuracy_diag, component_diag, results_df
            )
        out["figures"] = figures

        # Save report if requested
        if output_dir is not None:
            self.generate_report(out, output_dir)

        return out

    # ------------------------------------------------------------------
    # Quick-run helpers
    # ------------------------------------------------------------------

    def run_accuracy_only(self, results_df: pd.DataFrame) -> pd.DataFrame:
        """Run accuracy module only.

        Args:
            results_df: Tidy projection results with ``projected_value``,
                ``actual_value``, ``horizon``, ``geography``.

        Returns:
            Diagnostics DataFrame.
        """
        return self._compute_accuracy_diagnostics(results_df)

    def run_comparison(
        self,
        method_results_dict: dict[str, pd.DataFrame],
        baseline_name: str | None = None,
    ) -> pd.DataFrame:
        """Run benchmark comparison across multiple methods.

        Args:
            method_results_dict: ``{model_name: results_df}``.
            baseline_name: Name of the baseline model.  If ``None``, uses
                the first key.

        Returns:
            Comparison DataFrame with diagnostics for each method.
        """
        all_diags: list[pd.DataFrame] = []

        for name, rdf in method_results_dict.items():
            diag = self._compute_accuracy_diagnostics(rdf)
            diag["model_name"] = name
            all_diags.append(diag)

        if not all_diags:
            return pd.DataFrame()

        combined = pd.concat(all_diags, ignore_index=True)

        # Add delta vs baseline
        if baseline_name is None:
            baseline_name = next(iter(method_results_dict))

        baseline_diag = combined.loc[combined["model_name"] == baseline_name]
        if not baseline_diag.empty:
            baseline_lookup = baseline_diag.set_index(
                ["metric_name", "horizon", "geography"]
            )["value"]
            deltas = []
            for _, row in combined.iterrows():
                key = (row["metric_name"], row["horizon"], row["geography"])
                base_val = baseline_lookup.get(key, float("nan"))
                deltas.append(row["value"] - base_val)
            combined["delta_vs_baseline"] = deltas

        return combined

    # ------------------------------------------------------------------
    # Report generation
    # ------------------------------------------------------------------

    def generate_report(
        self,
        evaluation_results: dict[str, Any],
        output_dir: str | Path,
    ) -> Path:
        """Save all evaluation outputs to *output_dir*.

        Args:
            evaluation_results: Output dict from ``run_full_evaluation``.
            output_dir: Target directory.

        Returns:
            Path to output directory.
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        # Save diagnostics
        acc = evaluation_results.get("accuracy_diagnostics")
        if acc is not None:
            acc.to_csv(out / "accuracy_diagnostics.csv", index=False)

        real = evaluation_results.get("realism_diagnostics")
        if real is not None and not real.empty:
            real.to_csv(out / "realism_diagnostics.csv", index=False)

        comp = evaluation_results.get("comparison")
        if comp is not None and not comp.empty:
            comp.to_csv(out / "comparison_diagnostics.csv", index=False)

        # Save scorecard
        entry = evaluation_results.get("scorecard")
        if isinstance(entry, ScorecardEntry):
            summary = self._scorecard.render_summary([entry])
            (out / "scorecard_summary.txt").write_text(summary)

        # Save figures
        figures = evaluation_results.get("figures", {})
        if figures and MATPLOTLIB_AVAILABLE:
            save_evaluation_report(
                out / "figures",
                evaluation_results.get("accuracy_diagnostics", pd.DataFrame()),
                figures,
            )

        logger.info("Evaluation report saved to %s", out)
        return out

    # ------------------------------------------------------------------
    # Internal computation helpers
    # ------------------------------------------------------------------

    def _compute_accuracy_diagnostics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute accuracy diagnostics for each horizon x geography."""
        records: list[dict[str, Any]] = []

        # Determine groupby columns
        group_cols = ["horizon"]
        if "geography" in df.columns:
            group_cols.append("geography")

        for keys, grp in df.groupby(group_cols):
            if len(group_cols) == 1:
                horizon = keys
                geography = "state"
            else:
                horizon, geography = keys

            proj = grp["projected_value"].values
            act = grp["actual_value"].values

            for metric_name in self.accuracy_metrics:
                func = _METRIC_FUNCS.get(metric_name)
                if func is None:
                    continue
                val = func(proj, act)
                geo_group = self._resolve_geography_group(str(geography))
                records.append(
                    {
                        "run_id": grp["run_id"].iloc[0] if "run_id" in grp.columns else "",
                        "metric_name": metric_name,
                        "metric_group": "accuracy",
                        "geography": str(geography),
                        "geography_group": geo_group,
                        "target": grp["target"].iloc[0] if "target" in grp.columns else "population",
                        "horizon": int(horizon),
                        "value": val,
                        "model_name": (
                            grp["model_name"].iloc[0]
                            if "model_name" in grp.columns
                            else ""
                        ),
                    }
                )

        return pd.DataFrame(records)

    def _compute_realism_diagnostics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute age-structure realism if age_group data is available."""
        if "age_group" not in df.columns:
            return pd.DataFrame()

        from .metrics import jensen_shannon_divergence

        records: list[dict[str, Any]] = []
        group_cols = ["horizon"]
        if "geography" in df.columns:
            group_cols.append("geography")

        # Filter to age-group level only (exclude totals)
        age_df = df.loc[df["age_group"] != "total"]
        if age_df.empty:
            return pd.DataFrame()

        for keys, grp in age_df.groupby(group_cols):
            if len(group_cols) == 1:
                horizon = keys
                geography = "state"
            else:
                horizon, geography = keys

            proj = grp.groupby("age_group")["projected_value"].sum()
            act = grp.groupby("age_group")["actual_value"].sum()

            if proj.sum() <= 0 or act.sum() <= 0:
                continue

            jsd = jensen_shannon_divergence(proj.values, act.values)
            records.append(
                {
                    "run_id": grp["run_id"].iloc[0] if "run_id" in grp.columns else "",
                    "metric_name": "jsd",
                    "metric_group": "realism",
                    "geography": str(geography),
                    "geography_group": self._resolve_geography_group(str(geography)),
                    "target": "age_distribution",
                    "horizon": int(horizon),
                    "value": jsd,
                }
            )

        return pd.DataFrame(records)

    def _resolve_geography_group(self, geography: str) -> str:
        """Map a FIPS code to its county group name."""
        for group_name, fips_list in self.county_groups.items():
            if geography in fips_list:
                return group_name
        if geography == "state":
            return "state"
        return "rural"

    def _generate_figures(
        self,
        accuracy_diag: pd.DataFrame,
        component_diag: pd.DataFrame | None,
        results_df: pd.DataFrame,
    ) -> dict[str, Any]:
        """Generate standard evaluation figures."""
        figures: dict[str, Any] = {}

        if accuracy_diag.empty:
            return figures

        try:
            figures["horizon_profile"] = plot_horizon_profile(
                accuracy_diag, "mape"
            )
        except Exception:
            logger.warning("Could not generate horizon profile plot", exc_info=True)

        try:
            figures["county_horizon_heatmap"] = plot_county_horizon_heatmap(
                accuracy_diag, "mape"
            )
        except Exception:
            logger.warning("Could not generate county heatmap", exc_info=True)

        try:
            figures["bias_map"] = plot_bias_map(accuracy_diag)
        except Exception:
            logger.warning("Could not generate bias map", exc_info=True)

        if component_diag is not None:
            try:
                figures["component_blame"] = plot_component_blame(component_diag)
            except Exception:
                logger.warning("Could not generate component blame plot", exc_info=True)

        try:
            figures["stability_scatter"] = plot_stability_scatter(
                accuracy_diag,
                near_term_max_horizon=self.near_term_max,
                long_term_min_horizon=self.long_term_min,
            )
        except Exception:
            logger.warning("Could not generate stability scatter", exc_info=True)

        return figures
