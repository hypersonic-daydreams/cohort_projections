"""Model scorecard: multi-axis scoring for projection evaluation.

Computes six top-level scores and a weighted composite for each model run,
enabling structured comparison between methods.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from .data_structures import ScorecardEntry

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default weights (overridden by config)
# ---------------------------------------------------------------------------
_DEFAULT_WEIGHTS: dict[str, float] = {
    "near_term_accuracy": 0.25,
    "long_term_accuracy": 0.25,
    "bias_calibration": 0.15,
    "age_structure_realism": 0.15,
    "robustness_stability": 0.10,
    "interpretability": 0.10,
}


class ModelScorecard:
    """Build and compare multi-axis scorecards for projection runs.

    Args:
        config: Dictionary containing at least ``scorecard_weights``,
            ``near_term_max_horizon``, and ``long_term_min_horizon``.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.weights: dict[str, float] = config.get(
            "scorecard_weights", _DEFAULT_WEIGHTS
        )
        self.near_term_max: int = config.get("near_term_max_horizon", 5)
        self.long_term_min: int = config.get("long_term_min_horizon", 10)

    # ------------------------------------------------------------------
    # Core builder
    # ------------------------------------------------------------------

    def build_scorecard(
        self,
        accuracy_diagnostics: pd.DataFrame,
        realism_diagnostics: pd.DataFrame | None = None,
        sensitivity_diagnostics: pd.DataFrame | None = None,
        interpretability_score: float | None = None,
        *,
        run_id: str = "",
        model_name: str = "",
    ) -> ScorecardEntry:
        """Build a scorecard from module outputs.

        Args:
            accuracy_diagnostics: DataFrame with ``metric_name``,
                ``horizon``, ``value`` (at minimum).
            realism_diagnostics: DataFrame with ``metric_name``, ``value``
                for age-structure / realism checks.  Optional.
            sensitivity_diagnostics: DataFrame with ``metric_name``,
                ``value`` for robustness/stability checks.  Optional.
            interpretability_score: Qualitative score between 0 and 1.
                If ``None``, defaults to 0.5.
            run_id: Run identifier.
            model_name: Model name.

        Returns:
            Populated ``ScorecardEntry``.
        """
        near_acc = self._compute_near_term_accuracy(accuracy_diagnostics)
        long_acc = self._compute_long_term_accuracy(accuracy_diagnostics)
        bias = self._compute_bias_calibration(accuracy_diagnostics)
        age_real = self._compute_age_realism(realism_diagnostics)
        robust = self._compute_robustness(
            accuracy_diagnostics, sensitivity_diagnostics
        )
        interp = interpretability_score if interpretability_score is not None else 0.5

        entry = ScorecardEntry(
            run_id=run_id,
            model_name=model_name,
            near_term_accuracy=near_acc,
            long_term_accuracy=long_acc,
            bias_calibration=bias,
            age_structure_realism=age_real,
            robustness_stability=robust,
            interpretability=interp,
            details={
                "near_term_max": self.near_term_max,
                "long_term_min": self.long_term_min,
            },
        )
        entry.composite_score = self.compute_composite(entry)
        return entry

    # ------------------------------------------------------------------
    # Composite computation
    # ------------------------------------------------------------------

    def compute_composite(self, entry: ScorecardEntry) -> float:
        """Compute weighted composite score for a scorecard entry.

        Lower composite is better.  Accuracy and bias axes are
        *error-like* (lower is better), while realism, robustness, and
        interpretability are *quality-like* (higher is better), so we
        invert the latter group for consistent direction.

        Args:
            entry: A populated ScorecardEntry.

        Returns:
            Weighted composite score (lower = better).
        """
        w = self.weights
        # Error axes (lower is better) - use directly
        score = (
            w.get("near_term_accuracy", 0) * entry.near_term_accuracy
            + w.get("long_term_accuracy", 0) * entry.long_term_accuracy
            + w.get("bias_calibration", 0) * abs(entry.bias_calibration)
            # Quality axes (higher is better) - invert so lower composite is better
            + w.get("age_structure_realism", 0) * (1.0 - entry.age_structure_realism)
            + w.get("robustness_stability", 0) * (1.0 - entry.robustness_stability)
            + w.get("interpretability", 0) * (1.0 - entry.interpretability)
        )
        return float(score)

    # ------------------------------------------------------------------
    # Comparison
    # ------------------------------------------------------------------

    def compare_scorecards(self, entries: list[ScorecardEntry]) -> pd.DataFrame:
        """Create a side-by-side comparison DataFrame.

        Args:
            entries: List of scorecard entries to compare.

        Returns:
            DataFrame with one row per entry and columns for each axis
            plus the composite score.
        """
        rows = [
            {
                "run_id": e.run_id,
                "model_name": e.model_name,
                "near_term_accuracy": e.near_term_accuracy,
                "long_term_accuracy": e.long_term_accuracy,
                "bias_calibration": e.bias_calibration,
                "age_structure_realism": e.age_structure_realism,
                "robustness_stability": e.robustness_stability,
                "interpretability": e.interpretability,
                "composite_score": e.composite_score,
            }
            for e in entries
        ]
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Human-readable summary
    # ------------------------------------------------------------------

    def render_summary(self, entries: list[ScorecardEntry]) -> str:
        """Generate a human-readable text summary of scorecard entries.

        Args:
            entries: List of scorecard entries.

        Returns:
            Formatted multi-line string.
        """
        lines: list[str] = ["=" * 60, "MODEL SCORECARD COMPARISON", "=" * 60, ""]

        for entry in entries:
            lines.append(f"Model: {entry.model_name}  (run: {entry.run_id})")
            lines.append("-" * 40)
            lines.append(f"  Near-term accuracy:    {entry.near_term_accuracy:.4f}")
            lines.append(f"  Long-term accuracy:    {entry.long_term_accuracy:.4f}")
            lines.append(f"  Bias / calibration:    {entry.bias_calibration:+.4f}")
            lines.append(f"  Age-structure realism: {entry.age_structure_realism:.4f}")
            lines.append(f"  Robustness / stability:{entry.robustness_stability:.4f}")
            lines.append(f"  Interpretability:      {entry.interpretability:.4f}")
            lines.append(f"  ** Composite score:    {entry.composite_score:.4f}")
            lines.append("")

        if len(entries) > 1:
            best = min(entries, key=lambda e: e.composite_score)
            lines.append(
                f"Best composite: {best.model_name} ({best.composite_score:.4f})"
            )
            lines.append("")

        lines.append("=" * 60)
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internal score computations
    # ------------------------------------------------------------------

    def _compute_near_term_accuracy(self, df: pd.DataFrame) -> float:
        """Mean MAPE across near-term horizons."""
        mask = (df["metric_name"] == "mape") & (df["horizon"] <= self.near_term_max)
        vals = df.loc[mask, "value"]
        if vals.empty:
            return float("nan")
        return float(vals.mean())

    def _compute_long_term_accuracy(self, df: pd.DataFrame) -> float:
        """Mean MAPE across long-term horizons."""
        mask = (df["metric_name"] == "mape") & (df["horizon"] >= self.long_term_min)
        vals = df.loc[mask, "value"]
        if vals.empty:
            return float("nan")
        return float(vals.mean())

    def _compute_bias_calibration(self, df: pd.DataFrame) -> float:
        """Mean signed percentage error across all horizons."""
        mask = df["metric_name"] == "mean_signed_percentage_error"
        vals = df.loc[mask, "value"]
        if vals.empty:
            return 0.0
        return float(vals.mean())

    def _compute_age_realism(self, df: pd.DataFrame | None) -> float:
        """Age-structure realism score from JSD values.

        Returns 1.0 minus the mean JSD, clamped to [0, 1].
        """
        if df is None or df.empty:
            return 0.5  # default when no realism data available
        mask = df["metric_name"] == "jsd"
        vals = df.loc[mask, "value"]
        if vals.empty:
            return 0.5
        return float(np.clip(1.0 - vals.mean(), 0.0, 1.0))

    def _compute_robustness(
        self,
        accuracy_df: pd.DataFrame,
        sensitivity_df: pd.DataFrame | None,
    ) -> float:
        """Robustness score based on cross-horizon stability.

        Uses inverse coefficient of variation of MAPE across horizons,
        normalised to [0, 1].  If sensitivity diagnostics are provided,
        also factors in perturbation response stability.
        """
        mape_vals = accuracy_df.loc[
            accuracy_df["metric_name"] == "mape", "value"
        ]
        if mape_vals.empty or mape_vals.std() == 0:
            base_robustness = 1.0
        else:
            cv = mape_vals.std() / max(mape_vals.mean(), 1e-10)
            base_robustness = float(np.clip(1.0 / (1.0 + cv), 0.0, 1.0))

        if sensitivity_df is not None and not sensitivity_df.empty:
            sens_vals = sensitivity_df.loc[
                sensitivity_df["metric_name"] == "mape", "value"
            ]
            if not sens_vals.empty and sens_vals.std() > 0:
                sens_cv = sens_vals.std() / max(sens_vals.mean(), 1e-10)
                sens_score = float(np.clip(1.0 / (1.0 + sens_cv), 0.0, 1.0))
                return 0.7 * base_robustness + 0.3 * sens_score

        return base_robustness
