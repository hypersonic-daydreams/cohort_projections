"""Canonical column sets, metric registry, and horizon helpers.

Shared schema definitions used across all evaluation modules to eliminate
duplication of column lists, metric mappings, and horizon-band logic.
"""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd

from .metrics import (
    mae,
    mape,
    mean_signed_error,
    mean_signed_percentage_error,
    median_absolute_percentage_error,
    rmse,
    wape,
)

# ---------------------------------------------------------------------------
# Canonical column sets
# ---------------------------------------------------------------------------

PROJECTION_RESULT_COLUMNS: frozenset[str] = frozenset({
    "run_id",
    "geography",
    "geography_type",
    "year",
    "horizon",
    "sex",
    "age_group",
    "target",
    "projected_value",
    "actual_value",
    "base_value",
})

COMPONENT_RECORD_COLUMNS: frozenset[str] = frozenset({
    "run_id",
    "geography",
    "year",
    "horizon",
    "component",
    "projected_component_value",
    "actual_component_value",
})

DIAGNOSTIC_RECORD_COLUMNS: frozenset[str] = frozenset({
    "run_id",
    "metric_name",
    "metric_group",
    "geography",
    "geography_group",
    "target",
    "horizon",
    "value",
    "comparison_run_id",
    "notes",
})

# Order matters for joins — kept as a list.
PROJECTION_JOIN_KEYS: list[str] = [
    "geography",
    "geography_type",
    "year",
    "horizon",
    "sex",
    "age_group",
    "target",
]

# ---------------------------------------------------------------------------
# Metric registry
# ---------------------------------------------------------------------------

METRIC_REGISTRY: dict[str, tuple[Callable[..., float], str]] = {
    "mae": (mae, "accuracy"),
    "rmse": (rmse, "accuracy"),
    "mape": (mape, "accuracy"),
    "median_ape": (median_absolute_percentage_error, "accuracy"),
    "wape": (wape, "accuracy"),
    "mean_signed_error": (mean_signed_error, "bias"),
    "mean_signed_percentage_error": (mean_signed_percentage_error, "bias"),
}

# ---------------------------------------------------------------------------
# Horizon bands
# ---------------------------------------------------------------------------


class HorizonBands:
    """Manage near-term / long-term horizon partitioning from config."""

    def __init__(self, near_max: int = 5, long_min: int = 10) -> None:
        self.near_max = near_max
        self.long_min = long_min

    def near_term_mask(self, df: pd.DataFrame) -> pd.Series:
        """Return a boolean mask for near-term horizons."""
        return df["horizon"] <= self.near_max

    def long_term_mask(self, df: pd.DataFrame) -> pd.Series:
        """Return a boolean mask for long-term horizons."""
        return df["horizon"] >= self.long_min
