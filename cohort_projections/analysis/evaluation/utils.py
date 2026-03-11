"""Shared utility functions for the evaluation framework.

Consolidates repeated patterns (validation, diagnostic-record construction,
county-group resolution, DataFrame lookups, safe plotting, and grouped-metric
computation) used across multiple evaluation modules.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import pandas as pd

from .schemas import METRIC_REGISTRY

logger = logging.getLogger(__name__)

# Guarded import of matplotlib Figure type for type hints.
try:
    from matplotlib.figure import Figure
except ImportError:  # pragma: no cover
    Figure = object  # type: ignore[assignment,misc]


# ---------------------------------------------------------------------------
# DataFrame validation
# ---------------------------------------------------------------------------


def validate_dataframe(
    df: pd.DataFrame,
    required_cols: set[str] | frozenset[str],
    label: str = "input",
) -> None:
    """Check for required columns in a DataFrame.

    Args:
        df: DataFrame to validate.
        required_cols: Set of column names that must be present.
        label: Human-readable label used in the error message.

    Raises:
        ValueError: If *df* is missing one or more required columns.
    """
    missing = set(required_cols) - set(df.columns)
    if missing:
        raise ValueError(
            f"{label} DataFrame missing required columns: {sorted(missing)}"
        )


# ---------------------------------------------------------------------------
# Diagnostic record factory
# ---------------------------------------------------------------------------


def make_diagnostic_record(
    run_id: str,
    metric_name: str,
    metric_group: str,
    geography: str,
    target: str,
    value: float,
    *,
    geography_group: str = "",
    horizon: int | None = None,
    comparison_run_id: str = "",
    notes: str = "",
) -> dict[str, Any]:
    """Build a dict compatible with :class:`DiagnosticRecord`.

    Args:
        run_id: Identifier for the projection run.
        metric_name: Name of the metric (e.g. ``"mape"``).
        metric_group: Metric family (e.g. ``"accuracy"``, ``"bias"``).
        geography: FIPS code, ``"state"``, or aggregate label.
        target: Target variable (e.g. ``"population"``).
        value: Computed metric value.
        geography_group: County-type group (e.g. ``"bakken"``, ``"rural"``).
        horizon: Forecast horizon in years, or ``None``.
        comparison_run_id: Run ID of a comparison run, if applicable.
        notes: Free-text annotation.

    Returns:
        Dictionary whose keys match :class:`DiagnosticRecord` fields.
    """
    return {
        "run_id": run_id,
        "metric_name": metric_name,
        "metric_group": metric_group,
        "geography": geography,
        "geography_group": geography_group,
        "target": target,
        "horizon": horizon,
        "value": value,
        "comparison_run_id": comparison_run_id,
        "notes": notes,
    }


# ---------------------------------------------------------------------------
# County-group resolution
# ---------------------------------------------------------------------------


def resolve_county_group(
    fips: str,
    county_groups: dict[str, list[str]],
) -> str:
    """Map a FIPS code to its county-type group name.

    Iterates *county_groups* (e.g. ``{"bakken": ["38053", ...], ...}``) and
    returns the first group whose list contains *fips*.  Returns ``"rural"``
    if no match is found.

    Args:
        fips: FIPS code to look up.
        county_groups: Mapping of group name to list of FIPS codes.

    Returns:
        County-group name, or ``"rural"`` as default.
    """
    for group_name, fips_list in county_groups.items():
        if fips in fips_list:
            return group_name
    return "rural"


# ---------------------------------------------------------------------------
# Lookup builder
# ---------------------------------------------------------------------------


def build_lookup(
    df: pd.DataFrame,
    key_cols: list[str],
    value_col: str,
) -> dict[tuple[Any, ...], Any]:
    """Build a lookup dictionary from a DataFrame.

    Args:
        df: Source DataFrame.
        key_cols: Columns whose values form the composite key (as a tuple).
        value_col: Column whose value is stored as the dict value.

    Returns:
        Mapping from ``tuple(row[key_cols])`` to ``row[value_col]``.
    """
    lookup: dict[tuple[Any, ...], Any] = {}
    for _, row in df.iterrows():
        key = tuple(row[col] for col in key_cols)
        lookup[key] = row[value_col]
    return lookup


# ---------------------------------------------------------------------------
# Safe plot wrapper
# ---------------------------------------------------------------------------


def safe_plot(
    plot_fn: Callable[..., Figure],
    *args: Any,
    **kwargs: Any,
) -> Figure | None:
    """Safely execute a plot function, catching exceptions.

    If *plot_fn* raises, the exception is logged as a warning and ``None``
    is returned instead of propagating the error.

    Args:
        plot_fn: Callable that returns a matplotlib ``Figure``.
        *args: Positional arguments forwarded to *plot_fn*.
        **kwargs: Keyword arguments forwarded to *plot_fn*.

    Returns:
        The ``Figure`` on success, or ``None`` on failure.
    """
    try:
        return plot_fn(*args, **kwargs)
    except Exception:
        logger.warning(
            "Could not generate plot from %s", plot_fn.__name__, exc_info=True
        )
        return None


# ---------------------------------------------------------------------------
# Grouped metric computation
# ---------------------------------------------------------------------------


def compute_grouped_metrics(
    df: pd.DataFrame,
    groupby_cols: list[str],
    metrics: dict[str, tuple[Callable[..., float], str]] | None = None,
    run_id: str = "",
    county_groups: dict[str, list[str]] | None = None,
    default_target: str = "population",
    extra_notes: str = "",
) -> list[dict[str, Any]]:
    """Compute metrics for each group and return diagnostic records.

    Extracts the repeated *groupby -> compute -> append* pattern found in
    :mod:`forecast_accuracy` and other modules.

    Args:
        df: DataFrame containing at least ``projected_value`` and
            ``actual_value`` columns, plus any columns listed in
            *groupby_cols*.
        groupby_cols: Columns to group by before computing metrics.
        metrics: Mapping of metric name to ``(function, metric_group)``.
            Defaults to :data:`METRIC_REGISTRY` if ``None``.
        run_id: Run identifier to embed in each record.  If empty and
            ``"run_id"`` is in *groupby_cols*, the value from the group
            keys is used instead.
        county_groups: Optional county-group mapping for geography
            resolution.
        default_target: Target label when ``"target"`` is not in
            *groupby_cols*.
        extra_notes: Text appended to the ``notes`` field of each record.

    Returns:
        List of dicts compatible with :class:`DiagnosticRecord`.
    """
    if metrics is None:
        metrics = METRIC_REGISTRY

    records: list[dict[str, Any]] = []

    for group_keys, grp in df.groupby(groupby_cols):
        # Normalise group_keys to a tuple even for a single groupby column
        if not isinstance(group_keys, tuple):
            group_keys = (group_keys,)

        key_map = dict(zip(groupby_cols, group_keys, strict=True))

        proj = grp["projected_value"]
        act = grp["actual_value"]

        # Resolve per-record fields from group keys or defaults
        effective_run_id = run_id or str(key_map.get("run_id", ""))
        horizon = int(key_map["horizon"]) if "horizon" in key_map else None
        geography = str(key_map["geography"]) if "geography" in key_map else "state"
        target = str(key_map["target"]) if "target" in key_map else default_target

        # Resolve geography group
        if county_groups is not None and geography != "state":
            geo_group = resolve_county_group(geography, county_groups)
        elif geography == "state" or (
            "geography_type" in grp.columns
            and grp["geography_type"].iloc[0] == "state"
        ):
            geo_group = "state"
        else:
            geo_group = ""

        for metric_name, (metric_fn, metric_group) in metrics.items():
            val = metric_fn(proj, act)
            records.append(
                make_diagnostic_record(
                    run_id=effective_run_id,
                    metric_name=metric_name,
                    metric_group=metric_group,
                    geography=geography,
                    target=target,
                    value=val,
                    geography_group=geo_group,
                    horizon=horizon,
                    notes=extra_notes,
                )
            )

    return records
