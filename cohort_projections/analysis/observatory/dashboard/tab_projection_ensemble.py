"""Projection ensemble tab for the Observatory Panel dashboard.

Spaghetti plots, state-level error-over-time, and uncertainty fan charts
for comparing projection curves across benchmark runs.  All charts react
to shared origin-year and run-selector filter widgets.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pandas as pd
import panel as pn
import plotly.graph_objects as go

from cohort_projections.analysis.observatory.dashboard import theme, widgets

if TYPE_CHECKING:
    from cohort_projections.analysis.observatory.dashboard.data_manager import (
        DashboardDataManager,
    )

logger = logging.getLogger(__name__)

# Standard origin years for walk-forward validation.
_ORIGIN_YEARS = [2005, 2010, 2015, 2020]


# ---------------------------------------------------------------------------
# Color helper
# ---------------------------------------------------------------------------

def _run_color(run_id: str, run_ids: list[str]) -> str:
    """Return a consistent color for a run based on its position."""
    idx = run_ids.index(run_id) if run_id in run_ids else 0
    return theme.EXPERIMENT_COLORS[idx % len(theme.EXPERIMENT_COLORS)]


# ---------------------------------------------------------------------------
# Section builders (bound to widget values via pn.bind)
# ---------------------------------------------------------------------------

def _build_spaghetti_plot(
    origin_year: str,
    selected_runs: list[str],
    curves: pd.DataFrame,
    state_metrics: pd.DataFrame,
    champion_id: str | None,
    all_run_ids: list[str],
) -> pn.pane.Plotly:
    """State-level spaghetti plot: one line per run, optionally filtered by origin."""
    if not selected_runs or curves.empty:
        return pn.pane.Plotly(go.Figure(), sizing_mode="stretch_width")

    filtered = curves[curves["run_id"].isin(selected_runs)].copy()
    if filtered.empty:
        return pn.pane.Plotly(go.Figure(), sizing_mode="stretch_width")

    # Apply origin year filter
    if origin_year != "All" and "origin_year" in filtered.columns:
        origin_int = int(origin_year)
        filtered = filtered[filtered["origin_year"] == origin_int]
        if filtered.empty:
            return pn.pane.Plotly(go.Figure(), sizing_mode="stretch_width")

    fig = go.Figure()

    # --- Actual population line ---
    if not state_metrics.empty:
        actual_cols = {"validation_year", "actual_state"}
        if actual_cols.issubset(state_metrics.columns):
            actuals = (
                state_metrics[["validation_year", "actual_state"]]
                .drop_duplicates()
                .sort_values("validation_year")
            )
            if not actuals.empty:
                fig.add_trace(
                    go.Scatter(
                        x=actuals["validation_year"],
                        y=actuals["actual_state"],
                        name="Actual (Census PEP)",
                        mode="lines+markers",
                        line={"color": "black", "width": 3},
                        marker={"size": 5, "symbol": "circle"},
                        legendgroup="actual",
                        hovertemplate=(
                            "<b>Actual</b><br>"
                            "Year: %{x}<br>"
                            "Population: %{y:,.0f}<extra></extra>"
                        ),
                    )
                )

    # --- Determine which column holds projected state population ---
    proj_col = None
    for candidate in ("projected_state", "projected_population", "population"):
        if candidate in filtered.columns:
            proj_col = candidate
            break

    if proj_col is None:
        fig.add_annotation(
            text="No projected population column found in curves data.",
            showarrow=False,
        )
        return pn.pane.Plotly(fig, sizing_mode="stretch_width")

    year_col = "year" if "year" in filtered.columns else "validation_year"
    if year_col not in filtered.columns:
        return pn.pane.Plotly(go.Figure(), sizing_mode="stretch_width")

    # Determine origins present in the data
    has_origin = "origin_year" in filtered.columns
    origins = sorted(filtered["origin_year"].unique()) if has_origin else [None]

    # --- Projection curves per run ---
    for run_id in selected_runs:
        run_data = filtered[filtered["run_id"] == run_id]
        if run_data.empty:
            continue

        is_champion = run_id == champion_id
        color = _run_color(run_id, all_run_ids)
        line_style = "dash" if is_champion else "solid"
        line_width = 2.5 if is_champion else 1.8

        for i, origin in enumerate(origins):
            if origin is not None:
                odf = run_data[run_data["origin_year"] == origin].sort_values(year_col)
            else:
                odf = run_data.sort_values(year_col)

            if odf.empty:
                continue

            show_legend = i == 0
            origin_label = f" (origin {origin})" if origin is not None else ""

            fig.add_trace(
                go.Scatter(
                    x=odf[year_col],
                    y=odf[proj_col],
                    name=run_id if show_legend else run_id,
                    mode="lines",
                    line={"color": color, "width": line_width, "dash": line_style},
                    legendgroup=run_id,
                    showlegend=show_legend,
                    opacity=0.85 if not is_champion else 0.95,
                    hovertemplate=(
                        f"<b>{run_id}</b>{origin_label}<br>"
                        "Year: %{x}<br>"
                        "Projected: %{y:,.0f}<extra></extra>"
                    ),
                )
            )

    title_suffix = f" (Origin {origin_year})" if origin_year != "All" else " (All Origins)"
    fig.update_layout(**theme.get_plotly_layout_defaults())
    fig.update_layout(
        title=f"State Population Projection Ensemble{title_suffix}",
        xaxis_title="Year",
        yaxis_title="Population",
        yaxis_tickformat=",",
        legend={"orientation": "h", "y": -0.18, "x": 0, "xanchor": "left"},
        height=550,
    )

    return pn.pane.Plotly(fig, sizing_mode="stretch_width")


def _build_error_over_time(
    origin_year: str,
    selected_runs: list[str],
    state_metrics: pd.DataFrame,
    all_run_ids: list[str],
) -> pn.pane.Plotly:
    """Line plot of percentage error vs horizon, one line per run."""
    if not selected_runs or state_metrics.empty:
        return pn.pane.Plotly(go.Figure(), sizing_mode="stretch_width")

    filtered = state_metrics[state_metrics["run_id"].isin(selected_runs)].copy()
    if filtered.empty:
        return pn.pane.Plotly(go.Figure(), sizing_mode="stretch_width")

    # Apply origin year filter
    if origin_year != "All" and "origin_year" in filtered.columns:
        origin_int = int(origin_year)
        filtered = filtered[filtered["origin_year"] == origin_int]
        if filtered.empty:
            return pn.pane.Plotly(go.Figure(), sizing_mode="stretch_width")

    # Determine the horizon and error columns
    horizon_col = None
    for candidate in ("horizon", "forecast_horizon", "horizon_years"):
        if candidate in filtered.columns:
            horizon_col = candidate
            break

    error_col = None
    for candidate in ("pct_error", "state_pct_error", "ape"):
        if candidate in filtered.columns:
            error_col = candidate
            break

    if horizon_col is None or error_col is None:
        return pn.pane.Plotly(go.Figure(), sizing_mode="stretch_width")

    fig = go.Figure()

    for run_id in selected_runs:
        run_data = filtered[filtered["run_id"] == run_id]
        if run_data.empty:
            continue

        color = _run_color(run_id, all_run_ids)

        # Aggregate across origins if showing all
        if origin_year == "All" and "origin_year" in run_data.columns:
            agg = (
                run_data.groupby(horizon_col)[error_col]
                .mean()
                .reset_index()
                .sort_values(horizon_col)
            )
        else:
            agg = run_data[[horizon_col, error_col]].sort_values(horizon_col)

        fig.add_trace(
            go.Scatter(
                x=agg[horizon_col],
                y=agg[error_col],
                name=run_id,
                mode="lines+markers",
                line={"color": color, "width": 2},
                marker={"size": 5},
                hovertemplate=(
                    f"<b>{run_id}</b><br>"
                    "Horizon: %{x} yr<br>"
                    "Error: %{y:.2f}%<extra></extra>"
                ),
            )
        )

    fig.add_hline(y=0, line_width=1, line_color=theme.SDC_DARK_GRAY)

    title_suffix = f" (Origin {origin_year})" if origin_year != "All" else " (Mean Across Origins)"
    fig.update_layout(**theme.get_plotly_layout_defaults())
    fig.update_layout(
        title=f"State Error Over Forecast Horizon{title_suffix}",
        xaxis_title="Forecast Horizon (years)",
        yaxis_title="Percentage Error (%)",
        legend={"orientation": "h", "y": -0.15},
        height=420,
    )

    return pn.pane.Plotly(fig, sizing_mode="stretch_width")


def _build_uncertainty_fan(
    selected_runs: list[str],
    uncertainty: pd.DataFrame,
) -> pn.Column:
    """Fan chart from uncertainty_summary if data is available."""
    if uncertainty.empty:
        return pn.Column(
            widgets.empty_placeholder(
                "No uncertainty data available. "
                "Run benchmarks with uncertainty estimation enabled."
            )
        )

    if not selected_runs:
        return pn.Column(widgets.empty_placeholder("Select at least one run."))

    filtered = uncertainty[uncertainty["run_id"].isin(selected_runs)].copy()
    if filtered.empty:
        return pn.Column(
            widgets.empty_placeholder("No uncertainty data for selected runs.")
        )

    # Look for expected quantile columns
    year_col = None
    for candidate in ("year", "validation_year", "horizon"):
        if candidate in filtered.columns:
            year_col = candidate
            break

    quantile_sets = [
        ("p5", "p95"),
        ("p25", "p75"),
        ("p50", None),
    ]

    # Check which quantile columns exist
    available_quantiles = {
        q for q_pair in quantile_sets for q in q_pair if q is not None and q in filtered.columns
    }

    if year_col is None or not available_quantiles:
        return pn.Column(
            widgets.empty_placeholder(
                "Uncertainty data found but missing expected columns "
                f"(need year + quantile columns). Columns: {list(filtered.columns)}"
            )
        )

    fig = go.Figure()

    # Use first selected run for the fan chart
    run_id = selected_runs[0]
    run_data = filtered[filtered["run_id"] == run_id].sort_values(year_col)

    if run_data.empty:
        return pn.Column(
            widgets.empty_placeholder(f"No uncertainty data for run {run_id}.")
        )

    # p5-p95 band
    if "p5" in run_data.columns and "p95" in run_data.columns:
        fig.add_trace(
            go.Scatter(
                x=run_data[year_col],
                y=run_data["p95"],
                mode="lines",
                line={"width": 0},
                showlegend=False,
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=run_data[year_col],
                y=run_data["p5"],
                mode="lines",
                line={"width": 0},
                fill="tonexty",
                fillcolor="rgba(5,99,193,0.15)",
                name="90% CI (p5-p95)",
                hovertemplate="p5: %{y:,.0f}<extra></extra>",
            )
        )

    # p25-p75 band
    if "p25" in run_data.columns and "p75" in run_data.columns:
        fig.add_trace(
            go.Scatter(
                x=run_data[year_col],
                y=run_data["p75"],
                mode="lines",
                line={"width": 0},
                showlegend=False,
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=run_data[year_col],
                y=run_data["p25"],
                mode="lines",
                line={"width": 0},
                fill="tonexty",
                fillcolor="rgba(5,99,193,0.30)",
                name="50% CI (p25-p75)",
                hovertemplate="p25: %{y:,.0f}<extra></extra>",
            )
        )

    # Median line
    if "p50" in run_data.columns:
        fig.add_trace(
            go.Scatter(
                x=run_data[year_col],
                y=run_data["p50"],
                mode="lines",
                line={"color": theme.SDC_BLUE, "width": 2.5},
                name="Median (p50)",
                hovertemplate=(
                    f"<b>{run_id}</b><br>"
                    "Year: %{x}<br>"
                    "Median: %{y:,.0f}<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        **theme.get_plotly_layout_defaults(),
        title=f"Uncertainty Bands ({run_id})",
        xaxis_title="Year",
        yaxis_title="Population",
        yaxis_tickformat=",",
        height=450,
    )

    return pn.Column(pn.pane.Plotly(fig, sizing_mode="stretch_width"))


# ---------------------------------------------------------------------------
# Main tab builder
# ---------------------------------------------------------------------------

def build_projection_ensemble(dm: DashboardDataManager) -> pn.Column:
    """Build the projection ensemble tab with spaghetti plots and error curves.

    Parameters
    ----------
    dm:
        The dashboard data manager providing access to all observatory data.

    Returns
    -------
    pn.Column
        A Panel Column containing all ensemble visualizations wired to
        shared origin-year and run-selector filter widgets.
    """
    run_ids = dm.run_ids
    curves = dm.projection_curves
    state_metrics = dm.state_metrics
    uncertainty = dm.uncertainty_summary
    champion_id = dm.champion_id

    if not run_ids:
        return pn.Column(
            widgets.section_header(
                "Projection Ensemble",
                "No benchmark runs found.",
            ),
            widgets.empty_placeholder("Run benchmarks first to populate this tab."),
        )

    # ---- Section 1: Filters ----

    # Detect available origin years from the data
    available_origins: list[str] = ["All"]
    if not curves.empty and "origin_year" in curves.columns:
        detected = sorted(curves["origin_year"].dropna().unique().astype(int))
        available_origins += [str(y) for y in detected]
    elif not state_metrics.empty and "origin_year" in state_metrics.columns:
        detected = sorted(state_metrics["origin_year"].dropna().unique().astype(int))
        available_origins += [str(y) for y in detected]
    else:
        # Fallback to standard origins
        available_origins += [str(y) for y in _ORIGIN_YEARS]

    origin_selector = pn.widgets.Select(
        name="Origin Year",
        options=available_origins,
        value="All",
        sizing_mode="stretch_width",
    )

    run_selector = pn.widgets.MultiSelect(
        name="Select Experiments / Runs",
        options=run_ids,
        value=run_ids,
        size=min(len(run_ids), 8),
        sizing_mode="stretch_width",
    )

    # ---- Section 2: Spaghetti Plot (reactive) ----
    spaghetti = pn.bind(
        _build_spaghetti_plot,
        origin_year=origin_selector,
        selected_runs=run_selector,
        curves=curves,
        state_metrics=state_metrics,
        champion_id=champion_id,
        all_run_ids=run_ids,
    )

    # ---- Section 3: Error Over Time (reactive) ----
    error_plot = pn.bind(
        _build_error_over_time,
        origin_year=origin_selector,
        selected_runs=run_selector,
        state_metrics=state_metrics,
        all_run_ids=run_ids,
    )

    # ---- Section 4: Uncertainty Fan Chart (reactive) ----
    uncertainty_section = pn.bind(
        _build_uncertainty_fan,
        selected_runs=run_selector,
        uncertainty=uncertainty,
    )

    # ---- Assemble ----
    return pn.Column(
        widgets.section_header(
            "Projection Ensemble",
            f"{len(run_ids)} runs | "
            + (f"Champion: {champion_id}" if champion_id else "No champion detected"),
        ),
        pn.Card(
            pn.Row(origin_selector, sizing_mode="stretch_width"),
            run_selector,
            title="Filters",
            collapsed=False,
            sizing_mode="stretch_width",
        ),
        pn.Card(
            spaghetti,
            title="State-Level Spaghetti Plot",
            collapsed=False,
            sizing_mode="stretch_width",
        ),
        pn.Card(
            error_plot,
            title="State Error Over Forecast Horizon",
            collapsed=False,
            sizing_mode="stretch_width",
        ),
        pn.Card(
            uncertainty_section,
            title="Uncertainty Bands",
            collapsed=True,
            sizing_mode="stretch_width",
        ),
        sizing_mode="stretch_width",
    )
