"""Projection ensemble tab for the Observatory dashboard.

This tab emphasizes a focused comparison workflow: readable run presets,
latest-origin defaults, champion reference overlay, and a compact selected-run
summary alongside the charts.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pandas as pd
import panel as pn
import plotly.graph_objects as go

from cohort_projections.analysis.observatory.dashboard import theme, widgets
from cohort_projections.analysis.observatory.dashboard.data_manager import (
    RUN_SELECTION_PRESETS,
)
from cohort_projections.analysis.observatory.dashboard.theme import SDC_NAVY

if TYPE_CHECKING:
    from cohort_projections.analysis.observatory.dashboard.data_manager import (
        DashboardDataManager,
    )

logger = logging.getLogger(__name__)

_ORIGIN_YEARS = [2005, 2010, 2015, 2020]


def _run_color(run_id: str, run_ids: list[str]) -> str:
    """Return a consistent color for a bundle based on its position."""
    idx = run_ids.index(run_id) if run_id in run_ids else 0
    return theme.EXPERIMENT_COLORS[idx % len(theme.EXPERIMENT_COLORS)]


def _selected_method(dm: DashboardDataManager, run_id: str) -> str | None:
    """Return the method plotted for a selected benchmark bundle."""
    meta = dm.run_metadata[dm.run_metadata["run_id"] == run_id]
    if meta.empty:
        return None
    selected = meta.iloc[0].get("selected_method_id")
    if pd.notna(selected):
        return str(selected)
    reference = meta.iloc[0].get("reference_method_id")
    return str(reference) if pd.notna(reference) else None


def _filter_plot_frame(
    df: pd.DataFrame,
    *,
    run_id: str,
    method: str | None,
    origin_year: str,
    method_col: str,
) -> pd.DataFrame:
    """Filter a plot DataFrame to one bundle/method/origin combination."""
    filtered = df[df["run_id"] == run_id].copy()
    if filtered.empty:
        return filtered
    if method is not None and method_col in filtered.columns:
        method_filtered = filtered[filtered[method_col] == method]
        if not method_filtered.empty:
            filtered = method_filtered
    if origin_year != "All" and "origin_year" in filtered.columns:
        filtered = filtered[filtered["origin_year"] == int(origin_year)]
    return filtered


def _build_selected_run_summary(
    selected_runs: list[str],
    show_champion_reference: bool,
    dm: DashboardDataManager,
) -> pn.Column:
    """Compact ranked table of the bundles currently shown in the charts."""
    if dm.run_metadata.empty:
        return pn.Column(widgets.empty_placeholder("No run metadata available."))

    selected_ids = list(selected_runs)
    if show_champion_reference and dm.champion_id is not None:
        selected_ids = [dm.champion_id, *selected_ids]

    summary = dm.run_metadata[dm.run_metadata["run_id"].isin(selected_ids)].copy()
    if summary.empty:
        return pn.Column(widgets.empty_placeholder("Select at least one benchmark bundle."))

    summary["run"] = summary["run_id"].map(lambda run_id: dm.run_label(str(run_id), short=True))
    summary["status"] = summary["status_label"]
    summary["method"] = summary["selected_method_id"].fillna(summary["reference_method_id"])
    summary["config"] = summary["short_config"]
    summary["overall_mape"] = summary["selected_county_mape_overall"].round(3)
    summary["state_ape_short"] = summary["selected_state_ape_recent_short"].round(3)
    if dm.champion_id is not None:
        summary.loc[summary["run_id"] == dm.champion_id, "run"] = "Champion"

    display_df = summary[
        [
            col
            for col in [
                "run",
                "status",
                "method",
                "config",
                "overall_mape",
                "state_ape_short",
                "run_id",
            ]
            if col in summary.columns
        ]
    ].rename(columns={"run_id": "exact_run_id"})

    return widgets.metric_table(
        display_df,
        title="Selected Bundles",
        page_size=0,
        frozen_columns=["run"],
    )


def _add_actual_series(fig: go.Figure, state_metrics: pd.DataFrame) -> None:
    """Add the actual state population line when available."""
    if state_metrics.empty:
        return
    actual_cols = {"validation_year", "actual_state"}
    if not actual_cols.issubset(state_metrics.columns):
        return
    actuals = (
        state_metrics[["validation_year", "actual_state"]]
        .drop_duplicates()
        .sort_values("validation_year")
    )
    if actuals.empty:
        return
    fig.add_trace(
        go.Scatter(
            x=actuals["validation_year"],
            y=actuals["actual_state"],
            name="Actual (Census PEP)",
            mode="lines+markers",
            line={"color": "#111111", "width": 3},
            marker={"size": 5},
            hovertemplate="<b>Actual</b><br>Year: %{x}<br>Population: %{y:,.0f}<extra></extra>",
        )
    )


def _build_spaghetti_plot(
    origin_year: str,
    selected_runs: list[str],
    show_champion_reference: bool,
    emphasis_mode: str,
    curves: pd.DataFrame,
    state_metrics: pd.DataFrame,
    dm: DashboardDataManager,
) -> pn.pane.Plotly:
    """Projection comparison with a champion reference overlay."""
    if curves.empty:
        return pn.pane.Plotly(go.Figure(), sizing_mode="stretch_width")

    fig = go.Figure()
    _add_actual_series(fig, state_metrics)

    year_col = "year" if "year" in curves.columns else "validation_year"
    proj_col = next(
        (candidate for candidate in ("projected_state", "projected_population", "population") if candidate in curves.columns),
        None,
    )
    if proj_col is None or year_col not in curves.columns:
        return pn.pane.Plotly(go.Figure(), sizing_mode="stretch_width")

    if show_champion_reference and dm.champion_id is not None and dm.champion_method_id is not None:
        champion_curves = _filter_plot_frame(
            curves,
            run_id=dm.champion_id,
            method=dm.champion_method_id,
            origin_year=origin_year,
            method_col="method",
        )
        for idx, (_, origin_df) in enumerate(
            champion_curves.sort_values(year_col).groupby("origin_year" if "origin_year" in champion_curves.columns else year_col)
        ):
            fig.add_trace(
                go.Scatter(
                    x=origin_df[year_col],
                    y=origin_df[proj_col],
                    name="Champion",
                    mode="lines",
                    line={"color": SDC_NAVY, "width": 3.4, "dash": "solid"},
                    opacity=0.95,
                    legendgroup="champion",
                    showlegend=idx == 0,
                    hovertemplate="<b>Champion</b><br>Year: %{x}<br>Projected: %{y:,.0f}<extra></extra>",
                )
            )

    if not selected_runs:
        fig.update_layout(**theme.get_plotly_layout_defaults())
        fig.update_layout(title="State Population Projection Ensemble", height=520)
        return pn.pane.Plotly(fig, sizing_mode="stretch_width")

    for run_id in selected_runs:
        method = _selected_method(dm, run_id)
        run_curves = _filter_plot_frame(
            curves,
            run_id=run_id,
            method=method,
            origin_year=origin_year,
            method_col="method",
        )
        if run_curves.empty:
            continue

        run_label = dm.run_label(run_id, short=True)
        color = _run_color(run_id, dm.ordered_run_ids)
        line_width = 2.2 if emphasis_mode == "Balanced" else 1.9
        opacity = 0.84 if emphasis_mode == "Balanced" else 0.62

        group_col = "origin_year" if "origin_year" in run_curves.columns else year_col
        for idx, (_, origin_df) in enumerate(
            run_curves.sort_values(year_col).groupby(group_col)
        ):
            fig.add_trace(
                go.Scatter(
                    x=origin_df[year_col],
                    y=origin_df[proj_col],
                    name=run_label,
                    mode="lines",
                    line={"color": color, "width": line_width},
                    opacity=opacity,
                    legendgroup=run_id,
                    showlegend=idx == 0,
                    hovertemplate=f"<b>{run_label}</b><br>Year: %{{x}}<br>Projected: %{{y:,.0f}}<extra></extra>",
                )
            )

    title_suffix = f"Origin {origin_year}" if origin_year != "All" else "All Origins"
    fig.update_layout(**theme.get_plotly_layout_defaults())
    fig.update_layout(
        title=f"State Population Projection Ensemble ({title_suffix})",
        xaxis_title="Year",
        yaxis_title="Population",
        yaxis_tickformat=",",
        legend={"orientation": "h", "y": -0.18, "x": 0},
        height=540,
    )
    return pn.pane.Plotly(fig, sizing_mode="stretch_width")


def _build_error_over_time(
    origin_year: str,
    selected_runs: list[str],
    show_champion_reference: bool,
    state_metrics: pd.DataFrame,
    dm: DashboardDataManager,
) -> pn.pane.Plotly:
    """Line chart of state error vs forecast horizon."""
    if state_metrics.empty:
        return pn.pane.Plotly(go.Figure(), sizing_mode="stretch_width")

    horizon_col = next(
        (candidate for candidate in ("horizon", "forecast_horizon", "horizon_years") if candidate in state_metrics.columns),
        None,
    )
    error_col = next(
        (candidate for candidate in ("pct_error", "state_pct_error", "ape") if candidate in state_metrics.columns),
        None,
    )
    if horizon_col is None or error_col is None:
        return pn.pane.Plotly(go.Figure(), sizing_mode="stretch_width")

    fig = go.Figure()

    def _add_run(run_id: str, method: str | None, label: str, width: float, opacity: float) -> None:
        run_metrics = _filter_plot_frame(
            state_metrics,
            run_id=run_id,
            method=method,
            origin_year=origin_year,
            method_col="method",
        )
        if run_metrics.empty:
            return
        if origin_year == "All" and "origin_year" in run_metrics.columns:
            run_metrics = (
                run_metrics.groupby(horizon_col)[error_col]
                .mean()
                .reset_index()
                .sort_values(horizon_col)
            )
        else:
            run_metrics = run_metrics[[horizon_col, error_col]].sort_values(horizon_col)
        fig.add_trace(
            go.Scatter(
                x=run_metrics[horizon_col],
                y=run_metrics[error_col],
                name=label,
                mode="lines+markers",
                line={"color": _run_color(run_id, dm.ordered_run_ids), "width": width},
                marker={"size": 5},
                opacity=opacity,
                hovertemplate=f"<b>{label}</b><br>Horizon: %{{x}} yr<br>Error: %{{y:.2f}}%<extra></extra>",
            )
        )

    if show_champion_reference and dm.champion_id is not None and dm.champion_method_id is not None:
        champion_metrics = _filter_plot_frame(
            state_metrics,
            run_id=dm.champion_id,
            method=dm.champion_method_id,
            origin_year=origin_year,
            method_col="method",
        )
        if not champion_metrics.empty:
            if origin_year == "All" and "origin_year" in champion_metrics.columns:
                champion_metrics = (
                    champion_metrics.groupby(horizon_col)[error_col]
                    .mean()
                    .reset_index()
                    .sort_values(horizon_col)
                )
            else:
                champion_metrics = champion_metrics[[horizon_col, error_col]].sort_values(horizon_col)
            fig.add_trace(
                go.Scatter(
                    x=champion_metrics[horizon_col],
                    y=champion_metrics[error_col],
                    name="Champion",
                    mode="lines+markers",
                    line={"color": SDC_NAVY, "width": 3.2},
                    marker={"size": 5},
                    hovertemplate="<b>Champion</b><br>Horizon: %{x} yr<br>Error: %{y:.2f}%<extra></extra>",
                )
            )

    for run_id in selected_runs:
        method = _selected_method(dm, run_id)
        _add_run(
            run_id,
            method,
            dm.run_label(run_id, short=True),
            width=2.0,
            opacity=0.85,
        )

    fig.add_hline(y=0, line_width=1, line_color=theme.SDC_DARK_GRAY)
    title_suffix = f"Origin {origin_year}" if origin_year != "All" else "Mean Across Origins"
    fig.update_layout(**theme.get_plotly_layout_defaults())
    fig.update_layout(
        title=f"State Error Over Forecast Horizon ({title_suffix})",
        xaxis_title="Forecast Horizon (years)",
        yaxis_title="Percentage Error (%)",
        legend={"orientation": "h", "y": -0.18},
        height=420,
    )
    return pn.pane.Plotly(fig, sizing_mode="stretch_width")


def _build_uncertainty_fan(
    selected_runs: list[str],
    show_champion_reference: bool,
    uncertainty: pd.DataFrame,
    dm: DashboardDataManager,
) -> pn.Column:
    """Fan chart for the first selected bundle (or champion reference)."""
    if uncertainty.empty:
        return pn.Column(
            widgets.empty_placeholder(
                "No uncertainty data available. Run benchmarks with uncertainty estimation enabled."
            )
        )

    if not selected_runs and not show_champion_reference:
        return pn.Column(widgets.empty_placeholder("Select at least one benchmark bundle."))

    run_id = selected_runs[0] if selected_runs else dm.champion_id
    if run_id is None:
        return pn.Column(widgets.empty_placeholder("Select at least one benchmark bundle."))

    method = _selected_method(dm, run_id)
    run_data = _filter_plot_frame(
        uncertainty,
        run_id=run_id,
        method=method,
        origin_year="All",
        method_col="method",
    )
    if run_data.empty:
        return pn.Column(widgets.empty_placeholder("No uncertainty data for the current selection."))

    year_col = next((candidate for candidate in ("year", "validation_year", "horizon") if candidate in run_data.columns), None)
    if year_col is None:
        return pn.Column(widgets.empty_placeholder("Uncertainty data is missing a year/horizon column."))
    run_data = run_data.sort_values(year_col)

    fig = go.Figure()
    if {"p5", "p95"}.issubset(run_data.columns):
        fig.add_trace(go.Scatter(x=run_data[year_col], y=run_data["p95"], mode="lines", line={"width": 0}, showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=run_data[year_col], y=run_data["p5"], mode="lines", line={"width": 0}, fill="tonexty", fillcolor="rgba(5,99,193,0.15)", name="90% band"))
    if {"p25", "p75"}.issubset(run_data.columns):
        fig.add_trace(go.Scatter(x=run_data[year_col], y=run_data["p75"], mode="lines", line={"width": 0}, showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=run_data[year_col], y=run_data["p25"], mode="lines", line={"width": 0}, fill="tonexty", fillcolor="rgba(5,99,193,0.32)", name="50% band"))
    if "p50" in run_data.columns:
        fig.add_trace(
            go.Scatter(
                x=run_data[year_col],
                y=run_data["p50"],
                mode="lines",
                line={"color": SDC_NAVY if run_id == dm.champion_id else theme.SDC_BLUE, "width": 2.6},
                name="Median",
            )
        )

    fig.update_layout(**theme.get_plotly_layout_defaults())
    fig.update_layout(
        title=f"Uncertainty Bands ({dm.run_label(str(run_id), short=True)})",
        xaxis_title="Forecast Horizon",
        yaxis_title="Population / Error Distribution",
        height=430,
    )
    return pn.Column(pn.pane.Plotly(fig, sizing_mode="stretch_width"))


def build_projection_ensemble(dm: DashboardDataManager) -> pn.Column:
    """Build the projection ensemble tab."""
    if not dm.run_ids:
        return pn.Column(
            widgets.section_header("Projection Ensemble", "No benchmark runs found."),
            widgets.empty_placeholder("Run benchmarks first to populate this tab."),
        )

    available_origins: list[str] = ["All"]
    if not dm.projection_curves.empty and "origin_year" in dm.projection_curves.columns:
        detected = sorted(dm.projection_curves["origin_year"].dropna().unique().astype(int))
        available_origins += [str(year) for year in detected]
    elif not dm.state_metrics.empty and "origin_year" in dm.state_metrics.columns:
        detected = sorted(dm.state_metrics["origin_year"].dropna().unique().astype(int))
        available_origins += [str(year) for year in detected]
    else:
        available_origins += [str(year) for year in _ORIGIN_YEARS]

    default_origin = available_origins[-1] if len(available_origins) > 1 else available_origins[0]
    default_runs = dm.preset_run_ids(RUN_SELECTION_PRESETS[0])

    preset_selector = pn.widgets.Select(
        name="Comparison Preset",
        options=list(RUN_SELECTION_PRESETS),
        value=RUN_SELECTION_PRESETS[0],
        width=240,
    )
    origin_selector = pn.widgets.Select(
        name="Origin Year",
        options=available_origins,
        value=default_origin,
        width=180,
    )
    emphasis_selector = pn.widgets.Select(
        name="Line Emphasis",
        options=["Champion focus", "Balanced"],
        value="Champion focus",
        width=180,
    )
    show_champion_reference = pn.widgets.Checkbox(
        name="Include champion reference",
        value=True,
    )
    run_selector = pn.widgets.MultiChoice(
        name="Runs",
        options=dm.run_option_map(),
        value=default_runs,
        placeholder="Search benchmark bundles...",
        delete_button=True,
        search_option_limit=12,
        max_items=max(6, len(dm.run_ids)),
        sizing_mode="stretch_width",
    )

    def _apply_preset(event: object) -> None:
        del event
        run_selector.value = dm.preset_run_ids(str(preset_selector.value))

    preset_selector.param.watch(_apply_preset, "value")

    selected_summary = pn.panel(
        pn.bind(
            _build_selected_run_summary,
            selected_runs=run_selector,
            show_champion_reference=show_champion_reference,
            dm=dm,
        ),
        loading_indicator=True,
    )
    spaghetti = pn.panel(
        pn.bind(
            _build_spaghetti_plot,
            origin_year=origin_selector,
            selected_runs=run_selector,
            show_champion_reference=show_champion_reference,
            emphasis_mode=emphasis_selector,
            curves=dm.projection_curves,
            state_metrics=dm.state_metrics,
            dm=dm,
        ),
        loading_indicator=True,
    )
    error_plot = pn.panel(
        pn.bind(
            _build_error_over_time,
            origin_year=origin_selector,
            selected_runs=run_selector,
            show_champion_reference=show_champion_reference,
            state_metrics=dm.state_metrics,
            dm=dm,
        ),
        loading_indicator=True,
    )
    uncertainty_section = pn.panel(
        pn.bind(
            _build_uncertainty_fan,
            selected_runs=run_selector,
            show_champion_reference=show_champion_reference,
            uncertainty=dm.uncertainty_summary,
            dm=dm,
        ),
        loading_indicator=True,
    )

    return pn.Column(
        widgets.section_header(
            "Projection Ensemble",
            "Focused variant curves with champion reference and latest-origin defaults.",
        ),
        pn.Card(
            pn.pane.HTML(
                '<div class="filters-help">The default preset loads the strongest challenger bundles. Keep the champion reference on to compare against the current production baseline.</div>',
                sizing_mode="stretch_width",
                stylesheets=[theme.DASHBOARD_CSS],
            ),
            pn.FlexBox(
                preset_selector,
                origin_selector,
                emphasis_selector,
                show_champion_reference,
                flex_wrap="wrap",
                sizing_mode="stretch_width",
                styles={"gap": "10px"},
            ),
            run_selector,
            title="Filters",
            collapsed=False,
            sizing_mode="stretch_width",
        ),
        pn.Card(
            selected_summary,
            title="Selected Bundles",
            collapsed=False,
            sizing_mode="stretch_width",
        ),
        pn.Card(
            spaghetti,
            title="State-Level Projection Curves",
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
