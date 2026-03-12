"""Scorecard comparison tab for the Observatory Panel dashboard.

Interactive side-by-side scorecard comparison with MAPE delta bar charts,
sentinel county analysis, Pareto frontier scatter, and county-group impact
tables.  All visualizations react to a shared run-selector widget.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pandas as pd
import panel as pn
import plotly.graph_objects as go

from cohort_projections.analysis.observatory.comparator import (
    METRIC_COLUMNS,
    SENTINEL_COLUMNS,
    _GROUP_TO_SCORECARD_COL,
)
from cohort_projections.analysis.observatory.dashboard import theme, widgets

if TYPE_CHECKING:
    from cohort_projections.analysis.observatory.dashboard.data_manager import (
        DashboardDataManager,
    )

logger = logging.getLogger(__name__)

# Metrics shown in the delta bar chart (scorecard col -> display label).
_DELTA_METRICS: list[tuple[str, str]] = [
    ("county_mape_overall", "Overall"),
    ("county_mape_urban_college", "College"),
    ("county_mape_rural", "Rural"),
    ("county_mape_bakken", "Bakken"),
]

# Sentinel county scorecard columns and their short labels.
_SENTINEL_LABELS: dict[str, str] = {
    "sentinel_cass_mape": "Cass",
    "sentinel_grand_forks_mape": "Grand Forks",
    "sentinel_ward_mape": "Ward",
    "sentinel_burleigh_mape": "Burleigh",
    "sentinel_williams_mape": "Williams",
    "sentinel_mckenzie_mape": "McKenzie",
}


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

def _build_scorecard_table(
    selected_runs: list[str],
    scorecards: pd.DataFrame,
    champion_id: str | None,
) -> pn.Column:
    """Filter scorecards to selected runs and render as a Tabulator."""
    if not selected_runs or scorecards.empty:
        return pn.Column(widgets.empty_placeholder("Select at least one run."))

    filtered = scorecards[scorecards["run_id"].isin(selected_runs)].copy()
    if filtered.empty:
        return pn.Column(widgets.empty_placeholder("No scorecard data for selected runs."))

    # Identify delta-style columns for highlighting
    delta_cols = [c for c in filtered.columns if c in METRIC_COLUMNS or c in SENTINEL_COLUMNS]

    return widgets.metric_table(
        filtered,
        title="Side-by-Side Scorecard",
        highlight_cols=delta_cols,
        page_size=0,
    )


def _build_delta_bar_chart(
    selected_runs: list[str],
    scorecards: pd.DataFrame,
    champion_id: str | None,
    all_run_ids: list[str],
) -> pn.pane.Plotly:
    """Grouped bar chart of MAPE deltas vs champion for key metrics."""
    if not selected_runs or scorecards.empty or champion_id is None:
        return pn.pane.Plotly(go.Figure(), sizing_mode="stretch_width")

    champion_rows = scorecards[scorecards["run_id"] == champion_id]
    if champion_rows.empty:
        return pn.pane.Plotly(go.Figure(), sizing_mode="stretch_width")
    champion = champion_rows.iloc[0]

    fig = go.Figure()

    for run_id in selected_runs:
        run_rows = scorecards[scorecards["run_id"] == run_id]
        if run_rows.empty:
            continue
        row = run_rows.iloc[0]
        color = _run_color(run_id, all_run_ids)

        labels = []
        vals = []
        for col, label in _DELTA_METRICS:
            if col in scorecards.columns:
                delta = float(row.get(col, 0)) - float(champion.get(col, 0))
                labels.append(label)
                vals.append(delta)

        # Color bars individually: green for improvement, red for regression
        bar_colors = [
            theme.GROWTH_GREEN if v <= 0 else theme.SDC_RED for v in vals
        ]

        fig.add_trace(
            go.Bar(
                name=run_id,
                x=labels,
                y=vals,
                marker_color=color,
                text=[f"{v:+.3f}" for v in vals],
                textposition="outside",
                textfont={"size": 10},
                hovertemplate=(
                    f"<b>{run_id}</b><br>"
                    "%{x}: %{y:+.4f} pp<extra></extra>"
                ),
            )
        )

    # Zero reference line
    fig.add_hline(y=0, line_width=1, line_color=theme.SDC_DARK_GRAY)

    fig.update_layout(**theme.get_plotly_layout_defaults())
    fig.update_layout(
        title="MAPE Delta vs Champion (negative = improvement)",
        yaxis_title="Delta (percentage points)",
        barmode="group",
        legend={"orientation": "h", "y": -0.15},
        height=420,
    )

    return pn.pane.Plotly(fig, sizing_mode="stretch_width")


def _build_sentinel_chart(
    selected_runs: list[str],
    scorecards: pd.DataFrame,
    all_run_ids: list[str],
) -> pn.pane.Plotly:
    """Grouped bar chart of absolute MAPE for the 6 sentinel counties."""
    if not selected_runs or scorecards.empty:
        return pn.pane.Plotly(go.Figure(), sizing_mode="stretch_width")

    fig = go.Figure()

    available_sentinels = [
        (col, label)
        for col, label in _SENTINEL_LABELS.items()
        if col in scorecards.columns
    ]
    if not available_sentinels:
        return pn.pane.Plotly(go.Figure(), sizing_mode="stretch_width")

    sentinel_labels = [label for _, label in available_sentinels]

    for run_id in selected_runs:
        run_rows = scorecards[scorecards["run_id"] == run_id]
        if run_rows.empty:
            continue
        row = run_rows.iloc[0]
        color = _run_color(run_id, all_run_ids)

        vals = [float(row.get(col, 0)) for col, _ in available_sentinels]

        fig.add_trace(
            go.Bar(
                name=run_id,
                x=sentinel_labels,
                y=vals,
                marker_color=color,
                text=[f"{v:.2f}" for v in vals],
                textposition="outside",
                textfont={"size": 10},
                hovertemplate=(
                    f"<b>{run_id}</b><br>"
                    "%{x}: %{y:.3f}%<extra></extra>"
                ),
            )
        )

    fig.update_layout(**theme.get_plotly_layout_defaults())
    fig.update_layout(
        title="Sentinel County MAPE (absolute)",
        yaxis_title="MAPE (%)",
        barmode="group",
        legend={"orientation": "h", "y": -0.15},
        height=420,
    )

    return pn.pane.Plotly(fig, sizing_mode="stretch_width")


def _build_pareto_scatter(
    selected_runs: list[str],
    x_metric: str,
    y_metric: str,
    dm: DashboardDataManager,
    all_run_ids: list[str],
) -> pn.pane.Plotly:
    """Scatter plot with Pareto frontier highlighted."""
    scorecards = dm.scorecards
    if not selected_runs or scorecards.empty:
        return pn.pane.Plotly(go.Figure(), sizing_mode="stretch_width")

    if x_metric not in scorecards.columns or y_metric not in scorecards.columns:
        return pn.pane.Plotly(go.Figure(), sizing_mode="stretch_width")

    filtered = scorecards[scorecards["run_id"].isin(selected_runs)].copy()
    if filtered.empty:
        return pn.pane.Plotly(go.Figure(), sizing_mode="stretch_width")

    # Compute Pareto frontier
    try:
        pareto_df = dm.comparator.pareto_frontier(x_metric, y_metric)
        pareto_run_ids = set(pareto_df["run_id"].tolist()) if "run_id" in pareto_df.columns else set()
    except (ValueError, KeyError):
        pareto_run_ids = set()

    fig = go.Figure()

    # Non-Pareto points
    non_pareto = filtered[~filtered["run_id"].isin(pareto_run_ids)]
    if not non_pareto.empty:
        fig.add_trace(
            go.Scatter(
                x=non_pareto[x_metric],
                y=non_pareto[y_metric],
                mode="markers+text",
                marker={
                    "size": 10,
                    "color": theme.SDC_MID_GRAY,
                    "line": {"width": 1, "color": theme.SDC_DARK_GRAY},
                },
                text=non_pareto["run_id"],
                textposition="top center",
                textfont={"size": 9},
                name="Other runs",
                hovertemplate=(
                    "<b>%{text}</b><br>"
                    f"{x_metric}: %{{x:.4f}}<br>"
                    f"{y_metric}: %{{y:.4f}}<extra></extra>"
                ),
            )
        )

    # Pareto-optimal points
    pareto_filtered = filtered[filtered["run_id"].isin(pareto_run_ids)]
    if not pareto_filtered.empty:
        pareto_colors = [
            _run_color(rid, all_run_ids) for rid in pareto_filtered["run_id"]
        ]
        fig.add_trace(
            go.Scatter(
                x=pareto_filtered[x_metric],
                y=pareto_filtered[y_metric],
                mode="markers+text",
                marker={
                    "size": 14,
                    "color": pareto_colors,
                    "symbol": "star",
                    "line": {"width": 1, "color": theme.SDC_NAVY},
                },
                text=pareto_filtered["run_id"],
                textposition="top center",
                textfont={"size": 10, "color": theme.SDC_NAVY},
                name="Pareto-optimal",
                hovertemplate=(
                    "<b>%{text}</b> (Pareto)<br>"
                    f"{x_metric}: %{{x:.4f}}<br>"
                    f"{y_metric}: %{{y:.4f}}<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        **theme.get_plotly_layout_defaults(),
        title="Pareto Frontier",
        xaxis_title=x_metric,
        yaxis_title=y_metric,
        height=480,
    )

    return pn.pane.Plotly(fig, sizing_mode="stretch_width")


def _build_county_group_table(dm: DashboardDataManager) -> pn.Column:
    """Table showing best variant per county group."""
    best = dm.comparator.best_variant_per_group()
    if not best:
        return pn.Column(widgets.empty_placeholder("No county group data available."))

    rows = []
    for group, info in best.items():
        rows.append({
            "County Group": group,
            "Best Run": info.get("run_id", "?"),
            "Config": info.get("config_id", "?"),
        })

    df = pd.DataFrame(rows)

    # Also add the county group impact deltas if available
    impact = dm.comparator.county_group_impact()
    if not impact.empty:
        return pn.Column(
            widgets.metric_table(
                df,
                title="Best Variant per County Group",
                page_size=0,
            ),
            widgets.metric_table(
                impact,
                title="County Group Impact (delta vs champion)",
                highlight_cols=[c for c in impact.columns if c.startswith("delta_")],
                page_size=0,
            ),
        )

    return widgets.metric_table(
        df,
        title="Best Variant per County Group",
        page_size=0,
    )


# ---------------------------------------------------------------------------
# Main tab builder
# ---------------------------------------------------------------------------

def build_scorecard_tab(dm: DashboardDataManager) -> pn.Column:
    """Build the interactive scorecard comparison tab.

    Parameters
    ----------
    dm:
        The dashboard data manager providing access to all observatory data.

    Returns
    -------
    pn.Column
        A Panel Column containing all scorecard sections wired to a shared
        run-selector widget.
    """
    run_ids = dm.run_ids
    scorecards = dm.scorecards
    champion_id = dm.champion_id

    if not run_ids:
        return pn.Column(
            widgets.section_header(
                "Scorecard Comparison",
                "No benchmark runs found.",
            ),
            widgets.empty_placeholder("Run benchmarks first to populate this tab."),
        )

    # ---- Section 1: Run Selector ----
    run_selector = pn.widgets.MultiSelect(
        name="Select Runs to Compare",
        options=run_ids,
        value=run_ids,
        size=min(len(run_ids), 8),
        sizing_mode="stretch_width",
    )

    # ---- Metric selectors for Pareto scatter ----
    metric_options = [c for c in METRIC_COLUMNS if c in scorecards.columns]
    # Add sentinel columns as options too
    metric_options += [c for c in SENTINEL_COLUMNS if c in scorecards.columns]

    pareto_x = pn.widgets.Select(
        name="X Axis Metric",
        options=metric_options,
        value="county_mape_overall" if "county_mape_overall" in metric_options else (metric_options[0] if metric_options else ""),
        sizing_mode="stretch_width",
    )
    pareto_y = pn.widgets.Select(
        name="Y Axis Metric",
        options=metric_options,
        value="state_ape_recent_short" if "state_ape_recent_short" in metric_options else (metric_options[1] if len(metric_options) > 1 else metric_options[0] if metric_options else ""),
        sizing_mode="stretch_width",
    )

    # ---- Section 2: Scorecard Table (reactive) ----
    scorecard_table = pn.bind(
        _build_scorecard_table,
        selected_runs=run_selector,
        scorecards=scorecards,
        champion_id=champion_id,
    )

    # ---- Section 3: MAPE Delta Bar Chart (reactive) ----
    delta_chart = pn.bind(
        _build_delta_bar_chart,
        selected_runs=run_selector,
        scorecards=scorecards,
        champion_id=champion_id,
        all_run_ids=run_ids,
    )

    # ---- Section 4: Sentinel County Comparison (reactive) ----
    sentinel_chart = pn.bind(
        _build_sentinel_chart,
        selected_runs=run_selector,
        scorecards=scorecards,
        all_run_ids=run_ids,
    )

    # ---- Section 5: Pareto Frontier (reactive) ----
    pareto_chart = pn.bind(
        _build_pareto_scatter,
        selected_runs=run_selector,
        x_metric=pareto_x,
        y_metric=pareto_y,
        dm=dm,
        all_run_ids=run_ids,
    )

    # ---- Section 6: County Group Impact (static) ----
    county_group_section = _build_county_group_table(dm)

    # ---- Assemble ----
    return pn.Column(
        widgets.section_header(
            "Scorecard Comparison",
            f"Comparing {len(run_ids)} benchmark runs"
            + (f" | Champion: {champion_id}" if champion_id else ""),
        ),
        pn.Card(
            run_selector,
            title="Run Selector",
            collapsed=False,
            sizing_mode="stretch_width",
        ),
        pn.Card(
            scorecard_table,
            title="Side-by-Side Scorecard",
            collapsed=False,
            sizing_mode="stretch_width",
        ),
        pn.Card(
            delta_chart,
            title="MAPE Delta vs Champion",
            collapsed=False,
            sizing_mode="stretch_width",
        ),
        pn.Card(
            sentinel_chart,
            title="Sentinel County Comparison",
            collapsed=False,
            sizing_mode="stretch_width",
        ),
        pn.Card(
            pn.Row(pareto_x, pareto_y, sizing_mode="stretch_width"),
            pareto_chart,
            title="Pareto Frontier",
            collapsed=False,
            sizing_mode="stretch_width",
        ),
        pn.Card(
            county_group_section,
            title="County Group Impact",
            collapsed=False,
            sizing_mode="stretch_width",
        ),
        sizing_mode="stretch_width",
    )
