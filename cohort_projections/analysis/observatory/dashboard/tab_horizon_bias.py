"""Horizon Bias tab — horizon analysis, bias maps, and county-level deep dives.

Provides multi-section analysis of projection accuracy by forecast horizon,
county-level heatmaps, bias direction analysis, county report cards, and
outlier detection.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
import panel as pn
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from cohort_projections.analysis.observatory.dashboard.data_manager import (
    DashboardDataManager,
)
from cohort_projections.analysis.observatory.dashboard.theme import (
    CATEGORY_COLORS,
    EXPERIMENT_COLORS,
    SDC_NAVY,
    SDC_RED,
    get_plotly_layout_defaults,
)
from cohort_projections.analysis.observatory.dashboard.widgets import (
    empty_placeholder,
    metric_table,
    section_header,
)

logger = logging.getLogger(__name__)

# Category constants matching the comparator module.
_ALL_CATEGORIES = ["All", "Rural", "Bakken", "Urban/College", "Reservation"]
_CATEGORY_ORDER = ["Bakken", "Urban/College", "Reservation", "Rural"]

# Grade color mapping for county report cards.
_GRADE_COLORS: dict[str, str] = {
    "A": "#00B050",
    "B": "#92D050",
    "C": "#FFC000",
    "D": "#ED7D31",
    "F": "#C00000",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run_options(dm: DashboardDataManager) -> dict[str, str]:
    """Return selector options with readable labels."""
    return {"All Runs": "All Runs", **dm.run_option_map()}


def _filter_by_category(
    df: pd.DataFrame,
    category: str,
    category_col: str = "category",
) -> pd.DataFrame:
    """Filter DataFrame by category, returning unfiltered if 'All'."""
    if category == "All" or category_col not in df.columns:
        return df
    return df[df[category_col] == category].copy()


def _color_for_run(idx: int) -> str:
    """Return a color from the experiment palette by index."""
    return EXPERIMENT_COLORS[idx % len(EXPERIMENT_COLORS)]


# ---------------------------------------------------------------------------
# Section 2: Horizon Profile (dual-panel Plotly chart)
# ---------------------------------------------------------------------------


def _build_horizon_profile(
    dm: DashboardDataManager,
    selected_run: str,
    category: str,
) -> pn.pane.Plotly:
    """Build dual-panel horizon profile: MAPE (left) and MPE/bias (right).

    Parameters
    ----------
    dm:
        Data manager providing ``annual_horizon_summary``.
    selected_run:
        A specific run ID or ``"All Runs"``.
    category:
        Category filter.

    Returns
    -------
    pn.pane.Plotly
        Panel pane wrapping a Plotly subplot figure.
    """
    ahs = dm.annual_horizon_summary
    if ahs.empty:
        return empty_placeholder("No annual horizon summary data available.")

    ahs = _filter_by_category(ahs, category)
    if ahs.empty:
        return empty_placeholder(
            f"No horizon data for category '{category}'."
        )

    # Determine MAPE and MPE columns
    mape_col = (
        "mean_county_mape"
        if "mean_county_mape" in ahs.columns
        else "county_mape"
        if "county_mape" in ahs.columns
        else None
    )
    mpe_col = (
        "mean_county_mpe"
        if "mean_county_mpe" in ahs.columns
        else "mean_state_mpe"
        if "mean_state_mpe" in ahs.columns
        else "county_mpe"
        if "county_mpe" in ahs.columns
        else None
    )
    horizon_col = "horizon" if "horizon" in ahs.columns else None

    if horizon_col is None or mape_col is None:
        return empty_placeholder(
            "Horizon summary missing required columns (horizon, mape)."
        )

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("MAPE vs Horizon", "MPE (Bias) vs Horizon"),
        horizontal_spacing=0.10,
    )
    layout_defaults = get_plotly_layout_defaults()

    if selected_run == "All Runs":
        for idx, run_id in enumerate(dm.run_ids):
            run_data = ahs[ahs["run_id"] == run_id].sort_values(horizon_col)
            if run_data.empty:
                continue
            color = _color_for_run(idx)
            fig.add_trace(
                go.Scatter(
                    x=run_data[horizon_col],
                    y=run_data[mape_col],
                    mode="lines+markers",
                    name=run_id,
                    line={"color": color},
                    marker={"size": 4},
                    legendgroup=run_id,
                    showlegend=True,
                ),
                row=1,
                col=1,
            )
            if mpe_col is not None and mpe_col in run_data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=run_data[horizon_col],
                        y=run_data[mpe_col],
                        mode="lines+markers",
                        name=run_id,
                        line={"color": color},
                        marker={"size": 4},
                        legendgroup=run_id,
                        showlegend=False,
                    ),
                    row=1,
                    col=2,
                )
    else:
        run_data = ahs[ahs["run_id"] == selected_run].copy()
        if run_data.empty:
            return empty_placeholder(
                f"No horizon data for run '{selected_run}'."
            )
        # Group by method if available
        group_col = (
            "method"
            if "method" in run_data.columns
            else "method_id"
            if "method_id" in run_data.columns
            else None
        )
        if group_col is not None:
            for idx, (method, grp) in enumerate(
                run_data.sort_values(horizon_col).groupby(group_col)
            ):
                color = _color_for_run(idx)
                fig.add_trace(
                    go.Scatter(
                        x=grp[horizon_col],
                        y=grp[mape_col],
                        mode="lines+markers",
                        name=str(method),
                        line={"color": color},
                        marker={"size": 4},
                        legendgroup=str(method),
                    ),
                    row=1,
                    col=1,
                )
                if mpe_col is not None and mpe_col in grp.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=grp[horizon_col],
                            y=grp[mpe_col],
                            mode="lines+markers",
                            name=str(method),
                            line={"color": color},
                            marker={"size": 4},
                            legendgroup=str(method),
                            showlegend=False,
                        ),
                        row=1,
                        col=2,
                    )
        else:
            run_data = run_data.sort_values(horizon_col)
            fig.add_trace(
                go.Scatter(
                    x=run_data[horizon_col],
                    y=run_data[mape_col],
                    mode="lines+markers",
                    name=selected_run,
                ),
                row=1,
                col=1,
            )
            if mpe_col is not None and mpe_col in run_data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=run_data[horizon_col],
                        y=run_data[mpe_col],
                        mode="lines+markers",
                        name=selected_run,
                        showlegend=False,
                    ),
                    row=1,
                    col=2,
                )

    # Add zero-bias reference line on the right panel
    fig.add_hline(
        y=0, line_dash="dash", line_color="#A0A0A0", line_width=1, row=1, col=2
    )

    fig.update_xaxes(title_text="Horizon (years)", row=1, col=1)
    fig.update_xaxes(title_text="Horizon (years)", row=1, col=2)
    fig.update_yaxes(title_text="MAPE (%)", row=1, col=1)
    fig.update_yaxes(title_text="MPE (%)", row=1, col=2)

    fig.update_layout(
        **{
            k: v
            for k, v in layout_defaults.items()
            if k not in ("xaxis", "yaxis", "legend")
        },
        height=450,
        title_text="Horizon Accuracy Profile",
        legend={"orientation": "h", "yanchor": "bottom", "y": -0.25},
    )

    return pn.pane.Plotly(fig, sizing_mode="stretch_width")


# ---------------------------------------------------------------------------
# Section 3: County Heatmap
# ---------------------------------------------------------------------------


def _build_county_heatmap(
    dm: DashboardDataManager,
    category: str,
) -> pn.pane.Plotly:
    """Build county x run MAPE heatmap.

    Counties are grouped by category on the Y-axis; runs appear on the X-axis.
    Color encodes mean absolute percentage error.

    Parameters
    ----------
    dm:
        Data manager providing ``county_metrics``.
    category:
        Category filter (``"All"`` for all categories).

    Returns
    -------
    pn.pane.Plotly
        Panel pane wrapping a Plotly Heatmap figure.
    """
    cm = dm.county_metrics
    if cm.empty:
        return empty_placeholder("No county metrics data available.")

    cm = _filter_by_category(cm, category)
    if cm.empty:
        return empty_placeholder(
            f"No county data for category '{category}'."
        )

    # Determine the error column
    error_col = (
        "pct_error"
        if "pct_error" in cm.columns
        else "mape"
        if "mape" in cm.columns
        else None
    )
    county_name_col = (
        "county_name"
        if "county_name" in cm.columns
        else "county"
        if "county" in cm.columns
        else None
    )
    category_col = "category" if "category" in cm.columns else None

    if error_col is None or county_name_col is None:
        return empty_placeholder(
            "County metrics missing required columns (pct_error/mape, county_name)."
        )

    # Aggregate: mean |pct_error| per county per run
    agg = (
        cm.groupby([county_name_col, "run_id"]
                    + ([category_col] if category_col else []))[error_col]
        .apply(lambda x: x.abs().mean())
        .reset_index()
        .rename(columns={error_col: "mape"})
    )

    if agg.empty:
        return empty_placeholder("No aggregated county data available.")

    # Pivot to county x run matrix
    index_cols = (
        [category_col, county_name_col] if category_col else [county_name_col]
    )
    pivot = agg.pivot_table(
        index=index_cols,
        columns="run_id",
        values="mape",
    )

    # Sort by category order then county name
    pivot = pivot.reset_index()
    if category_col and category_col in pivot.columns:
        cat_order_map = {c: i for i, c in enumerate(_CATEGORY_ORDER)}
        pivot["_cat_order"] = pivot[category_col].map(cat_order_map).fillna(99)
        pivot = pivot.sort_values(["_cat_order", county_name_col]).drop(
            columns=["_cat_order"]
        )
    else:
        pivot = pivot.sort_values(county_name_col)

    # Build y-labels and z-matrix
    if category_col and category_col in pivot.columns:
        y_labels = [
            f"{row[county_name_col]} ({str(row[category_col])[:3]})"
            for _, row in pivot.iterrows()
        ]
    else:
        y_labels = pivot[county_name_col].tolist()

    run_cols = sorted(
        [c for c in pivot.columns if c in dm.run_ids]
    )
    if not run_cols:
        return empty_placeholder("No run columns found in pivoted data.")

    z = pivot[run_cols].values

    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=run_cols,
            y=y_labels,
            colorscale=[
                [0.0, "#059669"],
                [0.3, "#ecfdf5"],
                [0.5, "#FFFFFF"],
                [0.7, "#fef2f2"],
                [1.0, "#dc2626"],
            ],
            zmin=0,
            zmax=max(
                30,
                float(np.nanmax(z)) if z.size > 0 else 30,
            ),
            text=[
                [f"{v:.1f}" if not pd.isna(v) else "" for v in row]
                for row in z
            ],
            texttemplate="%{text}",
            textfont={"size": 8},
            hovertemplate=(
                "County: %{y}<br>"
                "Run: %{x}<br>"
                "MAPE: %{z:.2f}%<extra></extra>"
            ),
            colorbar={"title": "MAPE (%)"},
        )
    )

    # Add category separator lines
    if category_col and category_col in pivot.columns:
        cumulative = 0
        for cat in _CATEGORY_ORDER:
            cat_count = len(pivot[pivot[category_col] == cat])
            if cat_count > 0:
                cumulative += cat_count
                if cumulative < len(pivot):
                    fig.add_hline(
                        y=cumulative - 0.5,
                        line_width=2,
                        line_color=SDC_NAVY,
                    )

    layout_defaults = get_plotly_layout_defaults()
    fig.update_layout(
        **{
            k: v
            for k, v in layout_defaults.items()
            if k not in ("xaxis", "yaxis")
        },
        title="County MAPE by Run",
        height=max(600, len(y_labels) * 16 + 100),
        yaxis={"autorange": "reversed", "dtick": 1, "tickfont": {"size": 9}},
        xaxis={"side": "top", "tickfont": {"size": 11}},
    )

    return pn.pane.Plotly(fig, sizing_mode="stretch_width")


# ---------------------------------------------------------------------------
# Section 4: Bias Analysis
# ---------------------------------------------------------------------------


def _build_bias_heatmap(
    dm: DashboardDataManager,
    selected_run: str,
) -> pn.pane.Plotly:
    """Build bias heatmap: category x horizon, colored by mean signed error.

    Blue = under-projection, Red = over-projection, White = unbiased.

    Parameters
    ----------
    dm:
        Data manager providing ``bias_analysis``.
    selected_run:
        A specific run ID or ``"All Runs"`` (faceted).

    Returns
    -------
    pn.pane.Plotly
        Panel pane with a bias heatmap.
    """
    ba = dm.bias_analysis
    if ba.empty:
        return empty_placeholder("No bias analysis data available.")

    # Determine column names
    category_col = "category" if "category" in ba.columns else None
    horizon_col = (
        "horizon_bucket"
        if "horizon_bucket" in ba.columns
        else "horizon"
        if "horizon" in ba.columns
        else None
    )
    error_col = (
        "mean_signed_pct_error"
        if "mean_signed_pct_error" in ba.columns
        else "mean_mpe"
        if "mean_mpe" in ba.columns
        else "mpe"
        if "mpe" in ba.columns
        else None
    )

    if category_col is None or horizon_col is None or error_col is None:
        return empty_placeholder(
            "Bias analysis missing required columns "
            "(category, horizon, mean_signed_pct_error)."
        )

    method_col = "method" if "method" in ba.columns else None

    if selected_run != "All Runs":
        ba = ba[ba["run_id"] == selected_run]
        if ba.empty:
            return empty_placeholder(
                f"No bias data for run '{selected_run}'."
            )

    runs_to_plot = (
        sorted(ba[method_col].dropna().astype(str).unique())
        if selected_run != "All Runs" and method_col is not None
        else [selected_run] if selected_run != "All Runs"
        else sorted(ba["run_id"].unique())
    )
    n_runs = len(runs_to_plot)

    if n_runs == 0:
        return empty_placeholder("No runs to display in bias heatmap.")

    fig = make_subplots(
        rows=1,
        cols=n_runs,
        subplot_titles=[str(r) for r in runs_to_plot],
        horizontal_spacing=0.05,
    )

    for col_idx, run_id in enumerate(runs_to_plot, start=1):
        if selected_run != "All Runs" and method_col is not None:
            run_ba = ba[ba[method_col].astype(str) == str(run_id)]
        else:
            run_ba = ba[ba["run_id"] == run_id]
        if run_ba.empty:
            continue

        pivot = run_ba.pivot_table(
            index=category_col,
            columns=horizon_col,
            values=error_col,
            aggfunc="mean",
        )
        if pivot.empty:
            continue

        # Sort categories
        ordered_cats = [c for c in _CATEGORY_ORDER if c in pivot.index]
        remaining = [c for c in pivot.index if c not in ordered_cats]
        pivot = pivot.reindex(ordered_cats + remaining)

        z_max = max(abs(pivot.values.min()), abs(pivot.values.max()), 5)

        fig.add_trace(
            go.Heatmap(
                z=pivot.values,
                x=[str(c) for c in pivot.columns],
                y=list(pivot.index),
                colorscale=[
                    [0.0, "#2166AC"],   # Blue — under-projection
                    [0.5, "#FFFFFF"],   # White — unbiased
                    [1.0, "#B2182B"],   # Red — over-projection
                ],
                zmin=-z_max,
                zmax=z_max,
                text=[
                    [f"{v:+.1f}" if not pd.isna(v) else "" for v in row]
                    for row in pivot.values
                ],
                texttemplate="%{text}",
                textfont={"size": 9},
                hovertemplate=(
                    "Category: %{y}<br>"
                    "Horizon: %{x}<br>"
                    "Bias: %{z:+.2f}%<extra></extra>"
                ),
                colorbar=(
                    {"title": "MPE (%)", "len": 0.8}
                    if col_idx == n_runs
                    else None
                ),
                showscale=col_idx == n_runs,
            ),
            row=1,
            col=col_idx,
        )

    layout_defaults = get_plotly_layout_defaults()
    fig.update_layout(
        **{
            k: v
            for k, v in layout_defaults.items()
            if k not in ("xaxis", "yaxis")
        },
        title_text="Bias Direction by Category and Horizon",
        height=350 + 20 * n_runs,
    )

    return pn.pane.Plotly(fig, sizing_mode="stretch_width")


def _build_top_counties_table(
    dm: DashboardDataManager,
    selected_run: str,
    category: str,
) -> pn.Column:
    """Build a ranked county table to pair with the dense heatmaps."""
    crc = dm.county_report_cards
    if crc.empty:
        return pn.Column(empty_placeholder("No county report card data available."))

    if selected_run != "All Runs":
        crc = crc[crc["run_id"] == selected_run]
    crc = _filter_by_category(crc, category)
    if crc.empty:
        return pn.Column(empty_placeholder("No counties match the current filters."))

    sort_col = "mape" if "mape" in crc.columns else "worst_case_abs_error"
    if sort_col not in crc.columns:
        return pn.Column(empty_placeholder("County report cards are missing a sortable error column."))

    display_cols = [
        col
        for col in [
            "county_name",
            "method",
            "category",
            "grade",
            "mape",
            "worst_case_error",
            "bias_direction",
            "run_id",
        ]
        if col in crc.columns
    ]
    display_df = (
        crc.sort_values(sort_col, ascending=False, na_position="last")
        .head(12)[display_cols]
        .copy()
    )
    if "run_id" in display_df.columns:
        display_df["run"] = display_df["run_id"].map(lambda run_id: dm.run_label(str(run_id), short=True))
        display_df = display_df.drop(columns=["run_id"])
        ordered_cols = ["county_name", "method", "category", "grade", "mape", "worst_case_error", "bias_direction", "run"]
        display_df = display_df[[col for col in ordered_cols if col in display_df.columns]]

    return metric_table(
        display_df,
        title="Highest-MAPE Counties",
        page_size=0,
        frozen_columns=["county_name"],
    )


# ---------------------------------------------------------------------------
# Section 5: County Report Cards
# ---------------------------------------------------------------------------


def _build_county_report_cards(
    dm: DashboardDataManager,
    selected_run: str,
    category: str,
) -> pn.Column:
    """Build a sortable, filterable county report card table.

    Parameters
    ----------
    dm:
        Data manager providing ``county_report_cards``.
    selected_run:
        Run filter. ``"All Runs"`` shows all.
    category:
        Category filter.

    Returns
    -------
    pn.Column
        Tabulator table of county grades and metrics.
    """
    crc = dm.county_report_cards
    if crc.empty:
        return pn.Column(
            empty_placeholder("No county report card data available.")
        )

    if selected_run != "All Runs":
        crc = crc[crc["run_id"] == selected_run]
    crc = _filter_by_category(crc, category)

    if crc.empty:
        return pn.Column(
            empty_placeholder("No report cards match the current filters.")
        )

    # Select display columns (flexible based on what exists)
    desired_cols = [
        "run_id",
        "county_name",
        "category",
        "grade",
        "mape",
        "mean_signed_error",
        "bias_direction",
        "worst_case_error",
    ]
    display_cols = [c for c in desired_cols if c in crc.columns]
    display_df = crc[display_cols].copy()

    # Build grade color formatters for Tabulator
    grade_formatters: dict[str, Any] = {}
    if "grade" in display_df.columns:
        grade_formatters["grade"] = {
            "type": "color",
            "color": "#FFFFFF",
            "backgroundColor": {
                "A": _GRADE_COLORS["A"],
                "B": _GRADE_COLORS["B"],
                "C": _GRADE_COLORS["C"],
                "D": _GRADE_COLORS["D"],
                "F": _GRADE_COLORS["F"],
            },
        }

    highlight_cols = [
        c for c in ["mape", "mean_signed_error", "worst_case_error"]
        if c in display_df.columns
    ]

    return metric_table(
        display_df,
        title="County Report Cards",
        highlight_cols=highlight_cols,
        page_size=20,
    )


# ---------------------------------------------------------------------------
# Section 6: Outlier Flags
# ---------------------------------------------------------------------------


def _build_outlier_scatter(
    dm: DashboardDataManager,
    selected_run: str,
    category: str,
) -> pn.pane.Plotly:
    """Build outlier scatter: pct_error (x) vs z_score (y).

    Parameters
    ----------
    dm:
        Data manager providing ``outlier_flags``.
    selected_run:
        Run filter.
    category:
        Category filter.

    Returns
    -------
    pn.pane.Plotly
        Scatter plot of outliers, or an empty placeholder.
    """
    of = dm.outlier_flags
    if of.empty:
        return empty_placeholder("No outlier flag data available.")

    if selected_run != "All Runs":
        of = of[of["run_id"] == selected_run]
    of = _filter_by_category(of, category)

    if of.empty:
        return empty_placeholder("No outlier data matches the current filters.")

    error_col = (
        "pct_error"
        if "pct_error" in of.columns
        else "abs_pct_error"
        if "abs_pct_error" in of.columns
        else None
    )
    zscore_col = (
        "z_score"
        if "z_score" in of.columns
        else "zscore"
        if "zscore" in of.columns
        else None
    )

    if error_col is None or zscore_col is None:
        return empty_placeholder(
            "Outlier data missing required columns (pct_error, z_score)."
        )

    # Build hover text
    hover_parts: list[str] = []
    for col in ["county_name", "origin_year", "horizon"]:
        if col in of.columns:
            hover_parts.append(f"{col}: %{{customdata[{len(hover_parts)}]}}")
    customdata_cols = [c for c in ["county_name", "origin_year", "horizon"] if c in of.columns]
    hover_template = "<br>".join(hover_parts) + "<extra></extra>" if hover_parts else ""

    fig = go.Figure()
    category_col = "category" if "category" in of.columns else None

    if category_col:
        for cat, grp in of.groupby(category_col):
            color = CATEGORY_COLORS.get(str(cat), "#A0A0A0")
            customdata = grp[customdata_cols].values if customdata_cols else None
            fig.add_trace(
                go.Scatter(
                    x=grp[error_col],
                    y=grp[zscore_col],
                    mode="markers",
                    name=str(cat),
                    marker={"color": color, "size": 6, "opacity": 0.7},
                    customdata=customdata,
                    hovertemplate=hover_template,
                )
            )
    else:
        customdata = of[customdata_cols].values if customdata_cols else None
        fig.add_trace(
            go.Scatter(
                x=of[error_col],
                y=of[zscore_col],
                mode="markers",
                name="Outliers",
                marker={"color": SDC_RED, "size": 6, "opacity": 0.7},
                customdata=customdata,
                hovertemplate=hover_template,
            )
        )

    # Add z-score threshold lines
    for threshold in [2.0, 3.0]:
        fig.add_hline(
            y=threshold,
            line_dash="dash",
            line_color="#A0A0A0",
            line_width=1,
            annotation_text=f"z={threshold}",
            annotation_position="top right",
        )

    layout_defaults = get_plotly_layout_defaults()
    fig.update_layout(
        **layout_defaults,
        title="Outlier Detection: Error vs Z-Score",
        xaxis_title="Percentage Error",
        yaxis_title="Z-Score",
        height=450,
    )

    return pn.pane.Plotly(fig, sizing_mode="stretch_width")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_horizon_bias_tab(dm: DashboardDataManager) -> pn.Column:
    """Build the Horizon Bias tab for the Observatory dashboard.

    Contains six sections: run/category selectors, horizon profile chart,
    county heatmap, bias analysis heatmap, county report cards, and outlier
    scatter plot.  All sections react to the selector widgets via ``pn.bind``.

    Parameters
    ----------
    dm:
        The :class:`DashboardDataManager` providing all data access.

    Returns
    -------
    pn.Column
        A Panel Column containing all horizon bias analysis sections.
    """
    # Section 1: Selectors
    run_selector = pn.widgets.Select(
        name="Run",
        options=_run_options(dm),
        value=dm.champion_id or "All Runs",
        width=250,
    )
    category_selector = pn.widgets.Select(
        name="Category",
        options=_ALL_CATEGORIES,
        value="All",
        width=200,
    )
    selector_row = pn.FlexBox(
        run_selector,
        category_selector,
        flex_wrap="wrap",
        sizing_mode="stretch_width",
        styles={"gap": "10px"},
    )

    # Section 2: Horizon Profile (reactive)
    horizon_profile = pn.panel(pn.bind(
        _build_horizon_profile,
        dm=dm,
        selected_run=run_selector,
        category=category_selector,
    ), loading_indicator=True)

    # Section 3: County Heatmap (reactive on category only)
    county_heatmap = pn.panel(pn.bind(
        _build_county_heatmap,
        dm=dm,
        category=category_selector,
    ), loading_indicator=True)

    # Section 4: Bias Analysis (reactive on run)
    bias_heatmap = pn.panel(pn.bind(
        _build_bias_heatmap,
        dm=dm,
        selected_run=run_selector,
    ), loading_indicator=True)

    # Section 5: County Report Cards (reactive)
    report_cards = pn.panel(pn.bind(
        _build_county_report_cards,
        dm=dm,
        selected_run=run_selector,
        category=category_selector,
    ), loading_indicator=True)

    top_counties = pn.panel(pn.bind(
        _build_top_counties_table,
        dm=dm,
        selected_run=run_selector,
        category=category_selector,
    ), loading_indicator=True)

    # Section 6: Outlier Flags (reactive)
    outlier_scatter = pn.panel(pn.bind(
        _build_outlier_scatter,
        dm=dm,
        selected_run=run_selector,
        category=category_selector,
    ), loading_indicator=True)

    return pn.Column(
        section_header(
            "Horizon & Bias Analysis",
            subtitle="Accuracy by forecast horizon, county-level deep dives, and outlier detection",
        ),
        selector_row,
        pn.layout.Divider(),
        section_header("Horizon Accuracy Profile"),
        horizon_profile,
        pn.layout.Divider(),
        section_header("County MAPE Heatmap"),
        county_heatmap,
        top_counties,
        pn.layout.Divider(),
        section_header(
            "Bias Direction",
            subtitle="Blue = under-projection, Red = over-projection",
        ),
        bias_heatmap,
        pn.layout.Divider(),
        section_header("County Report Cards"),
        report_cards,
        pn.layout.Divider(),
        section_header("Outlier Flags"),
        outlier_scatter,
        sizing_mode="stretch_width",
    )
