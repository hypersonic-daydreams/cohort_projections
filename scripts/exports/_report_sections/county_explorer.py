"""
County Explorer section for the interactive HTML report.

Produces:
    1. Sortable summary table (all 53 counties)
    2. Growth/decline horizontal bar chart
    3. Top 10 growing / Top 10 declining paired bar charts
    4. County comparison chart with dropdown selection

References:
    ADR-054: State-county aggregation (bottom-up)
"""

from __future__ import annotations

import logging
from typing import Any

import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)


def _build_county_table(scenarios_data: dict[str, Any], theme: Any) -> str:
    """Build a sortable HTML table of all 53 counties (baseline scenario)."""
    sdata = scenarios_data.get("baseline", {})
    county_summary = sdata.get("county_summary")
    if county_summary is None or county_summary.empty:
        return '<p class="placeholder">County summary data not available.</p>'

    df = county_summary.sort_values("growth_rate", ascending=False).copy()
    df["display_name"] = df["name"].str.replace(" County", "", regex=False)
    df["growth_pct"] = df["growth_rate"] * 100

    rows_html = []
    for _, row in df.iterrows():
        rate = float(row["growth_rate"])
        color = theme.growth_color(rate)
        growth_pct = f"{row['growth_pct']:+.1f}%"
        rows_html.append(f"""
            <tr>
                <td>{int(row['fips'])}</td>
                <td>{row['display_name']}</td>
                <td class="num">{theme.format_population(row['base_population'])}</td>
                <td class="num">{theme.format_population(row['final_population'])}</td>
                <td class="num">{theme.format_change(row['absolute_growth'])}</td>
                <td class="num" style="color: {color}; font-weight: 600;">{growth_pct}</td>
            </tr>
        """)

    table_html = f"""
        <div class="table-wrapper">
            <table class="data-table sortable" id="county-table">
                <thead>
                    <tr>
                        <th data-sort="int">FIPS</th>
                        <th data-sort="string">County</th>
                        <th data-sort="int" class="num">Base Pop (2025)</th>
                        <th data-sort="int" class="num">Projected Pop (2055)</th>
                        <th data-sort="int" class="num">Change</th>
                        <th data-sort="float" class="num">Growth Rate</th>
                    </tr>
                </thead>
                <tbody>
                    {"".join(rows_html)}
                </tbody>
            </table>
        </div>
    """
    return table_html


def _build_growth_bar_chart(scenarios_data: dict[str, Any], theme: Any) -> str:
    """Build horizontal bar chart of growth rates for all 53 counties."""
    template_name = theme.register_template()

    sdata = scenarios_data.get("baseline", {})
    county_summary = sdata.get("county_summary")
    if county_summary is None or county_summary.empty:
        return '<p class="placeholder">County summary data not available.</p>'

    df = county_summary.sort_values("growth_rate", ascending=True).copy()
    df["display_name"] = df["name"].str.replace(" County", "", regex=False)
    df["growth_pct"] = df["growth_rate"] * 100

    colors = [theme.growth_color(r) for r in df["growth_rate"]]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=df["display_name"],
        x=df["growth_pct"],
        orientation="h",
        marker_color=colors,
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Growth Rate: %{x:+.1f}%<br>"
            "<extra></extra>"
        ),
    ))

    fig.update_layout(
        template=template_name,
        title="County Growth Rates - Baseline Scenario (2025-2055)",
        xaxis_title="Growth Rate (%)",
        yaxis_title="",
        height=max(600, len(df) * 18),
        showlegend=False,
        margin={"l": 120},
    )

    # Add zero line
    fig.add_vline(x=0, line_width=1, line_color=theme.DARK_GRAY)

    return pio.to_html(fig, include_plotlyjs=False, full_html=False)


def _build_top_bottom_charts(scenarios_data: dict[str, Any], theme: Any) -> str:
    """Build paired bar charts for top 10 growing and top 10 declining counties."""
    template_name = theme.register_template()

    sdata = scenarios_data.get("baseline", {})
    county_summary = sdata.get("county_summary")
    if county_summary is None or county_summary.empty:
        return '<p class="placeholder">County summary data not available.</p>'

    df = county_summary.copy()
    df["display_name"] = df["name"].str.replace(" County", "", regex=False)
    df["growth_pct"] = df["growth_rate"] * 100

    # Top 10 growing
    top10 = df.nlargest(10, "growth_rate").sort_values("growth_rate", ascending=True)
    # Top 10 declining (bottom 10)
    bottom10 = df.nsmallest(10, "growth_rate").sort_values("growth_rate", ascending=True)

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Top 10 Growing Counties", "Top 10 Declining Counties"),
        horizontal_spacing=0.15,
    )

    # Growing
    fig.add_trace(go.Bar(
        y=top10["display_name"],
        x=top10["growth_pct"],
        orientation="h",
        marker_color=theme.GROWTH_COLOR,
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Growth: %{x:+.1f}%<br>"
            "Pop 2055: %{customdata:,.0f}<extra></extra>"
        ),
        customdata=top10["final_population"],
        showlegend=False,
    ), row=1, col=1)

    # Declining
    fig.add_trace(go.Bar(
        y=bottom10["display_name"],
        x=bottom10["growth_pct"],
        orientation="h",
        marker_color=theme.DECLINE_COLOR,
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Decline: %{x:+.1f}%<br>"
            "Pop 2055: %{customdata:,.0f}<extra></extra>"
        ),
        customdata=bottom10["final_population"],
        showlegend=False,
    ), row=1, col=2)

    fig.update_layout(
        template=template_name,
        title="Top 10 Growing and Declining Counties - Baseline (2025-2055)",
        height=400,
        margin={"l": 120},
    )

    fig.update_xaxes(title_text="Growth Rate (%)", row=1, col=1)
    fig.update_xaxes(title_text="Growth Rate (%)", row=1, col=2)

    return pio.to_html(fig, include_plotlyjs=False, full_html=False)


def _build_county_comparison(scenarios_data: dict[str, Any], theme: Any) -> str:
    """Build county comparison line chart with dropdown to select counties.

    Uses the baseline scenario yearly county data for time series comparison.
    Preselects the 5 largest counties by base population.
    """
    template_name = theme.register_template()

    sdata = scenarios_data.get("baseline", {})
    county_yearly = sdata.get("county_yearly")
    county_summary = sdata.get("county_summary")
    if county_yearly is None or county_yearly.empty:
        return '<p class="placeholder">County yearly data not available for comparison chart.</p>'
    if county_summary is None or county_summary.empty:
        return '<p class="placeholder">County summary data not available for comparison chart.</p>'

    # Get county names mapping
    name_map = dict(zip(
        county_summary["fips"].astype(int),
        county_summary["name"].str.replace(" County", "", regex=False),
        strict=False,
    ))

    # Aggregate yearly county data
    yearly = county_yearly.groupby(["fips", "year"])["population"].sum().reset_index()

    # Get top 5 counties by base pop
    top5 = county_summary.nlargest(5, "base_population")["fips"].astype(int).tolist()

    # Get all unique counties
    all_counties = sorted(yearly["fips"].unique())

    fig = go.Figure()

    # Add a trace for each county (visible if in top5, hidden otherwise)
    for fips in all_counties:
        cdata = yearly[yearly["fips"] == fips].sort_values("year")
        name = name_map.get(fips, f"FIPS {fips}")
        visible = fips in top5

        fig.add_trace(go.Scatter(
            x=cdata["year"],
            y=cdata["population"],
            name=name,
            mode="lines+markers",
            marker={"size": 3},
            visible=True if visible else "legendonly",
            hovertemplate=(
                f"<b>{name}</b><br>"
                "Year: %{x}<br>"
                "Population: %{y:,.0f}<extra></extra>"
            ),
        ))

    fig.update_layout(
        template=template_name,
        title="County Population Comparison - Baseline (2025-2055)",
        xaxis_title="Year",
        yaxis_title="Population",
        yaxis_tickformat=",",
        height=480,
        legend={
            "orientation": "v",
            "yanchor": "top",
            "y": 1.0,
            "xanchor": "left",
            "x": 1.02,
            "font": {"size": 10},
        },
    )

    return pio.to_html(fig, include_plotlyjs=False, full_html=False)


def build_county_explorer(scenarios_data: dict[str, Any], theme: Any) -> str:
    """Build the complete County Explorer section as an HTML string.

    Parameters
    ----------
    scenarios_data : dict
        Keyed by scenario name, each value is a dict with:
        - 'county_summary': DataFrame from countys_summary.csv
        - 'county_yearly': DataFrame of county x year population totals
    theme : module
        The _report_theme module.

    Returns
    -------
    str
        HTML fragment for the county explorer section.
    """
    parts = ['<h2>County Explorer</h2>']

    try:
        parts.append(_build_county_table(scenarios_data, theme))
    except Exception:
        logger.exception("Error building county table")
        parts.append('<p class="placeholder">Error generating county table.</p>')

    try:
        parts.append('<div class="chart-container">')
        parts.append(_build_growth_bar_chart(scenarios_data, theme))
        parts.append('</div>')
    except Exception:
        logger.exception("Error building growth bar chart")
        parts.append('<p class="placeholder">Error generating growth bar chart.</p>')

    try:
        parts.append('<div class="chart-container">')
        parts.append(_build_top_bottom_charts(scenarios_data, theme))
        parts.append('</div>')
    except Exception:
        logger.exception("Error building top/bottom charts")
        parts.append('<p class="placeholder">Error generating top/bottom charts.</p>')

    try:
        parts.append('<div class="chart-container">')
        parts.append(_build_county_comparison(scenarios_data, theme))
        parts.append('<p class="chart-note">Click county names in the legend to show/hide. '
                     'Top 5 counties by base population are shown by default.</p>')
        parts.append('</div>')
    except Exception:
        logger.exception("Error building county comparison")
        parts.append('<p class="placeholder">Error generating county comparison chart.</p>')

    return "\n".join(parts)
