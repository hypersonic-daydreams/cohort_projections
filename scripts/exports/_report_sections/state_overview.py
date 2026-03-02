"""
State Overview section for the interactive HTML report.

Produces:
    1. Headline KPI cards (base pop, projected pop per scenario, growth rates)
    2. Multi-scenario trend line (total population 2025-2055)
    3. Population pyramid with year slider (baseline)
    4. Age structure shift (stacked area: youth / working / elderly)

References:
    ADR-037: CBO-grounded scenario methodology
    ADR-054: State-county aggregation (bottom-up)
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

logger = logging.getLogger(__name__)

# Age group definitions
AGE_GROUP_BINS = list(range(0, 90, 5)) + [91]
AGE_GROUP_LABELS = [
    "0-4", "5-9", "10-14", "15-19", "20-24", "25-29", "30-34", "35-39",
    "40-44", "45-49", "50-54", "55-59", "60-64", "65-69", "70-74", "75-79",
    "80-84", "85+",
]

YOUTH_CUTOFF = 15   # 0-14
WORKING_END = 65    # 15-64
KEY_YEARS = [2025, 2030, 2035, 2040, 2045, 2050, 2055]
PYRAMID_YEARS = [2025, 2035, 2045, 2055]


def _build_kpi_cards(scenarios_data: dict[str, Any], theme: Any) -> str:
    """Build HTML headline KPI cards showing base and projected populations."""
    cards_html = []

    # Get base population from first available scenario
    base_pop = None
    for scenario_key in ["baseline", "high_growth", "restricted_growth"]:
        state_summary = scenarios_data.get(scenario_key, {}).get("state_summary")
        if state_summary is not None and not state_summary.empty:
            base_row = state_summary[state_summary["year"] == 2025]
            if not base_row.empty:
                base_pop = float(base_row["total_population"].iloc[0])
                break

    if base_pop is None:
        return '<div class="kpi-row"><p>No state data available.</p></div>'

    # Base population card
    cards_html.append(f"""
        <div class="kpi-card kpi-base">
            <div class="kpi-label">Base Population (2025)</div>
            <div class="kpi-value">{theme.format_population(base_pop)}</div>
            <div class="kpi-subtitle">Census PEP 2025 Vintage</div>
        </div>
    """)

    # Per-scenario projected population cards
    scenario_order = ["baseline", "high_growth", "restricted_growth"]
    for scenario_key in scenario_order:
        sdata = scenarios_data.get(scenario_key, {})
        state_summary = sdata.get("state_summary")
        if state_summary is None or state_summary.empty:
            continue

        final_row = state_summary[state_summary["year"] == 2055]
        if final_row.empty:
            continue

        final_pop = float(final_row["total_population"].iloc[0])
        growth = final_pop - base_pop
        rate = growth / base_pop if base_pop > 0 else 0.0

        color = theme.get_scenario_color(scenario_key)
        label = theme.get_scenario_label(scenario_key)

        cards_html.append(f"""
            <div class="kpi-card" style="border-top: 4px solid {color};">
                <div class="kpi-label">{label} (2055)</div>
                <div class="kpi-value" style="color: {color};">{theme.format_population(final_pop)}</div>
                <div class="kpi-subtitle">{theme.format_change(growth)} ({theme.format_percent(rate)})</div>
            </div>
        """)

    return f'<div class="kpi-row">{"".join(cards_html)}</div>'


def _build_trend_chart(scenarios_data: dict[str, Any], theme: Any) -> str:
    """Build multi-scenario trend line chart for state total population."""
    template_name = theme.register_template()
    fig = go.Figure()

    scenario_order = ["baseline", "high_growth", "restricted_growth"]
    for scenario_key in scenario_order:
        sdata = scenarios_data.get(scenario_key, {})
        state_summary = sdata.get("state_summary")
        if state_summary is None or state_summary.empty:
            continue

        df = state_summary.sort_values("year")
        color = theme.get_scenario_color(scenario_key)
        label = theme.get_scenario_label(scenario_key)

        fig.add_trace(go.Scatter(
            x=df["year"],
            y=df["total_population"],
            name=label,
            mode="lines+markers",
            line={"color": color, "width": 2.5},
            marker={"size": 4},
            hovertemplate=(
                f"<b>{label}</b><br>"
                "Year: %{x}<br>"
                "Population: %{y:,.0f}<extra></extra>"
            ),
        ))

    fig.update_layout(
        template=template_name,
        title="State Population Projections by Scenario (2025-2055)",
        xaxis_title="Year",
        yaxis_title="Total Population",
        yaxis_tickformat=",",
        height=420,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
    )

    return pio.to_html(fig, include_plotlyjs=False, full_html=False)


def _build_population_pyramid(scenarios_data: dict[str, Any], theme: Any) -> str:
    """Build population pyramid with year slider for baseline scenario."""
    template_name = theme.register_template()

    sdata = scenarios_data.get("baseline", {})
    state_parquet = sdata.get("state_parquet")
    if state_parquet is None or state_parquet.empty:
        return '<p class="placeholder">Baseline state parquet data not available for population pyramid.</p>'

    df = state_parquet.copy()

    # Assign 5-year age groups
    df["age_group"] = pd.cut(
        df["age"], bins=AGE_GROUP_BINS, right=False, labels=AGE_GROUP_LABELS,
    )
    df["age_group"] = df["age_group"].astype(str)

    # Aggregate across race by age_group x sex x year
    agg = (
        df.groupby(["year", "age_group", "sex"], observed=True)["population"]
        .sum()
        .reset_index()
    )

    # Build frames for animation slider
    fig = go.Figure()

    # Initial frame: first pyramid year
    initial_year = PYRAMID_YEARS[0]
    for sex, sign, color in [
        ("Male", -1, theme.SEX_COLORS["Male"]),
        ("Female", 1, theme.SEX_COLORS["Female"]),
    ]:
        mask = (agg["year"] == initial_year) & (agg["sex"] == sex)
        subset = agg[mask].copy()
        subset["age_group"] = pd.Categorical(
            subset["age_group"], categories=AGE_GROUP_LABELS, ordered=True,
        )
        subset = subset.sort_values("age_group")
        vals = subset["population"].values * sign

        fig.add_trace(go.Bar(
            y=subset["age_group"],
            x=vals,
            name=sex,
            orientation="h",
            marker_color=color,
            hovertemplate=(
                f"<b>{sex}</b><br>"
                "Age: %{y}<br>"
                "Population: %{customdata:,.0f}<extra></extra>"
            ),
            customdata=subset["population"].values,
        ))

    # Create slider steps
    steps = []
    frames = []
    for yr in PYRAMID_YEARS:
        frame_data = []
        for sex, sign in [("Male", -1), ("Female", 1)]:
            mask = (agg["year"] == yr) & (agg["sex"] == sex)
            subset = agg[mask].copy()
            subset["age_group"] = pd.Categorical(
                subset["age_group"], categories=AGE_GROUP_LABELS, ordered=True,
            )
            subset = subset.sort_values("age_group")
            vals = subset["population"].values * sign
            frame_data.append(go.Bar(
                y=subset["age_group"],
                x=vals,
                customdata=subset["population"].values,
            ))
        frames.append(go.Frame(data=frame_data, name=str(yr)))
        steps.append({
            "method": "animate",
            "args": [[str(yr)], {
                "mode": "immediate",
                "frame": {"duration": 300, "redraw": True},
                "transition": {"duration": 200},
            }],
            "label": str(yr),
        })

    fig.frames = frames

    # Compute max extent for symmetric axis
    agg.groupby(["year", "sex"])["population"].sum().max()
    max_bar = (
        agg[agg["year"].isin(PYRAMID_YEARS)]
        .groupby(["year", "age_group", "sex"])["population"]
        .sum()
        .max()
    )
    x_extent = float(max_bar) * 1.1

    fig.update_layout(
        template=template_name,
        title=f"Population Pyramid - Baseline ({PYRAMID_YEARS[0]})",
        xaxis={
            "title": "Population",
            "range": [-x_extent, x_extent],
            "tickformat": ",",
            # Show absolute values on tick labels
        },
        yaxis={"title": "Age Group", "categoryorder": "array", "categoryarray": AGE_GROUP_LABELS},
        barmode="overlay",
        bargap=0.05,
        height=520,
        sliders=[{
            "active": 0,
            "currentvalue": {"prefix": "Year: ", "font": {"size": 14}},
            "pad": {"t": 40},
            "steps": steps,
        }],
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
    )

    # Update title to reflect slider
    for i, yr in enumerate(PYRAMID_YEARS):
        fig.frames[i].layout = go.Layout(
            title=f"Population Pyramid - Baseline ({yr})"
        )

    return pio.to_html(fig, include_plotlyjs=False, full_html=False)


def _build_age_structure_chart(scenarios_data: dict[str, Any], theme: Any) -> str:
    """Build stacked area chart showing youth/working/elderly proportions over time."""
    template_name = theme.register_template()

    sdata = scenarios_data.get("baseline", {})
    state_summary = sdata.get("state_summary")
    if state_summary is None or state_summary.empty:
        return '<p class="placeholder">State summary data not available for age structure chart.</p>'

    df = state_summary.sort_values("year").copy()

    # Use the pre-computed age group columns from the state summary
    # Columns: population_under_18, population_working_age, population_65_plus
    if not all(
        col in df.columns
        for col in ["population_under_18", "population_working_age", "population_65_plus"]
    ):
        return '<p class="placeholder">Age structure columns not found in state summary.</p>'

    total = df["total_population"]
    youth_pct = df["population_under_18"] / total * 100
    working_pct = df["population_working_age"] / total * 100
    elderly_pct = df["population_65_plus"] / total * 100

    fig = go.Figure()

    for name, values, color in [
        ("Youth (Under 18)", youth_pct, theme.AGE_STRUCTURE_COLORS["Youth (0-14)"]),
        ("Working Age (18-64)", working_pct, theme.AGE_STRUCTURE_COLORS["Working Age (15-64)"]),
        ("Elderly (65+)", elderly_pct, theme.AGE_STRUCTURE_COLORS["Elderly (65+)"]),
    ]:
        fig.add_trace(go.Scatter(
            x=df["year"],
            y=values,
            name=name,
            mode="lines",
            stackgroup="one",
            line={"width": 0.5, "color": color},
            fillcolor=color,
            hovertemplate=(
                f"<b>{name}</b><br>"
                "Year: %{x}<br>"
                "Share: %{y:.1f}%<extra></extra>"
            ),
        ))

    fig.update_layout(
        template=template_name,
        title="Age Structure Shift - Baseline Scenario (2025-2055)",
        xaxis_title="Year",
        yaxis_title="Share of Total Population (%)",
        yaxis={"range": [0, 100]},
        height=400,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
    )

    return pio.to_html(fig, include_plotlyjs=False, full_html=False)


def build_state_overview(scenarios_data: dict[str, Any], theme: Any) -> str:
    """Build the complete State Overview section as an HTML string.

    Parameters
    ----------
    scenarios_data : dict
        Keyed by scenario name, each value is a dict with:
        - 'state_summary': DataFrame from states_summary.csv
        - 'state_parquet': DataFrame from state parquet (for pyramids)
    theme : module
        The _report_theme module with colors, formatters, and templates.

    Returns
    -------
    str
        HTML fragment for the state overview section.
    """
    parts = ['<h2>State Overview</h2>']

    try:
        parts.append(_build_kpi_cards(scenarios_data, theme))
    except Exception:
        logger.exception("Error building KPI cards")
        parts.append('<p class="placeholder">Error generating KPI cards.</p>')

    try:
        parts.append('<div class="chart-container">')
        parts.append(_build_trend_chart(scenarios_data, theme))
        parts.append('</div>')
    except Exception:
        logger.exception("Error building trend chart")
        parts.append('<p class="placeholder">Error generating trend chart.</p>')

    try:
        parts.append('<div class="chart-container">')
        parts.append(_build_population_pyramid(scenarios_data, theme))
        parts.append('</div>')
    except Exception:
        logger.exception("Error building population pyramid")
        parts.append('<p class="placeholder">Error generating population pyramid.</p>')

    try:
        parts.append('<div class="chart-container">')
        parts.append(_build_age_structure_chart(scenarios_data, theme))
        parts.append('</div>')
    except Exception:
        logger.exception("Error building age structure chart")
        parts.append('<p class="placeholder">Error generating age structure chart.</p>')

    return "\n".join(parts)
