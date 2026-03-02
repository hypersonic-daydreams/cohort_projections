"""
Place Projections section for the interactive HTML report.

Produces:
    1. Tier summary cards (HIGH / MODERATE / LOWER)
    2. Growth scatter plot (base pop vs growth rate, colored by tier)
    3. Housing-unit cross-validation scatter
    4. Top places sortable table

References:
    ADR-033: City/place projection methodology (share-of-county trending)
    ADR-060: Housing-unit method (complementary HU x PPH projections)
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

logger = logging.getLogger(__name__)


def _build_tier_cards(scenarios_data: dict[str, Any], theme: Any) -> str:
    """Build tier summary cards showing place counts and aggregate stats."""
    sdata = scenarios_data.get("baseline", {})
    qa_tier = sdata.get("qa_tier_summary")
    if qa_tier is None or qa_tier.empty:
        return '<p class="placeholder">Tier summary data not available.</p>'

    cards_html = []
    tier_order = ["HIGH", "MODERATE", "LOWER"]
    tier_descriptions = {
        "HIGH": "Population >10,000 | 5-year age groups",
        "MODERATE": "Population 2,500-10,000 | Broad age groups",
        "LOWER": "Population 500-2,500 | Total only",
    }

    for tier in tier_order:
        row = qa_tier[qa_tier["confidence_tier"] == tier]
        if row.empty:
            continue
        row = row.iloc[0]
        count = int(row["place_count"])
        color = theme.TIER_COLORS.get(tier, theme.NEUTRAL_COLOR)
        mean_rate = float(row["mean_growth_rate"]) if pd.notna(row["mean_growth_rate"]) else 0.0
        total_base = float(row["total_base_population"]) if pd.notna(row["total_base_population"]) else 0.0

        cards_html.append(f"""
            <div class="kpi-card" style="border-top: 4px solid {color};">
                <div class="kpi-label">{tier} Confidence</div>
                <div class="kpi-value" style="color: {color};">{count}</div>
                <div class="kpi-subtitle">{tier_descriptions.get(tier, '')}</div>
                <div class="kpi-detail">
                    Base Pop: {theme.format_population(total_base)}<br>
                    Avg Growth: {theme.format_percent(mean_rate)}
                </div>
            </div>
        """)

    return f'<div class="kpi-row">{"".join(cards_html)}</div>'


def _build_growth_scatter(scenarios_data: dict[str, Any], theme: Any) -> str:
    """Build scatter plot: base population (x) vs growth rate (y), colored by tier."""
    template_name = theme.register_template()

    sdata = scenarios_data.get("baseline", {})
    place_summary = sdata.get("place_summary")
    if place_summary is None or place_summary.empty:
        return '<p class="placeholder">Place summary data not available.</p>'

    # Filter to actual places (not balance_of_county)
    df = place_summary[place_summary["row_type"] == "place"].copy()
    if df.empty:
        return '<p class="placeholder">No place-level data available.</p>'

    df["growth_pct"] = df["growth_rate"] * 100

    fig = go.Figure()

    tier_order = ["HIGH", "MODERATE", "LOWER"]
    for tier in tier_order:
        mask = df["confidence_tier"] == tier
        subset = df[mask]
        if subset.empty:
            continue

        color = theme.TIER_COLORS.get(tier, theme.NEUTRAL_COLOR)

        fig.add_trace(go.Scatter(
            x=subset["base_population"],
            y=subset["growth_pct"],
            mode="markers",
            name=tier,
            marker={
                "color": color,
                "size": 8,
                "line": {"width": 0.5, "color": "white"},
            },
            text=subset["name"],
            hovertemplate=(
                "<b>%{text}</b><br>"
                "Base Pop: %{x:,.0f}<br>"
                "Growth: %{y:+.1f}%<br>"
                f"Tier: {tier}<extra></extra>"
            ),
        ))

    fig.update_layout(
        template=template_name,
        title="Place Growth by Size and Confidence Tier - Baseline (2025-2055)",
        xaxis_title="Base Population (2025)",
        xaxis_type="log",
        yaxis_title="Growth Rate (%)",
        height=480,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
    )

    # Add zero line
    fig.add_hline(y=0, line_width=1, line_color=theme.DARK_GRAY, line_dash="dash")

    return pio.to_html(fig, include_plotlyjs=False, full_html=False)


def _build_hu_comparison(scenarios_data: dict[str, Any], theme: Any) -> str:
    """Build scatter comparing share-trending vs housing-unit projections."""
    template_name = theme.register_template()

    sdata = scenarios_data.get("baseline", {})
    place_summary = sdata.get("place_summary")
    hu_projections = sdata.get("hu_projections")

    if hu_projections is None or hu_projections.empty:
        return '<p class="placeholder">Housing unit projection data not available.</p>'
    if place_summary is None or place_summary.empty:
        return '<p class="placeholder">Place summary data not available.</p>'

    # Get final-year HU projections
    hu_final = hu_projections[hu_projections["year"] == 2055].copy()
    if hu_final.empty:
        # Try the last available year
        max_yr = hu_projections["year"].max()
        hu_final = hu_projections[hu_projections["year"] == max_yr].copy()

    hu_final = hu_final.groupby("place_fips")["population_hu"].sum().reset_index()
    hu_final["place_fips"] = hu_final["place_fips"].astype(str)

    # Get share-trending final populations
    places = place_summary[place_summary["row_type"] == "place"].copy()
    places["place_fips"] = places["place_fips"].astype(str)

    merged = places.merge(
        hu_final,
        on="place_fips",
        how="inner",
    )

    if merged.empty:
        return '<p class="placeholder">No matching places for HU comparison.</p>'

    fig = go.Figure()

    # 1:1 reference line
    max_val = max(merged["final_population"].max(), merged["population_hu"].max())
    min_val = min(merged["final_population"].min(), merged["population_hu"].min())
    fig.add_trace(go.Scatter(
        x=[min_val * 0.8, max_val * 1.2],
        y=[min_val * 0.8, max_val * 1.2],
        mode="lines",
        line={"color": theme.MID_GRAY, "dash": "dash", "width": 1},
        name="1:1 Line",
        showlegend=True,
    ))

    # Scatter points colored by tier
    tier_order = ["HIGH", "MODERATE", "LOWER"]
    for tier in tier_order:
        mask = merged["confidence_tier"] == tier
        subset = merged[mask]
        if subset.empty:
            continue
        color = theme.TIER_COLORS.get(tier, theme.NEUTRAL_COLOR)

        fig.add_trace(go.Scatter(
            x=subset["final_population"],
            y=subset["population_hu"],
            mode="markers",
            name=tier,
            marker={"color": color, "size": 8, "line": {"width": 0.5, "color": "white"}},
            text=subset["name"],
            hovertemplate=(
                "<b>%{text}</b><br>"
                "Share-Trend: %{x:,.0f}<br>"
                "Housing-Unit: %{y:,.0f}<extra></extra>"
            ),
        ))

    fig.update_layout(
        template=template_name,
        title="Share-Trending vs Housing-Unit Projections (2055)",
        xaxis_title="Share-Trending Projection",
        yaxis_title="Housing-Unit Projection",
        xaxis_tickformat=",",
        yaxis_tickformat=",",
        height=480,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
    )

    return pio.to_html(fig, include_plotlyjs=False, full_html=False)


def _build_places_table(scenarios_data: dict[str, Any], theme: Any) -> str:
    """Build sortable table of all 90 places with key stats."""
    sdata = scenarios_data.get("baseline", {})
    place_summary = sdata.get("place_summary")
    if place_summary is None or place_summary.empty:
        return '<p class="placeholder">Place summary data not available.</p>'

    # Filter to actual places
    df = place_summary[place_summary["row_type"] == "place"].copy()
    df = df.sort_values("base_population", ascending=False)
    df["growth_pct"] = df["growth_rate"] * 100

    rows_html = []
    for _, row in df.iterrows():
        rate = float(row["growth_rate"]) if pd.notna(row["growth_rate"]) else 0.0
        color = theme.growth_color(rate)
        tier = str(row["confidence_tier"]) if pd.notna(row["confidence_tier"]) else ""
        tier_color = theme.TIER_COLORS.get(tier, theme.NEUTRAL_COLOR)
        growth_pct = f"{row['growth_pct']:+.1f}%" if pd.notna(row["growth_pct"]) else "N/A"

        base_share = f"{row['base_share'] * 100:.1f}%" if pd.notna(row["base_share"]) else ""
        final_share = f"{row['final_share'] * 100:.1f}%" if pd.notna(row["final_share"]) else ""

        rows_html.append(f"""
            <tr>
                <td>{row['name']}</td>
                <td><span class="tier-badge" style="background: {tier_color};">{tier}</span></td>
                <td class="num">{theme.format_population(row['base_population'])}</td>
                <td class="num">{theme.format_population(row['final_population'])}</td>
                <td class="num">{theme.format_change(row['absolute_growth'])}</td>
                <td class="num" style="color: {color}; font-weight: 600;">{growth_pct}</td>
                <td class="num">{base_share}</td>
                <td class="num">{final_share}</td>
            </tr>
        """)

    table_html = f"""
        <div class="table-wrapper">
            <table class="data-table sortable" id="places-table">
                <thead>
                    <tr>
                        <th data-sort="string">Place</th>
                        <th data-sort="string">Tier</th>
                        <th data-sort="int" class="num">Base Pop (2025)</th>
                        <th data-sort="int" class="num">Projected Pop (2055)</th>
                        <th data-sort="int" class="num">Change</th>
                        <th data-sort="float" class="num">Growth Rate</th>
                        <th data-sort="float" class="num">Base Share</th>
                        <th data-sort="float" class="num">Final Share</th>
                    </tr>
                </thead>
                <tbody>
                    {"".join(rows_html)}
                </tbody>
            </table>
        </div>
    """
    return table_html


def build_place_projections(scenarios_data: dict[str, Any], theme: Any) -> str:
    """Build the complete Place Projections section as an HTML string.

    Parameters
    ----------
    scenarios_data : dict
        Keyed by scenario name, each value is a dict with:
        - 'place_summary': DataFrame from places_summary.csv
        - 'hu_projections': DataFrame from housing_unit_projections.parquet
        - 'qa_tier_summary': DataFrame from qa/qa_tier_summary.csv
    theme : module
        The _report_theme module.

    Returns
    -------
    str
        HTML fragment for the place projections section.
    """
    parts = ['<h2>Place Projections</h2>']

    try:
        parts.append(_build_tier_cards(scenarios_data, theme))
    except Exception:
        logger.exception("Error building tier cards")
        parts.append('<p class="placeholder">Error generating tier cards.</p>')

    try:
        parts.append('<div class="chart-container">')
        parts.append(_build_growth_scatter(scenarios_data, theme))
        parts.append('</div>')
    except Exception:
        logger.exception("Error building growth scatter")
        parts.append('<p class="placeholder">Error generating growth scatter plot.</p>')

    try:
        parts.append('<div class="chart-container">')
        parts.append(_build_hu_comparison(scenarios_data, theme))
        parts.append('</div>')
    except Exception:
        logger.exception("Error building HU comparison")
        parts.append('<p class="placeholder">Error generating housing-unit comparison.</p>')

    try:
        parts.append(_build_places_table(scenarios_data, theme))
    except Exception:
        logger.exception("Error building places table")
        parts.append('<p class="placeholder">Error generating places table.</p>')

    return "\n".join(parts)
