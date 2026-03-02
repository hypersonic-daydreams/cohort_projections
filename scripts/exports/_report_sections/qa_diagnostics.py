"""
QA & Diagnostics section for the interactive HTML report.

Produces:
    1. Backtest variant comparison (grouped bar chart)
    2. Rolling-origin scores (line chart across windows)
    3. Tier performance table (MAPE/MedAPE by tier)
    4. Reconciliation summary (share-sum validation)

References:
    ADR-057: Rolling-origin backtests (expanding-window cross-validation)
    ADR-033: City/place projection methodology
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

logger = logging.getLogger(__name__)

VARIANT_COLORS = {
    "A-I": "#0563C1",
    "A-II": "#00B0F0",
    "B-I": "#00B050",
    "B-II": "#1F3864",
}


def _build_variant_comparison(backtesting_data: dict[str, Any], theme: Any) -> str:
    """Build grouped bar chart comparing backtest variant scores."""
    template_name = theme.register_template()

    variant_scores = backtesting_data.get("variant_scores")
    if variant_scores is None or variant_scores.empty:
        return '<p class="placeholder">Backtest variant scores not available.</p>'

    fig = go.Figure()

    # Group by window (primary, secondary)
    variant_scores["window"].unique()
    variants = sorted(variant_scores["variant_id"].unique())

    for variant in variants:
        vdata = variant_scores[variant_scores["variant_id"] == variant]
        color = VARIANT_COLORS.get(variant, theme.NEUTRAL_COLOR)

        # Get fitting/constraint method for hover
        if not vdata.empty:
            fit = vdata.iloc[0]["fitting_method"]
            constraint = vdata.iloc[0]["constraint_method"]
            label_detail = f"{fit.upper()} + {constraint.replace('_', ' ').title()}"
        else:
            label_detail = ""

        fig.add_trace(go.Bar(
            x=vdata["window"],
            y=vdata["score"],
            name=variant,
            marker_color=color,
            hovertemplate=(
                f"<b>{variant}</b> ({label_detail})<br>"
                "Window: %{x}<br>"
                "Score: %{y:.3f}<extra></extra>"
            ),
        ))

    fig.update_layout(
        template=template_name,
        title="Backtest Variant Comparison (Lower Score = Better)",
        xaxis_title="Backtest Window",
        yaxis_title="Composite Score (Weighted MedAPE)",
        barmode="group",
        height=380,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
    )

    return pio.to_html(fig, include_plotlyjs=False, full_html=False)


def _build_rolling_origin_chart(backtesting_data: dict[str, Any], theme: Any) -> str:
    """Build line chart showing rolling-origin variant scores across windows."""
    template_name = theme.register_template()

    rolling_scores = backtesting_data.get("rolling_origin_per_window")
    if rolling_scores is None or rolling_scores.empty:
        return '<p class="placeholder">Rolling-origin per-window scores not available.</p>'

    fig = go.Figure()

    variants = sorted(rolling_scores["variant_id"].unique())
    for variant in variants:
        vdata = rolling_scores[rolling_scores["variant_id"] == variant].sort_values("train_end")
        color = VARIANT_COLORS.get(variant, theme.NEUTRAL_COLOR)

        # Create a nice window label
        window_labels = vdata["window"].values

        fig.add_trace(go.Scatter(
            x=window_labels,
            y=vdata["score"],
            name=variant,
            mode="lines+markers",
            line={"color": color, "width": 2},
            marker={"size": 6},
            hovertemplate=(
                f"<b>{variant}</b><br>"
                "Window: %{x}<br>"
                "Score: %{y:.3f}<extra></extra>"
            ),
        ))

    fig.update_layout(
        template=template_name,
        title="Rolling-Origin Cross-Validation Scores by Window",
        xaxis_title="Train/Test Window",
        yaxis_title="Composite Score",
        height=380,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
        xaxis={"tickangle": -30},
    )

    return pio.to_html(fig, include_plotlyjs=False, full_html=False)


def _build_tier_performance(backtesting_data: dict[str, Any], theme: Any) -> str:
    """Build table showing backtest performance metrics by confidence tier."""
    tier_agg = backtesting_data.get("tier_aggregates")
    if tier_agg is None or tier_agg.empty:
        return '<p class="placeholder">Tier aggregate backtest data not available.</p>'

    # Filter to primary window, winning variant (B-II)
    winner = backtesting_data.get("backtest_winner", {})
    winner_variant = winner.get("winner_variant_id", "B-II")
    winner_window = winner.get("window", "primary")

    mask = (tier_agg["variant_id"] == winner_variant) & (tier_agg["window"] == winner_window)
    df = tier_agg[mask].copy()
    if df.empty:
        # Fallback: show all primary window data
        df = tier_agg[tier_agg["window"] == "primary"].copy()

    # Filter to scored tiers only
    scored_tiers = ["HIGH", "MODERATE", "LOWER"]
    df = df[df["confidence_tier"].isin(scored_tiers)].copy()

    if df.empty:
        return '<p class="placeholder">No tier performance data for winning variant.</p>'

    rows_html = []
    for _, row in df.iterrows():
        tier = row["confidence_tier"]
        color = theme.TIER_COLORS.get(tier, theme.NEUTRAL_COLOR)
        medape = f"{row['tier_medape']:.1f}%" if pd.notna(row["tier_medape"]) else "N/A"
        mean_me = f"{row['tier_mean_me']:+.1f}%" if pd.notna(row["tier_mean_me"]) else "N/A"
        p90 = f"{row['tier_p90_mape']:.1f}%" if pd.notna(row["tier_p90_mape"]) else "N/A"
        max_mape = f"{row['tier_max_mape']:.1f}%" if pd.notna(row["tier_max_mape"]) else "N/A"
        count = int(row["place_count"])

        rows_html.append(f"""
            <tr>
                <td><span class="tier-badge" style="background: {color};">{tier}</span></td>
                <td class="num">{count}</td>
                <td class="num">{medape}</td>
                <td class="num">{mean_me}</td>
                <td class="num">{p90}</td>
                <td class="num">{max_mape}</td>
            </tr>
        """)

    variant_label = f"Variant {winner_variant} ({winner_window} window)"
    table_html = f"""
        <h3>Backtest Tier Performance - {variant_label}</h3>
        <div class="table-wrapper">
            <table class="data-table" id="tier-performance-table">
                <thead>
                    <tr>
                        <th>Tier</th>
                        <th class="num">Places</th>
                        <th class="num">MedAPE</th>
                        <th class="num">Mean ME</th>
                        <th class="num">P90 MAPE</th>
                        <th class="num">Max MAPE</th>
                    </tr>
                </thead>
                <tbody>
                    {"".join(rows_html)}
                </tbody>
            </table>
        </div>
    """
    return table_html


def _build_reconciliation_summary(qa_data: dict[str, Any], theme: Any) -> str:
    """Build reconciliation / share-sum validation summary table."""
    share_sum = qa_data.get("share_sum_validation")
    if share_sum is None or share_sum.empty:
        return '<p class="placeholder">Share-sum validation data not available.</p>'

    # Aggregate: count satisfied vs not, by year
    summary = (
        share_sum.groupby("year")
        .agg(
            total_counties=("county_fips", "count"),
            satisfied=("constraint_satisfied", "sum"),
        )
        .reset_index()
    )
    summary["pct_satisfied"] = summary["satisfied"] / summary["total_counties"] * 100

    rows_html = []
    for _, row in summary.iterrows():
        pct = float(row["pct_satisfied"])
        color = theme.GROWTH_COLOR if pct >= 99.9 else theme.DECLINE_COLOR
        rows_html.append(f"""
            <tr>
                <td class="num">{int(row['year'])}</td>
                <td class="num">{int(row['total_counties'])}</td>
                <td class="num">{int(row['satisfied'])}</td>
                <td class="num" style="color: {color}; font-weight: 600;">{pct:.1f}%</td>
            </tr>
        """)

    table_html = f"""
        <h3>County-Place Share-Sum Validation</h3>
        <p class="chart-note">Verifies that place shares plus balance-of-county sum to 100% for each county-year.</p>
        <div class="table-wrapper">
            <table class="data-table" id="reconciliation-table">
                <thead>
                    <tr>
                        <th class="num">Year</th>
                        <th class="num">Counties</th>
                        <th class="num">Satisfied</th>
                        <th class="num">% Satisfied</th>
                    </tr>
                </thead>
                <tbody>
                    {"".join(rows_html)}
                </tbody>
            </table>
        </div>
    """
    return table_html


def build_qa_diagnostics(
    backtesting_data: dict[str, Any],
    qa_data: dict[str, Any],
    theme: Any,
) -> str:
    """Build the complete QA & Diagnostics section as an HTML string.

    Parameters
    ----------
    backtesting_data : dict
        Contains:
        - 'variant_scores': DataFrame from backtest_variant_scores.csv
        - 'rolling_origin_per_window': DataFrame from rolling_origin_per_window_scores.csv
        - 'rolling_origin_aggregated': DataFrame from rolling_origin_aggregated_scores.csv
        - 'tier_aggregates': DataFrame from backtest_tier_aggregates.csv
        - 'backtest_winner': dict from backtest_winner.json
        - 'rolling_origin_winner': dict from rolling_origin_winner.json
    qa_data : dict
        Contains:
        - 'tier_summary': DataFrame from qa_tier_summary.csv
        - 'reconciliation': DataFrame from qa_reconciliation_magnitude.csv
        - 'share_sum_validation': DataFrame from qa_share_sum_validation.csv
        - 'outlier_flags': DataFrame from qa_outlier_flags.csv
    theme : module
        The _report_theme module.

    Returns
    -------
    str
        HTML fragment for the QA diagnostics section.
    """
    parts = ['<h2>QA &amp; Diagnostics</h2>']

    try:
        parts.append('<div class="chart-container">')
        parts.append(_build_variant_comparison(backtesting_data, theme))
        parts.append('</div>')
    except Exception:
        logger.exception("Error building variant comparison")
        parts.append('<p class="placeholder">Error generating variant comparison chart.</p>')

    try:
        parts.append('<div class="chart-container">')
        parts.append(_build_rolling_origin_chart(backtesting_data, theme))
        parts.append('</div>')
    except Exception:
        logger.exception("Error building rolling-origin chart")
        parts.append('<p class="placeholder">Error generating rolling-origin chart.</p>')

    try:
        parts.append(_build_tier_performance(backtesting_data, theme))
    except Exception:
        logger.exception("Error building tier performance")
        parts.append('<p class="placeholder">Error generating tier performance table.</p>')

    try:
        parts.append(_build_reconciliation_summary(qa_data, theme))
    except Exception:
        logger.exception("Error building reconciliation summary")
        parts.append('<p class="placeholder">Error generating reconciliation summary.</p>')

    return "\n".join(parts)
