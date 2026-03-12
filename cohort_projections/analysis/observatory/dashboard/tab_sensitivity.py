"""Sensitivity tab — parameter sensitivity, tornado charts, and recommendations.

Provides sensitivity analysis, parameter response plots, experiment
recommendations from the recommender, persistent weakness alerts, and
residual diagnostics.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd
import panel as pn
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from cohort_projections.analysis.observatory.dashboard.data_manager import (
    DashboardDataManager,
)
from cohort_projections.analysis.observatory.dashboard.theme import (
    EXPERIMENT_COLORS,
    ORANGE,
    SDC_BLUE,
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

# Threshold values for residual diagnostic highlighting.
_AUTOCORR_THRESHOLD = 0.3
_HET_R2_THRESHOLD = 0.3
_SHAPIRO_ALPHA = 0.05


# ---------------------------------------------------------------------------
# Section 1: Tornado Chart
# ---------------------------------------------------------------------------


def _build_tornado_chart(dm: DashboardDataManager) -> pn.pane.Plotly:
    """Build a horizontal tornado chart showing parameter sensitivity.

    Each parameter has opposing bars for low/high perturbation effect.
    Sorted by total swing (largest impact at top).

    Parameters
    ----------
    dm:
        Data manager providing ``sensitivity_summary``.

    Returns
    -------
    pn.pane.Plotly
        Tornado chart, or placeholder if no data.
    """
    ss = dm.sensitivity_summary
    if ss.empty:
        return empty_placeholder("No sensitivity summary data available.")

    # Determine the swing column
    swing_col = (
        "mape_swing"
        if "mape_swing" in ss.columns
        else "swing_state_error"
        if "swing_state_error" in ss.columns
        else None
    )

    # If no dedicated swing column, compute from parameter-level aggregation
    # using county_mape_overall or similar metric columns.
    metric_col = (
        "county_mape_overall"
        if "county_mape_overall" in ss.columns
        else "mape"
        if "mape" in ss.columns
        else None
    )
    param_col = (
        "parameter"
        if "parameter" in ss.columns
        else "param"
        if "param" in ss.columns
        else None
    )

    if param_col is None:
        return empty_placeholder(
            "Sensitivity summary missing 'parameter' column."
        )

    # Try to build tornado data from per-parameter low/high values
    if swing_col is not None and param_col in ss.columns:
        # Aggregate by parameter: total swing
        param_swings = (
            ss.groupby(param_col)[swing_col]
            .sum()
            .sort_values(ascending=True)
        )
        if param_swings.empty:
            return empty_placeholder("No parameter swing data to display.")

        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                y=param_swings.index.tolist(),
                x=param_swings.values.tolist(),
                orientation="h",
                marker_color=SDC_BLUE,
                name="Total Swing",
                hovertemplate=(
                    "Parameter: %{y}<br>"
                    "Swing: %{x:.4f}<extra></extra>"
                ),
            )
        )
    elif metric_col is not None:
        # Compute from raw sensitivity data: for each parameter, find min and
        # max metric values and compute the swing.
        records: list[dict[str, Any]] = []
        for param, grp in ss.groupby(param_col):
            vals = grp[metric_col].dropna()
            if len(vals) < 2:
                continue
            low = float(vals.min())
            high = float(vals.max())
            center = float(vals.mean())
            records.append({
                "parameter": str(param),
                "low_effect": low - center,
                "high_effect": high - center,
                "total_swing": high - low,
            })

        if not records:
            return empty_placeholder(
                "Insufficient data to compute parameter sensitivity."
            )

        tornado_df = pd.DataFrame(records).sort_values(
            "total_swing", ascending=True
        )

        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                y=tornado_df["parameter"],
                x=tornado_df["low_effect"],
                orientation="h",
                marker_color=SDC_BLUE,
                name="Low perturbation",
                hovertemplate=(
                    "Parameter: %{y}<br>"
                    "Effect: %{x:+.4f}<extra></extra>"
                ),
            )
        )
        fig.add_trace(
            go.Bar(
                y=tornado_df["parameter"],
                x=tornado_df["high_effect"],
                orientation="h",
                marker_color=SDC_RED,
                name="High perturbation",
                hovertemplate=(
                    "Parameter: %{y}<br>"
                    "Effect: %{x:+.4f}<extra></extra>"
                ),
            )
        )
        fig.update_layout(barmode="relative")
    else:
        return empty_placeholder(
            "Cannot build tornado chart — missing metric or swing columns."
        )

    # Add center axis line
    fig.add_vline(x=0, line_width=1, line_color="#595959")

    layout_defaults = get_plotly_layout_defaults()
    fig.update_layout(
        **layout_defaults,
        title="Parameter Sensitivity (Tornado Chart)",
        xaxis_title="Effect on Metric",
        yaxis_title="",
        height=max(350, len(fig.data[0].y) * 35 + 100) if fig.data else 400,
    )

    return pn.pane.Plotly(fig, sizing_mode="stretch_width")


# ---------------------------------------------------------------------------
# Section 2: Parameter Response Plot
# ---------------------------------------------------------------------------


def _build_parameter_response(dm: DashboardDataManager) -> pn.pane.Plotly:
    """Build parameter response subplots: value (x) vs metric delta (y).

    For each parameter that has been varied, shows how changes in the
    parameter value affect the overall MAPE delta. Includes a trend line
    when 3+ data points exist.

    Parameters
    ----------
    dm:
        Data manager providing ``recommender.parameter_sensitivity_summary()``.

    Returns
    -------
    pn.pane.Plotly
        Subplot figure with one panel per parameter.
    """
    try:
        sensitivity = dm.recommender.parameter_sensitivity_summary()
    except Exception:
        logger.exception("Failed to get parameter sensitivity summary.")
        sensitivity = pd.DataFrame()

    if sensitivity.empty:
        return empty_placeholder(
            "No parameter sensitivity data available for response plots."
        )

    param_col = "parameter" if "parameter" in sensitivity.columns else None
    value_col = "value" if "value" in sensitivity.columns else None
    delta_col = (
        "county_mape_overall_delta"
        if "county_mape_overall_delta" in sensitivity.columns
        else None
    )

    if param_col is None or value_col is None or delta_col is None:
        return empty_placeholder(
            "Sensitivity data missing required columns "
            "(parameter, value, county_mape_overall_delta)."
        )

    # Filter to numeric values only
    def _is_numeric(v: Any) -> bool:
        if isinstance(v, bool):
            return False
        try:
            float(v)
            return True
        except (TypeError, ValueError):
            return False

    numeric_mask = sensitivity[value_col].apply(_is_numeric)
    sensitivity = sensitivity[numeric_mask].copy()
    sensitivity["_value_f"] = sensitivity[value_col].apply(float)

    if sensitivity.empty:
        return empty_placeholder(
            "No numeric parameter values found for response plots."
        )

    params = sorted(sensitivity[param_col].unique())
    n_params = len(params)

    if n_params == 0:
        return empty_placeholder("No varied parameters to plot.")

    n_cols = min(n_params, 3)
    n_rows = (n_params + n_cols - 1) // n_cols

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=params,
        horizontal_spacing=0.08,
        vertical_spacing=0.12,
    )

    for idx, param in enumerate(params):
        row = idx // n_cols + 1
        col = idx % n_cols + 1
        grp = sensitivity[sensitivity[param_col] == param].sort_values("_value_f")

        if grp.empty:
            continue

        color = EXPERIMENT_COLORS[idx % len(EXPERIMENT_COLORS)]

        # Scatter points
        fig.add_trace(
            go.Scatter(
                x=grp["_value_f"],
                y=grp[delta_col],
                mode="markers",
                marker={"color": color, "size": 8},
                name=param,
                showlegend=False,
                hovertemplate=(
                    f"{param}=%{{x}}<br>"
                    f"Delta: %{{y:+.4f}}pp<extra></extra>"
                ),
            ),
            row=row,
            col=col,
        )

        # Trend line if 3+ points
        if len(grp) >= 3:
            import numpy as np

            xs = grp["_value_f"].values
            ys = grp[delta_col].dropna().values
            if len(xs) == len(ys) and len(xs) >= 3:
                coeffs = np.polyfit(xs, ys, 1)
                x_range = np.linspace(xs.min(), xs.max(), 50)
                y_trend = np.polyval(coeffs, x_range)
                fig.add_trace(
                    go.Scatter(
                        x=x_range,
                        y=y_trend,
                        mode="lines",
                        line={"color": color, "dash": "dash", "width": 1.5},
                        showlegend=False,
                        hoverinfo="skip",
                    ),
                    row=row,
                    col=col,
                )

        # Zero reference line
        fig.add_hline(
            y=0,
            line_dash="dot",
            line_color="#A0A0A0",
            line_width=1,
            row=row,
            col=col,
        )

    layout_defaults = get_plotly_layout_defaults()
    fig.update_layout(
        **{
            k: v
            for k, v in layout_defaults.items()
            if k not in ("xaxis", "yaxis")
        },
        title_text="Parameter Response Curves",
        height=max(350, n_rows * 300),
    )

    return pn.pane.Plotly(fig, sizing_mode="stretch_width")


# ---------------------------------------------------------------------------
# Section 3: Recommendations Table
# ---------------------------------------------------------------------------


def _build_recommendations_table(dm: DashboardDataManager) -> pn.Column:
    """Build the recommendations table from the recommender.

    Calls ``dm.recommender.suggest_next_experiments()`` and displays
    results as a styled Tabulator, color-coded by priority.

    Parameters
    ----------
    dm:
        Data manager providing ``recommender``.

    Returns
    -------
    pn.Column
        Tabulator table of recommendations, or placeholder.
    """
    try:
        recommendations = dm.recommender.suggest_next_experiments()
    except Exception:
        logger.exception("Failed to generate experiment recommendations.")
        return pn.Column(
            empty_placeholder("Unable to generate recommendations.")
        )

    if not recommendations:
        return pn.Column(
            empty_placeholder(
                "No experiment recommendations — insufficient data or "
                "all promising directions have been explored."
            )
        )

    records = [
        {
            "priority": rec.priority,
            "parameter": rec.parameter,
            "suggested_value": str(rec.suggested_value),
            "direction": rec.direction,
            "rationale": rec.rationale,
            "expected_impact": rec.expected_impact,
            "requires_code_change": rec.requires_code_change,
        }
        for rec in recommendations
    ]

    rec_df = pd.DataFrame(records)

    # Sort by priority
    rec_df = rec_df.sort_values("priority").reset_index(drop=True)

    return metric_table(
        rec_df,
        title="Experiment Recommendations",
        page_size=0,
    )


# ---------------------------------------------------------------------------
# Section 4: Persistent Weaknesses
# ---------------------------------------------------------------------------


def _build_weakness_cards(dm: DashboardDataManager) -> pn.Column:
    """Build alert cards for persistent weaknesses.

    Each weakness where no challenger improves over the champion is
    shown as an amber/red styled card.

    Parameters
    ----------
    dm:
        Data manager providing ``recommender``.

    Returns
    -------
    pn.Column
        Column of alert cards, or a success message.
    """
    try:
        weaknesses = dm.recommender.identify_persistent_weaknesses()
    except Exception:
        logger.exception("Failed to identify persistent weaknesses.")
        return pn.Column(
            empty_placeholder("Unable to compute persistent weaknesses.")
        )

    if weaknesses.empty:
        return pn.Column(
            pn.pane.Alert(
                "No persistent weaknesses detected — all metrics have at "
                "least one challenger improving over the champion.",
                alert_type="success",
            )
        )

    # Filter to actual weaknesses (delta >= 0 means no improvement)
    persistent = weaknesses[
        weaknesses["best_challenger_delta"].notna()
        & (weaknesses["best_challenger_delta"] >= 0)
    ]

    if persistent.empty:
        return pn.Column(
            pn.pane.Alert(
                "All metrics have at least one variant improving over the champion.",
                alert_type="success",
            )
        )

    cards: list[pn.pane.HTML] = []
    for _, row in persistent.iterrows():
        metric = row["metric"]
        champ_val = row.get("champion_value")
        best_delta = row.get("best_challenger_delta")
        best_run = row.get("best_challenger_run", "")

        champ_str = f"{champ_val:.4f}" if pd.notna(champ_val) else "N/A"
        delta_str = f"{best_delta:+.4f}" if pd.notna(best_delta) else "N/A"

        # Determine severity color
        severity_color = (
            SDC_RED if (pd.notna(best_delta) and best_delta > 0.1) else ORANGE
        )

        html = f"""
        <div style="
            border-left: 4px solid {severity_color};
            background: #FFF8F0;
            padding: 12px 16px;
            margin-bottom: 8px;
            border-radius: 0 6px 6px 0;
            font-family: 'Segoe UI', Roboto, Arial, sans-serif;
        ">
            <div style="font-weight:700; color:{SDC_NAVY}; margin-bottom:4px">
                {metric}
            </div>
            <div style="color:#595959; font-size:0.9em">
                Champion value: <strong>{champ_str}</strong>
                &nbsp;|&nbsp;
                Best challenger delta: <span style="color:{severity_color};
                font-weight:600">{delta_str}</span>
                {f'&nbsp;|&nbsp; Best run: {best_run}' if best_run else ''}
            </div>
            <div style="color:#595959; font-size:0.85em; margin-top:4px">
                No tested variant improves this metric. Consider designing
                experiments specifically targeting this area.
            </div>
        </div>
        """
        cards.append(pn.pane.HTML(html, sizing_mode="stretch_width"))

    return pn.Column(*cards, sizing_mode="stretch_width")


# ---------------------------------------------------------------------------
# Section 5: Residual Diagnostics
# ---------------------------------------------------------------------------


def _build_residual_diagnostics(dm: DashboardDataManager) -> pn.Column:
    """Build residual diagnostics table with violation highlighting.

    Displays horizon-bucketed residual statistics and highlights cells
    indicating potential violations (high autocorrelation, non-normality,
    high heteroscedasticity).

    Parameters
    ----------
    dm:
        Data manager providing ``residual_diagnostics``.

    Returns
    -------
    pn.Column
        Styled diagnostics table, or placeholder.
    """
    rd = dm.residual_diagnostics
    if rd.empty:
        return pn.Column(
            empty_placeholder("No residual diagnostics data available.")
        )

    # Select display columns (flexible)
    desired_cols = [
        "run_id",
        "horizon_bucket",
        "mean_autocorr_lag1",
        "het_r2",
        "shapiro_w",
        "normal_at_05",
        "error_skew",
        "error_kurtosis",
    ]
    display_cols = [c for c in desired_cols if c in rd.columns]
    if not display_cols:
        return pn.Column(
            empty_placeholder(
                "Residual diagnostics table has no recognized columns."
            )
        )

    display_df = rd[display_cols].copy()

    # Highlight columns where violations are possible
    highlight_cols = [
        c
        for c in ["mean_autocorr_lag1", "het_r2"]
        if c in display_df.columns
    ]

    return metric_table(
        display_df,
        title="Residual Diagnostics by Horizon",
        highlight_cols=highlight_cols,
        page_size=20,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_sensitivity_tab(dm: DashboardDataManager) -> pn.Column:
    """Build the Sensitivity tab for the Observatory dashboard.

    Contains five sections: tornado chart, parameter response plots,
    experiment recommendations, persistent weaknesses, and residual
    diagnostics.

    Parameters
    ----------
    dm:
        The :class:`DashboardDataManager` providing all data access.

    Returns
    -------
    pn.Column
        A Panel Column containing all sensitivity analysis sections.
    """
    return pn.Column(
        section_header(
            "Sensitivity & Recommendations",
            subtitle="Parameter exploration, sensitivity analysis, and next-experiment guidance",
        ),
        # Section 1: Tornado Chart
        pn.layout.Divider(),
        section_header(
            "Parameter Sensitivity",
            subtitle="Horizontal bars show the effect of each parameter on projection accuracy",
        ),
        _build_tornado_chart(dm),
        # Section 2: Parameter Response Plot
        pn.layout.Divider(),
        section_header(
            "Parameter Response Curves",
            subtitle="How each varied parameter affects the overall MAPE delta",
        ),
        _build_parameter_response(dm),
        # Section 3: Recommendations Table
        pn.layout.Divider(),
        section_header(
            "Experiment Recommendations",
            subtitle="Automated suggestions for the next experiments to run",
        ),
        _build_recommendations_table(dm),
        # Section 4: Persistent Weaknesses
        pn.layout.Divider(),
        section_header(
            "Persistent Weaknesses",
            subtitle="Metrics where no tested variant improves over the champion",
        ),
        _build_weakness_cards(dm),
        # Section 5: Residual Diagnostics
        pn.layout.Divider(),
        section_header(
            "Residual Diagnostics",
            subtitle="Statistical health checks on projection residuals by horizon",
        ),
        _build_residual_diagnostics(dm),
        sizing_mode="stretch_width",
    )
