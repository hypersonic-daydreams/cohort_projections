"""Reusable widget factories for the Observatory Panel dashboard.

Provides styled KPI cards, status badges, metric tables, section headers,
and other building blocks consumed by the dashboard tabs.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import panel as pn

from cohort_projections.analysis.observatory.dashboard.theme import (
    DASHBOARD_CSS,
    SDC_DARK_GRAY,
    SDC_NAVY,
    TABULATOR_STYLESHEET,
)

# ---------------------------------------------------------------------------
# KPI Card
# ---------------------------------------------------------------------------


def kpi_card(
    title: str,
    value: str | int | float,
    delta: float | None = None,
    color: str | None = None,
) -> pn.pane.HTML:
    """Render a styled KPI card with a large value, label, and optional delta.

    Parameters
    ----------
    title:
        Short label displayed below the number.
    value:
        The primary metric value.  Formatted as-is if a string; otherwise
        displayed with reasonable numeric formatting.
    delta:
        Optional change value.  Negative deltas (improvements for error
        metrics) are shown in green; positive in red.
    color:
        Override the value text color.  Defaults to SDC Navy.
    """
    value_color = color or SDC_NAVY
    if isinstance(value, float):
        value_str = f"{value:.2f}"
    elif isinstance(value, int):
        value_str = f"{value:,}"
    else:
        value_str = str(value)

    delta_html = ""
    if delta is not None:
        if delta < 0:
            arrow = "&#9660;"  # down arrow — improvement
            css_class = "negative"
        elif delta > 0:
            arrow = "&#9650;"  # up arrow — regression
            css_class = "positive"
        else:
            arrow = "&#8212;"  # em-dash — no change
            css_class = "neutral"
        delta_html = (
            f'<div class="kpi-delta {css_class}">'
            f"{arrow} {delta:+.2f}"
            f"</div>"
        )

    html = (
        f'<div class="kpi-card">'
        f'  <div class="kpi-value" style="color:{value_color}">{value_str}</div>'
        f'  <div class="kpi-label">{title}</div>'
        f"  {delta_html}"
        f"</div>"
    )
    return pn.pane.HTML(
        html,
        width=180,
        min_width=180,
        stylesheets=[DASHBOARD_CSS],
    )


# ---------------------------------------------------------------------------
# Status Badge
# ---------------------------------------------------------------------------

_STATUS_DISPLAY: dict[str, tuple[str, str]] = {
    "passed_all_gates": ("PASSED", "badge-passed"),
    "needs_human_review": ("REVIEW", "badge-review"),
    "failed_hard_gate": ("FAILED", "badge-failed"),
    "untested": ("UNTESTED", "badge-untested"),
    "champion": ("CHAMPION", "badge-champion"),
}


def status_badge(status: str) -> pn.pane.HTML:
    """Render a colored status badge.

    Parameters
    ----------
    status:
        One of ``passed_all_gates``, ``needs_human_review``,
        ``failed_hard_gate``, ``untested``, ``champion``.
        Unknown statuses render as gray.
    """
    label, css_class = _STATUS_DISPLAY.get(
        status.lower().strip(),
        (status.upper(), "badge-untested"),
    )
    html = f'<span class="badge {css_class}">{label}</span>'
    return pn.pane.HTML(html, stylesheets=[DASHBOARD_CSS])


# ---------------------------------------------------------------------------
# Delta Formatter
# ---------------------------------------------------------------------------


def delta_formatter(
    value: float,
    threshold: float = 0,
    fmt: str = ".2f",
) -> str:
    """Return an HTML string coloring *value* green (improvement) or red.

    For error metrics a *negative* delta is an improvement (green).

    Parameters
    ----------
    value:
        The delta value to format.
    threshold:
        The boundary between improvement and regression.  Defaults to 0.
    fmt:
        Python format spec for the numeric display.
    """
    if value < threshold:
        color = "#00B050"  # green — improvement
    elif value > threshold:
        color = "#C00000"  # red — regression
    else:
        color = SDC_DARK_GRAY
    return f'<span style="color:{color};font-weight:600">{value:{fmt}}</span>'


# ---------------------------------------------------------------------------
# Metric Table (Tabulator)
# ---------------------------------------------------------------------------


def metric_table(
    df: pd.DataFrame,
    title: str = "",
    highlight_cols: list[str] | None = None,
    page_size: int = 15,
    frozen_columns: list[str] | None = None,
    formatters: dict[str, Any] | None = None,
) -> pn.Column:
    """Wrap a DataFrame in a styled :class:`pn.widgets.Tabulator`.

    Parameters
    ----------
    df:
        The data to display.
    title:
        Optional title rendered above the table.
    highlight_cols:
        Column names whose cells should receive a green/red background
        based on sign (negative = green, positive = red).
    page_size:
        Number of rows per page.  Set to 0 to disable pagination.
    """
    if df.empty:
        return pn.Column(empty_placeholder("No data available for this table."))

    table_formatters: dict[str, Any] = dict(formatters or {})
    if highlight_cols:
        for col in highlight_cols:
            if col in df.columns:
                table_formatters[col] = {
                    "type": "progress",
                    "min": float(df[col].min()) if not df[col].empty else -1,
                    "max": float(df[col].max()) if not df[col].empty else 1,
                    "color": ["#00B050", "#C00000"],
                }

    resolved_frozen_columns = frozen_columns
    if resolved_frozen_columns is None:
        candidates = [
            "run",
            "run_id",
            "variant_id",
            "county_name",
            "metric",
            "parameter",
        ]
        resolved_frozen_columns = [col for col in candidates if col in df.columns][:2]

    tabulator_kwargs: dict[str, Any] = {
        "sizing_mode": "stretch_width",
        "theme": "simple",
        "header_filters": True,
        "show_index": False,
        "layout": "fit_data_stretch",
        "stylesheets": [TABULATOR_STYLESHEET],
        "row_height": 34,
    }
    if table_formatters:
        tabulator_kwargs["formatters"] = table_formatters
    if resolved_frozen_columns:
        tabulator_kwargs["frozen_columns"] = resolved_frozen_columns
    if page_size > 0:
        tabulator_kwargs["pagination"] = "remote"
        tabulator_kwargs["page_size"] = page_size

    table = pn.widgets.Tabulator(df, **tabulator_kwargs)

    components: list[Any] = []
    if title:
        components.append(section_header(title))
    components.append(table)
    return pn.Column(*components, sizing_mode="stretch_width")


# ---------------------------------------------------------------------------
# Empty Placeholder
# ---------------------------------------------------------------------------


def empty_placeholder(message: str = "No data available.") -> pn.pane.HTML:
    """Render a styled empty-state placeholder.

    Parameters
    ----------
    message:
        Text to display in the empty state.
    """
    html = f'<div class="empty-placeholder">{message}</div>'
    return pn.pane.HTML(
        html,
        sizing_mode="stretch_width",
        stylesheets=[DASHBOARD_CSS],
    )


# ---------------------------------------------------------------------------
# Section Header
# ---------------------------------------------------------------------------


def section_header(title: str, subtitle: str = "") -> pn.pane.HTML:
    """Render a styled section header with an optional subtitle.

    Parameters
    ----------
    title:
        Primary heading text.
    subtitle:
        Optional secondary description text.
    """
    subtitle_html = ""
    if subtitle:
        subtitle_html = f'<p class="subtitle">{subtitle}</p>'
    html = (
        f'<div class="section-header">'
        f"  <h2>{title}</h2>"
        f"  {subtitle_html}"
        f"</div>"
    )
    return pn.pane.HTML(
        html,
        sizing_mode="stretch_width",
        stylesheets=[DASHBOARD_CSS],
    )
