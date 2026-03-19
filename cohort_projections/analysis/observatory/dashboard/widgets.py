"""Reusable widget factories for the Observatory Panel dashboard.

Provides styled KPI cards, status badges, metric tables, section headers,
workflow steppers, progress rings, and other building blocks consumed by the
dashboard tabs.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import panel as pn

from cohort_projections.analysis.observatory.dashboard.theme import (
    DASHBOARD_CSS,
    GROWTH_GREEN,
    SDC_BLUE,
    SDC_DARK_GRAY,
    SDC_NAVY,
    SDC_RED,
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
        delta_html = f'<div class="kpi-delta {css_class}">{arrow} {delta:+.2f}</div>'

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
    priority_columns: list[str] | None = None,
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
    priority_columns:
        When provided, only these columns are initially visible.  Other
        columns are hidden but can be re-enabled via the Tabulator column
        visibility controls.
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

    # Build hidden_columns from priority_columns if specified.
    hidden_columns: list[str] = []
    if priority_columns:
        priority_set = set(priority_columns)
        frozen_set = set(resolved_frozen_columns or [])
        hidden_columns = [
            col for col in df.columns if col not in priority_set and col not in frozen_set
        ]

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
    if hidden_columns:
        tabulator_kwargs["hidden_columns"] = hidden_columns
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


def section_header(
    title: str,
    subtitle: str = "",
    tooltip: str = "",
) -> pn.pane.HTML:
    """Render a styled section header with an optional subtitle and tooltip.

    Parameters
    ----------
    title:
        Primary heading text.
    subtitle:
        Optional secondary description text.
    tooltip:
        Optional help text shown in a hover tooltip next to the title.
    """
    subtitle_html = ""
    if subtitle:
        subtitle_html = f'<p class="subtitle">{subtitle}</p>'
    tooltip_html = ""
    if tooltip:
        escaped = tooltip.replace('"', "&quot;").replace("<", "&lt;").replace(">", "&gt;")
        tooltip_html = (
            f'<span class="obs-tooltip">i<span class="obs-tooltip-text">{escaped}</span></span>'
        )
    html = f'<div class="section-header">  <h2>{title}{tooltip_html}</h2>  {subtitle_html}</div>'
    return pn.pane.HTML(
        html,
        sizing_mode="stretch_width",
        stylesheets=[DASHBOARD_CSS],
    )


def markdown_card(
    title: str,
    body: str,
    *,
    collapsed: bool = False,
    min_width: int | None = None,
) -> pn.Card:
    """Render a simple Markdown-based card with shared dashboard styling.

    Parameters
    ----------
    title:
        Card title shown in the header.
    body:
        Markdown body content.
    collapsed:
        Whether the card should start collapsed.
    min_width:
        Optional minimum width to help side-by-side dashboard layouts.
    """
    return pn.Card(
        pn.pane.Markdown(body, sizing_mode="stretch_width"),
        title=title,
        collapsed=collapsed,
        sizing_mode="stretch_width",
        min_width=min_width,
    )


# ---------------------------------------------------------------------------
# Workflow Stepper
# ---------------------------------------------------------------------------


def workflow_stepper(
    steps: list[str],
    active: int = 0,
    completed: list[int] | None = None,
) -> pn.pane.HTML:
    """Render a horizontal step-progress bar.

    Parameters
    ----------
    steps:
        Labels for each step (e.g. ``["Launch", "Monitor", "Review", "Decide"]``).
    active:
        Zero-based index of the currently active step.
    completed:
        Zero-based indices of completed steps.
    """
    completed_set = set(completed or [])
    parts: list[str] = []
    for i, label in enumerate(steps):
        if i in completed_set:
            css = "obs-step completed"
            num = "&#10003;"  # checkmark
        elif i == active:
            css = "obs-step active"
            num = str(i + 1)
        else:
            css = "obs-step"
            num = str(i + 1)
        parts.append(
            f'<div class="{css}"><span class="obs-step-num">{num}</span><span>{label}</span></div>'
        )
        if i < len(steps) - 1:
            conn_css = (
                "obs-step-connector completed" if i in completed_set else "obs-step-connector"
            )
            parts.append(f'<div class="{conn_css}"></div>')

    html = f'<div class="obs-workflow-stepper">{"".join(parts)}</div>'
    return pn.pane.HTML(html, sizing_mode="stretch_width", stylesheets=[DASHBOARD_CSS])


# ---------------------------------------------------------------------------
# Progress Ring
# ---------------------------------------------------------------------------


def progress_ring(
    pct: float,
    label: str = "",
    status: str = "running",
) -> pn.pane.HTML:
    """Render a circular progress indicator.

    Parameters
    ----------
    pct:
        Progress percentage (0–100).
    label:
        Compact text shown inside the ring (e.g. ``"12/20"``).
    status:
        One of ``"running"``, ``"complete"``, ``"mixed"``, ``"failed"``.
        Controls the fill color and animation.
    """
    color_map = {
        "running": SDC_BLUE,
        "complete": GROWTH_GREEN,
        "mixed": "#FFC000",
        "failed": SDC_RED,
    }
    color = color_map.get(status, SDC_BLUE)
    anim_class = " running" if status == "running" else ""
    pct_clamped = max(0.0, min(100.0, pct))
    pct_display = f"{pct_clamped:.0f}%"

    html = (
        f'<div class="obs-progress-ring{anim_class}" '
        f'style="--progress:{pct_clamped};--ring-color:{color}">'
        f'<div class="obs-ring-label">'
        f'<span class="obs-ring-pct">{pct_display}</span>'
        f'<span class="obs-ring-text">{label}</span>'
        f"</div></div>"
    )
    return pn.pane.HTML(html, width=130, height=130, stylesheets=[DASHBOARD_CSS])


# ---------------------------------------------------------------------------
# Candidate Feed
# ---------------------------------------------------------------------------


def candidate_feed(
    candidates: pd.DataFrame,
    max_items: int = 5,
) -> pn.pane.HTML:
    """Render a compact live feed of recently completed experiment candidates.

    Parameters
    ----------
    candidates:
        DataFrame with columns ``candidate_id``, ``outcome``,
        ``county_mape_overall``, ``delta_county_mape_overall``.
    max_items:
        Maximum number of items to display.
    """
    if candidates.empty:
        return pn.pane.HTML(
            '<div class="obs-candidate-feed">'
            '<div style="padding:12px;color:#A0A0A0;font-size:0.88em;text-align:center">'
            "Waiting for results&hellip;</div></div>",
            sizing_mode="stretch_width",
            stylesheets=[DASHBOARD_CSS],
        )

    items: list[str] = []
    for _, row in candidates.head(max_items).iterrows():
        name = str(row.get("candidate_id", "—"))
        outcome = str(row.get("outcome", ""))
        badge_map = {
            "passed_all_gates": ("&#10003;", "#00B050"),
            "needs_human_review": ("?", "#FFC000"),
            "failed_hard_gate": ("&#10007;", "#C00000"),
        }
        icon, dot_color = badge_map.get(outcome, ("&#8226;", "#A0A0A0"))

        mape_val = row.get("county_mape_overall")
        mape_str = f"{mape_val:.2f}%" if pd.notna(mape_val) else "—"

        delta_val = row.get("delta_county_mape_overall")
        if pd.notna(delta_val):
            delta_class = "improved" if float(delta_val) < 0 else "regressed"
            delta_str = f"{float(delta_val):+.2f}"
        else:
            delta_class = ""
            delta_str = ""

        items.append(
            f'<div class="obs-candidate-feed-item">'
            f'<span style="color:{dot_color};font-size:1.1em">{icon}</span>'
            f'<span class="obs-cf-name">{name}</span>'
            f'<span class="obs-cf-metric">{mape_str}</span>'
            f'<span class="obs-cf-delta {delta_class}">{delta_str}</span>'
            f"</div>"
        )

    html = f'<div class="obs-candidate-feed">{"".join(items)}</div>'
    return pn.pane.HTML(html, sizing_mode="stretch_width", stylesheets=[DASHBOARD_CSS])


# ---------------------------------------------------------------------------
# Hero Metric
# ---------------------------------------------------------------------------


def hero_metric(
    value: str,
    label: str,
    delta: float | None = None,
    color: str | None = None,
) -> pn.pane.HTML:
    """Render a large primary metric display.

    Parameters
    ----------
    value:
        Formatted metric value string (e.g. ``"8.86%"``).
    label:
        Short descriptor (e.g. ``"Champion County Error"``).
    delta:
        Optional change value.  Negative = improvement (green).
    color:
        Accent color for the underline bar.
    """
    accent = color or SDC_BLUE
    delta_html = ""
    if delta is not None:
        if delta < 0:
            d_color = GROWTH_GREEN
            arrow = "&#9660;"
        elif delta > 0:
            d_color = SDC_RED
            arrow = "&#9650;"
        else:
            d_color = SDC_DARK_GRAY
            arrow = "&#8212;"
        delta_html = (
            f'<div class="obs-mh-delta" style="color:{d_color}">'
            f"{arrow} {delta:+.2f} vs best challenger</div>"
        )
    html = (
        f'<div class="obs-metric-hero">'
        f'<div class="obs-mh-value">{value}</div>'
        f'<div class="obs-mh-label">{label}</div>'
        f"{delta_html}"
        f'<div class="obs-mh-underline" style="background:{accent}"></div>'
        f"</div>"
    )
    return pn.pane.HTML(html, sizing_mode="stretch_width", stylesheets=[DASHBOARD_CSS])


# ---------------------------------------------------------------------------
# Filter Bar
# ---------------------------------------------------------------------------


def filter_bar(*widgets: Any) -> pn.Row:
    """Wrap filter widgets in a standardized horizontal bar.

    Parameters
    ----------
    *widgets:
        Panel widgets to arrange horizontally.
    """
    return pn.Row(
        *widgets,
        css_classes=["obs-filter-bar"],
        sizing_mode="stretch_width",
        stylesheets=[DASHBOARD_CSS],
    )


# ---------------------------------------------------------------------------
# Info Tooltip
# ---------------------------------------------------------------------------


def info_tooltip(text: str) -> pn.pane.HTML:
    """Render a small ``(i)`` icon with a CSS-only hover tooltip.

    Parameters
    ----------
    text:
        Help text shown on hover.
    """
    escaped = text.replace('"', "&quot;").replace("<", "&lt;").replace(">", "&gt;")
    html = f'<span class="obs-tooltip">i<span class="obs-tooltip-text">{escaped}</span></span>'
    return pn.pane.HTML(html, width=26, stylesheets=[DASHBOARD_CSS])


# ---------------------------------------------------------------------------
# Completion Banner
# ---------------------------------------------------------------------------


def completion_banner(
    total: int,
    best_name: str = "",
    best_delta: float = 0.0,
    status: str = "success",
) -> pn.pane.HTML:
    """Render a post-completion status banner with next-step guidance.

    Parameters
    ----------
    total:
        Number of experiments that finished.
    best_name:
        Display name of the best candidate.
    best_delta:
        MAPE delta of the best candidate vs champion.
    status:
        One of ``"success"``, ``"mixed"``, ``"failed"``.
    """
    icons = {"success": "&#10004;", "mixed": "&#9888;", "failed": "&#10008;"}
    titles = {
        "success": f"Search Complete &mdash; {total} experiment(s) finished",
        "mixed": f"Search finished with errors &mdash; {total} experiment(s) ran",
        "failed": f"All {total} experiment(s) failed",
    }
    icon = icons.get(status, icons["success"])
    title = titles.get(status, titles["success"])

    detail = ""
    if status == "success" and best_name:
        direction = "improved" if best_delta < 0 else "changed"
        detail = (
            f"Best candidate: <strong>{best_name}</strong> "
            f"(MAPE {direction} by {abs(best_delta):.2f}pp)"
        )
    elif status == "mixed":
        detail = (
            "Some experiments succeeded. Review the candidates table below, "
            "then expand the Log Output card to diagnose failures."
        )
    elif status == "failed":
        detail = "Expand the Log Output card below to see error details."

    # Build next-steps guidance based on status
    if status == "failed":
        next_steps = (
            "<strong>Next steps:</strong>"
            "<ol>"
            "<li>Expand <em>Log Output</em> below to identify failures</li>"
            "<li>Fix the underlying issue, then click <em>Start Exploring</em> again</li>"
            "</ol>"
        )
    elif status == "mixed":
        next_steps = (
            "<strong>Next steps:</strong>"
            "<ol>"
            "<li>Review the <em>candidates table</em> below for successful results</li>"
            "<li>Expand <em>Log Output</em> to diagnose any failures</li>"
            "<li>Open the <strong>Decision Brief</strong> tab to see what is usable, blocked, or inconclusive</li>"
            "<li>Open the <strong>Experiment History</strong> tab to compare results over time</li>"
            "</ol>"
        )
    else:
        next_steps = (
            "<strong>Next steps:</strong>"
            "<ol>"
            "<li>Review the <em>candidates table</em> below to see all results</li>"
            "<li>Click <em>Review Results</em> to start the guided review flow</li>"
            "<li>Open <strong>Scorecards</strong> for detailed comparisons</li>"
            "<li>Open <strong>Experiment History</strong> to compare results over time</li>"
            "<li>Open <strong>Projections</strong> to visualize population curves</li>"
            "<li>Click <em>Start Exploring</em> to search further with different parameters</li>"
            "</ol>"
        )

    html = (
        f'<div class="obs-completion-banner {status}">'
        f'<div class="obs-cb-icon">{icon}</div>'
        f'<div class="obs-cb-title">{title}</div>'
        f'<div class="obs-cb-detail">{detail}</div>'
        f'<div class="obs-cb-next-steps">{next_steps}</div>'
        f"</div>"
    )
    return pn.pane.HTML(html, sizing_mode="stretch_width", stylesheets=[DASHBOARD_CSS])


# ---------------------------------------------------------------------------
# Illustrated Empty State
# ---------------------------------------------------------------------------

_EMPTY_STATE_SVGS: dict[str, str] = {
    "search": (
        '<svg width="64" height="64" viewBox="0 0 64 64" fill="none">'
        '<circle cx="28" cy="28" r="18" stroke="#0563C1" stroke-width="3" fill="#E9F1FC"/>'
        '<line x1="41" y1="41" x2="56" y2="56" stroke="#0563C1" stroke-width="3" '
        'stroke-linecap="round"/>'
        '<circle cx="28" cy="28" r="8" stroke="#0563C1" stroke-width="2" fill="none" '
        'stroke-dasharray="4 3"/>'
        "</svg>"
    ),
    "rocket": (
        '<svg width="64" height="64" viewBox="0 0 64 64" fill="none">'
        '<path d="M32 8 C32 8 44 20 44 36 L36 44 L28 44 L20 36 C20 20 32 8 32 8Z" '
        'fill="#E9F1FC" stroke="#0563C1" stroke-width="2"/>'
        '<circle cx="32" cy="28" r="4" fill="#0563C1"/>'
        '<path d="M26 44 L22 56 L28 48" stroke="#00B050" stroke-width="2" fill="none"/>'
        '<path d="M38 44 L42 56 L36 48" stroke="#00B050" stroke-width="2" fill="none"/>'
        "</svg>"
    ),
    "check": (
        '<svg width="64" height="64" viewBox="0 0 64 64" fill="none">'
        '<circle cx="32" cy="32" r="24" fill="#E2F3E8" stroke="#00B050" stroke-width="2"/>'
        '<path d="M22 32 L29 39 L42 25" stroke="#00B050" stroke-width="3" fill="none" '
        'stroke-linecap="round" stroke-linejoin="round"/>'
        "</svg>"
    ),
}


def illustrated_empty_state(
    message: str,
    illustration: str = "search",
) -> pn.pane.HTML:
    """Render an empty-state placeholder with a simple inline SVG illustration.

    Parameters
    ----------
    message:
        Descriptive text shown below the illustration.
    illustration:
        One of ``"search"``, ``"rocket"``, ``"check"``.
    """
    svg = _EMPTY_STATE_SVGS.get(illustration, _EMPTY_STATE_SVGS["search"])
    html = f'<div class="obs-empty-state">{svg}<div class="obs-es-message">{message}</div></div>'
    return pn.pane.HTML(html, sizing_mode="stretch_width", stylesheets=[DASHBOARD_CSS])


# ---------------------------------------------------------------------------
# Terminal Output
# ---------------------------------------------------------------------------


def terminal_output(text: str, max_height: int = 200) -> pn.pane.HTML:
    """Render log output in a dark-themed terminal-style pane.

    Parameters
    ----------
    text:
        Raw text content (typically log output).
    max_height:
        Maximum height in pixels before scrolling.
    """
    escaped = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    html = f'<pre class="obs-terminal" style="max-height:{max_height}px">{escaped}</pre>'
    return pn.pane.HTML(html, sizing_mode="stretch_width", stylesheets=[DASHBOARD_CSS])


# ---------------------------------------------------------------------------
# Next Step Bar (guided review)
# ---------------------------------------------------------------------------


def build_review_step_bar(
    dm_selection_state: Any,
    tabs: Any,
    current_step: int,
    total_steps: int,
    next_tab_index: int | None,
    next_tab_label: str = "",
) -> pn.Column:
    """Build a conditional "Next Step" navigation bar for guided review mode.

    Only renders when ``dm_selection_state.review_mode`` is True. Otherwise
    returns an empty Column.

    Parameters
    ----------
    dm_selection_state:
        The ``DashboardSelectionState`` instance.
    tabs:
        The root ``pn.Tabs`` widget (used to navigate).
    current_step:
        1-based step number for this tab (e.g. 1 of 3).
    total_steps:
        Total number of review steps.
    next_tab_index:
        Tab index to navigate to on click.  ``None`` for the final step.
    next_tab_label:
        Label for the next tab button text.
    """
    if tabs is None:
        return pn.Column()

    container = pn.Column(sizing_mode="stretch_width")

    def _maybe_render() -> pn.pane.HTML | None:
        if not getattr(dm_selection_state, "review_mode", False):
            return None

        badge = f'<span class="obs-step-badge">Step {current_step} of {total_steps}</span>'
        if next_tab_index is not None:
            btn_label = f"Next: {next_tab_label} &rarr;"
        else:
            btn_label = "Complete Review &#10003;"

        return pn.pane.HTML(
            f'<div class="obs-next-step-bar">'
            f"{badge}"
            f'<span class="obs-nsb-label" style="flex:1"></span>'
            f'<span class="obs-btn primary" id="obs-next-step">{btn_label}</span>'
            f"</div>",
            sizing_mode="stretch_width",
            stylesheets=[DASHBOARD_CSS],
        )

    bar_pane = _maybe_render()
    if bar_pane is not None:
        # We need a button for actual navigation since HTML spans can't trigger
        # Panel callbacks. Use a real button styled to match.
        if next_tab_index is not None:
            btn = pn.widgets.Button(
                name=f"Next: {next_tab_label}",
                button_type="primary",
                width=200,
                height=38,
            )
            btn.on_click(lambda e: setattr(tabs, "active", next_tab_index))
        else:
            btn = pn.widgets.Button(
                name="Complete Review",
                button_type="success",
                width=200,
                height=38,
            )

            def _complete(e: Any) -> None:
                dm_selection_state.review_mode = False
                dm_selection_state.review_step = 0

            btn.on_click(_complete)

        step_badge = pn.pane.HTML(
            f'<span class="obs-step-badge">Step {current_step} of {total_steps}</span>',
            stylesheets=[DASHBOARD_CSS],
        )
        container.append(
            pn.Row(
                step_badge,
                pn.Spacer(sizing_mode="stretch_width"),
                btn,
                sizing_mode="stretch_width",
                css_classes=["obs-next-step-bar"],
                stylesheets=[DASHBOARD_CSS],
            )
        )

    return container
