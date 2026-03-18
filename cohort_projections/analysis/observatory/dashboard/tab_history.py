"""Longitudinal benchmark-history tab for the Projection Observatory."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pandas as pd
import panel as pn
import plotly.graph_objects as go

from cohort_projections.analysis.observatory.dashboard import theme, widgets

if TYPE_CHECKING:
    from cohort_projections.analysis.observatory.dashboard.data_manager import (
        DashboardDataManager,
    )

logger = logging.getLogger(__name__)

_METRIC_LABELS: dict[str, str] = {
    "county_mape_overall": "County Overall",
    "county_mape_rural": "County Rural",
    "county_mape_bakken": "County Bakken",
    "county_mape_urban_college": "County Urban/College",
    "state_ape_recent_short": "State Recent Short",
    "state_ape_recent_medium": "State Recent Medium",
}
_CATEGORY_METRICS: tuple[str, ...] = (
    "county_mape_overall",
    "county_mape_rural",
    "county_mape_bakken",
    "county_mape_urban_college",
)
_STATUS_ORDER: tuple[str, ...] = (
    "champion",
    "passed_all_gates",
    "needs_human_review",
    "failed_hard_gate",
    "untested",
)


def _selected_statuses(
    statuses: list[str] | tuple[str, ...] | None, dm: DashboardDataManager
) -> list[str]:
    """Normalize status filters to a concrete list."""
    if statuses:
        return [str(status) for status in statuses]
    history = dm.longitudinal_run_history
    if history.empty or "status_code" not in history.columns:
        return list(_STATUS_ORDER)
    discovered = [
        str(status) for status in history["status_code"].dropna().astype(str).unique().tolist()
    ]
    return [status for status in _STATUS_ORDER if status in discovered] or discovered


def _filter_history(
    dm: DashboardDataManager, statuses: list[str] | tuple[str, ...] | None
) -> pd.DataFrame:
    """Return the longitudinal run history filtered to statuses."""
    history = dm.longitudinal_run_history
    if history.empty:
        return history
    selected = _selected_statuses(statuses, dm)
    if not selected:
        return history
    return history[history["status_code"].isin(selected)].copy()


def _filter_delta_history(
    dm: DashboardDataManager, statuses: list[str] | tuple[str, ...] | None
) -> pd.DataFrame:
    """Return the long metric-delta history filtered to statuses."""
    delta_history = dm.metric_delta_history
    if delta_history.empty:
        return delta_history
    selected = _selected_statuses(statuses, dm)
    if not selected:
        return delta_history
    return delta_history[delta_history["status_code"].isin(selected)].copy()


def _history_takeaway_text(dm: DashboardDataManager) -> str:
    """Return a plain-language summary of the benchmark history trajectory."""
    history = dm.longitudinal_run_history
    if history.empty:
        return (
            "No benchmark history is available yet. Run benchmark bundles first, then "
            "return here to see how challengers and champion baselines changed over time."
        )

    earliest = history["run_date_sort"].min()
    latest = history["run_date_sort"].max()
    review_count = int((history["status_code"] == "needs_human_review").sum())
    failed_count = int((history["status_code"] == "failed_hard_gate").sum())

    earliest_str = earliest.strftime("%Y-%m-%d") if pd.notna(earliest) else "unknown"
    latest_str = latest.strftime("%Y-%m-%d") if pd.notna(latest) else "unknown"
    parts = [
        f"**History coverage:** {len(history)} benchmark bundles from {earliest_str} to {latest_str}.",
    ]

    passed = history[history["status_code"] == "passed_all_gates"]
    if not passed.empty and "delta_county_mape_overall" in passed.columns:
        best = passed.sort_values(
            "delta_county_mape_overall", ascending=True, na_position="last"
        ).iloc[0]
        parts.append(
            f"**Best accepted challenger:** {best['display_name']} at "
            f"{float(best['delta_county_mape_overall']):+.3f} county-MAPE delta vs the champion-at-run baseline."
        )

    if review_count or failed_count:
        parts.append(
            f"**Governance queue:** {review_count} bundle(s) still need review and {failed_count} failed hard gates."
        )
    else:
        parts.append(
            "**Governance queue:** No historical bundles are currently waiting for review or marked failed."
        )

    if "reference_county_mape_overall" in history.columns:
        champion_first = history.sort_values(
            "run_date_sort", ascending=True, na_position="last"
        ).iloc[0]
        champion_last = history.sort_values(
            "run_date_sort", ascending=False, na_position="last"
        ).iloc[0]
        first_metric = champion_first.get("reference_county_mape_overall")
        last_metric = champion_last.get("reference_county_mape_overall")
        if pd.notna(first_metric) and pd.notna(last_metric):
            parts.append(
                f"**Champion baseline drift:** county MAPE moved from {float(first_metric):.3f} "
                f"to {float(last_metric):.3f} across the recorded benchmark history."
            )

    return "\n\n".join(parts)


def _build_history_takeaway_card(dm: DashboardDataManager) -> pn.Card:
    """Render the executive-summary card for the history tab."""
    return widgets.markdown_card(
        "Executive Summary",
        _history_takeaway_text(dm),
        min_width=420,
    )


def _build_history_table(
    dm: DashboardDataManager, statuses: list[str] | tuple[str, ...] | None
) -> pn.Column:
    """Render a searchable table of all benchmark bundles through time."""
    history = _filter_history(dm, statuses)
    if history.empty:
        return pn.Column(
            widgets.empty_placeholder("No benchmark history matches the current filter.")
        )

    display_cols = [
        col
        for col in [
            "run_date_display",
            "display_name",
            "status_label",
            "selected_method_id",
            "selected_county_mape_overall",
            "delta_county_mape_overall",
            "selected_state_ape_recent_short",
            "delta_state_ape_recent_short",
            "next_action",
            "run_id",
        ]
        if col in history.columns
    ]
    display = history[display_cols].rename(
        columns={
            "run_date_display": "date",
            "display_name": "run",
            "status_label": "status",
            "selected_method_id": "method",
            "selected_county_mape_overall": "county_mape_overall",
            "delta_county_mape_overall": "delta_county_mape_overall",
            "selected_state_ape_recent_short": "state_ape_recent_short",
            "delta_state_ape_recent_short": "delta_state_ape_recent_short",
            "next_action": "next_action",
            "run_id": "exact_run_id",
        }
    )
    return widgets.metric_table(
        display,
        title="Historical Run Timeline",
        page_size=12,
        frozen_columns=["date", "run"],
    )


def _build_champion_history_chart(
    dm: DashboardDataManager, statuses: list[str] | tuple[str, ...] | None
) -> pn.pane.Plotly:
    """Plot the champion-at-run baseline alongside selected challengers over time."""
    history = _filter_history(dm, statuses)
    if history.empty or "run_date_sort" not in history.columns:
        return pn.pane.Plotly(go.Figure(), sizing_mode="stretch_width")

    _required = ["run_date_sort", "reference_county_mape_overall", "selected_county_mape_overall"]
    if not all(c in history.columns for c in _required):
        return pn.pane.Plotly(go.Figure(), sizing_mode="stretch_width")
    chart_data = history.dropna(subset=_required, how="any").copy()
    if chart_data.empty:
        return pn.pane.Plotly(go.Figure(), sizing_mode="stretch_width")
    chart_data = chart_data.sort_values("run_date_sort", ascending=True, na_position="last")

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=chart_data["run_date_sort"],
            y=chart_data["reference_county_mape_overall"],
            mode="lines+markers",
            name="Champion-at-run baseline",
            line={"color": theme.SDC_NAVY, "width": 3},
            marker={"size": 8},
            hovertemplate=(
                "<b>Champion-at-run</b><br>%{x|%Y-%m-%d}<br>County MAPE: %{y:.4f}<extra></extra>"
            ),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=chart_data["run_date_sort"],
            y=chart_data["selected_county_mape_overall"],
            mode="markers",
            name="Selected challenger/result row",
            marker={
                "size": 11,
                "color": [
                    theme.STATUS_COLORS.get(str(status), theme.SDC_MID_GRAY)
                    for status in chart_data["status_code"]
                ],
                "line": {"width": 1, "color": theme.SDC_DARK_GRAY},
            },
            text=chart_data["display_name"],
            hovertemplate=(
                "<b>%{text}</b><br>%{x|%Y-%m-%d}<br>Selected county MAPE: %{y:.4f}<extra></extra>"
            ),
        )
    )
    fig.update_layout(**theme.get_plotly_layout_defaults())
    fig.update_layout(
        title="Champion-at-Run Baseline Through Time",
        yaxis_title="County MAPE",
        height=430,
        legend={"orientation": "h", "y": -0.18},
    )
    return pn.pane.Plotly(fig, sizing_mode="stretch_width")


def _build_metric_delta_heatmap(
    dm: DashboardDataManager,
    statuses: list[str] | tuple[str, ...] | None,
    metrics: list[str] | tuple[str, ...] | None,
) -> pn.pane.Plotly:
    """Heatmap of selected-vs-reference metric deltas across the full run history."""
    delta_history = _filter_delta_history(dm, statuses)
    if delta_history.empty:
        return pn.pane.Plotly(go.Figure(), sizing_mode="stretch_width")

    metric_filter = [
        metric for metric in (metrics or list(_METRIC_LABELS)) if metric in _METRIC_LABELS
    ]
    if metric_filter:
        delta_history = delta_history[delta_history["metric"].isin(metric_filter)]
    if delta_history.empty:
        return pn.pane.Plotly(go.Figure(), sizing_mode="stretch_width")

    ordered = delta_history.sort_values(
        ["run_date_sort", "run_id", "metric"], ascending=[True, True, True]
    )
    ordered["run_axis_label"] = ordered.apply(
        lambda row: (
            f"{row['run_date_sort']:%Y-%m-%d}<br>{str(row['display_name'])[:24]}"
            if pd.notna(row["run_date_sort"])
            else f"unknown<br>{str(row['display_name'])[:24]}"
        ),
        axis=1,
    )
    ordered["metric_label"] = ordered["metric"].map(_METRIC_LABELS).fillna(ordered["metric"])
    pivot = ordered.pivot_table(
        index="metric_label",
        columns="run_axis_label",
        values="delta_value",
        aggfunc="first",  # type: ignore[arg-type]
    )
    if pivot.empty:
        return pn.pane.Plotly(go.Figure(), sizing_mode="stretch_width")

    fig = go.Figure(
        data=go.Heatmap(
            z=pivot.values,
            x=list(pivot.columns),
            y=list(pivot.index),
            colorscale="RdYlGn_r",
            zmid=0,
            colorbar={"title": "Delta"},
            hovertemplate="<b>%{y}</b><br>%{x}<br>Delta: %{z:+.4f}<extra></extra>",
        )
    )
    fig.update_layout(**theme.get_plotly_layout_defaults())
    fig.update_layout(
        title="Metric Delta Heatmap vs Champion-at-Run",
        xaxis_title="Benchmark bundle",
        yaxis_title="Metric",
        height=460,
    )
    return pn.pane.Plotly(fig, sizing_mode="stretch_width")


def _build_category_trend_chart(
    dm: DashboardDataManager, statuses: list[str] | tuple[str, ...] | None
) -> pn.pane.Plotly:
    """Line chart of average category-level deltas through time."""
    delta_history = _filter_delta_history(dm, statuses)
    if delta_history.empty:
        return pn.pane.Plotly(go.Figure(), sizing_mode="stretch_width")

    chart = delta_history[delta_history["metric"].isin(_CATEGORY_METRICS)].copy()
    if chart.empty:
        return pn.pane.Plotly(go.Figure(), sizing_mode="stretch_width")

    grouped = (
        chart.groupby(["run_date_sort", "metric"], dropna=False)["delta_value"].mean().reset_index()
    )
    grouped = grouped.dropna(subset=["run_date_sort"])
    if grouped.empty:
        return pn.pane.Plotly(go.Figure(), sizing_mode="stretch_width")

    fig = go.Figure()
    for metric in _CATEGORY_METRICS:
        metric_rows = grouped[grouped["metric"] == metric]
        if metric_rows.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=metric_rows["run_date_sort"],
                y=metric_rows["delta_value"],
                mode="lines+markers",
                name=_METRIC_LABELS.get(metric, metric),
                line={"width": 2.4},
                hovertemplate="<b>%{fullData.name}</b><br>%{x|%Y-%m-%d}<br>Mean delta: %{y:+.4f}<extra></extra>",
            )
        )

    fig.add_hline(y=0, line_width=1, line_color=theme.SDC_DARK_GRAY)
    fig.update_layout(**theme.get_plotly_layout_defaults())
    fig.update_layout(
        title="Category-Level Performance Change Through Time",
        yaxis_title="Mean delta vs champion-at-run",
        height=430,
        legend={"orientation": "h", "y": -0.18},
    )
    return pn.pane.Plotly(fig, sizing_mode="stretch_width")


def _build_outcome_history_chart(
    dm: DashboardDataManager, statuses: list[str] | tuple[str, ...] | None
) -> pn.pane.Plotly:
    """Stacked bar chart of accepted/rejected/review history over time."""
    timeline = dm.status_timeline
    if timeline.empty:
        return pn.pane.Plotly(go.Figure(), sizing_mode="stretch_width")

    selected = _selected_statuses(statuses, dm)
    if selected:
        timeline = timeline[timeline["status_code"].isin(selected)]
    if timeline.empty:
        return pn.pane.Plotly(go.Figure(), sizing_mode="stretch_width")

    fig = go.Figure()
    for status in _STATUS_ORDER:
        status_rows = timeline[timeline["status_code"] == status]
        if status_rows.empty:
            continue
        fig.add_trace(
            go.Bar(
                x=status_rows["run_date_display"],
                y=status_rows["run_count"],
                name=status.replace("_", " ").title(),
                marker_color=theme.STATUS_COLORS.get(status, theme.SDC_MID_GRAY),
                hovertemplate="<b>%{fullData.name}</b><br>%{x}<br>Runs: %{y}<extra></extra>",
            )
        )
    fig.update_layout(**theme.get_plotly_layout_defaults())
    fig.update_layout(
        title="Accepted vs Rejected Challenger History",
        xaxis_title="Run date",
        yaxis_title="Benchmark bundles",
        barmode="stack",
        height=410,
        legend={"orientation": "h", "y": -0.18},
    )
    return pn.pane.Plotly(fig, sizing_mode="stretch_width")


def build_history_tab(dm: DashboardDataManager) -> pn.Column:
    """Build the longitudinal benchmark-history dashboard tab."""
    if not dm.run_ids:
        return pn.Column(
            widgets.section_header("History", "No benchmark runs found."),
            widgets.illustrated_empty_state(
                "Run benchmarks to see longitudinal history trends.", "rocket"
            ),
        )

    available_statuses = _selected_statuses(None, dm)
    status_selector = pn.widgets.MultiChoice(
        name="Statuses",
        options=available_statuses,
        value=available_statuses,
        delete_button=True,
        max_items=max(5, len(available_statuses)),
        sizing_mode="stretch_width",
    )
    metric_selector = pn.widgets.MultiChoice(
        name="Heatmap Metrics",
        options=list(_METRIC_LABELS.keys()),
        value=list(_METRIC_LABELS.keys()),
        delete_button=True,
        max_items=len(_METRIC_LABELS),
        sizing_mode="stretch_width",
    )

    return pn.Column(
        widgets.section_header(
            "History",
            "Use this tab to review benchmark history as a trajectory instead of isolated runs.",
        ),
        pn.FlexBox(
            widgets.markdown_card(
                "Use This Tab To",
                "Read the full benchmark archive over time: which challengers were accepted, "
                "which were rejected or flagged for review, how the champion-at-run baseline "
                "moved, and where category-level gains or regressions have clustered.\n\n"
                "Start with the summary and timeline table, then move to the heatmap and "
                "trend charts when you want to understand the full experimentation path.",
                min_width=420,
            ),
            _build_history_takeaway_card(dm),
            flex_wrap="wrap",
            sizing_mode="stretch_width",
            styles={"gap": "12px"},
        ),
        pn.Card(
            pn.pane.HTML(
                '<div class="filters-help">Filter the longitudinal view by status or focus the heatmap on a subset of metrics.</div>',
                sizing_mode="stretch_width",
                stylesheets=[theme.DASHBOARD_CSS],
            ),
            pn.FlexBox(
                status_selector,
                metric_selector,
                flex_wrap="wrap",
                sizing_mode="stretch_width",
                styles={"gap": "10px"},
            ),
            title="History Filters",
            collapsed=False,
            sizing_mode="stretch_width",
        ),
        pn.Card(
            pn.bind(_build_history_table, dm=dm, statuses=status_selector),
            title="All Historical Runs By Date",
            collapsed=False,
            sizing_mode="stretch_width",
        ),
        pn.Card(
            pn.bind(_build_champion_history_chart, dm=dm, statuses=status_selector),
            title="Champion History Over Time",
            collapsed=False,
            sizing_mode="stretch_width",
        ),
        pn.Card(
            pn.bind(
                _build_metric_delta_heatmap,
                dm=dm,
                statuses=status_selector,
                metrics=metric_selector,
            ),
            title="Delta Trends By Metric",
            collapsed=False,
            sizing_mode="stretch_width",
        ),
        pn.Card(
            pn.bind(_build_category_trend_chart, dm=dm, statuses=status_selector),
            title="Category-Level Performance Changes",
            collapsed=False,
            sizing_mode="stretch_width",
        ),
        pn.Card(
            pn.bind(_build_outcome_history_chart, dm=dm, statuses=status_selector),
            title="Accepted vs Rejected Challenger History",
            collapsed=False,
            sizing_mode="stretch_width",
        ),
        sizing_mode="stretch_width",
    )
