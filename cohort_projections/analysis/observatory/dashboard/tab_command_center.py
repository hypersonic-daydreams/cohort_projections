"""Command Center tab for the Observatory dashboard.

This is the dashboard home page. It prioritizes current decision context,
run health, and next actions before exposing the lower-level benchmark index.
"""

from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import panel as pn

from cohort_projections.analysis.observatory.dashboard.data_manager import (
    DashboardDataManager,
)
from cohort_projections.analysis.observatory.dashboard.theme import (
    DASHBOARD_CSS,
    GROWTH_GREEN,
    SDC_BLUE,
    SDC_NAVY,
    STATUS_COLORS,
)
from cohort_projections.analysis.observatory.dashboard.widgets import (
    empty_placeholder,
    kpi_card,
    metric_table,
    section_header,
)

logger = logging.getLogger(__name__)

_OBSERVATORY_SCRIPT = (
    Path(__file__).resolve().parents[4] / "scripts" / "analysis" / "observatory.py"
)
_PROJECT_ROOT = Path(__file__).resolve().parents[4]


def _count_by_outcome(dm: DashboardDataManager, outcome: str) -> int:
    """Count experiment log entries matching *outcome*."""
    log = dm.experiment_log
    if log.empty or "outcome" not in log.columns:
        return 0
    return int((log["outcome"].str.lower() == outcome.lower()).sum())


def _champion_mape(dm: DashboardDataManager) -> float | None:
    """Return the champion's county-level overall MAPE."""
    champion_id = dm.champion_id
    if champion_id is None:
        return None
    rows = dm.run_metadata[dm.run_metadata["run_id"] == champion_id]
    if rows.empty:
        return None
    value = rows.iloc[0].get("reference_county_mape_overall")
    metric = _as_float(value)
    if metric is not None:
        return metric
    value = rows.iloc[0].get("selected_county_mape_overall")
    return _as_float(value)


def _summary_card(title: str, headline: str, detail: str, tone: str = "primary") -> pn.pane.HTML:
    """Render a compact decision card."""
    html = (
        f'<div class="summary-card {tone}">'
        f'  <div class="eyebrow">{title}</div>'
        f'  <div class="headline">{headline}</div>'
        f'  <div class="detail">{detail}</div>'
        f"</div>"
    )
    return pn.pane.HTML(
        html,
        width=280,
        min_width=240,
        stylesheets=[DASHBOARD_CSS],
    )


def _fmt_metric(value: object) -> str:
    """Format a metric for compact HTML rendering."""
    metric = _as_float(value)
    if metric is None:
        return "N/A"
    return f"{metric:.4f}"


def _as_float(value: object) -> float | None:
    """Coerce a scalar metric value to float when possible."""
    if value is None or pd.isna(value):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return None


def _run_observatory_command(args: list[str]) -> str:
    """Run the observatory CLI and return combined stdout/stderr."""
    cmd = [sys.executable, str(_OBSERVATORY_SCRIPT), *args]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
            cwd=str(_PROJECT_ROOT),
        )
    except subprocess.TimeoutExpired:
        return "Command timed out after 120 seconds."
    except Exception as exc:  # pragma: no cover - defensive UI guard
        return f"Error running command: {exc}"

    output = result.stdout
    if result.stderr:
        output += "\n--- stderr ---\n" + result.stderr
    return output or "(no output)"


def _best_tested_challenger(dm: DashboardDataManager) -> pd.Series | None:
    """Return the strongest non-champion benchmark row for summary cards."""
    if dm.run_metadata.empty:
        return None
    challengers = dm.run_metadata.copy()
    if dm.champion_id is not None:
        challengers = challengers[challengers["run_id"] != dm.champion_id]
    challengers = challengers[
        challengers["selected_county_mape_overall"].notna()
    ]
    if challengers.empty:
        return None
    return challengers.sort_values(
        ["selected_county_mape_overall", "run_date_sort"],
        ascending=[True, False],
        na_position="last",
    ).iloc[0]


def _build_kpi_row(dm: DashboardDataManager) -> pn.FlexBox:
    """Top-line KPI strip with mobile-safe wrapping."""
    total_runs = len(dm.run_ids)

    tested_count = 0
    untested_count = 0
    if dm.catalog is not None:
        variants_df = dm.catalog.list_variants()
        if not variants_df.empty and "tested" in variants_df.columns:
            tested_count = int(variants_df["tested"].sum())
            untested_count = int((~variants_df["tested"]).sum())

    cards = [
        kpi_card("Total Runs", total_runs, color=SDC_NAVY),
        kpi_card("Experiments Tested", tested_count, color=SDC_BLUE),
        kpi_card(
            "Passed Gates",
            _count_by_outcome(dm, "passed_all_gates"),
            color=GROWTH_GREEN,
        ),
        kpi_card(
            "Needs Review",
            _count_by_outcome(dm, "needs_human_review"),
            color=STATUS_COLORS["needs_human_review"],
        ),
        kpi_card(
            "Untested Variants",
            untested_count,
            color=STATUS_COLORS["untested"],
        ),
        kpi_card(
            "Champion MAPE",
            f"{_champion_mape(dm):.2f}%"
            if _champion_mape(dm) is not None
            else "N/A",
            color=SDC_NAVY,
        ),
    ]
    return pn.FlexBox(
        *cards,
        flex_wrap="wrap",
        sizing_mode="stretch_width",
        styles={"gap": "12px"},
    )


def _build_decision_strip(dm: DashboardDataManager) -> pn.FlexBox:
    """Decision-oriented summary cards for the first screen."""
    champion_mape = _champion_mape(dm)
    champion_row = (
        dm.run_metadata[dm.run_metadata["run_id"] == dm.champion_id].iloc[0]
        if dm.champion_id is not None
        and not dm.run_metadata[dm.run_metadata["run_id"] == dm.champion_id].empty
        else None
    )
    best_variant = _best_tested_challenger(dm)
    review_queue = dm.run_metadata[dm.run_metadata["status_code"] == "needs_human_review"]

    recommendations = dm.recommender.suggest_next_experiments(1)
    top_recommendation = recommendations[0] if recommendations else None

    champion_card = _summary_card(
        "Current Champion",
        f"{champion_mape:.2f}% overall MAPE"
        if champion_mape is not None
        else "Champion unavailable",
        (
            f"{champion_row['display_name']} | "
            f"{champion_row.get('reference_method_id', champion_row.get('selected_method_id', ''))}"
            if champion_row is not None
            else "No champion metadata found."
        ),
        tone="primary",
    )

    best_variant_mape = (
        _as_float(best_variant.get("selected_county_mape_overall"))
        if best_variant is not None
        else None
    )
    if best_variant is not None and best_variant_mape is not None and champion_mape is not None:
        delta = best_variant_mape - champion_mape
        best_card = _summary_card(
            "Best Challenger",
            str(best_variant["display_name"]),
            f"{best_variant_mape:.2f}% overall MAPE ({delta:+.2f} vs champion)",
            tone="success" if delta <= 0 else "warning",
        )
    else:
        best_card = _summary_card(
            "Best Challenger",
            "No challenger ranked yet",
            "Run more benchmark bundles to populate variant comparisons.",
            tone="warning",
        )

    if review_queue.empty:
        review_card = _summary_card(
            "Review Queue",
            "No pending reviews",
            "All completed runs are either passed, failed hard, or still untested.",
            tone="success",
        )
    else:
        latest_review = review_queue.sort_values(
            "run_date_sort", ascending=False, na_position="last"
        ).iloc[0]
        review_card = _summary_card(
            "Review Queue",
            f"{len(review_queue)} run(s) need review",
            f"Latest: {latest_review['display_name']}",
            tone="warning",
        )

    if top_recommendation is None:
        recommendation_card = _summary_card(
            "Next Recommendation",
            "No recommendation yet",
            "The recommender needs more completed experiment history.",
            tone="primary",
        )
    else:
        recommendation_card = _summary_card(
            "Next Recommendation",
            f"{top_recommendation.parameter} -> {top_recommendation.suggested_value}",
            top_recommendation.rationale,
            tone="primary",
        )

    return pn.FlexBox(
        champion_card,
        best_card,
        review_card,
        recommendation_card,
        flex_wrap="wrap",
        sizing_mode="stretch_width",
        styles={"gap": "12px"},
    )


def _build_champion_card(dm: DashboardDataManager) -> pn.Card:
    """Detailed champion snapshot for quick inspection."""
    champion_id = dm.champion_id
    if champion_id is None:
        return pn.Card(
            empty_placeholder("No champion detected."),
            title="Champion Snapshot",
            sizing_mode="stretch_width",
        )

    scorecards = dm.scorecards
    champ_rows = scorecards[scorecards["run_id"] == champion_id]
    if champ_rows.empty:
        return pn.Card(
            empty_placeholder("Champion scorecard not found."),
            title="Champion Snapshot",
            sizing_mode="stretch_width",
        )

    champion_row = champ_rows[
        champ_rows["status_at_run"].fillna("").str.lower() == "champion"
    ]
    champion = champion_row.iloc[0] if not champion_row.empty else champ_rows.iloc[0]

    sentinel_cols = [c for c in champion.index if c.startswith("sentinel_")]
    sentinel_rows = []
    for col in sentinel_cols:
        value = champion.get(col)
        if pd.notna(value):
            label = (
                col.replace("sentinel_", "")
                .replace("_mape", "")
                .replace("_", " ")
                .title()
            )
            sentinel_rows.append((label, float(value)))
    sentinel_rows = sorted(sentinel_rows, key=lambda item: item[1], reverse=True)

    sentinel_html = ""
    if sentinel_rows:
        sentinel_html = (
            '<h4 style="margin:12px 0 6px 0;color:#1F3864">Highest Sentinel MAPEs</h4>'
            '<table style="width:100%;border-collapse:collapse">'
            + "".join(
                (
                    f'<tr><td style="padding:3px 12px 3px 0;color:#5A6C84">{label}</td>'
                    f'<td style="padding:3px 0;font-weight:600">{value:.2f}</td></tr>'
                )
                for label, value in sentinel_rows[:4]
            )
            + "</table>"
        )

    html = f"""
    <div style="font-family:'Aptos','Segoe UI',Arial,sans-serif">
      <table style="width:100%;border-collapse:collapse">
        <tr><td style="padding:4px 12px 4px 0;color:#5A6C84;width:150px">Run</td><td style="padding:4px 0;font-weight:600">{dm.run_label(champion_id)}</td></tr>
        <tr><td style="padding:4px 12px 4px 0;color:#5A6C84">Method</td><td style="padding:4px 0;font-weight:600">{champion.get('method_id', 'N/A')}</td></tr>
        <tr><td style="padding:4px 12px 4px 0;color:#5A6C84">Config</td><td style="padding:4px 0;font-weight:600">{champion.get('config_id', 'N/A')}</td></tr>
        <tr><td style="padding:4px 12px 4px 0;color:#5A6C84">Overall MAPE</td><td style="padding:4px 0;font-weight:700;color:{SDC_NAVY}">{_fmt_metric(champion.get('county_mape_overall'))}</td></tr>
        <tr><td style="padding:4px 12px 4px 0;color:#5A6C84">State APE (short)</td><td style="padding:4px 0;font-weight:600">{_fmt_metric(champion.get('state_ape_recent_short'))}</td></tr>
        <tr><td style="padding:4px 12px 4px 0;color:#5A6C84">State APE (medium)</td><td style="padding:4px 0;font-weight:600">{_fmt_metric(champion.get('state_ape_recent_medium'))}</td></tr>
      </table>
      {sentinel_html}
    </div>
    """
    return pn.Card(
        pn.pane.HTML(html, sizing_mode="stretch_width"),
        title="Champion Snapshot",
        sizing_mode="stretch_width",
    )


def _build_index_table(dm: DashboardDataManager) -> pn.Card:
    """Readable run index with labels, status, and next action."""
    if dm.run_metadata.empty:
        return pn.Card(
            empty_placeholder("No benchmark runs found."),
            title="Run Index",
            sizing_mode="stretch_width",
        )

    display_df = dm.run_metadata.copy()
    display_df["run"] = display_df["display_name"]
    display_df["status"] = display_df["status_label"]
    display_df["config"] = display_df["short_config"].replace("", pd.NA)
    display_df["overall_mape"] = display_df["selected_county_mape_overall"].round(3)
    display_df["state_ape_short"] = display_df["selected_state_ape_recent_short"].round(3)
    display_df["run_date"] = display_df["run_date_label"]

    columns = [
        column
        for column in [
            "run",
            "status",
            "run_date",
            "selected_method_id",
            "config",
            "overall_mape",
            "state_ape_short",
            "next_action",
            "run_id",
        ]
        if column in display_df.columns
    ]
    display_df = display_df[columns].rename(
        columns={
            "selected_method_id": "method",
            "next_action": "next_action",
            "run_id": "exact_run_id",
        }
    )

    return pn.Card(
        metric_table(
            display_df,
            page_size=8,
            frozen_columns=["run"],
        ),
        title="Run Index",
        sizing_mode="stretch_width",
    )


def _build_action_buttons(dm: DashboardDataManager) -> pn.Card:
    """Action row and output console."""
    output_pane = pn.widgets.TextAreaInput(
        value="Use the preview actions below to inspect pending or recommended runs.",
        disabled=True,
        height=220,
        sizing_mode="stretch_width",
        name="Output",
    )

    def on_refresh(event: Any) -> None:
        dm.refresh()
        output_pane.value = "Data refreshed successfully."

    def on_run_pending(event: Any) -> None:
        output_pane.value = "Running 'run-pending --dry-run'...\n"
        output_pane.value = _run_observatory_command(["run-pending", "--dry-run"])

    def on_run_recommended(event: Any) -> None:
        output_pane.value = "Running 'run-recommended --dry-run'...\n"
        output_pane.value = _run_observatory_command(["run-recommended", "--dry-run"])

    btn_refresh = pn.widgets.Button(
        name="Refresh Data",
        button_type="primary",
        width=150,
    )
    btn_refresh.on_click(on_refresh)

    btn_pending = pn.widgets.Button(
        name="Preview Pending",
        button_type="warning",
        width=160,
    )
    btn_pending.on_click(on_run_pending)

    btn_recommended = pn.widgets.Button(
        name="Preview Recommended",
        button_type="warning",
        width=190,
    )
    btn_recommended.on_click(on_run_recommended)

    buttons = pn.FlexBox(
        btn_refresh,
        btn_pending,
        btn_recommended,
        flex_wrap="wrap",
        sizing_mode="stretch_width",
        styles={"gap": "10px"},
    )

    return pn.Card(buttons, output_pane, title="Actions", sizing_mode="stretch_width")


def _build_weaknesses_panel(dm: DashboardDataManager) -> pn.Card:
    """Persistent weaknesses from the recommender."""
    try:
        weaknesses = dm.recommender.identify_persistent_weaknesses()
    except Exception:  # pragma: no cover - defensive UI guard
        logger.exception("Failed to identify persistent weaknesses.")
        return pn.Card(
            empty_placeholder("Unable to compute persistent weaknesses."),
            title="Persistent Weaknesses",
            sizing_mode="stretch_width",
        )

    if weaknesses.empty:
        return pn.Card(
            pn.pane.Alert(
                "No persistent weaknesses detected. Every tracked county group has at least one tested variant that improves over the champion.",
                alert_type="success",
            ),
            title="Persistent Weaknesses",
            sizing_mode="stretch_width",
        )

    persistent = weaknesses[
        weaknesses["best_challenger_delta"].notna()
        & (weaknesses["best_challenger_delta"] >= 0)
    ]
    if persistent.empty:
        return pn.Card(
            pn.pane.Alert(
                "No persistent weaknesses. The tested variant set covers every tracked county group.",
                alert_type="success",
            ),
            title="Persistent Weaknesses",
            sizing_mode="stretch_width",
        )

    display_df = persistent.copy()
    display_df["champion_value"] = display_df["champion_value"].round(4)
    display_df["best_challenger_delta"] = display_df["best_challenger_delta"].round(4)
    display_df = display_df.rename(
        columns={
            "metric": "Metric",
            "champion_value": "Champion Value",
            "best_challenger_delta": "Best Delta",
            "best_challenger_run": "Best Challenger Run",
        }
    )

    return pn.Card(
        pn.pane.Alert(
            f"{len(display_df)} metric(s) still have no improving challenger.",
            alert_type="warning",
        ),
        metric_table(
            display_df,
            page_size=0,
            frozen_columns=["Metric"],
        ),
        title="Persistent Weaknesses",
        sizing_mode="stretch_width",
    )


def build_command_center(dm: DashboardDataManager) -> pn.Column:
    """Build the dashboard home page."""
    return pn.Column(
        section_header(
            "Command Center",
            subtitle="Current state, decision context, and quick actions",
        ),
        _build_kpi_row(dm),
        _build_decision_strip(dm),
        _build_champion_card(dm),
        _build_index_table(dm),
        _build_action_buttons(dm),
        _build_weaknesses_panel(dm),
        sizing_mode="stretch_width",
    )
