"""Command Center tab — the Observatory dashboard home page.

Provides a high-level status overview with KPI cards, champion information,
benchmark index, action buttons for running experiments, and persistent
weakness detection.
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
    GROWTH_GREEN,
    SDC_BLUE,
    SDC_NAVY,
    SDC_RED,
    STATUS_COLORS,
)
from cohort_projections.analysis.observatory.dashboard.widgets import (
    empty_placeholder,
    kpi_card,
    metric_table,
    section_header,
)

logger = logging.getLogger(__name__)

# Path to the observatory CLI script (for subprocess calls).
_OBSERVATORY_SCRIPT = (
    Path(__file__).resolve().parents[4] / "scripts" / "analysis" / "observatory.py"
)
_PROJECT_ROOT = Path(__file__).resolve().parents[4]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _count_by_outcome(dm: DashboardDataManager, outcome: str) -> int:
    """Count experiment log entries matching *outcome*."""
    log = dm.experiment_log
    if log.empty or "outcome" not in log.columns:
        return 0
    return int((log["outcome"].str.lower() == outcome.lower()).sum())


def _champion_mape(dm: DashboardDataManager) -> float | None:
    """Return the champion's county_mape_overall, or None."""
    champion_id = dm.champion_id
    if champion_id is None:
        return None
    scorecards = dm.scorecards
    if scorecards.empty or "county_mape_overall" not in scorecards.columns:
        return None
    champ_rows = scorecards[scorecards["run_id"] == champion_id]
    if champ_rows.empty:
        return None
    return float(champ_rows.iloc[0]["county_mape_overall"])


def _run_observatory_command(args: list[str]) -> str:
    """Run an observatory CLI command via subprocess and return its output.

    Parameters
    ----------
    args:
        Arguments to pass after ``observatory.py`` (e.g. ``["run-pending", "--dry-run"]``).

    Returns
    -------
    str
        Combined stdout and stderr from the subprocess.
    """
    cmd = [sys.executable, str(_OBSERVATORY_SCRIPT), *args]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
            cwd=str(_PROJECT_ROOT),
        )
        output = result.stdout
        if result.stderr:
            output += "\n--- stderr ---\n" + result.stderr
        return output or "(no output)"
    except subprocess.TimeoutExpired:
        return "Command timed out after 120 seconds."
    except Exception as exc:
        return f"Error running command: {exc}"


# ---------------------------------------------------------------------------
# Row builders
# ---------------------------------------------------------------------------


def _build_kpi_row(dm: DashboardDataManager) -> pn.Row:
    """Row 1: KPI cards summarising observatory state."""
    total_runs = len(dm.run_ids)

    # Tested variant count from catalog
    tested_count = 0
    untested_count = 0
    if dm.catalog is not None:
        variants_df = dm.catalog.list_variants()
        if not variants_df.empty and "tested" in variants_df.columns:
            tested_count = int(variants_df["tested"].sum())
            untested_count = int((~variants_df["tested"]).sum())

    passed = _count_by_outcome(dm, "passed_all_gates")
    review = _count_by_outcome(dm, "needs_human_review")
    champ_mape = _champion_mape(dm)

    cards = [
        kpi_card("Total Runs", total_runs, color=SDC_NAVY),
        kpi_card("Experiments Tested", tested_count, color=SDC_BLUE),
        kpi_card("Passed Gates", passed, color=GROWTH_GREEN),
        kpi_card("Needs Review", review, color=STATUS_COLORS.get("needs_human_review", "#FFC000")),
        kpi_card("Untested Variants", untested_count, color=STATUS_COLORS.get("untested", "#A0A0A0")),
        kpi_card(
            "Champion MAPE",
            f"{champ_mape:.2f}%" if champ_mape is not None else "N/A",
            color=SDC_NAVY,
        ),
    ]
    return pn.Row(*cards, sizing_mode="stretch_width")


def _build_champion_card(dm: DashboardDataManager) -> pn.Card:
    """Row 2: Champion information card."""
    champion_id = dm.champion_id

    if champion_id is None:
        return pn.Card(
            empty_placeholder("No champion detected."),
            title="Champion",
            sizing_mode="stretch_width",
        )

    scorecards = dm.scorecards
    if scorecards.empty:
        return pn.Card(
            empty_placeholder("No scorecard data available."),
            title="Champion",
            sizing_mode="stretch_width",
        )

    champ_rows = scorecards[scorecards["run_id"] == champion_id]
    if champ_rows.empty:
        return pn.Card(
            empty_placeholder(f"Champion {champion_id} not found in scorecards."),
            title="Champion",
            sizing_mode="stretch_width",
        )

    champ = champ_rows.iloc[0]
    method_id = champ.get("method_id", "N/A")
    config_id = champ.get("config_id", "N/A")
    overall_mape = champ.get("county_mape_overall")
    state_short = champ.get("state_ape_recent_short")
    state_medium = champ.get("state_ape_recent_medium")

    # Sentinel values
    sentinel_cols = [c for c in champ.index if c.startswith("sentinel_")]
    sentinel_rows = ""
    for col in sentinel_cols:
        val = champ.get(col)
        if pd.notna(val):
            label = col.replace("sentinel_", "").replace("_mape", "").replace("_", " ").title()
            sentinel_rows += (
                f'<tr><td style="padding:2px 12px 2px 0;color:#595959">{label}</td>'
                f'<td style="padding:2px 0;font-weight:600">{val:.4f}</td></tr>'
            )

    html = f"""
    <div style="font-family:'Segoe UI',Roboto,Arial,sans-serif">
      <table style="width:100%;border-collapse:collapse">
        <tr>
          <td style="padding:4px 12px 4px 0;color:#595959;width:140px">Run ID</td>
          <td style="padding:4px 0;font-weight:600">{champion_id}</td>
        </tr>
        <tr>
          <td style="padding:4px 12px 4px 0;color:#595959">Method</td>
          <td style="padding:4px 0;font-weight:600">{method_id}</td>
        </tr>
        <tr>
          <td style="padding:4px 12px 4px 0;color:#595959">Config</td>
          <td style="padding:4px 0;font-weight:600">{config_id}</td>
        </tr>
        <tr>
          <td style="padding:4px 12px 4px 0;color:#595959">Overall MAPE</td>
          <td style="padding:4px 0;font-weight:700;color:{SDC_NAVY}">
            {f'{overall_mape:.4f}' if pd.notna(overall_mape) else 'N/A'}
          </td>
        </tr>
        <tr>
          <td style="padding:4px 12px 4px 0;color:#595959">State APE (short)</td>
          <td style="padding:4px 0;font-weight:600">
            {f'{state_short:.4f}' if pd.notna(state_short) else 'N/A'}
          </td>
        </tr>
        <tr>
          <td style="padding:4px 12px 4px 0;color:#595959">State APE (medium)</td>
          <td style="padding:4px 0;font-weight:600">
            {f'{state_medium:.4f}' if pd.notna(state_medium) else 'N/A'}
          </td>
        </tr>
      </table>
      {'<h4 style="margin:12px 0 4px 0;color:#1F3864">Sentinel Counties</h4>'
       '<table style="width:100%;border-collapse:collapse">'
       + sentinel_rows +
       '</table>' if sentinel_rows else ''}
    </div>
    """
    return pn.Card(
        pn.pane.HTML(html, sizing_mode="stretch_width"),
        title="Champion",
        sizing_mode="stretch_width",
    )


def _build_index_table(dm: DashboardDataManager) -> pn.Card:
    """Row 3: Benchmark index table."""
    index_df = dm.index

    if index_df.empty:
        return pn.Card(
            empty_placeholder("No benchmark runs found."),
            title="Benchmark Index",
            sizing_mode="stretch_width",
        )

    display_cols = [
        c for c in ["run_id", "method_id", "config_id", "date", "status"]
        if c in index_df.columns
    ]
    display_df = index_df[display_cols] if display_cols else index_df

    return pn.Card(
        metric_table(display_df, page_size=10),
        title="Benchmark Index",
        sizing_mode="stretch_width",
    )


def _build_action_buttons(dm: DashboardDataManager) -> pn.Card:
    """Row 4: Action buttons with subprocess output."""
    output_pane = pn.widgets.TextAreaInput(
        value="Click a button above to see output here.",
        disabled=True,
        height=250,
        sizing_mode="stretch_width",
        name="Output",
    )

    def on_refresh(event: Any) -> None:
        dm.refresh()
        output_pane.value = "Data refreshed successfully."

    def on_run_pending(event: Any) -> None:
        output_pane.value = "Running 'run-pending --dry-run'...\n"
        result = _run_observatory_command(["run-pending", "--dry-run"])
        output_pane.value = result

    def on_run_recommended(event: Any) -> None:
        output_pane.value = "Running 'run-recommended --dry-run'...\n"
        result = _run_observatory_command(["run-recommended", "--dry-run"])
        output_pane.value = result

    btn_refresh = pn.widgets.Button(
        name="Refresh Data",
        button_type="primary",
        sizing_mode="fixed",
        width=150,
    )
    btn_refresh.on_click(on_refresh)

    btn_pending = pn.widgets.Button(
        name="Run Pending (Preview)",
        button_type="warning",
        sizing_mode="fixed",
        width=180,
    )
    btn_pending.on_click(on_run_pending)

    btn_recommended = pn.widgets.Button(
        name="Run Recommended (Preview)",
        button_type="warning",
        sizing_mode="fixed",
        width=210,
    )
    btn_recommended.on_click(on_run_recommended)

    button_row = pn.Row(
        btn_refresh,
        btn_pending,
        btn_recommended,
        sizing_mode="stretch_width",
    )

    return pn.Card(
        button_row,
        output_pane,
        title="Actions",
        sizing_mode="stretch_width",
    )


def _build_weaknesses_panel(dm: DashboardDataManager) -> pn.Card:
    """Row 5: Persistent weaknesses from the recommender."""
    try:
        weaknesses = dm.recommender.identify_persistent_weaknesses()
    except Exception:
        logger.exception("Failed to identify persistent weaknesses.")
        return pn.Card(
            empty_placeholder("Unable to compute persistent weaknesses."),
            title="Persistent Weaknesses",
            sizing_mode="stretch_width",
        )

    if weaknesses.empty:
        return pn.Card(
            pn.pane.Alert(
                "No persistent weaknesses detected — all metrics have at "
                "least one challenger improving over the champion.",
                alert_type="success",
            ),
            title="Persistent Weaknesses",
            sizing_mode="stretch_width",
        )

    # Filter to actual weaknesses (no challenger improves)
    persistent = weaknesses[
        weaknesses["best_challenger_delta"].notna()
        & (weaknesses["best_challenger_delta"] >= 0)
    ]

    if persistent.empty:
        return pn.Card(
            pn.pane.Alert(
                "No persistent weaknesses — every metric has at least one "
                "variant that improves over the champion.",
                alert_type="success",
            ),
            title="Persistent Weaknesses",
            sizing_mode="stretch_width",
        )

    # Build a warning table
    rows_html = ""
    for _, row in persistent.iterrows():
        metric = row["metric"]
        champ_val = row["champion_value"]
        best_delta = row["best_challenger_delta"]
        champ_str = f"{champ_val:.4f}" if pd.notna(champ_val) else "N/A"
        delta_str = f"{best_delta:+.4f}" if pd.notna(best_delta) else "N/A"
        rows_html += (
            f"<tr>"
            f'<td style="padding:4px 12px 4px 0">{metric}</td>'
            f'<td style="padding:4px 12px 4px 0">{champ_str}</td>'
            f'<td style="padding:4px 0;color:{SDC_RED};font-weight:600">'
            f"{delta_str}</td>"
            f"</tr>"
        )

    html = f"""
    <div style="font-family:'Segoe UI',Roboto,Arial,sans-serif">
      <p style="color:#C00000;font-weight:600;margin-bottom:8px">
        The following metrics have no challenger variant that improves over
        the champion. Consider designing new experiments targeting these areas.
      </p>
      <table style="width:100%;border-collapse:collapse">
        <thead>
          <tr style="border-bottom:2px solid #D9D9D9">
            <th style="padding:4px 12px 4px 0;text-align:left">Metric</th>
            <th style="padding:4px 12px 4px 0;text-align:left">Champion Value</th>
            <th style="padding:4px 0;text-align:left">Best Challenger Delta</th>
          </tr>
        </thead>
        <tbody>
          {rows_html}
        </tbody>
      </table>
    </div>
    """

    return pn.Card(
        pn.pane.Alert(
            f"{len(persistent)} persistent weakness(es) detected.",
            alert_type="warning",
        ),
        pn.pane.HTML(html, sizing_mode="stretch_width"),
        title="Persistent Weaknesses",
        sizing_mode="stretch_width",
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_command_center(dm: DashboardDataManager) -> pn.Column:
    """Build the Command Center tab — the Observatory dashboard home page.

    Parameters
    ----------
    dm:
        The :class:`DashboardDataManager` providing all data access.

    Returns
    -------
    pn.Column
        A Panel Column containing KPI cards, champion info, benchmark index,
        action buttons, and persistent weakness analysis.
    """
    return pn.Column(
        section_header(
            "Command Center",
            subtitle="Observatory status overview and quick actions",
        ),
        _build_kpi_row(dm),
        _build_champion_card(dm),
        _build_index_table(dm),
        _build_action_buttons(dm),
        _build_weaknesses_panel(dm),
        sizing_mode="stretch_width",
    )
