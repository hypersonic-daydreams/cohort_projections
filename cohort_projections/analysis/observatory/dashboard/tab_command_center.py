"""Command Center tab for the Observatory dashboard.

This is the dashboard home page. It prioritizes current decision context,
run health, and next actions before exposing the lower-level benchmark index.
"""

from __future__ import annotations

import datetime as dt
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import panel as pn
import yaml

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
    if value is None or pd.isna(value):  # type: ignore[call-overload]
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


def _default_search_id() -> str:
    """Build a timestamped search ID for dashboard launches."""
    return f"search-{dt.datetime.now(tz=dt.UTC).strftime('%Y%m%d-%H%M%S')}"


def _search_progress_html(session_row: pd.Series | None) -> str:
    """Render a simple HTML progress bar for one autonomous-search session."""
    if session_row is None:
        return "<div><strong>No autonomous-search session selected.</strong></div>"

    progress_pct = float(session_row.get("progress_pct", 0.0) or 0.0)
    status = str(session_row.get("status", "unknown") or "unknown")
    total = int(session_row.get("total", 0) or 0)
    planned = int(session_row.get("planned", 0) or 0)
    running = int(session_row.get("running", 0) or 0)
    completed = int(session_row.get("completed", 0) or 0)
    failed = int(session_row.get("failed", 0) or 0)
    tone = STATUS_COLORS.get(
        "needs_human_review" if failed else status,
        SDC_BLUE,
    )
    return f"""
    <div style="font-family:'Aptos','Segoe UI',Arial,sans-serif">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px">
        <strong>{session_row.get("search_id", "")}</strong>
        <span style="color:#5A6C84;text-transform:uppercase;font-size:12px">{status}</span>
      </div>
      <div style="width:100%;height:14px;background:#E7ECF3;border-radius:999px;overflow:hidden">
        <div style="width:{progress_pct:.1f}%;height:14px;background:{tone};border-radius:999px"></div>
      </div>
      <div style="display:flex;justify-content:space-between;gap:12px;flex-wrap:wrap;margin-top:8px;font-size:13px;color:#334E68">
        <span>Progress: {completed + failed}/{total}</span>
        <span>Completed: {completed}</span>
        <span>Failed: {failed}</span>
        <span>Running: {running}</span>
        <span>Planned: {planned}</span>
      </div>
    </div>
    """


def _search_session_detail_html(session_row: pd.Series | None) -> str:
    """Render search-session metadata and artifact availability."""
    if session_row is None:
        return "<div style='color:#5A6C84'>No autonomous-search sessions found yet.</div>"

    artifact_lines = []
    for label, key in [
        ("Candidate summary CSV", "candidate_summary_csv"),
        ("Candidate summary JSON", "candidate_summary_json"),
        ("Search report", "search_report_markdown"),
        ("Observatory report", "observatory_report_html"),
    ]:
        value = str(session_row.get(key, "") or "")
        artifact_lines.append(
            f"<li><strong>{label}:</strong> {value if value else 'not written yet'}</li>"
        )

    return f"""
    <div style="font-family:'Aptos','Segoe UI',Arial,sans-serif">
      <table style="width:100%;border-collapse:collapse">
        <tr><td style="padding:4px 12px 4px 0;color:#5A6C84;width:160px">Created</td><td>{session_row.get("created_at", "")}</td></tr>
        <tr><td style="padding:4px 12px 4px 0;color:#5A6C84">Updated</td><td>{session_row.get("updated_at", "")}</td></tr>
        <tr><td style="padding:4px 12px 4px 0;color:#5A6C84">Base revision</td><td>{session_row.get("resolved_base_revision", session_row.get("base_revision", ""))}</td></tr>
        <tr><td style="padding:4px 12px 4px 0;color:#5A6C84">Process</td><td>{session_row.get("process_status", "unknown")} (pid: {session_row.get("dashboard_pid", "n/a")})</td></tr>
        <tr><td style="padding:4px 12px 4px 0;color:#5A6C84">Session directory</td><td>{session_row.get("session_dir", "")}</td></tr>
      </table>
      <div style="margin-top:10px">
        <strong>Artifacts</strong>
        <ul style="margin:6px 0 0 18px;padding:0">
          {"".join(artifact_lines)}
        </ul>
      </div>
    </div>
    """


def _best_tested_challenger(dm: DashboardDataManager) -> pd.Series | None:
    """Return the strongest non-champion benchmark row for summary cards."""
    if dm.run_metadata.empty:
        return None
    challengers = dm.run_metadata.copy()
    if dm.champion_id is not None:
        challengers = challengers[challengers["run_id"] != dm.champion_id]
    challengers = challengers[challengers["selected_county_mape_overall"].notna()]
    if challengers.empty:
        return None
    return challengers.sort_values(
        ["selected_county_mape_overall", "run_date_sort"],
        ascending=[True, False],
        na_position="last",
    ).iloc[0]


def _command_center_summary(dm: DashboardDataManager) -> str:
    """Return a plain-language summary of the current decision state."""
    champion_row = (
        dm.run_metadata[dm.run_metadata["run_id"] == dm.champion_id].iloc[0]
        if dm.champion_id is not None
        and not dm.run_metadata[dm.run_metadata["run_id"] == dm.champion_id].empty
        else None
    )
    champion_mape = _champion_mape(dm)
    best_variant = _best_tested_challenger(dm)
    if not dm.run_metadata.empty and "status_code" in dm.run_metadata.columns:
        review_queue = dm.run_metadata[dm.run_metadata["status_code"] == "needs_human_review"]
    else:
        review_queue = dm.run_metadata.iloc[:0]

    recommendations = dm.recommender.suggest_next_experiments(1)
    top_recommendation = recommendations[0] if recommendations else None

    summary_parts: list[str] = []
    if champion_row is not None and champion_mape is not None:
        summary_parts.append(
            f"Current champion: {champion_row['display_name']} at {champion_mape:.2f}% county error."
        )
    elif champion_row is not None:
        summary_parts.append(f"Current champion: {champion_row['display_name']}.")

    best_variant_mape = (
        _as_float(best_variant.get("selected_county_mape_overall"))
        if best_variant is not None
        else None
    )
    if best_variant is not None and best_variant_mape is not None and champion_mape is not None:
        delta = best_variant_mape - champion_mape
        summary_parts.append(
            f"Best tested challenger: {best_variant['display_name']} at "
            f"{best_variant_mape:.2f}% ({delta:+.2f} vs champion)."
        )
    elif best_variant is not None:
        summary_parts.append(f"Best tested challenger: {best_variant['display_name']}.")

    if review_queue.empty:
        summary_parts.append("No completed runs are waiting for human review.")
    else:
        summary_parts.append(f"{len(review_queue)} run(s) currently need human review.")

    if top_recommendation is not None:
        summary_parts.append(
            "Top suggested next experiment: "
            f"{top_recommendation.parameter} -> {top_recommendation.suggested_value}."
        )

    return " ".join(summary_parts) or "No completed Observatory run history is available yet."


def _make_tab_button(
    *,
    name: str,
    button_type: str,
    width: int,
    target_index: int,
    tabs: pn.Tabs | None,
) -> pn.widgets.Button:
    """Create a button that activates a dashboard tab when clicked."""
    button = pn.widgets.Button(
        name=name,
        button_type=button_type,
        width=width,
        disabled=tabs is None,
    )

    if tabs is not None:

        def _activate_tab(event: Any) -> None:
            tabs.active = target_index

        button.on_click(_activate_tab)

    return button


def _build_start_here_card(dm: DashboardDataManager, tabs: pn.Tabs | None) -> pn.Card:
    """Render a compact orientation panel with the current situation and workflow nav."""
    current_state = pn.pane.Markdown(f"**Current situation:** {_command_center_summary(dm)}")
    workflow = pn.pane.Markdown(
        "1. Review the situation summary and decision cards below.\n"
        "2. **Start Exploring** to launch an autonomous search with smart defaults, or expand controls to customize.\n"
        "3. Use the tabs to compare challengers, inspect projections, or check diagnostics."
    )
    buttons = pn.FlexBox(
        _make_tab_button(
            name="Scorecards",
            button_type="default",
            width=120,
            target_index=3,
            tabs=tabs,
        ),
        _make_tab_button(
            name="Projections",
            button_type="default",
            width=120,
            target_index=4,
            tabs=tabs,
        ),
        _make_tab_button(
            name="Horizon & Bias",
            button_type="default",
            width=130,
            target_index=5,
            tabs=tabs,
        ),
        _make_tab_button(
            name="Sensitivity",
            button_type="default",
            width=120,
            target_index=6,
            tabs=tabs,
        ),
        flex_wrap="wrap",
        sizing_mode="stretch_width",
        styles={"gap": "8px"},
    )

    return pn.Card(
        current_state,
        workflow,
        buttons,
        title="Start Here",
        sizing_mode="stretch_width",
    )


def _build_quick_start_card(dm: DashboardDataManager) -> pn.Card:
    """Quick launcher with resource controls and smart defaults.

    Exposes only the settings a user needs to allocate system resources and
    scope the search — CPU cores and maximum experiment runs.  Everything
    else (search ID, batch budget, pending/recommended limits, recipe
    inclusion) is derived from the search policy automatically.
    """
    import os

    status_pane = pn.pane.HTML(sizing_mode="stretch_width")
    output_pane = pn.widgets.TextAreaInput(
        value="",
        disabled=True,
        height=120,
        sizing_mode="stretch_width",
        name="",
        visible=False,
    )

    # Derive smart defaults from the search policy
    available_cores = os.cpu_count() or 12
    default_workers = min(12, available_cores)
    default_batch_budget = int(dm.search_policy.default_run_budget)
    default_max_total = 20
    include_recipes = bool(dm.search_policy.include_recipe_catalog)

    # Resource controls — the only inputs on the Quick Start card
    cpu_slider = pn.widgets.IntSlider(
        name="CPU cores to use",
        start=2,
        end=available_cores,
        step=2,
        value=default_workers,
        width=280,
    )
    max_runs_spinner = pn.widgets.IntInput(
        name="Max experiments to run",
        value=default_max_total,
        step=5,
        start=1,
        end=200,
        width=160,
    )

    # Check if there's already an active search session
    active_id = dm.active_search_id
    has_active = active_id is not None and active_id != ""

    if has_active:
        status_pane.object = (
            f'<div style="color:#334E68;font-size:0.92em">'
            f"Active search: <strong>{active_id}</strong> — "
            f"progress updates appear in the Search Progress section below."
            f"</div>"
        )

    def _on_start(event: Any) -> None:
        import time

        search_id = _default_search_id()
        session_root = dm.search_session_root
        session_root.mkdir(parents=True, exist_ok=True)
        pid_path = session_root / f".{search_id}.dashboard.pid"
        log_path = session_root / f".{search_id}.dashboard.log"
        meta_path = session_root / f".{search_id}.dashboard_meta.yaml"
        workers = cpu_slider.value
        total_runs = max_runs_spinner.value
        cmd = [
            sys.executable,
            str(_OBSERVATORY_SCRIPT),
            "search-auto",
            "--search-id",
            search_id,
            "--batch-run-budget",
            str(default_batch_budget),
            "--max-total-runs",
            str(total_runs),
            "--workers-per-run",
            str(workers),
            "--max-pending",
            str(int(dm.search_policy.default_max_pending)),
            "--max-recommended",
            str(int(dm.search_policy.default_max_recommended)),
        ]
        if include_recipes:
            cmd.append("--include-recipe-catalog")
        meta_path.write_text(
            yaml.safe_dump(
                {
                    "search_id": search_id,
                    "launched_at": dt.datetime.now(tz=dt.UTC).isoformat(),
                    "base_revision": "HEAD",
                    "command": " ".join(cmd[2:]),
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        with log_path.open("a", encoding="utf-8") as handle:
            process = subprocess.Popen(  # noqa: S603
                cmd,
                cwd=str(_PROJECT_ROOT),
                stdout=handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        pid_path.write_text(str(process.pid), encoding="utf-8")

        # Wait briefly to detect early failures (e.g. dirty checkout,
        # missing config).  The process typically crashes within 1-2s
        # if it is going to fail at startup.
        time.sleep(2)
        exit_code = process.poll()
        output_pane.visible = True

        if exit_code is not None:
            # Process already exited — read the log for the error
            log_tail = ""
            if log_path.exists():
                lines = log_path.read_text(encoding="utf-8").strip().splitlines()
                log_tail = "\n".join(lines[-15:])
            output_pane.value = (
                f"Search {search_id} failed to start (exit code {exit_code}).\n\n"
                f"--- Log output ---\n{log_tail}"
            )
            status_pane.object = (
                f'<div style="color:#C00;font-weight:600;font-size:0.92em">'
                f"Search <strong>{search_id}</strong> failed to start. "
                f"See details below."
                f"</div>"
            )
        else:
            output_pane.value = (
                f"Launched: {search_id}  (PID {process.pid})\n"
                f"CPU cores: {workers} | Max experiments: {total_runs}\n"
                f"Log: {log_path.relative_to(_PROJECT_ROOT)}\n\n"
                f"Progress will appear in the Search Progress section below."
            )
            status_pane.object = (
                f'<div style="color:#00B050;font-weight:600;font-size:0.92em">'
                f"Search <strong>{search_id}</strong> launched successfully."
                f"</div>"
            )

    btn_start = pn.widgets.Button(
        name="Start Exploring",
        button_type="success",
        width=200,
        height=44,
    )
    btn_start.on_click(_on_start)

    description = pn.pane.Markdown(
        "Set your resource budget and hit **Start Exploring**. "
        "The system will automatically plan, run, and evaluate "
        "projection experiments. Expand **Autonomous Search (Advanced)** "
        "below for session management and fine-grained control.",
        styles={"color": "#5A6C84", "font-size": "0.9em"},
    )

    resource_row = pn.Row(
        cpu_slider,
        pn.Spacer(width=20),
        max_runs_spinner,
        sizing_mode="stretch_width",
    )

    return pn.Card(
        description,
        resource_row,
        pn.Row(btn_start, sizing_mode="stretch_width"),
        status_pane,
        output_pane,
        title="Quick Start",
        sizing_mode="stretch_width",
    )


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
            "Champion Error (MAPE)",
            f"{_champion_mape(dm):.2f}%" if _champion_mape(dm) is not None else "N/A",
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
    if not dm.run_metadata.empty and "status_code" in dm.run_metadata.columns:
        review_queue = dm.run_metadata[dm.run_metadata["status_code"] == "needs_human_review"]
    else:
        review_queue = dm.run_metadata.iloc[:0]

    recommendations = dm.recommender.suggest_next_experiments(1)
    top_recommendation = recommendations[0] if recommendations else None

    champion_card = _summary_card(
        "Current Champion",
        f"{champion_mape:.2f}% county error"
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
            f"{best_variant_mape:.2f}% county error ({delta:+.2f} vs champion)",
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


def _queue_health_snapshot(dm: DashboardDataManager) -> dict[str, Any]:
    """Return the queue-health metrics surfaced on the command center."""
    inventory = {}
    if dm.catalog is not None:
        try:
            inventory = dm.catalog.get_inventory_summary()
        except Exception:  # pragma: no cover - defensive UI guard
            logger.exception("Failed to read catalog inventory summary.")
            inventory = {}

    review_queue = (
        int((dm.run_metadata["status_code"] == "needs_human_review").sum())
        if not dm.run_metadata.empty and "status_code" in dm.run_metadata.columns
        else 0
    )

    recommendations = []
    try:
        recommendations = dm.recommender.suggest_next_experiments(5)
    except Exception:  # pragma: no cover - defensive UI guard
        logger.exception("Failed to read recommendation queue.")

    runnable_recommendations = sum(
        1 for rec in recommendations if not getattr(rec, "requires_code_change", False)
    )

    return {
        "untested_runnable": int(inventory.get("untested_runnable", 0) or 0),
        "untested_requires_code_change": int(
            inventory.get("untested_requires_code_change", 0) or 0
        ),
        "grid_blocked": int(inventory.get("grid_blocked", 0) or 0),
        "grid_blocked_ids": list(inventory.get("grid_blocked_ids", []) or []),
        "review_queue": review_queue,
        "runnable_recommendations": runnable_recommendations,
    }


def _build_queue_health_card(dm: DashboardDataManager) -> pn.Card:
    """Queue readiness panel for unattended-run planning."""
    snapshot = _queue_health_snapshot(dm)
    blocked_grids = snapshot["grid_blocked_ids"]

    lines = [
        f"Variants you can run now: {snapshot['untested_runnable']}",
        f"Variants still blocked by code changes: {snapshot['untested_requires_code_change']}",
        f"Runs waiting for human review: {snapshot['review_queue']}",
        f"Config-only recommendations ready to preview: {snapshot['runnable_recommendations']}",
    ]
    if blocked_grids:
        lines.append(f"Blocked grids: {', '.join(blocked_grids)}")
    else:
        lines.append("Blocked grids: none")

    command_md = (
        "Advanced CLI loop:\n"
        "1. `python scripts/analysis/observatory.py status`\n"
        "2. `python scripts/analysis/observatory.py run-pending --dry-run --run-budget 3 --resume-file data/analysis/experiments/sweeps/observatory_pending_resume.json`\n"
        "3. Re-run without `--dry-run` once the queue looks correct.\n"
        "4. Review `needs_human_review` runs before any SOP-003 promotion decision."
    )

    tone = "warning" if blocked_grids or snapshot["review_queue"] else "success"
    alert_type = "warning" if tone == "warning" else "success"
    headline = (
        "Attention needed before unattended queueing."
        if tone == "warning"
        else "Queue looks runnable within current guardrails."
    )

    return pn.Card(
        pn.pane.Alert(headline, alert_type=alert_type),
        pn.pane.Markdown(
            "Use this section when deciding whether the unattended queue is safe to run. "
            "If you only need to inspect what would happen next, the preview buttons below do not execute experiments."
        ),
        pn.pane.Markdown("\n".join(f"- {line}" for line in lines)),
        pn.pane.Markdown(command_md),
        title="Queue Health",
        collapsed=True,
        sizing_mode="stretch_width",
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

    champion_row = champ_rows[champ_rows["status_at_run"].fillna("").str.lower() == "champion"]
    champion = champion_row.iloc[0] if not champion_row.empty else champ_rows.iloc[0]

    sentinel_cols = [c for c in champion.index if c.startswith("sentinel_")]
    sentinel_rows = []
    for col in sentinel_cols:
        value = champion.get(col)
        if pd.notna(value):
            label = col.replace("sentinel_", "").replace("_mape", "").replace("_", " ").title()
            sentinel_rows.append((label, float(value)))
    sentinel_rows = sorted(sentinel_rows, key=lambda item: item[1], reverse=True)

    sentinel_html = ""
    if sentinel_rows:
        sentinel_html = (
            '<h4 style="margin:12px 0 6px 0;color:#1F3864">Highest Sentinel County Errors (MAPE)</h4>'
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
        <tr><td style="padding:4px 12px 4px 0;color:#5A6C84">Method</td><td style="padding:4px 0;font-weight:600">{champion.get("method_id", "N/A")}</td></tr>
        <tr><td style="padding:4px 12px 4px 0;color:#5A6C84">Config</td><td style="padding:4px 0;font-weight:600">{champion.get("config_id", "N/A")}</td></tr>
        <tr><td style="padding:4px 12px 4px 0;color:#5A6C84">County Error (MAPE)</td><td style="padding:4px 0;font-weight:700;color:{SDC_NAVY}">{_fmt_metric(champion.get("county_mape_overall"))}</td></tr>
        <tr><td style="padding:4px 12px 4px 0;color:#5A6C84">Recent State Error (APE, short)</td><td style="padding:4px 0;font-weight:600">{_fmt_metric(champion.get("state_ape_recent_short"))}</td></tr>
        <tr><td style="padding:4px 12px 4px 0;color:#5A6C84">Recent State Error (APE, medium)</td><td style="padding:4px 0;font-weight:600">{_fmt_metric(champion.get("state_ape_recent_medium"))}</td></tr>
      </table>
      {sentinel_html}
    </div>
    """
    return pn.Card(
        pn.pane.HTML(html, sizing_mode="stretch_width"),
        title="Champion Snapshot",
        collapsed=True,
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
    display_df["review_status"] = display_df["status_label"]
    display_df["config"] = display_df["short_config"].replace("", pd.NA)
    display_df["county_error_mape"] = display_df["selected_county_mape_overall"].round(3)
    display_df["recent_state_error_ape"] = display_df["selected_state_ape_recent_short"].round(3)
    display_df["run_date"] = display_df["run_date_label"]

    columns = [
        column
        for column in [
            "run",
            "review_status",
            "run_date",
            "selected_method_id",
            "config",
            "county_error_mape",
            "recent_state_error_ape",
            "next_action",
            "run_id",
        ]
        if column in display_df.columns
    ]
    display_df = display_df[columns].rename(
        columns={
            "selected_method_id": "method",
            "next_action": "recommended_next_step",
            "run_id": "run_id",
        }
    )

    return pn.Card(
        metric_table(
            display_df,
            page_size=8,
            frozen_columns=["run"],
        ),
        title="Run Index",
        collapsed=True,
        sizing_mode="stretch_width",
    )


def _derive_parallelism(core_budget: int) -> tuple[int, int]:
    """Map a core budget to (parallel_runs, workers_per_run).

    Below 14 cores: single run, all cores as workers.
    14+ cores: two parallel runs sharing the budget evenly.
    """
    parallel_runs = 1 if core_budget < 14 else 2
    workers_per_run = max(1, core_budget // parallel_runs)
    return parallel_runs, workers_per_run


def _core_alloc_label(core_budget: int) -> str:
    parallel_runs, workers_per_run = _derive_parallelism(core_budget)
    if parallel_runs == 1:
        return f"→ {workers_per_run} workers, 1 run at a time"
    return f"→ {parallel_runs} parallel runs × {workers_per_run} workers = {parallel_runs * workers_per_run} cores"


def _build_action_buttons(dm: DashboardDataManager) -> pn.Card:
    """Action row and output console."""
    # CPU core allocation slider
    cpu_budget = pn.widgets.IntSlider(
        name="CPU Core Budget",
        start=2,
        end=24,
        step=2,
        value=12,
        width=320,
    )
    alloc_label = pn.pane.Markdown(
        _core_alloc_label(cpu_budget.value),
        styles={"color": "#555", "margin-top": "4px"},
    )

    def _on_cpu_budget_change(event: Any) -> None:
        alloc_label.object = _core_alloc_label(event.new)

    cpu_budget.param.watch(_on_cpu_budget_change, "value")

    output_pane = pn.widgets.TextAreaInput(
        value="Use the preview actions below to inspect what would run next. Launch buttons run in the background and return immediately.",
        disabled=True,
        height=220,
        sizing_mode="stretch_width",
        name="Output",
    )

    def on_refresh(event: Any) -> None:
        dm.refresh()
        output_pane.value = "Data refreshed successfully."

    def on_run_pending(event: Any) -> None:
        output_pane.value = (
            "Running 'run-pending --dry-run --run-budget 3 --resume-file "
            "data/analysis/experiments/sweeps/observatory_pending_resume.json'...\n"
        )
        output_pane.value = _run_observatory_command(
            [
                "run-pending",
                "--dry-run",
                "--run-budget",
                "3",
                "--resume-file",
                "data/analysis/experiments/sweeps/observatory_pending_resume.json",
            ]
        )

    def on_run_recommended(event: Any) -> None:
        output_pane.value = (
            "Running 'run-recommended --dry-run --run-budget 2 --resume-file "
            "data/analysis/experiments/sweeps/observatory_recommended_resume.json'...\n"
        )
        output_pane.value = _run_observatory_command(
            [
                "run-recommended",
                "--dry-run",
                "--run-budget",
                "2",
                "--resume-file",
                "data/analysis/experiments/sweeps/observatory_recommended_resume.json",
            ]
        )

    def _launch_sweep(subcommand: str, resume_filename: str) -> None:
        """Launch run-pending or run-recommended as a background subprocess."""
        parallel_runs, workers_per_run = _derive_parallelism(cpu_budget.value)
        sweeps_dir = _PROJECT_ROOT / "data" / "analysis" / "experiments" / "sweeps"
        sweeps_dir.mkdir(parents=True, exist_ok=True)
        import datetime as _dt

        stamp = _dt.datetime.now(tz=_dt.UTC).strftime("%Y%m%dT%H%M%SZ")
        log_path = sweeps_dir / "logs" / f"{subcommand}_{stamp}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        resume_file = sweeps_dir / resume_filename
        cmd = [
            sys.executable,
            str(_OBSERVATORY_SCRIPT),
            subcommand,
            "--parallel-runs",
            str(parallel_runs),
            "--workers-per-run",
            str(workers_per_run),
            "--resume-file",
            str(resume_file),
        ]
        with log_path.open("a", encoding="utf-8") as handle:
            process = subprocess.Popen(  # noqa: S603
                cmd,
                cwd=str(_PROJECT_ROOT),
                stdout=handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        output_pane.value = (
            f"Launched {subcommand} in the background (PID {process.pid}).\n"
            f"CPU budget: {cpu_budget.value} cores "
            f"({parallel_runs} parallel run(s) × {workers_per_run} workers)\n"
            f"Log: {log_path.relative_to(_PROJECT_ROOT)}\n"
            f"Resume: {resume_file.relative_to(_PROJECT_ROOT)}\n\n"
            f"Command: {' '.join(cmd[2:])}"
        )

    def on_launch_pending(event: Any) -> None:
        _launch_sweep("run-pending", "observatory_pending_resume.json")

    def on_launch_recommended(event: Any) -> None:
        _launch_sweep("run-recommended", "observatory_recommended_resume.json")

    btn_refresh = pn.widgets.Button(
        name="Refresh Data",
        button_type="primary",
        width=150,
    )
    btn_refresh.on_click(on_refresh)

    btn_pending = pn.widgets.Button(
        name="Preview Runnable Queue",
        button_type="warning",
        width=190,
    )
    btn_pending.on_click(on_run_pending)

    btn_recommended = pn.widgets.Button(
        name="Preview Recommended Queue",
        button_type="warning",
        width=220,
    )
    btn_recommended.on_click(on_run_recommended)

    btn_launch_pending = pn.widgets.Button(
        name="Launch Run-Pending",
        button_type="success",
        width=180,
    )
    btn_launch_pending.on_click(on_launch_pending)

    btn_launch_recommended = pn.widgets.Button(
        name="Launch Run-Recommended",
        button_type="success",
        width=210,
    )
    btn_launch_recommended.on_click(on_launch_recommended)

    cpu_row = pn.Row(
        pn.Column(cpu_budget, alloc_label),
        sizing_mode="stretch_width",
    )

    preview_buttons = pn.FlexBox(
        btn_refresh,
        btn_pending,
        btn_recommended,
        flex_wrap="wrap",
        sizing_mode="stretch_width",
        styles={"gap": "10px"},
    )

    launch_buttons = pn.FlexBox(
        btn_launch_pending,
        btn_launch_recommended,
        flex_wrap="wrap",
        sizing_mode="stretch_width",
        styles={"gap": "10px"},
    )

    guidance = pn.pane.Markdown(
        "**Preview** actions show what would run without executing anything. "
        "**Launch** buttons start a real background sweep using the CPU budget above — "
        "check the log file and progress JSON in `data/analysis/experiments/sweeps/` to monitor progress. "
        "Run `observatory.py refresh && observatory.py compare` when done."
    )

    return pn.Card(
        section_header("CPU Resources"),
        cpu_row,
        section_header("Preview"),
        preview_buttons,
        section_header("Launch (Background)"),
        launch_buttons,
        guidance,
        output_pane,
        title="Manual Sweep Actions (Advanced)",
        collapsed=True,
        sizing_mode="stretch_width",
    )


def _build_search_progress_card(dm: DashboardDataManager) -> pn.Card:
    """Live-progress view for active search sessions.

    Always visible on the Command Center.  Shows the active session's
    progress bar, process health, best candidates, and a live log tail.
    """
    session_select = pn.widgets.Select(
        name="Search Session",
        options=dm.search_session_option_map() or {"No sessions yet": ""},
        value=dm.active_search_id or "",
        sizing_mode="stretch_width",
    )
    progress_pane = pn.pane.HTML(sizing_mode="stretch_width")
    best_candidates_box = pn.Column(sizing_mode="stretch_width")
    log_tail_pane = pn.widgets.TextAreaInput(
        value="",
        disabled=True,
        height=160,
        sizing_mode="stretch_width",
        name="Live Log",
    )
    current_experiment_pane = pn.pane.HTML(sizing_mode="stretch_width")

    def _selected_session_row() -> pd.Series | None:
        sessions = dm.search_sessions
        if sessions.empty:
            return None
        search_id = session_select.value or dm.active_search_id
        if search_id:
            matches = sessions[sessions["search_id"] == search_id]
            if not matches.empty:
                return matches.iloc[0]
        return sessions.iloc[0]

    def _current_experiment_html(
        search_id: str,
        session_row: pd.Series | None,
    ) -> str:
        """Render info about the currently running experiment, if any."""
        if session_row is None:
            return ""
        process_status = str(session_row.get("process_status", "unknown"))
        is_running = bool(session_row.get("dashboard_process_running", False))
        running_count = int(session_row.get("running", 0) or 0)
        completed = int(session_row.get("completed", 0) or 0)
        failed = int(session_row.get("failed", 0) or 0)
        total = int(session_row.get("total", 0) or 0)

        # Process health indicator
        if is_running:
            indicator = '<span style="color:#00B050;font-weight:600">&#9679; Process running</span>'
        elif process_status == "stopped" and completed + failed >= total > 0:
            indicator = '<span style="color:#0563C1;font-weight:600">&#9632; Search complete</span>'
        elif process_status == "stopped":
            indicator = '<span style="color:#C00;font-weight:600">&#9632; Process stopped</span>'
        else:
            indicator = f'<span style="color:#5A6C84">Process: {process_status}</span>'

        # Currently running experiments
        activity = ""
        if running_count > 0:
            candidates = dm.search_session_candidates(search_id)
            if not candidates.empty and "status" in candidates.columns:
                active = candidates[candidates["status"].str.lower() == "running"]
                if not active.empty:
                    names = active["candidate_id"].tolist()[:3]
                    activity = (
                        f'<div style="margin-top:6px;color:#334E68;font-size:0.9em">'
                        f"Running now: <strong>{', '.join(str(n) for n in names)}</strong>"
                        f"{'...' if len(active) > 3 else ''}"
                        f"</div>"
                    )

        # Elapsed time
        elapsed = ""
        created = str(session_row.get("created_at", "") or "")
        if created:
            try:
                start = dt.datetime.fromisoformat(created)
                if start.tzinfo is None:
                    start = start.replace(tzinfo=dt.UTC)
                delta = dt.datetime.now(tz=dt.UTC) - start
                hours, remainder = divmod(int(delta.total_seconds()), 3600)
                mins, secs = divmod(remainder, 60)
                elapsed = f"{hours}h {mins}m elapsed" if hours > 0 else f"{mins}m {secs}s elapsed"
            except (ValueError, TypeError):
                pass

        return f"""
        <div style="font-family:'Aptos','Segoe UI',Arial,sans-serif;
                    display:flex;justify-content:space-between;align-items:center;
                    flex-wrap:wrap;gap:8px;margin-bottom:4px">
            {indicator}
            <span style="color:#5A6C84;font-size:0.85em">{elapsed}</span>
        </div>
        {activity}
        """

    def _refresh_progress(*, prefer_search_id: str | None = None) -> None:
        dm.refresh_search_sessions()
        options = dm.search_session_option_map()
        if not options:
            session_select.options = {"No sessions yet": ""}
            session_select.value = ""
            progress_pane.object = _search_progress_html(None)
            current_experiment_pane.object = ""
            best_candidates_box[:] = [
                empty_placeholder("No search results yet. Use Quick Start above to begin.")
            ]
            log_tail_pane.value = ""
            return

        session_select.options = options
        desired = (
            prefer_search_id
            or session_select.value
            or dm.active_search_id
            or next(iter(options.values()))
        )
        if desired not in options.values():
            desired = next(iter(options.values()))
        session_select.value = desired

        selected = _selected_session_row()
        progress_pane.object = _search_progress_html(selected)

        selected_search_id = str(selected["search_id"]) if selected is not None else ""

        # Current experiment and process health
        current_experiment_pane.object = _current_experiment_html(
            selected_search_id,
            selected,
        )

        # Best candidates
        best_candidates = (
            dm.search_session_best_candidates(selected_search_id)
            if selected_search_id
            else pd.DataFrame()
        )
        if best_candidates.empty:
            best_candidates_box[:] = [empty_placeholder("No completed candidates yet.")]
        else:
            best_cols = [
                col
                for col in [
                    "candidate_id",
                    "outcome",
                    "county_mape_overall",
                    "delta_county_mape_overall",
                ]
                if col in best_candidates.columns
            ]
            best_candidates_box[:] = [
                metric_table(
                    best_candidates[best_cols],
                    page_size=5,
                    frozen_columns=["candidate_id"],
                )
            ]

        # Live log tail
        log_text = (
            dm.search_session_log_tail(selected_search_id, max_chars=4000)
            if selected_search_id
            else ""
        )
        log_tail_pane.value = log_text

    def _on_stop(event: Any) -> None:
        search_id = (session_select.value or "").strip()
        if search_id:
            dm.stop_search_session(search_id)
            _refresh_progress(prefer_search_id=search_id)

    def _on_refresh(event: Any) -> None:
        _refresh_progress()

    btn_stop = pn.widgets.Button(name="Stop", button_type="danger", width=80)
    btn_stop.on_click(_on_stop)
    btn_refresh = pn.widgets.Button(name="Refresh", button_type="default", width=90)
    btn_refresh.on_click(_on_refresh)

    # Auto-refresh every 5 seconds
    if pn.state.curdoc is not None:
        pn.state.add_periodic_callback(_refresh_progress, period=5000, start=True)

    session_select.param.watch(
        lambda event: _refresh_progress(prefer_search_id=str(event.new)), "value"
    )
    _refresh_progress(prefer_search_id=dm.active_search_id)

    return pn.Card(
        pn.Row(session_select, btn_refresh, btn_stop, sizing_mode="stretch_width"),
        current_experiment_pane,
        progress_pane,
        best_candidates_box,
        log_tail_pane,
        title="Search Progress",
        sizing_mode="stretch_width",
    )


def _build_autonomous_search_card(dm: DashboardDataManager) -> pn.Card:
    """Full autonomous-search controls — starts collapsed for reduced clutter.

    Power users expand this card to customize search parameters, preview
    plans, inspect session details, and view full candidate/report output.
    """
    session_select = pn.widgets.Select(
        name="Search Session",
        options=dm.search_session_option_map() or {"No sessions yet": ""},
        value=dm.active_search_id or "",
        sizing_mode="stretch_width",
    )
    search_id_input = pn.widgets.TextInput(
        name="Launch Search ID",
        value=_default_search_id(),
        sizing_mode="stretch_width",
    )
    cpu_budget_search = pn.widgets.IntSlider(
        name="CPU Core Budget",
        start=2,
        end=24,
        step=2,
        value=12,
        width=320,
    )
    cpu_budget_search_label = pn.pane.Markdown(
        f"→ {cpu_budget_search.value} workers per experiment (search-auto runs one experiment at a time)",
        styles={"color": "#555", "margin-top": "4px"},
    )

    def _on_cpu_search_change(event: Any) -> None:
        cpu_budget_search_label.object = (
            f"→ {event.new} workers per experiment (search-auto runs one experiment at a time)"
        )

    cpu_budget_search.param.watch(_on_cpu_search_change, "value")

    batch_run_budget = pn.widgets.IntInput(
        name="Batch Run Budget",
        value=int(dm.search_policy.default_run_budget),
        step=1,
        start=1,
        sizing_mode="stretch_width",
    )
    max_total_runs = pn.widgets.IntInput(
        name="Max Total Runs",
        value=20,
        step=1,
        start=1,
        sizing_mode="stretch_width",
    )
    max_pending = pn.widgets.IntInput(
        name="Max Pending",
        value=int(dm.search_policy.default_max_pending),
        step=1,
        start=0,
        sizing_mode="stretch_width",
    )
    max_recommended = pn.widgets.IntInput(
        name="Max Recommended",
        value=int(dm.search_policy.default_max_recommended),
        step=1,
        start=0,
        sizing_mode="stretch_width",
    )
    overwrite_box = pn.widgets.Checkbox(name="Overwrite existing session", value=False)
    include_recipes_box = pn.widgets.Checkbox(
        name="Include recipe catalog",
        value=bool(dm.search_policy.include_recipe_catalog),
    )

    detail_pane = pn.pane.HTML(sizing_mode="stretch_width")
    sessions_table_box = pn.Column(sizing_mode="stretch_width")
    candidates_table_box = pn.Column(sizing_mode="stretch_width")
    report_preview_box = pn.Column(sizing_mode="stretch_width")
    log_preview = pn.widgets.TextAreaInput(
        value="",
        disabled=True,
        height=220,
        sizing_mode="stretch_width",
        name="Launch Log Tail",
    )
    output_pane = pn.widgets.TextAreaInput(
        value="",
        disabled=True,
        height=220,
        sizing_mode="stretch_width",
        name="Output",
    )

    def _planner_args() -> list[str]:
        return [
            "--max-pending",
            str(max_pending.value),
            "--max-recommended",
            str(max_recommended.value),
        ]

    def _selected_session_row() -> pd.Series | None:
        sessions = dm.search_sessions
        if sessions.empty:
            return None
        search_id = session_select.value or dm.active_search_id
        if search_id:
            matches = sessions[sessions["search_id"] == search_id]
            if not matches.empty:
                return matches.iloc[0]
        return sessions.iloc[0]

    def _refresh_search_views(*, prefer_search_id: str | None = None) -> None:
        dm.refresh_search_sessions()
        sessions = dm.search_sessions

        options = dm.search_session_option_map()
        if not options:
            session_select.options = {"No sessions yet": ""}
            session_select.value = ""
            detail_pane.object = _search_session_detail_html(None)
            sessions_table_box[:] = [empty_placeholder("No autonomous-search sessions found.")]
            candidates_table_box[:] = [empty_placeholder("No candidate preview available yet.")]
            report_preview_box[:] = [empty_placeholder("No search report is available yet.")]
            log_preview.value = ""
            return

        session_select.options = options
        desired = (
            prefer_search_id
            or session_select.value
            or dm.active_search_id
            or next(iter(options.values()))
        )
        if desired not in options.values():
            desired = next(iter(options.values()))
        session_select.value = desired

        selected = _selected_session_row()
        detail_pane.object = _search_session_detail_html(selected)

        display_sessions = sessions.copy()
        display_sessions["progress"] = (
            display_sessions["progress_count"].astype(int).astype(str)
            + "/"
            + display_sessions["total"].astype(int).astype(str)
        )
        display_sessions["progress_pct"] = display_sessions["progress_pct"].round(1)
        session_cols = [
            col
            for col in [
                "search_id",
                "status",
                "progress",
                "progress_pct",
                "running",
                "planned",
                "failed",
                "updated_at",
            ]
            if col in display_sessions.columns
        ]
        sessions_table_box[:] = [
            metric_table(
                display_sessions[session_cols].rename(
                    columns={
                        "search_id": "search",
                        "progress_pct": "progress_pct",
                        "updated_at": "updated",
                    }
                ),
                page_size=5,
                frozen_columns=["search"],
            )
        ]

        selected_search_id = str(selected["search_id"]) if selected is not None else ""
        candidates = (
            dm.search_session_candidates(selected_search_id)
            if selected_search_id
            else pd.DataFrame()
        )
        if candidates.empty:
            candidates_table_box[:] = [
                empty_placeholder(
                    "No candidate summary is available for the selected search session yet."
                )
            ]
        else:
            candidate_cols = [
                col
                for col in [
                    "candidate_id",
                    "source",
                    "execution_mode",
                    "status",
                    "outcome",
                    "run_id",
                    "primary_metric_name",
                    "county_mape_overall",
                    "delta_county_mape_overall",
                ]
                if col in candidates.columns
            ]
            candidates_table_box[:] = [
                metric_table(
                    candidates[candidate_cols],
                    page_size=8,
                    frozen_columns=["candidate_id"],
                )
            ]

        report_text = (
            dm.search_session_report_markdown(selected_search_id) if selected_search_id else ""
        )
        observatory_report_path = (
            str(selected.get("observatory_report_html", "") or "") if selected is not None else ""
        )
        if report_text:
            report_preview_box[:] = [
                pn.pane.Markdown(report_text, sizing_mode="stretch_width"),
                pn.pane.Markdown(
                    f"Observatory HTML report: `{observatory_report_path or 'not written yet'}`"
                ),
            ]
        else:
            report_preview_box[:] = [
                empty_placeholder(
                    "No Markdown search report is available for the selected session yet."
                ),
                pn.pane.Markdown(
                    f"Observatory HTML report: `{observatory_report_path or 'not written yet'}`"
                ),
            ]

        log_preview.value = (
            dm.search_session_log_tail(selected_search_id) if selected_search_id else ""
        )

    def _preview_plan(event: Any) -> None:
        search_id = search_id_input.value.strip() or _default_search_id()
        args = ["search-plan", "--search-id", search_id, *_planner_args()]
        if include_recipes_box.value:
            args.append("--include-recipe-catalog")
        if overwrite_box.value:
            args.append("--overwrite")
        output_pane.value = _run_observatory_command(args)
        _refresh_search_views(prefer_search_id=search_id)

    def _preview_next_batch(event: Any) -> None:
        search_id = (session_select.value or search_id_input.value).strip()
        if not search_id:
            output_pane.value = "Select or create a search session first."
            return
        output_pane.value = _run_observatory_command(
            [
                "search-run",
                "--search-id",
                search_id,
                "--run-budget",
                str(batch_run_budget.value),
                "--dry-run",
            ]
        )
        _refresh_search_views(prefer_search_id=search_id)

    def _launch_search(event: Any) -> None:
        search_id = search_id_input.value.strip() or _default_search_id()
        session_root = dm.search_session_root
        session_root.mkdir(parents=True, exist_ok=True)
        pid_path = session_root / f".{search_id}.dashboard.pid"
        log_path = session_root / f".{search_id}.dashboard.log"
        meta_path = session_root / f".{search_id}.dashboard_meta.yaml"
        cmd = [
            sys.executable,
            str(_OBSERVATORY_SCRIPT),
            "search-auto",
            "--search-id",
            search_id,
            "--batch-run-budget",
            str(batch_run_budget.value),
            "--max-total-runs",
            str(max_total_runs.value),
            "--workers-per-run",
            str(cpu_budget_search.value),
            *_planner_args(),
        ]
        if include_recipes_box.value:
            cmd.append("--include-recipe-catalog")
        if overwrite_box.value:
            cmd.append("--overwrite")
        meta_path.write_text(
            yaml.safe_dump(
                {
                    "search_id": search_id,
                    "launched_at": dt.datetime.now(tz=dt.UTC).isoformat(),
                    "base_revision": "HEAD",
                    "command": " ".join(cmd[2:]),
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        with log_path.open("a", encoding="utf-8") as handle:
            process = subprocess.Popen(  # noqa: S603
                cmd,
                cwd=str(_PROJECT_ROOT),
                stdout=handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        pid_path.write_text(str(process.pid), encoding="utf-8")
        output_pane.value = (
            f"Launched autonomous search in the background: {search_id}\n"
            f"Command: {' '.join(cmd[2:])}\n"
            "Use the Search Progress card or Refresh button to watch progress."
        )
        session_select.value = search_id
        _refresh_search_views(prefer_search_id=search_id)
        search_id_input.value = _default_search_id()

    def _refresh_only(event: Any) -> None:
        _refresh_search_views()
        output_pane.value = "Autonomous-search views refreshed from session files."

    def _stop_search(event: Any) -> None:
        search_id = (session_select.value or "").strip()
        if not search_id:
            output_pane.value = "Select a dashboard-launched search session first."
            return
        output_pane.value = dm.stop_search_session(search_id)
        _refresh_search_views(prefer_search_id=search_id)

    btn_refresh = pn.widgets.Button(name="Refresh", button_type="primary", width=100)
    btn_refresh.on_click(_refresh_only)
    btn_preview_plan = pn.widgets.Button(name="Preview Plan", button_type="warning", width=140)
    btn_preview_plan.on_click(_preview_plan)
    btn_preview_batch = pn.widgets.Button(
        name="Preview Next Batch", button_type="warning", width=170
    )
    btn_preview_batch.on_click(_preview_next_batch)
    btn_launch = pn.widgets.Button(name="Launch Search-Auto", button_type="success", width=180)
    btn_launch.on_click(_launch_search)
    btn_stop = pn.widgets.Button(name="Stop Search", button_type="danger", width=140)
    btn_stop.on_click(_stop_search)

    session_select.param.watch(
        lambda event: _refresh_search_views(prefer_search_id=str(event.new)), "value"
    )
    _refresh_search_views(prefer_search_id=dm.active_search_id)

    cpu_search_row = pn.Column(
        cpu_budget_search,
        cpu_budget_search_label,
        sizing_mode="stretch_width",
    )

    controls = pn.FlexBox(
        search_id_input,
        batch_run_budget,
        max_total_runs,
        max_pending,
        max_recommended,
        cpu_search_row,
        flex_wrap="wrap",
        sizing_mode="stretch_width",
        styles={"gap": "10px"},
    )
    toggles = pn.FlexBox(
        overwrite_box,
        include_recipes_box,
        flex_wrap="wrap",
        sizing_mode="stretch_width",
        styles={"gap": "18px"},
    )
    buttons = pn.FlexBox(
        btn_refresh,
        btn_preview_plan,
        btn_preview_batch,
        btn_launch,
        btn_stop,
        flex_wrap="wrap",
        sizing_mode="stretch_width",
        styles={"gap": "10px"},
    )

    return pn.Card(
        session_select,
        detail_pane,
        section_header("Launch Settings"),
        controls,
        toggles,
        buttons,
        output_pane,
        section_header("All Sessions"),
        sessions_table_box,
        section_header("Candidate Details"),
        candidates_table_box,
        section_header("Search Report"),
        report_preview_box,
        section_header("Log Tail"),
        log_preview,
        title="Autonomous Search (Advanced)",
        collapsed=True,
        sizing_mode="stretch_width",
    )


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
        weaknesses["best_challenger_delta"].notna() & (weaknesses["best_challenger_delta"] >= 0)
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
        collapsed=True,
        sizing_mode="stretch_width",
    )


def build_command_center(
    dm: DashboardDataManager,
    tabs: pn.Tabs | None = None,
) -> pn.Column:
    """Build the dashboard home page.

    Parameters
    ----------
    dm:
        Dashboard data manager supplying run metadata, scorecards, and
        recommendations.
    tabs:
        Optional parent tab layout. When provided, the start-here buttons can
        switch directly to the related dashboard tabs.

    Returns
    -------
    pn.Column
        The assembled command-center layout.
    """
    return pn.Column(
        section_header(
            "Command Center",
            subtitle=(
                "Compare projection variants, inspect decision evidence, and "
                "decide what to run or promote next."
            ),
        ),
        # -- Above the fold: orientation, situation, and one-click action --
        _build_start_here_card(dm, tabs),
        _build_kpi_row(dm),
        _build_decision_strip(dm),
        _build_quick_start_card(dm),
        # -- Live progress for active searches --
        _build_search_progress_card(dm),
        # -- Everything below is collapsed by default (progressive disclosure) --
        _build_autonomous_search_card(dm),
        _build_queue_health_card(dm),
        _build_champion_card(dm),
        _build_index_table(dm),
        _build_action_buttons(dm),
        _build_weaknesses_panel(dm),
        sizing_mode="stretch_width",
    )
