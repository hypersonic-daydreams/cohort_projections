"""Command Center tab for the Observatory dashboard.

This is the dashboard home page. It prioritizes one primary action path:
understand the current session outcome, launch the next search safely, and
route the user into guided review only when the evidence is ready.
"""

from __future__ import annotations

import datetime as dt
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any, TypedDict

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
    layout_mode_classes,
)
from cohort_projections.analysis.observatory.dashboard.widgets import (
    candidate_feed,
    empty_placeholder,
    hero_metric,
    kpi_card,
    markdown_card,
    metric_table,
    progress_ring,
    section_header,
    terminal_output,
)
from cohort_projections.analysis.observatory.decision_support import (
    build_search_session_summary,
)

logger = logging.getLogger(__name__)

_OBSERVATORY_SCRIPT = (
    Path(__file__).resolve().parents[4] / "scripts" / "analysis" / "observatory.py"
)
_PROJECT_ROOT = Path(__file__).resolve().parents[4]


class _LaunchPreset(TypedDict):
    """Structured settings for one simple launch preset."""

    cpu_budget: int
    max_total_runs: int
    batch_run_budget: int
    summary: str


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


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


def _best_tested_challenger(dm: DashboardDataManager) -> pd.Series | None:
    """Return the strongest non-champion benchmark row for summary cards."""
    if dm.run_metadata.empty:
        return None
    if "selected_county_mape_overall" not in dm.run_metadata.columns:
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


def _derive_parallelism(core_budget: int) -> tuple[int, int]:
    """Map a core budget to (parallel_runs, workers_per_run).

    Below 14 cores: single run, all cores as workers.
    14+ cores: two parallel runs sharing the budget evenly.
    """
    parallel_runs = 1 if core_budget < 14 else 2
    workers_per_run = max(1, core_budget // parallel_runs)
    return parallel_runs, workers_per_run


def _core_alloc_label(core_budget: int) -> str:
    """Return a human-readable label for the CPU allocation."""
    parallel_runs, workers_per_run = _derive_parallelism(core_budget)
    if parallel_runs == 1:
        return f"-> {workers_per_run} workers, 1 run at a time"
    return f"-> {parallel_runs} parallel runs x {workers_per_run} workers = {parallel_runs * workers_per_run} cores"


# ---------------------------------------------------------------------------
# Tested utility functions (kept unchanged for backward compatibility)
# ---------------------------------------------------------------------------


def _search_progress_html(session_row: pd.Series | None) -> str:
    """Render a simple HTML progress bar for one autonomous-search session.

    Retained as a tested helper even though the UI now uses ``progress_ring``.
    """
    if session_row is None:
        return "<div><strong>No autonomous-search session selected.</strong></div>"

    progress_pct = float(session_row.get("progress_pct", 0.0) or 0.0)
    status = str(session_row.get("status", "unknown") or "unknown")
    total = int(session_row.get("total", 0) or 0)
    planned = int(session_row.get("planned", 0) or 0)
    running = int(session_row.get("running", 0) or 0)
    completed = int(session_row.get("completed", 0) or 0)
    failed = int(session_row.get("failed", 0) or 0)
    is_stopped = not bool(session_row.get("dashboard_process_running", False))

    if is_stopped and failed > 0 and completed == 0:
        tone = "#C00000"
        status_label = "STOPPED"
        status_color = "#C00000"
    elif failed > 0:
        tone = STATUS_COLORS.get("needs_human_review", "#FFC000")
        status_label = status.upper()
        status_color = "#5A6C84"
    else:
        tone = STATUS_COLORS.get(status, SDC_BLUE)
        status_label = status.upper()
        status_color = "#5A6C84"

    failed_span = (
        f'<span style="color:#C00000;font-weight:600">Failed: {failed}</span>'
        if failed > 0
        else f"<span>Failed: {failed}</span>"
    )
    completed_span = (
        f'<span style="color:#00B050;font-weight:600">Completed: {completed}</span>'
        if completed > 0
        else f"<span>Completed: {completed}</span>"
    )

    return f"""
    <div style="font-family:'Aptos','Segoe UI',Arial,sans-serif">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px">
        <strong>{session_row.get("search_id", "")}</strong>
        <span style="color:{status_color};text-transform:uppercase;font-size:12px;
               font-weight:600">{status_label}</span>
      </div>
      <div style="width:100%;height:14px;background:#E7ECF3;border-radius:999px;overflow:hidden">
        <div style="width:{progress_pct:.1f}%;height:14px;background:{tone};border-radius:999px"></div>
      </div>
      <div style="display:flex;justify-content:space-between;gap:12px;flex-wrap:wrap;margin-top:8px;font-size:13px;color:#334E68">
        <span>Progress: {completed + failed}/{total}</span>
        {completed_span}
        {failed_span}
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


def _command_center_summary(dm: DashboardDataManager) -> str:
    """Return a plain-language summary of the current decision state."""
    brief = getattr(dm, "decision_brief", {})
    main_reason = str(
        brief.get("main_reason") or brief.get("session_headline") or brief.get("headline") or ""
    )
    next_step = str(
        brief.get("recommended_next_step")
        or brief.get("recommended_action")
        or brief.get("session_recommendation")
        or ""
    )
    status_label = str(brief.get("user_status_label", "") or "")
    escalation = str(brief.get("escalation_guidance", "") or "")
    blocker = str(brief.get("session_blocker_summary", "") or "")
    if status_label or main_reason or next_step:
        parts = []
        if status_label:
            parts.append(f"Current decision state: {status_label}.")
        if main_reason:
            parts.append(main_reason)
        if next_step:
            parts.append(f"Next step: {next_step}")
        if blocker:
            parts.append(f"Blocker: {blocker}")
        if escalation:
            parts.append(f"Escalation: {escalation}.")
        return " ".join(parts).strip()

    champion_row = (
        dm.run_metadata[dm.run_metadata["run_id"] == dm.champion_id].iloc[0]
        if dm.champion_id is not None
        and not dm.run_metadata[dm.run_metadata["run_id"] == dm.champion_id].empty
        else None
    )
    champion_mape_val = _champion_mape(dm)
    best_variant = _best_tested_challenger(dm)
    if not dm.run_metadata.empty and "status_code" in dm.run_metadata.columns:
        review_queue = dm.run_metadata[dm.run_metadata["status_code"] == "needs_human_review"]
    else:
        review_queue = dm.run_metadata.iloc[:0]

    recommendations = dm.recommender.suggest_next_experiments(1)
    top_recommendation = recommendations[0] if recommendations else None

    summary_parts: list[str] = []
    if champion_row is not None and champion_mape_val is not None:
        summary_parts.append(
            f"Current champion: {champion_row['display_name']} at {champion_mape_val:.2f}% county error."
        )
    elif champion_row is not None:
        summary_parts.append(f"Current champion: {champion_row['display_name']}.")

    best_variant_mape = (
        _as_float(best_variant.get("selected_county_mape_overall"))
        if best_variant is not None
        else None
    )
    if best_variant is not None and best_variant_mape is not None and champion_mape_val is not None:
        delta = best_variant_mape - champion_mape_val
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


def _build_onboarding_card(dm: DashboardDataManager) -> pn.Card:
    """Explain the first action path when the dashboard has little or no history."""
    has_archive = bool(dm.run_ids)
    archive_note = (
        "You can still inspect the existing benchmark archive below while you decide what to explore next."
        if has_archive
        else "The first completed benchmark bundle will unlock the guided review tabs automatically."
    )
    body = "\n".join(
        [
            "The Projection Observatory compares projection variants, tracks search sessions, and guides review decisions.",
            "",
            "**What Start Exploring produces:** a bounded search session, benchmark bundles for completed candidates, and a recommended next route when the run finishes.",
            "",
            "**When results become reviewable:** as soon as the first usable benchmark bundle lands, the dashboard can route you into Decision Brief and the guided review tabs.",
            "",
            "**Where blocked results go:** blocked or inconclusive sessions stay on the recovery path and send you to blocker resolution instead of deep analysis by default.",
            "",
            archive_note,
        ]
    )
    return markdown_card(
        "Start Here",
        body,
        css_classes=["obs-primary-workflow-card"],
    )


def _build_decision_brief_card(dm: DashboardDataManager) -> pn.Card:
    """Compact decision brief surfaced directly on the Command Center."""
    brief = dm.decision_brief
    subject = str(brief.get("primary_subject_label", "") or "Current front-runner")
    raw_subject_id = str(brief.get("raw_subject_id", "") or "")
    status_label = str(brief.get("user_status_label", "") or "Needs more evidence")
    confidence_label = str(brief.get("confidence_label", "") or "Low confidence")
    main_reason = str(brief.get("main_reason", "") or "No decision evidence is available yet.")
    next_step = str(brief.get("recommended_next_step", "") or "Inspect the current evidence.")
    escalation = str(brief.get("escalation_guidance", "") or "Safe to continue alone")
    safe_verdict = str(
        brief.get("safe_to_recommend_label", "") or "Not yet — collect more evidence first."
    )

    body = [
        f"**Outcome:** {status_label}",
        "",
        f"**Current focus:** {subject}",
        "",
        f"**Confidence:** {confidence_label}",
        "",
        f"**Main reason:** {main_reason}",
        "",
        f"**Next action:** {next_step}",
        "",
        f"**Escalation guidance:** {escalation}",
        "",
        f"**Safe to recommend?** {safe_verdict}",
    ]
    if raw_subject_id and raw_subject_id != subject:
        body.extend(["", f"**Reference ID:** `{raw_subject_id}`"])
    return markdown_card("Decision Brief", "\n".join(body), css_classes=["obs-compact-review-card"])


def _queue_health_snapshot(dm: DashboardDataManager) -> dict[str, Any]:
    """Return the queue-health metrics surfaced on the command center.

    Kept as a tested utility even though no dedicated card calls it directly.
    """
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


def _enter_guided_review(
    dm: DashboardDataManager,
    tabs: pn.Tabs | None = None,
) -> None:
    """Enable guided review mode and navigate to the Decision Brief when possible."""
    if hasattr(dm, "initialize_guided_review_shortlist"):
        try:
            dm.initialize_guided_review_shortlist()
        except Exception:  # pragma: no cover - defensive UI guard
            logger.exception("Failed to seed guided-review shortlist.")
    dm.selection_state.review_mode = True
    dm.selection_state.review_step = 1
    if tabs is not None:
        tabs.active = 1


# ---------------------------------------------------------------------------
# Hero metric
# ---------------------------------------------------------------------------


def _build_hero_metric(dm: DashboardDataManager) -> pn.pane.HTML:
    """Render the champion MAPE as the primary hero metric with challenger delta."""
    champ_mape = _champion_mape(dm)
    if champ_mape is None:
        return hero_metric("N/A", "Champion County Error (MAPE)")

    best = _best_tested_challenger(dm)
    best_mape = _as_float(best.get("selected_county_mape_overall")) if best is not None else None
    delta = None
    if best_mape is not None:
        delta = best_mape - champ_mape

    return hero_metric(
        f"{champ_mape:.2f}%",
        "Champion County Error (MAPE)",
        delta=delta,
        color=SDC_BLUE,
    )


# ---------------------------------------------------------------------------
# KPI grid (5 compact cards)
# ---------------------------------------------------------------------------


def _build_kpi_grid(dm: DashboardDataManager) -> pn.FlexBox | pn.pane.HTML:
    """Five smaller KPI cards in a compact flex grid."""
    total_runs = len(dm.run_ids)

    tested_count = 0
    untested_count = 0
    if dm.catalog is not None:
        variants_df = dm.catalog.list_variants()
        if not variants_df.empty and "tested" in variants_df.columns:
            tested_count = int(variants_df["tested"].sum())
            untested_count = int((~variants_df["tested"]).sum())

    if total_runs == 0 and tested_count == 0:
        return pn.pane.HTML(
            '<div style="text-align:center;padding:20px 16px;background:#F8FBFF;'
            "border:1px dashed #B8CBE0;border-radius:12px;color:#5A6C84;"
            'font-size:0.95em">'
            '<strong style="color:#1F3864">No benchmark results yet.</strong><br>'
            "Use the Launch Section to run your first autonomous search. "
            "KPIs will populate as results come in."
            "</div>",
            sizing_mode="stretch_width",
        )

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
    ]
    return pn.FlexBox(
        *cards,
        flex_wrap="wrap",
        sizing_mode="stretch_width",
        css_classes=layout_mode_classes("obs-kpi-grid"),
        styles={"gap": "10px"},
    )


# ---------------------------------------------------------------------------
# Decision strip (2 cards: Champion + Best Challenger)
# ---------------------------------------------------------------------------


def _build_decision_strip(dm: DashboardDataManager) -> pn.FlexBox | pn.pane.HTML:
    """Two-card decision strip: Current Champion and Best Challenger."""
    if dm.run_metadata.empty:
        return pn.pane.HTML(
            '<div style="text-align:center;padding:24px 16px;background:'
            "linear-gradient(180deg, #F8FBFF 0%, #E9F1FC 100%);"
            'border:1px solid #D9E3F0;border-radius:12px">'
            '<div style="color:#1F3864;font-size:1.1em;font-weight:700;'
            'margin-bottom:8px">'
            "Decision cards appear here after your first benchmark run</div>"
            '<div style="color:#5A6C84;font-size:0.9em;line-height:1.5">'
            "You will see: current champion accuracy and best challenger comparison."
            "</div></div>",
            sizing_mode="stretch_width",
        )

    champ_mape = _champion_mape(dm)
    champion_row = (
        dm.run_metadata[dm.run_metadata["run_id"] == dm.champion_id].iloc[0]
        if dm.champion_id is not None
        and not dm.run_metadata[dm.run_metadata["run_id"] == dm.champion_id].empty
        else None
    )
    best_variant = _best_tested_challenger(dm)

    champion_card = _summary_card(
        "Current Champion",
        f"{champ_mape:.2f}% county error" if champ_mape is not None else "Champion unavailable",
        (
            f"{champion_row.get('display_name', champion_row.get('run_id', ''))} | "
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
    if best_variant is not None and best_variant_mape is not None and champ_mape is not None:
        delta = best_variant_mape - champ_mape
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

    return pn.FlexBox(
        champion_card,
        best_card,
        flex_wrap="wrap",
        sizing_mode="stretch_width",
        styles={"gap": "12px"},
    )


# ---------------------------------------------------------------------------
# Subprocess launch helper (shared by simple + advanced)
# ---------------------------------------------------------------------------


def _launch_search_auto(
    *,
    search_id: str,
    cpu_budget: int,
    max_total_runs: int,
    batch_run_budget: int,
    max_pending: int,
    max_recommended: int,
    include_recipes: bool,
    overwrite: bool,
    status_pane: pn.pane.HTML,
    output_pane: pn.pane.HTML,
    session_dir: Path,
) -> subprocess.Popen | None:
    """Write sidecar metadata, PID file, and launch ``search-auto`` subprocess.

    Returns the ``Popen`` handle on success, or ``None`` if the process
    crashes during the 2-second startup window.
    """
    import time

    session_dir.mkdir(parents=True, exist_ok=True)
    pid_path = session_dir / f".{search_id}.dashboard.pid"
    log_path = session_dir / f".{search_id}.dashboard.log"
    meta_path = session_dir / f".{search_id}.dashboard_meta.yaml"

    cmd = [
        sys.executable,
        str(_OBSERVATORY_SCRIPT),
        "search-auto",
        "--search-id",
        search_id,
        "--batch-run-budget",
        str(batch_run_budget),
        "--max-total-runs",
        str(max_total_runs),
        "--workers-per-run",
        str(cpu_budget),
        "--max-pending",
        str(max_pending),
        "--max-recommended",
        str(max_recommended),
    ]
    if include_recipes:
        cmd.append("--include-recipe-catalog")
    if overwrite:
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

    # Wait briefly to detect early failures
    time.sleep(2)
    exit_code = process.poll()

    if exit_code is not None:
        log_tail = ""
        if log_path.exists():
            lines = log_path.read_text(encoding="utf-8").strip().splitlines()
            log_tail = "\n".join(lines[-15:])
        output_pane.object = terminal_output(
            f"Search {search_id} failed to start (exit code {exit_code}).\n\n"
            f"--- Log output ---\n{log_tail}",
            max_height=180,
        ).object
        status_pane.object = (
            f'<div style="color:#C00;font-weight:600;font-size:0.92em">'
            f"Search <strong>{search_id}</strong> failed to start. "
            f"See details below."
            f"</div>"
        )
        return None

    output_pane.object = terminal_output(
        f"Launched: {search_id}  (PID {process.pid})\n"
        f"CPU cores: {cpu_budget} | Max experiments: {max_total_runs}\n"
        f"Log: {log_path.relative_to(_PROJECT_ROOT)}\n\n"
        f"Progress will appear in the Search Progress section.",
        max_height=180,
    ).object
    status_pane.object = (
        f'<div style="color:#00B050;font-weight:600;font-size:0.92em">'
        f"Search <strong>{search_id}</strong> launched successfully."
        f"</div>"
    )
    return process


# ---------------------------------------------------------------------------
# Launch section (consolidated Quick Start + Advanced)
# ---------------------------------------------------------------------------


def _build_launch_section(dm: DashboardDataManager) -> pn.Card:
    """Unified launch section combining simple and advanced search controls.

    Simple mode (always visible): CPU slider, max experiments, hint text,
    Start Exploring button, status/output panes.

    Advanced toggle (collapsed inner card): search-auto parameters, session
    management, preview/launch buttons, candidate tables, report, and log.
    """
    import os

    # --- Status and output panes (shared by simple and advanced) ---
    status_pane = pn.pane.HTML(sizing_mode="stretch_width")
    output_pane = pn.pane.HTML(sizing_mode="stretch_width")

    # --- Smart defaults ---
    available_cores = os.cpu_count() or 12
    default_workers = min(12, available_cores)
    default_batch_budget = int(dm.search_policy.default_run_budget)
    default_max_total = 20
    default_include_recipes = bool(dm.search_policy.include_recipe_catalog)

    preset_defaults: dict[str, _LaunchPreset] = {
        "Quick check": {
            "cpu_budget": min(max(2, available_cores // 3), 8),
            "max_total_runs": 8,
            "batch_run_budget": max(1, min(default_batch_budget, 2)),
            "summary": (
                "Runs a small safe batch to see whether the direction looks promising. "
                "Use this when you want a quick signal before investing in a larger search."
            ),
        },
        "Standard exploration": {
            "cpu_budget": default_workers,
            "max_total_runs": default_max_total,
            "batch_run_budget": default_batch_budget,
            "summary": (
                "Runs the recommended balanced search. This is the default starting point for "
                "a junior demographer because it usually produces reviewable evidence without "
                "opening too many branches at once."
            ),
        },
        "Deeper search": {
            "cpu_budget": min(available_cores, max(default_workers, 16)),
            "max_total_runs": 36,
            "batch_run_budget": max(default_batch_budget, 4),
            "summary": (
                "Runs a larger search budget to explore more variants after the first pass. "
                "Use this when the earlier review ended in mixed signal or no clear winner."
            ),
        },
    }

    # --- Simple mode widgets ---
    preset_state: dict[str, str] = {"value": "Standard exploration"}
    launch_note = pn.pane.HTML(sizing_mode="stretch_width")
    cpu_slider = pn.widgets.IntSlider(
        name="CPU cores to use",
        start=2,
        end=available_cores,
        step=1,
        value=default_workers,
        width=280,
    )
    max_runs_spinner = pn.widgets.IntInput(
        name="Max experiments to run",
        value=default_max_total,
        step=5,
        start=1,
        end=200,
        width=180,
    )

    preset_buttons: dict[str, pn.widgets.Button] = {}

    def _apply_preset(preset_name: str) -> None:
        preset_state["value"] = preset_name
        selected = preset_defaults[preset_name]
        cpu_slider.value = int(selected["cpu_budget"])
        max_runs_spinner.value = int(selected["max_total_runs"])
        for name, button in preset_buttons.items():
            button.button_type = "primary" if name == preset_name else "light"
        launch_note.object = (
            '<div style="color:#334E68;font-size:0.9em;margin:2px 0 10px 0">'
            f"<strong>{preset_name}:</strong> {selected['summary']} "
            "Results become reviewable after the first completed benchmark bundle is written."
            "</div>"
        )

    def _make_preset_handler(preset_name: str) -> Any:
        def _handler(event: Any) -> None:
            del event
            _apply_preset(preset_name)

        return _handler

    for preset_name in preset_defaults:
        button = pn.widgets.Button(
            name=preset_name,
            button_type="light",
            sizing_mode="stretch_width",
            height=52,
        )
        button.on_click(_make_preset_handler(preset_name))
        preset_buttons[preset_name] = button

    preset_button_row = pn.FlexBox(
        *preset_buttons.values(),
        flex_wrap="wrap",
        sizing_mode="stretch_width",
        css_classes=layout_mode_classes("obs-preset-button-row"),
        styles={"gap": "10px"},
    )
    _apply_preset(preset_state["value"])

    # --- Next-recommendation hint ---
    recommendations = dm.recommender.suggest_next_experiments(1)
    top_rec = recommendations[0] if recommendations else None
    if top_rec is not None:
        code_change_note = (
            " This suggestion needs a code change before it can run from this screen."
            if getattr(top_rec, "requires_code_change", False)
            else ""
        )
        hint_text = (
            '<div style="color:#334E68;font-size:0.9em;margin:4px 0 6px 0">'
            "<strong>Suggested next test:</strong> "
            f"Try <strong>{top_rec.parameter} = {top_rec.suggested_value}</strong>."
            "</div>"
            f'<div style="color:#5A6C84;font-size:0.85em;margin:0 0 8px 0">'
            f"Why: {top_rec.rationale}{code_change_note}"
            "</div>"
        )
    else:
        hint_text = (
            '<div style="color:#5A6C84;font-size:0.88em;margin:4px 0 8px 0">'
            "No recommendation available yet. Run more tested history to unlock suggested next experiments."
            "</div>"
        )
    hint_pane = pn.pane.HTML(hint_text, sizing_mode="stretch_width")

    # Active session notice
    active_id = dm.active_search_id
    has_active = active_id is not None and active_id != ""
    if has_active:
        status_pane.object = (
            f'<div style="color:#334E68;font-size:0.92em">'
            f"Active search: <strong>{active_id}</strong> -- "
            f"progress updates appear in the Search Progress section."
            f"</div>"
        )

    # --- Simple Start button ---
    def _on_simple_start(event: Any) -> None:
        preset_name = preset_state["value"]
        preset = preset_defaults[preset_name]
        _launch_search_auto(
            search_id=_default_search_id(),
            cpu_budget=cpu_slider.value,
            max_total_runs=max_runs_spinner.value,
            batch_run_budget=int(preset["batch_run_budget"]),
            max_pending=int(dm.search_policy.default_max_pending),
            max_recommended=int(dm.search_policy.default_max_recommended),
            include_recipes=default_include_recipes,
            overwrite=False,
            status_pane=status_pane,
            output_pane=output_pane,
            session_dir=dm.search_session_root,
        )

    btn_start = pn.widgets.Button(
        name="Start Exploring",
        button_type="success",
        width=200,
        height=44,
    )
    btn_start.on_click(_on_simple_start)

    resource_row = pn.FlexBox(
        cpu_slider,
        max_runs_spinner,
        flex_wrap="wrap",
        sizing_mode="stretch_width",
        styles={"gap": "14px"},
    )
    customize_card = pn.Card(
        resource_row,
        pn.pane.HTML(
            '<div style="color:#5A6C84;font-size:0.85em">'
            "Use these controls only when you need to override the preset defaults."
            "</div>",
            sizing_mode="stretch_width",
        ),
        title="Customize Launch Settings",
        collapsed=True,
        sizing_mode="stretch_width",
    )

    # ---------------------------------------------------------------
    # Advanced controls (collapsed inner card)
    # ---------------------------------------------------------------
    search_id_input = pn.widgets.TextInput(
        name="Launch Search ID",
        value=_default_search_id(),
        sizing_mode="stretch_width",
    )
    batch_run_budget_input = pn.widgets.IntInput(
        name="Batch Run Budget",
        value=default_batch_budget,
        step=1,
        start=1,
        sizing_mode="stretch_width",
    )
    max_total_runs_input = pn.widgets.IntInput(
        name="Max Total Runs",
        value=default_max_total,
        step=1,
        start=1,
        sizing_mode="stretch_width",
    )
    max_pending_input = pn.widgets.IntInput(
        name="Max Pending",
        value=int(dm.search_policy.default_max_pending),
        step=1,
        start=0,
        sizing_mode="stretch_width",
    )
    max_recommended_input = pn.widgets.IntInput(
        name="Max Recommended",
        value=int(dm.search_policy.default_max_recommended),
        step=1,
        start=0,
        sizing_mode="stretch_width",
    )
    overwrite_box = pn.widgets.Checkbox(name="Overwrite existing session", value=False)
    include_recipes_box = pn.widgets.Checkbox(
        name="Include recipe catalog",
        value=default_include_recipes,
    )

    # Advanced session management widgets
    adv_session_select = pn.widgets.Select(
        name="Search Session",
        options=dm.search_session_option_map() or {"No sessions yet": ""},
        value=dm.active_search_id or "",
        sizing_mode="stretch_width",
    )
    adv_detail_pane = pn.pane.HTML(sizing_mode="stretch_width")
    adv_sessions_table_box = pn.Column(sizing_mode="stretch_width")
    adv_candidates_table_box = pn.Column(sizing_mode="stretch_width")
    adv_report_preview_box = pn.Column(sizing_mode="stretch_width")
    adv_log_preview = pn.pane.HTML(sizing_mode="stretch_width")
    adv_output_pane = pn.pane.HTML(sizing_mode="stretch_width")

    def _adv_selected_session_row() -> pd.Series | None:
        sessions = dm.search_sessions
        if sessions.empty:
            return None
        search_id = adv_session_select.value or dm.active_search_id
        if search_id:
            matches = sessions[sessions["search_id"] == search_id]
            if not matches.empty:
                return matches.iloc[0]
        return sessions.iloc[0]

    def _adv_refresh_views(*, prefer_search_id: str | None = None) -> None:
        """Refresh all advanced-section session views."""
        dm.refresh_search_sessions()
        sessions = dm.search_sessions
        options = dm.search_session_option_map()
        if not options:
            adv_session_select.options = {"No sessions yet": ""}
            adv_session_select.value = ""
            adv_detail_pane.object = _search_session_detail_html(None)
            adv_sessions_table_box[:] = [empty_placeholder("No autonomous-search sessions found.")]
            adv_candidates_table_box[:] = [empty_placeholder("No candidate preview available yet.")]
            adv_report_preview_box[:] = [empty_placeholder("No search report is available yet.")]
            adv_log_preview.object = ""
            return

        adv_session_select.options = options
        desired = (
            prefer_search_id
            or adv_session_select.value
            or dm.active_search_id
            or next(iter(options.values()))
        )
        if desired not in options.values():
            desired = next(iter(options.values()))
        adv_session_select.value = desired

        selected = _adv_selected_session_row()
        adv_detail_pane.object = _search_session_detail_html(selected)

        # Sessions table
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
        adv_sessions_table_box[:] = [
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

        # Candidates table
        selected_search_id = str(selected["search_id"]) if selected is not None else ""
        candidates = (
            dm.search_session_candidates(selected_search_id)
            if selected_search_id
            else pd.DataFrame()
        )
        if candidates.empty:
            adv_candidates_table_box[:] = [
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
            adv_candidates_table_box[:] = [
                metric_table(
                    candidates[candidate_cols],
                    page_size=8,
                    frozen_columns=["candidate_id"],
                )
            ]

        # Report preview
        report_text = (
            dm.search_session_report_markdown(selected_search_id) if selected_search_id else ""
        )
        observatory_report_path = (
            str(selected.get("observatory_report_html", "") or "") if selected is not None else ""
        )
        if report_text:
            adv_report_preview_box[:] = [
                pn.pane.Markdown(report_text, sizing_mode="stretch_width"),
                pn.pane.Markdown(
                    f"Observatory HTML report: `{observatory_report_path or 'not written yet'}`"
                ),
            ]
        else:
            adv_report_preview_box[:] = [
                empty_placeholder(
                    "No Markdown search report is available for the selected session yet."
                ),
                pn.pane.Markdown(
                    f"Observatory HTML report: `{observatory_report_path or 'not written yet'}`"
                ),
            ]

        # Log tail
        log_text = dm.search_session_log_tail(selected_search_id) if selected_search_id else ""
        adv_log_preview.object = (
            terminal_output(log_text, max_height=220).object if log_text else ""
        )

    # --- Advanced button callbacks ---
    def _adv_preview_plan(event: Any) -> None:
        sid = search_id_input.value.strip() or _default_search_id()
        args = [
            "search-plan",
            "--search-id",
            sid,
            "--max-pending",
            str(max_pending_input.value),
            "--max-recommended",
            str(max_recommended_input.value),
        ]
        if include_recipes_box.value:
            args.append("--include-recipe-catalog")
        if overwrite_box.value:
            args.append("--overwrite")
        result = _run_observatory_command(args)
        adv_output_pane.object = terminal_output(result, max_height=220).object
        _adv_refresh_views(prefer_search_id=sid)

    def _adv_launch_search(event: Any) -> None:
        sid = search_id_input.value.strip() or _default_search_id()
        _launch_search_auto(
            search_id=sid,
            cpu_budget=cpu_slider.value,
            max_total_runs=max_total_runs_input.value,
            batch_run_budget=batch_run_budget_input.value,
            max_pending=max_pending_input.value,
            max_recommended=max_recommended_input.value,
            include_recipes=include_recipes_box.value,
            overwrite=overwrite_box.value,
            status_pane=status_pane,
            output_pane=output_pane,
            session_dir=dm.search_session_root,
        )
        adv_session_select.value = sid
        _adv_refresh_views(prefer_search_id=sid)
        search_id_input.value = _default_search_id()

    def _adv_stop_search(event: Any) -> None:
        sid = (adv_session_select.value or "").strip()
        if not sid:
            adv_output_pane.object = terminal_output(
                "Select a dashboard-launched search session first.", max_height=80
            ).object
            return
        result = dm.stop_search_session(sid)
        adv_output_pane.object = terminal_output(result, max_height=120).object
        _adv_refresh_views(prefer_search_id=sid)

    def _adv_refresh_only(event: Any) -> None:
        _adv_refresh_views()
        adv_output_pane.object = terminal_output(
            "Autonomous-search views refreshed from session files.", max_height=80
        ).object

    # --- Manual sweep callbacks (from old _build_action_buttons) ---
    def _on_run_pending_preview(event: Any) -> None:
        result = _run_observatory_command(
            [
                "run-pending",
                "--dry-run",
                "--run-budget",
                "3",
                "--resume-file",
                "data/analysis/experiments/sweeps/observatory_pending_resume.json",
            ]
        )
        adv_output_pane.object = terminal_output(result, max_height=220).object

    def _on_run_recommended_preview(event: Any) -> None:
        result = _run_observatory_command(
            [
                "run-recommended",
                "--dry-run",
                "--run-budget",
                "2",
                "--resume-file",
                "data/analysis/experiments/sweeps/observatory_recommended_resume.json",
            ]
        )
        adv_output_pane.object = terminal_output(result, max_height=220).object

    def _launch_sweep(subcommand: str, resume_filename: str) -> None:
        """Launch run-pending or run-recommended as a background subprocess."""
        parallel_runs, workers_per_run = _derive_parallelism(cpu_slider.value)
        sweeps_dir = _PROJECT_ROOT / "data" / "analysis" / "experiments" / "sweeps"
        sweeps_dir.mkdir(parents=True, exist_ok=True)
        stamp = dt.datetime.now(tz=dt.UTC).strftime("%Y%m%dT%H%M%SZ")
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
        with log_path.open("a", encoding="utf-8") as fh:
            process = subprocess.Popen(  # noqa: S603
                cmd,
                cwd=str(_PROJECT_ROOT),
                stdout=fh,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        adv_output_pane.object = terminal_output(
            f"Launched {subcommand} in the background (PID {process.pid}).\n"
            f"CPU budget: {cpu_slider.value} cores "
            f"({parallel_runs} parallel run(s) x {workers_per_run} workers)\n"
            f"Log: {log_path.relative_to(_PROJECT_ROOT)}\n"
            f"Resume: {resume_file.relative_to(_PROJECT_ROOT)}\n\n"
            f"Command: {' '.join(cmd[2:])}",
            max_height=180,
        ).object

    def _on_launch_pending(event: Any) -> None:
        _launch_sweep("run-pending", "observatory_pending_resume.json")

    def _on_launch_recommended(event: Any) -> None:
        _launch_sweep("run-recommended", "observatory_recommended_resume.json")

    # --- Advanced buttons ---
    btn_adv_refresh = pn.widgets.Button(name="Refresh", button_type="primary", width=100)
    btn_adv_refresh.on_click(_adv_refresh_only)
    btn_adv_preview_plan = pn.widgets.Button(name="Preview Plan", button_type="warning", width=140)
    btn_adv_preview_plan.on_click(_adv_preview_plan)
    btn_adv_launch = pn.widgets.Button(name="Launch Search-Auto", button_type="success", width=180)
    btn_adv_launch.on_click(_adv_launch_search)
    btn_adv_stop = pn.widgets.Button(name="Stop Search", button_type="danger", width=140)
    btn_adv_stop.on_click(_adv_stop_search)

    btn_pending_preview = pn.widgets.Button(
        name="Preview Runnable Queue", button_type="warning", width=190
    )
    btn_pending_preview.on_click(_on_run_pending_preview)
    btn_recommended_preview = pn.widgets.Button(
        name="Preview Recommended Queue", button_type="warning", width=220
    )
    btn_recommended_preview.on_click(_on_run_recommended_preview)
    btn_launch_pending = pn.widgets.Button(
        name="Launch Run-Pending", button_type="success", width=180
    )
    btn_launch_pending.on_click(_on_launch_pending)
    btn_launch_recommended = pn.widgets.Button(
        name="Launch Run-Recommended", button_type="success", width=210
    )
    btn_launch_recommended.on_click(_on_launch_recommended)

    adv_session_select.param.watch(
        lambda event: _adv_refresh_views(prefer_search_id=str(event.new)), "value"
    )
    _adv_refresh_views(prefer_search_id=dm.active_search_id)

    adv_controls = pn.FlexBox(
        search_id_input,
        batch_run_budget_input,
        max_total_runs_input,
        max_pending_input,
        max_recommended_input,
        flex_wrap="wrap",
        sizing_mode="stretch_width",
        styles={"gap": "10px"},
    )
    adv_toggles = pn.FlexBox(
        overwrite_box,
        include_recipes_box,
        flex_wrap="wrap",
        sizing_mode="stretch_width",
        styles={"gap": "18px"},
    )
    adv_search_buttons = pn.FlexBox(
        btn_adv_refresh,
        btn_adv_preview_plan,
        btn_adv_launch,
        btn_adv_stop,
        flex_wrap="wrap",
        sizing_mode="stretch_width",
        styles={"gap": "10px"},
    )
    adv_sweep_buttons = pn.FlexBox(
        btn_pending_preview,
        btn_recommended_preview,
        btn_launch_pending,
        btn_launch_recommended,
        flex_wrap="wrap",
        sizing_mode="stretch_width",
        styles={"gap": "10px"},
    )

    advanced_card = pn.Card(
        adv_session_select,
        adv_detail_pane,
        section_header("Search-Auto Settings"),
        adv_controls,
        adv_toggles,
        adv_search_buttons,
        adv_output_pane,
        section_header("Manual Sweep Actions"),
        adv_sweep_buttons,
        section_header("All Sessions"),
        adv_sessions_table_box,
        section_header("Candidate Details"),
        adv_candidates_table_box,
        section_header("Search Report"),
        adv_report_preview_box,
        section_header("Log Tail"),
        adv_log_preview,
        title="Advanced Controls",
        collapsed=True,
        sizing_mode="stretch_width",
    )

    return pn.Card(
        pn.pane.HTML(
            '<div style="color:#1F3864;font-size:0.95em;font-weight:600;margin-bottom:4px">'
            "Choose how broad the next search should be."
            "</div>",
            sizing_mode="stretch_width",
        ),
        preset_button_row,
        hint_pane,
        launch_note,
        pn.Row(btn_start, sizing_mode="stretch_width"),
        status_pane,
        output_pane,
        customize_card,
        advanced_card,
        title="Launch Experiments",
        sizing_mode="stretch_width",
        css_classes=["obs-primary-launch-card"],
    )


# ---------------------------------------------------------------------------
# Search Progress card (Phase 3 monitoring overhaul)
# ---------------------------------------------------------------------------


def _build_search_progress_card(
    dm: DashboardDataManager,
    tabs: pn.Tabs | None = None,
) -> pn.Card:
    """Live-progress view for active search sessions.

    Shows progress ring, process health badges, candidate feed during
    monitoring, completion banner when done, and terminal-styled log output.
    """
    session_select = pn.widgets.Select(
        name="Search Session",
        options=dm.search_session_option_map() or {"No sessions yet": ""},
        value=dm.active_search_id or "",
        sizing_mode="stretch_width",
    )
    progress_area = pn.Column(sizing_mode="stretch_width")
    review_action_row = pn.Row(sizing_mode="stretch_width")
    best_candidates_box = pn.Column(sizing_mode="stretch_width")
    log_pane = pn.pane.HTML(sizing_mode="stretch_width")
    log_card = pn.Card(
        log_pane,
        title="Log Output",
        collapsed=True,
        sizing_mode="stretch_width",
    )
    current_experiment_pane = pn.pane.HTML(sizing_mode="stretch_width")

    btn_primary_action = pn.widgets.Button(
        name="Review Results",
        button_type="primary",
        width=190,
        visible=False,
    )
    primary_action_route: dict[str, str] = {"value": "review"}

    def _on_primary_action(event: Any) -> None:
        route = primary_action_route["value"]
        if route in {"review", "resolve_blocker", "senior_review"}:
            _enter_guided_review(dm, tabs)
            return
        if tabs is not None:
            tabs.active = 0

    btn_primary_action.on_click(_on_primary_action)

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

    def _process_health_badge(session_row: pd.Series | None, search_id: str) -> str:
        """Return styled HTML badge for process health status."""
        if session_row is None:
            return ""
        is_running = bool(session_row.get("dashboard_process_running", False))
        running_count = int(session_row.get("running", 0) or 0)
        completed = int(session_row.get("completed", 0) or 0)
        failed = int(session_row.get("failed", 0) or 0)
        total = int(session_row.get("total", 0) or 0)
        process_status = str(session_row.get("process_status", "unknown"))

        if is_running and running_count > 0:
            badge = '<span class="badge badge-passed">RUNNING EXPERIMENTS</span>'
        elif is_running:
            badge = '<span class="badge badge-passed">PROCESS ACTIVE</span>'
        elif process_status == "stopped" and completed + failed >= total > 0 and failed == 0:
            badge = '<span class="badge badge-champion">SEARCH COMPLETE</span>'
        elif process_status == "stopped":
            badge = '<span class="badge badge-failed">PROCESS STOPPED</span>'
        else:
            badge = f'<span class="badge badge-untested">{process_status.upper()}</span>'

        # Activity info
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
        updated = str(session_row.get("updated_at", "") or "")
        if created:
            try:
                start = dt.datetime.fromisoformat(created)
                if start.tzinfo is None:
                    start = start.replace(tzinfo=dt.UTC)
                if is_running:
                    delta = dt.datetime.now(tz=dt.UTC) - start
                    hours, remainder = divmod(int(delta.total_seconds()), 3600)
                    mins, secs = divmod(remainder, 60)
                    elapsed = (
                        f"{hours}h {mins}m elapsed" if hours > 0 else f"{mins}m {secs}s elapsed"
                    )
                elif updated and completed + failed >= total > 0:
                    end = dt.datetime.fromisoformat(updated)
                    if end.tzinfo is None:
                        end = end.replace(tzinfo=dt.UTC)
                    delta_t = end - start
                    hours, remainder = divmod(int(delta_t.total_seconds()), 3600)
                    mins, secs = divmod(remainder, 60)
                    elapsed = (
                        f"Completed in {hours}h {mins}m"
                        if hours > 0
                        else f"Completed in {mins}m {secs}s"
                    )
            except (ValueError, TypeError):
                pass

        return (
            f"<div style=\"font-family:'Aptos','Segoe UI',Arial,sans-serif;"
            f"display:flex;justify-content:space-between;align-items:center;"
            f'flex-wrap:wrap;gap:8px;margin-bottom:4px">'
            f"{badge}"
            f'<span style="color:#5A6C84;font-size:0.85em">{elapsed}</span>'
            f"</div>"
            f"{activity}"
        )

    def _refresh_progress(*, prefer_search_id: str | None = None) -> None:
        """Refresh the search progress card content."""
        dm.refresh_search_sessions()
        options = dm.search_session_option_map()
        if not options:
            session_select.options = {"No sessions yet": ""}
            session_select.value = ""
            progress_area[:] = [
                empty_placeholder("No search results yet. Use the Launch Section to begin.")
            ]
            review_action_row[:] = []
            current_experiment_pane.object = ""
            best_candidates_box[:] = [empty_placeholder("No search results yet.")]
            log_pane.object = ""
            return

        session_select.options = options
        active = dm.active_search_id
        desired = prefer_search_id or active or session_select.value or next(iter(options.values()))
        if desired not in options.values():
            desired = next(iter(options.values()))
        session_select.value = desired

        selected = _selected_session_row()
        selected_search_id = str(selected["search_id"]) if selected is not None else ""

        # --- Determine state for progress ring vs completion banner ---
        is_running = (
            bool(selected.get("dashboard_process_running", False))
            if selected is not None
            else False
        )
        process_status = (
            str(selected.get("process_status", "unknown")) if selected is not None else "unknown"
        )
        completed = int(selected.get("completed", 0) or 0) if selected is not None else 0
        failed = int(selected.get("failed", 0) or 0) if selected is not None else 0
        total = int(selected.get("total", 0) or 0) if selected is not None else 0
        progress_pct = (
            float(selected.get("progress_pct", 0.0) or 0.0) if selected is not None else 0.0
        )

        search_done = (
            not is_running
            and process_status in ("complete", "stopped")
            and completed + failed >= total > 0
        )

        if search_done:
            # Completion state -- show banner instead of ring
            session_candidates = (
                dm.search_session_candidates(selected_search_id)
                if selected_search_id
                else pd.DataFrame()
            )
            session_summary = build_search_session_summary(
                session_candidates,
                search_id=selected_search_id,
                status=str(selected.get("status", "")) if selected is not None else "",
                history_index_present=bool(dm.benchmark_history_snapshot["index_present"]),
                incomplete_bundle_count=int(
                    dm.benchmark_history_snapshot["incomplete_bundle_count"]
                ),
            )
            best_candidates = (
                dm.search_session_best_candidates(selected_search_id)
                if selected_search_id
                else pd.DataFrame()
            )
            outcome_body = [
                f"**Outcome:** {session_summary.get('user_status_label', 'Needs more evidence')}",
                "",
                f"**Confidence:** {session_summary.get('confidence_label', 'Low confidence')}",
                "",
                f"**Evidence quality:** {session_summary.get('evidence_quality', 'Partial evidence')}",
                "",
                f"**Main reason:** {session_summary.get('main_reason', session_summary.get('session_headline', ''))}",
            ]
            main_gain = str(session_summary.get("main_gain", "") or "")
            if main_gain:
                outcome_body.extend(["", f"**Main gain:** {main_gain}"])
            main_tradeoff = str(session_summary.get("main_tradeoff", "") or "")
            if main_tradeoff:
                outcome_body.extend(["", f"**Main tradeoff:** {main_tradeoff}"])
            outcome_body.extend(
                [
                    "",
                    f"**Next action:** {session_summary.get('recommended_next_step', 'Inspect the session details.')}",
                    "",
                    f"**Escalation guidance:** {session_summary.get('escalation_guidance', 'Safe to continue alone')}",
                ]
            )
            progress_area[:] = [
                markdown_card(
                    "Session Outcome",
                    "\n".join(outcome_body),
                    min_width=420,
                )
            ]
            primary_action_route["value"] = str(session_summary.get("next_action_route", "review"))
            btn_primary_action.name = str(
                session_summary.get("next_action_label", "Review Results")
            )
            btn_primary_action.button_type = {
                "review": "primary",
                "resolve_blocker": "warning",
                "continue_exploring": "success",
                "senior_review": "warning",
                "monitor": "default",
            }.get(primary_action_route["value"], "primary")
            btn_primary_action.visible = True
            review_action_row[:] = [
                btn_primary_action,
                pn.pane.HTML(
                    (
                        '<div style="color:#5A6C84;font-size:0.88em;padding-top:8px">'
                        "The button above follows the recommended route from the current session "
                        "outcome instead of sending every result into the same review path."
                        "</div>"
                    ),
                    sizing_mode="stretch_width",
                ),
            ]
        else:
            # Running / not started -- show progress ring + status text
            ring_status = "running" if is_running else "mixed" if failed > 0 else "running"
            ring_label = f"{completed + failed}/{total}" if total > 0 else ""
            ring = progress_ring(progress_pct, label=ring_label, status=ring_status)

            status_text = (
                f'<div style="font-size:0.92em;color:#334E68">'
                f"<strong>{selected.get('search_id', '')}</strong> -- "
                f"{str(selected.get('status', 'unknown')).upper()}"
                f"</div>"
                if selected is not None
                else ""
            )
            status_html = pn.pane.HTML(status_text, sizing_mode="stretch_width")

            progress_area[:] = [pn.Row(ring, status_html, sizing_mode="stretch_width")]
            btn_primary_action.visible = False
            review_action_row[:] = []

        # Process health badges
        current_experiment_pane.object = _process_health_badge(selected, selected_search_id)

        # Best candidates -- candidate_feed during monitoring, metric_table when done
        best_candidates = (
            dm.search_session_best_candidates(selected_search_id)
            if selected_search_id
            else pd.DataFrame()
        )
        if best_candidates.empty:
            best_candidates_box[:] = [empty_placeholder("No completed candidates yet.")]
        elif search_done:
            best_cols = [
                col
                for col in [
                    "candidate_id",
                    "decision_label",
                    "outcome",
                    "county_mape_overall",
                    "delta_county_mape_overall",
                    "headline",
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
        else:
            best_candidates_box[:] = [candidate_feed(best_candidates, max_items=5)]

        # Log tail -- terminal output widget
        log_text = (
            dm.search_session_log_tail(selected_search_id, max_chars=4000)
            if selected_search_id
            else ""
        )
        if log_text:
            log_pane.object = terminal_output(log_text, max_height=200).object
        else:
            log_pane.object = ""

        # Auto-expand log when there are errors, but never force-collapse
        # (respect the user's manual expand/collapse choice).
        has_errors = selected is not None and (
            int(selected.get("failed", 0) or 0) > 0
            and not bool(selected.get("dashboard_process_running", False))
        )
        if has_errors and log_card.collapsed:
            log_card.collapsed = False

        # Adaptive polling frequency
        if _periodic_cb is not None:
            any_running = (
                not dm.search_sessions.empty
                and dm.search_sessions["dashboard_process_running"].any()
            )
            if any_running and _periodic_cb.period != 5000:
                _periodic_cb.period = 5000
            elif not any_running and _periodic_cb.period != 30000:
                _periodic_cb.period = 30000

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

    # Auto-refresh every 5 seconds while a search is running
    _periodic_cb = None
    if pn.state.curdoc is not None:
        _periodic_cb = pn.state.add_periodic_callback(_refresh_progress, period=5000, start=True)

    session_select.param.watch(
        lambda event: _refresh_progress(prefer_search_id=str(event.new)), "value"
    )
    _refresh_progress(prefer_search_id=dm.active_search_id)

    return pn.Card(
        pn.Row(session_select, btn_refresh, btn_stop, sizing_mode="stretch_width"),
        current_experiment_pane,
        progress_area,
        review_action_row,
        best_candidates_box,
        log_card,
        title="Search Progress",
        sizing_mode="stretch_width",
        css_classes=["obs-primary-workflow-card"],
    )


# ---------------------------------------------------------------------------
# Champion snapshot and run index (preserved)
# ---------------------------------------------------------------------------


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
            '<h4 style="margin:12px 0 6px 0;color:#1F3864">Highest Priority County Errors</h4>'
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
        <tr><td style="padding:4px 12px 4px 0;color:#5A6C84">County Error</td><td style="padding:4px 0;font-weight:700;color:{SDC_NAVY}">{_fmt_metric(champion.get("county_mape_overall"))}</td></tr>
        <tr><td style="padding:4px 12px 4px 0;color:#5A6C84">Recent State Error (short window)</td><td style="padding:4px 0;font-weight:600">{_fmt_metric(champion.get("state_ape_recent_short"))}</td></tr>
        <tr><td style="padding:4px 12px 4px 0;color:#5A6C84">Recent State Error (medium window)</td><td style="padding:4px 0;font-weight:600">{_fmt_metric(champion.get("state_ape_recent_medium"))}</td></tr>
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
    display_df["run"] = display_df.get("display_name", pd.Series("", index=display_df.index))
    display_df["review_status"] = display_df.get(
        "status_label", pd.Series("", index=display_df.index)
    )
    display_df["config"] = display_df.get(
        "short_config", pd.Series("", index=display_df.index)
    ).replace("", pd.NA)
    if "selected_county_mape_overall" in display_df.columns:
        display_df["county_error_mape"] = display_df["selected_county_mape_overall"].round(3)
    else:
        display_df["county_error_mape"] = pd.NA
    if "selected_state_ape_recent_short" in display_df.columns:
        display_df["recent_state_error_ape"] = display_df["selected_state_ape_recent_short"].round(
            3
        )
    else:
        display_df["recent_state_error_ape"] = pd.NA
    display_df["run_date"] = display_df.get("run_date_label", pd.Series("", index=display_df.index))

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


# ---------------------------------------------------------------------------
# Main layout
# ---------------------------------------------------------------------------


def build_command_center(
    dm: DashboardDataManager,
    tabs: pn.Tabs | None = None,
) -> pn.Column:
    """Build the Command Center tab with a two-column layout.

    Parameters
    ----------
    dm:
        Dashboard data manager supplying run metadata, scorecards, and
        recommendations.
    tabs:
        Optional parent tab layout (unused after removing Start Here card,
        kept for API compatibility).

    Returns
    -------
    pn.Column
        The assembled command-center layout.
    """

    def _layout_section(area: str, *components: Any) -> pn.Column:
        return pn.Column(
            *components,
            css_classes=layout_mode_classes(
                "obs-command-center-section",
                f"obs-cc-area-{area}",
            ),
            sizing_mode="stretch_width",
        )

    workflow_card = (
        _build_search_progress_card(dm, tabs=tabs)
        if not dm.search_sessions.empty
        else _build_onboarding_card(dm)
    )

    sections: list[Any] = [
        _layout_section("session", workflow_card),
        _layout_section("launch", _build_launch_section(dm)),
        _layout_section("brief", _build_decision_brief_card(dm)),
        _layout_section("kpis", _build_kpi_grid(dm)),
        _layout_section("hero", _build_hero_metric(dm)),
        _layout_section("strip", _build_decision_strip(dm)),
    ]

    if dm.run_ids:
        sections.extend(
            [
                _layout_section("runindex", _build_index_table(dm)),
                _layout_section("champion", _build_champion_card(dm)),
            ]
        )

    command_center_grid = pn.FlexBox(
        *sections,
        css_classes=layout_mode_classes("obs-command-center-grid"),
        sizing_mode="stretch_width",
    )

    return pn.Column(
        section_header(
            "Command Center",
            subtitle=(
                "Compare projection variants, inspect decision evidence, and "
                "decide what to run or promote next."
            ),
        ),
        command_center_grid,
        sizing_mode="stretch_width",
    )
