"""Workspace readiness and UX-state helpers for the Observatory dashboard.

Centralizes first-run preflight checks and the high-level workspace states
used by the launcher, data manager, command center, and analytical tabs.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from typing import Any

from cohort_projections.analysis.observatory.results_store import ResultsStore
from cohort_projections.analysis.observatory.search_policy import load_search_policy
from cohort_projections.data.popest_shared import resolve_popest_root

WORKSPACE_SETUP_NEEDED = "setup_needed"
WORKSPACE_EMPTY_READY = "empty_ready"
WORKSPACE_SEARCH_IN_PROGRESS = "search_in_progress"
WORKSPACE_REVIEW_READY = "review_ready"
WORKSPACE_RECOVERY_NEEDED = "recovery_needed"
WORKSPACE_RECOMMENDATION_READY = "recommendation_ready"
WORKSPACE_SENIOR_JUDGMENT_NEEDED = "senior_judgment_needed"


@dataclass(frozen=True)
class ReadinessIssue:
    """One workstation readiness or recovery issue."""

    code: str
    label: str
    impact: str
    action: str
    severity: str = "error"


def _issue(
    code: str,
    label: str,
    impact: str,
    action: str,
    *,
    severity: str = "error",
) -> ReadinessIssue:
    """Build one readiness issue."""
    return ReadinessIssue(
        code=code,
        label=label,
        impact=impact,
        action=action,
        severity=severity,
    )


def run_dashboard_preflight() -> dict[str, Any]:
    """Inspect local workstation readiness before building the full dashboard."""
    issues: list[ReadinessIssue] = []
    dependency_specs = {
        "panel": importlib.util.find_spec("panel"),
        "plotly": importlib.util.find_spec("plotly"),
    }
    missing_dependencies = sorted(name for name, spec in dependency_specs.items() if spec is None)
    if missing_dependencies:
        issues.append(
            _issue(
                "missing_dashboard_deps",
                "Dashboard dependencies are missing",
                (
                    "The Projection Observatory cannot render correctly until the "
                    "dashboard dependencies are installed."
                ),
                (
                    "Run `uv sync --extra dev --extra dashboard` from the project root "
                    "and launch the Observatory again."
                ),
            )
        )

    try:
        popest_root = resolve_popest_root(None)
        popest_issue: ReadinessIssue | None = None
    except FileNotFoundError as exc:
        popest_root = None
        popest_issue = _issue(
            "missing_popest_archive",
            "Shared Census archive is not configured",
            (
                "Benchmarking and walk-forward validation depend on the shared Census "
                "PEP archive, so some Observatory actions will fail until it is available."
            ),
            (
                f"{exc} Set `CENSUS_POPEST_DIR` to the shared archive root, for example "
                "`export CENSUS_POPEST_DIR=~/workspace/shared-data/census/popest`."
            ),
        )
        issues.append(popest_issue)

    store = ResultsStore.from_config()
    history_dir = store._history_dir  # noqa: SLF001
    run_dirs = (
        [path for path in history_dir.iterdir() if path.is_dir()] if history_dir.exists() else []
    )
    complete_bundle_count = sum(
        1
        for run_dir in run_dirs
        if (run_dir / "summary_scorecard.csv").exists() and (run_dir / "manifest.json").exists()
    )
    incomplete_bundle_count = max(0, len(run_dirs) - complete_bundle_count)
    index_present = (history_dir / "index.csv").exists()

    if run_dirs and not index_present:
        issues.append(
            _issue(
                "missing_benchmark_index",
                "Benchmark archive metadata is incomplete",
                (
                    "The dashboard found benchmark run directories but the benchmark "
                    "index is missing, so review guidance would be unreliable."
                ),
                (
                    "Rebuild or restore `data/analysis/benchmark_history/index.csv`, or "
                    "re-register benchmark bundles before relying on review guidance."
                ),
                severity="warning",
            )
        )
    if incomplete_bundle_count > 0:
        issues.append(
            _issue(
                "incomplete_benchmark_bundles",
                "Some benchmark bundles are incomplete",
                (
                    f"{incomplete_bundle_count} bundle(s) are missing required benchmark "
                    "artifacts, so those runs should not be treated as review-ready."
                ),
                (
                    "Repair the affected bundles or rerun those benchmarks before using "
                    "them for review or recommendation."
                ),
                severity="warning",
            )
        )

    search_policy = load_search_policy()
    search_session_root = search_policy.session_root
    if not search_session_root.exists():
        issues.append(
            _issue(
                "missing_search_session_root",
                "Search session directory does not exist yet",
                (
                    "The Observatory can still launch new searches, but there is no "
                    "existing autonomous-search session history to inspect."
                ),
                (
                    f"Launch a search to create `{search_session_root}`, or create the "
                    "directory manually if the configured path is wrong."
                ),
                severity="warning",
            )
        )

    state = WORKSPACE_EMPTY_READY
    blocking_issues = [issue for issue in issues if issue.severity == "error"]
    if blocking_issues:
        state = WORKSPACE_SETUP_NEEDED
    elif (
        run_dirs
        and (not index_present or incomplete_bundle_count > 0)
        and complete_bundle_count == 0
    ):
        state = WORKSPACE_RECOVERY_NEEDED
    elif complete_bundle_count > 0 and index_present:
        state = WORKSPACE_REVIEW_READY

    return {
        "state": state,
        "history_dir": history_dir,
        "history_index_present": index_present,
        "bundle_count": len(run_dirs),
        "complete_bundle_count": complete_bundle_count,
        "incomplete_bundle_count": incomplete_bundle_count,
        "search_session_root": search_session_root,
        "popest_root": popest_root,
        "issues": [issue.__dict__ for issue in issues],
        "blocking_issues": [issue.__dict__ for issue in blocking_issues],
        "ready_for_dashboard": state != WORKSPACE_SETUP_NEEDED,
    }


def resolve_workspace_state(
    *,
    preflight: dict[str, Any] | None,
    benchmark_brief: dict[str, Any],
    session_brief: dict[str, Any],
    active_search_id: str | None,
) -> dict[str, Any]:
    """Return the canonical UX workspace state for the Observatory."""
    preflight_state = str((preflight or {}).get("state", "") or "")
    if preflight_state == WORKSPACE_SETUP_NEEDED:
        return _state_payload(
            WORKSPACE_SETUP_NEEDED,
            "Get Ready",
            "Fix workstation setup before relying on the Observatory.",
            "Resolve the setup blockers listed on the readiness screen, then reopen the dashboard.",
            "Start here",
            "setup",
        )

    complete_bundle_count = int((preflight or {}).get("complete_bundle_count", 0) or 0)
    index_present = bool((preflight or {}).get("history_index_present", False))
    incomplete_bundle_count = int((preflight or {}).get("incomplete_bundle_count", 0) or 0)

    if active_search_id:
        return _state_payload(
            WORKSPACE_SEARCH_IN_PROGRESS,
            "Continue Monitoring",
            "An autonomous search is actively producing evidence.",
            "Keep monitoring the session outcome. The dashboard will route you into review once usable evidence is available.",
            "Search is running",
            "monitor",
        )

    if complete_bundle_count == 0 and (
        preflight_state == WORKSPACE_RECOVERY_NEEDED
        or not index_present
        or incomplete_bundle_count > 0
    ):
        return _state_payload(
            WORKSPACE_RECOVERY_NEEDED,
            "Recover Broken Evidence",
            "The archive contains incomplete or inconsistent benchmark artifacts.",
            "Repair the benchmark archive or rerun the affected bundles before treating the results as review-ready.",
            "Needs repair",
            "recovery",
        )

    benchmark_state = str(benchmark_brief.get("decision_state", "") or "")
    session_state = str(session_brief.get("session_decision_state", "") or "")

    if complete_bundle_count > 0 and index_present:
        if benchmark_state == "recommended":
            return _state_payload(
                WORKSPACE_RECOMMENDATION_READY,
                "Prepare Recommendation",
                "The strongest benchmark-backed candidate is ready to bring forward for senior review.",
                "Prepare a concise recommendation packet, then validate details in the analytical tabs before escalation.",
                "Ready for recommendation",
                "recommendation",
            )
        if benchmark_state == "ready_for_review":
            return _state_payload(
                WORKSPACE_REVIEW_READY,
                "Review Best Candidate",
                "Reviewable benchmark evidence is available now.",
                "Start in Decision Brief, then use the guided review tabs to confirm plausibility before making a provisional recommendation.",
                "Ready for review",
                "review",
            )
        if benchmark_state in {"mixed_signal", "failed_hard_gate"}:
            return _state_payload(
                WORKSPACE_SENIOR_JUDGMENT_NEEDED,
                "Ask For Senior Review",
                "The best available evidence has meaningful tradeoffs or a hard-gate issue.",
                "Use the analytical tabs to understand the tradeoff, then escalate to a senior analyst rather than making a solo recommendation.",
                "Needs senior judgment",
                "senior_review",
            )

    if session_state in {"mixed_signal", "failed_hard_gate"}:
        return _state_payload(
            WORKSPACE_SENIOR_JUDGMENT_NEEDED,
            "Ask For Senior Review",
            "The latest session produced evidence that still needs senior judgment.",
            "Review the tradeoffs and blockers, then bring the case to a senior analyst instead of advancing it alone.",
            "Needs senior judgment",
            "senior_review",
        )

    if session_state == "blocked_by_data_or_runtime":
        return _state_payload(
            WORKSPACE_RECOVERY_NEEDED,
            "Recover Broken Evidence",
            "The latest session did not produce usable benchmark-backed evidence.",
            "Resolve the blocker or rerun the affected search before treating the session as reviewable.",
            "Needs repair",
            "recovery",
        )

    return _state_payload(
        WORKSPACE_EMPTY_READY,
        "Start First Exploration",
        "The workstation is ready, but no reviewable benchmark evidence exists yet.",
        "Launch a bounded search first. The first usable benchmark bundle will unlock guided review and recommendation prep.",
        "Ready to start",
        "launch",
    )


def workspace_context_message(
    workspace_state: str,
    tab_name: str,
    *,
    direct_mode: bool,
) -> str:
    """Return a short context strip for one analytical tab."""
    prefix = "Direct Explore mode:" if direct_mode else "Guided mode:"

    messages: dict[str, dict[str, str]] = {
        "Scorecards": {
            WORKSPACE_EMPTY_READY: "Open this after your first successful benchmark to compare which candidate actually improved the tracked error measures.",
            WORKSPACE_REVIEW_READY: "Start here to verify that the current front-runner really beats the champion on the most important metrics.",
            WORKSPACE_RECOMMENDATION_READY: "Use this to confirm the headline gain before packaging a provisional recommendation.",
            WORKSPACE_SENIOR_JUDGMENT_NEEDED: "Use this to see exactly where the gain and regression split, so you can frame the tradeoff clearly for senior review.",
            WORKSPACE_RECOVERY_NEEDED: "Use this only after the archive is repaired; until then, any scorecard conclusions may be incomplete.",
        },
        "Projections": {
            WORKSPACE_EMPTY_READY: "This tab becomes useful once you have a reviewable candidate and need to check whether the projected paths still look plausible.",
            WORKSPACE_REVIEW_READY: "Use this after Scorecards to make sure the forecast paths still look demographically credible.",
            WORKSPACE_RECOMMENDATION_READY: "Use this to confirm that the candidate you plan to recommend still looks plausible, not just accurate on backtests.",
            WORKSPACE_SENIOR_JUDGMENT_NEEDED: "Use this to show whether the tradeoff produces implausible forecast paths or only a tolerable accuracy shift.",
            WORKSPACE_RECOVERY_NEEDED: "Projection curves are secondary until the broken evidence state is resolved.",
        },
        "Horizon & Bias": {
            WORKSPACE_EMPTY_READY: "Open this after you have reviewable evidence and need to understand where the candidate wins or loses over time.",
            WORKSPACE_REVIEW_READY: "Use this to see whether gains hold across horizons and county types or only in a narrow slice.",
            WORKSPACE_RECOMMENDATION_READY: "Use this to pressure-test the recommendation before taking it to senior review.",
            WORKSPACE_SENIOR_JUDGMENT_NEEDED: "Use this to locate the exact horizon or geography where the candidate becomes questionable.",
            WORKSPACE_RECOVERY_NEEDED: "Bias diagnostics should not drive decisions while archive evidence is incomplete.",
        },
        "Sensitivity": {
            WORKSPACE_EMPTY_READY: "This tab is mainly for deeper follow-up once you already have a candidate worth discussing.",
            WORKSPACE_REVIEW_READY: "Use this last to see whether the candidate behaves stably enough to trust the review recommendation.",
            WORKSPACE_RECOMMENDATION_READY: "Use this to confirm the recommendation is not resting on a brittle parameter combination.",
            WORKSPACE_SENIOR_JUDGMENT_NEEDED: "Use this to show whether the tradeoff is a stable methodological choice or a fragile one-off result.",
            WORKSPACE_RECOVERY_NEEDED: "Sensitivity results are secondary until the broken evidence state is repaired.",
        },
    }
    body = messages.get(tab_name, {}).get(
        workspace_state,
        "Use this tab to inspect the detailed evidence directly.",
    )
    return f"**{prefix}** {body}"


def _state_payload(
    state: str,
    route_title: str,
    summary: str,
    next_step: str,
    badge_label: str,
    dominant_route: str,
) -> dict[str, Any]:
    """Return one normalized workspace-state payload."""
    return {
        "state": state,
        "route_title": route_title,
        "summary": summary,
        "next_step": next_step,
        "badge_label": badge_label,
        "dominant_route": dominant_route,
    }


def build_preflight_screen_markdown(preflight: dict[str, Any]) -> str:
    """Return the readiness-screen body for a blocked workstation."""
    issues = preflight.get("blocking_issues") or preflight.get("issues") or []
    lines = [
        "The Projection Observatory is not ready to guide decisions on this workstation yet.",
        "",
        "Resolve the blockers below, then launch the dashboard again.",
    ]
    for issue in issues:
        lines.extend(
            [
                "",
                f"**{issue.get('label', 'Issue')}**",
                "",
                f"- Impact: {issue.get('impact', '')}",
                f"- Action: {issue.get('action', '')}",
            ]
        )
    return "\n".join(lines)
