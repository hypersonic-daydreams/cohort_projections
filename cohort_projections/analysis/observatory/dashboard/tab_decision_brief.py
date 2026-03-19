"""Decision Brief tab for the Observatory dashboard."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd
import panel as pn

from cohort_projections.analysis.observatory.dashboard.widgets import (
    build_review_step_bar,
    markdown_card,
    metric_table,
    section_header,
)

if TYPE_CHECKING:
    from cohort_projections.analysis.observatory.dashboard.data_manager import (
        DashboardDataManager,
    )


def _decision_brief_markdown(dm: DashboardDataManager) -> str:
    """Return the primary narrative brief for the current dashboard state."""
    brief = dm.decision_brief
    source = str(brief.get("source", "") or "")
    state = str(
        brief.get("decision_state") or brief.get("session_decision_state") or "not_executed"
    )
    headline = str(
        brief.get("headline")
        or brief.get("session_headline")
        or "No decision evidence is available yet."
    )
    explanation = str(brief.get("explanation") or "")
    if not explanation:
        explanation = str(brief.get("session_blocker_summary") or headline)
    recommendation = str(
        brief.get("recommended_action")
        or brief.get("session_recommendation")
        or brief.get("recommended_next_step")
        or "Inspect the available evidence before deciding."
    )
    recommendation_id = str(brief.get("recommendation_candidate_id", "") or "")

    lines = [
        f"**Decision state:** `{state}`",
        "",
        f"**What just happened:** {headline}",
        "",
        f"**Why this is the current state:** {explanation}",
        "",
    ]
    if recommendation_id:
        lines.extend(
            [
                f"**Current best option:** `{recommendation_id}`",
                "",
            ]
        )
    lines.append(f"**What to do next:** {recommendation}")
    if source == "search_session":
        lines.extend(
            [
                "",
                "This brief is using search-session evidence because benchmark-backed review data is missing or incomplete.",
            ]
        )
    return "\n".join(lines)


def _reviewability_markdown(dm: DashboardDataManager) -> str:
    """Summarize whether the current evidence is reviewable."""
    brief = dm.decision_brief
    snapshot = dm.benchmark_history_snapshot
    state = str(
        brief.get("decision_state") or brief.get("session_decision_state") or "not_executed"
    )
    if brief.get("source") == "search_session":
        successful = int(brief.get("successful_benchmark_count", 0) or 0)
        inconclusive = int(brief.get("inconclusive_count", 0) or 0)
        blocker = str(brief.get("session_blocker_summary", "") or "No blocker recorded.")
        return "\n".join(
            [
                f"**Reviewability:** `{state}`",
                "",
                f"- Successful benchmarked candidates: `{successful}`",
                f"- Inconclusive candidates: `{inconclusive}`",
                f"- Benchmark index present: `{snapshot['index_present']}`",
                f"- Incomplete bundles detected: `{snapshot['incomplete_bundle_count']}`",
                f"- Primary blocker: {blocker}",
            ]
        )

    return "\n".join(
        [
            f"**Reviewability:** `{state}`",
            "",
            f"- Benchmark index present: `{snapshot['index_present']}`",
            f"- Bundles on disk: `{snapshot['bundle_count']}`",
            f"- Complete bundles: `{snapshot['complete_bundle_count']}`",
            f"- Incomplete bundles: `{snapshot['incomplete_bundle_count']}`",
            "- Benchmark-backed evidence is available for direct review in the downstream tabs.",
        ]
    )


def _candidate_snapshot(dm: DashboardDataManager) -> pn.Column:
    """Return a small candidate snapshot table when session evidence exists."""
    session = dm.session_review_data
    candidates = session.get("candidates")
    if not isinstance(candidates, pd.DataFrame) or candidates.empty:
        return pn.Column()
    display_cols = [
        col
        for col in [
            "candidate_id",
            "decision_label",
            "outcome",
            "best_metric_name",
            "best_metric_delta",
            "headline",
        ]
        if col in candidates.columns
    ]
    if not display_cols:
        return pn.Column()
    prioritized = ["candidate_id", "decision_label", "outcome", "best_metric_delta", "headline"]
    return metric_table(
        candidates[display_cols],
        title="Candidate Decision Snapshot",
        page_size=10,
        frozen_columns=["candidate_id"],
        priority_columns=[col for col in prioritized if col in display_cols],
    )


def build_decision_brief_tab(
    dm: DashboardDataManager,
    tabs: pn.Tabs | None = None,
) -> pn.Column:
    """Build the Decision Brief tab."""
    cards: list[Any] = [
        markdown_card("Decision Brief", _decision_brief_markdown(dm), min_width=420),
        markdown_card("Reviewability", _reviewability_markdown(dm), min_width=320),
    ]
    candidate_snapshot = _candidate_snapshot(dm)
    if len(candidate_snapshot) > 0:
        cards.append(candidate_snapshot)

    return pn.Column(
        section_header(
            "Decision Brief",
            subtitle=(
                "Start here after a run or search finishes. This view tells you whether the "
                "results are usable, what matters most, what is blocked, and where to go next."
            ),
        ),
        build_review_step_bar(
            dm.selection_state,
            tabs,
            current_step=1,
            total_steps=4,
            next_tab_index=2 if tabs is not None else None,
            next_tab_label="Scorecards",
        ),
        pn.FlexBox(*cards, flex_wrap="wrap", sizing_mode="stretch_width", styles={"gap": "12px"}),
        sizing_mode="stretch_width",
    )
