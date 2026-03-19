"""Decision Brief tab for the Observatory dashboard."""

from __future__ import annotations

import html
from typing import TYPE_CHECKING, Any

import pandas as pd
import panel as pn

from cohort_projections.analysis.observatory.dashboard.theme import (
    DASHBOARD_CSS,
    layout_mode_classes,
)
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
    status_labels = {
        "recommended": "Best candidate so far",
        "ready_for_review": "Ready for human review",
        "mixed_signal": "Needs judgment",
        "blocked_by_data_or_runtime": "Blocked",
        "failed_hard_gate": "Do not promote",
        "not_executed": "Needs more evidence",
    }
    subject = str(
        brief.get("primary_subject_label")
        or brief.get("best_label")
        or brief.get("recommendation_candidate_id")
        or "Current focus"
    )
    raw_subject_id = str(brief.get("raw_subject_id", "") or "")
    if not raw_subject_id:
        raw_subject_id = str(brief.get("recommendation_candidate_id", "") or "")
    recommendation = str(
        brief.get("recommended_next_step")
        or brief.get("recommended_action")
        or brief.get("session_recommendation")
        or "Inspect the available evidence before deciding."
    )
    main_reason = str(
        brief.get("main_reason")
        or brief.get("explanation")
        or brief.get("session_blocker_summary")
        or brief.get("headline")
        or brief.get("session_headline")
        or "No decision evidence is available yet."
    )
    confidence_label = str(brief.get("confidence_label", "") or "Low confidence")
    safe_verdict = str(
        brief.get("safe_to_recommend_label", "") or "Not yet — collect more evidence first."
    )

    lines = [
        f"**Outcome:** {brief.get('user_status_label', status_labels.get(state, 'Needs more evidence'))}",
        "",
        f"**Current focus:** {subject}",
        "",
        f"**Confidence:** {confidence_label}",
        "",
        f"**Main reason:** {main_reason}",
        "",
        f"**Safe to recommend?** {safe_verdict}",
        "",
        f"**Next action:** {recommendation}",
        "",
        f"**Escalation guidance:** {brief.get('escalation_guidance', 'Safe to continue alone')}",
    ]
    if raw_subject_id and raw_subject_id != subject:
        lines.extend(["", f"**Reference ID:** `{raw_subject_id}`"])
    if source == "search_session":
        lines.extend(
            [
                "",
                "This brief is using the latest search session because benchmark-backed archive review is missing or incomplete.",
            ]
        )
    return "\n".join(lines)


def _reviewability_markdown(dm: DashboardDataManager) -> str:
    """Summarize whether the current evidence is reviewable."""
    brief = dm.decision_brief
    snapshot = dm.benchmark_history_snapshot
    state = str(brief.get("user_status_label", "") or "Needs more evidence")
    if brief.get("source") == "search_session":
        successful = int(brief.get("successful_benchmark_count", 0) or 0)
        inconclusive = int(brief.get("inconclusive_count", 0) or 0)
        blocker = str(brief.get("session_blocker_summary", "") or "No blocker recorded.")
        return "\n".join(
            [
                f"**Reviewability:** {state}",
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
            f"**Reviewability:** {state}",
            "",
            f"- Benchmark index present: `{snapshot['index_present']}`",
            f"- Bundles on disk: `{snapshot['bundle_count']}`",
            f"- Complete bundles: `{snapshot['complete_bundle_count']}`",
            f"- Incomplete bundles: `{snapshot['incomplete_bundle_count']}`",
            "- Benchmark-backed evidence is available for direct review in the downstream tabs.",
        ]
    )


def _what_matters_markdown(dm: DashboardDataManager) -> str:
    """Return the main gain/tradeoff/risk summary for the current brief."""
    brief = dm.decision_brief
    main_gain = str(brief.get("main_gain", "") or "No confirmed gain is available yet.")
    main_tradeoff = str(
        brief.get("main_tradeoff", "") or "No blocking tradeoff is highlighted in the summary."
    )
    evidence_quality = str(brief.get("evidence_quality", "") or "Partial evidence")
    risk_flags = brief.get("risk_flags", [])
    if not isinstance(risk_flags, list):
        risk_flags = []
    risk_text = (
        ", ".join(str(flag) for flag in risk_flags if str(flag).strip())
        or "No special risk flags recorded."
    )
    return "\n".join(
        [
            f"**Evidence quality:** {evidence_quality}",
            "",
            f"**Main gain:** {main_gain}",
            "",
            f"**Main tradeoff:** {main_tradeoff}",
            "",
            f"**Blockers or risk flags:** {risk_text}",
        ]
    )


def _review_checklist_card(dm: DashboardDataManager) -> pn.Card:
    """Render the guided-review checklist."""
    checklist = dm.decision_brief.get("review_checklist", [])
    if not isinstance(checklist, list) or not checklist:
        return markdown_card(
            "Review Checklist",
            "No checklist is available for the current decision summary.",
            min_width=320,
        )

    status_icon = {"yes": "Yes", "no": "No"}
    lines: list[str] = []
    for item in checklist:
        if not isinstance(item, dict):
            continue
        label = str(item.get("label", "") or "").strip()
        status = status_icon.get(str(item.get("status", "") or "").strip().lower(), "No")
        detail = str(item.get("detail", "") or "").strip()
        if not label:
            continue
        lines.extend([f"**{label}:** {status}", "", detail, ""])
    return markdown_card("Review Checklist", "\n".join(lines).strip(), min_width=360)


def _verdict_strip_html(dm: DashboardDataManager) -> str:
    """Return a verdict-first HTML summary for the current brief."""
    brief = dm.decision_brief
    status_label = str(brief.get("user_status_label", "") or "Needs more evidence")
    confidence_label = str(brief.get("confidence_label", "") or "Low confidence")
    subject = str(
        brief.get("primary_subject_label")
        or brief.get("best_label")
        or brief.get("recommendation_candidate_id")
        or "Current focus"
    )
    main_reason = str(
        brief.get("main_reason")
        or brief.get("explanation")
        or brief.get("session_headline")
        or brief.get("headline")
        or "No decision evidence is available yet."
    )
    next_step = str(
        brief.get("recommended_next_step")
        or brief.get("recommended_action")
        or brief.get("session_recommendation")
        or "Inspect the current evidence before deciding."
    )
    escalation = str(brief.get("escalation_guidance", "") or "Safe to continue alone")
    safe_verdict = str(
        brief.get("safe_to_recommend_label", "") or "Not yet — collect more evidence first."
    )
    raw_subject_id = str(brief.get("raw_subject_id", "") or "")
    source = str(brief.get("source", "") or "")

    safe_class = "caution"
    if bool(brief.get("safe_to_recommend", False)):
        safe_class = "safe"
    elif "blocked" in status_label.lower() or "do not promote" in status_label.lower():
        safe_class = "blocked"

    fields = [
        ("Outcome", status_label, False),
        ("Confidence", confidence_label, False),
        ("Main reason", main_reason, True),
        ("Next action", next_step, True),
        ("Escalation guidance", escalation, False),
    ]
    field_html = "".join(
        (
            '<div class="obs-verdict-item">'
            f'<div class="obs-verdict-label">{html.escape(label)}</div>'
            f'<div class="obs-verdict-value{" long" if is_long else ""}">{html.escape(value)}</div>'
            "</div>"
        )
        for label, value, is_long in fields
    )

    reference_parts: list[str] = []
    if raw_subject_id and raw_subject_id != subject:
        reference_parts.append(f"Reference ID: {raw_subject_id}")
    if source == "search_session":
        reference_parts.append(
            "Using the latest search-session evidence because benchmark-backed archive review is incomplete or unavailable."
        )
    reference_html = (
        f'<div class="obs-reference-note">{html.escape(" ".join(reference_parts))}</div>'
        if reference_parts
        else ""
    )

    return (
        '<div class="obs-verdict-strip">'
        '<div class="obs-verdict-top">'
        '<div class="obs-verdict-badges">'
        f'<span class="obs-verdict-pill">{html.escape(status_label)}</span>'
        f'<span class="obs-verdict-pill">Current focus: {html.escape(subject)}</span>'
        "</div>"
        f'<span class="obs-safe-pill {safe_class}">Safe to recommend? {html.escape(safe_verdict)}</span>'
        "</div>"
        f'<div class="obs-verdict-grid">{field_html}</div>'
        f"{reference_html}"
        "</div>"
    )


def _candidate_snapshot(dm: DashboardDataManager) -> pn.Column:
    """Return a small candidate snapshot table when session evidence exists."""
    session = dm.session_review_data
    candidates = session.get("candidates")
    if not isinstance(candidates, pd.DataFrame) or candidates.empty:
        return pn.Column()
    display_df = candidates.copy()
    if "primary_subject_label" in display_df.columns:
        display_df["candidate"] = display_df["primary_subject_label"]
    elif "candidate_id" in display_df.columns:
        display_df["candidate"] = display_df["candidate_id"].astype(str).str.replace("-", " ")
    if "user_status_label" in display_df.columns:
        display_df["outcome"] = display_df["user_status_label"]
    if "next_action_label" in display_df.columns:
        display_df["next_step"] = display_df["next_action_label"]
    display_cols = [
        col
        for col in [
            "candidate",
            "outcome",
            "best_metric_delta",
            "headline",
            "next_step",
        ]
        if col in display_df.columns
    ]
    if not display_cols:
        return pn.Column()
    prioritized = ["candidate", "outcome", "best_metric_delta", "headline", "next_step"]
    return metric_table(
        display_df[display_cols],
        title="Candidate Decision Snapshot",
        page_size=10,
        frozen_columns=["candidate"],
        priority_columns=[col for col in prioritized if col in display_cols],
    )


def build_decision_brief_tab(
    dm: DashboardDataManager,
    tabs: pn.Tabs | None = None,
) -> pn.Column:
    """Build the Decision Brief tab."""
    cards: list[Any] = [
        _review_checklist_card(dm),
        markdown_card(
            "What Matters Most",
            _what_matters_markdown(dm),
            min_width=360,
            css_classes=["obs-compact-review-card"],
        ),
        markdown_card(
            "Reviewability & Archive Details",
            _reviewability_markdown(dm),
            min_width=320,
            collapsed=True,
            css_classes=["obs-compact-review-card"],
        ),
    ]
    candidate_snapshot = _candidate_snapshot(dm)
    if len(candidate_snapshot) > 0:
        cards.append(
            pn.Card(
                candidate_snapshot,
                title="Candidate Snapshot",
                collapsed=True,
                sizing_mode="stretch_width",
                css_classes=["obs-compact-review-card"],
            )
        )

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
            total_steps=5,
            next_tab_index=2 if tabs is not None else None,
            next_tab_label="Scorecards",
        ),
        pn.pane.HTML(
            _verdict_strip_html(dm),
            sizing_mode="stretch_width",
            stylesheets=[DASHBOARD_CSS],
        ),
        pn.FlexBox(
            *cards,
            flex_wrap="wrap",
            sizing_mode="stretch_width",
            css_classes=layout_mode_classes("obs-decision-grid"),
            styles={"gap": "12px"},
        ),
        sizing_mode="stretch_width",
    )
