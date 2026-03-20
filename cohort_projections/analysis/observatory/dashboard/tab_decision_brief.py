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
    operational_label = str(brief.get("operational_evidence_label", "") or "Operationally clean")
    operational_summary = str(
        brief.get("operational_evidence_summary", "") or "Operational evidence is clean."
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
        f"**Operational quality:** {operational_label}",
        "",
        operational_summary,
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


def _what_matters_html(dm: DashboardDataManager) -> str:
    """Return a structured HTML insight card for gain/tradeoff/risk."""
    brief = dm.decision_brief
    main_gain = html.escape(
        str(brief.get("main_gain", "") or "No confirmed gain is available yet.")
    )
    main_tradeoff = html.escape(
        str(brief.get("main_tradeoff", "") or "No blocking tradeoff is highlighted.")
    )
    risk_flags = brief.get("risk_flags", [])
    if not isinstance(risk_flags, list):
        risk_flags = []
    risk_text = html.escape(
        ", ".join(str(flag) for flag in risk_flags if str(flag).strip())
        or "No special risk flags recorded."
    )
    evidence_quality = html.escape(str(brief.get("evidence_quality", "") or "Partial evidence"))
    return (
        f'<div style="margin-bottom:8px">'
        f'<span class="obs-text-caption">Evidence quality: {evidence_quality}</span></div>'
        '<div class="obs-insight-card">'
        '<div class="obs-insight-section gain">'
        '<div class="obs-insight-label">Main Gain</div>'
        f'<div class="obs-insight-value">{main_gain}</div>'
        "</div>"
        '<div class="obs-insight-section tradeoff">'
        '<div class="obs-insight-label">Main Tradeoff</div>'
        f'<div class="obs-insight-value">{main_tradeoff}</div>'
        "</div>"
        '<div class="obs-insight-section risk">'
        '<div class="obs-insight-label">Blockers &amp; Risk Flags</div>'
        f'<div class="obs-insight-value">{risk_text}</div>'
        "</div>"
        "</div>"
    )


def _review_checklist_html(dm: DashboardDataManager) -> str:
    """Render a structured HTML checklist for the current brief."""
    checklist = dm.decision_brief.get("review_checklist", [])
    if not isinstance(checklist, list) or not checklist:
        return '<div class="obs-text-body" style="padding:16px;color:#7A8CA0">No checklist is available for the current decision summary.</div>'

    items: list[str] = []
    for item in checklist:
        if not isinstance(item, dict):
            continue
        label = html.escape(str(item.get("label", "") or "").strip())
        raw_status = str(item.get("status", "") or "").strip().lower()
        detail = html.escape(str(item.get("detail", "") or "").strip())
        if not label:
            continue
        if raw_status == "yes":
            css_class = "passed"
            icon = "&#10003;"
        elif raw_status == "no":
            css_class = "failed"
            icon = "&#10007;"
        else:
            css_class = "pending"
            icon = "?"
        items.append(
            f'<div class="obs-checklist-item {css_class}">'
            f'<span class="obs-check-icon">{icon}</span>'
            f"<div>"
            f'<div class="obs-check-label">{label}</div>'
            f'<div class="obs-check-detail">{detail}</div>'
            f"</div></div>"
        )
    return "".join(items)


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
    operational_label = str(brief.get("operational_evidence_label", "") or "Operationally clean")
    raw_subject_id = str(brief.get("raw_subject_id", "") or "")
    source = str(brief.get("source", "") or "")

    safe_class = "caution"
    if bool(brief.get("safe_to_recommend", False)):
        safe_class = "safe"
    elif "blocked" in status_label.lower() or "do not promote" in status_label.lower():
        safe_class = "blocked"

    fields: list[tuple[str, str, bool, str]] = [
        ("Outcome", status_label, False, "tint-blue"),
        ("Confidence", confidence_label, False, "tint-blue"),
        ("Main reason", main_reason, True, ""),
        ("Operational quality", operational_label, False, ""),
        ("Next action", next_step, True, "tint-green" if safe_class == "safe" else ""),
        ("Escalation guidance", escalation, False, ""),
    ]
    field_html = "".join(
        (
            f'<div class="obs-verdict-item{" " + tint if tint else ""}">'
            f'<div class="obs-verdict-label">{html.escape(label)}</div>'
            f'<div class="obs-verdict-value{" long" if is_long else ""}">{html.escape(value)}</div>'
            "</div>"
        )
        for label, value, is_long, tint in fields
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
        f'<div class="obs-verdict-strip {safe_class}">'
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
    context_card = markdown_card(
        "Why Open This Now",
        "\n".join(
            [
                f"**Workspace state:** {getattr(dm, 'workspace_state', {}).get('badge_label', 'Ready')}",
                "",
                getattr(dm, "workspace_state", {}).get(
                    "next_step",
                    "Use this tab to understand what happened, what matters most, and where to go next.",
                ),
            ]
        ),
        variant="subtle",
        min_width=320,
        css_classes=["obs-compact-review-card"],
    )

    # Structured checklist card
    checklist_card = pn.Card(
        pn.pane.HTML(
            _review_checklist_html(dm),
            sizing_mode="stretch_width",
            stylesheets=[DASHBOARD_CSS],
        ),
        title="Review Checklist",
        sizing_mode="stretch_width",
        min_width=320,
        css_classes=["obs-compact-review-card"],
    )

    # Structured "What Matters Most" card
    what_matters_card = pn.Card(
        pn.pane.HTML(
            _what_matters_html(dm),
            sizing_mode="stretch_width",
            stylesheets=[DASHBOARD_CSS],
        ),
        title="What Matters Most",
        sizing_mode="stretch_width",
        css_classes=["obs-card-prominent", "obs-compact-review-card"],
    )

    # Collapsible extras
    collapsible_cards: list[Any] = [
        markdown_card(
            "Reviewability & Archive Details",
            _reviewability_markdown(dm),
            variant="subtle",
            collapsed=True,
            css_classes=["obs-compact-review-card"],
        ),
    ]
    if getattr(dm.selection_state, "recommendation_package_path", ""):
        collapsible_cards.append(
            markdown_card(
                "Recommendation Packet",
                "\n".join(
                    [
                        "A review packet has been prepared for the current front-runner.",
                        "",
                        f"**Package path:** `{getattr(dm.selection_state, 'recommendation_package_path', '')}`",
                        "",
                        "Bring this forward for senior review after validating the details in the analytical tabs.",
                    ]
                ),
                variant="subtle",
                collapsed=True,
                css_classes=["obs-compact-review-card"],
            )
        )
    candidate_snapshot = _candidate_snapshot(dm)
    if len(candidate_snapshot) > 0:
        collapsible_cards.append(
            pn.Card(
                candidate_snapshot,
                title="Candidate Snapshot",
                collapsed=True,
                sizing_mode="stretch_width",
                css_classes=["obs-card-subtle", "obs-compact-review-card"],
            )
        )

    btn_direct = pn.widgets.Button(
        name="Open Details Directly",
        button_type="light",
        width=180,
    )

    def _open_details(event: object) -> None:
        del event
        dm.selection_state.experience_mode = "direct"
        dm.selection_state.review_mode = False
        if tabs is not None:
            tabs.active = 2

    btn_direct.on_click(_open_details)

    # Linear reading flow: verdict -> context + checklist -> what matters -> extras
    return pn.Column(
        section_header(
            "Decision Brief",
            subtitle=(
                "Start here after a run or search finishes. This view tells you whether the "
                "results are usable, what matters most, what is blocked, and where to go next."
            ),
        ),
        pn.pane.HTML(
            _verdict_strip_html(dm),
            sizing_mode="stretch_width",
            stylesheets=[DASHBOARD_CSS],
        ),
        pn.FlexBox(
            context_card,
            checklist_card,
            flex_wrap="wrap",
            sizing_mode="stretch_width",
            css_classes=layout_mode_classes("obs-decision-grid"),
            styles={"gap": "12px"},
        ),
        what_matters_card,
        *collapsible_cards,
        pn.Row(btn_direct, sizing_mode="stretch_width"),
        build_review_step_bar(
            dm.selection_state,
            tabs,
            current_step=1,
            total_steps=5,
            next_tab_index=2 if tabs is not None else None,
            next_tab_label="Scorecards",
        ),
        sizing_mode="stretch_width",
    )
