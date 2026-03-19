---
title: "Projection Observatory UI/UX Review and Implementation Backlog"
created: 2026-03-13T13:30:00-05:00
status: historical-backlog-complete
author: Codex
purpose: >
  Capture a first-time-user UX review of the Projection Observatory dashboard
  and convert the findings into a concrete, implementation-ready backlog.
related_docs:
  - DEVELOPMENT_TRACKER.md
  - docs/guides/observatory-start-here.md
  - docs/plans/benchmarking-process-improvement-roadmap.md
  - scripts/analysis/observatory_dashboard.py
  - cohort_projections/analysis/observatory/dashboard/
---

# Projection Observatory UI/UX Review and Implementation Backlog

> Status note (2026-03-19): `OBS-UX-01` through `OBS-UX-38` are implemented.
> This document is retained as historical implementation history, not an open
> backlog. Use `DEVELOPMENT_TRACKER.md` for current work and
> `docs/plans/README.md` for planning-doc status.

## Review Frame

Primary user story:

> "I am the State Demographer opening the Projection Observatory for the first
> time. Can I tell what this system is for, what needs my attention now, what I
> should do next, and where to go for supporting evidence?"

Review questions:

1. Is the UI intuitive on first open?
2. Does it lead the user through the process visually and explicitly?
3. Is the purpose of each component easy to understand?
4. Is the most important information visible together rather than scattered?

## Executive Assessment

Overall conclusion: the dashboard is a strong expert tool, but before this
review it was not yet fully self-guiding for a first-time operator.

What already worked well:

- Fresh launch intent was correct: `Command Center` is the right home tab.
- The tab order was already sensible for the real workflow.
- The `Command Center` already surfaced the champion, best challenger, review
  queue, recommendation, queue health, and next actions.
- The desktop layout was stable and remained usable at narrower laptop widths.

What needed improvement:

- First-open behavior could drift because the `pn.serve` launcher reused a
  mutable app object, allowing later sessions to inherit the previously active
  tab.
- The landing page showed status but did not explicitly tell the user what the
  Projection Observatory is or how to use it.
- Core decision context was visible, but the process still relied on the user
  inferring the next tab and the next question.
- The UI still used too much internal or statistical shorthand (`MAPE`, `APE`,
  `sentinel`, `resolved_status`, `config_only`) without enough plain-language
  framing.
- Technical tabs remained information-dense and expert-friendly rather than
  self-explanatory.

## Concrete Backlog

| ID | Work Item | Priority | Status | Acceptance Criteria | Primary Touch Points |
|----|-----------|----------|--------|---------------------|----------------------|
| OBS-UX-01 | Isolate dashboard session state | P0 | implemented_2026-03-13 | Each new browser session gets a fresh app instance and consistently opens on `Command Center` | `scripts/analysis/observatory_dashboard.py` |
| OBS-UX-02 | Add first-run orientation on `Command Center` | P0 | implemented_2026-03-13 | Landing page defines the Observatory in plain language and gives a clear ordered workflow | `tab_command_center.py` |
| OBS-UX-03 | Add plain-language current-state summary | P0 | implemented_2026-03-13 | First screen explains champion, best challenger, review load, and next suggested experiment in one paragraph | `tab_command_center.py` |
| OBS-UX-04 | Add cross-tab navigation shortcuts from `Command Center` | P0 | implemented_2026-03-13 | User can jump directly from the landing page to variants, scorecards, projections, diagnostics, and recommendations | `app.py`, `tab_command_center.py` |
| OBS-UX-05 | Replace Command Center jargon with decision language | P0 | implemented_2026-03-13 | Queue health, champion snapshot, action buttons, and run index use plainer labels | `tab_command_center.py` |
| OBS-UX-06 | Add task-oriented framing to every major tab | P1 | implemented_2026-03-13 | Each tab begins with a short `Use this tab to...` explanation in plain language | `tab_experiment_tracker.py`, `tab_scorecard.py`, `tab_projection_ensemble.py`, `tab_horizon_bias.py`, `tab_sensitivity.py` |
| OBS-UX-07 | Add executive summaries above dense comparison views | P1 | implemented_2026-03-13 | `Scorecards` and `Projections` start with a concise takeaway before charts/tables | `tab_scorecard.py`, `tab_projection_ensemble.py` |
| OBS-UX-08 | Push advanced diagnostics behind progressive disclosure | P1 | implemented_2026-03-13 | Residual tests, outlier scatter, and deeper QA live below clear summaries or in collapsible sections | `tab_horizon_bias.py`, `tab_sensitivity.py` |
| OBS-UX-09 | Preserve shortlist context across tabs | P1 | implemented_2026-03-15 | Selected bundles/runs remain aligned when moving between comparison, projection, and diagnostic tabs | dashboard tab modules + shared data manager |
| OBS-UX-10 | Improve scanability of dense tables | P2 | implemented_2026-03-15 | Internal-only fields are hidden by default, labels are friendlier, and review-action columns remain visible | `tab_command_center.py`, `tab_experiment_tracker.py`, `widgets.py` |

## First Implementation Slice Completed On 2026-03-13

Implemented in this session:

- `OBS-UX-01`: switched the launcher to serve a fresh dashboard factory per
  session instead of a shared mutable app instance.
- `OBS-UX-02`: added a `Start Here` panel with the Observatory definition,
  ordered workflow, and navigation shortcuts.
- `OBS-UX-03`: added a plain-language `Current situation` summary.
- `OBS-UX-04`: wired `Command Center` buttons to the corresponding tabs.
- `OBS-UX-05`: renamed several `Command Center` labels to make the decision
  context easier to read.
- `OBS-UX-06`: added task-oriented framing to `Experiments`, `Scorecards`,
  `Projections`, `Horizon & Bias`, and `Sensitivity`.
- `OBS-UX-07`: added executive summaries above the dense `Scorecards` and
  `Projections` comparison views.
- `OBS-UX-08`: moved advanced diagnostics behind collapsible cards in
  `Horizon & Bias` and `Sensitivity`.

## Completion Update (2026-03-15)

- `OBS-UX-09`: added shared shortlist state in the dashboard data manager so
  `Scorecards`, `Projections`, and `Horizon & Bias` stay aligned when moving
  between tabs.
- `OBS-UX-10`: cleaned up dense decision-support tables with friendlier labels,
  visible action/context columns, and reduced exposure of internal-only fields
  in the Experiment Tracker.

## Progressive Disclosure and Quick Start (2026-03-17)

New user story driving this slice:

> "I want to open the Observatory and start exploring improvements to the
> projections without wading through configuration controls and dense
> reference panels before I can take action."

| ID | Work Item | Priority | Status | Acceptance Criteria | Primary Touch Points |
|----|-----------|----------|--------|---------------------|----------------------|
| OBS-UX-11 | Add one-click Quick Start launcher | P0 | implemented_2026-03-17 | A single "Start Exploring" button with smart defaults launches `search-auto` without requiring any configuration | `tab_command_center.py` |
| OBS-UX-12 | Add lightweight Search Progress card | P0 | implemented_2026-03-17 | Active search session progress, best candidates, stop/refresh — always visible without expanding advanced controls | `tab_command_center.py` |
| OBS-UX-13 | Collapse secondary panels by default | P0 | implemented_2026-03-17 | Queue Health, Champion Snapshot, Run Index, Manual Sweep Actions, and Persistent Weaknesses start collapsed | `tab_command_center.py` |
| OBS-UX-14 | Collapse Autonomous Search advanced controls | P0 | implemented_2026-03-17 | Full search configuration (search ID, CPU budget, batch/max budgets, toggles, preview/launch buttons) starts collapsed under "Autonomous Search (Advanced)" | `tab_command_center.py` |
| OBS-UX-15 | Streamline Start Here card | P1 | implemented_2026-03-17 | Compact orientation with 3-step workflow and 4 tab-navigation buttons; removed verbose introductory text | `tab_command_center.py` |
| OBS-UX-16 | Rename legacy Actions card | P1 | implemented_2026-03-17 | Renamed to "Manual Sweep Actions (Advanced)" to clarify its role relative to Quick Start and Autonomous Search | `tab_command_center.py` |

Implemented in this session:

- `OBS-UX-11`: added a `Quick Start` card immediately after the decision strip.
  Uses smart defaults from the search policy (12 CPU cores, policy-derived batch
  budget, 20 max total runs, recipe inclusion per policy). Generates a timestamped
  search ID automatically. One green "Start Exploring" button.
- `OBS-UX-12`: added a `Search Progress` card that shows the active session's
  progress bar, best completed candidates, and stop/refresh controls. Auto-refreshes
  every 5 seconds. Always visible — does not require expanding the advanced card.
- `OBS-UX-13`: all secondary reference panels (Queue Health, Champion Snapshot,
  Run Index, Manual Sweep Actions, Persistent Weaknesses) now render with
  `collapsed=True` so they are available but do not contribute to first-open
  cognitive load.
- `OBS-UX-14`: the full Autonomous Search control surface is now in a card titled
  "Autonomous Search (Advanced)" that starts collapsed. Power users expand it to
  customize search parameters, preview plans, or inspect session detail/logs.
- `OBS-UX-15`: the Start Here card was streamlined to a compact 3-step workflow
  and 4 tab-navigation buttons pointing to Scorecards, Projections, Horizon & Bias,
  and Sensitivity (corrected tab indices for the 7-tab layout).
- `OBS-UX-16`: the legacy Actions card was renamed to "Manual Sweep Actions
  (Advanced)" and starts collapsed.

New Command Center layout order (top to bottom):

1. Start Here (compact orientation + tab nav)
2. KPI row (at-a-glance metrics)
3. Decision strip (champion / challenger / review / recommendation cards)
4. **Quick Start** (one-click launch — new)
5. **Search Progress** (live progress + best candidates — new)
6. Autonomous Search (Advanced) — collapsed
7. Queue Health — collapsed
8. Champion Snapshot — collapsed
9. Run Index — collapsed
10. Manual Sweep Actions (Advanced) — collapsed
11. Persistent Weaknesses — collapsed

## Dashboard UI/UX Overhaul (2026-03-18)

Comprehensive UI/UX overhaul across all 6 phases. Driven by the observation
that the dashboard, despite 16 prior UX improvements, still felt scattered
and text-heavy with no guided workflow.

| ID | Work Item | Priority | Status | Acceptance Criteria | Primary Touch Points |
|----|-----------|----------|--------|---------------------|----------------------|
| OBS-UX-17 | New CSS theme classes for stepper, progress ring, two-column layout, tooltips, terminal output | P0 | implemented_2026-03-18 | All new visual components render correctly with SDC branding | `theme.py` |
| OBS-UX-18 | New widget factories (workflow_stepper, progress_ring, candidate_feed, hero_metric, completion_banner, terminal_output, illustrated_empty_state, info_tooltip, filter_bar) | P0 | implemented_2026-03-18 | 28 new unit tests pass for all widget factories | `widgets.py` |
| OBS-UX-19 | Command Center two-column layout with metric hierarchy | P0 | implemented_2026-03-18 | Hero metric (champion MAPE) prominent; 2 decision cards instead of 4; 2-column responsive grid that collapses below 900px | `tab_command_center.py` |
| OBS-UX-20 | Unified launch section (consolidate 3 launch surfaces) | P0 | implemented_2026-03-18 | Single "Launch Experiments" card with simple mode (Start Exploring) and collapsed advanced controls; shared subprocess helper | `tab_command_center.py` |
| OBS-UX-21 | Polished monitoring experience (progress ring, candidate feed, completion banner) | P0 | implemented_2026-03-18 | CSS circular progress ring with animation; live candidate feed during monitoring; celebratory completion banner with "Review Results" CTA; terminal-styled log output | `tab_command_center.py` |
| OBS-UX-22 | Workflow stepper above tabs | P0 | implemented_2026-03-18 | 4-step stepper (Launch → Monitor → Review → Decide) updates based on active tab and search state | `app.py` |
| OBS-UX-23 | Guided review mode with Next Step navigation | P1 | implemented_2026-03-18 | After search completes, analytical tabs show step badges and "Next" buttons navigating Scorecards → Projections → Horizon & Bias → Sensitivity | `app.py`, `data_manager.py`, analytical tabs |
| OBS-UX-24 | Merge Experiments + History into single tab | P1 | implemented_2026-03-18 | New "Experiment History" tab with 4 nested sub-tabs (Catalog, Timeline, Trends, Log); 7 tabs reduced to 6 | `tab_experiment_history.py`, `app.py` |
| OBS-UX-25 | Remove "Use This Tab To" explanation cards | P1 | implemented_2026-03-18 | All verbose explanation cards replaced with compact info tooltips on section headers | All analytical tab modules |
| OBS-UX-26 | Standardize filter bar placement | P2 | implemented_2026-03-18 | All tabs use `filter_bar()` wrapper at the top instead of ad-hoc Card/FlexBox patterns | `tab_scorecard.py`, `tab_projection_ensemble.py`, `tab_horizon_bias.py` |
| OBS-UX-27 | Illustrated empty states | P2 | implemented_2026-03-18 | First-run empty states show inline SVG illustrations (rocket, search, check) instead of plain gray text | All tabs |
| OBS-UX-28 | Tabulator column priority | P2 | implemented_2026-03-18 | `metric_table()` supports `priority_columns` parameter to show only key columns by default | `widgets.py` |

Implemented in this session:

- `OBS-UX-17` through `OBS-UX-18`: New theme and widget foundation (12 CSS
  classes, 10 widget factories, 28 new tests).
- `OBS-UX-19` through `OBS-UX-21`: Command Center completely restructured from
  11 vertical sections to a two-column layout. Three launch surfaces merged
  into one. Monitoring upgraded from raw HTML progress bars to animated CSS
  rings, live candidate feeds, and celebratory completion banners.
- `OBS-UX-22` through `OBS-UX-23`: Workflow stepper and guided review mode
  added, giving users a clear visual path through the analysis workflow.
- `OBS-UX-24` through `OBS-UX-26`: Experiments and History merged, explanation
  cards removed, filters standardized across all tabs.
- `OBS-UX-27` through `OBS-UX-28`: Illustrated empty states and column priority
  reduce visual clutter and improve first-run experience.

## Junior-Demographer Guided Journey (2026-03-19)

New user story driving this slice:

> "I am a junior demographer using the Projection Observatory to improve the
> projections. I want the dashboard to tell me what happened, what matters
> most, what I should look at next, and whether I am safe to make a
> recommendation or need help."

| ID | Work Item | Priority | Status | Acceptance Criteria | Primary Touch Points |
|----|-----------|----------|--------|---------------------|----------------------|
| OBS-UX-29 | Add user-facing decision-state layer | P0 | implemented_2026-03-19 | Decision summaries expose plain-language status, confidence, CTA route, escalation guidance, and review checklist fields instead of only internal state codes | `decision_support.py`, dashboard decision surfaces |
| OBS-UX-30 | Make launch flow preset-first | P0 | implemented_2026-03-19 | `Launch Experiments` defaults to Quick/Standard/Deeper presets, while CPU/run-budget controls move behind collapsed customization | `tab_command_center.py` |
| OBS-UX-31 | Add state-aware post-search dominant CTA | P0 | implemented_2026-03-19 | Search completion routes to `Review Results`, `Resolve Blocker`, `Continue Exploring`, or `Ask For Senior Review` based on session evidence state | `tab_command_center.py`, `decision_support.py` |
| OBS-UX-32 | Seed guided review with one primary comparison | P1 | implemented_2026-03-19 | Entering guided review auto-selects the strongest available comparison run and preserves it across analytical tabs | `data_manager.py`, `tab_command_center.py` |
| OBS-UX-33 | Turn Decision Brief into a review hub | P1 | implemented_2026-03-19 | Decision Brief shows outcome summary, evidence quality, gain/tradeoff framing, risk flags, safe-to-recommend verdict, and checklist before dense tables | `tab_decision_brief.py` |
| OBS-UX-34 | Add interpretation-first review prompts across analytical tabs | P1 | implemented_2026-03-19 | Scorecards, Projections, Horizon & Bias, and Sensitivity begin with review questions and interpretation-first takeaways rather than chart-first expert framing | analytical tab modules |

Implemented in this session:

- `OBS-UX-29`: central decision payloads now include stable user-facing fields
  such as `user_status_label`, `confidence_label`, `next_action_label`,
  `next_action_route`, `safe_to_recommend`, `blocker_category_label`,
  `escalation_guidance`, and `review_checklist`.
- `OBS-UX-30`: the launch surface now centers on `Quick check`, `Standard
  exploration`, and `Deeper search`, while CPU and max-run controls moved into
  a collapsed `Customize Launch Settings` card.
- `OBS-UX-31`: post-search monitoring now ends with a `Session Outcome` card
  and a single dominant CTA chosen from the current evidence state instead of
  always pushing the user into the same review path.
- `OBS-UX-32`: guided review now seeds the shortlist with the strongest
  available run so the analytical tabs open on a focused comparison rather
  than a generic multi-run state.
- `OBS-UX-33`: the `Decision Brief` tab now acts as the review hub with
  outcome summary, gain/tradeoff/risk framing, reviewability context, and a
  structured checklist.
- `OBS-UX-34`: analytical tabs now open with plain-language review questions
  and interpretation-first summaries tailored to the junior-demographer
  workflow.

## Portrait-Oriented Guided Review Follow-up (2026-03-19)

Follow-up user story from the visual browser review on the workstation's
portrait display:

> "I am using the Projection Observatory on a `1440x2560` portrait monitor.
> I need the dashboard to read top-to-bottom as a guided decision flow rather
> than as a landscape expert console."

| ID | Work Item | Priority | Status | Acceptance Criteria | Primary Touch Points |
|----|-----------|----------|--------|---------------------|----------------------|
| OBS-UX-35 | Add automatic portrait layout mode | P0 | implemented_2026-03-19 | Portrait-oriented viewports trigger stacked workflow layout, compact shell spacing, guided-tab emphasis, and hover-only Plotly chrome | `theme.py`, `app.py`, shared widgets |
| OBS-UX-36 | Rebuild Command Center for portrait primary-path flow | P0 | implemented_2026-03-19 | Portrait mode orders `Session Outcome / Start Here -> Launch -> Decision Brief snapshot -> KPI grid -> secondary detail cards`, with preset buttons enlarged and reference panels demoted | `tab_command_center.py` |
| OBS-UX-37 | Make Decision Brief and guided navigation verdict-first in portrait | P1 | implemented_2026-03-19 | Decision Brief starts with a verdict strip and checklist, keeps raw IDs/archive details secondary, and makes the guided next-step bar sticky in portrait | `tab_decision_brief.py`, `theme.py`, analytical tabs |
| OBS-UX-38 | Replace portrait-first wide tables with stacked summaries and placeholders | P1 | implemented_2026-03-19 | Scorecards/Projections/Sensitivity lead with interpretation-first cards, wide raw tables collapse by default, and empty chart states render explanatory placeholders instead of blank plot shells | `tab_scorecard.py`, `tab_projection_ensemble.py`, `tab_sensitivity.py`, tests |

Implemented in this session:

- `OBS-UX-35`: added orientation-aware layout helpers and CSS that switch the
  shell, `Command Center`, guided review tabs, filter bars, and next-step bar
  into a compact portrait-friendly flow while also de-emphasizing the
  `Experiment History` tab during guided review.
- `OBS-UX-36`: reordered the `Command Center` around one primary action path,
  added a real onboarding card when no search session exists, promoted the
  `Session Outcome` / `Search Progress` card to the top of the flow, and
  rendered preset launch options as large segmented buttons instead of a dense
  radio strip.
- `OBS-UX-37`: turned `Decision Brief` into a verdict-first hub with a visible
  safe-to-recommend pill, standardized outcome/confidence/reason/next-action
  ordering, and a sticky portrait next-step bar across the guided sequence.
- `OBS-UX-38`: moved `Selected Bundles` and raw recommendation tables behind
  collapsed advanced cards, introduced a compact projection chip summary and
  stacked sensitivity recommendation cards, collapsed lower-signal scorecard
  diagnostics by default, and replaced blank Plotly shells with explanatory
  placeholders when the underlying data is missing.

New tab layout (6 tabs):

1. Command Center (two-column: hero metric + decision strip + search progress | KPI grid + launch + reference)
2. **Experiment History** (merged: Catalog, Timeline, Trends, Log sub-tabs)
3. Scorecards
4. Projections
5. Horizon & Bias
6. Sensitivity

## Verification Notes

This review was based on:

- live browser screenshots from a clean first-open session,
- live rendering of all six tabs,
- a narrower laptop-width check for the `Command Center`,
- portrait-monitor rendering at `1440x2560`,
- direct code inspection of the dashboard launcher and tab builders.
- Module import verification and full dashboard test suite (18 tests passing)
  for the 2026-03-17 progressive disclosure changes.
- 257 observatory tests passing, mypy clean, ruff clean for the 2026-03-18
  UI/UX overhaul (46 dashboard-specific tests including 28 new widget tests).
- 79 dashboard tests passing and 292 observatory tests passing, with targeted
  portrait-layout, decision-brief, projection-summary, and recommendation-card
  regression coverage for the 2026-03-19 portrait follow-up.
