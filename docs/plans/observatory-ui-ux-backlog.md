---
title: "Projection Observatory UI/UX Review and Implementation Backlog"
created: 2026-03-13T13:30:00-05:00
status: active-backlog
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

## Verification Notes

This review was based on:

- live browser screenshots from a clean first-open session,
- live rendering of all six tabs,
- a narrower laptop-width check for the `Command Center`,
- direct code inspection of the dashboard launcher and tab builders.
