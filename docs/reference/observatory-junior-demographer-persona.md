# Projection Observatory Persona: Junior Demographer

## Purpose

Use this persona when designing, reviewing, or refining the Projection
Observatory UI/UX. The goal is to keep the dashboard usable for someone who is
capable, careful, and motivated, but still building confidence in projection
method evaluation.

## Persona Summary

**Name:** Alex Rivera
**Role:** Junior Demographer / Research Analyst
**Experience level:** Early career, 1-3 years of applied demographic work
**Domain strengths:** Population estimates, cohort-component concepts,
county-level patterns, QA instincts
**Confidence level:** Moderate on demography, low-to-moderate on model
comparison and promotion decisions

Alex understands why projection improvement matters, but does not yet trust
their own judgment when comparing alternative methods. They are willing to
explore, but they want the Observatory to help them avoid careless mistakes and
to make the next step feel obvious.

## Background

Alex can:

- read projection tables and demographic summaries,
- understand concepts like cohort aging, fertility, mortality, migration, and
  county subgroup variation,
- recognize that some improvements are tradeoffs rather than pure wins,
- follow a structured workflow when the system clearly signals what to do.

Alex struggles with:

- translating statistical shorthand into decision confidence,
- knowing when a result is "interesting" versus "ready for review",
- interpreting failures caused by runtime, data, or artifact issues,
- deciding which tab to open next when many views are available.

## Primary User Story

> I am a junior demographer using the Projection Observatory to improve the
> population projections. I want the dashboard to tell me what happened, what
> matters most, what I should look at next, and whether I am safe to make a
> recommendation or need help.

## Goals

- Find promising candidate changes without needing to understand every control.
- Tell whether a search produced usable evidence or only noise.
- Understand the difference between:
  `blocked`, `mixed signal`, `ready for review`, and `do not promote`.
- Move through the workflow in order without guessing:
  launch -> monitor -> review -> decide.
- Build confidence through plain-language summaries before inspecting dense
  tables and diagnostics.

## Friction Points To Design Around

- Alex hesitates when the UI uses unexplained abbreviations like `MAPE`, `APE`,
  `sentinel`, `hard gate`, or raw parameter names without interpretation.
- Alex loses confidence when the UI tells them to go somewhere that does not
  exist, or when a button/step seems implied but not actually available.
- Alex will avoid advanced panels if they look risky or if the impact is not
  obvious.
- Alex may confuse "best numeric score" with "promotion-ready" unless the UI
  explicitly separates the two.

## What Good UX Feels Like For Alex

- The first screen answers:
  what is this, what is the current situation, and what should I do next?
- The launch path is safe by default and uses sensible settings.
- Monitoring feels active and reassuring rather than technical and opaque.
- Completed runs end with a clear recommendation and a visible "review now"
  action.
- Each tab starts with a short takeaway before showing charts and tables.
- Advanced diagnostics are available, but clearly marked as optional or
  secondary.
- The system distinguishes:
  `best candidate so far`, `usable evidence`, and `promotion recommendation`.

## Decision-Making Heuristics

When optimizing for Alex, prefer these patterns:

- Plain-language labels before technical labels.
- Recommended next action before raw metrics.
- One strong suggested path before multiple equivalent choices.
- Safe defaults before flexible configuration.
- Explanations of tradeoffs before exposing advanced diagnostics.
- Visible blockers with concrete recovery steps.

## Anti-Patterns

Avoid these when designing for this persona:

- Surfacing internal IDs or method codes as the primary explanation.
- Showing decision states without telling the user what to do next.
- Requiring the user to infer workflow order from tab names alone.
- Mixing operational failures with methodological failures without distinction.
- Presenting success banners that do not lead into an actionable review flow.

## Observatory-Specific UX Requirements

Future Observatory work should assume Alex needs:

- a visible ordered workflow,
- a trustworthy "next step" recommendation,
- a real guided review mode after search completion,
- plain-language interpretations of recommendation status,
- corrected and current navigation cues,
- suggestions that explain why a parameter is worth testing,
- help distinguishing "interesting result" from "decision-ready evidence".

## Acceptance Questions For Future Sessions

When reviewing changes, ask:

1. On first open, would Alex know what the Observatory is for?
2. After running a search, would Alex know whether to review, rerun, or stop?
3. Does the UI tell Alex where to go next without making them guess?
4. Are expert diagnostics available without dominating the first-run
   experience?
5. Would Alex trust the interface enough to keep exploring instead of asking a
   senior analyst for help immediately?
