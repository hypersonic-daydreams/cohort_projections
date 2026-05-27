# Planning Documents Index

Canonical inventory of development-plan, backlog, and feature-idea documents.
Use this file to determine whether a document is current, supporting-only, or
historical before treating it as open work.

## Canonical Current Planning Sources

| Document | Role | Status |
|----------|------|--------|
| [`DEVELOPMENT_TRACKER.md`](../../DEVELOPMENT_TRACKER.md) | Canonical current-state tracker, open work, documentation queue, and deferred work | current |
| [`experiment-catalog.md`](./experiment-catalog.md) | Active CF-001 experiment and feature-idea catalog for county benchmark follow-ons | current |
| [`061-college-fix-model-revision.md`](../governance/adrs/061-college-fix-model-revision.md) | Active ADR for CF-001 methodology decision | current_proposed |
| [`2026-03-09-m2026r1-vs-m2026.md`](../reviews/benchmark_decisions/2026-03-09-m2026r1-vs-m2026.md) | Pending benchmark decision record for CF-001 promotion review | current_pending_human_review |
| [`2026-public-projection-release-handoff/`](./2026-public-projection-release-handoff/) | Active PUB-2026 public PDF and consolidated-download handoff package for marketing | current |

## Current Supporting Analysis

These documents still inform active work, but they are not themselves the
canonical backlog.

| Document | Role | Status |
|----------|------|--------|
| [`2026-03-04-projection-accuracy-analysis.md`](../reviews/2026-03-04-projection-accuracy-analysis.md) | Backtesting analysis with prioritized action items that feed CF-001 experiment ideas | supporting_current |
| [`2026-03-04-college-fix-research-implications.md`](../reviews/2026-03-04-college-fix-research-implications.md) | Research synthesis behind ADR-061 and CF-001 | supporting_current |
| [`observatory-start-here.md`](../guides/observatory-start-here.md) | Current Projection Observatory entry point and reading order | supporting_current |
| [`benchmarking-workflow.md`](../guides/benchmarking-workflow.md) | Current benchmark execution and review workflow | supporting_current |
| [`observatory-search-loop.md`](../guides/observatory-search-loop.md) | Current bounded-queue operating guide for the Observatory | supporting_current |

## Historical Or Completed Planning Records

These documents are still useful as rationale or implementation history, but
their task lists are no longer open by default.

| Document | Role | Status |
|----------|------|--------|
| [`benchmarking-process-improvement-roadmap.md`](./benchmarking-process-improvement-roadmap.md) | Historical roadmap for Observatory and benchmarking follow-ons; items P0-P10 are delivered | historical_complete |
| [`observatory-ui-ux-backlog.md`](./observatory-ui-ux-backlog.md) | Historical UI/UX implementation log; OBS-UX-01 through OBS-UX-38 are delivered | historical_complete |
| [`benchmarking-p0-implementation-plan.md`](./benchmarking-p0-implementation-plan.md) | Historical BM-001 implementation plan; BM-001 is complete | historical_complete |
| [`evaluation-blueprint.md`](./evaluation-blueprint.md) | Design blueprint for the evaluation framework; implementation landed under PP-006 / ADR-063 | historical_implemented |
| [`pp3-s08-implementation-kickoff.md`](./pp3-s08-implementation-kickoff.md) | Historical place-projection kickoff packet; PP-003 is complete | historical_complete |
| [`population-projection-explosion-bug-investigation-2026-02-13.md`](./population-projection-explosion-bug-investigation-2026-02-13.md) | Closed investigation and remediation record | historical_closed |
| [`census-method-assessment-and-path-forward.md`](./census-method-assessment-and-path-forward.md) | Historical methodology assessment and gap analysis | historical_reference |
| [`implementation-plan-census-method-upgrade.md`](./implementation-plan-census-method-upgrade.md) | Historical implementation plan for the February Census-method upgrade track | historical_reference |
| [`convergence-interpolation-scaling-factors.md`](./convergence-interpolation-scaling-factors.md) | Early February convergence-wiring plan, superseded by later implementation | historical_reference |
| [`audit-implementation-plan-and-findings.md`](./audit-implementation-plan-and-findings.md) | Closed audit of the February Census-method planning docs | historical_closed |
| [`PACKAGE_EXTRACTION_PLAN.md`](../governance/plans/PACKAGE_EXTRACTION_PLAN.md) | Historical package-extraction plan | historical_complete |
| [`PACKAGE_EXTRACTION_TRACKER.md`](../governance/plans/PACKAGE_EXTRACTION_TRACKER.md) | Completed package-extraction tracker | historical_complete |
| [`REPOSITORY_EVALUATION.md`](../REPOSITORY_EVALUATION.md) | December 28, 2025 repository snapshot; superseded by the tracker | historical_snapshot |

## Working Rule

If a historical document is reopened or a new backlog item is discovered, add
the work to `DEVELOPMENT_TRACKER.md` before treating it as active.
