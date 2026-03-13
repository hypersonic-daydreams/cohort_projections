# Projection Observatory Start Here

Use this guide as the single entry point for continuing Projection Observatory
work. It links the current status, the safe operating workflow, and the
remaining improvement backlog so you do not have to reconstruct the navigation
path from multiple documents.

## Read In This Order

1. `DEVELOPMENT_TRACKER.md`
   - Read the `PP-007 Projection Observatory Operational Hardening` section for
     the current readiness statement, governance boundary, and preserved session
     findings.
2. `docs/guides/observatory-search-loop.md`
   - Read this before running bounded unattended queues.
3. `docs/guides/benchmarking-workflow.md`
   - Read this for challenger review, decision records, and alias-promotion
     workflow under SOP-003.
4. `docs/plans/benchmarking-process-improvement-roadmap.md`
   - Read this for the full follow-on improvement backlog and rationale.
5. `docs/guides/configuration-reference.md`
   - Read the Observatory config sections when changing thresholds, paths, or
     variant-catalog behavior.

## Current State (2026-03-13)

- BM-001 experiment infrastructure is implemented.
- OBS-001 Observatory analysis, CLI, dashboard, and reporting layer is
  implemented.
- PP-007 operational hardening is complete.
- The Observatory is ready for supervised comparison and bounded unattended
  search-loop execution with resume files and run budgets.
- The Observatory is not an auto-promotion system. Promotion still requires
  SOP-003 review and alias updates.

## What Exists Today

- CLI: `scripts/analysis/observatory.py`
- Interactive Panel dashboard: `scripts/analysis/observatory_dashboard.py`
- Static experiment dashboard builder:
  `scripts/analysis/build_experiment_dashboard.py`
- Search-loop operator guide: `docs/guides/observatory-search-loop.md`
- Full benchmark workflow guide: `docs/guides/benchmarking-workflow.md`
- Full backlog roadmap: `docs/plans/benchmarking-process-improvement-roadmap.md`
- Runtime config: `config/observatory_config.yaml`
- Variant catalog: `config/observatory_variants.yaml`

## Remaining Improvement Tracks

### Decision Policy And Governance

- `P1` explicit promotion thresholds
- `P2` hard-gates versus tradeoff metrics split
- `P3` stricter metric-contract and schema enforcement
- `P8` automatic post-promotion revalidation
- `P10` operational-quality tracking alongside accuracy

### UI/UX And Decision Support

- `P4` longitudinal benchmark-history dashboard views
- `P5` promotion package builder
- `P9` finer methodological segmentation lenses

Related delivered capability: the interactive experiment dashboard already
exists. The remaining gap is making longitudinal benchmark history and review
artifacts easier to interpret over time.

### Throughput And Search-Loop Scale

- `P6` benchmark runtime optimization

Current guidance: optimize benchmark internals first. Queue execution remains
intentionally sequential unless throughput later proves to be the real
constraint.

### Scope Expansion

- `P7` extend the benchmarking/versioning framework beyond county scope to
  place-level methods

## Recommended Remaining Implementation Order

This is the practical follow-on order from the active roadmap after removing
the already-completed BM-001 and PP-007 foundation work:

1. `P6` runtime optimization
2. `P1` explicit promotion thresholds
3. `P2` hard-gates versus tradeoffs split
4. `P3` schema enforcement
5. `P8` post-promotion revalidation
6. `P4` longitudinal dashboard
7. `P5` promotion package builder
8. `P9` segmentation refinement
9. `P7` place-scope extension
10. `P10` operational-quality tracking

## Session Start Shortcut

When starting a new Observatory implementation session:

1. Read the `PP-007` section in `DEVELOPMENT_TRACKER.md`.
2. Run `python scripts/analysis/observatory.py status`.
3. Confirm whether the session is about operation, UI/UX, or follow-on
   capability work.
4. Use the roadmap to select the next backlog item before editing code.
