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
3. `docs/guides/observatory-autonomous-search.md`
   - Read this before running deterministic autonomous search in isolated
     worktrees.
4. `docs/guides/benchmarking-workflow.md`
   - Read this for challenger review, decision records, and alias-promotion
     workflow under SOP-003.
5. `docs/plans/benchmarking-process-improvement-roadmap.md`
   - Read this for the full follow-on improvement backlog and rationale.
6. `docs/guides/configuration-reference.md`
   - Read the Observatory config sections when changing thresholds, paths, or
     variant-catalog behavior.

## Current State (2026-03-15)

- BM-001 experiment infrastructure is implemented.
- OBS-001 Observatory analysis, CLI, dashboard, and reporting layer is
  implemented.
- PP-007 operational hardening is complete.
- Deterministic autonomous search with isolated mirror/worktree execution is
  implemented.
- The recipe catalog now includes enabled search-only benchmark methods that
  clone the challenger under unique sandbox-only method IDs for recent-window
  and mortality-factor search.
- The Observatory is ready for supervised comparison and bounded unattended
  search-loop execution with resume files and run budgets.
- The Observatory is not an auto-promotion system. Promotion still requires
  SOP-003 review and alias updates.

## What Exists Today

- CLI: `scripts/analysis/observatory.py`
- Interactive Panel dashboard: `scripts/analysis/observatory_dashboard.py`
- Windows/WSL one-click launcher: `scripts/windows/launch_projection_observatory.cmd`
- Windows shortcut installer:
  `scripts/windows/install_projection_observatory_shortcuts.cmd`
- Static experiment dashboard builder:
  `scripts/analysis/build_experiment_dashboard.py`
- Search-loop operator guide: `docs/guides/observatory-search-loop.md`
- Autonomous search guide: `docs/guides/observatory-autonomous-search.md`
- Full benchmark workflow guide: `docs/guides/benchmarking-workflow.md`
- Full backlog roadmap: `docs/plans/benchmarking-process-improvement-roadmap.md`
- Concrete UI/UX review + backlog: `docs/plans/observatory-ui-ux-backlog.md`
- Runtime config: `config/observatory_config.yaml`
- Autonomous search policy: `config/observatory_search_policy.yaml`
- Deterministic recipe catalog: `config/observatory_recipes.yaml`
- Variant catalog: `config/observatory_variants.yaml`
- One-command unattended launcher:
  `python scripts/analysis/observatory.py search-auto`
- Enabled search-only recipe families now cover convergence-window widening,
  mortality-improvement sensitivity, and selected interaction terms around
  `m2026r1` without touching production aliases.
- The dashboard `Command Center` now includes an `Autonomous Search` panel for
  previewing search plans, launching `search-auto`, stopping dashboard-launched
  searches, tracking session progress, previewing search reports/log tails, and
  inspecting candidate/result summaries from persisted search sessions.
- A dedicated `History` tab now provides longitudinal benchmark-history views
  across all runs, champion-at-run baselines, metric delta heatmaps,
  category-level trend lines, and accepted/rejected challenger history.

## Follow-On Roadmap Status

The previously documented follow-on roadmap items are now implemented in the
current codebase:

- `OBS-UX-09` and `OBS-UX-10` are complete.
- `P1`/`P2` promotion-threshold formalization and hard-gate vs tradeoff split
  are implemented in the machine-readable evaluation policy.
- `P3` schema enforcement is implemented through centralized benchmark-contract
  validation for manifests, scorecards, and the benchmark index.
- `P5` promotion package building is implemented via
  `scripts/analysis/build_promotion_package.py`.
- `P6` runtime evidence is now surfaced from per-run runtime summaries.
- `P7` scope-aware bundle registration is implemented via
  `scripts/analysis/register_benchmark_bundle.py` for future place-scope
  bundles.
- `P8` post-promotion revalidation is implemented in
  `scripts/analysis/promote_method.py --revalidate`.
- `P9` segmentation refinement is implemented in the benchmark scorecard with
  reservation, smallest-county, volatile-oil, and college-heavy non-core
  lenses.
- `P10` operational-quality tracking is now recorded in benchmark artifacts and
  Observatory status output.

## Session Start Shortcut

When starting a new Observatory implementation session:

1. Read the `PP-007` section in `DEVELOPMENT_TRACKER.md`.
2. Run `python scripts/analysis/observatory.py status`.
3. Confirm whether the session is about operation, UI/UX, or follow-on
   capability work.
4. Use the roadmap to select the next backlog item before editing code.
