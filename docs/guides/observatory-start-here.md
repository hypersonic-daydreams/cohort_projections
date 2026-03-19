# Projection Observatory Start Here

Use this guide as the single entry point for continuing Projection Observatory
work. It links the current status, the safe operating workflow, and the
planning-doc inventory so you do not have to reconstruct the navigation path
from multiple documents.

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
5. `docs/plans/README.md`
   - Read this to separate current planning docs from historical roadmaps and
     completed backlogs.
6. `docs/plans/benchmarking-process-improvement-roadmap.md`
   - Read this only for historical roadmap rationale behind delivered
     Observatory and benchmarking features.
7. `docs/guides/configuration-reference.md`
   - Read the Observatory config sections when changing thresholds, paths, or
     variant-catalog behavior.

## Current State (2026-03-19)

- BM-001 experiment infrastructure is implemented.
- OBS-001 Observatory analysis, CLI, dashboard, and reporting layer is
  implemented.
- PP-007 operational hardening is complete.
- Deterministic autonomous search with isolated mirror/worktree execution is
  implemented.
- The recipe catalog now includes enabled search-only benchmark methods that
  clone the challenger under unique sandbox-only method IDs for recent-window
  and mortality-factor search.
- The live dashboard now includes the junior-demographer guided-review pass and
  the portrait-oriented follow-up pass. The default UI is optimized to answer:
  what happened, whether the evidence is usable, what to open next, and
  whether the user is safe to recommend a result or needs help.
- The local launcher now performs a workstation preflight before the full
  dashboard renders. If dashboard dependencies or the shared Census archive are
  missing, the user sees a dedicated readiness screen instead of a half-working
  dashboard.
- The dashboard now resolves one canonical workspace state across the launcher,
  decision brief, and command center: `setup needed`, `start first
  exploration`, `continue monitoring`, `review best candidate`, `prepare
  recommendation`, `recover broken evidence`, or `ask for senior review`.
- Guided mode remains the default, while a top-level `Explore Directly` mode
  lets experienced users jump straight into the analytical tabs without the
  guided-review emphasis.
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
- Planning-doc inventory: `docs/plans/README.md`
- Historical roadmap rationale: `docs/plans/benchmarking-process-improvement-roadmap.md`
- Historical UI/UX implementation backlog: `docs/plans/observatory-ui-ux-backlog.md`
- Runtime config: `config/observatory_config.yaml`
- Autonomous search policy: `config/observatory_search_policy.yaml`
- Deterministic recipe catalog: `config/observatory_recipes.yaml`
- Variant catalog: `config/observatory_variants.yaml`
- One-command unattended launcher:
  `python scripts/analysis/observatory.py search-auto`
- Enabled search-only recipe families now cover convergence-window widening,
  mortality-improvement sensitivity, and selected interaction terms around
  `m2026r1` without touching production aliases.
- The dashboard `Command Center` uses progressive disclosure to reduce
  first-open cognitive load. The top-to-bottom flow is now:
  `Session Outcome / Start Here -> Launch Experiments -> Decision Brief ->
  Scorecards -> Projections -> Horizon & Bias -> Sensitivity`.
- The `Launch Experiments` card centers on a one-click `Start Exploring`
  action with `Quick check`, `Standard exploration`, and `Deeper search`
  presets. CPU/run-budget controls, manual sweep actions, and other operator
  controls remain available in collapsed advanced cards.
- Completed searches now end in a `Session Outcome` card with one dominant
  CTA chosen from the evidence state: `Review Results`, `Resolve Blocker`,
  `Continue Exploring`, or `Ask For Senior Review`.
- Guided review now starts in `Decision Brief`, which acts as the review hub:
  verdict strip first, checklist second, then evidence quality, gains,
  tradeoffs, and escalation guidance.
- When benchmark-backed evidence is strong enough, the Command Center now
  surfaces a `Prepare Recommendation` route and can build a lightweight review
  package for senior sign-off using the existing promotion-package workflow.
- `Scorecards`, `Projections`, `Horizon & Bias`, and `Sensitivity` now lead
  with review questions and interpretation-first summaries rather than making
  the user start from raw tables/charts.
- Every analytical tab now begins with a short workspace-state context strip so
  junior users know why the tab matters before dropping into the details.
- The dashboard now detects portrait-oriented viewports automatically. On the
  current workstation's `1440x2560` portrait monitor it stacks the primary
  workflow vertically, compresses shell chrome, keeps guided navigation sticky,
  de-emphasizes `Experiment History` during guided review, and hides Plotly
  modebars until hover.
- `Experiment History` remains the archive/reference tab for longitudinal
  benchmark-history views across all runs, champion-at-run baselines, metric
  delta heatmaps, category-level trend lines, and accepted/rejected
  challenger history.

## Follow-On Roadmap Status

The previously documented follow-on roadmap items are now implemented in the
current codebase:

- `OBS-UX-09` through `OBS-UX-16` are complete (see
  `docs/plans/observatory-ui-ux-backlog.md`).
- `OBS-UX-17` through `OBS-UX-38` are also complete, including workflow
  stepper/guided review, the junior-demographer decision-support pass, and the
  portrait-oriented dashboard follow-up (see
  `docs/plans/observatory-ui-ux-backlog.md`).
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
- `P10` operational-quality tracking is now active in policy-backed decisions:
  benchmark artifacts still record runtime/reproducibility/completeness, and
  the Observatory now uses those signals to block unusable evidence, downgrade
  reproducibility/runtime warnings to review-required states, and require
  operationally clean benchmark evidence before `Prepare Recommendation`
  appears.
- `P1` history smoke tests and the convenience latest-pointer updater are now
  implemented. `data/analysis/benchmark_history/latest/` provides alias-aligned
  quick lookup for the newest benchmark bundle matching the current immutable
  alias target, while dated run directories and `index.csv` remain canonical.

## Session Start Shortcut

When starting a new Observatory implementation session:

1. Read the `PP-007` section in `DEVELOPMENT_TRACKER.md`.
2. Run `python scripts/analysis/observatory.py status`.
3. Confirm whether the session is about operation, UI/UX, or follow-on
   capability work.
4. If new Observatory work is opened, record it in `DEVELOPMENT_TRACKER.md`
   first; use `docs/plans/README.md` to separate current tasks from historical
   roadmap documents.
