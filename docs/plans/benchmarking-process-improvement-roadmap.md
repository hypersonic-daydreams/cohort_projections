---
title: "Benchmarking Process Improvement Roadmap"
created: 2026-03-09T17:00:00-05:00
status: active-roadmap
author: Codex
purpose: >
  Capture the next improvements to the versioned benchmarking workflow so
  future sessions can extend the process without re-deriving the design.
related_docs:
  - docs/governance/sops/SOP-003-method-benchmarking-versioning-promotion.md
  - docs/guides/observatory-start-here.md
  - docs/guides/observatory-search-loop.md
  - docs/guides/benchmarking-workflow.md
  - docs/reviews/benchmark_decisions/2026-03-09-m2026r1-vs-m2026.md
related_adrs:
  - ADR-061 (College Fix model revision)
---

# Benchmarking Process Improvement Roadmap

## Purpose

This document captures the recommended next improvements to the benchmark,
versioning, and promotion workflow introduced under SOP-003.

The workflow is now usable:

- immutable method profiles exist,
- benchmark bundles are append-only,
- challenger-vs-champion decision records can be generated,
- champion aliases can be promoted without mutating historical methods.

The remaining work is about making the process:

- more explicit,
- less manual,
- faster to execute,
- more robust against drift,
- easier to interpret over many sessions,
- suitable for semi-autonomous AI agent operation.

## Current Strengths

The workflow already does the important things correctly:

- old methods are preserved under immutable IDs,
- new methods are evaluated head-to-head,
- benchmark runs persist manifests and scorecards,
- promotion is alias-based and therefore reversible,
- the first real benchmark run has been completed and archived.

That means future process improvements should optimize usability and rigor, not redesign the core model.

## How To Use This Roadmap

Use this document for the full follow-on backlog after the Observatory
foundation work. If you need a quicker entry point, start with
`docs/guides/observatory-start-here.md` and then return here for the full item
descriptions.

## Status Snapshot (2026-03-13)

The roadmap below spans both completed foundation work and remaining follow-on
items. Current state:

- Implemented: `P0` agent experiment orchestration via `BM-001`
- Implemented: `P0.5` experiment log via `BM-001`
- Implemented: `P1` stub evaluation policy via `BM-001`
- Implemented: Observatory baseline package (`OBS-001`) with CLI, dashboard,
  recommender, comparator, report generation, and variant catalog
- Implemented: operational hardening (`OBS-01` through `OBS-09`) for bounded
  unattended queueing under SOP-003 governance
- Implemented: interactive experiment dashboard on 2026-03-12

The remaining roadmap starts with completing the full `P1` promotion-threshold
formalization and the follow-on items from `P2` onward.

## Priority Roadmap

## P0: Agent Experiment Orchestration Layer

### Problem

The current workflow assumes a human operator threading together 4-5 CLI commands with correct arguments: register a profile, run the benchmark suite, generate a decision record, evaluate results, optionally promote. An AI agent can follow this sequence, but it requires re-deriving the full command chain each time, which is slow, error-prone, and blocks semi-autonomous operation.

There is no declarative way for an agent to know "what to try next," no single entry point that goes from hypothesis to logged result, and no machine-readable policy for deciding whether results require human review or can be autonomously logged and moved past.

### Improvement

#### 1. Experiment Spec Format

Define a declarative YAML experiment spec that captures everything an agent needs to run a single experiment:

```yaml
experiment_id: exp-20260310-conv-hold-25yr
hypothesis: >
  Extending convergence hold from 20 to 25 years will reduce long-horizon
  overshoot in college counties without materially worsening rural accuracy.
expected_improvement:
  - county_mape_urban_college reduction
risk_areas:
  - county_mape_rural regression if hold is too long
  - sensitivity instability from extended extrapolation
base_method: m2026r1
base_config: cfg-20260309-college-fix-v1
config_delta:
  convergence_schedule:
    medium_hold_years: 25
scope: county
benchmark_label: conv_hold_25yr_test
requested_by: agent  # or human
```

Store pending specs in `data/analysis/experiments/pending/` and completed specs in `data/analysis/experiments/completed/`.

#### 2. Single-Command Orchestrator

Build `scripts/analysis/run_experiment.py` that:

1. reads an experiment spec,
2. derives a new `method_id` and `config_id` from the base + delta (or uses explicitly provided ones),
3. registers the immutable method profile,
4. runs `run_benchmark_suite.py` with correct arguments,
5. evaluates hard gates and tradeoff thresholds against the promotion policy (P1),
6. writes an experiment log entry (P0.5),
7. moves the spec from `pending/` to `completed/`,
8. returns a structured result: `passed_all_gates`, `needs_human_review`, `failed_hard_gate`, or `inconclusive`.

The orchestrator never promotes automatically — it only classifies results. Promotion remains a separate, human-gated step.

#### 3. Autonomous vs. Review-Required Classification

Define a machine-readable evaluation policy (extends P1 promotion thresholds):

```yaml
agent_evaluation_policy:
  auto_log_and_continue:
    # Agent can mark "candidate passed" and move to next experiment if:
    hard_gates_passed: true
    no_tradeoff_regression_beyond:
      county_mape_rural: 0.10
      county_mape_bakken: 0.50
      county_mape_overall: 0.05
  flag_for_human_review:
    # Agent must stop and request review if:
    - any hard gate fails
    - any tradeoff metric regresses beyond tolerance
    - sensitivity instability flag is set
    - state APE improves but county MAPE worsens (mixed signal)
  never_autonomous:
    # Always requires human decision:
    - promotion (alias update)
    - rejection of a method (permanent status change)
    - changes to the evaluation policy itself
```

#### 4. Git Branch Conventions for Agent Experiments

Agent-driven experiments should use predictable branch names:

- `experiment/<experiment_id>` for code changes needed by the experiment
- Branches are created from the current feature branch, not from master
- If the experiment requires no code changes (config-only delta), no branch is needed

This keeps agent-driven code exploration navigable in git history without polluting the main feature branch.

### Why This Matters

This is the single highest-leverage improvement for enabling AI agents to iterate on model improvements semi-independently. Without it, every experiment cycle requires a human (or agent) to manually reconstruct the full command sequence and interpret results ad hoc. With it, an agent can process a queue of experiments, log all results immutably, and only interrupt the human when something genuinely requires judgment.

---

## P0.5: Experiment Log

### Problem

The benchmark archive records *what happened* mechanically (methods, configs, git state, metrics) but not *why it was tried* or *what was learned*. When reviewing a sequence of 10+ experiments after the fact, the manifests and scorecards show outcomes but not the reasoning thread that connects them. This makes post-hoc review slow and makes it hard to avoid re-testing hypotheses that were already explored.

### Improvement

Add an append-only experiment log alongside the benchmark index:

**Location:** `data/analysis/experiments/experiment_log.csv`

**Required columns:**

| Column | Description |
|--------|-------------|
| `experiment_id` | Unique identifier (format: `exp-YYYYMMDD-short-slug`) |
| `run_date` | ISO date of execution |
| `hypothesis` | One-sentence description of what was being tested |
| `base_method` | The method this experiment builds on |
| `config_delta_summary` | Human-readable summary of what changed |
| `run_id` | Link to benchmark run (if executed) |
| `outcome` | `passed` / `failed_hard_gate` / `mixed_signal` / `inconclusive` / `not_run` |
| `key_metrics_summary` | Compact summary: e.g., "state APE -0.2, rural MAPE +0.02" |
| `interpretation` | What the result means for the hypothesis |
| `next_action` | `proceed_to_next` / `flag_for_review` / `promote_candidate` / `abandon_line` |
| `agent_or_human` | Who ran the experiment |
| `spec_path` | Path to the full experiment spec YAML |

**Complementary YAML log** (richer detail): `data/analysis/experiments/experiment_log.yaml`

Each entry mirrors the CSV but allows multi-line hypothesis, interpretation, and structured config delta fields. The CSV is for quick scanning and dashboard integration; the YAML is for full post-hoc review.

### Interaction with Existing Artifacts

- The experiment log does **not** replace the benchmark index — it links to it via `run_id`.
- The experiment log does **not** replace decision records — decision records are for promotion decisions; the experiment log is for the broader exploration trajectory.
- The experiment spec YAML (from P0) is the input; the experiment log entry is the output.

### Why This Matters

This turns the benchmarking archive from "a collection of individual run results" into "a navigable research journal." It directly supports:

- **Future analysis:** Which hypotheses have been tested? Which lines of inquiry were productive?
- **Reporting:** Summarize the improvement trajectory for stakeholders.
- **Agent continuity:** A new agent session can read the log to understand what has already been tried and what the current frontier is, without re-exploring dead ends.
- **Post-hoc review:** A human reviewer can trace the full chain from hypothesis → experiment → result → interpretation → next action.

---

## P1: Add Explicit Promotion Thresholds

### Problem

The current decision process is directionally correct, but still partly qualitative.
That leaves too much room for inconsistent judgments across sessions.

### Improvement

Define explicit promotion thresholds for the benchmark scorecard.

At minimum, formalize:

- minimum improvement required for `state_ape_recent_short`,
- minimum improvement required for `state_ape_recent_medium`,
- maximum tolerated regression for `county_mape_rural`,
- maximum tolerated regression for `county_mape_bakken`,
- whether overall county MAPE must improve or can remain statistically neutral,
- whether certain sentinel counties have veto power.

### Recommendation

Implement a `promotion_policy` block that can be stored and versioned, for example:

```yaml
promotion_policy:
  state_ape_recent_short_min_improvement: 0.10
  state_ape_recent_medium_min_improvement: 0.25
  county_mape_rural_max_regression: 0.10
  county_mape_bakken_max_regression: 0.50
  require_zero_hard_constraint_regression: true
```

### Why This Matters

This is the single highest-leverage improvement for cross-session consistency.

---

## P2: Split Hard Gates From Tradeoff Metrics

### Problem

Some metrics are true failures; others are legitimate tradeoffs. These should not be treated the same way.

### Improvement

Separate scorecard outputs into:

- `hard_gates`
- `tradeoff_metrics`

Hard gates should include only:

- negative population violations,
- scenario ordering violations,
- true aggregation failures,
- missing artifacts or broken manifests.

Tradeoff metrics should include:

- state APE,
- county MAPE,
- category-specific MAPE,
- sentinel county deltas,
- sensitivity-instability indicators.

### Why This Matters

This makes decisions cleaner and avoids false “regression” language for normal model tradeoffs.

---

## P3: Version the Metric Contract More Rigorously

### Problem

The workflow already records `benchmark_contract_version`, but the enforcement is still light.

### Improvement

Add tests that fail if:

- required scorecard columns disappear,
- manifest fields change without an intentional version bump,
- index schema changes without migration handling.

### Recommendation

Create a schema test module that validates:

- manifest JSON shape,
- `summary_scorecard.csv` required columns,
- `index.csv` append-only compatibility.

### Why This Matters

This prevents silent benchmark drift and protects longitudinal comparability.

---

## P4: Add a Longitudinal Dashboard

### Problem

The data exists, but comparing many runs still requires manual inspection of multiple files.

Related delivered capability: the interactive experiment dashboard already
exists and covers experiment tracking, scorecards, horizon analysis, and county
heatmaps. The remaining gap here is a benchmark-history-centric longitudinal
view over champion changes, accepted/rejected challengers, and metric deltas
through time.

### Improvement

Build a lightweight dashboard or report over `data/analysis/benchmark_history/index.csv`.

Minimum views:

- all historical runs by date,
- champion history over time,
- delta trends by metric,
- category-level performance changes,
- accepted vs rejected challenger history.

### Why This Matters

This turns the archive into an actual decision-support system rather than a file store.

---

## P5: Add a Promotion Package Builder

### Problem

The approval artifacts are still assembled indirectly from several files.

### Improvement

Add a script that packages:

- the summary scorecard,
- comparison JSON,
- draft decision record,
- key plots or tables,
- manifest,
- alias targets before/after promotion.

### Why This Matters

This makes review faster and reduces the chance that approval decisions are made from partial evidence.

---

## P6: Improve Runtime Efficiency

### Problem

The benchmark suite is correct but not fast. The dominant bottleneck is **sequential iteration over independent projections**, not computation speed or data I/O.

Walk-forward validation runs 636 projections sequentially (4 origins × 53 counties × 3 methods). Sensitivity analysis runs 5,400+ projections sequentially. Each individual projection takes only ~100-150ms, but strung together the sequential loop structure turns fast math into slow runs.

Meanwhile, the existing codebase already has `ProcessPoolExecutor` parallelism working in `multi_geography.py` for production runs — the analysis scripts simply don't use it.

### Bottleneck Breakdown

| Component | Time | Parallelizable? |
|-----------|------|-----------------|
| Data loading (snapshots, rates) | ~3-5 sec | No (one-time startup) |
| Walk-forward loop (636 projections) | ~2-3 min | **Yes — all independent** |
| Sensitivity loop (5,400+ projections) | ~10-15 min | **Yes — all independent** |
| QC diagnostics | ~2-5 sec | Minor |
| Metrics & reporting | ~5-10 sec | Minor |
| **Full benchmark suite total** | **~15-25 min** | |

### Improvement

#### Tier 1: Parallelize county-method loops (5-8x speedup)

Apply `ProcessPoolExecutor` (same pattern as `multi_geography.py`) to:

- the nested origin/county/method loop in `walk_forward_validation.py` (lines 1369-1431),
- the perturbation loop in `sensitivity_analysis.py`.

Shared data (snapshots, rates) loads once at startup; each worker receives its `(origin_year, fips, method_name)` tuple. Results collect via `as_completed()` with a progress bar.

| Scenario | Current (sequential) | Parallel (8 cores) | Speedup |
|----------|---------------------|---------------------|---------|
| Walk-forward (636 runs) | ~2-3 min | ~20-30 sec | ~5-8x |
| Sensitivity (5,400+ runs) | ~10-15 min | ~1.5-2.5 min | ~5-8x |
| Full benchmark suite | ~15-25 min | ~3-5 min | ~5-8x |

This roadmap item is intentionally about benchmark-internal throughput, not
Observatory queue concurrency. The queue runner remains deliberately sequential
unless faster benchmark internals still leave throughput as the main bottleneck.

Implemented slices on 2026-03-13:

- `scripts/analysis/sensitivity_analysis.py` now supports worker-based parallel
  execution with deterministic result ordering and sequential fallback on
  worker failure.
- `scripts/analysis/walk_forward_validation.py` now supports county-level
  annual-validation parallelism with deterministic merge order and per-county
  sequential fallback on worker failure, while preserving the older
  origin-level worker path for direct callers.
- `scripts/analysis/run_benchmark_suite.py` now routes the shared `--workers`
  control into both sensitivity analysis and county-level annual validation.

Remaining gap:

- End-to-end runtime measurement to quantify which benchmark stages remain the
  dominant bottleneck after the sensitivity and annual-validation
  parallelization slices.
- County-level data extract caching and any remaining walk-forward hot spots
  beyond the annual benchmark path.

#### Tier 2: Cache county-level data extracts (2-3x additional)

Pre-compute per-county DataFrame views once at startup rather than filtering 636 times from the full snapshot DataFrames.

#### Tier 3: Minor I/O improvements (marginal)

- Convert Census 2000 Excel to parquet (saves ~2-3 sec).
- These are not worth prioritizing given the Tier 1 opportunity.

### What Is NOT Worth Optimizing

- **Projection math**: Already ~100ms per county. Numpy/Cython would not help meaningfully.
- **Parquet I/O**: Already fast; loaded once at startup.
- **Rate computations**: Already cached per-method in the dispatch system.

### Why This Matters

Faster runs lower the friction for iterative model improvement and allow more challenger methods to be tested. The 5-8x speedup from Tier 1 is dramatic, not incremental, and directly compounds with the iterative benchmarking workflow — every improvement cycle gets a faster feedback loop.

---

## P7: Extend the Framework Beyond County Scope

### Problem

The workflow is currently county-ready. Place-level evaluation is not yet first-class.

### Improvement

Extend SOP-003 tooling to support:

- place champion aliases,
- place benchmark bundles,
- place-level scorecards,
- place-specific decision records.

Potential first use cases:

- comparing place share-trending variants,
- comparing place cross-check methods,
- comparing housing-unit method calibration variants.

### Why This Matters

The benchmark/versioning process should be consistent across county and place workflows.

---

## P8: Add Automatic Post-Promotion Revalidation

### Problem

A promoted alias change is logged, but there is not yet a required clean “champion re-baseline” run after promotion.

### Improvement

After any approved promotion:

1. update the alias,
2. rerun the champion alone under the promoted config,
3. store a clean post-promotion benchmark bundle,
4. mark that bundle as the new reference baseline.

### Why This Matters

This gives each champion a canonical baseline bundle for future comparisons.

---

## P9: Improve Segmentation

### Problem

The current category breakdown is useful but still coarse.

### Improvement

Add additional segments where methodological tradeoffs differ:

- college-heavy non-core counties,
- smallest-population counties,
- reservation counties as their own promotion lens,
- highly volatile oil counties separate from other Bakken counties,
- perhaps top decile and bottom decile by population.

### Why This Matters

Better segmentation makes it easier to see where a challenger truly helps and where it may still need refinement.

---

## P10: Track Operational Quality Alongside Accuracy

### Problem

The process currently emphasizes accuracy metrics, but not enough operational metrics.

### Improvement

Track:

- benchmark runtime,
- reproducibility logging success,
- missing-input failures,
- benchmark artifact completeness,
- benchmark rerun determinism where relevant.

### Why This Matters

A method that is slightly more accurate but much more fragile should not be judged only on projection metrics.

## Recommended Remaining Implementation Order (2026-03-13)

Completed foundation work is intentionally removed from the ordering below.
What remains:

1. **P6 runtime optimization** — highest immediate leverage for faster iteration across every future benchmark cycle
2. **P1 promotion thresholds** — finish the full machine-readable thresholds beyond the current stub policy
3. **P2 hard-gates vs tradeoffs split** — clarify classification semantics before adding more automation and UI
4. **P3 schema enforcement** — protect longitudinal comparability and prevent silent drift
5. **P8 post-promotion revalidation** — make promotions leave behind a clean new baseline
6. **P4 longitudinal dashboard** — turn benchmark history into a first-class decision surface
7. **P5 promotion package builder** — shorten the human review loop
8. **P9 segmentation refinement** — make tradeoffs easier to interpret in the right county groups
9. **P7 place-scope extension** — expand the same governance and comparison pattern to place methods
10. **P10 operational quality tracking** — round out accuracy-first evaluation with robustness and operability signals

> **Foundation already in place:** `P0`, `P0.5`, and the `P1` stub are delivered through `BM-001`; the Observatory baseline and bounded unattended queue hardening are delivered through `OBS-001` and `PP-007`.

## Session-Start Shortcut

When future sessions ask how to improve the benchmarking process or the
Projection Observatory:

- Start with `docs/guides/observatory-start-here.md` for the navigation path.
- Use `docs/plans/benchmarking-process-improvement-roadmap.md` for the full backlog details.

Then confirm whether the work should be treated as:

- process-only documentation,
- automation implementation,
- policy tightening,
- scope expansion beyond county benchmarks,
- or Observatory UI/UX and decision-support improvements.

When an agent session is tasked with running experiments:

1. Check `data/analysis/experiments/pending/` for queued experiment specs.
2. Check `data/analysis/experiments/experiment_log.csv` for what has already been tried.
3. Run the next pending experiment via `scripts/analysis/run_experiment.py`, or manually thread the CLI commands per `docs/guides/benchmarking-workflow.md` when the session needs direct control.
4. Log the result in the experiment log before moving to the next experiment.
5. Flag any `needs_human_review` results and stop on that line of inquiry until reviewed.
