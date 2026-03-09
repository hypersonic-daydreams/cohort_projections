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
- easier to interpret over many sessions.

## Current Strengths

The workflow already does the important things correctly:

- old methods are preserved under immutable IDs,
- new methods are evaluated head-to-head,
- benchmark runs persist manifests and scorecards,
- promotion is alias-based and therefore reversible,
- the first real benchmark run has been completed and archived.

That means future process improvements should optimize usability and rigor, not redesign the core model.

## Priority Roadmap

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

## Recommended Implementation Order

Recommended sequence (revised to account for iterative benchmarking workflow):

1. **P6 runtime optimization** — unlocks faster iteration on all subsequent improvements; 5-8x speedup compounds with every benchmark cycle
2. P1 promotion thresholds
3. P2 hard-gates vs tradeoffs split
4. P3 schema enforcement
5. P8 post-promotion revalidation
6. P4 dashboard
7. P5 promotion package builder
8. P9 segmentation refinement
9. P7 place-scope extension
10. P10 operational quality tracking

> **Rationale for P6 promotion:** The original ordering assumed a stable, infrequent benchmarking cadence. With active iterative model development (e.g., ADR-061 college fix revision), benchmark runs happen multiple times per session. Cutting each run from ~15-25 minutes to ~3-5 minutes directly accelerates the development feedback loop and makes all other improvements cheaper to validate.

## Session-Start Shortcut

When future sessions ask how to improve the benchmarking process, start here:

- `docs/plans/benchmarking-process-improvement-roadmap.md`

Then confirm whether the work should be treated as:

- process-only documentation,
- automation implementation,
- policy tightening,
- or scope expansion beyond county benchmarks.
