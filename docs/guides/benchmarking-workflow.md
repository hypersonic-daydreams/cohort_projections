# Benchmarking Workflow

Practical guide for running the versioned benchmark workflow defined in
[SOP-003](../governance/sops/SOP-003-method-benchmarking-versioning-promotion.md).

This guide is the operator-facing companion to SOP-003:

- SOP-003 defines the governance rules, schemas, and required artifacts.
- This guide shows the day-to-day commands and decision flow.
- If you are continuing Projection Observatory work, start with
  [Projection Observatory Start Here](./observatory-start-here.md).
- For bounded multi-variant queueing in the Projection Observatory, also read
  [Projection Observatory Search-Loop Guide](./observatory-search-loop.md).

## When To Use This Guide

Use this guide when you are:

- introducing a challenger method or config,
- benchmarking a challenger against the current champion,
- generating a decision record for review,
- promoting an approved method by alias update.

Do not use this guide for:

- routine code cleanup that does not affect results,
- publication-only reruns under an unchanged champion,
- ADR drafting without benchmark execution.

## Related Observatory Docs

- [Projection Observatory Start Here](./observatory-start-here.md) for status,
  reading order, and the follow-on backlog
- [Projection Observatory Search-Loop Guide](./observatory-search-loop.md) for
  bounded unattended queueing
- [Benchmarking Process Improvement Roadmap](../plans/benchmarking-process-improvement-roadmap.md)
  for the remaining multi-session implementation queue

## Preflight Checklist

1. Activate the project environment:
   ```bash
   source .venv/bin/activate
   ```
2. Confirm the relevant method IDs are registered in:
   - `scripts/analysis/walk_forward_validation.py`
3. Confirm immutable profiles exist in:
   - `config/method_profiles/`
4. Confirm aliases are current in:
   - `config/method_profiles/aliases.yaml`
5. Read the active status in:
   - `DEVELOPMENT_TRACKER.md`

## Core Concepts

### Immutable Objects

- `method_id`: algorithm identity
- `config_id`: benchmarked config identity
- `run_id`: benchmark execution identity

These never change after publication.

### Mutable Objects

- aliases in `config/method_profiles/aliases.yaml`

Aliases are pointers only. They are not provenance.

## Standard Workflow

### 1. Register the Challenger

Create a new immutable profile in `config/method_profiles/` before running any benchmark.

Example files:

- `config/method_profiles/m2026__cfg-20260223-baseline-v1.yaml`
- `config/method_profiles/m2026r1__cfg-20260309-college-fix-v1.yaml`

Minimum profile fields:

- `method_id`
- `config_id`
- `scope`
- `status`
- `created_date`
- `created_from`
- `description`
- `code_refs`
- `adr_refs`
- `resolved_config`

### 2. Run the Canonical Benchmark Suite

Run the challenger head-to-head against the current champion.

Example:

```bash
source .venv/bin/activate
python scripts/analysis/run_benchmark_suite.py \
  --scope county \
  --champion-method m2026 \
  --champion-config cfg-20260223-baseline-v1 \
  --challenger-method m2026r1 \
  --challenger-config cfg-20260309-college-fix-v1 \
  --benchmark-label college_fix_head_to_head
```

The run creates a new append-only bundle under:

- `data/analysis/benchmark_history/<run_id>/`

Required outputs include:

- `manifest.json`
- `summary_scorecard.csv`
- `comparison_to_champion.json`
- detailed state/county/sensitivity/QC outputs

### 3. Generate the Draft Decision Record

After the run finishes:

```bash
source .venv/bin/activate
python scripts/analysis/compare_benchmark_runs.py \
  --run-id <run_id>
```

This creates a draft review in:

- `docs/reviews/benchmark_decisions/YYYY-MM-DD-<challenger>-vs-<champion>.md`

### 4. Review the Evidence

The minimum review inputs are:

- `summary_scorecard.csv`
- `comparison_to_champion.json`
- the generated benchmark decision record

Primary promotion lens:

- recent-origin state APE
- recent-origin signed bias
- urban/college county performance
- overall county MAPE
- zero hard-constraint regression

### 5. Approve or Reject

If the challenger should become champion, mark the decision record as:

```markdown
| Status | Approved |
```

If not approved, leave the alias unchanged.

### 6. Promote By Alias Update Only

Never rename the challenger to the old champion method ID.

Promotion command:

```bash
source .venv/bin/activate
python scripts/analysis/promote_method.py \
  --scope county \
  --alias county_champion \
  --method-id m2026r1 \
  --config-id cfg-20260309-college-fix-v1 \
  --decision-id 2026-03-09-m2026r1-vs-m2026
```

This updates:

- `config/method_profiles/aliases.yaml`
- `data/analysis/benchmark_history/promotion_history.csv`

## What To Read In A Scorecard

The highest-signal fields are:

- `state_ape_recent_short`
- `state_ape_recent_medium`
- `state_signed_bias_recent`
- `county_mape_overall`
- `county_mape_urban_college`
- `county_mape_rural`
- `county_mape_bakken`

Sentinel counties:

- Cass
- Grand Forks
- Ward
- Burleigh
- Williams
- McKenzie

Constraint fields:

- `negative_population_violations`
- `scenario_order_violations`
- `aggregation_violations`
- `sensitivity_instability_flag`

## Decision Heuristics

Default rule:

- promote when the challenger improves recent-origin state error and key county metrics without introducing hard-constraint regression.

Default caution cases:

- improvement concentrated only in one geography type,
- small gains offset by materially worse Bakken or rural performance,
- sensitivity profile becomes much less stable,
- benchmark changes depend on fragile rounding or schema artifacts.

## Directory Map

| Path | Purpose |
|------|---------|
| `config/method_profiles/` | Immutable method/config profiles |
| `config/method_profiles/aliases.yaml` | Mutable champion/candidate/reference pointers |
| `data/analysis/benchmark_history/` | Append-only run bundles |
| `docs/reviews/benchmark_decisions/` | Human review decisions |
| `scripts/analysis/run_benchmark_suite.py` | Canonical benchmark runner |
| `scripts/analysis/compare_benchmark_runs.py` | Draft decision record generator |
| `scripts/analysis/promote_method.py` | Alias-based promotion tool |

## Current Example

First completed benchmark bundle:

- `data/analysis/benchmark_history/br-20260309-160948-m2026r1-ecb4498/`

Draft decision record:

- `docs/reviews/benchmark_decisions/2026-03-09-m2026r1-vs-m2026.md`

## Common Failure Modes

### Silent drift

Symptom:

- old method results change after helper refactors

Response:

- create a new `method_id` instead of mutating the old one

### Benchmark overwrite

Symptom:

- rerun writes into an old benchmark directory

Response:

- treat this as a process failure; benchmark runs must be append-only

### Alias confusion

Symptom:

- a user references `county_champion` as if it were provenance

Response:

- resolve the alias to its immutable `method_id` and `config_id`

### False constraint regression from rounding

Symptom:

- county sums disagree with rounded state totals by sub-person margins

Response:

- use the documented tolerance from the benchmarking module rather than treating this as a methodological regression

## Experiment Orchestration (BM-001)

The experiment orchestrator provides a single-command pipeline for running
benchmark experiments from a declarative spec. It wraps the standard workflow
(register profile → run suite → evaluate → log) into one step.

### Experiment Spec Format

Experiment specs are YAML files stored in `data/analysis/experiments/pending/`.
Schema: `config/experiment_spec_schema.yaml`.

Required fields:

- `experiment_id` — unique ID (format: `exp-YYYYMMDD-short-slug`)
- `hypothesis` — what is being tested
- `base_method` — method_id to build on
- `base_config` — config_id to build on
- `config_delta` — nested config changes from the base
- `scope` — benchmark scope (`county`)
- `benchmark_label` — human-readable label
- `requested_by` — `agent` or `human`

Example:

```yaml
experiment_id: exp-20260310-convergence-hold
hypothesis: >
  Extending convergence hold from 20 to 25 years will reduce long-horizon
  overshoot in college counties.
base_method: m2026r1
base_config: cfg-20260309-college-fix-v1
config_delta:
  convergence:
    hold_after_year: 2030
scope: county
benchmark_label: convergence-hold-2030
requested_by: agent
```

### Running an Experiment

```bash
source .venv/bin/activate
python scripts/analysis/run_experiment.py --spec data/analysis/experiments/pending/exp-20260310-convergence-hold.yaml
```

Optional flags:

- `--dry-run` — validate spec and print resolved config without running
- `--policy <path>` — override default evaluation policy
- `--profile-dir <path>` — override method profile directory

The orchestrator:

1. validates the spec,
2. derives `method_id` and `config_id` (or uses overrides),
3. checks `METHOD_DISPATCH` registration,
4. creates an immutable method profile,
5. runs `run_benchmark_suite.py` as a subprocess,
6. evaluates results against the evaluation policy,
7. writes an experiment log entry,
8. moves the spec from `pending/` to `completed/`,
9. prints a structured JSON result.

The orchestrator **never promotes** — it only classifies results.

### Evaluation Policy

Location: `config/benchmark_evaluation_policy.yaml`

The policy defines:

- **Hard gates** (zero tolerance): `negative_population_violations`,
  `scenario_order_violations`, `aggregation_violations`
- **Tradeoff thresholds** (max regression): `county_mape_rural` (0.10),
  `county_mape_bakken` (0.50), `county_mape_overall` (0.05)

Classification outcomes:

| Outcome | Meaning |
|---------|---------|
| `passed_all_gates` | All gates passed, all tradeoffs within threshold |
| `needs_human_review` | Gates pass but tradeoff breached or sensitivity flag |
| `failed_hard_gate` | At least one hard gate violated |
| `inconclusive` | Benchmark failed or code changes required |

### Experiment Log

Location: `data/analysis/experiments/experiment_log.csv`
Schema: `config/experiment_log_schema.yaml`

The log is append-only. Each entry records:

- `experiment_id`, `run_date`, `hypothesis`, `base_method`
- `config_delta_summary`, `run_id`, `outcome`
- `key_metrics_summary`, `interpretation`, `next_action`
- `agent_or_human`, `spec_path`

Query utilities in `cohort_projections/analysis/experiment_log.py`:

- `read_experiment_log()` — full log as DataFrame
- `get_tested_hypotheses()` — set of experiment_ids (dedup check)

### Agent Experiment Workflow

1. Check `data/analysis/experiments/pending/` for queued specs.
2. Check `experiment_log.csv` for what has already been tried.
3. Run the next pending experiment via `run_experiment.py --spec <path>`.
4. Review the JSON output for the classification.
5. Flag any `needs_human_review` results and stop on that line of inquiry.

### Directory Map (Updated)

| Path | Purpose |
|------|---------|
| `config/experiment_spec_schema.yaml` | Experiment spec contract |
| `config/experiment_log_schema.yaml` | Log entry contract |
| `config/benchmark_evaluation_policy.yaml` | Machine-readable gates and thresholds |
| `data/analysis/experiments/pending/` | Queued experiment specs |
| `data/analysis/experiments/completed/` | Executed experiment specs |
| `data/analysis/experiments/experiment_log.csv` | Append-only experiment journal |
| `scripts/analysis/run_experiment.py` | Single-command experiment orchestrator |
| `cohort_projections/analysis/evaluation_policy.py` | Policy evaluation logic |
| `cohort_projections/analysis/experiment_log.py` | Log read/write utilities |
| `scripts/analysis/build_experiment_dashboard.py` | Interactive experiment comparison dashboard |
| `data/analysis/experiments/experiment_dashboard.html` | Generated dashboard output |

## Dashboard and Visualization

After running experiments, generate an interactive HTML dashboard to compare
results across all completed experiments:

```bash
source .venv/bin/activate
python scripts/analysis/build_experiment_dashboard.py
```

The dashboard is written to `data/analysis/experiments/experiment_dashboard.html`
and includes five interactive tabs:

1. **Experiment Tracker** — catalog status, KPIs, outcome badges, MAPE deltas
2. **Spaghetti Plot** — overlaid state projection curves with origin-year filtering
3. **Scorecard Comparison** — grouped bar chart of deltas + full metrics table
4. **Horizon Analysis** — MAPE and bias degradation by forecast horizon
5. **County Detail** — heatmap of county-level performance by experiment

Optional flags:

- `--output <path>` — write dashboard to a custom location
- `--no-plotly-js` — load Plotly.js from CDN (smaller file, requires internet)

The dashboard reads from existing benchmark history artifacts; it does not
re-run any experiments.

## Batch Sweeps

The sweep runner automates running multiple experiments in sequence with dedup
checking and dashboard regeneration.

### Spec List Mode

Run specific experiment specs in order:

```bash
source .venv/bin/activate
python scripts/analysis/run_experiment_sweep.py \
  --specs data/analysis/experiments/pending/exp-001.yaml \
         data/analysis/experiments/pending/exp-002.yaml
```

### Grid Mode

Generate specs from a parameter grid and run them all:

```bash
source .venv/bin/activate
python scripts/analysis/run_experiment_sweep.py --grid config/grids/blend-sweep.yaml
```

Grid YAML format:

```yaml
base_method: m2026r1
base_config: cfg-20260309-college-fix-v1
scope: county
requested_by: agent
parameters:
  college_blend_factor: [0.5, 0.6, 0.7, 0.8, 0.9]
mode: cartesian  # or "zip" for paired parameters
```

### Pending Mode

Run all specs queued in the pending directory:

```bash
source .venv/bin/activate
python scripts/analysis/run_experiment_sweep.py --pending
```

### Behavior

- **Dedup**: Before running each spec, the sweep checks the experiment log.
  Already-tested experiment IDs are skipped with a message.
- **Fault tolerance**: If a single experiment fails, the sweep logs the
  failure and continues with the next spec.
- **Dashboard**: After all specs complete, the dashboard is automatically
  regenerated via `build_experiment_dashboard.py`.
- **Summary table**: A summary table is printed showing experiment outcomes
  and key metric deltas.

## Projection Observatory

The Projection Observatory is a unified analysis and comparison layer built on
top of the experiment and sweep infrastructure. It consolidates all benchmark
results into a single store, computes N-way comparisons, and recommends the
next experiments to run.

### How It Integrates

The Observatory does not replace the sweep or experiment runner. It delegates
to them for execution:

1. **Sweep** produces benchmark run bundles under `data/analysis/benchmark_history/`.
2. The Observatory's `ResultsStore` loads all run scorecards from that history.
3. `ObservatoryComparator` ranks runs, computes deltas against the champion, and
   identifies Pareto-optimal configurations.
4. `ObservatoryRecommender` analyzes tested parameter ranges and the variant
   catalog to suggest what to try next.
5. `run-pending` and `run-recommended` generate experiment specs from the
   variant catalog and invoke the sweep runner for execution.

### CLI Commands

```bash
source .venv/bin/activate
python scripts/analysis/observatory.py status          # Inventory and catalog status
python scripts/analysis/observatory.py compare         # Full N-way comparison report
python scripts/analysis/observatory.py rank <metric>   # Rank all runs by a specific metric
python scripts/analysis/observatory.py recommend       # Next-experiment suggestions
python scripts/analysis/observatory.py run-pending     # Run all untested catalog variants
python scripts/analysis/observatory.py run-pending --dry-run  # Preview without executing
python scripts/analysis/observatory.py run-recommended     # Run config-only recommendations
python scripts/analysis/observatory.py run-recommended --dry-run  # Preview recommendations
python scripts/analysis/observatory.py diff <id1> <id2>    # Head-to-head run comparison
python scripts/analysis/observatory.py history         # Chronological experiment progression
python scripts/analysis/observatory.py report          # Generate HTML observatory report
python scripts/analysis/observatory.py refresh         # Rebuild results cache
python scripts/analysis/observatory.py --format json status  # Machine-readable output
```

### Execution Workflows

**`run-pending`** iterates over all variants in the catalog that have no
recorded results, generates experiment specs via `catalog.generate_spec()`,
and runs them through the sweep infrastructure. Use `--dry-run` to preview
which variants would be executed.

**`run-recommended`** asks the recommender for suggestions (boundary
extensions, untested catalog entries, interaction candidates), filters to
config-only variants, generates specs, and runs them. Use `--dry-run` to
preview the recommendations without execution.

### Configuration

| File | Purpose |
|------|---------|
| `config/observatory_config.yaml` | Paths (history dir, cache dir, experiment log, variant catalog), champion/challenger method IDs, comparison metrics, recommendation thresholds |
| `config/observatory_variants.yaml` | Variant definitions (11 experiments EXP-A through EXP-K), parameter bounds, grid sweep definitions |

### Results Flow

```
experiment spec → sweep runner → benchmark_history/<run_id>/
    → observatory ResultsStore (loads scorecards)
    → compare / rank / recommend
```

### Directory Map (Observatory)

| Path | Purpose |
|------|---------|
| `cohort_projections/analysis/observatory/` | Observatory package (results_store, comparator, recommender, variant_catalog, report) |
| `config/observatory_config.yaml` | Observatory settings |
| `config/observatory_variants.yaml` | Variant catalog and grid definitions |
| `scripts/analysis/observatory.py` | CLI entry point |
| `data/analysis/observatory/` | Cache directory for observatory results |
| `tests/test_analysis/test_observatory_*.py` | 176 tests across 5 test files |

## Related Documents

- [SOP-003](../governance/sops/SOP-003-method-benchmarking-versioning-promotion.md)
- [testing-workflow.md](./testing-workflow.md)
- [DEVELOPMENT_TRACKER.md](../../DEVELOPMENT_TRACKER.md)
- [Benchmarking Process Improvement Roadmap](../plans/benchmarking-process-improvement-roadmap.md)
- [BM-001 Implementation Plan](../plans/benchmarking-p0-implementation-plan.md)
