# Benchmarking Workflow

Practical guide for running the versioned benchmark workflow defined in
[SOP-003](../governance/sops/SOP-003-method-benchmarking-versioning-promotion.md).

This guide is the operator-facing companion to SOP-003:

- SOP-003 defines the governance rules, schemas, and required artifacts.
- This guide shows the day-to-day commands and decision flow.

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

## Related Documents

- [SOP-003](../governance/sops/SOP-003-method-benchmarking-versioning-promotion.md)
- [testing-workflow.md](./testing-workflow.md)
- [DEVELOPMENT_TRACKER.md](../../DEVELOPMENT_TRACKER.md)
