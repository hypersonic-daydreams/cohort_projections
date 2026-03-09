---
title: "P0/P0.5/P1 Implementation Plan — Agent Experiment Infrastructure"
created: 2026-03-09
status: ready-for-implementation
parent: docs/plans/benchmarking-process-improvement-roadmap.md
tracker_id: BM-001
purpose: >
  Decompose the P0 (orchestrator), P0.5 (experiment log), and P1-stub
  (promotion thresholds) roadmap items into discrete, agent-assignable tasks
  with clear file ownership, dependencies, and definitions of done.
---

# P0/P0.5/P1 Implementation Plan

## Coordination Model

This plan is designed for a session that uses **sub-agents working in parallel**
on independent tasks, with sequential hand-offs where dependencies exist.

### Key Principles

1. **File ownership**: Each task lists the files it creates or modifies. Two tasks never modify the same file.
2. **Dependency arrows**: Tasks list prerequisites. An agent should not start a task until its prerequisites are complete.
3. **Append-only outputs**: Experiment logs, benchmark index, and promotion history are append-only. Multiple agents can write to these without conflict as long as writes are atomic (one entry per append).
4. **Immutable inputs**: Method profiles, benchmark bundles, and experiment specs are write-once. No task modifies an existing profile or spec.
5. **No code deletion**: New code is additive. Existing scripts (`run_benchmark_suite.py`, `compare_benchmark_runs.py`, `promote_method.py`) are called by the orchestrator, not replaced.

### Agent Types Needed

| Agent Role | Scope | Parallelizable With |
|------------|-------|---------------------|
| Schema Agent | Define experiment spec format, evaluation policy schema, experiment log schema | All others (runs first) |
| Orchestrator Agent | Build `run_experiment.py` | Policy Agent, Log Agent (after Schema) |
| Policy Agent | Build evaluation policy YAML and validation logic | Orchestrator Agent, Log Agent (after Schema) |
| Log Agent | Build experiment log writer and reader utilities | Orchestrator Agent, Policy Agent (after Schema) |
| Test Agent | Write tests for all new modules | Runs after Orchestrator + Policy + Log |
| Docs Agent | Update workflow guide, AGENTS.md, tracker | Runs after Test |

---

## Task Breakdown

### Phase 1: Schemas and Directory Structure (Sequential — runs first)

#### BM-001-01: Create experiment directory structure and schemas

**Creates:**
- `data/analysis/experiments/pending/` (directory, already created)
- `data/analysis/experiments/completed/` (directory, already created)
- `data/analysis/experiments/.gitkeep` (preserves empty dirs in git)
- `config/experiment_spec_schema.yaml` (canonical schema for experiment specs)
- `config/experiment_log_schema.yaml` (canonical schema for log entries)

**Definition of Done:**
- Schema files define all required and optional fields with types and constraints
- Schema files include inline examples
- Directory structure exists and is tracked in git

**Dependencies:** None

---

### Phase 2: Core Implementation (Parallel — three agents can work simultaneously)

#### BM-001-02: Build evaluation policy and validation logic

**Creates:**
- `config/benchmark_evaluation_policy.yaml` (machine-readable gates and thresholds)
- `cohort_projections/analysis/evaluation_policy.py` (load policy, evaluate scorecard against policy, return classification)

**Modifies:** Nothing existing

**Definition of Done:**
- Policy YAML defines `hard_gates`, `tradeoff_thresholds`, and `agent_classification_rules`
- `evaluate_scorecard(scorecard_row, policy)` returns one of: `passed_all_gates`, `needs_human_review`, `failed_hard_gate`, `inconclusive`
- Hard gates: zero tolerance for `negative_population_violations`, `scenario_order_violations`, `aggregation_violations`
- Tradeoff thresholds: configurable max-regression values for county MAPE categories
- Policy is loaded from YAML, not hard-coded

**Dependencies:** BM-001-01 (schema awareness)

**Key Design Constraint:** The evaluation function must consume the same `summary_scorecard.csv` row format that `build_summary_scorecard()` already produces. No scorecard schema changes.

---

#### BM-001-03: Build experiment log writer and reader

**Creates:**
- `cohort_projections/analysis/experiment_log.py` (append entry, read log, query by experiment_id/method/outcome)
- `data/analysis/experiments/experiment_log.csv` (initialized with header row only)

**Modifies:** Nothing existing

**Definition of Done:**
- `append_experiment_entry(entry_dict)` atomically appends one row to CSV
- `read_experiment_log()` returns a DataFrame of all entries
- `get_tested_hypotheses()` returns a set of experiment_ids for dedup checks
- CSV header matches the schema from BM-001-01
- First write creates the file; subsequent writes append
- Entries include: `experiment_id`, `run_date`, `hypothesis`, `base_method`, `config_delta_summary`, `run_id`, `outcome`, `key_metrics_summary`, `interpretation`, `next_action`, `agent_or_human`, `spec_path`

**Dependencies:** BM-001-01 (schema)

---

#### BM-001-04: Build experiment orchestrator

**Creates:**
- `scripts/analysis/run_experiment.py` (CLI entry point)

**Modifies:** Nothing existing (calls existing scripts as subprocesses or imports their functions)

**Definition of Done:**
- Accepts `--spec <path>` argument pointing to an experiment spec YAML
- Reads spec, validates against schema
- Derives `method_id` and `config_id` from spec (or uses explicitly provided ones)
- Creates immutable method profile in `config/method_profiles/`
- Calls `run_benchmark_suite.py` logic (import, not subprocess) with correct champion/challenger args
- Loads the resulting scorecard and evaluates against policy (BM-001-02)
- Writes experiment log entry (BM-001-03)
- Moves spec from `pending/` to `completed/`
- Prints structured JSON result to stdout: `{ "outcome": "...", "run_id": "...", "experiment_id": "..." }`
- Does NOT promote — only classifies

**Dependencies:** BM-001-01, BM-001-02, BM-001-03

**Key Design Constraints:**
- The orchestrator wraps existing tooling. It must not duplicate logic from `benchmarking.py` or `run_benchmark_suite.py`.
- Method profile generation from `base_config + config_delta` must produce a valid profile that passes `load_method_profile()` validation.
- If the spec's `config_delta` requires changes to `METHOD_DISPATCH` in `walk_forward_validation.py`, the orchestrator should detect this and return `inconclusive` with a message explaining that code changes are needed (config-only deltas are the autonomous path; code deltas require human/agent intervention).

---

### Phase 3: Testing (Sequential — after Phase 2)

#### BM-001-05: Write tests for evaluation policy

**Creates:**
- `tests/test_analysis/test_evaluation_policy.py`

**Definition of Done:**
- Tests `evaluate_scorecard()` with synthetic scorecards that trigger each classification
- Tests policy loading from YAML
- Tests edge cases: exactly-at-threshold, missing fields, multiple violations
- All tests pass

**Dependencies:** BM-001-02

---

#### BM-001-06: Write tests for experiment log

**Creates:**
- `tests/test_analysis/test_experiment_log.py`

**Definition of Done:**
- Tests append, read, dedup query
- Tests CSV creation on first write
- Tests append-only property (no overwrite)
- All tests pass

**Dependencies:** BM-001-03

---

#### BM-001-07: Write tests for experiment orchestrator

**Creates:**
- `tests/test_analysis/test_run_experiment.py`

**Definition of Done:**
- Tests spec parsing and validation
- Tests method profile derivation from base + delta
- Tests end-to-end flow with mocked benchmark suite (synthetic scorecard)
- Tests spec movement from pending to completed
- Tests classification output for each outcome type
- Tests that config-only deltas work autonomously
- Tests that code-requiring deltas return `inconclusive`
- All tests pass

**Dependencies:** BM-001-04, BM-001-05, BM-001-06

---

### Phase 4: Documentation and Tracker (Sequential — after Phase 3)

#### BM-001-08: Update documentation

**Modifies:**
- `docs/guides/benchmarking-workflow.md` (add experiment orchestration section)
- `DEVELOPMENT_TRACKER.md` (mark BM-001 tasks as complete)

**Creates:**
- Nothing new (extend existing docs)

**Definition of Done:**
- Benchmarking workflow guide includes: experiment spec format reference, `run_experiment.py` usage, evaluation policy reference, experiment log location and format
- Tracker updated with completion evidence

**Dependencies:** BM-001-07

---

## Dependency Graph

```
BM-001-01 (schemas)
    ├──→ BM-001-02 (policy)      ─┐
    ├──→ BM-001-03 (log)         ─┤── can run in parallel
    │                              │
    └──→ BM-001-04 (orchestrator) ←┘── depends on 02 + 03
              │
              ├──→ BM-001-05 (policy tests)      ─┐
              ├──→ BM-001-06 (log tests)          ─┤── can run in parallel
              └──→ BM-001-07 (orchestrator tests) ←┘── depends on 05 + 06
                        │
                        └──→ BM-001-08 (docs)
```

## Parallel Execution Plan

**Wave 1** (1 agent): BM-001-01

**Wave 2** (3 agents in parallel):
- Agent A: BM-001-02 (evaluation policy)
- Agent B: BM-001-03 (experiment log)
- Agent C: BM-001-04 (orchestrator — can start structure while waiting for 02/03, but integration depends on them)

**Wave 3** (up to 3 agents in parallel):
- Agent D: BM-001-05 (policy tests)
- Agent E: BM-001-06 (log tests)
- Agent F: BM-001-07 (orchestrator tests — may need 05/06 patterns)

**Wave 4** (1 agent): BM-001-08 (docs)

---

## Files Created vs Modified Summary

### New Files (safe for parallel creation — no conflicts)

| File | Task | Purpose |
|------|------|---------|
| `config/experiment_spec_schema.yaml` | 01 | Experiment spec contract |
| `config/experiment_log_schema.yaml` | 01 | Log entry contract |
| `config/benchmark_evaluation_policy.yaml` | 02 | Machine-readable gates and thresholds |
| `cohort_projections/analysis/evaluation_policy.py` | 02 | Policy evaluation logic |
| `cohort_projections/analysis/experiment_log.py` | 03 | Log read/write utilities |
| `data/analysis/experiments/experiment_log.csv` | 03 | Append-only experiment journal |
| `scripts/analysis/run_experiment.py` | 04 | Single-command orchestrator |
| `tests/test_analysis/test_evaluation_policy.py` | 05 | Policy tests |
| `tests/test_analysis/test_experiment_log.py` | 06 | Log tests |
| `tests/test_analysis/test_run_experiment.py` | 07 | Orchestrator tests |

### Modified Files (sequential access only)

| File | Task | Change |
|------|------|--------|
| `docs/guides/benchmarking-workflow.md` | 08 | Add experiment orchestration section |
| `DEVELOPMENT_TRACKER.md` | 08 | Add BM-001 checklist, mark complete |

### Untouched Files (read-only references)

| File | Used By | How |
|------|---------|-----|
| `scripts/analysis/run_benchmark_suite.py` | 04 | Called/imported by orchestrator |
| `cohort_projections/analysis/benchmarking.py` | 02, 03, 04 | Imports: `load_method_profile`, `build_summary_scorecard`, etc. |
| `scripts/analysis/walk_forward_validation.py` | 04 | Checked for METHOD_DISPATCH registration |
| `config/method_profiles/aliases.yaml` | 04 | Read for champion resolution |
| `data/analysis/benchmark_history/index.csv` | 04 | Appended by existing benchmark suite |

---

## Acceptance Criteria (Full BM-001)

1. An agent can write an experiment spec YAML and run `python scripts/analysis/run_experiment.py --spec <path>`.
2. The orchestrator produces an immutable benchmark bundle, a log entry, and a classification.
3. The experiment log is append-only and queryable.
4. The evaluation policy is loaded from YAML, not hard-coded.
5. No existing scripts, modules, or benchmark data are modified or deleted.
6. All new code has tests that pass.
7. A human reviewer can read the experiment log and trace from hypothesis to outcome to next action.

---

## Session-Start Instructions

When starting the implementation session:

1. Read this plan first.
2. Confirm the current test baseline: `pytest tests/ -q -k "not test_residual_computation_single_period"`.
3. Start with BM-001-01 (schemas) — this unblocks everything else.
4. After BM-001-01, launch Wave 2 agents in parallel.
5. After Wave 2, launch Wave 3 test agents.
6. Finish with BM-001-08 (docs and tracker update).
7. Final verification: full test suite passes, experiment log is empty but valid, evaluation policy loads correctly.
