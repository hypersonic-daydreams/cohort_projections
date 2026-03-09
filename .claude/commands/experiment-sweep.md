---
description: Autonomously run a batch of experiments from the catalog, using sub-agents
allowed-tools: Bash, Read, Edit, Write, Glob, Grep, Agent, AskUserQuestion
---

# Experiment Sweep

You are executing the `/experiment-sweep` command for the cohort_projections project.
This command autonomously runs multiple experiments from the experiment catalog,
using sub-agents for parallel execution, and documents all outcomes.

## Before You Begin

Gather context by running these steps **in parallel**:

1. **Read the experiment catalog** — `docs/plans/experiment-catalog.md`
2. **Read the current aliases** — `config/method_profiles/aliases.yaml`
3. **List available method profiles** — `config/method_profiles/*.yaml`
4. **Check pending experiments** — `data/analysis/experiments/pending/`
5. **Check completed experiments** — `data/analysis/experiments/completed/`
6. **Read the experiment log** — `data/analysis/experiments/experiment_log.csv`
7. **Read the spec schema** — `config/experiment_spec_schema.yaml`
8. **Read the evaluation policy** — `config/benchmark_evaluation_policy.yaml`

Present a status summary:

```
Sweep Status
  Champion:  <method_id> / <config_id>
  Candidate: <method_id> / <config_id>
  Catalog entries: <count total> (<count untested>)
  Pending specs: <count>
  Completed experiments: <count>
  Log entries: <count>
```

## Step 1: Select Experiments

Cross-reference the catalog against the experiment log to find untested experiments.
Identify which catalog entries have NOT been run yet.

**Selection rules:**
- Default: run the next 2-3 experiments from the catalog's recommended order
- If the user specifies experiments (e.g., "run EXP-A and EXP-B"), use those
- If the user says "all tier 1", run all Tier 1 experiments
- Skip any experiment whose `experiment_id` already appears in the log
- Never run more than 4 experiments in a single sweep (resource constraint)

Present the selected experiments and ask for confirmation:

```
Selected for this sweep:
  1. EXP-A: convergence-medium-hold-5 — extend medium hold from 3 to 5 years
  2. EXP-B: college-blend-70 — increase blend factor from 0.5 to 0.7

Proceed? (yes / edit selection / no)
```

## Step 2: Generate Specs

For each selected experiment, generate a YAML spec file using the catalog entry.
Use today's date (`date +%Y%m%d`) for the experiment_id.

Write all specs to `data/analysis/experiments/pending/`.

## Step 3: Validate All Specs

Run dry-run validation for each spec **sequentially** (they share state):

```bash
source .venv/bin/activate && python scripts/analysis/run_experiment.py \
  --spec data/analysis/experiments/pending/<experiment_id>.yaml \
  --dry-run
```

If any spec fails validation:
- Show the error
- Fix it
- Re-validate
- If unfixable, remove from the sweep and note why

Report validation results:

```
Validation Results
  ✓ exp-20260309-convergence-medium-hold-5 — valid
  ✓ exp-20260309-college-blend-70 — valid
```

## Step 4: Run Experiments

**IMPORTANT:** Run experiments **sequentially, not in parallel**. Each benchmark run
is resource-intensive and they may share intermediate state (METHOD_DISPATCH registration,
profile creation). Running them in parallel risks race conditions.

For each validated spec, run the full benchmark:

```bash
source .venv/bin/activate && python scripts/analysis/run_experiment.py \
  --spec data/analysis/experiments/pending/<experiment_id>.yaml
```

Each run will take several minutes. After each completes:
1. Capture the JSON output
2. Record the classification
3. Note key metrics
4. Continue to the next experiment

## Step 5: Document Results

After all experiments complete, update the experiment catalog with results.

### 5a. Update the Catalog

Edit `docs/plans/experiment-catalog.md` to add a **Results** row to each tested
experiment's table:

```markdown
| Results | `passed_all_gates` — rural MAPE improved 0.03, Bakken MAPE unchanged. Run: `br-YYYYMMDD-...` |
```

### 5b. Write a Sweep Summary

Append a sweep summary section at the bottom of the catalog:

```markdown
## Sweep Results — YYYY-MM-DD

| Experiment | Classification | Key Delta | Recommendation |
|------------|---------------|-----------|----------------|
| EXP-A | passed_all_gates | rural MAPE -0.03 | Consider promotion |
| EXP-B | needs_human_review | college MAPE -0.02, Bakken +0.08 | Review Bakken regression |
```

### 5c. Present Summary to User

Show a concise results table and interpretation:

- Which experiments passed all gates?
- Which need human review?
- Which failed?
- What's the recommended next step?

If any experiment `passed_all_gates` and is clearly superior, suggest running
`/experiment` with a combined config delta that stacks the winning changes.

## Adaptive Behavior

### If an experiment reveals a new idea
Note it in the catalog under a new "Generated" tier, but do NOT run it in
this sweep. Let the user decide in a future sweep.

### If two experiments interact
If EXP-A and EXP-B both pass, and they modify related parameters, note that
a combined experiment should be considered. Add it to the catalog.

### If all experiments fail
Investigate whether the failures share a root cause (e.g., a code bug in
METHOD_DISPATCH, a data issue). Report the pattern rather than individual failures.

## Key Constraints

- **Never promote** — sweeps only classify. Promotion requires `/experiment` or manual review.
- **Never modify projection code** — sweeps only test config changes.
- **Always log** — every run must appear in `experiment_log.csv`, even failures.
- **Respect the catalog order** — unless the user overrides, follow the recommended sequence.
- **Stop on hard gate failure pattern** — if 2+ experiments fail the same hard gate,
  stop the sweep and investigate rather than burning through remaining specs.

## Quick Reference

| File | Purpose |
|------|---------|
| `docs/plans/experiment-catalog.md` | Experiment ideas and results |
| `config/experiment_spec_schema.yaml` | Spec format reference |
| `config/benchmark_evaluation_policy.yaml` | Gate thresholds |
| `data/analysis/experiments/pending/` | Queued specs |
| `data/analysis/experiments/completed/` | Executed specs |
| `data/analysis/experiments/experiment_log.csv` | Append-only result journal |

Now begin by gathering context (the parallel reads listed in "Before You Begin").
