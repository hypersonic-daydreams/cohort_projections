---
description: Draft, validate, and run a benchmark experiment spec
allowed-tools: Bash, Read, Edit, Write, Glob, Grep, Agent, AskUserQuestion
---

# Benchmark Experiment

You are executing the `/experiment` command for the cohort_projections project.
This command walks through the full experiment lifecycle: draft a spec, validate it,
and optionally run it through the benchmark pipeline.

## Before You Begin

Gather context by running these steps **in parallel**:

1. **Read the current aliases** — `config/method_profiles/aliases.yaml`
2. **List available method profiles** — `config/method_profiles/*.yaml` (not aliases)
3. **Check pending experiments** — `data/analysis/experiments/pending/`
4. **Check completed experiments** — `data/analysis/experiments/completed/`
5. **Read the experiment log** — `data/analysis/experiments/experiment_log.csv`
6. **Read the spec schema** — `config/experiment_spec_schema.yaml`

Present a brief status summary:

```
Benchmark Status
  Champion:  <method_id> / <config_id>
  Candidate: <method_id> / <config_id>  (or "none")
  Pending experiments: <count>
  Completed experiments: <count>
  Log entries: <count>
```

## Step 1: Understand the Change

Ask the user what they want to test. If they have already described it, proceed.

Good experiments have:
- A **single config change** (or small, coherent set of changes)
- A **falsifiable hypothesis** — state the expected direction of improvement
- A **clear base** — which existing method/config to build on

Help the user articulate:
1. What config parameter(s) are changing?
2. What do they expect to improve?
3. What might get worse?

If the user is unsure what to test, read the current champion profile and
`config/projection_config.yaml` to suggest tunable parameters.

## Step 2: Draft the Experiment Spec

Generate a YAML spec using this template. Today's date for the ID is from `date +%Y%m%d`.

```yaml
experiment_id: "exp-YYYYMMDD-short-slug"
hypothesis: >
  One sentence: what changes and what improvement is expected.
base_method: "<method_id>"
base_config: "<config_id>"
config_delta:
  # Only the changed parameters, using projection_config.yaml key paths
  section:
    parameter: new_value
scope: "county"
benchmark_label: "human-readable-label"
requested_by: "human"
expected_improvement:
  - "metric expected to improve"
risk_areas:
  - "metric or geography that might degrade"
notes:
  - "Any additional context"
```

**Rules for the spec:**
- `experiment_id` format: `exp-YYYYMMDD-short-slug` (2-4 hyphenated words)
- `base_method` and `base_config` must reference an existing profile
- `config_delta` must contain at least one key-value pair
- `scope` is always `county` for now
- `requested_by` is `human` when initiated via this command

Show the draft to the user and ask for confirmation or edits.

## Step 3: Write the Spec File

Once the user approves, write the spec to:

```
data/analysis/experiments/pending/<experiment_id>.yaml
```

## Step 4: Validate (Dry Run)

Run the orchestrator in dry-run mode to validate the spec:

```bash
source .venv/bin/activate && python scripts/analysis/run_experiment.py \
  --spec data/analysis/experiments/pending/<experiment_id>.yaml \
  --dry-run
```

If validation fails:
- Show the error
- Fix the spec
- Re-validate

If validation passes, show the resolved config summary and ask the user
whether to proceed with the full benchmark run.

## Step 5: Run (Optional)

If the user says yes, execute the full benchmark:

```bash
source .venv/bin/activate && python scripts/analysis/run_experiment.py \
  --spec data/analysis/experiments/pending/<experiment_id>.yaml
```

This will take several minutes. Run it and monitor for completion.

When done, present:
- **Classification**: passed_all_gates / needs_human_review / failed_hard_gate / inconclusive
- **Key metrics** from the JSON output
- **Interpretation** and recommended next action

If the user says no (or wants to run later), confirm that the spec is saved
in `pending/` and remind them they can run it with:

```bash
python scripts/analysis/run_experiment.py --spec data/analysis/experiments/pending/<experiment_id>.yaml
```

## Quick Reference

### Evaluation gates

| Gate | Threshold |
|------|-----------|
| negative_population_violations | 0 (hard) |
| scenario_order_violations | 0 (hard) |
| aggregation_violations | 0 (hard) |
| county_mape_rural | max +0.10 regression |
| county_mape_bakken | max +0.50 regression |
| county_mape_overall | max +0.05 regression |

### Key files

| File | Purpose |
|------|---------|
| `config/experiment_spec_schema.yaml` | Spec format reference |
| `config/benchmark_evaluation_policy.yaml` | Gate thresholds |
| `config/method_profiles/aliases.yaml` | Current champion/candidate pointers |
| `data/analysis/experiments/experiment_log.csv` | Append-only result journal |
| `docs/guides/benchmarking-workflow.md` | Full workflow documentation |

### After the experiment

- `passed_all_gates` → Consider promotion (separate workflow)
- `needs_human_review` → Examine scorecard and decision record before proceeding
- `failed_hard_gate` → Investigate constraint violations, revise approach
- `inconclusive` → Debug code/data issues before retrying

Now begin by gathering context (the parallel reads listed in "Before You Begin").
