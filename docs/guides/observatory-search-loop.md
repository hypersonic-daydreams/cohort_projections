# Projection Observatory Search-Loop Guide

Short operator guide for running the Projection Observatory as a bounded,
resumable search loop without drifting from the governance rules in
[SOP-003](../governance/sops/SOP-003-method-benchmarking-versioning-promotion.md).

Start with [Projection Observatory Start Here](./observatory-start-here.md) if
you need the current status, reading order, or the full follow-on backlog. This
guide only covers safe queue operation.

## Use This When

Use this guide when you want the Observatory to queue and run multiple
config-only variants with limited supervision.

Do not use this guide to auto-promote a challenger. Promotion decisions still
require the review and alias-update workflow in SOP-003.

## Guardrails

1. Activate the project environment before any Python command:

```bash
source .venv/bin/activate
```

2. Treat the queue runner as intentionally sequential.
   - The current unattended pattern is bounded, resumable, one-at-a-time execution.
   - Use run budgets and resume files instead of launching an unbounded batch.

3. Keep the same resume file for repeated invocations of the same queue.
   - This preserves checkpoint state for completed and failed attempts.

4. Review `needs_human_review` results before any promotion discussion.
   - The Observatory can classify and rank.
   - It does not replace SOP-003 review.

## Safe Operating Loop

### 1. Check current inventory and blockers

```bash
source .venv/bin/activate
python scripts/analysis/observatory.py status
```

Look for:

- `Untested runnable`
- `Untested requiring code changes`
- `Blocked grids`
- active review queue in the dashboard or experiment log

### 2. Preview a bounded pending queue

```bash
source .venv/bin/activate
python scripts/analysis/observatory.py run-pending \
  --dry-run \
  --run-budget 3 \
  --resume-file data/analysis/experiments/sweeps/observatory_pending_resume.json
```

Use this preview to confirm:

- the queue order is sensible,
- the budget is small enough for unattended execution,
- the reported resume file is the one you intend to keep using.

### 3. Run the bounded queue

```bash
source .venv/bin/activate
python scripts/analysis/observatory.py run-pending \
  --run-budget 3 \
  --retry-failures 1 \
  --resume-file data/analysis/experiments/sweeps/observatory_pending_resume.json
```

Notes:

- Re-running the same command reuses the checkpoint and skips already completed work.
- `--retry-failures 1` allows one extra attempt for prior failed specs in that resume file.
- Keep budgets small until the queue behavior matches expectations.

### 4. Preview or run recommender-generated work separately

```bash
source .venv/bin/activate
python scripts/analysis/observatory.py run-recommended \
  --dry-run \
  --run-budget 2 \
  --resume-file data/analysis/experiments/sweeps/observatory_recommended_resume.json
```

Use a separate resume file for recommendation queues so catalog sweeps and
recommendation-driven follow-ups do not overwrite each other's checkpoint state.

### 5. Inspect results before deciding what to promote

```bash
source .venv/bin/activate
python scripts/analysis/observatory.py recommend
python scripts/analysis/observatory.py report --output data/analysis/observatory/latest_report.html
```

Then review:

- `passed_all_gates` versus `needs_human_review`,
- trade-offs against the current champion,
- whether the improvement is broad enough to justify promotion under SOP-003.

## Current Stop Conditions

Pause unattended queueing when any of the following appear:

- a blocked grid or non-injectable parameter is surfaced,
- repeated failures exhaust the retry budget,
- `needs_human_review` accumulates faster than you are reviewing it,
- the recommender begins surfacing only code-change-required follow-ups.

## Recommended Defaults

- Pending queue budget: `3`
- Recommendation queue budget: `2`
- Retry failures: `1`
- Concurrency: sequential only

These defaults are conservative on purpose. Increase them only after the queue
outputs remain stable over several runs.
