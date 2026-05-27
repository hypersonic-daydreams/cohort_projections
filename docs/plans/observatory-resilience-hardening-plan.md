# Projection Observatory Resilience Hardening Plan

Created: 2026-05-27
Status: Proposed follow-on hardening plan
Scope: Projection Observatory dashboard, deep-search controller, sandbox orchestration, and session recovery

## Purpose

This note captures recommended follow-on work after the 2026-05-27 deep-search
launch issues. The failures were primarily orchestration and artifact-contract
problems, not demographic methodology failures. The goal is to make the
Projection Observatory easier to operate during active project work, clearer
when something fails, and more resilient to partial runs or interrupted
background processes.

This document is a proposed hardening plan. Treat `DEVELOPMENT_TRACKER.md` as
the source of truth for active work; move items there before implementation.

## Observed Failure Modes

The 2026-05-27 session exposed several reliability gaps:

1. Dashboard launches delegate most validation to the child CLI process. If the
   child exits early, the dashboard only detects it through a short startup
   poll and log tail.
2. Deep search can write a planned session before execution preflight has
   passed, leaving a half-real `search-*` folder after startup failure.
3. Candidate state is marked `running` in bulk before work begins. If the
   process is interrupted, session state can become ambiguous.
4. Operational blockers are represented as completed candidates, which can
   make a session look healthier than it is.
5. Sandbox/worktree assumptions are too implicit. Worktree provisioning, data
   overlays, benchmark bundle locations, experiment logs, and completed specs
   all need explicit contract checks.
6. Dashboard progress depends mostly on PID checks and session files. It needs
   stronger heartbeat and recovery semantics.
7. Dirty-check behavior was originally too blunt. It blocked unrelated
   planning and marketing files even though they could not affect benchmark
   reproducibility. This specific issue was partially addressed on 2026-05-27
   by classifying only material paths as blocking.

## Design Goals

- Preserve reproducibility without blocking unrelated documentation work.
- Fail before session planning when a launch cannot run.
- Make every failed state visible and actionable in the dashboard.
- Keep completed benchmark evidence distinguishable from operational blockers.
- Make interrupted searches resumable or explicitly archivable.
- Centralize preflight, state, and artifact-contract logic so dashboard and CLI
  behavior stay aligned.

## Recommended Work

### 1. Shared Launch Preflight

Create a shared `SearchLaunchPreflight` used by both the dashboard and CLI
before `plan_session()`.

It should check:

- material dirty files and non-blocking dirty files;
- base revision resolution;
- search ID collision and overwrite policy;
- search pack existence and schema validity;
- configured session, runtime, mirror, and worktree directories;
- shared data availability needed for benchmark execution;
- active/stale dashboard-launched process state;
- whether benchmark history and experiment logs are writable.

The result should be structured as `pass`, `warn`, or `block`, with issue codes,
plain-language impact, and next action. The dashboard should render this report
before launching rather than requiring the user to inspect logs after failure.

### 2. Explicit Session State Machine

Replace ad hoc status strings with a small state machine:

- `preflight_failed`
- `planned`
- `running`
- `interrupted`
- `finished`
- `failed_startup`
- `archived`

Only create a normal planned session after launch preflight passes. If planning
itself fails, write a separate launch diagnostic artifact rather than a normal
search session.

### 3. Heartbeat And Recovery

Have deep search write a heartbeat artifact while running, for example:

```text
data/analysis/experiments/search_runs/<search_id>/runtime/heartbeat.json
```

The heartbeat should include PID, process group, current phase, current
candidate IDs, last progress timestamp, and a monotonic sequence number. The
dashboard should combine PID matching, heartbeat freshness, and session status
to determine whether a search is active, stale, interrupted, or finished.

When a session is stale, the dashboard should offer clear actions:

- `Resume`
- `Archive failed launch`
- `Rerun with same settings`
- `Open logs`

### 4. Better Candidate Outcome Taxonomy

Separate operational execution state from benchmark outcome.

Recommended candidate execution states:

- `planned`
- `running`
- `completed_benchmark`
- `operational_blocker`
- `candidate_failed`
- `skipped_existing`

Recommended benchmark outcomes should remain focused on evidence quality, such
as `recommended`, `needs_human_review`, `failed_hard_gate`, and `inconclusive`.

The dashboard should report operational blockers explicitly as blockers, not as
normal completed candidates.

### 5. Artifact Contract Validation

Add a contract layer for each candidate run. A successful benchmark-backed
candidate should prove that the following artifacts exist and agree:

- candidate spec;
- completed spec;
- experiment log row;
- run ID;
- benchmark bundle;
- benchmark manifest;
- summary scorecard;
- runtime summary;
- benchmark index row;
- candidate summary row.

When a contract fails, the candidate should be quarantined with a structured
reason and the session should continue only if the operational-failure policy
allows it.

### 6. Defensive Worktree Lifecycle

Move worktree creation into the per-candidate error boundary and make cleanup
safe even if creation fails. Worktree cleanup should never mask the primary
failure. Candidate logs should include whether failure happened during:

- worktree creation;
- recipe application;
- patch validation;
- targeted tests;
- experiment execution;
- artifact harvest;
- worktree cleanup.

### 7. End-To-End Smoke Coverage

Add a fast integration smoke test that runs one minimal search candidate and
asserts the full lifecycle:

- preflight pass;
- session planned;
- heartbeat written;
- candidate transitions through running to final state;
- logs written;
- candidate summary written;
- search report written;
- no stale running state remains.

Add a second recovery smoke test that simulates a stale heartbeat or dead PID
and verifies the dashboard/session layer marks the run as interrupted instead
of active.

## Suggested Implementation Order

1. Implement shared launch preflight and use it before `plan_session()`.
2. Add explicit startup-failure and preflight-failure diagnostics.
3. Add candidate outcome taxonomy changes and dashboard summary updates.
4. Add heartbeat writing and stale-session detection.
5. Harden worktree lifecycle and cleanup.
6. Add artifact-contract validation helpers.
7. Add end-to-end smoke and recovery tests.
8. Update operator docs after behavior stabilizes.

## Near-Term Patch Candidate

The next pragmatic change should be:

1. Add `SearchLaunchPreflight`.
2. Run it from `deep-search` before `plan_session()`.
3. Run it from the dashboard before spawning the child process.
4. Render blocking and warning issues directly in the Command Center.
5. Write a launch diagnostic artifact when preflight blocks.

That patch would eliminate the most confusing behavior from the 2026-05-27
session: a failed launch creating a normal-looking search folder.

## Out Of Scope

- Methodology changes affecting projection results.
- Race/ethnicity, geography, or output-format changes.
- Automatic method promotion. SOP-003 review and alias updates remain required.
- Replacing the deterministic search-pack and recipe model.
