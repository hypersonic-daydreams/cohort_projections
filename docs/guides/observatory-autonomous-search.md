# Projection Observatory Autonomous Search

Deterministic operating guide for running the Projection Observatory as a
code-driven search process in isolated git worktrees.

This guide extends the bounded queue workflow in
[Projection Observatory Search-Loop Guide](./observatory-search-loop.md).
Use this when you want the system to plan and run many candidates with minimal
supervision while guaranteeing that the live checkout and production aliases are
not mutated by the search run itself.

## Design Boundary

- Search execution is deterministic and code-driven.
- The controller runs candidates in disposable worktrees created from a bare
  mirror, not in the live checkout.
- The controller can write append-only artifacts and reports.
- The controller cannot promote a champion, update aliases, or merge branches.
- Search-only recipe candidates can now clone a registered base method into a
  unique sandbox-only method ID at benchmark time, so recipe experiments stay
  separate from production aliases and the live checkout.

## Files

- CLI: `python scripts/analysis/observatory.py search-plan|search-run|search-status|search-report|search-auto`
- Policy: `config/observatory_search_policy.yaml`
- Recipe catalog: `config/observatory_recipes.yaml`
- Session state: `data/analysis/experiments/search_runs/<search_id>/`
- Runtime mirror/worktrees: `data/analysis/observatory_runtime/`

## Safe Loop

### Fastest Hands-Off Launch

If you want the most straightforward unattended run, use `search-auto`:

```bash
source .venv/bin/activate
python scripts/analysis/observatory.py search-auto \
  --search-id search-20260315-a \
  --batch-run-budget 5
```

This will:

- create or overwrite the autonomous-search session,
- plan pending variants, recommendations, and enabled recipe candidates,
- run the session in repeated batches until it is exhausted or hits your caps,
- refresh the Observatory cache,
- write a per-candidate summary CSV/JSON plus a Markdown search report into the
  session directory,
- write a full Observatory HTML report into the same session directory unless
  `--skip-observatory-report` is used.

If you prefer to operate from the dashboard, open the Projection Observatory
`Command Center` tab and use the `Autonomous Search` card. It can:

- preview a `search-plan`,
- preview the next `search-run --dry-run` batch,
- launch `search-auto` in the background,
- stop dashboard-launched `search-auto` processes with `SIGTERM`,
- auto-refresh against persisted `session.yaml` state,
- show progress bars, session counts, and candidate/result tables for the
  selected search session,
- preview the selected session's Markdown search report, best completed
  candidates, and dashboard-captured launch log tail.

The current enabled recipe catalog is intentionally broader than the original
three-point pilot. It now covers:

- a convergence-window family (`recent/medium` = `2/2`, `2/3`, `3/3`),
- a mortality-improvement family (`0.003`, `0.004`, `0.006`),
- three interaction recipes that combine the most informative window and
  mortality settings.

That gives the autonomous search a small but usable lattice around the current
`m2026r1` baseline, which is enough to estimate directionality and rough local
shape instead of only testing one-sided point moves.

### 1. Plan a session

```bash
source .venv/bin/activate
python scripts/analysis/observatory.py search-plan \
  --search-id search-20260315-a \
  --base-revision HEAD \
  --max-pending 3 \
  --max-recommended 5
```

This creates a persisted session with planned specs under
`data/analysis/experiments/search_runs/<search_id>/planned_specs/`.

### 2. Preview the run

```bash
source .venv/bin/activate
python scripts/analysis/observatory.py search-run \
  --search-id search-20260315-a \
  --run-budget 3 \
  --dry-run
```

### 3. Execute the run

```bash
source .venv/bin/activate
python scripts/analysis/observatory.py search-run \
  --search-id search-20260315-a \
  --run-budget 3
```

The controller will:

- assert the live checkout is clean,
- create disposable worktrees from the bare mirror,
- apply any approved deterministic code recipes,
- run the experiment orchestrator inside the worktree,
- harvest benchmark bundles and experiment-log entries back into the canonical
  Observatory stores,
- write patches, logs, and copied specs into the session directory.

### 4. Inspect status and report

```bash
source .venv/bin/activate
python scripts/analysis/observatory.py search-status --search-id search-20260315-a
python scripts/analysis/observatory.py search-report --search-id search-20260315-a
```

## Guardrails

- The live checkout must be clean before `search-run`.
- Protected files such as `config/method_profiles/aliases.yaml` are blocked
  from recipe mutation by policy.
- Recipe mutations are restricted to allowlisted roots.
- Changed Python files are syntax-checked with `py_compile` before the
  experiment runs.
- Worktrees are removed automatically unless `--keep-worktrees` is used.
- Promotion remains a separate human action through SOP-003.

## Current Intended Use

Use autonomous search for:

- config-only exploration at larger scale than the bounded pending queue,
- deterministic code recipes and search-only method clones that have a distinct
  experiment method ID,
- unattended experimentation where you want to come back later to benchmark
  evidence rather than to merged code.

Do not use autonomous search for:

- free-form code generation,
- automatic promotion,
- direct edits in the live checkout,
- recipe definitions that silently change the current champion method in place.
