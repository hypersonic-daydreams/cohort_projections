# Projection Observatory Deep Search

Deterministic operating guide for running the Projection Observatory as a
guided deep-search process in isolated git worktrees.

This guide extends the bounded queue workflow in
[Projection Observatory Search-Loop Guide](./observatory-search-loop.md).
Use it when you want the system to plan, execute, replan, and summarize many
candidates with minimal supervision while guaranteeing that the live checkout
and production aliases are not mutated by the search run itself.

`search-auto` still exists, but it is now a backward-compatible alias for the
same deep-search controller. Treat `deep-search` as the canonical command name.

## Design Boundary

- Deep search is deterministic and policy-driven.
- The CPU budget is the main user control. Parallel run count and workers per
  run are derived from that shared allocator.
- Deterministic policy and ranking remain authoritative.
- Optional AI synthesis is advisory only. It may summarize or critique the
  evidence, but it cannot pick winners, set decision states, or promote a
  method.
- The controller runs candidates in disposable worktrees created from a bare
  mirror, not in the live checkout.
- The controller can write append-only artifacts and reports, including
  journals, frontiers, and briefs.
- The controller cannot promote a champion, update aliases, or merge branches.
- Search packs define the objective order, seeds, guardrails, and optional code
  mutators for one search domain. `cf001` is the first pack.
- Search-only recipe candidates can clone a registered base method into a
  unique sandbox-only method ID at benchmark time, so recipe experiments stay
  separate from production aliases and the live checkout.
- If AI-generated narrative is carried into a formal decision record,
  methodology proposal, or review memo, treat that narrative as external AI
  analysis and follow
  [SOP-001](../governance/sops/SOP-001-external-ai-analysis-integration.md).

## Files

- CLI:
  `python scripts/analysis/observatory.py deep-search|search-plan|search-run|search-status|search-report|search-auto`
- Core control plane:
  `cohort_projections/analysis/observatory/deep_search.py`
- Optional synthesis layer:
  `cohort_projections/analysis/observatory/ai_synthesis.py`
- Policy: `config/observatory_search_policy.yaml`
- Search packs: `config/observatory_search_packs/*.yaml`
- Recipe catalog: `config/observatory_recipes.yaml`
- Session state: `data/analysis/experiments/search_runs/<search_id>/`
- Runtime mirror/worktrees: `data/analysis/observatory_runtime/`

## Safe Loop

### Fastest Hands-Off Launch

If you want the most straightforward unattended run, use `deep-search`:

```bash
source .venv/bin/activate
python scripts/analysis/observatory.py deep-search \
  --search-id search-20260315-a \
  --cpu-budget 12 \
  --search-pack cf001
```

This will:

- create or overwrite the deep-search session,
- resolve a search pack, CPU budget, time budget, and hidden batch defaults,
- plan pending variants, recommendations, and enabled recipe candidates,
- run repeated batches, replan adaptively, and stop on exhaustion, plateau,
  runtime cap, candidate cap, or operational-failure budget,
- quarantine orchestration failures as `operational_blocker` artifacts instead
  of treating them as model evidence,
- refresh the Observatory cache,
- write per-session artifacts including `session.yaml`,
  `search_journal.jsonl`, `frontier.csv`, `candidate_summary.csv/json`,
  `deep_search_brief.json/.md`, and optional `ai_brief.json/.md`,
- write a full Observatory HTML report into the same session directory unless
  `--skip-observatory-report` is used.

If you prefer to operate from the dashboard, open the Projection Observatory
`Command Center` tab. The primary path is now:

1. choose `CPU cores to use`,
2. click `Begin Deep Search`.

Live progress and best candidates appear in the `Search Progress` card, which
auto-refreshes every 5 seconds.

For fine-grained control, expand `Expert Controls`. It can:

- preview a `search-plan`,
- preview the next `search-run --dry-run` batch,
- launch `deep-search` in the background with custom parameters,
- stop dashboard-launched `deep-search` or `search-auto` processes with
  `SIGTERM`,
- show session tables, candidate details, search reports, and log tails for
  the selected search session.

The current recipe/search-pack surface is intentionally broader than the
original three-point pilot. The first `cf001` pack and recipe lattice cover:

- a convergence-window family (`recent/medium` = `2/2`, `2/3`, `3/3`),
- a mortality-improvement family (`0.003`, `0.004`, `0.006`),
- three interaction recipes that combine the most informative window and
  mortality settings.

That gives deep search a small but usable lattice around the current
`m2026r1` baseline, which is enough to estimate directionality and rough local
shape instead of only testing one-sided point moves.

### 1. Plan a session

```bash
source .venv/bin/activate
python scripts/analysis/observatory.py search-plan \
  --search-id search-20260315-a \
  --base-revision HEAD \
  --search-pack cf001 \
  --cpu-budget 12 \
  --time-budget-hours 6 \
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
- validate changed files against allowlisted roots, protected-path rules, patch
  size limits, Python compilation, and any targeted tests declared by the pack
  or candidate,
- run the experiment orchestrator inside the worktree,
- harvest benchmark bundles and experiment-log entries back into the canonical
  Observatory stores,
- write patches, logs, copied specs, and validation reports into the session
  directory.

### 4. Inspect status and report

```bash
source .venv/bin/activate
python scripts/analysis/observatory.py search-status --search-id search-20260315-a
python scripts/analysis/observatory.py search-report --search-id search-20260315-a
```

For the end-to-end command, `deep-search` and `search-auto` call the same
controller:

```bash
source .venv/bin/activate
python scripts/analysis/observatory.py deep-search --cpu-budget 12 --search-pack cf001
python scripts/analysis/observatory.py search-auto --cpu-budget 12 --search-pack cf001
```

## Session Artifacts

Every deep-search session writes a directory under
`data/analysis/experiments/search_runs/<search_id>/` with:

- `session.yaml`: canonical session metadata and candidate state
- `search_journal.jsonl`: append-only event log
- `planned_specs/`: generated experiment specs
- `completed_specs/`: copied completed specs
- `candidate_summary.csv` / `candidate_summary.json`: flattened candidate table
- `frontier.csv`: best candidates ordered by pack objective order
- `deep_search_brief.json` / `deep_search_brief.md`: deterministic review brief
- `ai_brief.json` / `ai_brief.md`: optional advisory AI summary when enabled
- `search_report.md`: broader candidate/session report
- `patches/`, `logs/`, `profiles/`, `quarantine/`, `code_candidates/`: sandbox
  artifacts for code-backed or blocked candidates

## Guardrails

- The live checkout must be clean before `search-run`.
- Protected files such as `config/method_profiles/aliases.yaml` are blocked
  from recipe mutation by policy.
- Recipe mutations are restricted to allowlisted roots.
- Changed Python files are syntax-checked with `py_compile` before the
  experiment runs.
- Code-backed candidates are further limited by deep-search policy
  (`max_patch_files`, `max_patch_lines`) and may be blocked before benchmarking
  if targeted tests fail.
- Benchmark bundles missing `manifest.json`, `summary_scorecard.csv`,
  `runtime_summary.json`, or reproducibility/runtime flags are quarantined as
  `operational_blocker`.
- Optional AI synthesis is suppressed automatically when the deterministic claim
  checker detects contradictions against run IDs, metric keys, or decision
  state.
- Worktrees are removed automatically unless `--keep-worktrees` is used.
- Promotion remains a separate human action through SOP-003.

## Current Intended Use

Use deep search for:

- config-only exploration at larger scale than the bounded pending queue,
- deterministic code recipes and search-only method clones that have a distinct
  experiment method ID,
- unattended experimentation where you want to come back later to benchmark
  evidence rather than to merged code,
- dashboard-driven guided experimentation where the user mostly verifies the
  Deep Search Brief and drill-down tabs.

Do not use deep search for:

- free-form code generation,
- automatic promotion,
- direct edits in the live checkout,
- recipe definitions that silently change the current champion method in place.
