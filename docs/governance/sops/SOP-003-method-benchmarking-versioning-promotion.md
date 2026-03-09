# SOP-003: Method Benchmarking, Versioning, and Promotion

## Document Information

| Field | Value |
|-------|-------|
| SOP ID | 003 |
| Status | Active |
| Created | 2026-03-09 |
| Last Updated | 2026-03-09 |
| Owner | Project Lead |

---

## 1. Purpose

This SOP defines the standard workflow for iterative improvement of the North Dakota population projection methods while preserving:

- immutable historical method definitions,
- append-only benchmark evidence,
- reversible production promotion decisions, and
- cross-session auditability.

This SOP is both:

1. the operating procedure to follow once the workflow exists, and
2. the implementation specification for the automation required to support it.

The goal is to ensure the project can answer, at any later date:

- what method was used,
- what changed from the previous method,
- what evidence justified the change,
- whether the change improved benchmark performance, and
- how to revert to the prior champion if needed.

---

## 2. Scope

### In Scope

- Versioning projection methods whose behavior may change results
- Versioning benchmarked configuration profiles
- Defining champion vs challenger evaluation workflow
- Persisting benchmark manifests, scorecards, and comparison artifacts
- Promotion, rejection, supersession, and deprecation of methods
- Automation requirements for benchmark orchestration and run logging
- Migration of existing methods (`sdc_2024`, `m2026`, `m2026r1`) into the versioned workflow

### Out of Scope

- Replacing ADRs for methodology decisions
- Replacing `DEVELOPMENT_TRACKER.md` as the active-status source of truth
- CI/CD deployment outside this repository
- Rewriting existing backtest, sensitivity, uncertainty, or QC logic
- Deleting historical methods or historical benchmark outputs

---

## 3. Prerequisites

### Required Access

- [ ] Write access to the repository
- [ ] Ability to run commands in the project virtual environment
- [ ] Ability to create benchmark artifacts under `data/analysis/`

### Required Knowledge

- Familiarity with [AGENTS.md](../../AGENTS.md)
- Familiarity with ADR workflow in [docs/governance/adrs/README.md](../adrs/README.md)
- Familiarity with testing guidance in [docs/guides/testing-workflow.md](../../guides/testing-workflow.md)
- Understanding that methodology changes affecting results remain Tier 3 decisions requiring human approval before production promotion

### Required Tools

- Project virtual environment (`source .venv/bin/activate`)
- Existing validation scripts:
  - `scripts/analysis/walk_forward_validation.py`
  - `scripts/analysis/sensitivity_analysis.py`
  - `scripts/analysis/uncertainty_analysis.py`
  - `scripts/analysis/qc_diagnostics.py`
- Reproducibility helper: `cohort_projections/utils/reproducibility.py`

---

## 4. Non-Negotiable Rules

1. **No silent method mutation.**
   If a code or config change can change benchmark results, it must receive a new immutable `method_id` and, if applicable, a new immutable `config_id`.

2. **No benchmark overwrite.**
   Benchmark outputs are append-only. A rerun creates a new `run_id`; it does not replace a prior run directory.

3. **No production promotion without comparison to the current champion.**
   Every candidate method must be benchmarked head-to-head against the frozen champion.

4. **No benchmark without a frozen config snapshot.**
   Every run must persist the fully resolved config used during execution, even if the source config was assembled from multiple files.

5. **No mutable alias as the only provenance record.**
   Aliases such as `champion` or `current_candidate` are convenience pointers only. Audit trails must point to immutable IDs.

6. **No deletion of prior methods for cleanliness.**
   Old methods may be `deprecated` or `superseded`, but remain available for reference and reruns unless a separate archival decision is recorded.

7. **No benchmark script without reproducibility logging.**
   All new automation scripts defined by this SOP must use `log_execution(...)`.

---

## 5. Versioning Model

### 5.1 Artifact Types

| Artifact Type | Purpose | Mutability | Example |
|---------------|---------|------------|---------|
| `method_id` | Immutable algorithm identity | Immutable | `m2026`, `m2026r1` |
| `config_id` | Immutable benchmarked config identity | Immutable | `cfg-20260309-college-fix-v1` |
| `run_id` | Immutable execution identity | Immutable | `br-20260309-154500-m2026r1-a1b2c3d` |
| `alias` | Human-friendly pointer | Mutable | `county_champion`, `place_champion` |
| `decision_id` | Human review record identity | Immutable | `2026-03-09-m2026r1-vs-m2026` |

### 5.2 Required Naming Conventions

#### Method IDs

- Lowercase letters, digits, and underscores only
- Must be stable once published in the registry
- Must not be reused for changed behavior

Recommended patterns:

- Baseline lineage: `m2026`
- Revision lineage: `m2026r1`, `m2026r2`
- Focused branch: `m2026_fertility_v2`
- Place variant: `place_bii_ro1`

#### Config IDs

Format:

```text
cfg-YYYYMMDD-short-slug-vN
```

Example:

```text
cfg-20260309-college-fix-v1
```

#### Run IDs

Format:

```text
br-YYYYMMDD-HHMMSS-primarymethod-shortgit
```

Example:

```text
br-20260309-154500-m2026r1-a1b2c3d
```

### 5.3 Method Lifecycle States

| State | Meaning |
|-------|---------|
| `candidate` | Implemented and benchmarkable, not approved for production |
| `accepted` | Evidence supports the method as a viable option |
| `champion` | Current production method selected for official outputs |
| `superseded` | Former champion or accepted method replaced by a newer method |
| `rejected` | Benchmarked and not selected |
| `deprecated` | Retained for reference; not recommended for new comparison work |

Only one method per scope should hold the `champion` alias at a time.

---

## 6. Required Repository Layout

The following locations are required to implement this SOP.

| Path | Purpose | Status |
|------|---------|--------|
| `config/method_profiles/` | Immutable method/config profiles used for benchmarked methods | To implement |
| `config/method_profiles/aliases.yaml` | Mutable alias map from role to immutable method/config IDs | To implement |
| `data/analysis/benchmark_history/` | Append-only benchmark run artifacts | To implement |
| `data/analysis/benchmark_history/index.csv` | Append-only benchmark run index | To implement |
| `data/analysis/benchmark_history/latest/` | Optional convenience pointers or symlinks to latest reports | Optional |
| `docs/reviews/benchmark_decisions/` | Human-readable decision records comparing challenger vs champion | To implement |
| `scripts/analysis/run_benchmark_suite.py` | Canonical automation entrypoint for benchmark runs | To implement |
| `scripts/analysis/compare_benchmark_runs.py` | Canonical comparison/diff reporting tool | To implement |
| `scripts/analysis/promote_method.py` | Controlled alias update tool for champion promotion | To implement |
| `tests/test_analysis/test_benchmarking.py` | Automation and schema tests for the new workflow | To implement |

### 6.1 Method Profile Contract

Each immutable profile file under `config/method_profiles/` must contain:

- `method_id`
- `config_id`
- `scope` (`county`, `place`, `statewide_validation`, etc.)
- `status`
- `created_date`
- `created_from`
- `description`
- `code_refs`
- `adr_refs`
- `resolved_config`
- `notes`

The `resolved_config` block must be a full benchmark-ready snapshot, not a partial diff.

### 6.2 Alias Contract

`aliases.yaml` must hold only mutable pointers. Example:

```yaml
county_champion:
  method_id: m2026
  config_id: cfg-20260223-baseline-v1

county_candidate:
  method_id: m2026r1
  config_id: cfg-20260309-college-fix-v1
```

If aliases are changed, the prior mapping must be recorded in the promotion decision record.

---

## 7. Procedure

### Phase 1: Register the Experiment

**Objective**: Define a challenger method without mutating the current champion.

**Inputs**:

- Existing champion method/config
- Hypothesis for improvement
- ADR or review motivating the change

**Steps**:

1. **Write the hypothesis**
   - State the exact problem being addressed
   - State the expected improvement
   - State the expected regression risk

2. **Assign immutable IDs**
   - Create a new `method_id` if behavior may change results
   - Create a new `config_id` if parameters differ from prior benchmarked config

3. **Register the method profile**
   - Add a full immutable config snapshot under `config/method_profiles/`
   - Set status to `candidate`
   - Record ADRs, code references, and source lineage

4. **Link the work**
   - Cross-reference the experiment in the relevant ADR or review document
   - If the work is active, note it in `DEVELOPMENT_TRACKER.md`

**Outputs**:

- New immutable method profile
- Registered candidate method/config
- Written hypothesis in ADR or review context

**Checkpoint**: The challenger exists under a new immutable ID before benchmark execution begins.

---

### Phase 2: Implement the Challenger

**Objective**: Add new behavior without altering historical behavior of prior methods.

**Inputs**:

- Registered `method_id`
- Registered `config_id`

**Steps**:

1. **Add code under a new method path or registry entry**
   - Extend `METHOD_DISPATCH` or equivalent registry
   - Do not redefine the behavior of existing method IDs

2. **Preserve old code paths**
   - Shared helpers may be refactored if behavior is preserved
   - If a refactor changes output behavior, it still requires a new `method_id`

3. **Add tests**
   - Validate registry lookup
   - Validate reproducibility of config resolution
   - Validate old and new methods remain independently runnable

**Outputs**:

- New code path or registry entry
- Tests covering method registration and execution

**Checkpoint**: Champion and challenger can both run in the same codebase without conditional hand-editing.

---

### Phase 3: Run the Canonical Benchmark Suite

**Objective**: Produce an append-only, reproducible evidence bundle comparing champion and challenger.

**Inputs**:

- Champion method/config
- Challenger method/config
- Canonical benchmark contract

**Canonical command interface (required implementation)**:

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

**Required benchmark battery**:

1. Walk-forward validation
2. Sensitivity analysis
3. Uncertainty analysis
4. QC diagnostics
5. Relevant invariant tests or integration checks

**Steps**:

1. **Resolve methods and configs**
   - Load immutable method profiles
   - Materialize a fully resolved config snapshot for each method

2. **Capture provenance**
   - Git commit hash
   - Dirty worktree flag
   - Python version
   - Command line
   - Script versions
   - Input artifact paths and hashes

3. **Run the benchmark battery**
   - Use the same benchmark contract for champion and challenger
   - Do not change metrics or horizons mid-run

4. **Write append-only artifacts**
   - Create a new `run_id`
   - Create a new run directory
   - Write manifest, scorecards, and detailed outputs

5. **Append the index**
   - Add one row per method per run to `index.csv`

**Outputs**:

- Append-only benchmark run directory
- Run manifest
- Summary scorecard
- Updated benchmark index

**Checkpoint**: A full evidence bundle exists for both champion and challenger, with no overwritten prior run.

---

### Phase 4: Review and Decide

**Objective**: Convert benchmark evidence into a tracked decision.

**Inputs**:

- Benchmark run artifacts
- Champion vs challenger diff
- Relevant ADRs and reviews

**Steps**:

1. **Generate the comparison**
   - Run the required comparison tool:
   ```bash
   source .venv/bin/activate
   python scripts/analysis/compare_benchmark_runs.py \
     --run-id br-20260309-154500-m2026r1-a1b2c3d \
     --output-dir docs/reviews/benchmark_decisions/
   ```

2. **Create a decision record**
   - Use the template in `docs/governance/sops/templates/benchmark-decision-record.md`
   - Record material improvements and regressions
   - Record unresolved questions

3. **Assign one decision**
   - `promote`
   - `accept_but_do_not_promote`
   - `retain_champion`
   - `reject`
   - `needs_more_segmentation`

4. **Record state changes**
   - Update method lifecycle status
   - If promotion is approved, mark the old champion as `superseded`

**Outputs**:

- Decision record under `docs/reviews/benchmark_decisions/`
- Updated method statuses

**Checkpoint**: A human-readable, dated record explains the benchmark outcome.

---

### Phase 5: Promote or Revert

**Objective**: Change production pointers without rewriting historical methods.

**Inputs**:

- Approved decision record
- Current alias map

**Steps**:

1. **Promote by alias update only**
   - Do not rename the challenger to the champion's historical `method_id`
   - Update `aliases.yaml` so the champion alias points to the approved method/config pair

2. **Record the transition**
   - Log old alias target and new alias target
   - Link to the decision record and benchmark `run_id`

3. **Support reversion**
   - Reversion must be possible by changing the alias back to the previous immutable pair

**Required command interface**:

```bash
source .venv/bin/activate
python scripts/analysis/promote_method.py \
  --scope county \
  --alias county_champion \
  --method-id m2026r1 \
  --config-id cfg-20260309-college-fix-v1 \
  --decision-id 2026-03-09-m2026r1-vs-m2026
```

**Outputs**:

- Updated alias pointer
- Promotion audit record

**Checkpoint**: Production selection changed without mutating old methods or deleting prior champion references.

---

### Phase 6: Maintain the History

**Objective**: Keep the benchmark archive auditable over multiple sessions.

**Steps**:

1. Never edit prior run manifests except to correct non-semantic metadata errors with explicit annotation.
2. Never remove prior benchmark rows from `index.csv`.
3. Do not reuse deprecated or rejected method IDs for new work.
4. Keep `latest/` pointers optional and disposable; treat dated run directories as canonical.
5. When benchmark metrics change structurally, increment a `benchmark_contract_version` and document the change.

**Checkpoint**: The archive remains longitudinally comparable across sessions.

---

## 8. Required Artifacts and Schemas

### 8.1 Benchmark Run Directory

Each benchmark run must create:

```text
data/analysis/benchmark_history/<run_id>/
├── manifest.json
├── resolved_configs/
│   ├── <method_id>__<config_id>.yaml
├── summary_scorecard.json
├── summary_scorecard.csv
├── state_metrics.csv
├── county_metrics.csv
├── sensitivity_summary.csv
├── uncertainty_summary.csv
├── qc_summary.json
├── comparison_to_champion.json
└── execution_log.json
```

Additional detailed files may be included, but the files above are required.

### 8.2 `manifest.json` Required Fields

| Field | Description |
|-------|-------------|
| `run_id` | Immutable benchmark run identifier |
| `benchmark_contract_version` | Version of the metric/output contract |
| `created_at_utc` | ISO timestamp |
| `git_commit` | Commit hash |
| `git_dirty` | Boolean dirty-worktree flag |
| `command` | Full CLI invocation |
| `scope` | Evaluation scope |
| `methods` | Array of method/config pairs included in the run |
| `champion_method_id` | Champion method used for comparison |
| `challenger_method_ids` | Challenger methods included in the run |
| `input_artifacts` | Paths and hashes of key inputs |
| `output_artifacts` | Paths of generated outputs |
| `script_versions` | Scripts/modules used to produce the run |
| `execution_log_run_id` | `log_execution` reference if available |

### 8.3 `summary_scorecard.csv` Required Columns

At minimum:

| Column | Description |
|--------|-------------|
| `run_id` | Benchmark run identifier |
| `method_id` | Immutable method identifier |
| `config_id` | Immutable config identifier |
| `scope` | Evaluation scope |
| `status_at_run` | Method status when run |
| `state_ape_recent_short` | Recent-origin short-horizon state APE |
| `state_ape_recent_medium` | Recent-origin medium-horizon state APE |
| `state_signed_bias_recent` | Recent-origin signed bias |
| `county_mape_overall` | Overall county MAPE |
| `county_mape_urban_college` | Urban/college county MAPE |
| `county_mape_rural` | Rural county MAPE |
| `county_mape_bakken` | Bakken county MAPE |
| `sentinel_cass_mape` | Sentinel county metric |
| `sentinel_grand_forks_mape` | Sentinel county metric |
| `sentinel_ward_mape` | Sentinel county metric |
| `sentinel_burleigh_mape` | Sentinel county metric |
| `sentinel_williams_mape` | Sentinel county metric |
| `negative_population_violations` | Hard constraint count |
| `scenario_order_violations` | Hard constraint count |
| `aggregation_violations` | Hard constraint count |
| `sensitivity_instability_flag` | Whether the method is materially fragile |

If the metric contract evolves, add columns; do not rename or remove prior columns without a schema version bump.

### 8.4 `index.csv` Required Columns

`index.csv` is the append-only longitudinal ledger. One row per method per run.

| Column | Description |
|--------|-------------|
| `run_id` | Benchmark run identifier |
| `run_date` | Run date |
| `method_id` | Method identifier |
| `config_id` | Config identifier |
| `scope` | Evaluation scope |
| `benchmark_label` | Human-friendly label |
| `benchmark_contract_version` | Metric contract version |
| `git_commit` | Commit hash |
| `decision_id` | Decision record if later assigned |
| `decision_status` | `pending`, `promoted`, `retained`, `rejected`, etc. |
| `is_champion_at_run` | Boolean |
| `summary_scorecard_path` | Path to scorecard |
| `manifest_path` | Path to manifest |

### 8.5 Decision Record

Each head-to-head review must create:

```text
docs/reviews/benchmark_decisions/YYYY-MM-DD-<challenger>-vs-<champion>.md
```

Use the template in:

- `docs/governance/sops/templates/benchmark-decision-record.md`

---

## 9. Quality Gates

| Gate | Criteria | Responsible |
|------|----------|-------------|
| Gate 1 | Champion and challenger have distinct immutable IDs | Implementer |
| Gate 2 | Relevant tests, Ruff, and MyPy pass for changed code | Implementer |
| Gate 3 | Benchmark run includes manifest, config snapshots, and scorecard | Automation |
| Gate 4 | Hard constraints show zero critical violations | Automation |
| Gate 5 | Challenger is compared directly to current champion | Automation |
| Gate 6 | Decision record explains improvements, regressions, and verdict | Reviewer |
| Gate 7 | Production promotion is approved by human reviewer when results may change official outputs | Project Lead |

---

## 10. Required Automation Components

The following implementation items are required to fully operationalize this SOP.

| Priority | Component | Required Behavior |
|----------|-----------|-------------------|
| P0 | Method profile loader | Read immutable profiles and alias pointers |
| P0 | Benchmark suite runner | Orchestrate walk-forward, sensitivity, uncertainty, and QC in one command |
| P0 | Manifest writer | Capture config snapshots, hashes, commit, command, and outputs |
| P0 | Benchmark index updater | Append rows without rewriting history |
| P0 | Comparison report builder | Produce challenger-vs-champion diff artifacts |
| P0 | Promotion tool | Update alias pointers only, with audit trail |
| P1 | Schema validation tests | Validate manifest and scorecard contract |
| P1 | History smoke tests | Confirm old run directories remain readable |
| P1 | Latest-pointer updater | Refresh convenience `latest/` links without affecting canonical history |
| P2 | HTML dashboard | Read `index.csv` and summarize longitudinal performance |

### 10.1 Required Script Responsibilities

#### `scripts/analysis/run_benchmark_suite.py`

Must:

- accept explicit champion/challenger method and config IDs,
- resolve immutable profiles,
- run the canonical benchmark battery,
- write the full run directory,
- append `index.csv`,
- use `log_execution(...)`.

#### `scripts/analysis/compare_benchmark_runs.py`

Must:

- accept a `run_id`,
- compare challenger against champion within that run,
- emit machine-readable diff plus human-readable summary.

#### `scripts/analysis/promote_method.py`

Must:

- validate an approved `decision_id`,
- update alias pointers only,
- write a promotion audit entry,
- refuse promotion if the target method lacks benchmark evidence.

---

## 11. Decision Rules

The default promotion logic is:

1. Prefer lower recent-origin state APE.
2. Prefer less negative recent-origin signed bias.
3. Prefer improved urban/college county performance.
4. Reject methods that improve one headline metric by creating material hard-constraint failures.
5. Reject methods that improve a narrow slice while materially worsening overall county performance unless the decision record explicitly approves the tradeoff.

### 11.1 Required Verdict Labels

| Verdict | Meaning |
|---------|---------|
| `promote` | Challenger becomes champion alias target |
| `accept_but_do_not_promote` | Challenger is valid and retained for future comparison |
| `retain_champion` | Evidence insufficient to replace champion |
| `reject` | Challenger should not be used further in its current form |
| `needs_more_segmentation` | Challenger shows mixed results and needs narrower testing |

---

## 12. Migration Plan for Existing Methods

The repository currently contains benchmark-relevant methods including `sdc_2024`, `m2026`, and `m2026r1`.

To onboard them into this SOP:

1. Create immutable profiles for each current method/config pair.
2. Register `m2026` as the current county champion unless replaced by later approved decision.
3. Register `m2026r1` as a `candidate` until its head-to-head benchmark decision is recorded.
4. Backfill at least one benchmark history record for the current champion and one for the current challenger if available.
5. Do not rename existing method IDs during migration.

This migration is the minimum needed before any future challenger promotion should occur.

---

## 13. Troubleshooting

### Issue: A helper refactor changes old benchmark outputs

**Symptom**: `m2026` reruns no longer match prior benchmark evidence after unrelated cleanup.

**Cause**: Shared helper behavior changed without method re-baselining.

**Resolution**:

- Treat the changed behavior as a new method version.
- Restore old behavior for the historical method if feasible.
- If exact restoration is not feasible, document the break explicitly and create a migration note before further benchmarking.

### Issue: A benchmark run cannot be reproduced because the source config changed

**Symptom**: `projection_config.yaml` no longer matches prior results.

**Cause**: The run used a mutable config file without frozen snapshot capture.

**Resolution**:

- Use the resolved config snapshot stored in the benchmark run directory.
- Treat missing snapshots as a process failure and do not use the run as canonical evidence.

### Issue: `latest/` reports disagree with historical run artifacts

**Symptom**: A convenience report conflicts with a dated run directory.

**Cause**: `latest/` pointer was refreshed after a newer run.

**Resolution**:

- Treat dated run directories and `index.csv` as canonical.
- Treat `latest/` as convenience output only.

---

## 14. Related Documentation

- [AGENTS.md](../../AGENTS.md) - Project operating constraints
- [docs/governance/adrs/005-configuration-management-strategy.md](../adrs/005-configuration-management-strategy.md) - Config governance context
- [docs/governance/adrs/009-logging-error-handling-strategy.md](../adrs/009-logging-error-handling-strategy.md) - Logging and error handling
- [docs/governance/adrs/056-testing-strategy-maturation.md](../adrs/056-testing-strategy-maturation.md) - Testing expectations
- [docs/governance/adrs/057-rolling-origin-backtests.md](../adrs/057-rolling-origin-backtests.md) - Rolling-origin benchmark precedent
- [docs/governance/adrs/061-college-fix-model-revision.md](../adrs/061-college-fix-model-revision.md) - Current challenger-method context
- [docs/reviews/2026-03-04-projection-accuracy-analysis.md](../../reviews/2026-03-04-projection-accuracy-analysis.md) - Current improvement priorities
- [docs/guides/testing-workflow.md](../../guides/testing-workflow.md) - Test execution guidance

---

## 15. Revision History

| Date | Version | Author | Changes |
|------|---------|--------|---------|
| 2026-03-09 | 1.0 | Codex | Initial version |

---

*Template Basis: SOP template v1.0*
