# SOP-004: Experimental Methodology Branch Workflow

## Document Information

| Field | Value |
|-------|-------|
| SOP ID | 004 |
| Status | Active |
| Created | 2026-03-04 |
| Last Updated | 2026-03-04 |
| Owner | Project Lead |

---

## 1. Purpose

This SOP defines the feature-branch workflow for developing, testing, and deploying experimental methodology changes to the cohort projection pipeline. The goal is to ensure that production configuration on `master` is never modified until changes have passed walk-forward validation and received Tier 3 (human) review approval.

Methodology changes -- new projection method variants, parameter tuning, convergence schedule revisions, GQ corrections, and model revisions -- carry risk of silently degrading projection accuracy if deployed without systematic validation. This SOP enforces a branch-based isolation pattern where experimental work is developed, validated, and approved before it reaches production.

**Origin**: This SOP was created after the CF-001 (College Fix Model Revision, ADR-061) workflow, where production config (`projection_config.yaml`) was inadvertently modified on `master` before walk-forward validation was complete. This SOP prevents that class of error by making the branch boundary explicit and enforceable.

---

## 2. Scope

### In Scope
- Methodology changes that affect projection outputs (new rates, convergence schedules, smoothing parameters, GQ corrections)
- New method variants added to `METHOD_DISPATCH` in `walk_forward_validation.py` or `sensitivity_analysis.py`
- Changes to `projection_config.yaml` that alter projection behavior
- ADRs proposing methodology changes (through their full lifecycle)
- Updates to `docs/methodology.md` that reflect new methodology

### Out of Scope
- Bug fixes to existing production methodology (commit directly to `master`)
- Documentation-only changes (ADR status updates, README edits, SOP updates)
- Export/visualization scripts that do not affect projection outputs
- Infrastructure changes that do not alter projection results (CLI argument parsing, logging, output formatting)
- Backward-compatible code additions where defaults preserve existing behavior (see Section 5 decision table)

---

## 3. Prerequisites

### Required Knowledge
- Familiarity with project structure (see [AGENTS.md](../../../AGENTS.md))
- Understanding of `METHOD_DISPATCH` registry pattern in `walk_forward_validation.py`
- Understanding of the walk-forward validation methodology (see [ADR-057](../adrs/057-rolling-origin-backtests.md))
- Familiarity with `projection_config.yaml` structure and parameter effects

### Required Tools
- Python environment with project dependencies (`uv sync`)
- Git with pre-commit configured
- Access to validation data (walk-forward requires historical population snapshots)

### Required Access
- [ ] Write access to the repository
- [ ] Ability to run `walk_forward_validation.py` and `sensitivity_analysis.py`
- [ ] Ability to request Tier 3 human review from project lead

---

## 4. Procedure

### Phase 1: Branch Setup

**Objective**: Create an isolated feature branch for all experimental methodology work.

**Steps**:

1. **Create feature branch** from `master`:
   ```bash
   git checkout master
   git pull
   git checkout -b feature/{tracker-id}-{description}
   ```
   - Naming convention: `feature/{tracker-id}-{description}` (e.g., `feature/cf-001-college-fix-revision`)
   - The tracker ID should reference a DEVELOPMENT_TRACKER.md entry or ADR number

2. **Verify clean starting point**:
   ```bash
   pytest -k "not test_residual_computation_single_period"
   ```
   - All tests must pass before beginning experimental work

3. **Identify backward-compatible infrastructure changes**:
   - Changes that add new parameters with defaults preserving existing behavior MAY be committed to `master` first (see Section 5 decision table)
   - Example: adding `fraction=1.0` to `subtract_gq_from_populations()` where the default preserves current behavior
   - If infrastructure changes go to `master` first, rebase the feature branch afterward:
     ```bash
     git checkout feature/{tracker-id}-{description}
     git rebase master
     ```

**Outputs**:
- Feature branch created and checked out
- Clean test baseline confirmed

**Checkpoint**: Feature branch exists, all tests pass on branch

---

### Phase 2: Implementation

**Objective**: Implement the experimental methodology on the feature branch with full validation infrastructure.

**Steps**:

1. **Add new method variant to METHOD_DISPATCH**:
   - In `walk_forward_validation.py`, register the new variant in the `METHOD_DISPATCH` dictionary
   - Implement `prepare_{variant}()` and `project_{variant}()` functions following the existing pattern
   - The variant name should be descriptive (e.g., `m2026r1`, `college_fix_v2`)

2. **Add sensitivity analysis support** (if applicable):
   - Register the variant in `sensitivity_analysis.py`
   - Define parameter sweep ranges for the new methodology

3. **Implement core methodology changes**:
   - Modify or create modules in `cohort_projections/` as needed
   - Follow existing code patterns: type hints, Google-style docstrings, modular design
   - All production code changes stay on the feature branch

4. **Update configuration on the feature branch only**:
   - Modify `config/projection_config.yaml` as needed for the new methodology
   - **CRITICAL**: Do NOT commit config changes to `master`. Config changes stay on the feature branch until Phase 4 approval.

5. **Write unit tests**:
   - Add tests for new parameters, functions, and edge cases
   - Tests go on the feature branch alongside the code they test

6. **Update documentation on the feature branch**:
   - Create or update the relevant ADR (Status: Proposed)
   - Update `docs/methodology.md` with the new methodology description
   - Document design decisions and rationale

7. **Run tests on the feature branch**:
   ```bash
   pytest -k "not test_residual_computation_single_period"
   ```

**Outputs**:
- New METHOD_DISPATCH variant with prepare/project functions
- Updated config (on feature branch only)
- Unit tests for new functionality
- ADR and methodology documentation

**Checkpoint**: All tests pass on the feature branch

---

### Phase 3: Validation

**Objective**: Validate the experimental methodology against the current production method using walk-forward backtesting.

**Steps**:

1. **Run walk-forward validation**:
   ```bash
   python scripts/analysis/walk_forward_validation.py \
       --methods sdc_2024 m2026 {new_method} \
       --run-label {tracker-id}-{description}
   ```
   - Always include `sdc_2024` (SDC benchmark) and `m2026` (current production) for comparison
   - Use a descriptive run label matching the tracker ID

2. **Run sensitivity analysis** (if applicable):
   ```bash
   python scripts/analysis/sensitivity_analysis.py \
       --method {new_method} \
       --run-label {tracker-id}-sensitivity
   ```

3. **Compare results across methods**:
   - County-level MAPE (Mean Absolute Percentage Error)
   - State-level APE (Absolute Percentage Error)
   - College county accuracy (Cass, Grand Forks, Ward)
   - Scenario ordering consistency (baseline < high growth for all counties)
   - Directional accuracy (growth/decline matches actuals)

4. **Archive validation outputs**:
   - Validation results are saved with the run label for reproducibility
   - Commit validation outputs to the feature branch (not `master`)
   - Include summary tables in the ADR or a linked validation report

5. **Document validation findings**:
   - Update the ADR with validation results
   - Note any counties or scenarios where the new method underperforms
   - Provide a clear recommendation (adopt, iterate, or reject)

**Outputs**:
- Walk-forward validation results with run label
- Sensitivity analysis results (if applicable)
- Comparative accuracy metrics across methods
- Updated ADR with validation findings and recommendation

**Checkpoint**: Validation results archived and documented; recommendation stated

---

### Phase 4: Tier 3 Review and Merge

**Objective**: Obtain human approval and merge validated methodology into production.

**Steps**:

1. **Present validation results for Tier 3 review**:
   - Share validation summary with the project lead
   - Include: method description, accuracy comparison, trade-offs, recommendation
   - Provide access to full validation outputs for detailed review

2. **Await decision**:

   **If approved**:
   1. Update ADR status: Proposed --> Accepted
   2. Add Implementation Results section to the ADR
   3. Merge feature branch to `master`:
      ```bash
      git checkout master
      git merge feature/{tracker-id}-{description}
      ```
      This brings config changes into production as part of the merge.
   4. Re-run the production projection pipeline:
      ```bash
      python scripts/projections/run_all_projections.py
      ```
   5. Update `DEVELOPMENT_TRACKER.md` to reflect completion
   6. Update `docs/methodology.md` if not already current
   7. Sync:
      ```bash
      ./scripts/bisync.sh
      ```

   **If rejected**:
   1. Iterate on the feature branch (return to Phase 2) if feedback is actionable
   2. Or close: update ADR status to Rejected with rationale, delete feature branch
      ```bash
      git branch -d feature/{tracker-id}-{description}
      ```

3. **Post-merge verification**:
   ```bash
   pytest -k "not test_residual_computation_single_period"
   ```
   - All tests must pass on `master` after merge

**Outputs**:
- Merged feature branch (if approved)
- Updated ADR with final status
- Updated DEVELOPMENT_TRACKER.md
- Fresh production projection outputs

**Checkpoint**: `master` has validated methodology, all tests pass, pipeline re-run complete

---

## 5. Branch Decision Table

Use this table to determine whether a change belongs on a feature branch or can go directly to `master`.

| Change Type | Branch | Example |
|-------------|--------|---------|
| New METHOD_DISPATCH variant | Feature | `m2026r1` variant functions |
| Config parameter changes affecting outputs | Feature | `college_age.age_groups` extension |
| New convergence schedules or rates | Feature | County-specific dampening factors |
| New smoothing parameters | Feature | Migration smoothing window changes |
| New backward-compatible parameters (default preserves behavior) | Master OK | `fraction=1.0` on `subtract_gq_from_populations()` |
| METHOD_DISPATCH infrastructure and documentation | Master OK | Registry documentation, CLI argument parsing |
| Unit tests for new functionality | Feature | GQ fraction tests, new variant tests |
| ADR (Proposed status) | Feature | ADR-061 College Fix proposal |
| ADR status update to Accepted/Rejected | Master OK | Status field change after Tier 3 review |
| Bug fixes to existing methodology | Master | ADR-049 smoothing propagation fix |
| Export/visualization scripts | Master OK | New report builder scripts |
| `docs/methodology.md` for new methodology | Feature | New method description |
| `docs/methodology.md` corrections/clarifications | Master OK | Fixing a typo or clarifying existing text |

**Rule of thumb**: If the change could alter the numbers produced by `run_all_projections.py`, it belongs on a feature branch.

---

## 6. Anti-Patterns

The following practices violate this SOP and must be avoided:

| Anti-Pattern | Why It Is Dangerous | Correct Approach |
|-------------|---------------------|------------------|
| Changing `projection_config.yaml` on `master` before validation | Production projections silently change; no comparative baseline | Config changes stay on feature branch until Phase 4 merge |
| Committing experimental method parameters to `master` | Unvalidated methodology enters production | All experimental parameters on feature branch |
| Running `run_all_projections.py` with unvalidated methodology | Produces outputs that may be shared or published prematurely | Only run production pipeline after Tier 3 approval and merge |
| Merging feature branch without walk-forward results | No evidence the methodology improves accuracy | Phase 3 validation is a prerequisite for Phase 4 |
| Skipping Tier 3 human review | Removes the final quality gate for methodology changes | All methodology changes require project lead sign-off |

---

## 7. Quality Gates

| Gate | Criteria | Responsible |
|------|----------|-------------|
| Branch created | Feature branch follows naming convention | Developer / AI Agent |
| Tests pass on branch | `pytest` returns 0 on feature branch | Developer / AI Agent |
| Walk-forward validation complete | Results archived with run label; comparative metrics documented | Developer / AI Agent |
| Tier 3 review approved | Project lead has reviewed validation results and approved | Project Lead (human) |
| Post-merge tests pass | `pytest` returns 0 on `master` after merge | Developer / AI Agent |
| Production pipeline re-run | `run_all_projections.py` completes with new methodology | Developer / AI Agent |

---

## 8. Troubleshooting

### Feature branch has diverged significantly from master

**Symptom**: Merge conflicts when attempting Phase 4 merge.

**Resolution**: Rebase the feature branch onto `master` periodically during development:
```bash
git checkout feature/{tracker-id}-{description}
git rebase master
```
Resolve conflicts, then re-run tests to confirm nothing broke.

### Walk-forward validation shows mixed results

**Symptom**: New method is better for some counties but worse for others.

**Resolution**: Document the trade-offs explicitly in the ADR. The Tier 3 reviewer needs to see where the method improves and where it regresses. Consider whether the regression is in high-priority counties (e.g., the four metro counties) or low-population counties where small absolute errors produce large percentage errors.

### Backward-compatible change accidentally alters outputs

**Symptom**: A change committed to `master` as "backward-compatible" turns out to change projection results.

**Resolution**: Revert the change on `master` immediately. Move it to a feature branch. The test suite should catch this if projection output tests exist, but verify by running `run_all_projections.py` and comparing outputs to the previous run.

### Multiple experimental branches need the same infrastructure

**Symptom**: Two feature branches both need the same new parameter or METHOD_DISPATCH infrastructure.

**Resolution**: Commit the shared infrastructure to `master` first (if it is genuinely backward-compatible with appropriate defaults). Both feature branches can then rebase onto `master` to pick up the shared code.

---

## 9. Related Documentation

- [AGENTS.md](../../../AGENTS.md) -- AI agent governance and quality standards
- [ADR-061](../adrs/061-college-fix-model-revision.md) -- First methodology change developed under this SOP
- [ADR-057](../adrs/057-rolling-origin-backtests.md) -- Walk-forward validation methodology
- [DEVELOPMENT_TRACKER.md](../../../DEVELOPMENT_TRACKER.md) -- Current project status and tracker IDs
- [SOP-001](./SOP-001-external-ai-analysis-integration.md) -- External AI Analysis Integration
- [SOP-002](./SOP-002-data-processing-documentation.md) -- Data Processing Documentation
- `scripts/analysis/walk_forward_validation.py` -- Walk-forward validation with METHOD_DISPATCH
- `scripts/analysis/sensitivity_analysis.py` -- Sensitivity analysis framework
- `config/projection_config.yaml` -- Production projection configuration

---

## 10. Revision History

| Date | Version | Author | Changes |
|------|---------|--------|---------|
| 2026-03-04 | 1.0 | Claude Opus 4.6 / N. Haarstad | Initial version derived from CF-001 workflow |

---

*SOP Version: 1.0*
