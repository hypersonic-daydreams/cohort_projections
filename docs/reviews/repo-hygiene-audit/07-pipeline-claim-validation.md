# Pipeline Claim Validation: Fragmented and Incomplete Pipeline

| Attribute | Value |
|-----------|-------|
| **Date** | 2026-02-26 |
| **Timestamp (UTC)** | 2026-02-26T16:57:49Z |
| **Timestamp (Local)** | 2026-02-26 10:57:49 CST |
| **Scope** | Validate synthesis claim: "Pipeline is fragmented and incomplete" |
| **Primary Reference** | [00-synthesis-summary.md](./00-synthesis-summary.md) |
| **Related Audit Detail** | [02-scripts-workflow.md](./02-scripts-workflow.md) |
| **Status** | Complete |

---

## Claim Under Review

From [00-synthesis-summary.md](./00-synthesis-summary.md):

> "Pipeline is fragmented and incomplete: run_complete_pipeline.sh only calls 3 of 7 steps; numbering collisions; ghost references to nonexistent files."

---

## Executive Verdict

**Verdict: The claim is materially true.**

The issue is **not** that `scripts/pipeline/` is deprecated. The directory and core projection scripts are active and recently modified.  
The specific problem is that the orchestration/documentation layer is inconsistent: parts still reflect an older 3-step workflow while newer required/adjacent steps and runner names were added later.

---

## Evidence Summary

### 1. `run_complete_pipeline.sh` is a 3-step runner

`scripts/pipeline/run_complete_pipeline.sh` explicitly runs:

1. `01_process_demographic_data.py`
2. `02_run_projections.py`
3. `03_export_results.py`

It labels these as **STEP 1/3**, **2/3**, **3/3**.

### 2. Additional numbered pipeline scripts exist but are not called by the shell runner

These scripts are present in `scripts/pipeline/`:

- `00_prepare_processed_data.py`
- `01_compute_residual_migration.py`
- `01b_compute_convergence.py`
- `01c_compute_mortality_improvement.py`

None are invoked by `run_complete_pipeline.sh`.

### 3. Numbering collision is real

Two different step files share the `01_` prefix:

- `01_process_demographic_data.py`
- `01_compute_residual_migration.py`

Plus letter-suffixed step files:

- `01b_compute_convergence.py`
- `01c_compute_mortality_improvement.py`

This makes step ordering ambiguous.

### 4. Ghost reference is real

`scripts/projections/run_all_projections.py` is referenced in multiple current docs/instructions, but the file does not exist.

Examples of active references include:

- `AGENTS.md`
- `README.md`
- `CLAUDE.md`
- `docs/NAVIGATION.md`
- `DEVELOPMENT_TRACKER.md`

Current `scripts/projections/` contains `run_pep_projections.py` and `run_sdc_2024_comparison.py`, but no `run_all_projections.py`.

### 5. This is current workflow debt, not an archived/deprecated subsystem

Evidence that this is current:

- `scripts/pipeline/02_run_projections.py`, `03_export_results.py`, and `scripts/projections/run_pep_projections.py` have recent commits on **2026-02-26**.
- ADR implementation notes reference these scripts as active implementation files (e.g., ADR-054).
- `REPOSITORY_INVENTORY.md` lists `scripts/pipeline/` runner/scripts as active files.

Evidence that parts are stale:

- `run_complete_pipeline.sh` last changed on **2025-12-18** and still reflects the original 3-step design.
- Top-level docs still instruct use of nonexistent `scripts/projections/run_all_projections.py`.

---

## Interpretation

The audit statement is accurate in substance:

- The pipeline surface area has evolved.
- Orchestration and docs were not fully reconciled.
- Operators can follow valid-looking instructions that are now incomplete or broken.

This is best described as **active pipeline with stale orchestration/documentation seams**, not fully deprecated code.

---

## Decision-Relevant Notes

1. The project currently has multiple competing "entry points" for running projections, and at least one is nonexistent.
2. The `run_complete_pipeline.sh` script can still run, but it does not represent the full modern processing story implied by newer migration/mortality/convergence workflows.
3. The inconsistency is high-impact for reproducibility and onboarding, even though core engine code appears active and maintained.

