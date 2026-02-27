# B05 Strategy Decisions (Owner-Approved)

**Date:** 2026-02-27  
**Batch:** B05 (`repository_footprint_and_data_hygiene`)  
**Purpose:** Record owner-approved archive/move/delete strategy decisions required to unblock B05 implementation planning and execution.

## Decisions

1. **Extract `sdc_2024_replication/` to its own repository** under the local `demography/` directory (WSL).
   - **Repo name must remain `sdc_2024_replication`** to preserve the existing GitHub remote linkage.
   - Target location (WSL): `~/workspace/demography/sdc_2024_replication/` (or equivalent under the `demography/` workspace root).
   - Implementation must **update or replace** any links and filepath references that break as a result of the move.
   - The extraction work should be treated as a **boundary change** with explicit inventory, validation, and rollback steps.

2. **Canonical location for SDC rate CSVs stays in the extracted `sdc_2024_replication` repository.**
   - Any duplicate copies in this repository should be removed only after:
     - all in-repo references are updated, and
     - claim-check replay gates confirm no regressions.

3. **Open triage items will be investigated before final placement decisions.**
   - Agents will inventory affected items and propose new canonical locations and retention policies (archive vs delete) for review.

## Implementation Notes / Requirements

- **References inventory required:** enumerate all in-repo references that point to `sdc_2024_replication/` paths and to the SDC rate CSV locations.
- **Provenance preservation:** if any data artifacts are moved, keep provenance and metadata intact (and continue honoring `data/raw/` immutability).
- **Rollback plan:** extraction must include an explicit rollback trigger and steps.

## Pending Proposals (Agent Work)

Agents should propose new locations and retention rules for:
- Root clutter set from `RHA-015` (e.g., `2025_popest_data/`, `journal_article_pdfs/`, stray root `.md` artifacts, root `.xlsx`, and `ingest*.log`).
- Stale scenario exports from `RHA-017` (`data/exports/low_growth/`).
- Empty placeholder directories from `RHA-026` (including `data/raw/census/{acs,decennial,pep}` and `docs/governance/templates/`).
