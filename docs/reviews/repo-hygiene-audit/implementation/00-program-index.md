# Repo Hygiene Audit Implementation Program Index

**Status:** Execution In Progress (B05 Wave 1 complete)  
**Audit:** `repo-hygiene-audit`  
**Source of truth for verified claims:** `../verification/claims_registry.yaml`

## Purpose

This directory organizes verified repo-hygiene findings into a tracked implementation
program with batch-level evidence, gate records, and execution outcomes.

Goals:
- Keep all findings discoverable from one entry point.
- Harmonize findings into actionable batches.
- Define dry-run and go/no-go gates before code or structure changes.
- Preserve execution evidence as batches move from preflight to completion.

## Program Artifacts

| File | Purpose |
|---|---|
| `00-program-index.md` | Entry point and operating model |
| `01-findings-catalog.yaml` | Canonical planning catalog keyed to `RHA-*` claim IDs |
| `02-action-batches.yaml` | Sequenced remediation waves with dependencies and risk profiles |
| `03-dry-run-validation-matrix.md` | Batch-level dry-run procedures and quality gates |
| `04-go-no-go-checklist.md` | Decision checklist to authorize implementation |
| `05-dashboard-template.md` | Reusable status dashboard format for execution phase |

## Upstream Inputs

- Verified evidence dashboard: `../verification/progress.md`
- Verified claims registry: `../verification/claims_registry.yaml`
- Evidence artifacts: `../verification/evidence/*.json`

## Operating Model

1. Curate and maintain metadata in `01-findings-catalog.yaml`.
2. Keep sequencing and dependencies current in `02-action-batches.yaml`.
3. For each selected batch, run the dry-run gates in `03-dry-run-validation-matrix.md`.
4. Complete `04-go-no-go-checklist.md` before implementation.
5. Track execution using `05-dashboard-template.md`.

## Guardrails

- Do not edit audit report findings text (`00-07*.md`) during planning.
- Keep planning artifacts reproducible as implementation evolves.
- Preserve claim IDs (`RHA-*`) as stable join keys across all planning artifacts.
