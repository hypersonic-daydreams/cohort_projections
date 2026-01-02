# Hygiene Improvement Proposal: The "Knowledge Intelligence" System

**Goal**: Elevate Repository Hygiene from "File Tracking" to "Semantics Tracking".

## Executive Summary
Our current "Repository Intelligence" system is excellent at tracking *files* (existence, modification time). However, it is blind to the *content* of our governance documents. It knows `ADR-005.md` exists, but it doesn't know if it is `Accepted`, `Deprecated`, or when it was `Last Reviewed`.

This proposal recommends expanding the system to parse and index this metadata into PostgreSQL, allowing us to query the "health" of our project knowledge just as we query the health of our data.

## 1. Unified Governance Structure

### Current State
*   **ADRs**: `docs/adr/` (Good)
*   **SOPs**: `docs/sops/` (Good)
*   **Templates**: Scattered (`docs/sops/templates/`, `docs/adr/020-reports/REPORT_TEMPLATE.md`, root `TEMPLATE.md`).

### Recommendation: `docs/governance` Unification
Move all governance artifacts into a coherent hierarchy.
```
docs/
  governance/
    adrs/           # Architecture Decision Records
    sops/           # Standard Operating Procedures
    templates/      # Centralized templates for EVERYTHING
    reports/        # Formal sub-agent reports
```
**Benefit**: Single source of truth. "If it rules the project, it's in `docs/governance`".

## 2. "Knowledge Intelligence" Database Expansion

### Current Limitation
`code_inventory` table only has: `filepath`, `status` (active/deleted), `last_validated`.

### Recommendation: `governance_inventory` Table
Create a new table (or view) to track the *semantic* state of documentation.

**Proposed Schema**:
*   `id`: UUID
*   `file_id`: FK to `code_inventory`
*   `doc_type`: `ADR`, `SOP`, `REPORT`
*   `doc_id`: `001`, `022`, etc.
*   `title`: "Fertility Rate Processing"
*   `status`: `DRAFT`, `PROPOSED`, `ACCEPTED`, `DEPRECATED`
*   `last_reviewed_date`: DATE
*   `next_review_due`: DATE (Calculated)

### Automation
Update `scan_repository.py` to:
1.  Detect if a file is in `docs/governance/`.
2.  Parse the Markdown metadata table (common in our docs).
3.  Upsert this metadata into `governance_inventory`.

## 3. Automated Staleness Checks

Once metadata is in Postgres, we can run simple SQL queries to answer critical questions:
*   "Which active SOPs haven't been reviewed in 6 months?"
*   "Do we have any 'Proposed' ADRs that are >30 days old?" (Stalled decisions).

**Action**: A weekly generic report (or pre-commit warning) listing "Stale Knowledge".

## 4. Workflows as Code

### Recommendation: "SOP-Runner"
Instead of just reading SOPs, consider an interactive CLI tool (`scripts/run_sop.py 001`) that:
1.  Reads `SOP-001.md`.
2.  Parses the checklist items.
3.  Prompts the user interactively: "Have you completed Phase A? [y/N]".
4.  Logs the run to `run_history` (just like a data projection!).

**Benefit**: This treats *process execution* as a form of *data processing*, maximizing repeatability.

## Summary of Prudent Next Steps
1.  **Refactor**: Centralize templates into `docs/governance/templates`.
2.  **Schema**: Add `metadata` JSONB column to `code_inventory` (least brittle approach).
3.  **Parser**: Upgrade `scan_repository.py` to extract Frontmatter/Table metadata.
