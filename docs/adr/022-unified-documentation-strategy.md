# 22. Unified Documentation & Reproducibility Strategy

Date: 2026-01-01
Status: Proposed

## Context
The project operates in a hybrid environment: serving the North Dakota Department of Commerce (state agency) while producing research for academic publication. This creates dual requirements:
1.  **Reproducibility**: Exact parameters and data versions used for a projection must be logging and retrievable for "Methodology" sections.
2.  **Unified Access**: Documentation is currently scattered ("strewn about") across scripts, READMEs, and ADRs. A single entry point is needed without sacrificing the developer convenience of co-located documentation.

## Decision
We will implement a **Unified Documentation & Reproducibility System** backed by the `cohort_projections_meta` PostgreSQL database.

### 1. Unified Documentation Index
Instead of manually maintaining a "Master Documentation" file, we will auto-generate it from the `documentation_links` table.

-   **Co-location**: Documentation (READMEs, methodology notes) remains next to the code it describes.
-   **Centralization**: A process `scripts/intelligence/generate_docs_index.py` will query the DB to build `docs/INDEX.md`, organizing links by function (Analysis, ETL, etc.) and status.

### 2. Methodological Reproducibility (`run_history`)
We will add a `run_history` table to the metadata database to auto-log execution details for key scripts (projections, statistical models).

**Schema**:
-   `id`: UUID
-   `script_name`: (FK to code_inventory)
-   `execution_time`: Timestamp
-   `git_commit`: Hash
-   `parameters`: JSONB (e.g., `{ "start_year": 2024, "scenario": "high_migration" }`)
-   `input_manifest`: JSONB (List of input file IDs/Checksums)
-   `output_manifest`: JSONB (List of output file paths)

### 3. "Living Methodology"
We will create a template for the "Methodology Section" of papers that pulls directly from `run_history`.
-   *Example*: "We ran the cohort component model (version `a1b2c`) using the `high_migration` scenario. Input data included Census PEP 2024 (checksum `xyz`)..." -> This text can be partially checking against the DB.

## Consequences
### Positive
-   **No "Stale" Index**: The documentation index is always in sync with the repository state (guaranteed by the pre-commit hook).
-   **Audit Ready**: Any result can be traced back to the exact code version and data inputs used.
-   **Operates as One**: Developers write docs locally; consumers read them globally via the generated index.

### Negative
-   **Database Dependency**: Documentation exploration relies on the DB being up-to-date (mitigated by pre-commit hooks).
-   **Integration Effort**: Key scripts must be updated to write to `run_history`.

## Compliance
-   **Dataverse/State**: Meets requirements for audit trails.
-   **Academic**: Supports "Replication Package" standards.
