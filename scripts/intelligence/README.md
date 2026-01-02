# Repository Intelligence Scripts

This directory contains the tooling for the **Repository Intelligence System**, which maintains the `cohort_projections_meta` PostgreSQL database.

## Scripts

### Core Scanners
-   **`scan_repository.py`**: The "Big Bang" scanner. Recursively scans the entire repo (respecting `.gitignore`), extracts docstrings, and populates the `code_inventory` table. Use this to reset or initialize the system.
-   **`link_documentation.py`**: Heuristically links code files to documentation (e.g., matching `README.md` in parent dirs). Populates `documentation_links`.

### Code Lifecycle
-   **`archive_manager.py`**: The "Garbage Collector". Moves files marked as `deprecated` in the database to the `archive/` directory to keep the main tree clean.

### Generators
-   **`generate_inventory_docs.py`**: Generates [REPOSITORY_INVENTORY.md](../../REPOSITORY_INVENTORY.md) from the DB. This is the primary map for AI agents.
-   **`generate_docs_index.py`**: Generates [docs/INDEX.md](../../docs/INDEX.md), the unified documentation index.

## Database Schema
The system uses the `cohort_projections_meta` database. See `data/metadata/migrations/` for schema definitions.

-   `code_inventory`: File paths, status, descriptions.
-   `documentation_links`: Links between code and docs.
-   `run_history`: Execution logs for reproducibility.
