# codebase-catalog (vendored minimal build)

This repository uses a PostgreSQL-backed “repository intelligence” system.

The full `codebase_catalog` implementation was originally maintained outside the repo.
To keep the environment reproducible here, this directory provides:
- `update-code-inventory`: best-effort inventory refresh (skips if DB unavailable)
- `check-data-manifest`: minimal manifest validation (fails fast on missing manifest)
