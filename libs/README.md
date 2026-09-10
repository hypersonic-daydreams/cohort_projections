# libs/

Status: current · Last Updated: 2026-06-23

Orientation README for `libs/`. This directory is a **second package root** alongside the
main `cohort_projections/` package. It holds three small, separately-versioned support
packages that are vendored into this repo as editable installs. Each has its own
`pyproject.toml` and `README.md`, and each is wired into the environment via
`[tool.uv.sources]` in the repository-root `pyproject.toml`:

```toml
project-utils    = { path = "libs/project_utils",    editable = true }
evidence-review  = { path = "libs/evidence_review",  editable = true }
codebase-catalog = { path = "libs/codebase_catalog", editable = true }
```

## Why they are a separate package root

These packages were originally maintained **outside** this repository (e.g. a shared
`~/workspace/libs/…` location). They are vendored here as minimal/editable copies so that
environment resolution is reproducible on any machine even when those external paths are
not available. Keeping them under `libs/` — rather than folding them into
`cohort_projections/` — preserves their independent identity and versioning while still
letting the main package import them directly (`from project_utils import ...`, etc.).

## The three packages

- **`project_utils`** — lightweight **configuration and logging** utilities (`config.py`,
  `logging.py`) used across the `cohort_projections` codebase. The most actively imported
  of the three.
- **`evidence_review`** — a **vendored stub** of a fuller package maintained externally;
  present mainly to keep environment resolution reproducible when the external editable
  path is unavailable.
- **`codebase_catalog`** — a **vendored minimal build** of a PostgreSQL-backed "repository
  intelligence" system. It exposes two console-script entry points (`pyproject.toml`):
  - `update-code-inventory` — best-effort inventory refresh (no-ops when the DB is
    unavailable).
  - `check-data-manifest` — minimal data-manifest validation (fails fast on a missing
    manifest).
  Both are wired as pre-commit hooks in `.pre-commit-config.yaml`.

### Note on `codebase_catalog` (hygiene decision 2)

Per the repository hygiene proposal (decision 2), the bespoke inventory side of
`codebase_catalog` — the `update-code-inventory` CLI and the orphaned
`REPOSITORY_INVENTORY.md` chain it fed — is **being deprecated in favor of the workspace
ledger** (`workspace_silver.ledger`, owned by the command-center repo), which already
indexes this repo and provides current-vs-stale signaling. The **`check-data-manifest`
hook is retained** as a useful fast-fail guard. Do not invest in repairing the local
`cohort_projections_meta` inventory DB; route inventory/"what's current vs. archived"
questions to the workspace ledger instead.
