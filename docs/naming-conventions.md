# Naming & Metadata Conventions

| Attribute | Value |
|-----------|-------|
| **Status** | current |
| **Last Updated** | 2026-06-23 |
| **Owner** | Repository maintainers |
| **Source** | `docs/REPOSITORY_HYGIENE_PROPOSAL_2026-06-23.md` §7B (recorded decisions) |

These rules encode the conventions that already mostly exist in the repository,
so they read as intentional rather than inconsistent. They are adoptable for any
new file. Apply them to new work; do **not** mass-rename existing files (that is
high-churn and breaks cross-references).

---

## 1. Filename Casing by Category

| Category | Pattern | Example |
|----------|---------|---------|
| Governance/decision docs (ADR, SOP) | `NNN-kebab-title.md`, zero-padded 3 digits | `061-college-fix-model-revision.md`, `SOP-005-public-facing-prose-voice.md` |
| Time-stamped docs (reviews, postmortems) | `YYYY-MM-DD-kebab-title.md` (ISO date first) | `2026-03-01-publication-output-signoff.md` |
| Plans / guides / methodology | `kebab-case.md` | `observatory-start-here.md`, `methodology.md` |
| Stable root governance docs | `SCREAMING_SNAKE_CASE.md` | `AGENTS.md`, `CLAUDE.md`, `DEVELOPMENT_TRACKER.md`, `README.md` |
| `docs/` top-level docs | SCREAMING for stable entry-points, `snake_case` for technical/methodology support docs — pick one per file deliberately | `BIGQUERY_SETUP.md` (setup entry-point) vs. `census_api_usage.md` (support doc) |
| Python modules | `snake_case.py` | `residual_migration.py` |

---

## 2. ISO Date-Stamping

- Use **ISO `YYYY-MM-DD`** in filenames, headers, and prose. Never `YYYYMMDD`
  or `MM-DD-YYYY` for new files.
- In a time-stamped doc filename, the date comes **first**, then the kebab
  title: `2026-06-23-some-review.md`.

**Good:** `2026-03-01-pp3-imp14-results.md`
**Bad:** `pp3-imp14-results-20260301.md` (date not ISO-first)

> Note: data/output **filenames** use a compact `YYYYMMDD` token by long-standing
> convention (e.g. `nd_population_projections_provisional_20260217.xlsx`). That
> is the one place compact dates persist; prose and doc filenames use ISO.

---

## 3. Run-ID in Output Filenames

Output/export files carry a **run identifier**, not just a date — so an export
can be traced to the run that produced it from the name alone.

- Pattern: `nd_<thing>_<YYYYMMDD>_<run-id>.<ext>`, e.g.
  `nd_state_projection_20260611_m2026r1.parquet`.
- The run-id ↔ config-sha mapping is documented in `data/projections/README.md`
  (and run metadata under `data/metadata/projection_run_*.json`).

**Good:** `nd_county_projection_20260611_m2026r1.csv`
**Bad:** `nd_county_projection_20260611.csv` (date present, run not traceable)

---

## 4. Status Lives in the Directory, Not the Filename

Do **not** encode lifecycle status in filenames (no `-DONE`, `-OLD`,
`-DEPRECATED` suffixes). A closed item should not still look active in a
listing, and a renamed file should not break links.

- Prefer **moving** finished/superseded items into the correct archive zone
  (see `ARCHIVE_STRATEGY.md`) over per-file status suffixes.
- The directory + the file's metadata header carries status.
- **The one earned exception:** a genuinely immutable artifact may take a
  `-LOCKED` suffix (e.g. a locked config). Verify no tooling globs the file by
  exact name before adding the suffix.

**Good:** move a finished plan to `docs/archive/` and add a manifest entry.
**Bad:** rename it to `some-plan-COMPLETE.md` in place.

Real example from this repo: `population-projection-explosion-bug-investigation-2026-02-13.md`
reads as active from its name even though it is closed — its status is only
recoverable from `docs/plans/README.md`. New work should avoid this by moving
closed items rather than relying on the reader to look up status elsewhere.

---

## 5. ADR / Work-ID Cross-Reference Format

- Always cite with a **hyphen, no space**: `ADR-061`, `SOP-005`, `PP-005`,
  `CF-001`, `PUB-2026`.
- Never `ADR 061` or `ADR_061` — the space form fragments search and tooling.

**Good:** "per ADR-065, the CBO-adjusted baseline is the active scenario."
**Bad:** "per ADR 065 …"

See `docs/work-registry.md` for the full set of identifier schemes.

---

## 6. Metadata Header Block

Every non-trivial markdown document opens with a metadata block. Code modules
use a docstring header per SOP-002 instead.

Required fields:

```
| Attribute | Value |
|-----------|-------|
| **Status** | current \| supporting \| historical \| deprecated \| locked |
| **Last Updated** | YYYY-MM-DD |
| **Owner** | <person or role> |
```

Add `Supersedes` / `Superseded-By` rows when a document replaces or is replaced
by another. Status vocabulary:

| Status | Meaning |
|--------|---------|
| `current` | Authoritative and active. |
| `supporting` | A valid input/reference, not a primary source of truth. |
| `historical` | Kept for the record; describes a past state. |
| `deprecated` | Superseded; retained only for auditability. |
| `locked` | Immutable (e.g. a finalized release artifact). |

**Good:** the header at the top of this very file.
**Bad:** a multi-hundred-line doc with no date or status, so a reader cannot
tell at a glance whether it is current (the failure mode of the retired
`REPOSITORY_INVENTORY.md`).
