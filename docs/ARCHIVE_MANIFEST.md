# Archive Manifest

| Attribute | Value |
|-----------|-------|
| **Status** | current |
| **Last Updated** | 2026-06-23 |
| **Owner** | Repository maintainers |
| **Governs** | The three archive zones defined in `ARCHIVE_STRATEGY.md` |

The single lookup for "what is archived, and why." Grouped by zone. Large file
groups are summarized by directory + its stale reason rather than listed
file-by-file. The archival convention and lifecycle are in `ARCHIVE_STRATEGY.md`.

**Zone summary (counts on disk at last update):**

| Zone | Path | Files | Tracked? |
|------|------|-------|----------|
| Deprecated data outputs | `data_archives/` (renamed from `archived/`) | ~112 | gitignored |
| Deprecated documentation | `docs/archive/` | 13 | git-tracked |
| Transient artifacts | `scratch/archive/` | 79 | gitignored |
| **Inbound** (being moved) | `data/projections/{high_growth,restricted_growth}/` → `data_archives/` | ~594 data files | gitignored |

---

## Zone 1 — `data_archives/` (deprecated data outputs)

Gitignored, on-disk only. Renamed from the former `archived/`. Each subdir
should carry a `README_STALE.md` per `ARCHIVE_STRATEGY.md`.

| Item | Reason / ADR | Last-valid-date | Disposition |
|------|--------------|-----------------|-------------|
| `low_growth_exports/county/` (53 CSV + 53 Excel, gzipped per-county `low_growth` projections) | Deprecated `low_growth` scenario; superseded by the CBO-adjusted single-baseline model (ADR-065). | 2026-02-17 | Retired — keep for auditability; do not regenerate. |
| `low_growth_exports/summaries/` (5 summary CSVs: county age distribution, dependency ratios, growth rates, race composition, sex ratio) | Same as above — derived from the deprecated `low_growth` run. | 2026-02-17 | Retired. |
| `ward_county_nd_population_2008_2024.xlsx` | One-off Ward County historical extract; superseded by canonical PEP inputs in `data/processed/`. | ~2026-01-05 | Retired. |

> Action item: add a `README_STALE.md` to `data_archives/` (and its
> `low_growth_exports/` subtree) stating the above, per the lifecycle in
> `ARCHIVE_STRATEGY.md`.

---

## Zone 2 — `docs/archive/` (deprecated documentation)

Git-tracked. Index maintained in `docs/archive/README.md`. Each file retains its
original content below an archive header.

| Item | Reason / ADR | Archived-date | Disposition |
|------|--------------|---------------|-------------|
| `SESSION_SUMMARY.md` | Session-specific notes. | 2025-12-31 | Historical. |
| `SESSION_CONTINUATION_SUMMARY.md` | Session-specific notes. | 2025-12-31 | Historical. |
| `BIGQUERY_DATA_SUMMARY.md` | Session-specific notes. | 2025-12-31 | Historical. |
| `IMPLEMENTATION_SUMMARY.md` | Implementation complete. | 2025-12-31 | Historical. |
| `FERTILITY_IMPLEMENTATION_SUMMARY.md` | Implementation complete. | 2025-12-31 | Historical. |
| `GEOGRAPHIC_MODULE_SUMMARY.md` | Implementation complete. | 2025-12-31 | Historical. |
| `OUTPUT_MODULE_IMPLEMENTATION.md` | Implementation complete. | 2025-12-31 | Historical. |
| `PARALLEL_IMPLEMENTATION_SUMMARY.md` | Implementation complete. | 2025-12-31 | Historical. |
| `PIPELINE_IMPLEMENTATION_SUMMARY.md` | Implementation complete. | 2025-12-31 | Historical. |
| `PROJECT_STATUS.md` | Superseded by `DEVELOPMENT_TRACKER.md`. | 2026-02-02 | Superseded. |
| `REPOSITORY_HYGIENE_TRACKER.md` | Completed (all 11 task groups done). | 2026-02-02 | Completed. |
| `DEVELOPMENT_TRACKER_2026-02-26.md` | Historical snapshot of the tracker; superseded by the live `DEVELOPMENT_TRACKER.md`. | 2026-02-26 | Historical snapshot. |
| `README.md` | Index for this zone (not itself archived content). | 2026-02-02 | Active index. |

---

## Zone 3 — `scratch/archive/` (transient artifacts)

Gitignored, auto-cleanable, no preservation guarantee. Summarized by directory.

| Item (directory) | Contents | Reason | Last-valid-date | Disposition |
|------------------|----------|--------|-----------------|-------------|
| `2025_popest_data/` (46 files incl. `charts/`) | Census PEP 2025-vintage release data, draft news releases (`.docx`/`.md`/`.xlsx`), components-of-change tables, and diverging-stacked-bar / waterfall charts. Has its own `README.md`. | One-off analysis around the Jan-2026 PEP 2025 release; superseded once Vintage 2025 was folded into the canonical inputs (ADR-066). | ~2026-01/02 | Transient — keep until next scratch cleanup. |
| `repo_hygiene_evidence_superseded/` (33 files) | Timestamped per-finding JSON evidence (`…Z_RHA-NNN.json`) from an earlier repo-hygiene audit pass. | Superseded by the 2026-06-23 hygiene proposal; retained only as raw evidence. | 2026-02-27 | Transient — auto-cleanable. |

---

## Inbound — being moved into `data_archives/` (hygiene Decision 1)

These STALE scenario directories currently still live under `data/projections/`
and carry their own `README_STALE.md`. Per the recorded decision they are being
moved into `data_archives/`, carrying their stale markers with them.

| Item | Reason / ADR | Last-valid-date | Disposition |
|------|--------------|-----------------|-------------|
| `data/projections/high_growth/` (~297 data files + `README_STALE.md`) | Pre-ADR-065/066 outputs. ADR-065 made the CBO-adjusted baseline the only active scenario and kept `high_growth` only as an inactive internal sensitivity; ADR-066 corrected the base population to PEP Vintage 2025. State total in this dir: 799,358 (2025) → 1,078,346 (2055), not comparable to the current baseline. | 2026-02-26 (run `projection_run_20260226_150927`) | Retire as stale — move to `data_archives/`; do not regenerate. |
| `data/projections/restricted_growth/` (~297 data files + `README_STALE.md`) | Pre-ADR-065/066 outputs. ADR-065 retained `restricted_growth` only as a deprecated, inactive alias; ADR-066 corrected the base population. Diverges from the current baseline (2055: 828,470 vs. baseline 876,479). | 2026-02-26 (run `projection_run_20260226_150733`) | Retire as stale — move to `data_archives/`; do not regenerate. |

> Related stale marker, **not** being moved: `data/projections/exports/geojson/`
> carries a `README_STALE.md` (GeoJSON exports built from the 2026-02-26 runs,
> pre-ADR-065/066). It stays under `data/projections/exports/` for now;
> disposition tracked with the export tree.

---

## Source Vaults (NOT archived — listed here so they are never confused for archive content)

Per `ARCHIVE_STRATEGY.md`, these are immutable reference data under `data/raw/`
and must **never** be deleted. They are outside the archive taxonomy:

| Path | Nature |
|------|--------|
| `data/raw/immigration/rpc_vault/` | Source immigration reference data (vault; renamed from `rpc_archives/`). |
| `data/raw/nd_sdc_2024_projections/source_files/vault/` | Source SDC-2024 projection files (vault; renamed from `source_files/backup/`). |
