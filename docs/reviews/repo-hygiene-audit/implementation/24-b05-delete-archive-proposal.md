# B05 Delete/Archive Placement Proposal (Wave 2)

**Date (UTC):** 2026-02-27  
**Batch:** `B05` (`repository_footprint_and_data_hygiene`)  
**Operator:** codex

## Purpose

Track RB-005 action item 3 from proposal through owner-confirmed execution for root clutter,
stale exports, and empty placeholders.

## Candidate Inventory Snapshot

### Root clutter set (`RHA-015`)

| Item | Type | Tracked | Size | Proposed Action | Proposed Destination / Policy |
|---|---|---|---:|---|---|
| `RCLONE_TEST` | file | yes | 4.0K | remove from git (done in Wave 1) | add ignore rule to prevent re-tracking |
| `2025_popest_data/` | dir | no | 4.6M | archive | `scratch/archive/2025_popest_data/` (retain 180 days) |
| `journal_article_pdfs/` | dir | no | 33M | move to SDC repo | `../sdc_2024_replication/journal_article_pdfs/` |
| `chatgpt_feedback_on_v0.9.md` | file | no | 20K | archive | `../sdc_2024_replication/revisions/root_artifacts/` |
| `formula_audit_article-0.9-production_20260112_205726.md` | file | no | 16K | archive | `../sdc_2024_replication/revisions/root_artifacts/` |
| `ward_county_nd_population_2008_2024.xlsx` | file | no | 16K | archived (done 2026-02-27) | `archived/ward_county_nd_population_2008_2024.xlsx` |
| `ingest.log` | file | no | 4.0K | delete | ephemeral log |
| `ingest_full.log` | file | no | 4.0K | delete | ephemeral log |
| `ingest_nohup.log` | file | no | 16K | delete | ephemeral log |

### Stale export set (`RHA-017`)

- `data/exports/low_growth/` contains `111` files (latest timestamp: `2026-02-17`).
- No matching `data/projections/low_growth/` directory exists.

Owner decision + execution:
1. Archived from `data/exports/low_growth/` to `archived/low_growth_exports/` (2026-02-27).
2. Retain in archive for now; revisit deletion after B05/B06 closeout.

### Empty placeholder set (`RHA-026`)

Current non-`.gitkeep` entry counts:
- `data/raw/census/acs`: `0`
- `data/raw/census/decennial`: `0`
- `data/raw/census/pep`: `0`
- `docs/governance/templates`: `0`

Owner decision:
1. Keep empty placeholder directories as-is for now.
2. No directory removal in this wave.

## SDC Rate Dedup Evidence (`RHA-018`)

SHA-256 short hashes by location (`data`, `data_updated`, `data/processed/immigration/rates`):

- `adjustment_factors_by_county.csv`: identical across all three (`UNIQUE_HASH_COUNT=1`)
- `base_population_by_county.csv`: 2 variants (`UNIQUE_HASH_COUNT=2`), `data_updated` matches `data/processed`
- `fertility_rates_by_county.csv`: 2 variants (`UNIQUE_HASH_COUNT=2`), `data_updated` matches `data/processed`
- `migration_rates_by_county.csv`: 3 variants (`UNIQUE_HASH_COUNT=3`)
- `survival_rates_by_county.csv`: 2 variants (`UNIQUE_HASH_COUNT=2`), `data_updated` matches `data/processed`

Proposed Wave 2 policy:
1. Canonical source stays in extracted `sdc_2024_replication` repo (`data/` + `data_updated/` as documented there).
2. Keep this repo's `data/processed/immigration/rates/` copies only if actively consumed by projection code.
3. For each file, record source-of-truth mapping and checksum in a short manifest before deleting duplicates.

## Owner Decisions Applied (2026-02-27)

1. Created root `archived/` directory.
2. Archived stale low-growth exports to `archived/low_growth_exports/`.
3. Kept empty placeholder directories unchanged.
4. Moved `ward_county_nd_population_2008_2024.xlsx` to `archived/`.

## Remaining Approval Needed for Wave 2

1. Archive destination decision for `2025_popest_data/`.
2. Placement decisions for remaining root clutter files/logs noted in this document.

## Status

- RB-005 action 3: **in progress (proposal delivered + partial execution completed)**
- Wave 2 destructive execution: **partial complete; remaining destinations pending**
