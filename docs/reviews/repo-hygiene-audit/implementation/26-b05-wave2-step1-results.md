# B05 Wave 2 Step 1 Results

## Batch

- Batch ID: `B05`
- Batch Name: `repository_footprint_and_data_hygiene`
- Date (UTC): 2026-02-27
- Step: `Wave 2 / Step 1` (owner-confirmed archive actions)

## Owner Decisions Executed

1. Created root archive directory: `archived/`.
2. Moved stale exports:
   - from `data/exports/low_growth/`
   - to `archived/low_growth_exports/`
3. Kept placeholder directories unchanged:
   - `data/raw/census/acs`
   - `data/raw/census/decennial`
   - `data/raw/census/pep`
   - `docs/governance/templates`
4. Moved root Excel artifact:
   - from `ward_county_nd_population_2008_2024.xlsx`
   - to `archived/ward_county_nd_population_2008_2024.xlsx`

## File-System Verification

Command outcomes:
- `LOW_GROWTH_ARCHIVED=1`
- `WARD_FILE_ARCHIVED=1`
- `LOW_GROWTH_REMAINING=0`
- `WARD_FILE_REMAINING=0`

Archive structure now includes:
- `archived/low_growth_exports/`
- `archived/ward_county_nd_population_2008_2024.xlsx`

## B05 Claim Replay (Post-Step)

Command:
```bash
source .venv/bin/activate
python scripts/reviews/run_claim_checks.py run \
  --claim-id RHA-011 --claim-id RHA-012 --claim-id RHA-015 \
  --claim-id RHA-017 --claim-id RHA-018 --claim-id RHA-026
```

Result summary:
- `RHA-011`: PASS (`1/1`)
- `RHA-012`: PASS (`1/1`)
- `RHA-015`: PASS (`1/1`)
- `RHA-017`: FAIL (`0/1`) expected drift after low-growth archival
- `RHA-018`: PASS (`1/1`)
- `RHA-026`: PASS (`1/1`)

Evidence log:
- `check-replay-b05-wave2-step1.txt`

## Status

- Wave 2 Step 1: `complete`
- B05 overall: `in_progress`
- Remaining Wave 2 work: finish archive placement for remaining root-clutter items and execute SDC extraction step.
