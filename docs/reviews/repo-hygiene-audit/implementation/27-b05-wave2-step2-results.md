# B05 Wave 2 Step 2 Results

## Batch

- Batch ID: `B05`
- Batch Name: `repository_footprint_and_data_hygiene`
- Date (UTC): 2026-02-27
- Step: `Wave 2 / Step 2` (remaining archive placements + SDC extraction)

## Scope Executed

1. Extracted in-repo SDC subtree to sibling repository location:
   - from `./sdc_2024_replication/`
   - to `../sdc_2024_replication/`
2. Completed remaining root-clutter placements:
   - moved `2025_popest_data/` to `scratch/archive/2025_popest_data/`
   - moved `journal_article_pdfs/` to `../sdc_2024_replication/journal_article_pdfs/`
   - moved root markdown artifacts to `../sdc_2024_replication/revisions/root_artifacts/`
   - deleted root ingest logs (`ingest.log`, `ingest_full.log`, `ingest_nohup.log`)
3. Retargeted root symlinks:
   - `journal_article_output -> ../sdc_2024_replication/scripts/statistical_analysis/journal_article/output`
   - `journal_article_versions -> ../sdc_2024_replication/scripts/statistical_analysis/journal_article/output/versions`
4. Applied SDC rate dedup policy:
   - removed in-repo duplicate rate CSVs under `data/processed/immigration/rates/`
   - canonical files remain in sibling repo `data/` and `data_updated/`

## File-System Verification

Command outcomes:
- `SDC_SIBLING_EXISTS=1`
- `SDC_IN_REPO_EXISTS=0`
- `ROOT_CLUTTER_REMAINING=0`
- `POPEST_ARCHIVED=1`
- `ROOT_ARTIFACTS_MOVED=1`
- `JOURNAL_ARTICLE_PDFS_MOVED=1`
- `IN_REPO_RATE_FILES=0`
- `SIBLING_CANONICAL_RATE_FILES=5`

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
- `RHA-017`: PASS (`1/1`)
- `RHA-018`: PASS (`1/1`)
- `RHA-026`: PASS (`1/1`)

Evidence artifacts:
- `docs/reviews/repo-hygiene-audit/verification/evidence/20260227T181233Z_RHA-011.json`
- `docs/reviews/repo-hygiene-audit/verification/evidence/20260227T181234Z_RHA-012.json`
- `docs/reviews/repo-hygiene-audit/verification/evidence/20260227T181234Z_RHA-015.json`
- `docs/reviews/repo-hygiene-audit/verification/evidence/20260227T181234Z_RHA-017.json`
- `docs/reviews/repo-hygiene-audit/verification/evidence/20260227T181234Z_RHA-018.json`
- `docs/reviews/repo-hygiene-audit/verification/evidence/20260227T181234Z_RHA-026.json`

## Status

- Wave 2 Step 2: `complete`
- B05 overall: `complete`
- Remaining B05 work: none
