---
title: "v0.8.6 Status Update Memo"
date: 2026-01-04
context: "v0.8.6 critique implementation"
status: "update"
---

# v0.8.6 Status Update Memo (2026-01-04)

## Summary
- Data extensions are largely complete: ACS moved-from-abroad proxy (2010-2023), PEP components 2000-2024 with regime markers, LPR state totals FY2000-FY2023 (FY2012 from yearbook PDF), refugee/SIV series for FY2021-FY2024 (partial for FY2021/FY2023/FY2024) with OCR-assisted FY2024 coverage, manifests and validation logs updated.
- Remaining data gaps: refugee FY2022 state-by-nationality panel (ND-only placeholder); pre-2000 Census components not yet integrated.

## Reattempted Data Acquisition (Results)
- RPC FY2022 arrivals PDF: page 1 is text, pages 2+ are image-only; extraction with `pdfplumber` fails without OCR.
- RPC FY2024 arrivals PDF: text is encoded (cid glyphs); extraction yields unreadable labels.
- RPC XLSX alternatives for FY2022/FY2024: common rpc.state.gov XLSX endpoints return 404.
- OHSS LPR yearbooks FY2007-FY2013: CDN blocks programmatic downloads (HTTP 403). Manual download required.
- Census pre-2000 components: `/programs-surveys/popest/datasets/1980-1990/` exposes `state/asrh/st_int_asrh.txt` (age/sex/race/h, not components of change). `1990-2000` state/intercensal directories are not indexable via HTTP listing.

## Next Actions (Agreed / Suggested)
- Manual download of OHSS LPR yearbook ZIPs for FY2007-FY2013, then ingest with `sdc_2024_replication/data_immigration_policy/scripts/process_dhs_lpr_data.py`.
- Decide whether to run OCR for FY2022/FY2024 RPC PDFs to recover state-by-nationality panels.
- Confirm whether pre-2000 components are still required before further sourcing.

## Tests
- `uv run pytest tests/ -q` (768 passed, 5 skipped; warnings from `cohort_projections/output/visualizations.py` about `set_ticklabels`).

## References
- Remaining tasks tracker: `sdc_2024_replication/revisions/v0.8.6/remaining_tasks_v0.8.6.md`
- Archived full tracker: `sdc_2024_replication/revisions/v0.8.6/progress_tracker_v0.8.6_critique_v0.8.5.md`
- Response notes: `sdc_2024_replication/revisions/v0.8.6/response_to_critique.md`
- Data manifest: `data/DATA_MANIFEST.md`
- Validation log: `data/processed/immigration/analysis/immigration_v0.8.6_validation.md`
