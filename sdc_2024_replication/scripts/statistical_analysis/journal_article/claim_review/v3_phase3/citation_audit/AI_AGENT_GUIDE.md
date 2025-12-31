# AI Agent Guide: Citation Audit (APA 7th)

## Goal
Validate that every in-text citation key resolves to a BibTeX entry, and that each BibTeX entry meets APA 7th required metadata.

## Run (from repo root)
```bash
source .venv/bin/activate
python sdc_2024_replication/scripts/statistical_analysis/journal_article/claim_review/v3_phase3/citation_audit/check_citations.py \
  --tex-root sdc_2024_replication/scripts/statistical_analysis/journal_article \
  --bib-file sdc_2024_replication/scripts/statistical_analysis/journal_article/references.bib
```
If the main TeX file is not `main.tex`, add `--main-tex path/to/file.tex`.

## Outputs (consume these)
- `citation_audit_report.json`: summary counts + missing/uncited keys.
- `citation_entries.jsonl`: per-entry APA audit records (canonical for agents).
- `citation_audit_report.html`: highlighted report with missing fields marked.
- `citation_fixes_schema.json`: schema for optional fixes input.

## What to Check
- Missing in BibTeX: any citation key used in text but absent from `.bib`.
- Uncited BibTeX: entries in `.bib` that never appear in text.
- Duplicate BibTeX keys: entries with the same citation key.
- APA required fields: entries with `status = missing_required` in JSONL.

## How to Report Findings
- For each key with `missing_required`, list the missing fields and entry type.
- For each missing/uncited key, list the file/line occurrences from the report.
- For duplicate keys, list the key and confirm which entry to keep.
- If `nocite{*}` is present, ignore uncited-entry warnings.

## Optional: Provide Fixes
- Create a JSONL file following `citation_fixes_schema.json`.
- Each record can add fields (e.g., `url`, `publisher`, `year`) or override `entry_type`.
- Use `--fixes-file path.jsonl` to apply fixes before re-running the audit.

## Notes
- APA edition is pinned to 7th in the script.
- The audit checks metadata completeness only (not punctuation or author-year formatting).
