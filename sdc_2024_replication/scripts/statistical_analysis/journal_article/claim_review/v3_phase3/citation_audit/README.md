# Citation Audit (APA Compatibility)

Purpose: Ensure every in-text citation key resolves to a BibTeX entry, every BibTeX entry is cited in the text (unless `\nocite{*}` is used), and entries meet APA 7th completeness requirements.

This audit focuses on citation integrity (key matching) plus APA 7th field completeness. It does not fully validate punctuation or author-year formatting, but it flags missing required metadata.

## Inputs
- LaTeX sources (follows `\input`, `\include`, `\subfile`, `\import`, `\subimport` from `main.tex` when available)
- `references.bib`

## Outputs
- `citation_audit_report.json` - Structured report for AI agents
- `citation_audit_report.md` - Human-readable summary
- `citation_audit_report.html` - Highlighted report with missing fields marked
- `citation_entries.jsonl` - APA completeness audit per entry (canonical)
- `citation_entry_schema.json` - JSON Schema for `citation_entries.jsonl`
- `citation_fixes_schema.json` - JSON Schema for optional fixes input
- `AI_AGENT_GUIDE.md` - Succinct workflow for AI agents

## Usage
```bash
source .venv/bin/activate
python sdc_2024_replication/scripts/statistical_analysis/journal_article/claim_review/v3_phase3/citation_audit/check_citations.py \
  --tex-root sdc_2024_replication/scripts/statistical_analysis/journal_article \
  --bib-file sdc_2024_replication/scripts/statistical_analysis/journal_article/references.bib
```

### Additional Options
- `--main-tex path1.tex path2.tex`: Explicit TeX entry points to follow includes.
- `--scan-all`: Scan all `.tex` files under `--tex-root`.
- `--extra-cite-commands cmd1 cmd2`: Add custom citation command names that do not include `cite`.
- `--fixes-file path.jsonl`: Apply structured fixes (see `citation_fixes_schema.json`).

### Strict Mode
Use `--strict` to exit non-zero when missing/uncited entries, APA-required fields, or duplicate BibTeX keys are found.

## Notes on APA Style
- APA 7th requires consistent author-year citations and complete reference entries.
- This audit validates key matching and required metadata fields only.
- If APA formatting issues are found, they should be corrected in the LaTeX style configuration or BibTeX entries.

## Limitations
- Custom citation macros that do not include `cite` in the command name must be added via `--extra-cite-commands`.
- `\nocite{*}` disables uncited-entry evaluation; this is noted in the report.
