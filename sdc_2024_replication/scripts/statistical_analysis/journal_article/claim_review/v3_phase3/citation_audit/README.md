# Citation Audit (APA Compatibility)

Purpose: Ensure every in-text citation key resolves to a BibTeX entry, every BibTeX entry is cited in the text (unless `\nocite{*}` is used), and entries meet APA 7th completeness requirements.

This audit focuses on citation integrity (key matching) plus APA 7th field completeness. It does not fully validate punctuation or author-year formatting, but it flags missing required metadata.

## Inputs
- LaTeX sources (defaults to `main.tex`, `sections/`, and `revision_sections/` under the journal article root)
- `references.bib`

## Outputs
- `citation_audit_report.json` - Structured report for AI agents
- `citation_audit_report.md` - Human-readable summary
- `citation_entries.jsonl` - APA completeness audit per entry (canonical)
- `citation_entry_schema.json` - JSON Schema for `citation_entries.jsonl`
- `AI_AGENT_GUIDE.md` - Succinct workflow for AI agents

## Usage
```bash
source .venv/bin/activate
python sdc_2024_replication/scripts/statistical_analysis/journal_article/claim_review/v3_phase3/citation_audit/check_citations.py \
  --tex-root sdc_2024_replication/scripts/statistical_analysis/journal_article \
  --bib-file sdc_2024_replication/scripts/statistical_analysis/journal_article/references.bib
```

### Strict Mode
Use `--strict` to exit non-zero when missing/uncited entries or APA-required fields are found.

## Notes on APA Style
- APA 7th requires consistent author-year citations and complete reference entries.
- This audit validates key matching and required metadata fields only.
- If APA formatting issues are found, they should be corrected in the LaTeX style configuration or BibTeX entries.

## Limitations
- Multi-line citation commands may not report exact line numbers; the key match still works.
- Custom citation macros that do not include `cite` in the command name are not detected.
- `\nocite{*}` disables uncited-entry evaluation; this is noted in the report.
