# AI Agent Guide: Claim + Argument Review (v5_p305_complete)

## Goal
Create a complete, linked inventory of claims and arguments for the v5_p305_complete PDF, then verify each claim with evidence.

## Files You Should Use
- `claims/claims_manifest.jsonl` (canonical claim list)
- `argument_map/argument_map.jsonl` (canonical argument map)
- `claims/claim_schema.json` and `argument_map/argument_schema.json` (schemas)
- `document_metadata.json` (PDF hash for consistency)

## Workflow (Agent Order)
1) **Extract claims**
   - Add discrete claims to `claims_manifest.jsonl` with source locations.
   - Use parsers for repeatable extraction:
     - `python claims/build_section_claims.py --section "SectionName" --dry-run` (preview)
     - `python claims/build_section_claims.py --section "SectionName" --write` (update manifest)
   - Run QA after extraction:
     - All sections: `python claims/qa_claims.py --all-sections --compare-section`
     - Single section: `python claims/qa_claims.py --section "SectionName"`
2) **Annotate support types**
   - Fill `support_primary` and `support_alternative` for each claim.
3) **Map arguments**
   - Create argument nodes and link them to claim IDs.
4) **Verify claims**
   - Add evidence in `evidence/C####.md` and update claim `status`.

## Available Sections
- Abstract (pages 1-2)
- Introduction (pages 3-5)
- Data and Methods (pages 5-19)
- Results (pages 19-31)
- Discussion (pages 31-45)
- Conclusion (pages 45-47)
- Appendix (pages 51-61)

## Claim Rules
- Split multi‑clause statements into separate claims.
- Each numeric result is its own claim.
- Methods statements are claims.
- Figure/table captions are claims.

## Argument Rules
- Use Toulmin roles: claim, grounds, warrant, backing, qualifier, rebuttal.
- Link nodes via `supports_argument_ids` and `rebuts_argument_ids`.
- Use `argument_group_id` to group a chain.

## Reporting Expectations
- Update `claims_manifest.jsonl` and `argument_map.jsonl` only.
- Do not overwrite the PDF or raw outputs.
- If a warrant is missing, add a claim and link it.
