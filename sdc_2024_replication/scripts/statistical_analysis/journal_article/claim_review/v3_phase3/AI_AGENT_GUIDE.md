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
2) **Annotate support types**
   - Fill `support_primary` and `support_alternative` for each claim.
3) **Map arguments**
   - Create argument nodes and link them to claim IDs.
4) **Verify claims**
   - Add evidence in `evidence/C####.md` and update claim `status`.

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
