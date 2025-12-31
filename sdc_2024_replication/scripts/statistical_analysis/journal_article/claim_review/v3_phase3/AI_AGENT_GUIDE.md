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

## Argument Mapping Rules

### Required Toulmin Structure
Each argument group MUST have at minimum:
1. **Claim** - Central assertion (supports_argument_ids: [])
2. **Grounds** - Evidence supporting the claim (supports_argument_ids: [claim_id])
3. **Warrant** - Logical bridge explaining WHY grounds support claim (supports_argument_ids: [claim_id])

Optional (recommended where applicable):
4. **Backing** - Additional support for the warrant
5. **Qualifier** - Scope limitations ("typically", "in most cases")
6. **Rebuttal** - Counter-arguments or exceptions

### Key Distinction
- **GROUNDS = EVIDENCE** (data, statistics, observations from the paper)
- **WARRANTS = REASONING** (why the evidence supports the conclusion)
- Data alone is NOT an argument - every claim needs BOTH grounds AND warrant

### Example
```
Claim: "North Dakota's migration is highly volatile"
  ↑
Grounds: "CV of 82.5%, range from 30 to 3,000 migrants" [C0273, C0274]
  ↑
Warrant: "A CV this high means annual flows can deviate from the mean
          by nearly the mean itself, making prediction difficult" [C0276]
```

### Argument Mapping Tools
```bash
# Audit existing groups for completeness
python argument_map/map_section_arguments.py --audit

# Get prompt for mapping a section
python argument_map/map_section_arguments.py --section Results

# Get prompt for completing a specific group
python argument_map/map_section_arguments.py --group G020
```

### Linking Rules
- Use `supports_argument_ids` to link grounds→claim, warrant→claim, backing→warrant
- Use `rebuts_argument_ids` for counter-arguments
- Use `argument_group_id` (G###) to group related nodes

## Reporting Expectations
- Update `claims_manifest.jsonl` and `argument_map.jsonl` only.
- Do not overwrite the PDF or raw outputs.
- If a warrant is missing, identify the claim that serves as warrant and link it.
- Every argument group must pass the audit check (claim + grounds + warrant).
