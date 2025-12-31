# Claim Segmentation and Support Annotation Guidelines

## Claim Granularity
- Split multi-clause statements into separate claims when each clause can be verified independently.
- Each quantitative result is its own claim.
- Methods statements ("we estimate", "we apply", "we model") are claims.
- Definitions and scope statements are claims.
- Each figure/table caption is a claim, plus any key numeric or causal statements inside.

## Required Fields
- `claim_id`: C0001, C0002, ...
- `claim_text`: Verbatim or lightly normalized text.
- `claim_type`: descriptive, comparative, causal, forecast, methodological, definition, normative.
- `source`: Minimum is `pdf_page`.
- `status`: Start as `unassigned`.

## Support Annotation (Separate Step)
After claims are listed, add two support-type annotations for each claim:
- `support_primary`: Most appropriate or robust evidence/support type.
- `support_alternative`: Alternative evidence/support type if primary is not feasible.

Use short, concrete descriptions (e.g., "replicate regression with provided data", "cite peer-reviewed estimate", "recalculate from table", "official dataset cross-check").

## Argument Map Integration
- When argument mapping begins, link relevant argument node IDs in `argument_ids`.
- If a missing warrant or grounds is identified, add the missing claim to the manifest.

## Examples
- claim_type: forecast
  support_primary: "replicate model output from code and data"
  support_alternative: "cross-check against independent projection source"

- claim_type: causal
  support_primary: "verify identification strategy and re-estimate effect"
  support_alternative: "cite prior causal estimate with comparable design"
