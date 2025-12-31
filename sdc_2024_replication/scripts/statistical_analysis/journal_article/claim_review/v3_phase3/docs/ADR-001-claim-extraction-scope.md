# ADR-001: Claim Extraction Scope and Purpose

## Status
Accepted

## Context
During review of the argument map viewer, a critical gap was identified: the claim manifest and argument map were capturing **study-generated findings** (results, methods, conclusions) but missing **externally-sourced claims** that situate and justify the paper.

### Examples of Missing Claim Types

**From the Abstract:**
1. "Demographic literature has extensively examined migration to gateway states" — literature characterization claim
2. "Small peripheral states have received comparatively little rigorous empirical attention" — research gap claim

**From the Introduction:**
1. "North Dakota has a population of approximately 780,000" — factual claim (potentially outdated)
2. "North Dakota ranks 47th among US states" — factual claim requiring verification
3. "Natural population growth has stagnated" — demographic trend claim
4. "Domestic out-migration persists" — demographic trend claim
5. "International migration has emerged as a critical and at times dominant component of population change" — demographic trend claim
6. "The Great Plains has experienced sustained population decline throughout much of the twentieth and early twenty-first century" — historical claim

### The Fundamental Problem
The current system conflates two distinct verification needs:

| Claim Type | Source of Evidence | Verification Method |
|------------|-------------------|---------------------|
| **Study-generated** | This paper's analysis | Check methodology, rerun code, verify statistics |
| **Externally-sourced** | Citations, external data | Verify citation accuracy, check source currency, confirm facts |

Study-generated claims are verified by our analytical pipeline. Externally-sourced claims require **citation tracing** and **fact-checking against authoritative sources**.

## Decision

### 1. Distinguish Two Claim Categories

**Category A: Externally-Sourced Claims**
- Factual assertions about the world (populations, rankings, trends)
- Literature characterizations ("extensive research exists on X")
- Research gap claims ("Y has received little attention")
- Historical claims
- Claims attributed to cited sources

**Category B: Study-Generated Claims**
- Methodological descriptions of this study
- Statistical results produced by this study
- Interpretations and conclusions from this study's analysis

### 2. Prioritize Category A for Manifest Completeness
Category A claims are the primary verification target because:
- They can be wrong or outdated without the author knowing
- They require external fact-checking
- Citation-to-claim mapping reveals whether sources actually support the claims made
- Unsupported or poorly-supported claims weaken the paper's foundation

### 3. Track Citation-to-Claim Relationships
For each externally-sourced claim, capture:
- `citation_keys`: Which citations are invoked to support this claim
- `support_type`: How the citation supports the claim (direct quote, paraphrase, interpretation)
- `verification_status`: unverified, verified, disputed, outdated
- `verification_notes`: Specific findings from fact-checking

### 4. Flag Claims Requiring Current Data
Claims about current facts (populations, rankings) should cite the most current authoritative source, not secondary sources that may be outdated. Flag claims where:
- A secondary source is cited for a fact available from primary sources
- The cited source is more than 2 years old for time-sensitive data
- Census/official data exists but isn't cited

## Consequences

### Positive
- Complete coverage of verifiable claims
- Clear mapping of citations to specific claims they support
- Identification of unsupported or weakly-supported claims
- Detection of outdated facts before publication
- Improved citation hygiene

### Negative
- Larger claim manifest (more granular extraction)
- More complex extraction process (must identify external vs. study-generated)
- Additional verification work for externally-sourced claims

## Implementation Notes

### Revised Diagnosis: What's Actually Missing

Upon detailed review, the claim manifest **does contain** the externally-sourced claims identified above:
- C0001: "North Dakota has a population of approximately 780,000" ✓
- C0002: "North Dakota ranks 47th among U.S. states in population" ✓
- C0005: "Natural population growth has stagnated in North Dakota" ✓
- C0006: "Domestic out-migration persists in North Dakota" ✓
- C0007: "International migration has emerged as a critical...component" ✓
- C0008: "The Great Plains has experienced sustained population decline..." ✓
- C0062: "The demographic literature has extensively examined migration to gateway states" ✓
- C0063: "Small peripheral states have received comparatively little rigorous empirical attention" ✓

The argument map also includes these claims in G005 (A0021-A0029), properly structured with grounds and warrants.

**The actual gap is citation-to-claim linking, not claim extraction.**

Current schema:
```json
{"claim_id": "C0001", "claim_text": "North Dakota has a population of approximately 780,000",
 "support_primary": "external_data", "support_alternative": "citation"}
```

The `support_primary` and `support_alternative` fields describe *what type* of evidence would be appropriate, but **not which specific citation is actually invoked** in the paper text.

### The Real Question: "What is Wilson et al. providing?"

When the user asks this, they want to know:
1. Which claims in the paper are supported by the Wilson et al. citation?
2. Is Wilson et al. the right source for those claims, or should we cite primary data?
3. Is the information from Wilson et al. still current?

We cannot answer these questions because we don't have:
- `citation_keys`: Which BibTeX keys appear in the same sentence/paragraph
- `citation_type`: Whether the claim is a direct quote, paraphrase, or interpretation
- `source_category`: Whether the claim requires external verification or is study-generated

### Required Schema Update

```json
{
  "claim_id": "C0001",
  "claim_text": "North Dakota has a population of approximately 780,000",
  "claim_type": "descriptive",
  "source_category": "external",           // NEW: external | study_generated
  "citation_keys": ["wilson2022"],         // NEW: BibTeX keys from paper
  "citation_accuracy": "unverified",       // NEW: see values below
  "support_primary": "external_data",
  "support_alternative": "citation",
  "verification_status": "unverified",     // NEW: unverified | verified | disputed | outdated
  "verification_notes": ""                 // NEW: specific findings
}
```

**`citation_accuracy` values** (for tracking whether citations actually support claims):
- `unverified`: Citation present but not yet checked against source
- `verified`: Citation checked and accurately supports the claim
- `partial`: Citation supports part of the claim but not all
- `unsupported`: Citation does not actually support this claim
- `uncited`: No citation provided for an external claim (needs one)

This avoids assuming citations are accurate just because they're present.

### Second Gap: Orphan Claims Not in Argument Map

Analysis reveals **138 claims (31%)** are not linked to any argument node:
- Appendix: 39 orphan claims
- Data and Methods: 45 orphan claims
- Results: 27 orphan claims
- Conclusion: 15 orphan claims
- Discussion: 12 orphan claims

These claims exist in the manifest but don't appear in the HTML viewer because the argument map only references 314 of 452 claims.

**Design decision needed**: How should orphan claims appear in the visualization?

Option A: **Standalone nodes** - Show orphan claims as unconnected nodes, making visible which claims lack argumentative support
Option B: **Link to existing groups** - Identify which argument group each orphan claim belongs to and add appropriate role
Option C: **Separate "unstructured claims" section** - Keep orphans visible but distinct from structured arguments

**Recommendation**: Option A (standalone nodes) best serves the goal of "making it easier to visualize and understand" - seeing which claims float free helps identify:
- Claims that need supporting evidence
- Evidence that needs to be linked to claims
- Patterns where one piece of evidence supports multiple claims
- Warrants that are reused across arguments

### Viewer Enhancement Needed

The HTML viewer should show:
1. All 452 claims (not just the 314 currently linked)
2. Clear visual distinction between:
   - Structured arguments (claim + grounds + warrant)
   - Orphan claims (no argumentative structure)
   - Shared evidence (grounds used by multiple claims)
   - Shared warrants (reasoning reused across arguments)
3. Filter/highlight for:
   - Externally-sourced claims (Category A - verification priority)
   - Study-generated claims (Category B)
   - Claims with `citation_accuracy: unverified`

### Process Changes Required

1. **Schema update**: Add `source_category`, `citation_keys`, `citation_accuracy`, `verification_status`, `verification_notes`
2. **Citation extraction pass**: Re-read paper to match citations to claims (without assuming accuracy)
3. **Orphan claim handling**: Add orphan claims to argument_map.jsonl as standalone nodes
4. **Viewer update**: Modify build_viewer.py to show orphan claims and enable filtering
5. **Cross-reference analysis**: Identify shared grounds/warrants across argument groups

## Related Documents
- `claims/claims_manifest.jsonl` - needs schema update for citation_keys
- `argument_map/argument_map.jsonl` - needs orphan claims added as nodes
- `argument_map/build_viewer.py` - needs update to show all claims and enable filtering
- `citation_audit/citation_entries.jsonl` - already has 73 BibTeX entries to cross-reference
- `claims/extract_claims.py` - needs prompt revision to extract citation context
- `AI_AGENT_GUIDE.md` - needs updated extraction rules
