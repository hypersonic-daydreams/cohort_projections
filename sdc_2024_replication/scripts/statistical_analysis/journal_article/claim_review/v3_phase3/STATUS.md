# Claim Review Plan and Status - v5_p305_complete Draft

## Objective
Create a complete, claim-level manifest for the v5_p305_complete PDF, map the argument structure, annotate preferred evidence types, and track verification outcomes for systematic revision.

## Workflow Checklist
- [x] Scaffold claim review workspace
- [x] Snapshot PDF and record metadata hash
- [x] Scaffold argument map workspace and schema
- [x] Extract text with page anchors into `extracted/`
- [x] Build claim manifest for Abstract + Introduction (pilot granularity check)
- [x] Build claim manifest for remaining sections
- [x] Extend repeatable claim-parsing to all sections (`build_section_claims.py`)
- [x] Run QA layer on all sections
- [ ] Add support-type annotations (primary + alternative) for all claims
- [ ] Build argument map for Abstract + Introduction (pilot structure check)
- [ ] Build argument map for remaining sections
- [ ] Export Graphviz graphs (per-argument and full-paper)
- [ ] Run citation audit (APA 7th completeness) and address missing/orphaned references
- [ ] Assign claims to agents for verification
- [ ] Collect evidence notes in `evidence/` and update statuses
- [ ] Summarize results in `exports/claims_dashboard.md`

## Notes
- Canonical data file: `claims/claims_manifest.jsonl`
- Support annotation question: "What two types of evidence or support would be most appropriate to support this claim?"
  - `support_primary` = most appropriate or robust support
  - `support_alternative` = alternative support if primary is not feasible

## Session Notes (2025-12-31)
- Text extraction: `mutool draw -F txt -o extracted/page-%d.txt source/article_draft_v5_p305_complete.pdf 1-61` completed; 61 page files present in `extracted/`.
- Draft claim manifest generated: `claims/claims_manifest.jsonl` currently has 1332 claims across sections (Abstract 15, Introduction 49, Data and Methods 227, Results 465, Discussion 165, Conclusion 38, Appendix 373).
- Introduction claims regenerated via a reusable parser script: `claims/build_intro_claims.py` (pages 3-5, 49 claims, pdf_page anchors corrected to 3/4/5).
  - Re-run from `v3_phase3/`: `python claims/build_intro_claims.py --write` (venv required).
- QA layer added: `claims/qa_claims.py` validates manifest structure, flags likely parsing artifacts, and can compare Introduction claims to parser output.
  - Intro QA run: `python claims/qa_claims.py --section Introduction --expected-pages 3 4 5 --compare-intro` (0 errors, 3 heuristic warnings for sentence starts such as "With"/"Because").
- Parsing rules applied for Introduction (repeatable in the script):
  - Drop repeated page headers and numeric-only lines.
  - Trim line-break hyphenation (remove trailing `-` and join), and join em-dash line breaks without inserting a space.
  - Strip list numbering (`1.`–`4.`) and convert list lead-in colons to sentence boundaries.
  - Protect common abbreviations (e.g., `U.S.`, `et al.`) during sentence splitting.
  - Assign `pdf_page` by line-offset tracking so cross-page sentences anchor to the page where they begin.
- Pitfalls/known limits noted during the Introduction pass:
  - Hyphenation rule assumes trailing `-` is a line-break artifact; if a true compound hyphen falls at line end, it will be merged incorrectly.
  - Abbreviation protection uses a placeholder character; literal tildes would be restored to periods (not observed in the PDF).
  - Claim-type labels are still heuristic and should be audited later; research questions are forced to `methodological`.
- Known issues in the current manifest (needs cleanup):
  - Some claims are still too coarse and should be split (multi-clause sentences, especially in Abstract and Results narrative).
  - Line-break hyphenation is partly fixed (common prefixes preserved), but some words still appear broken (e.g., "grav-ity" in a few claims). Extend the hyphenation preserve list or post-clean.
  - Plot/axis labels mostly filtered, but at least one mixed claim remains (e.g., a Discussion claim that includes figure axis text from page 32). Needs additional filtering for figure labels before sentence assembly.
  - Claim types are heuristic-only; several labels are likely off (e.g., some descriptive claims tagged as methodological/forecast).
- Draft manifest for non-Introduction sections was produced via ad-hoc Python parsing in the shell (not saved to a script). If repeating, re-run extraction + rebuild with a dedicated script or rework the parser in-session.

## Session Notes (2025-12-31 - Continued)

### Generalized Section Parser (`build_section_claims.py`)

A generalized claim parser was created that extends `build_intro_claims.py` to all sections:

```bash
# Dry-run to preview claims for any section
python claims/build_section_claims.py --section "Data and Methods" --dry-run

# Update manifest (preserves existing claim IDs)
python claims/build_section_claims.py --section Introduction --write
```

**Section Configuration (page ranges):**
| Section | Pages | Manifest Claims | Parser Claims |
|---------|-------|-----------------|---------------|
| Abstract | 1-2 | 15 | 12 |
| Introduction | 3-5 | 49 | 49 (matched) |
| Data and Methods | 5-19 | 227 | 185 |
| Results | 19-31 | 465 | 238 |
| Discussion | 31-45 | 165 | 124 |
| Conclusion | 45-47 | 38 | 35 |
| Appendix | 51-61 | 373 | 103 |

**Claim Count Discrepancies Explained:**
- **Introduction matches exactly** (49 claims) - parser validated against existing
- Other sections have higher manifest counts because ad-hoc parsing:
  - Broke table data into individual cell claims (e.g., "Table 5: Liberia LQ is 40.83")
  - Split sentences at semicolons and clause boundaries more aggressively
  - Included figure axis labels and notes as separate claims
- The generalized parser consolidates these into full-sentence claims

### Enhanced QA Layer (`qa_claims.py`)

Extended QA capabilities to support all sections:

```bash
# Basic QA on all sections
python claims/qa_claims.py --all-sections

# QA with parser comparison (shows discrepancies)
python claims/qa_claims.py --all-sections --compare-section

# Single section QA
python claims/qa_claims.py --section "Results" --compare-section
```

**QA Summary (2025-12-31):**
- Total claims: 1332
- Errors: 0
- Warnings: 392 (heuristic flags)

**Common warning categories:**
1. Claims not ending with sentence punctuation (table cells, list items)
2. Claims starting with lowercase (continuation fragments)
3. Claims starting with conjunctions ("With", "Because", "And")
4. Possible figure/axis label fragments ("ACF", "Panel", "Density")
5. High numeric density (table data claims)
6. Duplicate claim text (standard notation like "*** p < 0.001")

### Parsing Rules (Applied to All Sections)

1. **Header Filtering:**
   - Drop repeated document title
   - Drop standalone page numbers
   - Drop numeric-only lines (years, table values)

2. **Line Normalization:**
   - Remove list numbering (`1.`-`4.`)
   - Convert list lead-in colons to periods
   - Join line-break hyphenation intelligently

3. **Sentence Splitting:**
   - Protect abbreviations: U.S., et al., e.g., i.e., etc., Fig., Eq., Vol., No., pp.
   - Split on sentence-ending punctuation followed by whitespace
   - Track character offsets for page assignment

4. **Page Assignment:**
   - Assign pdf_page based on where sentence begins (not ends)
   - Track line offsets through merged text blob

5. **Claim Type Inference (Heuristic):**
   - `methodological`: analysis, method, study, framework, regression
   - `causal`: effect, impact, policy-associated, difference-in-differences
   - `comparative`: relative, higher, lower, exceeds, compared to
   - `forecast`: projection, scenario, future, Monte Carlo
   - `normative`: should, must, need to
   - `definition`: is defined, refers to, means
   - `descriptive`: default fallback

### Known Pitfalls and Limitations

1. **Hyphenation at line breaks:**
   - Assumes trailing `-` is a line-break artifact
   - True compound hyphens at line end will be incorrectly merged
   - Workaround: Check for lowercase continuation to decide

2. **Abbreviation protection:**
   - Uses `~` placeholder; literal tildes would be incorrectly restored
   - Not observed in this document

3. **Table/Figure content:**
   - Parser filters obvious axis labels but some mixed claims remain
   - Existing manifest preserves fine-grained table claims intentionally
   - Design choice: Keep granular table claims for verification purposes

4. **Claim type labels:**
   - Purely heuristic; should be manually audited
   - Research questions forced to `methodological`
   - Some misclassifications expected (e.g., descriptive tagged as forecast)

5. **Cross-page sentences:**
   - Anchored to the page where the sentence begins
   - May cause slight page assignment discrepancies vs visual PDF

### Parser Design Philosophy

The generalized parser produces **sentence-level claims** while the existing manifest uses **finer-grained claims** for certain content types:

- **Sentence-level (parser default):** Each complete sentence = one claim
- **Cell-level (manifest):** Table cells and individual statistics as claims

Both approaches are valid:
- Sentence-level: Easier to verify, clearer context
- Cell-level: More granular tracking, specific value verification

The QA layer now compares both approaches to surface discrepancies without requiring exact match.

## Session Notes (2025-12-31 - LLM-Based Extraction)

### New LLM-Assisted Claim Extraction (`extract_claims.py`)

The manifest was cleared and rebuilt using an LLM-based approach that extracts claims based on **IDEAS/meaning** rather than sentence punctuation. This produces higher-quality claims that represent discrete verifiable assertions.

**New Tool:**
```bash
# List chunks for a section
python claims/extract_claims.py --section Abstract --list-chunks

# Extract claims from specific chunk (for parallel agents)
python claims/extract_claims.py --section "Data and Methods" --chunk 5

# Extract all chunks at once (outputs guidance for LLM agent)
python claims/extract_claims.py --section Results --all-chunks
```

**Approach:**
1. Text is chunked by page with ~1500-2000 character chunks
2. Each chunk is processed by an LLM agent with clear extraction guidelines
3. Agents extract discrete IDEAS as claims, not raw sentences
4. Claims are assigned types: descriptive, methodological, comparative, causal, forecast, definition, normative
5. Multiple agents run in parallel for speed

**Final Claim Counts (LLM Extraction):**
| Section | Claims | ID Range |
|---------|--------|----------|
| Introduction | 60 | C0001-C0060 |
| Abstract | 30 | C0061-C0090 |
| Conclusion | 53 | C0091-C0143 |
| Data and Methods | 129 | C0144-C0272 |
| Results | 69 | C0273-C0341 |
| Discussion | 36 | C0342-C0377 |
| Appendix | 75 | C0378-C0452 |
| **TOTAL** | **452** | C0001-C0452 |

**Claim Type Distribution:**
| Type | Count |
|------|-------|
| descriptive | 188 |
| methodological | 181 |
| comparative | 20 |
| causal | 18 |
| definition | 18 |
| normative | 14 |
| forecast | 13 |

### QA Validation Results

Final QA run: **0 errors, 70 warnings**

Most warnings are acceptable stylistic variations:
- Claims not ending with periods (by design - claims are IDEAS, not sentences)
- Claims starting with "with", "because" (valid subordinate clause claims)
- Figure/axis label fragments (3 instances - manual review recommended)
- Missing expected pages in Discussion (pages 32-39 are mostly figures/tables)

### Key Improvements Over Sentence-Based Parsing

1. **Semantic granularity:** Claims represent discrete verifiable assertions, not sentence fragments
2. **Better claim typing:** LLM assigns types based on meaning, not keyword heuristics
3. **Reduced noise:** Filters table data, axis labels, and structural artifacts
4. **Consistent ID scheme:** Sequential C0001-C0452 with no gaps
5. **Parallel extraction:** 3-7 agents run simultaneously per section for speed

### Files Modified
- `claims/claims_manifest.jsonl` - Rebuilt with 452 LLM-extracted claims
- `claims/extract_claims.py` - New LLM extraction tool (created)
- `claims/qa_claims.py` - Extended with section support (existing)

## Session Notes (2025-12-31 - Support Annotations & Argument Map)

### Support Type Annotations

All 452 claims now have `support_primary` and `support_alternative` annotations indicating the most appropriate evidence types for verification.

**Support Type Distribution (Primary):**
| Type | Count |
|------|-------|
| quantitative_data | 145 |
| methodology_reference | 116 |
| citation | 79 |
| logical_inference | 41 |
| model_output | 27 |
| external_data | 18 |
| visual_evidence | 9 |
| qualitative_description | 7 |
| cross_validation | 6 |
| theoretical_framework | 4 |

### Argument Map (Abstract + Introduction)

Built Toulmin-style argument maps for Abstract and Introduction sections:

**Summary:**
- Total nodes: 59
- Argument groups: 9 (G001-G009)

**Nodes by Role:**
| Role | Count |
|------|-------|
| grounds | 18 |
| warrant | 12 |
| claim | 11 |
| backing | 8 |
| qualifier | 6 |
| rebuttal | 4 |

**Argument Groups:**
- G001: Research gap justification (5 nodes)
- G002: North Dakota's distinctive migration profile (5 nodes)
- G003: Policy impact evidence (5 nodes)
- G004: Forecast uncertainty implications (5 nodes)
- G005: Migration's demographic importance (9 nodes)
- G006: Methodological challenges (7 nodes)
- G007: Research questions and design (7 nodes)
- G008: Study contributions (7 nodes)
- G009: Scope limitations (9 nodes)

### Files Modified
- `claims/claims_manifest.jsonl` - Added support_primary and support_alternative to all 452 claims
- `argument_map/argument_map.jsonl` - Created with 59 argument nodes
