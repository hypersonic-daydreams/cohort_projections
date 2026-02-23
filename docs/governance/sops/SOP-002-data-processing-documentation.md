# SOP-002: Data Processing Script and Data Source Documentation

## Document Information

| Field | Value |
|-------|-------|
| SOP ID | 002 |
| Status | Active |
| Created | 2026-02-23 |
| Last Updated | 2026-02-23 |
| Owner | Project Lead |

---

## 1. Purpose

This SOP defines the documentation standards for data processing scripts and data source files in the cohort projection system. The goal is to ensure that every data transformation is fully traceable: a future reader (or AI agent) should be able to understand not just *what* was done but *why*, reproduce the results, and write up the methodology for publication.

**Origin**: Derived from the ADR-053 implementation workflow (2026-02-23), where ND-specific fertility and mortality rates were built from CDC WONDER and NVSR data. The documentation produced during that work became the template for this SOP.

---

## 2. Scope

### In Scope
- Module docstrings for data processing scripts (`scripts/data/build_*.py`, `scripts/data/ingest_*.py`)
- `DATA_SOURCE_NOTES.md` files in data directories (`data/raw/{category}/`)
- ADR status updates and Implementation Results sections
- Output file provenance (what produced it, when, from what inputs)

### Out of Scope
- Core engine module documentation (covered by Google-style docstrings per AGENTS.md)
- Test documentation
- Export/visualization script documentation
- Journal article or publication-ready text

---

## 3. Prerequisites

### Required Knowledge
- Familiarity with project structure (see [AGENTS.md](../../../AGENTS.md))
- Understanding of ADR format (see [ADR README](../adrs/README.md))
- Knowledge of the data source being processed

### Reference Examples
The following files exemplify the standards in this SOP:
- `scripts/data/build_nd_fertility_rates.py` — ADR-053 Part A
- `scripts/data/build_nd_survival_rates.py` — ADR-053 Part B
- `data/raw/fertility/DATA_SOURCE_NOTES.md` — ND-specific fertility section
- `data/raw/mortality/DATA_SOURCE_NOTES.md` — ND state life tables section

---

## 4. Procedure

### Phase 1: Script Module Docstring

**Objective**: Every data processing script has a self-contained docstring sufficient for methodology reproduction.

**Required elements** (in this order):

```python
"""
One-line summary of what the script does.

Created: YYYY-MM-DD
ADR: NNN (Part X — short title)    # if applicable
Author: who created/modified it

Purpose
-------
2-4 sentences explaining WHY this script exists. What problem does it solve?
What was wrong with the previous approach? Quantify the impact if possible
(e.g., "national rates understate ND fertility by ~15%").

Method
------
Numbered steps describing the processing logic. Each step should be concrete
enough that someone could reimplement from this description alone.

Key design decisions
--------------------
Bullet points explaining non-obvious choices. Each should include:
- **Bold label**: What was decided
- Why that choice was made (rationale)
- What the alternative was and why it was rejected (trade-off)

Validation results (YYYY-MM-DD)
-------------------------------
Actual computed values from the most recent run:
- Key metric 1: computed value (target: X, status: within Y%)
- Key metric 2: computed value (published: X, delta: Y)
Include enough detail that someone can assess whether results are plausible.

Inputs
------
- path/to/input/file
    Description. Source/provenance. Download date if external.

Output
------
- path/to/output/file
    Description. Row count. Schema summary.

Usage
-----
    python scripts/data/script_name.py
"""
```

**Quality checklist:**
- [ ] Could someone reimplement the method from the docstring alone?
- [ ] Are all input files listed with provenance (source URL, download date)?
- [ ] Are validation results actual numbers, not just "should be reasonable"?
- [ ] Are design decisions documented with rationale, not just stated?
- [ ] Is the ADR reference included?

---

### Phase 2: DATA_SOURCE_NOTES.md

**Objective**: Every `data/raw/{category}/` directory has a living document tracking all data files, their sources, and how they're used.

**When to update:**
- Adding new data files to a `data/raw/` directory
- Replacing an existing data pipeline with a new one
- Changing the processing methodology for existing data

**Required sections for new data entries:**

```markdown
### Section Title — Added YYYY-MM-DD

| File | Description | Rows | Download Date |
|------|-------------|------|---------------|
| filename.ext | What it contains | N | YYYY-MM-DD |

**Source:** Full URL or citation
**Query parameters:** (for web-based sources like CDC WONDER)
**Processing script:** `scripts/data/build_*.py`

**Column definitions** or **Race mapping** (if applicable):
| Source Column | Project Column | Notes |
|--------------|----------------|-------|
| ... | ... | ... |

**Validation:**
| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| ... | ... | ... | ... |
```

**When replacing an existing pipeline**, add a Historical Notes section:

```markdown
## Historical Notes

Prior to [ADR/date], the project used [old source] from [old file].
That file remains in the repository as a reference. The switch to
[new source] [quantify impact, e.g., "increased projected births by ~15%"].
```

---

### Phase 3: ADR Lifecycle Updates

**Objective**: ADRs reflect implementation reality, not just the original proposal.

**When implementing an ADR:**

1. **Update status**: "Proposed" → "Accepted" when implementation begins
2. **Add Implemented date**: `## Implemented\nYYYY-MM-DD`
3. **Update Last Reviewed**: To current date

**When implementation is complete:**

4. **Add Implementation Results section** before References:

```markdown
## Implementation Results (YYYY-MM-DD)

### Part A: [Component Name]

- **Script**: `scripts/data/build_*.py`
- **Output**: `data/raw/path/to/output.csv` (N rows)
- **Key metric**: value (target: X, status: within Y%)
- **Key metric**: value (vs. actual/published: X)

### Files Created/Modified

| File | Action | Description |
|------|--------|-------------|
| `path/to/file` | Created/Modified | Brief description |
```

5. **Correct factual errors**: If implementation reveals that assumptions in the proposal were wrong (e.g., ND mortality is above national, not below), update the Context section with corrected values and annotate what changed.

---

### Phase 4: Verification

**Objective**: Confirm documentation is complete before committing.

**Checklist:**

- [ ] Script module docstring has all required elements (Phase 1)
- [ ] `DATA_SOURCE_NOTES.md` updated for any new/changed data files (Phase 2)
- [ ] ADR status updated to reflect current state (Phase 3)
- [ ] ADR Implementation Results section added with actual metrics (Phase 3)
- [ ] Config file changes documented (if input paths changed)
- [ ] Historical Notes added if replacing an existing pipeline

---

## 5. Artifacts

### Documentation Artifacts Produced

| Artifact | Location | When Created |
|----------|----------|--------------|
| Script docstring | Module-level `"""..."""` | When creating/modifying processing scripts |
| DATA_SOURCE_NOTES.md | `data/raw/{category}/DATA_SOURCE_NOTES.md` | When adding data files |
| ADR Implementation Results | `docs/governance/adrs/NNN-*.md` | When implementation is complete |

---

## 6. Quality Gates

| Gate | Criteria | Responsible |
|------|----------|-------------|
| Docstring complete | All 9 required elements present | AI Agent / Developer |
| DATA_SOURCE_NOTES updated | New files documented with provenance | AI Agent / Developer |
| ADR lifecycle current | Status and Implementation Results reflect reality | AI Agent / Developer |
| Validation results recorded | Actual numbers, not placeholders | AI Agent / Developer |

---

## 7. Troubleshooting

### "I don't know the download date for a data file"

**Resolution**: Check git history (`git log --follow path/to/file`) or file modification time. If neither is available, use "Unknown — pre-YYYY" with the earliest plausible date.

### "The ADR proposal had incorrect assumptions"

**Resolution**: Correct the factual errors in the Context section directly. Add a note in the Implementation Results section explaining what was discovered (e.g., "Contrary to the original proposal..."). Do not leave known-incorrect statements in the ADR.

### "The script is too simple to need all 9 docstring elements"

**Resolution**: All data processing scripts that produce files consumed by the pipeline need the full docstring. Short utility scripts (< 50 lines, single function, no external data) can use a shortened version with just Purpose, Inputs, Output, and Usage.

---

## 8. Related Documentation

- [AGENTS.md Section 5](../../../AGENTS.md) — Quality Standards (references this SOP)
- [SOP-001](./SOP-001-external-ai-analysis-integration.md) — External AI Analysis Integration
- [ADR-053](../adrs/053-nd-specific-vital-rates.md) — First implementation using this standard
- [data-sources-workflow.md](../../guides/data-sources-workflow.md) — Data acquisition guide

---

## 9. Revision History

| Date | Version | Author | Changes |
|------|---------|--------|---------|
| 2026-02-23 | 1.0 | Claude Opus 4.6 / N. Haarstad | Initial version derived from ADR-053 documentation workflow |

---

*SOP Version: 1.0*
