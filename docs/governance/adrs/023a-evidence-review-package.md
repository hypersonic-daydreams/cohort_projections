# ADR-023a: Evidence Review Package

## Purpose

This document specifies the extraction of citation audit, claims analysis, and argument mapping features into a standalone `evidence-review` Python package.

**Parent ADR**: [ADR-023: Package Extraction Strategy](023-package-extraction-strategy.md)

## Package Overview

| Attribute | Value |
|-----------|-------|
| **Package Name** | `evidence-review` |
| **Repository** | `~/workspace/libs/evidence_review` |
| **Python Module** | `evidence_review` |
| **Primary Purpose** | Quality assurance for written documents: citation integrity, claim support verification, argument structure analysis |

## Scope

### Document Types Supported

The package supports quality review for:
- Academic journal articles
- White papers
- Internal reports and memos
- Press releases
- Policy briefs
- Grant proposals
- Any persuasive or evidence-based writing

### Feature Modules

The package contains three integrated feature modules:

#### 1. Citation Audit (`evidence_review.citations`)

**Source**: `sdc_2024_replication/scripts/.../claim_review/v3_phase3/citation_audit/`

**Capabilities**:
- Parse LaTeX source files (following `\input`, `\include`, `\subfile`, `\import`)
- Extract citation keys from LaTeX using configurable patterns
- Parse BibTeX files with full handling of nested braces, string macros, field concatenation
- Validate APA 7th Edition requirements per entry type
- Map 22+ BibTeX entry types to canonical APA categories
- Generate multi-format reports (JSON, Markdown, HTML, JSONL)
- Support structured fix application via JSONL input
- CI/CD integration via strict mode exit codes

**Files to Extract**:
- `check_citations.py` (1,595 lines) - Core engine
- `citation_entry_schema.json` - Schema for APA completeness audit
- `citation_fixes_schema.json` - Schema for citation corrections
- `README.md`, `AI_AGENT_GUIDE.md` - Documentation

**Dependencies**: Python stdlib only (argparse, json, logging, re, pathlib, typing, collections, datetime)

#### 2. Claims Analysis (`evidence_review.claims`)

**Source**: `sdc_2024_replication/scripts/.../claim_review/v3_phase3/claims/`

**Capabilities**:
- LLM-assisted semantic claim extraction framework
- Claim type classification (descriptive, comparative, causal, forecast, methodological, definition, normative)
- Section-aware parsing (Abstract, Introduction, Methods, Results, Discussion, Conclusion, Appendix)
- Text chunking for parallel agent processing (~1500-2000 chars)
- Claim manifest with full source location tracking
- Support types (primary/alternative) and priority classification
- Status workflow (unassigned -> assigned -> in_review -> verified/disputed/needs_revision)
- Cross-referencing to argument IDs for traceability

**Files to Extract**:
- `extract_claims.py` - LLM-assisted extraction framework
- `build_section_claims.py` - Generalized section parser
- `qa_claims.py` - Quality assurance checks
- `claim_schema.json` - Claim manifest schema
- `claim_guidelines.md` - Extraction guidelines

**Dependencies**: Python stdlib only (argparse, json, sys, dataclasses, pathlib, typing)

#### 3. Argumentation Mapping (`evidence_review.argumentation`)

**Source**: `sdc_2024_replication/scripts/.../claim_review/v3_phase3/argument_map/`

**Capabilities**:
- Toulmin model implementation (Claim, Grounds, Warrant, Backing, Qualifier, Rebuttal)
- Argument graph construction with supports/rebuts relationships
- Graphviz DOT graph generation with role-specific visual styling
- Interactive HTML viewer generation
- Cross-referencing to claims manifest via claim IDs
- Status workflow (draft -> reviewed -> reconciled)
- Group-based argument organization

**Files to Extract**:
- `generate_argument_graphs.py` (175+ lines) - Graphviz DOT generation
- `map_section_arguments.py` - LLM guidance for mapping
- `build_viewer.py` - Interactive viewer generation
- `argument_schema.json` - Toulmin structure schema
- `argument_guidelines.md` - Mapping guidelines
- `ARGUMENTATION_METHOD.md` - Toulmin framework documentation

**Dependencies**: Python stdlib for core logic; graphviz (system binary) for visualization

### Data Flow Between Modules

```
                    ┌─────────────────┐
                    │   LaTeX/BibTeX  │
                    │     Source      │
                    └────────┬────────┘
                             │
                             ▼
                ┌────────────────────────┐
                │    citation_audit      │
                │  (APA 7th validation)  │
                └────────────┬───────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ citation_keys   │◄──────────────┐
                    │    (JSONL)      │               │
                    └─────────────────┘               │
                                                      │
    ┌─────────────────┐                               │
    │  Document Text  │                               │
    │   (by section)  │                               │
    └────────┬────────┘                               │
             │                                        │
             ▼                                        │
    ┌─────────────────┐                               │
    │     claims      │───── citation_keys field ─────┘
    │  (extraction)   │
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │ claims_manifest │
    │    (JSONL)      │
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │  argumentation  │◄──── claim_ids linkage
    │   (Toulmin)     │
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │ argument_map    │
    │ (JSONL + DOT)   │
    └─────────────────┘
```

### Interchange Format

All modules use JSONL (JSON Lines) as the canonical data interchange format:
- `citation_entries.jsonl` - Citation audit results
- `claims_manifest.jsonl` - Extracted claims
- `argument_map.jsonl` - Argument nodes

JSON Schema files define structure for each format.

## Package Structure

```
evidence_review/
├── pyproject.toml
├── uv.lock
├── .envrc
├── .gitignore
├── README.md
├── AGENTS.md                          # AI agent guidance
├── evidence_review/
│   ├── __init__.py
│   ├── citations/
│   │   ├── __init__.py
│   │   ├── auditor.py                 # Core audit engine
│   │   ├── parsers.py                 # LaTeX/BibTeX parsing
│   │   ├── apa7.py                    # APA 7th Edition rules
│   │   ├── reporters.py               # Multi-format reporting
│   │   └── schemas/
│   │       ├── citation_entry.json
│   │       └── citation_fixes.json
│   ├── claims/
│   │   ├── __init__.py
│   │   ├── extractor.py               # Claim extraction
│   │   ├── chunker.py                 # Text chunking
│   │   ├── qa.py                      # Quality assurance
│   │   └── schemas/
│   │       └── claim.json
│   ├── argumentation/
│   │   ├── __init__.py
│   │   ├── toulmin.py                 # Toulmin model
│   │   ├── graph_builder.py           # DOT graph generation
│   │   ├── viewer.py                  # HTML viewer
│   │   └── schemas/
│   │       └── argument.json
│   └── cli/
│       ├── __init__.py
│       ├── audit_citations.py         # CLI: check-citations
│       ├── extract_claims.py          # CLI: extract-claims
│       └── map_arguments.py           # CLI: map-arguments
├── docs/
│   ├── citations.md
│   ├── claims.md
│   ├── argumentation.md
│   └── ai_agent_guide.md
└── tests/
    ├── unit/
    │   ├── test_bibtex_parser.py
    │   ├── test_latex_parser.py
    │   ├── test_apa7_rules.py
    │   └── test_toulmin.py
    ├── integration/
    │   └── test_full_workflow.py
    └── fixtures/
        ├── sample.tex
        ├── sample.bib
        └── expected_outputs/
```

## CLI Entry Points

The package will expose three CLI commands via pyproject.toml entry points:

```toml
[project.scripts]
check-citations = "evidence_review.cli.audit_citations:main"
extract-claims = "evidence_review.cli.extract_claims:main"
map-arguments = "evidence_review.cli.map_arguments:main"
```

## API Design

### Citation Audit API

```python
from evidence_review.citations import CitationAuditor, APA7Validator

# Initialize auditor
auditor = CitationAuditor(
    latex_root="article/main.tex",
    bibtex_files=["references.bib"],
    citation_patterns=[r"\\cite\{([^}]+)\}", r"\\citep\{([^}]+)\}"]
)

# Run audit
results = auditor.audit()

# Validate against APA 7th
validator = APA7Validator()
issues = validator.validate(results.entries)

# Generate reports
results.to_json("audit_report.json")
results.to_markdown("audit_report.md")
results.to_html("audit_report.html")
results.to_jsonl("citation_entries.jsonl")
```

### Claims Analysis API

```python
from evidence_review.claims import ClaimExtractor, ClaimManifest

# Initialize extractor
extractor = ClaimExtractor(
    chunk_size=1800,
    sections=["introduction", "methods", "results", "discussion"]
)

# Extract claims from text
claims = extractor.extract(
    text="The population of North Dakota increased by 15% between 2010 and 2020...",
    section="results",
    page_number=12
)

# Build manifest
manifest = ClaimManifest()
manifest.add_claims(claims)
manifest.link_citations(citation_keys=["census2020", "smith2021"])
manifest.to_jsonl("claims_manifest.jsonl")
```

### Argumentation API

```python
from evidence_review.argumentation import ArgumentMapper, ToulminNode

# Initialize mapper
mapper = ArgumentMapper(claims_manifest="claims_manifest.jsonl")

# Add argument nodes
claim = mapper.add_node(
    role="claim",
    text="North Dakota's population growth was driven by oil industry expansion.",
    claim_ids=["C0001", "C0002"]
)

grounds = mapper.add_node(
    role="grounds",
    text="Employment in oil extraction increased 340% between 2008 and 2014.",
    supports=[claim.id]
)

# Generate visualization
mapper.to_dot("argument_graph.dot")
mapper.to_png("argument_graph.png")  # Requires graphviz
mapper.to_html("argument_viewer.html")
```

## Migration Plan

### Phase 1: Package Scaffolding (30 min)

1. Create repository at `~/workspace/libs/evidence_review`
2. Initialize with `uv init`
3. Create `.envrc` per REPOSITORY_STANDARDS.md
4. Set up basic package structure

### Phase 2: Code Migration (2-3 hours)

1. **Citation Audit**: Move and refactor `check_citations.py` into modular structure
2. **Claims**: Move extraction scripts and schemas
3. **Argumentation**: Move graph generation and viewer code
4. **Schemas**: Consolidate JSON schemas into package

### Phase 3: API Refinement (1 hour)

1. Design clean public API for each module
2. Add `__init__.py` exports
3. Create CLI entry points
4. Write comprehensive docstrings

### Phase 4: Testing (1 hour)

1. Port existing tests (`tests/test_tools/test_citation_audit.py`)
2. Add unit tests for new module boundaries
3. Create integration tests for full workflow
4. Verify all tests pass

### Phase 5: Documentation (30 min)

1. Create package README.md
2. Create AGENTS.md for AI agent guidance
3. Create per-module documentation
4. Add usage examples

### Phase 6: Integration (30 min)

1. Add package to cohort_projections as editable dependency
2. Update any imports in cohort_projections
3. Verify cohort_projections tests pass
4. Update REPOSITORY_INVENTORY.md

## Testing Strategy

### Unit Tests

- BibTeX parser: field extraction, nested braces, string macros
- LaTeX parser: citation key extraction, include following
- APA 7th rules: required/recommended fields per entry type
- Toulmin model: node creation, relationship validation

### Integration Tests

- Full audit workflow: LaTeX + BibTeX -> reports
- Full claim workflow: text -> manifest
- Full argument workflow: claims -> graph visualization
- Cross-module: citation keys linked to claims linked to arguments

### Fixtures

Provide sample documents covering edge cases:
- Multi-file LaTeX projects
- Complex BibTeX with strings and cross-references
- Various document sections
- Multiple argument structures

## Dependencies

### Required (Python stdlib)

- `argparse` - CLI argument parsing
- `json` - JSON/JSONL handling
- `logging` - Logging
- `re` - Regular expressions
- `pathlib` - Path handling
- `typing` - Type hints
- `collections` - Data structures
- `datetime` - Timestamps
- `dataclasses` - Data classes

### Optional (External)

- `graphviz` (system binary) - Graph visualization (graceful degradation if absent)

### Development Only

- `pytest` - Testing
- `pytest-cov` - Coverage
- `ruff` - Linting
- `mypy` - Type checking

## Success Criteria

The extraction is complete when:

1. [ ] Package is installable via `uv add --editable ../libs/evidence_review`
2. [ ] All three CLI commands work: `check-citations`, `extract-claims`, `map-arguments`
3. [ ] All unit and integration tests pass
4. [ ] cohort_projections can import and use all functionality
5. [ ] cohort_projections tests still pass
6. [ ] Documentation is complete and accurate
7. [ ] Package is listed in REPOSITORY_INVENTORY.md

## Future Enhancements

After initial extraction, consider:

1. **Document format expansion**: Support for DOCX, Google Docs, plain text
2. **Citation style expansion**: Support for Chicago, MLA, IEEE beyond APA 7th
3. **LLM integration**: Direct integration with Claude API for claim extraction
4. **Web interface**: Flask/FastAPI wrapper for non-CLI usage
5. **VS Code extension**: Real-time citation and claim checking

## Related Documents

- [ADR-023](023-package-extraction-strategy.md): Parent extraction strategy
- [ADR-019](019-argument-mapping-claim-review-process.md): Original claim review process design
- [check_citations.py](../../sdc_2024_replication/scripts/statistical_analysis/journal_article/claim_review/v3_phase3/citation_audit/check_citations.py): Source code
- [claim_schema.json](../../sdc_2024_replication/scripts/statistical_analysis/journal_article/claim_review/v3_phase3/claims/claim_schema.json): Claim schema
- [argument_schema.json](../../sdc_2024_replication/scripts/statistical_analysis/journal_article/claim_review/v3_phase3/argument_map/argument_schema.json): Argument schema
