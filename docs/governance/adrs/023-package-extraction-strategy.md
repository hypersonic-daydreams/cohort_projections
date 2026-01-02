# ADR-023: Package Extraction Strategy

## Status
Accepted

## Date
2026-01-02

## Context

The cohort_projections project has grown to include features that are:
1. **Decoupled** from the core projection functionality
2. **Reusable** across other projects in the workspace
3. **Generic** enough to serve broader use cases

This creates maintenance overhead and missed opportunities for code reuse. The workspace already contains infrastructure to support shared packages (`libs/` category, REPOSITORY_STANDARDS.md).

### Requirements

- Extracted packages must be importable by cohort_projections and other projects
- Extraction must not break existing functionality
- Packages must follow workspace standards (uv, direnv, pyproject.toml)
- Documentation and tests must transfer with the code

### Challenges

- Determining appropriate package boundaries
- Managing cross-repository coordination
- Avoiding premature abstraction
- Ensuring discoverability of extracted packages

## Decision

### Decision 1: Extract Three Feature Sets as Separate Packages

**Decision**: Create three new packages in the `~/workspace/libs/` category:

| Package | Repository | Module | Source Location | Purpose |
|---------|------------|--------|-----------------|---------|
| `evidence-review` | `evidence_review` | `evidence_review` | `sdc_2024_replication/.../claim_review/v3_phase3/` | Citation audit, claims analysis, argument mapping |
| `project-utils` | `project_utils` | `project_utils` | `cohort_projections/utils/{config_loader,logger}.py` | Configuration loading and logging setup |
| `codebase-catalog` | `codebase_catalog` | `codebase_catalog` | `scripts/intelligence/`, `scripts/hooks/` | Repository scanning, code inventory, pre-commit hooks |

**Rationale**:
- All three feature sets have **zero coupling** to the core projection engine
- All use only Python stdlib or minimal external dependencies
- All have clear, well-defined boundaries
- Workspace infrastructure (`libs/`, REPOSITORY_STANDARDS.md) already supports this pattern

**Alternatives Considered**:
- **Monorepo subpackages**: Rejected; doesn't enable reuse across other repositories
- **Micro-packages (one per feature)**: Rejected for academic-article-review; citation/claims/arguments share a domain
- **Leave in place**: Rejected; misses reuse opportunities and increases maintenance burden

### Decision 2: Defer demographic-calcs Extraction

**Decision**: Do NOT extract demographic utility functions at this time.

**Rationale**:
- These functions (`cohort_projections/utils/demographic_utils.py`) are likely to evolve significantly based on this project's needs
- Premature extraction would create API stability pressure that could constrain development
- Future extraction is anticipated once the interfaces stabilize

**Trigger for Reconsideration**:
- When two or more projects need the same demographic functions
- When the function signatures have been stable for 6+ months

### Decision 3: Expanded Scope for academic-article-review

**Decision**: The academic-article-review package will support multiple document types, not just academic papers.

**Document Types**:
- Academic journal articles (original scope)
- White papers
- Internal reports
- Press releases
- Policy briefs
- Any document requiring citation integrity and claim support verification

**Rationale**:
- The underlying workflows (citation audit, claim extraction, argument mapping) are document-type agnostic
- Expanding scope increases reuse value across the organization
- The Toulmin model of argumentation applies to any persuasive writing

### Decision 4: Phased Extraction Approach

**Decision**: Extract packages in phases, validating workflow before proceeding.

| Phase | Package | Priority | Est. Effort |
|-------|---------|----------|-------------|
| 1a | `project-utils` | Highest | 1-2 hours |
| 1b | `evidence-review` | High | 3-4 hours |
| 2 | `codebase-catalog` | Medium | 2-3 hours |

**Rationale**:
- Phase 1a validates the extraction workflow with the simplest package
- Phase 1b addresses the original motivation (citation/claims features)
- Phase 2 proceeds after workflow is proven

## Consequences

### Positive

1. **Separation of Concerns**: Citation tooling code is no longer mixed with projection code
2. **Reusability**: Packages can serve other demography projects (popest, nd_population, hhs_stats, etc.)
3. **Testing Isolation**: Each package has focused test suites
4. **Maintenance Clarity**: Bug location is obvious based on package boundaries
5. **Onboarding**: New team members understand system boundaries more easily

### Negative

1. **Coordination Overhead**: Changes may require updates across repositories
2. **Development Friction**: Editing library features requires managing multiple repos
3. **Discovery Challenge**: New developers must learn which packages exist

### Risks and Mitigations

**Risk**: Breaking changes in extracted packages cascade across dependent projects
- **Mitigation**: Use semantic versioning; dependent projects pin versions via `uv.lock`

**Risk**: Development friction slows iteration
- **Mitigation**: Use editable installs during development (`uv add --editable ../libs/package`)

**Risk**: Packages become orphaned or forgotten
- **Mitigation**: Update REPOSITORY_INVENTORY.md; add to category documentation

**Risk**: Premature abstraction creates unusable APIs
- **Mitigation**: Only extract features already proven in cohort_projections; defer evolving code

## Implementation Notes

### Extraction Process (Per Package)

1. **Create package repository** in `~/workspace/libs/`
2. **Scaffold structure** following REPOSITORY_STANDARDS.md
3. **Move source code** preserving git history where practical
4. **Transfer tests** and documentation
5. **Update cohort_projections** to import from package
6. **Add to REPOSITORY_INVENTORY.md**
7. **Validate** all tests pass in both locations

### Dependency Management

During active development:
```bash
cd ~/workspace/demography/cohort_projections
uv add --editable ../../libs/academic-article-review
```

For reproducibility/sharing:
```bash
uv add git+https://github.com/hypersonic-daydreams/academic-article-review@v0.1.0
```

### Testing Strategy

- Each extracted package must have its own test suite
- cohort_projections integration tests verify packages work together
- CI/CD runs tests on both package changes and dependent project changes

## References

1. **REPOSITORY_STANDARDS.md**: Workspace-wide standards for Python projects
2. **REPOSITORY_INVENTORY.md**: Catalog of all workspace repositories
3. **ADR-020/020a**: Example of parent-child ADR pattern

## Revision History

- **2026-01-02**: Initial version (ADR-023) - Package extraction strategy decision

## Related ADRs

- [ADR-023a](023a-evidence-review-package.md): Evidence Review package specification
- [ADR-023b](023b-project-utils-package.md): Project Utils package specification
- [ADR-023c](023c-codebase-catalog-package.md): Codebase Catalog package specification
- [ADR-005](005-configuration-management-strategy.md): Configuration management (being extracted)
- [ADR-009](009-logging-error-handling-strategy.md): Logging strategy (being extracted)
- [ADR-019](019-argument-mapping-claim-review-process.md): Claim review process (being extracted)
