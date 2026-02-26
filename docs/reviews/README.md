# Review Documents

Analytical reviews and quality assurance documents for the North Dakota Population Projection System. These are working documents that identify issues, validate methodology, and inform ADR decisions.

## Index

| Date | Review | Scope | Related ADRs |
|------|--------|-------|-------------|
| 2026-02-17 | [Export Text Review](2026-02-17-export-text-review.md) | Methodology text accuracy in export workbooks | ADR-037 |
| 2026-02-17 | [Methodology Text Audit](2026-02-17-methodology-text-audit.md) | Production code vs. methodology text alignment | ADR-001, ADR-002, ADR-035, ADR-036 |
| 2026-02-17 | [Vintage 2025 Census Data Analysis](2026-02-17-vintage-2025-census-data-analysis.md) | V2025 PEP data quality and incorporation | ADR-039 |
| 2026-02-17 | [Bakken Migration Dampening Review](2026-02-17-bakken-migration-dampening-review.md) | Oil county growth rates and dampening effectiveness | ADR-040 |
| 2026-02-18 | [Projection Output Sanity Check](2026-02-18-projection-output-sanity-check.md) | Full output review: 4 scenarios, 53 counties, demographic plausibility | ADR-037, ADR-039, ADR-040 |
| 2026-02-18 | [Sanity Check Investigation Reports](2026-02-18-sanity-check-investigations/README.md) | Deep investigation of 7 findings; all P0-P2 fixes implemented | ADR-042, ADR-043, ADR-044, ADR-045, ADR-046 |
| 2026-02-23 | [Projection Output Review and ADR Assessment](2026-02-23-projection-output-review.md) | Full 3-scenario review; ADR decisions for 051, 052, 036, 033; new finding on state-county aggregation | ADR-033, ADR-036, ADR-045, ADR-051, ADR-052, ADR-054 |
| 2026-02-23 | [Post-ADR-054 Sanity Check](2026-02-23-post-adr054-sanity-check.md) | 9-category validation: structural, SDC comparison, calibration, plausibility, age structure, extremes, scenario spread, race, historical trends | ADR-049, ADR-050, ADR-052, ADR-053, ADR-054 |
| 2026-02-23 | [Ward & Grand Forks Institutional Population Review](2026-02-23-ward-grand-forks-institutional-population-review.md) | Military base and college/university population handling in Ward and Grand Forks counties | ADR-049, ADR-052 |

## Conventions

- File naming: `YYYY-MM-DD-topic-name.md`
- Each review includes a metadata header (Date, Reviewer, Scope, Status)
- Reviews that lead to decisions should cross-reference the resulting ADR
- ADRs that are informed by reviews should back-link in a "Related Reviews" section

---

**Last Updated**: 2026-02-23
