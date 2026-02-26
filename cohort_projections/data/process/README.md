# Data Processing Module

Implementation-focused guide for `cohort_projections.data.process`.

**Last Updated:** 2026-02-26  
**Status:** Current  
**Projection Horizon:** 2025-2055

## Why This Exists

The `data.process` package transforms raw demographic source files into standardized, projection-ready datasets used by the cohort-component engine.

## Scope

- Normalize population, fertility, mortality, and migration inputs.
- Enforce project cohort conventions (age, sex, race/ethnicity, geography).
- Emit deterministic, typed data products consumed by projection scripts.

## Module Map

| Module | Responsibility | Primary Outputs |
|--------|----------------|-----------------|
| `base_population.py` | Build age x sex x race/ethnicity base matrices | Base cohort population tables |
| `fertility_rates.py` | Prepare age-specific fertility rates | ASFR tables by cohort/geography |
| `survival_rates.py` | Prepare mortality/survival inputs | Survival schedules by cohort |
| `migration_rates.py` | Prepare net migration rates | Migration rates by cohort/geography |
| `__init__.py` | Package exports | Stable import surface |

## Core Conventions

- Race/ethnicity must map to the six categories in `config/projection_config.yaml`.
- Geography follows state -> county -> place hierarchy.
- Age structure uses single-year ages with open-ended top cohort.
- Negative population values are invalid and should be rejected.

## Typical Workflow

1. Load harmonized source extracts from `data/processed/` or staged interim inputs.
2. Run module-level processors for required rate/base components.
3. Validate outputs for completeness, shape, and demographic plausibility.
4. Persist outputs to documented processed/interim destinations.
5. Run downstream pipeline steps that consume these outputs.

## Minimal Usage Example

```python
from cohort_projections.data.process.base_population import process_county_population

county_base = process_county_population(raw_data)
```

## Validation Expectations

- No negative populations.
- Expected geographies present (all ND counties when county scope is requested).
- Cohort completeness by age x sex x race/ethnicity.
- Warning thresholds follow project demographic guardrails.

## Related Documentation

- `docs/guides/data-sources-workflow.md`
- `docs/guides/testing-workflow.md`
- `docs/methodology.md`
- `docs/governance/adrs/001-fertility-rate-processing.md`
- `docs/governance/adrs/002-survival-rate-processing.md`
- `docs/governance/adrs/003-migration-rate-processing.md`

## Legacy Detailed Reference

The previous long-form module reference was archived for reproducibility:

- `cohort_projections/data/process/README_ARCHIVE_2026-02-26.md`

Use the archive for historical examples and exhaustive function-by-function narrative.
