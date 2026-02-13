---
title: "Audit: Implementation Plan & Findings Documents"
created: 2026-02-13T20:00:00-06:00
status: audit-complete
author: Claude Code (Opus 4.6)
audited_documents:
  - docs/plans/implementation-plan-census-method-upgrade.md
  - docs/plans/census-method-assessment-and-path-forward.md
purpose: >
  Independent verification of claims, data file references, and assumptions
  in the Census method upgrade planning documents. Conducted via parallel
  sub-agents auditing against actual files on disk and Census Bureau sources.
related_documents:
  - docs/plans/implementation-plan-census-method-upgrade.md (audited)
  - docs/plans/census-method-assessment-and-path-forward.md (audited)
---

# Audit: Implementation Plan & Findings Documents

## 1. Overview

Three independent sub-agents audited the planning documents on 2026-02-13.
This document records their findings, tracks resolution status, and captures
corrections made to the source documents.

---

## 2. Findings: Data File References

### 2.1 `base_population_by_county.csv` — Wrong Path

**Severity**: Blocking (referenced in 4 places in implementation plan)

**Claim**: File exists at `data/processed/base_population_by_county.csv` with
53 counties, 18 five-year age groups x 2 sexes.

**Finding**: File does NOT exist at `data/processed/`. Found at two locations:
- `sdc_2024_replication/data/base_population_by_county.csv`
- `data/processed/immigration/rates/base_population_by_county.csv`

**Verified schema**: Columns `[county_name, age_group, sex, population]`,
1,908 rows (53 counties x 18 age groups x 2 sexes). Matches requirements.

**Status**: RESOLVED — updated implementation plan references to correct path.

### 2.2 `survival_rates_sdc_2024_by_age_group.csv` — Wrong Path

**Severity**: Blocking (referenced in findings document Section 8.3)

**Claim**: File exists with 36 rows (18 age groups x 2 sexes), CDC ND 2020
life tables.

**Finding**: File exists at `data/processed/sdc_2024/survival_rates_sdc_2024_by_age_group.csv`.
Columns: `[age_group, age_start, age_end, sex, survival_rate_5yr,
survival_rate_1yr, source, notes]`. 36 rows. Source: `SDC_2024_CDC_ND_2020`.

**Status**: RESOLVED — updated implementation plan references to correct path.

### 2.3 PEP Vintage 2024 County Age-Sex Data — Initially Reported as Unavailable

**Severity**: Blocking (the sole remaining data gap in the implementation plan)

**Claim**: Can be fetched from Census Bureau API endpoint
`/data/2024/pep/charagegroups`.

**Initial finding (2026-02-13, earlier in session)**: Three independent agents
concluded the data was NOT published. This was INCORRECT — see correction below.

**Corrected finding (2026-02-13, later in session)**: The Census Bureau
published `cc-est2024-agesex-all.csv` on **June 26, 2025** at:
`https://www2.census.gov/programs-surveys/popest/datasets/2020-2024/counties/asrh/`

The file was not in our local archive because the earlier download session
(2026-02-03) only fetched `co-est2024-alldata.csv` (county totals). The
age-sex file (`cc-est2024-agesex-all.csv`, 8.3 MB) was published under a
different subdirectory (`asrh/` vs `totals/`).

**Downloaded and archived (2026-02-13)**:
- Parquet: `shared-data/census/popest/parquet/2020-2024/county/cc-est2024-agesex-all.parquet`
- 18,864 rows, 96 columns (wide format — age groups as columns, not AGEGRP rows)
- 53 ND counties confirmed (STATE=38), 6 YEAR codes (2020-2024 + base)
- Metadata: `shared-data/census/popest/metadata/cc-est2024-agesex-all.json`
- Added to `raw-archives/2020-2024-raw.zip` and `catalog.yaml`

**Format note**: Unlike `cc-est2020int-alldata` (long format with `AGEGRP`
column), this file uses **wide format** with columns like `AGE04_TOT`,
`AGE513_MALE`, `AGE85PLUS_FEM`. The data loader must pivot wide-to-long.

**Status**: RESOLVED — data is available. All 5 periods (2000-2024) can use
real Census data. No synthetic data needed.

### 2.4 Convergence Config Already Exists

**Severity**: Minor

**Claim** (implementation plan Section 4.4): Need to add convergence schedule
to `projection_config.yaml`.

**Finding**: Config already exists at lines 142-150:
```yaml
interpolation:
  method: "census_bureau_convergence"
  convergence_schedule:
    recent_to_medium_years: 5
    medium_hold_years: 10
    medium_to_longterm_years: 5
```

**Status**: RESOLVED — plan should say "extend existing config" not "add."

### 2.5 Dampening Config Mostly Exists

**Severity**: Minor

**Claim**: Need to add dampening config with boom_periods.

**Finding**: County list and 60% factor already exist in config (lines
129-138). Only `boom_periods` key needs to be added.

**Status**: RESOLVED — note in plan that only `boom_periods` is new.

---

## 3. Findings: Unverified Claims in Findings Document

### 3.1 SDC Variable Dampening Multipliers (0.2, 0.6, 0.5, 0.7)

**Severity**: Critical — presented as established fact in Section 8.1

**Claim**: SDC 2024 used specific variable multipliers on projected migration:
0.2 (2020-2025), 0.6 (2025-2035), 0.5 (2035-2040), 0.7 (2040-2050).

**Verification result**: The specific multipliers ARE documented in four
places within our codebase, all originating from prior workbook analysis:
- `sdc_2024_replication/README.md` (lines 233-240)
- `sdc_2024_replication/METHODOLOGY_SPEC.md` (lines 234-241)
- `docs/governance/adrs/017-sdc-2024-methodology-comparison.md` (line 93)
- `docs/methodology_comparison_sdc_2024.md` (lines 92-94)

The SDC public methodology report (`full_report_text.md`) says only
"rates were typically reduced to about 60%." The specific per-period
multipliers were extracted from examining the `Projections_Base_2023.xlsx`
workbook in a prior analysis session.

**Status**: RESOLVED — claim is substantiated by prior workbook analysis
documented in our replication files. Updated findings document to cite
sources explicitly.

### 3.2 "~32,000 Manual Adjustments"

**Severity**: Critical — presented as established fact in Section 8.2

**Claim**: SDC applied approximately 32,000 manual person-adjustments per
5-year projection period.

**Verification result**: The figure appears in two of our internal documents:
- `sdc_2024_replication/METHODOLOGY_SPEC.md` (line 285-286)
- `docs/governance/adrs/017-sdc-2024-methodology-comparison.md` (line 101)

However, the SDC public documentation does NOT quantify adjustments — it only
acknowledges that college-age and male migration adjustments were made. The
specific "32,000" number lacks a traceable derivation. It may represent a sum
of adjustment column values in the SDC workbook, but the calculation was not
documented.

Math check: 53 counties x 18 age groups x 2 sexes x 6 projection periods =
11,448 cells total. If adjustments span all periods, ~3 adjustments/cell is
more plausible. But this is speculation.

**Status**: RESOLVED — updated findings document to qualify the number as
"documented in our replication analysis" rather than "SDC reported." Added
caveat that the specific figure lacks independent verification.

### 3.3 "Methodologically Cleaner" Assertion

**Severity**: Minor

**Claim** (Section 8.1): "Our approach is methodologically cleaner — dampening
historical data that contains boom effects, rather than inventing ad-hoc
multipliers for future periods."

**Finding**: This is a defensible methodological preference, not an objective
fact. Both approaches (dampening inputs vs. varying output multipliers) are
used by different state demographic offices.

**Status**: SHOULD REPHRASE — reframe as "Our approach differs from SDC 2024
in philosophy" rather than asserting superiority.

---

## 4. Findings: Architecture and Schema

### 4.1 Engine Migration Format

**Verified**: The engine currently expects migration rates as `[age, sex, race,
net_migration OR migration_rate]` with 1,092 rows per county (91 ages x 2
sexes x 6 races). The plan correctly identifies this and includes a bridge
function (`expand_5yr_migration_to_engine_format()`).

### 4.2 Census 2000 AGEGRP Codes

**Verified**: AGEGRP codes 0-18 where code 0 = Total (must be excluded).
Codes 1-18 = 18 actual five-year age groups. Plan must note to filter
AGEGRP > 0.

### 4.3 Age Group Alignment Across Sources

**Verified**: Both Census 2000 (AGEGRP 1-18) and cc-est2019 files use
matching 18 five-year groups: 0-4 through 85+.

### 4.4 Backward-Compatible Interface Changes

**Verified**: Proposed changes to `CohortComponentProjection.__init__`,
`project_single_year()`, and `multi_geography.py` are sound and
backward-compatible via optional parameters with None defaults.

---

## 5. Resolution Tracker

| # | Issue | Severity | Status | Resolution |
|---|-------|----------|--------|------------|
| 2.1 | base_population_by_county.csv path | Blocking | Resolved | File at `sdc_2024_replication/data/` and `data/processed/immigration/rates/` |
| 2.2 | survival_rates_sdc_2024_by_age_group.csv | Blocking | Resolved | File at `data/processed/sdc_2024/` |
| 2.3 | PEP 2024 age-sex data availability | Blocking | Resolved | Initially reported unavailable; actually published June 2025. Downloaded 2026-02-13. |
| 2.4 | Convergence config exists | Minor | Resolved | Plan updated to "extend existing config" |
| 2.5 | Dampening config mostly exists | Minor | Resolved | Only `boom_periods` key is new |
| 3.1 | SDC variable multipliers | Critical | Resolved | Sourced from prior workbook analysis; documented in replication files |
| 3.2 | 32K adjustments figure | Critical | Resolved | Qualified as unverified estimate; rephrased in findings doc |
| 3.3 | "Cleaner" assertion | Minor | Resolved | Rephrased as methodological difference, not superiority |
| 4.2 | AGEGRP=0 filtering | Minor | Resolved | Note added to data loader specs |

---

## 6. Research Agent Results

### 6.1 SDC Workbook Verification (completed 2026-02-13)

**Agent task**: Verify SDC variable dampening multipliers and manual
adjustment count claims against SDC source files and workbooks.

**Result**: Both claims are documented in our replication files
(`sdc_2024_replication/README.md`, `METHODOLOGY_SPEC.md`, ADR-017) but
originate from prior workbook analysis sessions, not from SDC public
documentation. The SDC report says only "typically reduced to about 60%."
The specific multipliers and adjustment count were extracted from examining
`Projections_Base_2023.xlsx` but the extraction methodology was not recorded.

**Action taken**: Updated findings document to cite our replication files as
the source and qualify claims appropriately.

### 6.2 Missing Data File Search (completed 2026-02-13)

**Agent task**: Locate `base_population_by_county.csv`,
`survival_rates_sdc_2024_by_age_group.csv`, and PEP 2020-2024 age-sex data.

**Result**:
- `base_population_by_county.csv` found at `sdc_2024_replication/data/` and
  `data/processed/immigration/rates/`. Schema verified: 1,908 rows, columns
  `[county_name, age_group, sex, population]`.
- `survival_rates_sdc_2024_by_age_group.csv` found at
  `data/processed/sdc_2024/`. Schema verified: 36 rows, columns
  `[age_group, age_start, age_end, sex, survival_rate_5yr, survival_rate_1yr,
  source, notes]`.
- PEP 2020-2024 age-sex data: Initially reported as unavailable. **CORRECTION**:
  `cc-est2024-agesex-all.csv` was published June 2025 and downloaded 2026-02-13.
  Now at `shared-data/census/popest/parquet/2020-2024/county/cc-est2024-agesex-all.parquet`
  (18,864 rows, 96 columns, wide format).

### 6.3 Census FTP / PEP 2024 Age-Sex Availability (completed 2026-02-13)

**Agent task**: Check Census Bureau FTP server and API for county-level
age-sex estimates covering 2020-2024.

**Initial result**: Agents concluded PEP Vintage 2024 county-level age-sex
data was not published. This was INCORRECT.

**Corrected result**: `cc-est2024-agesex-all.csv` was published June 26, 2025
at `https://www2.census.gov/programs-surveys/popest/datasets/2020-2024/counties/asrh/`.
The agents searched the wrong subdirectory (`totals/` instead of `asrh/`) and
the local archive only contained the totals file. Downloaded and archived
on 2026-02-13.

---

## 7. Decision: Handling the 2020-2024 Data Gap — SUPERSEDED

~~The Census Bureau has not published county-level age-sex estimates for
2020-2024.~~ **CORRECTION**: The data WAS published (June 2025) and has been
downloaded. See Section 2.3 correction above.

The original three options (A: 4 periods only, B: synthetic 2024,
C: wait for Census) are now moot. With `cc-est2024-agesex-all.csv` available,
all 5 periods (2000-2024) can use actual Census Bureau data:

| Period | Start Source | End Source |
|--------|-------------|-----------|
| 2000-2005 | Census 2000 file | Census 2000 file (POPESTIMATE2005) |
| 2005-2010 | Census 2000 file (POPESTIMATE2005) | cc-est2019 (YEAR=1) |
| 2010-2015 | cc-est2019 (YEAR=1) | cc-est2019 (YEAR=6) |
| 2015-2020 | cc-est2019 (YEAR=6) | cc-est2020int (YEAR=11) |
| 2020-2024 | cc-est2020int (YEAR=11) | cc-est2024-agesex-all (YEAR=6) |

**Status**: RESOLVED — 5 periods with real data. No synthetic data needed.

---

## 8. Corrections Applied to Source Documents

### 8.1 Implementation Plan (`implementation-plan-census-method-upgrade.md`)

| Section | Change | Reason |
|---------|--------|--------|
| 3 (Data Inventory) | Updated file paths for base_population and survival_rates | Wrong paths (audit 2.1, 2.2) |
| 3 (Data Gaps) | ~~Revised PEP 2024 entry to note data not yet published~~ → Corrected: data IS available, no remaining gaps | Initial: audit 2.3; Corrected: data found 2026-02-13 |
| 3 (Available Data) | Added `cc-est2024-agesex-all.parquet` to available data table | Data downloaded 2026-02-13 |
| 3 (Historical Matrix) | Updated 2020 source; changed 2024 from "✗ Not published" to "✓ Available" | Data found and downloaded |
| 4.2.2 | Updated data source comments; replaced `construct_synthetic_2024` with `load_pep_2020_2024_county_age_sex` | Real data available, no synthetic needed |
| 4.4 | Noted convergence config already exists; uncommented [2020, 2024] period | Audit 2.4 + data now available |

### 8.2 Findings Document (`census-method-assessment-and-path-forward.md`)

| Section | Change | Reason |
|---------|--------|--------|
| 8.1 | Added source citations for multiplier values | Audit 3.1 — unverified claim |
| 8.1 | Rephrased "methodologically cleaner" assertion | Audit 3.3 — opinion vs fact |
| 8.2 | Qualified "~32,000" figure with sourcing caveat | Audit 3.2 — unverified number |
| 8.3 | ~~Updated PEP 2024 status to "not yet published"~~ → Corrected: data IS published and downloaded | Initial: audit 2.3; Corrected: data found 2026-02-13 |

### 8.3 AGENTS.md

| Section | Change | Reason |
|---------|--------|--------|
| 6 (Data Conventions) | Added "Shared Census Data Archive" subsection | Document shared-data directory for agent use |
