# Export Script Text Content Review

| Attribute | Value |
|-----------|-------|
| **Date** | 2026-02-17 |
| **Reviewer** | Claude Code (Opus 4.6) |
| **Scope** | All non-data text in `build_detail_workbooks.py` and `build_provisional_workbook.py` |
| **Status** | Findings recorded; no changes made |

---

## High Priority: Factual Accuracy Issues

### 1. Fertility source discrepancy

The config says `source: "SEER"` with a 5-year average (2018–2022). The summary workbook says "CDC/NCHS age-specific fertility rates (2024 for major groups, 2022 national rates for AIAN/Asian)." The detail workbook says "SEER age-specific rates (2018–2022 average)." These three descriptions cannot all be correct. **The actual data pipeline needs to be audited**, and all three locations aligned.

### 2. Mortality year discrepancy

`projection_config.yaml` says `life_table_year: 2020` and `source: "SEER"`. The summary workbook says "CDC/NCHS life tables (2023)". The detail workbook says "CDC life tables" with no year. Same issue — verify and align.

### 3. Internal ADR references in user-facing text

Both scripts cite "ADR-037" in methodology footers. DHHS users have no access to ADRs. Replace with the actual CBO publication: `"CBO Demographic Outlook (Pub. 60875, Jan 2025; Pub. 61879, Jan 2026)"`.

### 4. Missing "conditional projection" caveat

ADR-037 explicitly recommends all outputs include a caveat that projections are conditional on assumptions, not forecasts. Neither workbook includes this. Suggested text: *"These projections are conditional on stated assumptions and should not be interpreted as forecasts."*

---

## Medium Priority: Clarity & Consistency

### 5. Scenario name mismatch between scripts

- **Detail workbook**: `"Baseline (Trend Continuation)"`, `"Restricted Growth (CBO Policy-Adjusted)"`, `"High Growth (Pre-Policy Elevated Immigration)"`
- **Summary workbook**: `"Baseline"`, `"Restricted Growth"`, `"High Growth"`

An analyst comparing the two may wonder if these are the same scenarios. Use the full names everywhere; use short names only for tab labels if length is a concern.

### 6. Ambiguous "Change" / "% Change" column headers

The detail workbook (`build_detail_workbooks.py:302`) uses bare `"Change"` and `"% Change"` without specifying the comparison period. The summary workbook is inconsistent — some sheets say `"Change (2025-2045)"` (good), others say just `"% Change"` (ambiguous). All should read `"Change (2025-2045)"` and `"% Change (2025-2045)"`.

### 7. Detail workbook methodology footer is too sparse

It's significantly less detailed than the summary workbook's version — omits BEBR method, Rogers-Castro allocation, "held constant" for fertility, and the life table year. Given that the detail workbooks are the primary analytical tools, they arguably need *more* methodology documentation, not less.

### 8. Misleading TOC generation line in detail workbook

`build_detail_workbooks.py:457–461`: `"Base: Census PEP 2000-2024"` could be read as "the base population comes from a 2000–2024 average." In reality, 2000–2024 is the migration data window. Change to: `"Base Year: 2025 (Census PEP 2024 Vintage)"`.

### 9. Detail workbook TOC title missing horizon

`build_detail_workbooks.py:448`: `"North Dakota Population Projections"` — should be `"North Dakota Population Projections 2025–2045"` to match the summary workbook.

### 10. No organization attribution

Neither workbook identifies who produced the projections. DHHS analysts receiving the file need to know the source (e.g., "Produced by the North Dakota State Data Center").

### 11. Total Dependency ratio missing its formula

Youth shows `"Youth Dependency (0-14 / 15-64)"` and Aged shows `"Aged Dependency (65+ / 15-64)"`, but Total just says `"Total Dependency"` with no formula. For consistency: `"Total Dependency ((0-14 + 65+) / 15-64)"`.

### 12. Summary workbook "% Chg" is ambiguous

`build_provisional_workbook.py:225–228`: State Summary columns read `"Baseline % Chg"` — is this year-over-year or cumulative from 2025? Change to `"Baseline % Chg from 2025"`.

---

## Low Priority: Polish

### 13. "Key Indicators" section header could be more specific

`build_detail_workbooks.py:362`: Currently `"Key Indicators"` — consider `"Dependency Ratios"` since that's the only content. Keep "Key Indicators" only if additional indicators (median age, sex ratio) are planned.

### 14. TOC descriptions use data-professional jargon

In the summary workbook TOC:

- `"wide format for easy comparison"` → `"side-by-side comparison"`
- `"long format (scenario x year) for pivoting and charting"` → `"one row per scenario-year combination (for charts and pivot tables)"`

### 15. "Range (High - Restricted)" label

`build_provisional_workbook.py:316`: "Range" may confuse readers who expect it to include all three scenarios. Consider `"Spread (High − Restricted)"` or `"Difference (High − Restricted)"`.

### 16. Inconsistent total row labels

Detail workbook: `"Total"`. Summary workbook age structure: `"TOTAL"`. Summary county detail: `"STATE TOTAL"`. Minor but worth standardizing.

### 17. Missing data availability note

ADR-038 states that advanced users needing race breakdowns or annual granularity should use parquet files. Neither workbook communicates this. A one-line note in the methodology footer would be useful.

### 18. Consider a shared constants module

Creating something like `scripts/exports/_methodology.py` with canonical methodology text, scenario descriptions, and labels would prevent future drift between the two scripts.

---

## Things That Are Good (No Changes Needed)

- Region names and county-to-region mapping are correct and complete (all 53 counties verified)
- Sheet tab naming follows ADR-038 conventions
- Age group bins in the summary workbook (0–4, 5–17, 18–24, 25–44, 45–64, 65+) align well with DHHS program eligibility boundaries
- Back-to-TOC navigation links in detail workbooks
- County count annotations in TOC region entries (e.g., "3 counties")
- "STATE TOTAL" confirmation that state totals are county sums
- Provisional label and filename designation for document control
- Dependency ratio formulas embedded in labels (Youth and Aged)
