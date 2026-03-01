# 2026-02-28 Place Projection Output Contract (PP3-S06)

| Attribute | Value |
|-----------|-------|
| **Date** | 2026-02-28 |
| **Reviewer** | Claude (AI Agent) -- requires human review before acceptance |
| **Scope** | PP3-S06 output contract defining Phase 1 deliverables for city/place projections |
| **Status** | Revised -- human decisions incorporated 2026-02-28; final approval at PP3-S07 gate |
| **Related ADR** | ADR-033 |

---

## 1. Scenario Coverage

Place-level outputs are generated for all three active scenarios:

| Scenario Key | Label | Place Outputs |
|--------------|-------|---------------|
| `baseline` | Baseline (Trend Continuation) | Yes |
| `restricted_growth` | Restricted Growth (CBO Policy-Adjusted) | Yes |
| `high_growth` | High Growth (Elevated Immigration) | Yes |

Share-trending is applied independently per scenario: each scenario's county projection yields its own set of place totals. The same historical share trends are used across scenarios; only the county denominators differ.

## 2. Projection Output Files

### 2.1 Directory Layout

Place outputs follow the existing `data/projections/{scenario}/{level}/` convention:

```
data/projections/
  baseline/
    county/                          # existing
    state/                           # existing
    place/                           # NEW
      nd_place_3825700_projection_2025_2055_baseline.parquet
      nd_place_3825700_projection_2025_2055_baseline_metadata.json
      nd_place_3825700_projection_2025_2055_baseline_summary.csv
      ...
      places_summary.csv             # all-place summary (mirrors countys_summary.csv)
      places_metadata.json           # run-level metadata (mirrors countys_metadata.json)
  restricted_growth/
    place/                           # same structure
  high_growth/
    place/                           # same structure
```

### 2.2 File Naming Convention

Pattern: `nd_place_{place_fips}_projection_{start}_{end}_{scenario}.{ext}`

- `{place_fips}`: 7-digit GEOID (state + place FIPS, e.g., `3825700` for Fargo)
- `{start}`: base year (`2025`)
- `{end}`: final year (`2055`)
- `{scenario}`: scenario key (`baseline`, `restricted_growth`, `high_growth`)
- `{ext}`: `parquet`, `csv`, or `json`

This mirrors the county pattern `nd_county_{county_fips}_projection_{start}_{end}_{scenario}.parquet`.

### 2.3 Formats

| Format | Purpose | Compression |
|--------|---------|-------------|
| Parquet | Machine use, pipeline consumption | gzip (per config `output.compression`) |
| CSV | Human-readable, summary tables | None |
| JSON | Per-place and run-level metadata | None |

### 2.4 Which Places Get Individual Files

Only places in the projection universe (population >= 500 per ADR-033 EXCLUDED threshold) receive individual output files. Per the S01 scope envelope:

- **HIGH + MODERATE + LOWER**: 90 places (9 + 9 + 72) get individual parquet/metadata/summary files
- **EXCLUDED** (<500): 265 places are omitted from projection output

## 3. Schema Definitions by Confidence Tier

### 3.1 HIGH Tier (>10,000 population) -- Full Age Groups

9 places (e.g., Fargo, Bismarck, Grand Forks, Minot, West Fargo, Williston, Dickinson, Mandan, Jamestown).

**Parquet schema:**

| Column | Type | Description |
|--------|------|-------------|
| `year` | `int64` | Projection year (2025-2055) |
| `age_group` | `string` | 5-year age group label (e.g., `0-4`, `5-9`, ..., `85+`) |
| `sex` | `string` | `Male` or `Female` |
| `population` | `float64` | Projected population for this cohort |

Age groups use the standard 18-bin scheme from the detail workbooks: `0-4`, `5-9`, `10-14`, `15-19`, `20-24`, `25-29`, `30-34`, `35-39`, `40-44`, `45-49`, `50-54`, `55-59`, `60-64`, `65-69`, `70-74`, `75-79`, `80-84`, `85+`.

**Note:** HIGH-tier places do not receive single-year-of-age or race/ethnicity detail. Age-sex distributions are allocated from the containing county's cohort-component output proportionally. This is a share-trending model, not a cohort-component model at the place level.

### 3.2 MODERATE Tier (2,500-10,000 population) -- Broad Age Groups

9 places (e.g., Valley City, Wahpeton, Devils Lake, Beulah, Grafton, Watford City, Rugby, Bottineau, Lisbon).

**Parquet schema:**

| Column | Type | Description |
|--------|------|-------------|
| `year` | `int64` | Projection year (2025-2055) |
| `age_group` | `string` | Broad age group label |
| `sex` | `string` | `Male` or `Female` |
| `population` | `float64` | Projected population for this cohort |

Broad age groups (6 bins): `0-17`, `18-24`, `25-44`, `45-64`, `65-84`, `85+`.

### 3.3 LOWER Tier (500-2,500 population) -- Total Only

72 places.

**Parquet schema:**

| Column | Type | Description |
|--------|------|-------------|
| `year` | `int64` | Projection year (2025-2055) |
| `population` | `float64` | Projected total population |

No age or sex breakdown. Total population only.

### 3.4 Summary CSV Schema (All Tiers)

The per-place `_summary.csv` and the aggregated `places_summary.csv` share a common schema, mirroring `countys_summary.csv`:

| Column | Type | Description |
|--------|------|-------------|
| `place_fips` | `string` | 7-digit place GEOID (or `bal_{county_fips}` for balance-of-county rows) |
| `name` | `string` | Place name (e.g., `Fargo city`) or `Balance of {County Name}` |
| `county_fips` | `string` | 5-digit primary county GEOID (from crosswalk) |
| `level` | `string` | Always `place` |
| `row_type` | `string` | `place` for projected places; `balance_of_county` for the unincorporated/rural remainder row |
| `confidence_tier` | `string` | `HIGH`, `MODERATE`, or `LOWER` (null for balance-of-county rows) |
| `base_population` | `float64` | Population at base year (2025) |
| `final_population` | `float64` | Population at final year (2055) |
| `absolute_growth` | `float64` | `final_population - base_population` |
| `growth_rate` | `float64` | `absolute_growth / base_population` |
| `base_share` | `float64` | Place share of county at base year |
| `final_share` | `float64` | Projected place share of county at final year |
| `processing_time` | `float64` | Seconds to process this place |

**Balance-of-county rows:** Each county with at least one projected place receives a `Balance of {County Name}` row in `places_summary.csv` (one per county). This row reports the residual population (county total minus sum of projected places) as a published output, making the county accounting fully transparent. Balance-of-county rows have `row_type = "balance_of_county"`, a synthetic FIPS of the form `bal_{county_fips}`, and a null `confidence_tier`. The same balance-of-county rows appear in the workbook summary tables alongside the projected places.

## 4. Metadata Fields

### 4.1 Per-Place Metadata JSON

Each place gets a `_metadata.json` file mirroring the county metadata pattern:

```json
{
  "geography": {
    "level": "place",
    "place_fips": "3825700",
    "name": "Fargo city",
    "county_fips": "38017",
    "confidence_tier": "HIGH",
    "base_population": 131564.0
  },
  "projection": {
    "base_year": 2025,
    "end_year": 2055,
    "scenario": "baseline",
    "method": "share_of_county_trending",
    "processing_date": "2026-XX-XXTXX:XX:XX+00:00"
  },
  "share_model": {
    "trend_type": "linear",
    "base_share": 0.652,
    "final_share": 0.668,
    "share_change": 0.016,
    "historical_window": "2000-2024",
    "crosswalk_vintage": "2020"
  },
  "summary_statistics": {
    "base_population": 131564.0,
    "final_population": 168764.5,
    "absolute_growth": 37200.5,
    "growth_rate": 0.283,
    "years_projected": 30
  },
  "validation": {
    "share_within_bounds": true,
    "share_sum_check_passed": true,
    "all_checks_passed": true
  },
  "processing_time_seconds": 0.12
}
```

### 4.2 Run-Level Metadata JSON (`places_metadata.json`)

Mirrors `countys_metadata.json`:

```json
{
  "level": "place",
  "num_geographies": 90,
  "successful": 90,
  "failed": 0,
  "by_tier": {
    "HIGH": 9,
    "MODERATE": 9,
    "LOWER": 72
  },
  "total_processing_time_seconds": 12.5,
  "processing_date": "2026-XX-XXTXX:XX:XX+00:00",
  "crosswalk_vintage": "2020",
  "model_version": "1.0.0"
}
```

### 4.3 Required Metadata on Every Output Record

Each parquet file must embed the following in its Parquet file metadata (key-value pairs in the file footer), enabling downstream consumers to identify provenance without needing the sidecar JSON:

| Key | Example Value |
|-----|---------------|
| `scenario` | `baseline` |
| `geography_level` | `place` |
| `place_fips` | `3825700` |
| `county_fips` | `38017` |
| `confidence_tier` | `HIGH` |
| `projection_base_year` | `2025` |
| `projection_end_year` | `2055` |
| `model_method` | `share_of_county_trending` |
| `model_version` | `1.0.0` |
| `crosswalk_vintage` | `2020` |
| `processing_date` | ISO 8601 timestamp |

## 5. QA Summary Tables

The pipeline must produce the following QA artifacts in `data/projections/{scenario}/place/qa/`:

### 5.1 Tier-Level Summary (`qa_tier_summary.csv`)

One row per confidence tier, aggregating across all places in that tier:

| Column | Description |
|--------|-------------|
| `confidence_tier` | `HIGH`, `MODERATE`, `LOWER` |
| `place_count` | Number of places in tier |
| `total_base_population` | Sum of base-year population |
| `total_final_population` | Sum of final-year population |
| `mean_growth_rate` | Average growth rate across places |
| `median_growth_rate` | Median growth rate |
| `min_growth_rate` | Minimum growth rate |
| `max_growth_rate` | Maximum growth rate |

### 5.2 Share-Sum Validation (`qa_share_sum_validation.csv`)

One row per county-year, verifying the county constraint:

| Column | Description |
|--------|-------------|
| `county_fips` | 5-digit county GEOID |
| `county_name` | County name |
| `year` | Projection year |
| `sum_place_shares` | Sum of all place shares in this county-year |
| `balance_of_county_share` | `1.0 - sum_place_shares` |
| `constraint_satisfied` | `true` if `sum_place_shares <= 1.0` |
| `rescaling_applied` | `true` if proportional rescaling was needed |

### 5.3 Outlier Flags (`qa_outlier_flags.csv`)

Flags places with unusual projection behavior for manual review:

| Column | Description |
|--------|-------------|
| `place_fips` | 7-digit place GEOID |
| `name` | Place name |
| `confidence_tier` | Tier label |
| `flag_type` | Type of flag (see below) |
| `flag_detail` | Description of the flagged condition |
| `year` | Year where flag applies (if year-specific) |

Flag types:
- `SHARE_REVERSAL`: place share trend reverses direction vs. historical
- `EXTREME_GROWTH`: projected growth rate exceeds tier-specific uncertainty band (HIGH: +/-10%, MODERATE: +/-15%, LOWER: +/-25% per year)
- `NEAR_ZERO_SHARE`: projected share drops below 0.1% of county
- `SHARE_RESCALED`: place was affected by proportional rescaling because county shares summed to >100%
- `POPULATION_DECLINE_TO_NEAR_ZERO`: projected population falls below 50

### 5.4 Balance-of-County Table (`qa_balance_of_county.csv`)

Derived residual for each county (county total minus sum of projected places):

| Column | Description |
|--------|-------------|
| `county_fips` | 5-digit county GEOID |
| `county_name` | County name |
| `year` | Projection year |
| `county_total` | County projection total |
| `sum_of_places` | Sum of place projections in this county |
| `balance_of_county` | `county_total - sum_of_places` |
| `balance_share` | `balance_of_county / county_total` |

## 6. Consistency Constraints

The following invariants must hold for every scenario and every projection year. The pipeline must validate these and fail loudly (not silently) on violation.

### 6.1 Hard Constraints (pipeline must enforce)

1. **Share bound**: For every place and year, `0 <= place_share <= 1.0`.
2. **County share sum**: For every county and year, `sum(place_shares) <= 1.0`. If trending produces a sum >1.0, apply proportional rescaling and log the event.
3. **Place-county consistency**: For every county and year, `sum(place_populations) <= county_total`. The remainder is the balance-of-county (unincorporated/rural).
4. **Non-negative population**: No place may have negative projected population in any year.
5. **Monotonic FIPS**: Place FIPS in the output universe must match the projection-universe crosswalk exactly (no orphan places, no missing places).
6. **Scenario ordering at state level**: `restricted_growth <= baseline <= high_growth` for state totals. (This is an existing invariant; place outputs must not break it.)

### 6.2 Soft Constraints (flag in QA, do not block)

1. **Balance-of-county non-negative**: `balance_of_county >= 0` for each county-year. A negative balance means place shares sum to more than the county total after population multiplication, which should not happen if share constraints hold, but should be flagged if it occurs due to floating-point issues.
2. **Share stability**: No place share should change by more than 20 percentage points over the 30-year horizon. Flag for review if exceeded.
3. **Tier-appropriate growth**: Growth rates outside the tier uncertainty band (HIGH +/-10%, MODERATE +/-15%, LOWER +/-25%) are flagged but not blocked.

## 7. Workbook Impacts

### 7.1 Recommendation: Separate Place Workbook

Create a new standalone place workbook rather than adding tabs to the existing county detail workbooks. Rationale:

- The existing detail workbooks (`nd_projections_{scenario}_detail_{datestamp}.xlsx`) already contain 62 sheets (1 TOC + 1 state + 8 regions + 53 counties = 63 sheets). Adding 90 place sheets would make them unwieldy.
- Place projections use a different methodology (share-trending vs. cohort-component) and carry different metadata (confidence tiers, share model parameters). A separate workbook makes the methodological distinction clear.
- Users interested in city projections are often a different audience than users of county age-sex detail.

### 7.2 New Workbook: `nd_projections_{scenario}_places_{datestamp}.xlsx`

Structure:

| Sheet | Content |
|-------|---------|
| Table of Contents | Summary of all places with hyperlinks, confidence tier labels, base/final population, growth rate, county assignment |
| HIGH tier places (9 sheets) | Age-group x sex tables at key years (2025, 2030, 2035, 2040, 2045, 2050, 2055), matching the county detail format with 5-year age groups |
| MODERATE tier places (9 sheets) | Broad age-group x sex tables at key years |
| LOWER tier places (1 combined sheet) | Total population at key years for all 72 LOWER-tier places in a single summary table (no individual sheets). **Prominent uncertainty caveat header** at the top of the sheet noting that LOWER-tier projections carry wider uncertainty bands and should be used with caution. |
| Methodology | Place projection methodology note, confidence tier definitions, caveats |

### 7.3 Impact on Existing Workbooks

- **Provisional workbook** (`build_provisional_workbook.py`): Add one new sheet `Places -- {scenario_short}` with a summary table (place name, county, tier, key-year populations, growth). This mirrors the existing `Counties -- {scenario_short}` sheets. No structural changes to existing sheets.
- **Detail workbooks** (`build_detail_workbooks.py`): No changes. County detail workbooks remain county-focused.
- **Export pipeline** (`03_export_results.py`): The `--places` flag already exists in the CLI. Implementation must wire it to the new place-level output directory and summary generation.

### 7.4 Methodology Text Update

Add a place-specific methodology line to `_methodology.py`:

```
"City/place projections: Share-of-county trending method (ADR-033). "
"Population-based confidence tiers: HIGH (>10,000, 5-year age groups), "
"MODERATE (2,500-10,000, broad age groups), LOWER (500-2,500, total only). "
"Place projections are constrained to county totals."
```

## 8. Dependencies

The following artifacts must exist before place output generation can execute:

| Dependency | Artifact | Status | Blocking? |
|------------|----------|--------|-----------|
| Place-county crosswalk | `data/processed/geographic/place_county_crosswalk_2020.csv` | PP3-S03 rules defined; file not yet built | Yes |
| Historical place time series | Assembled from `sub-est00int`, `sub-est2020int`, `sub-est2024` | PP3-S02 readiness confirmed; assembly script not yet written | Yes |
| Share-trend model | PP3-S04 model specification | Pending | Yes |
| County projections (all scenarios) | `data/projections/{scenario}/county/*.parquet` | Complete | No |
| Backtest validation results | PP3-S05 acceptance metrics | Pending | Yes |
| Confidence tier assignment | Derived from 2024 PEP population in crosswalk | Derivable from existing data | No |

## 9. Resolved Questions (Human Decisions -- 2026-02-28)

The following open questions were resolved by human review on 2026-02-28:

1. **LOWER-tier workbook treatment -- Combined sheet with caveat header.** All 72 LOWER-tier places appear on a single combined sheet in the place workbook. The sheet carries a prominent uncertainty caveat header warning users that LOWER-tier projections have wider uncertainty bands and should be used with caution. (See updated Section 7.2 sheet table.)

2. **Balance-of-county as output -- Publish as output.** A `Balance of {County Name}` row is included in `places_summary.csv` and the workbook summary for each county that has at least one projected place. This makes the county-level accounting fully transparent and eliminates the need for users to compute the residual manually. Balance-of-county rows use `row_type = "balance_of_county"` in the summary schema. (See updated Section 3.4 schema.) The QA table in Section 5.4 is retained for detailed year-by-year diagnostics.

3. **Race/ethnicity at place level -- Leave unmentioned.** No commitment to future race/ethnicity work at the place level is made in Phase 1 deliverables. The topic is intentionally omitted from the methodology text and output contract rather than flagged as a future-phase item.

4. **Key years -- Same 7 key years as county workbooks.** Place workbooks use the same 7 key years: 2025, 2030, 2035, 2040, 2045, 2050, 2055, consistent with the county detail workbooks. No reduction despite higher uncertainty at lower tiers.

---

| Attribute | Value |
|-----------|-------|
| **Last Updated** | 2026-02-28 |
| **Version** | 1.0 |
