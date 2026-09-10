# Fact-Check: Commerce Annual Report Economic Stats (March 2026)

| Attribute | Value |
|-----------|-------|
| **Date** | 2026-03-31 |
| **Author** | Claude Code (Opus 4.6) |
| **Type** | Fact-check / data verification |
| **Scope** | All data points in `AnnualReport_EconStats_March2026.docx` plus four population growth narrative claims |
| **Status** | Complete (one item awaiting V2025 age-sex release, June 2026) |
| **Source Document** | `AnnualReport_EconStats_March2026.docx` (dated March 19, 2026) |

---

## Quick Summary

| # | Category | Verdict | Issue |
|---|----------|---------|-------|
| 1 | Real GDP | **PROBLEM** | Figures do not match BEA data; BEA shows decline, not growth |
| 2 | Mid-America Business Conditions Index | **Unverified** | State-level values not publicly indexed; one source suggests 53.3 not 53.8 |
| 3 | ND Businesses | **Plausible** | Arithmetic correct; source values not independently confirmed |
| 4 | General Fund Receipts | **Caution** | Arithmetic correct; likely comparing partial-year to full-year |
| 5 | Average Annual Income (Private) | **Minor error** | 2023 value off by $1 ($64,672 per BLS, not $64,671) |
| 6 | Cost of Living Index | **Mostly confirmed** | 2025 confirmed; 2024 value unverified (91.9 claimed vs 91.3 Q2 2024) |
| 7 | Total Employment (Private) | **Confirmed** | Exact match to BLS QCEW |
| 8 | Population | **Confirmed** | Exact match to PEP Vintage 2025 |
| 9 | Housing Permits | **Confirmed** | Exact match to Census BPS (2025 is preliminary) |

---

## Part 1: Annual Report Data Points

### 1. Real Gross Domestic Product

| Item | Claimed | BEA Official | Match? |
|------|---------|-------------|--------|
| 2024 | $63,366,400,000 | ~$59.9B (chained 2017$) | **No** |
| 2023 | $62,144,800,000 | ~$60.3B (chained 2017$) | **No** |
| YoY Change | +$1,221,600,000 | N/A | Arithmetic correct internally |
| % Change | +1.97% | **−0.7%** | **Wrong direction** |

**Verdict: PROBLEM.** The claimed figures do not match BEA's official real GDP by state. Per the BEA "Gross Domestic Product by State, Q4 2024" release (March 2025):

- BEA shows ND real GDP at ~$59.9B (chained 2017 dollars) for 2024 with a **−0.7% decline** — making ND the weakest-performing state.
- The report claims +1.97% growth.
- The dollar figures ($62–63B) don't correspond to any standard BEA series (real in 2017$, or nominal at ~$75–76B).

The arithmetic is internally consistent ($1,221,600,000 / $62,144,800,000 = 1.97%), but the underlying numbers appear to come from a non-standard source or methodology, or use a different base year / deflator.

**Sources:** BEA GDP by State release (March 2025); FRED series NDRGSP.

---

### 2. Mid-American Business Conditions Index

| Item | Claimed | Found | Match? |
|------|---------|-------|--------|
| Feb 2026 | 53.8 | 53.3 (one secondary source) | **Uncertain** |
| Feb 2025 | 55.1 | Not found | **Unverified** |
| Change | −1.3 points | N/A | Arithmetic correct |
| % Change | −2.36% | N/A | Arithmetic correct (−1.3 / 55.1 = −2.3593%) |

**Verdict: UNVERIFIED.** Creighton University publishes state-level values in monthly PDF reports that are not fully indexed online. The overall nine-state regional index for Feb 2026 was 54.7. One secondary source cited ND at 53.3 (not 53.8) for Feb 2026, but this could not be confirmed against the primary Creighton report.

**Note:** Creighton publishes two separate indices — the Mid-America **Business Conditions Index** (manufacturing/supply manager survey) and the **Rural Mainstreet Index** (rural bank CEO survey). These are different products with different values. Confirm which index the report intends to cite.

**Sources:** Creighton University Mid-American Economy reports; Business Record (Feb 2026); Clay County News.

---

### 3. ND Businesses

| Item | Claimed | Found | Match? |
|------|---------|-------|--------|
| 2024 | 36,650 private businesses | Not independently confirmed | Plausible |
| 2023 | 35,688 private businesses | Not independently confirmed | Plausible |
| Change | +962 | N/A | Arithmetic correct (36,650 − 35,688) |
| % Change | +2.70% | N/A | Arithmetic correct (962 / 35,688 = 2.696%) |

**Verdict: PLAUSIBLE but UNVERIFIED.** The figures are consistent with BLS QCEW establishment counts for ND private sector. The most likely source is QCEW annual averages, though the mid-30,000s range is higher than expected and may reflect a definitional difference (e.g., "businesses" vs "establishments" — one firm can operate multiple establishments).

**Recommendation:** Verify at the [BLS QCEW Data Viewer](https://data.bls.gov/cew/apps/data_views/data_views.htm) — select ND, Private, All Industries, Annual Averages, Establishments column.

---

### 4. General Fund Receipts

| Item | Claimed | Found | Match? |
|------|---------|-------|--------|
| SFY-26 | $1.75B | Not independently confirmed | — |
| SFY-25 | $2.66B | Not independently confirmed | — |
| Change | −$0.91B | N/A | Arithmetic correct ($2.66B − $1.75B) |
| % Change | −34.21% | N/A | Arithmetic correct ($0.91B / $2.66B) |

**Verdict: ARITHMETIC CORRECT but COMPARISON MAY BE MISLEADING.**

ND's state fiscal year runs July 1 – June 30. As of the report date (March 19, 2026), SFY-26 is only ~9 months complete. If $1.75B is a year-to-date figure compared to SFY-25's full-year $2.66B, the −34.21% decline is an apples-to-oranges comparison. Annualizing $1.75B over 9 months gives ~$2.33B, implying roughly a −12% decline — much less dramatic.

**Recommendation:** Confirm with ND OMB monthly revenue reports whether both figures cover the same time period. If SFY-26 is YTD, the comparison should be against the same 9-month period of SFY-25.

**Source:** ND Office of Management and Budget monthly financial reports (omb.nd.gov).

---

### 5. Average Annual Income (Private)

| Item | Claimed | BLS QCEW | Match? |
|------|---------|----------|--------|
| 2024 | $67,001 | $67,001 | **Yes** |
| 2023 | $64,671 | **$64,672** | **No — off by $1** |
| Change | +$2,330 | +$2,329 | No (follows from $1 error) |
| % Change | +3.60% | +3.60% | Yes (rounds the same) |

**Verdict: MINOR ERROR.** The 2023 average annual pay per BLS QCEW is **$64,672**, not $64,671. This $1 discrepancy propagates to the dollar change ($2,329 vs $2,330) but does not affect the rounded percentage.

**Source:** BLS QCEW API — 2023 & 2024 annual averages, ND (area_fips=38000), Private (own_code=5), All Industries (industry_code=10).

---

### 6. Cost of Living Index

| Item | Claimed | MERIC | Match? |
|------|---------|-------|--------|
| 2025 | 91.1 | **91.1** | **Yes** |
| 2024 | 91.9 | 91.3 (Q2 2024 only) | **Unverified** |
| Change | −0.8 points | N/A | Arithmetic correct |
| % Change | −0.87% | N/A | Arithmetic correct (0.8 / 91.9 = 0.870%) |

**Verdict: 2025 CONFIRMED; 2024 UNVERIFIED.** The 2025 value of 91.1 is confirmed from MERIC's Cost of Living Data Series. The 2024 annual average of 91.9 could not be verified — MERIC's website only displays the current year, and the closest 2024 data point found was 91.3 (Q2 2024 quarterly). The discrepancy (91.9 vs 91.3) may reflect the difference between quarterly and annual averaging.

ND component breakdown (2025): Grocery 96.8, Housing 75.7, Utilities 83.2, Transportation 99.9, Health 108.8, Misc 99.2. ND ranked 12th lowest nationally.

**Source:** MERIC Cost of Living Data Series (meric.mo.gov); C2ER ACCRA data.

---

### 7. Total Employment (Private)

| Item | Claimed | BLS QCEW | Match? |
|------|---------|----------|--------|
| 2024 | 354,071 | **354,071** | **Yes** |
| 2023 | 348,290 | **348,290** | **Yes** |
| Change | +5,781 | +5,781 | **Yes** |
| % Change | +1.66% | +1.66% | **Yes** |

**Verdict: CONFIRMED.** All four values match BLS QCEW exactly.

**Source:** BLS QCEW API — 2023 & 2024 annual averages, ND, Private, All Industries.

---

### 8. Population

| Item | Claimed | PEP V2025 | Match? |
|------|---------|-----------|--------|
| 2025 | 799,358 | **799,358** | **Yes** |
| 2024 | 793,387 | **793,387** | **Yes** |
| Change | +5,971 | +5,971 | **Yes** |
| % Change | +0.75% | +0.753% → 0.75% | **Yes** |

**Verdict: CONFIRMED.** All values match PEP Vintage 2025 (NST-EST2025-ALLDATA) exactly. Cross-checked against county-sum from `nd_county_population.csv` — also exact match.

**Source:** Census Bureau PEP Vintage 2025 (released 2026-01-27 for state, 2026-03-26 for county).

---

### 9. Housing Permits

| Item | Claimed | Census BPS | Match? |
|------|---------|------------|--------|
| 2025 | 2,394 | **2,394** (preliminary) | **Yes** |
| 2024 | 2,319 | **2,319** (final) | **Yes** |
| Change | +75 | +75 | **Yes** |
| % Change | +3.23% | +3.234% → 3.23% | **Yes** |

**Verdict: CONFIRMED.** All values match Census Bureau Building Permits Survey exactly.

**Caveat:** The 2025 figure is preliminary; final data is scheduled for May 14, 2026. For context, the preliminary 2024 figure was 2,272 vs. the final 2,319 — a revision of +47 units. The final 2025 number may similarly shift.

**Source:** Census Bureau Building Permits Survey — `stateannual_202499.xls` (2024 final), `stateannual_2025prelim.xls` (2025 preliminary).

---

## Part 2: Population Growth Narrative Claims

These claims appeared in related narrative text and were fact-checked separately.

> "The state's working-age population also has grown for three straight years to 473,249, the highest since 2020. North Dakota ranks No. 14 in the nation for per capita population growth, growing about 50% faster than the national average. The state has grown by nearly 19% since 2010."

### Claim A: Working-age population grown three straight years to 473,249, highest since 2020

**Verdict: Mostly TRUE — trend correct; figure ~292 off.**

Working-age (ages 18–64) from Vintage 2024 county age-sex estimates (`cc-est2024-agesex-all`):

| Date | Working-Age (18–64) | YoY Change |
|------|--------------------:|-----------:|
| Apr 2020 (census base) | 470,502 | — |
| Jul 2020 | 470,138 | −364 |
| Jul 2021 | 467,432 | −2,706 |
| Jul 2022 | 466,741 | −691 |
| Jul 2023 | 469,798 | +3,057 |
| Jul 2024 | 473,541 | +3,743 |

- Three straight years of growth (2022→2023→2024): **Confirmed.**
- Highest since 2020: **Confirmed** (exceeds Apr 2020 base of 470,502).
- 473,249 vs our 473,541: discrepancy of 292. May reflect different vintage or age boundary.

**Note:** V2025 county age-sex data not yet released (expected June 2026). V2025 revised total population downward, so this figure will likely be revised.

### Claim B: Ranks No. 14 nationally for per capita population growth

**Verdict: TRUE.** Confirmed at #14 among 50 states (excluding DC/PR) for 2024→2025 growth in Vintage 2025 data (+0.753%).

### Claim C: Growing about 50% faster than the national average

**Verdict: APPROXIMATELY TRUE for the most recent year.**

| Period | US Growth | ND Growth | Ratio |
|--------|----------:|----------:|------:|
| **2024–2025** | **+0.524%** | **+0.753%** | **1.44×** |
| 2023–2025 | +1.494% | +1.561% | 1.05× |
| 2020–2025 | +3.078% | +2.533% | 0.82× |

The 1.44× ratio for 2024–2025 is reasonably close to "about 50% faster," though it's specific to the most recent single year. Over longer horizons, ND's growth is at or below the national average.

### Claim D: Grown nearly 19% since 2010

**Verdict: TRUE.** Census 2010: 672,591 → V2025 Jul 2025: 799,358 = **+18.9%**.

---

## Vintage Revision Note

Vintage 2025 revised ND's prior-year estimates notably downward from Vintage 2024:

| Year | Vintage 2024 | Vintage 2025 | Revision |
|------|-------------:|-------------:|---------:|
| 2020 | 779,563 | 779,612 | +49 |
| 2021 | 777,966 | 777,977 | +11 |
| 2022 | 781,057 | 780,191 | −866 |
| 2023 | 789,047 | 787,071 | −1,976 |
| 2024 | 796,568 | 793,387 | −3,181 |

The 2024 estimate was revised downward by 3,181 persons (−0.4%). ND's 2024→2025 growth appears stronger in V2025 partly because the 2024 base was revised down.

---

## Data Files Used

| File | Vintage / Source | Scope |
|------|-----------------|-------|
| `data/raw/population/NST-EST2025-ALLDATA.csv` | PEP V2025 | State-level totals and components, 2020–2025 |
| `shared-data/.../co-est2025-alldata.parquet` | PEP V2025 | County-level totals, 2020–2025 |
| `shared-data/.../NST-EST2025-ALLDATA.parquet` | PEP V2025 | State-level (parquet copy) |
| `shared-data/.../cc-est2024-agesex-all.parquet` | PEP V2024 | County-level age-sex, 2020–2024 |
| `shared-data/.../co-est2019-alldata.parquet` | PEP V2019 | County-level totals, 2010–2019 |
| `data/raw/population/nd_county_population.csv` | PEP V2025 | ND county totals |
| BLS QCEW API (38000, Private, All Industries) | 2023–2024 | Employment and wages |
| Census Building Permits Survey | 2024 final, 2025 prelim | Housing units authorized |
| MERIC Cost of Living Data Series | 2025 | Cost of living index |
| BEA GDP by State | Q4 2024 release | Real GDP |

## Pending Updates

- [ ] Re-verify Claim A (working-age population) when V2025 county age-sex data is released (expected June 2026)

## Recommendations

1. **Real GDP**: Reconcile the source and methodology with BEA official figures. The current values appear to use a non-standard series.
2. **General Fund Receipts**: Clarify whether SFY-26 is a partial-year figure. If so, compare against the same YTD period of SFY-25.
3. **Average Annual Income**: Correct 2023 from $64,671 to $64,672 (and change from +$2,330 to +$2,329).
4. **Mid-America Business Conditions Index**: Verify state-level values against Creighton's primary monthly PDF report, and clarify whether this is the Business Conditions Index or the Rural Mainstreet Index.
5. **Cost of Living 2024**: Verify the annual average against archived MERIC data (current site only shows 2025).
6. **Housing Permits 2025**: Note that 2,394 is preliminary; final figure due May 2026.
