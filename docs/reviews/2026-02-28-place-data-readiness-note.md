# 2026-02-28 Place Data Readiness Note (PP3-S02)

| Attribute | Value |
|-----------|-------|
| **Date** | 2026-02-28 |
| **Reviewer** | Codex (GPT-5) |
| **Scope** | PP3-S02 historical place data readiness check for Phase 1 city/place scoping |
| **Target Window** | 2000-2024 annual |
| **Status** | Complete (readiness with defined gaps) |
| **Related ADR** | ADR-033 |

---

## 1. Data Sources Checked

From the shared Census archive (`~/workspace/shared-data/census/popest/`):

| Dataset ID | Vintage | ND Place Rows | Years Available | Notes |
|------------|---------|---------------|-----------------|-------|
| `sub-est00int` | 2000-2010 | 357 | 2000-2010 (`POPESTIMATE2000`..`POPESTIMATE2010`) | Intercensal place totals |
| `sub-est2020int` | 2010-2020 | 357 | 2010-2019 (`POPESTIMATE2010`..`POPESTIMATE2019`) | Intercensal revisions through 2020 census cycle |
| `sub-est2024` | 2020-2024 | 355 | 2020-2024 (`POPESTIMATE2020`..`POPESTIMATE2024`) | Current postcensal place totals |

## 2. Coverage Result

Using handoff windows `2000-2009` (sub-est00int), `2010-2019` (sub-est2020int), and `2020-2024` (sub-est2024):

- Year coverage: **2000-2024 continuous** (`25` years)
- Total long-format rows: **8,915**
- Unique ND places in any year: **357**
- Per-year unique-place range: **355 to 357**
- Missing population cells: **0**

## 3. Identified Gaps and Required Rules

### Gap A: Place universe changes by 2020-2024

- `sub-est2010/2020int` has 357 ND places; `sub-est2024` has 355.
- Two historical places are no longer in the active 2020-2024 file:
  - `04740` Bantry city (2019 pop: 7)
  - `14140` Churchs Ferry city (2019 pop: 9)

**Required rule**: keep these as historical-only backtest records and exclude them from forward projection output universe.

### Gap B: County mapping missing in all place totals files

- In ND place files, `COUNTY` is uniformly `000` (no direct place->county assignment).

**Required rule**: build authoritative place->county mapping externally (handled in PP3-S03).

### Gap C: Local place reference schema mismatch

- `data/raw/geographic/nd_places.csv` is in Census wide format (`STATE`, `PLACE`, `NAME`, `POPESTIMATE*`) and does not match loader-required columns (`state_fips`, `place_fips`, `place_name`, `county_fips`).

**Required rule**: create/maintain a standardized place reference file (or processed derivative) that satisfies loader schema before implementation.

### Gap D: 2010-2019 vintage choice must be explicit

- `sub-est2019_all` and `sub-est2020int` both cover 2010-2019 but differ materially in aggregate ND totals (difference grows from +284 in 2010 to +8,537 in 2019 for `sub-est2020int` vs `sub-est2019_all`).

**Required rule**: lock `sub-est2020int` as canonical for 2010-2019 in Phase 1 to align with intercensal revisions.

## 4. Readiness Verdict

- **History coverage requirement (2000-2024): MET**.
- **Implementation readiness: PARTIAL** pending PP3-S03 mapping artifact and standardized place reference schema.

---

| Attribute | Value |
|-----------|-------|
| **Last Updated** | 2026-02-28 |
| **Version** | 1.0 |
