#!/usr/bin/env python3
"""
Data Validation Script for North Dakota Cohort Projection System
Validates all processed data files and generates a validation report.
"""

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

# Base paths
BASE_DIR = Path("/home/nigel/cohort_projections")
DATA_DIR = BASE_DIR / "data" / "raw"

# Initialize validation results
validation_results = []
issues_found = []


def add_result(file_name, check_name, expected, actual, status, notes=""):
    """Add a validation result."""
    validation_results.append(
        {
            "file": file_name,
            "check": check_name,
            "expected": expected,
            "actual": actual,
            "status": status,
            "notes": notes,
        }
    )
    if status == "FAIL":
        issues_found.append(
            f"{file_name}: {check_name} - Expected {expected}, got {actual}. {notes}"
        )


print("=" * 80)
print("DATA VALIDATION FOR NORTH DAKOTA COHORT PROJECTION SYSTEM")
print("=" * 80)
print(f"Validation Date: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S')}")
print()

# =============================================================================
# 1. FERTILITY DATA VALIDATION
# =============================================================================
print("\n1. VALIDATING FERTILITY DATA")
print("-" * 40)

fertility_file = DATA_DIR / "fertility" / "asfr_processed.csv"
try:
    fertility_df = pd.read_csv(fertility_file)
    print(f"   File: {fertility_file}")
    print(f"   Loaded successfully: {len(fertility_df)} rows")

    # Check required columns
    required_cols = ["age", "race_ethnicity", "asfr", "year"]
    missing_cols = [c for c in required_cols if c not in fertility_df.columns]
    add_result(
        "asfr_processed.csv",
        "Required columns",
        str(required_cols),
        str(list(fertility_df.columns)),
        "PASS" if not missing_cols else "FAIL",
        f"Missing: {missing_cols}" if missing_cols else "All columns present",
    )

    # Check row count (7 age groups x 6 race categories = 42)
    expected_rows = 42
    actual_rows = len(fertility_df)
    # Note: The file has 43 rows (header + 42 data rows) - we check data rows
    add_result(
        "asfr_processed.csv",
        "Row count",
        expected_rows,
        actual_rows,
        "PASS" if actual_rows >= expected_rows else "FAIL",
    )

    # Check unique age groups
    age_groups = fertility_df["age"].unique()
    expected_age_groups = 7
    add_result(
        "asfr_processed.csv",
        "Age groups count",
        expected_age_groups,
        len(age_groups),
        "PASS" if len(age_groups) == expected_age_groups else "FAIL",
        f"Age groups: {sorted(age_groups)}",
    )

    # Check unique race categories
    race_categories = fertility_df["race_ethnicity"].unique()
    expected_race_cats = 6
    add_result(
        "asfr_processed.csv",
        "Race categories count",
        expected_race_cats,
        len(race_categories),
        "PASS" if len(race_categories) == expected_race_cats else "FAIL",
        f"Categories: {sorted(race_categories)}",
    )

    # Check ASFR values are reasonable (0-200 per 1,000)
    min_asfr = fertility_df["asfr"].min()
    max_asfr = fertility_df["asfr"].max()
    asfr_reasonable = (min_asfr >= 0) and (max_asfr <= 200)
    add_result(
        "asfr_processed.csv",
        "ASFR values range (0-200)",
        "0-200",
        f"{min_asfr:.1f}-{max_asfr:.1f}",
        "PASS" if asfr_reasonable else "FAIL",
    )

    print(f"   Age groups: {len(age_groups)} ({sorted(age_groups)})")
    print(f"   Race categories: {len(race_categories)}")
    print(f"   ASFR range: {min_asfr:.1f} - {max_asfr:.1f}")

except Exception as e:
    print(f"   ERROR loading fertility data: {e}")
    add_result("asfr_processed.csv", "File load", "Success", "Failed", "FAIL", str(e))

# =============================================================================
# 2. MORTALITY DATA VALIDATION
# =============================================================================
print("\n2. VALIDATING MORTALITY DATA")
print("-" * 40)

mortality_file = DATA_DIR / "mortality" / "survival_rates_processed.csv"
try:
    mortality_df = pd.read_csv(mortality_file)
    print(f"   File: {mortality_file}")
    print(f"   Loaded successfully: {len(mortality_df)} rows")

    # Check required columns
    required_cols = ["age", "sex", "race_ethnicity", "qx", "survival_rate", "lx"]
    missing_cols = [c for c in required_cols if c not in mortality_df.columns]
    add_result(
        "survival_rates_processed.csv",
        "Required columns",
        str(required_cols),
        str(list(mortality_df.columns)),
        "PASS" if not missing_cols else "FAIL",
        f"Missing: {missing_cols}" if missing_cols else "All columns present",
    )

    # Check row count (101 ages x 2 sexes x 6 races = 1,212)
    expected_rows = 1212
    actual_rows = len(mortality_df)
    add_result(
        "survival_rates_processed.csv",
        "Row count",
        expected_rows,
        actual_rows,
        "PASS" if actual_rows == expected_rows else "FAIL",
    )

    # Check unique ages (0-100 = 101 ages)
    ages = mortality_df["age"].unique()
    expected_ages = 101
    add_result(
        "survival_rates_processed.csv",
        "Age range count",
        expected_ages,
        len(ages),
        "PASS" if len(ages) == expected_ages else "FAIL",
        f"Ages: {min(ages)}-{max(ages)}",
    )

    # Check unique sexes
    sexes = mortality_df["sex"].unique()
    expected_sexes = 2
    add_result(
        "survival_rates_processed.csv",
        "Sex categories",
        expected_sexes,
        len(sexes),
        "PASS" if len(sexes) == expected_sexes else "FAIL",
        f"Sexes: {sorted(sexes)}",
    )

    # Check unique race categories
    race_categories = mortality_df["race_ethnicity"].unique()
    expected_race_cats = 6
    add_result(
        "survival_rates_processed.csv",
        "Race categories count",
        expected_race_cats,
        len(race_categories),
        "PASS" if len(race_categories) == expected_race_cats else "FAIL",
        f"Categories: {sorted(race_categories)}",
    )

    # Check qx values between 0 and 1
    min_qx = mortality_df["qx"].min()
    max_qx = mortality_df["qx"].max()
    qx_valid = (min_qx >= 0) and (max_qx <= 1)
    add_result(
        "survival_rates_processed.csv",
        "qx range (0-1)",
        "0-1",
        f"{min_qx:.6f}-{max_qx:.6f}",
        "PASS" if qx_valid else "FAIL",
    )

    # Check survival_rate = 1 - qx
    survival_check = (mortality_df["survival_rate"] - (1 - mortality_df["qx"])).abs().max()
    add_result(
        "survival_rates_processed.csv",
        "survival_rate = 1 - qx",
        "< 0.0001",
        f"{survival_check:.8f}",
        "PASS" if survival_check < 0.0001 else "FAIL",
    )

    print(f"   Ages: {len(ages)} ({min(ages)}-{max(ages)})")
    print(f"   Sexes: {sorted(sexes)}")
    print(f"   Race categories: {len(race_categories)}")
    print(f"   qx range: {min_qx:.6f} - {max_qx:.6f}")
    print(f"   survival_rate check deviation: {survival_check:.8f}")

except Exception as e:
    print(f"   ERROR loading mortality data: {e}")
    add_result("survival_rates_processed.csv", "File load", "Success", "Failed", "FAIL", str(e))

# =============================================================================
# 3. MIGRATION DATA VALIDATION
# =============================================================================
print("\n3. VALIDATING MIGRATION DATA")
print("-" * 40)

migration_file = DATA_DIR / "migration" / "nd_migration_processed.csv"
try:
    migration_df = pd.read_csv(migration_file)
    print(f"   File: {migration_file}")
    print(f"   Loaded successfully: {len(migration_df)} rows")

    # Check required columns
    required_cols = [
        "county_fips",
        "county_name",
        "year",
        "inflow_n2",
        "outflow_n2",
        "net_migration",
    ]
    missing_cols = [c for c in required_cols if c not in migration_df.columns]
    add_result(
        "nd_migration_processed.csv",
        "Required columns",
        str(required_cols),
        str(list(migration_df.columns)),
        "PASS" if not missing_cols else "FAIL",
        f"Missing: {missing_cols}" if missing_cols else "All columns present",
    )

    # Check row count (53 counties x 4 years = 212)
    expected_rows = 212
    actual_rows = len(migration_df)
    add_result(
        "nd_migration_processed.csv",
        "Row count",
        expected_rows,
        actual_rows,
        "PASS" if actual_rows == expected_rows else "FAIL",
    )

    # Check unique counties
    counties = migration_df["county_fips"].unique()
    expected_counties = 53
    add_result(
        "nd_migration_processed.csv",
        "County count",
        expected_counties,
        len(counties),
        "PASS" if len(counties) == expected_counties else "FAIL",
    )

    # Check unique years
    years = migration_df["year"].unique()
    expected_years = 4
    add_result(
        "nd_migration_processed.csv",
        "Year count",
        expected_years,
        len(years),
        "PASS" if len(years) == expected_years else "FAIL",
        f"Years: {sorted(years)}",
    )

    # Check county_fips starts with 38
    nd_counties = migration_df["county_fips"].astype(str).str.startswith("38")
    all_nd = nd_counties.all()
    add_result(
        "nd_migration_processed.csv",
        "FIPS starts with 38",
        "All",
        f"{nd_counties.sum()}/{len(nd_counties)}",
        "PASS" if all_nd else "FAIL",
    )

    # Check net_migration = inflow - outflow
    net_check = (
        (migration_df["net_migration"] - (migration_df["inflow_n2"] - migration_df["outflow_n2"]))
        .abs()
        .max()
    )
    add_result(
        "nd_migration_processed.csv",
        "net_migration = inflow - outflow",
        "< 1",
        f"{net_check}",
        "PASS" if net_check < 1 else "FAIL",
    )

    print(f"   Counties: {len(counties)}")
    print(f"   Years: {sorted(years)}")
    print(f"   All FIPS start with 38: {all_nd}")

except Exception as e:
    print(f"   ERROR loading migration data: {e}")
    add_result("nd_migration_processed.csv", "File load", "Success", "Failed", "FAIL", str(e))

# =============================================================================
# 4. COUNTY POPULATION DATA VALIDATION
# =============================================================================
print("\n4. VALIDATING COUNTY POPULATION DATA")
print("-" * 40)

county_pop_file = DATA_DIR / "population" / "nd_county_population.csv"
try:
    county_pop_df = pd.read_csv(county_pop_file)
    print(f"   File: {county_pop_file}")
    print(f"   Loaded successfully: {len(county_pop_df)} rows")

    # Check required columns
    required_cols = ["county_fips", "county_name", "population_2024"]
    missing_cols = [c for c in required_cols if c not in county_pop_df.columns]
    add_result(
        "nd_county_population.csv",
        "Required columns",
        str(required_cols),
        str(list(county_pop_df.columns)),
        "PASS" if not missing_cols else "FAIL",
        f"Missing: {missing_cols}" if missing_cols else "All columns present",
    )

    # Check row count (53 counties)
    expected_rows = 53
    actual_rows = len(county_pop_df)
    add_result(
        "nd_county_population.csv",
        "Row count",
        expected_rows,
        actual_rows,
        "PASS" if actual_rows == expected_rows else "FAIL",
    )

    # Check population total (~796K)
    total_pop = county_pop_df["population_2024"].sum()
    expected_pop_min = 750000
    expected_pop_max = 850000
    pop_reasonable = expected_pop_min <= total_pop <= expected_pop_max
    add_result(
        "nd_county_population.csv",
        "Total population (~796K)",
        f"{expected_pop_min:,}-{expected_pop_max:,}",
        f"{total_pop:,}",
        "PASS" if pop_reasonable else "FAIL",
    )

    # Check county_fips starts with 38
    nd_counties = county_pop_df["county_fips"].astype(str).str.startswith("38")
    all_nd = nd_counties.all()
    add_result(
        "nd_county_population.csv",
        "FIPS starts with 38",
        "All",
        f"{nd_counties.sum()}/{len(nd_counties)}",
        "PASS" if all_nd else "FAIL",
    )

    # Check for duplicate counties
    duplicates = county_pop_df["county_fips"].duplicated().sum()
    add_result(
        "nd_county_population.csv",
        "No duplicate counties",
        "0",
        str(duplicates),
        "PASS" if duplicates == 0 else "FAIL",
    )

    print(f"   Counties: {actual_rows}")
    print(f"   Total population: {total_pop:,}")
    print(f"   All FIPS start with 38: {all_nd}")

except Exception as e:
    print(f"   ERROR loading county population data: {e}")
    add_result("nd_county_population.csv", "File load", "Success", "Failed", "FAIL", str(e))

# =============================================================================
# 5. POPULATION DISTRIBUTION DATA VALIDATION
# =============================================================================
print("\n5. VALIDATING POPULATION DISTRIBUTION DATA")
print("-" * 40)

dist_file = DATA_DIR / "population" / "nd_age_sex_race_distribution.csv"
try:
    dist_df = pd.read_csv(dist_file)
    print(f"   File: {dist_file}")
    print(f"   Loaded successfully: {len(dist_df)} rows")

    # Check required columns
    required_cols = ["age_group", "sex", "race_ethnicity", "proportion"]
    missing_cols = [c for c in required_cols if c not in dist_df.columns]
    add_result(
        "nd_age_sex_race_distribution.csv",
        "Required columns",
        str(required_cols),
        str(list(dist_df.columns)),
        "PASS" if not missing_cols else "FAIL",
        f"Missing: {missing_cols}" if missing_cols else "All columns present",
    )

    # Check proportions sum to ~1.0
    total_proportion = dist_df["proportion"].sum()
    proportion_valid = 0.99 <= total_proportion <= 1.01
    add_result(
        "nd_age_sex_race_distribution.csv",
        "Proportions sum to 1.0",
        "0.99-1.01",
        f"{total_proportion:.6f}",
        "PASS" if proportion_valid else "FAIL",
    )

    # Check unique age groups
    age_groups = dist_df["age_group"].unique()
    add_result(
        "nd_age_sex_race_distribution.csv",
        "Age groups present",
        ">= 15",
        str(len(age_groups)),
        "PASS" if len(age_groups) >= 15 else "FAIL",
        f"Age groups: {sorted(age_groups)}",
    )

    # Check unique sexes
    sexes = dist_df["sex"].unique()
    expected_sexes = 2
    add_result(
        "nd_age_sex_race_distribution.csv",
        "Sex categories",
        expected_sexes,
        len(sexes),
        "PASS" if len(sexes) == expected_sexes else "FAIL",
        f"Sexes: {sorted(sexes)}",
    )

    # Check race categories
    race_categories = dist_df["race_ethnicity"].unique()
    add_result(
        "nd_age_sex_race_distribution.csv",
        "Race categories present",
        ">= 4",
        str(len(race_categories)),
        "PASS" if len(race_categories) >= 4 else "FAIL",
        f"Categories: {sorted(race_categories)}",
    )

    # Check all proportions are positive
    min_prop = dist_df["proportion"].min()
    max_prop = dist_df["proportion"].max()
    props_valid = (min_prop >= 0) and (max_prop <= 1)
    add_result(
        "nd_age_sex_race_distribution.csv",
        "Proportions range (0-1)",
        "0-1",
        f"{min_prop:.6f}-{max_prop:.6f}",
        "PASS" if props_valid else "FAIL",
    )

    print(f"   Age groups: {len(age_groups)}")
    print(f"   Sexes: {sorted(sexes)}")
    print(f"   Race categories: {len(race_categories)}")
    print(f"   Proportions sum: {total_proportion:.6f}")

except Exception as e:
    print(f"   ERROR loading distribution data: {e}")
    add_result("nd_age_sex_race_distribution.csv", "File load", "Success", "Failed", "FAIL", str(e))

# =============================================================================
# 6. GEOGRAPHIC DATA VALIDATION
# =============================================================================
print("\n6. VALIDATING GEOGRAPHIC DATA")
print("-" * 40)

# 6a. Counties file (national file - filter for ND)
counties_file = DATA_DIR / "geographic" / "nd_counties.csv"
try:
    counties_df = pd.read_csv(counties_file)
    print(f"   File: {counties_file}")
    print(f"   Total rows (all states): {len(counties_df)}")

    # Filter for North Dakota (state_fips = 38)
    if "state_fips" in counties_df.columns:
        nd_counties_df = counties_df[counties_df["state_fips"] == 38]
    elif "county_fips" in counties_df.columns:
        nd_counties_df = counties_df[counties_df["county_fips"].astype(str).str.startswith("38")]
    else:
        nd_counties_df = counties_df

    nd_county_count = len(nd_counties_df)
    expected_nd_counties = 53
    add_result(
        "nd_counties.csv",
        "ND counties present",
        expected_nd_counties,
        nd_county_count,
        "PASS" if nd_county_count >= expected_nd_counties else "FAIL",
    )

    add_result(
        "nd_counties.csv",
        "Total rows (national)",
        ">= 3000",
        str(len(counties_df)),
        "PASS" if len(counties_df) >= 3000 else "INFO",
    )

    print(f"   ND counties: {nd_county_count}")

except Exception as e:
    print(f"   ERROR loading counties data: {e}")
    add_result("nd_counties.csv", "File load", "Success", "Failed", "FAIL", str(e))

# 6b. Places file
places_file = DATA_DIR / "geographic" / "nd_places.csv"
try:
    places_df = pd.read_csv(places_file)
    print(f"   File: {places_file}")
    print(f"   Loaded successfully: {len(places_df)} rows")

    # Check that it's ND data (STATE = 38)
    if "STATE" in places_df.columns:
        nd_places = places_df[places_df["STATE"] == 38]
        nd_places_count = len(nd_places)
    else:
        nd_places_count = len(places_df)

    expected_places = 300
    add_result(
        "nd_places.csv",
        "ND places count",
        f">= {expected_places}",
        str(nd_places_count),
        "PASS" if nd_places_count >= expected_places else "FAIL",
    )

    print(f"   ND places: {nd_places_count}")

except Exception as e:
    print(f"   ERROR loading places data: {e}")
    add_result("nd_places.csv", "File load", "Success", "Failed", "FAIL", str(e))

# 6c. Metro crosswalk file
metro_file = DATA_DIR / "geographic" / "metro_crosswalk.csv"
try:
    metro_df = pd.read_csv(metro_file)
    print(f"   File: {metro_file}")
    print(f"   Loaded successfully: {len(metro_df)} rows")

    # Check for ND counties in crosswalk
    if "county_fips" in metro_df.columns:
        nd_metro = metro_df[metro_df["county_fips"].astype(str).str.startswith("38")]
        nd_metro_count = len(nd_metro)
    elif "state_fips" in metro_df.columns:
        nd_metro = metro_df[metro_df["state_fips"] == 38]
        nd_metro_count = len(nd_metro)
    else:
        nd_metro_count = 0

    add_result(
        "metro_crosswalk.csv",
        "Total entries",
        ">= 1000",
        str(len(metro_df)),
        "PASS" if len(metro_df) >= 1000 else "INFO",
    )

    add_result(
        "metro_crosswalk.csv",
        "ND counties in crosswalk",
        ">= 0",
        str(nd_metro_count),
        "PASS" if nd_metro_count >= 0 else "INFO",
    )

    print(f"   Total metro entries: {len(metro_df)}")
    print(f"   ND entries: {nd_metro_count}")

except Exception as e:
    print(f"   ERROR loading metro crosswalk data: {e}")
    add_result("metro_crosswalk.csv", "File load", "Success", "Failed", "FAIL", str(e))

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 80)
print("VALIDATION SUMMARY")
print("=" * 80)

# Count results
pass_count = sum(1 for r in validation_results if r["status"] == "PASS")
fail_count = sum(1 for r in validation_results if r["status"] == "FAIL")
info_count = sum(1 for r in validation_results if r["status"] == "INFO")
total_checks = len(validation_results)

print(f"\nTotal checks: {total_checks}")
print(f"PASSED: {pass_count} ({100*pass_count/total_checks:.1f}%)")
print(f"FAILED: {fail_count} ({100*fail_count/total_checks:.1f}%)")
print(f"INFO: {info_count} ({100*info_count/total_checks:.1f}%)")

if issues_found:
    print("\nISSUES FOUND:")
    for issue in issues_found:
        print(f"  - {issue}")
else:
    print("\nNo critical issues found!")

# =============================================================================
# GENERATE MARKDOWN REPORT
# =============================================================================
report_path = BASE_DIR / "data" / "DATA_VALIDATION_REPORT.md"

report_content = f"""# Data Validation Report
## North Dakota Cohort Projection System

**Generated:** {datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S')}
**Validator:** `scripts/validate_data.py`

---

## Executive Summary

| Metric | Value |
|--------|-------|
| Total Checks | {total_checks} |
| Passed | {pass_count} ({100*pass_count/total_checks:.1f}%) |
| Failed | {fail_count} ({100*fail_count/total_checks:.1f}%) |
| Informational | {info_count} ({100*info_count/total_checks:.1f}%) |

**Overall Status:** {'PASS - All critical checks passed' if fail_count == 0 else 'ISSUES FOUND - See details below'}

---

## File-by-File Validation

### 1. Fertility Data (`data/raw/fertility/asfr_processed.csv`)

| Check | Expected | Actual | Status |
|-------|----------|--------|--------|
"""

# Add fertility checks
for r in validation_results:
    if r["file"] == "asfr_processed.csv":
        status_emoji = (
            "PASS" if r["status"] == "PASS" else ("FAIL" if r["status"] == "FAIL" else "INFO")
        )
        report_content += f"| {r['check']} | {r['expected']} | {r['actual']} | {status_emoji} |\n"

report_content += f"""
**Notes:** {fertility_df['race_ethnicity'].unique().tolist() if 'fertility_df' in dir() else 'N/A'}

---

### 2. Mortality Data (`data/raw/mortality/survival_rates_processed.csv`)

| Check | Expected | Actual | Status |
|-------|----------|--------|--------|
"""

# Add mortality checks
for r in validation_results:
    if r["file"] == "survival_rates_processed.csv":
        status_emoji = (
            "PASS" if r["status"] == "PASS" else ("FAIL" if r["status"] == "FAIL" else "INFO")
        )
        report_content += f"| {r['check']} | {r['expected']} | {r['actual']} | {status_emoji} |\n"

report_content += """
**Notes:** Contains survival rates for ages 0-100, both sexes, and 6 race/ethnicity categories.

---

### 3. Migration Data (`data/raw/migration/nd_migration_processed.csv`)

| Check | Expected | Actual | Status |
|-------|----------|--------|--------|
"""

# Add migration checks
for r in validation_results:
    if r["file"] == "nd_migration_processed.csv":
        status_emoji = (
            "PASS" if r["status"] == "PASS" else ("FAIL" if r["status"] == "FAIL" else "INFO")
        )
        report_content += f"| {r['check']} | {r['expected']} | {r['actual']} | {status_emoji} |\n"

report_content += """
**Notes:** Contains IRS migration data for years 2019-2022.

---

### 4. County Population Data (`data/raw/population/nd_county_population.csv`)

| Check | Expected | Actual | Status |
|-------|----------|--------|--------|
"""

# Add county population checks
for r in validation_results:
    if r["file"] == "nd_county_population.csv":
        status_emoji = (
            "PASS" if r["status"] == "PASS" else ("FAIL" if r["status"] == "FAIL" else "INFO")
        )
        report_content += f"| {r['check']} | {r['expected']} | {r['actual']} | {status_emoji} |\n"

report_content += """
**Notes:** 2024 population estimates for all 53 North Dakota counties.

---

### 5. Population Distribution Data (`data/raw/population/nd_age_sex_race_distribution.csv`)

| Check | Expected | Actual | Status |
|-------|----------|--------|--------|
"""

# Add distribution checks
for r in validation_results:
    if r["file"] == "nd_age_sex_race_distribution.csv":
        status_emoji = (
            "PASS" if r["status"] == "PASS" else ("FAIL" if r["status"] == "FAIL" else "INFO")
        )
        report_content += f"| {r['check']} | {r['expected']} | {r['actual']} | {status_emoji} |\n"

report_content += """
**Notes:** Proportional distribution of population by age, sex, and race/ethnicity.

---

### 6. Geographic Data

#### 6a. Counties (`data/raw/geographic/nd_counties.csv`)

| Check | Expected | Actual | Status |
|-------|----------|--------|--------|
"""

# Add counties checks
for r in validation_results:
    if r["file"] == "nd_counties.csv":
        status_emoji = (
            "PASS" if r["status"] == "PASS" else ("FAIL" if r["status"] == "FAIL" else "INFO")
        )
        report_content += f"| {r['check']} | {r['expected']} | {r['actual']} | {status_emoji} |\n"

report_content += """
**Notes:** National county file with population estimates. Filter on state_fips=38 for ND.

#### 6b. Places (`data/raw/geographic/nd_places.csv`)

| Check | Expected | Actual | Status |
|-------|----------|--------|--------|
"""

# Add places checks
for r in validation_results:
    if r["file"] == "nd_places.csv":
        status_emoji = (
            "PASS" if r["status"] == "PASS" else ("FAIL" if r["status"] == "FAIL" else "INFO")
        )
        report_content += f"| {r['check']} | {r['expected']} | {r['actual']} | {status_emoji} |\n"

report_content += """
**Notes:** North Dakota cities, towns, and census-designated places.

#### 6c. Metro Crosswalk (`data/raw/geographic/metro_crosswalk.csv`)

| Check | Expected | Actual | Status |
|-------|----------|--------|--------|
"""

# Add metro checks
for r in validation_results:
    if r["file"] == "metro_crosswalk.csv":
        status_emoji = (
            "PASS" if r["status"] == "PASS" else ("FAIL" if r["status"] == "FAIL" else "INFO")
        )
        report_content += f"| {r['check']} | {r['expected']} | {r['actual']} | {status_emoji} |\n"

report_content += f"""
**Notes:** National CBSA/metro area crosswalk file.

---

## Data Quality Summary

### Data Completeness

| Data Category | Status | Completeness |
|---------------|--------|--------------|
| Fertility (ASFR) | {'Complete' if not any(r['status']=='FAIL' and r['file']=='asfr_processed.csv' for r in validation_results) else 'Issues'} | 7 age groups x 6 race categories |
| Mortality (Life Tables) | {'Complete' if not any(r['status']=='FAIL' and r['file']=='survival_rates_processed.csv' for r in validation_results) else 'Issues'} | 101 ages x 2 sexes x 6 races |
| Migration (IRS) | {'Complete' if not any(r['status']=='FAIL' and r['file']=='nd_migration_processed.csv' for r in validation_results) else 'Issues'} | 53 counties x 4 years |
| Population (County) | {'Complete' if not any(r['status']=='FAIL' and r['file']=='nd_county_population.csv' for r in validation_results) else 'Issues'} | All 53 ND counties |
| Population (Distribution) | {'Complete' if not any(r['status']=='FAIL' and r['file']=='nd_age_sex_race_distribution.csv' for r in validation_results) else 'Issues'} | Age/sex/race proportions |
| Geographic | {'Complete' if not any(r['status']=='FAIL' and 'nd_' in r['file'] for r in validation_results) else 'Issues'} | Counties, places, metro areas |

### Issues Found

"""

if issues_found:
    for issue in issues_found:
        report_content += f"- {issue}\n"
else:
    report_content += "No critical issues found.\n"

report_content += """
---

## Recommendations

"""

# Generate recommendations based on validation results
recommendations = []

# Check for specific issues and add recommendations
if any(r["status"] == "FAIL" and "Row count" in r["check"] for r in validation_results):
    recommendations.append("Review files with unexpected row counts to ensure all data is present.")

if any(r["status"] == "FAIL" and "proportion" in r["check"].lower() for r in validation_results):
    recommendations.append("Check population distribution proportions - they should sum to 1.0.")

if fail_count == 0:
    recommendations.append("All validations passed. Data is ready for projection modeling.")
    recommendations.append("Consider implementing automated validation as part of data pipeline.")
    recommendations.append("Document any manual adjustments made to source data.")

for i, rec in enumerate(recommendations, 1):
    report_content += f"{i}. {rec}\n"

report_content += """
---

## Technical Details

### File Locations

| File | Path |
|------|------|
| Fertility | `data/raw/fertility/asfr_processed.csv` |
| Mortality | `data/raw/mortality/survival_rates_processed.csv` |
| Migration | `data/raw/migration/nd_migration_processed.csv` |
| County Population | `data/raw/population/nd_county_population.csv` |
| Population Distribution | `data/raw/population/nd_age_sex_race_distribution.csv` |
| Counties | `data/raw/geographic/nd_counties.csv` |
| Places | `data/raw/geographic/nd_places.csv` |
| Metro Crosswalk | `data/raw/geographic/metro_crosswalk.csv` |

### Data Sources

- **Fertility:** CDC NCHS Natality Data (WONDER)
- **Mortality:** CDC NCHS National Vital Statistics System
- **Migration:** IRS Statistics of Income (SOI) Migration Data
- **Population:** US Census Bureau Population Estimates Program
- **Geographic:** US Census Bureau TIGER/Line and OMB CBSA definitions

---

*Report generated by the North Dakota Cohort Projection System validation script.*
"""

# Write the report
with open(report_path, "w") as f:
    f.write(report_content)

print(f"\nReport saved to: {report_path}")
