"""
Test script for extracting structured data from ND Vital Events Summary (VES) PDFs.

This script demonstrates and validates PDF table extraction from the 2024 VES
using pdfplumber. It tests two approaches:
  1. Text-based parsing (RECOMMENDED for most pages): Uses extract_text() with
     known county names for fast, reliable whitespace-delimited parsing.
  2. Position-aware parsing (REQUIRED for sparse pages): Uses character-level
     x-positions to assign values to the correct year columns, handling cases
     where cells are blank (not NR, just empty) in the original PDF.

Findings:
  - Text-based parsing works on 21/24 county-year tables (all non-sparse ones)
  - Position-aware parsing handles ALL 24 tables including sparse pages
  - pdfplumber's extract_table() with "text" strategy fails on death rates and
    other pages where wider numeric values cause column misalignment
  - NR (Not Reportable) values appear frequently in small-county tables
  - Multi-word county names (GOLDEN VALLEY, GRAND FORKS) are handled correctly
  - Some pages use "County Average" instead of "Total" for the summary row
  - A few pages have genuinely blank cells (not NR, not 0) where the PDF omits
    the value entirely; these require position-aware parsing

VES PDF Structure (2024 edition, 46 pages):
  The VES contains county-level vital statistics tables, each with 53 county rows
  plus a state total, across 15 years (2010-2024). Key pages (0-indexed):

  Page 14: Census data (2020 population by county, 4 columns)
  Page 15: Resident births (counts)
  Page 16: Resident birth rates
  Page 17: Resident fertility rates
  Page 20: Teenage births
  Page 26: Out-of-wedlock births
  Page 30: Low weight births
  Page 32: Infant deaths
  Page 34: Neonatal deaths
  Page 36: Fetal deaths
  Page 38: Resident deaths (counts)
  Page 39: Resident death rates
  Page 40: Childhood & adolescent deaths
  Page 42: Marriages
  Page 44: Divorces

  Pages 4-5: Summary data for 2024 (single-year, multi-metric)
  Pages 9-10: Summary data for 2020-2024 (5-year, multi-metric)

Usage:
  python scripts/data/test_ves_extraction.py
  python scripts/data/test_ves_extraction.py --pdf data/raw/2024VES.pdf
  python scripts/data/test_ves_extraction.py --verbose
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd
import pdfplumber

# ── Constants ──────────────────────────────────────────────────────────────────

ND_COUNTIES: list[str] = [
    "ADAMS", "BARNES", "BENSON", "BILLINGS", "BOTTINEAU", "BOWMAN", "BURKE",
    "BURLEIGH", "CASS", "CAVALIER", "DICKEY", "DIVIDE", "DUNN", "EDDY",
    "EMMONS", "FOSTER", "GOLDEN VALLEY", "GRAND FORKS", "GRANT", "GRIGGS",
    "HETTINGER", "KIDDER", "LAMOURE", "LOGAN", "McHENRY", "McINTOSH",
    "McKENZIE", "McLEAN", "MERCER", "MORTON", "MOUNTRAIL", "NELSON",
    "OLIVER", "PEMBINA", "PIERCE", "RAMSEY", "RANSOM", "RENVILLE",
    "RICHLAND", "ROLETTE", "SARGENT", "SHERIDAN", "SIOUX", "SLOPE",
    "STARK", "STEELE", "STUTSMAN", "TOWNER", "TRAILL", "WALSH", "WARD",
    "WELLS", "WILLIAMS",
]

YEARS_2010_2024: list[str] = [str(y) for y in range(2010, 2025)]

# Page index (0-based) to table label mapping for county x year tables
COUNTY_YEAR_PAGES: dict[int, str] = {
    15: "births",
    16: "birth_rates",
    17: "fertility_rates",
    20: "teenage_births",
    21: "teenage_birth_rates",
    22: "teenage_birth_ratios",
    26: "out_of_wedlock_births",
    27: "out_of_wedlock_birth_ratios",
    30: "low_weight_births",
    31: "low_weight_birth_ratios",
    32: "infant_deaths",
    33: "infant_death_ratios",
    34: "neonatal_deaths",
    35: "neonatal_death_ratios",
    36: "fetal_deaths",
    37: "fetal_death_ratios",
    38: "deaths",
    39: "death_rates",
    40: "childhood_adolescent_deaths",
    41: "childhood_adolescent_death_rates",
    42: "marriages",
    43: "marriage_rates",
    44: "divorces",
    45: "divorce_rates",
}

# Summary row labels used across different pages
SUMMARY_ROW_NAMES: list[str] = ["Total", "County Average"]


# ── Core parsing functions ────────────────────────────────────────────────────


def _get_search_names(include_total: bool = True) -> list[str]:
    """Return county names plus optional summary row names, sorted longest first."""
    names = list(ND_COUNTIES)
    if include_total:
        names.extend(SUMMARY_ROW_NAMES)
    # Sort by length descending so "GRAND FORKS" matches before "GRAND"
    names.sort(key=len, reverse=True)
    return names


def parse_county_year_table(
    page: pdfplumber.page.Page,
    years: list[str] | None = None,
    include_total: bool = True,
) -> pd.DataFrame:
    """Parse a county-by-year table from a VES PDF page using text extraction.

    This is the fast approach for pages where all rows have the expected number
    of values. For pages with blank cells (e.g., low_weight_births), use
    parse_county_year_table_positional() instead.

    Parameters
    ----------
    page : pdfplumber.page.Page
        A pdfplumber page object.
    years : list[str], optional
        Expected year column headers. Defaults to 2010-2024.
    include_total : bool
        Whether to include the state total / county average row.

    Returns
    -------
    pd.DataFrame
        DataFrame with 'county' column and one column per year.
        NR values are preserved as strings; use convert_nr_to_nan() to clean.
    """
    if years is None:
        years = YEARS_2010_2024

    text = page.extract_text()
    if not text:
        return pd.DataFrame()

    lines = text.split("\n")
    search_names = _get_search_names(include_total)

    records: list[dict[str, str]] = []
    for line in lines:
        line = line.strip()
        if not line:
            continue

        matched_county = None
        for county in search_names:
            if line.startswith(county):
                rest = line[len(county):].strip()
                if rest and (
                    rest[0].isdigit() or rest.startswith(("NR", "-", "0"))
                ):
                    matched_county = county
                    break

        if matched_county:
            values_str = line[len(matched_county):].strip()
            values = values_str.split()
            if len(values) == len(years):
                # Normalize summary row names to "Total"
                county_name = (
                    "Total" if matched_county in SUMMARY_ROW_NAMES else matched_county
                )
                record = {"county": county_name}
                for year, val in zip(years, values, strict=False):
                    record[year] = val
                records.append(record)

    return pd.DataFrame(records)


def parse_county_year_table_positional(
    page: pdfplumber.page.Page,
    years: list[str] | None = None,
    include_total: bool = True,
    col_tolerance: float = 15.0,
) -> pd.DataFrame:
    """Parse a county-by-year table using character x-positions for column alignment.

    This approach reads individual character positions from the PDF and assigns
    each value to the nearest year column based on x-coordinate proximity.
    It correctly handles pages where some cells are genuinely blank (not NR),
    filling them with NaN.

    Parameters
    ----------
    page : pdfplumber.page.Page
        A pdfplumber page object.
    years : list[str], optional
        Expected year column headers. Defaults to 2010-2024.
    include_total : bool
        Whether to include the state total / county average row.
    col_tolerance : float
        Maximum x-distance (points) for a value to be assigned to a column.

    Returns
    -------
    pd.DataFrame
        DataFrame with 'county' column and one column per year.
        Missing cells are NaN; NR values are preserved as strings.
    """
    if years is None:
        years = YEARS_2010_2024

    chars = page.chars
    if not chars:
        return pd.DataFrame()

    # Group characters by row (y-position, rounded)
    rows_by_y: dict[float, list[dict]] = defaultdict(list)
    for c in chars:
        y = round(c["top"], 0)
        rows_by_y[y].append(c)

    def _chars_to_words(char_list: list[dict]) -> list[tuple[float, str]]:
        """Group adjacent characters into words, returning (x0, word) pairs."""
        sorted_chars = sorted(char_list, key=lambda x: x["x0"])
        words: list[tuple[float, str]] = []
        current_word = ""
        current_x0: float | None = None
        prev_x1 = 0.0

        for c in sorted_chars:
            if current_word and c["x0"] - prev_x1 > 3:
                if current_x0 is not None:
                    words.append((current_x0, current_word))
                current_word = c["text"]
                current_x0 = c["x0"]
            else:
                if not current_word:
                    current_x0 = c["x0"]
                current_word += c["text"]
            prev_x1 = c["x0"] + c.get("width", 5)

        if current_word and current_x0 is not None:
            words.append((current_x0, current_word))
        return words

    # Step 1: Find the header row to get column x-positions
    col_x_positions: dict[str, float] = {}
    for y in sorted(rows_by_y.keys()):
        words = _chars_to_words(rows_by_y[y])
        word_texts = [w[1] for w in words]
        if "2010" in word_texts and "2024" in word_texts:
            for x, text in words:
                if text in years:
                    col_x_positions[text] = x
            break

    if not col_x_positions or len(col_x_positions) != len(years):
        # Fall back to text-based parsing
        return parse_county_year_table(page, years, include_total)

    # Step 2: Parse data rows
    search_names = _get_search_names(include_total)
    records: list[dict[str, str]] = []

    for y in sorted(rows_by_y.keys()):
        words = _chars_to_words(rows_by_y[y])
        if not words:
            continue

        # Reconstruct the line text to match county names
        line_text = " ".join(w[1] for w in words)

        matched_county = None
        for county in search_names:
            if line_text.startswith(county):
                rest = line_text[len(county):].strip()
                if rest and (
                    rest[0].isdigit() or rest.startswith(("NR", "-", "0"))
                ):
                    matched_county = county
                    break

        if not matched_county:
            continue

        county_name = (
            "Total" if matched_county in SUMMARY_ROW_NAMES else matched_county
        )

        # Skip words that are part of the county name
        county_words = matched_county.split()
        data_words = words[len(county_words):]

        # Assign each data word to the nearest year column
        record: dict[str, str] = {"county": county_name}
        for x, val in data_words:
            best_year = None
            best_dist = float("inf")
            for yr, yr_x in col_x_positions.items():
                dist = abs(x - yr_x)
                if dist < best_dist:
                    best_dist = dist
                    best_year = yr
            if best_year and best_dist < col_tolerance:
                record[best_year] = val

        records.append(record)

    # Build DataFrame, filling missing years with NaN
    df = pd.DataFrame(records)
    for yr in years:
        if yr not in df.columns:
            df[yr] = pd.NA

    # Reorder columns
    cols = ["county"] + [yr for yr in years if yr in df.columns]
    return df[cols]


def convert_nr_to_nan(
    df: pd.DataFrame, year_cols: list[str] | None = None
) -> pd.DataFrame:
    """Convert NR values to NaN and cast year columns to float.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame from parse_county_year_table().
    year_cols : list[str], optional
        Columns to convert. Defaults to all non-county columns.

    Returns
    -------
    pd.DataFrame
        DataFrame with numeric year columns (NR -> NaN).
    """
    if year_cols is None:
        year_cols = [c for c in df.columns if c != "county"]

    df = df.copy()
    for col in year_cols:
        if col in df.columns:
            df[col] = df[col].replace("NR", pd.NA)
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def parse_census_data(page: pdfplumber.page.Page) -> pd.DataFrame:
    """Parse the census population data page (page index 14).

    This page has a different format: county, total population, females 15-44,
    females 15-19, children under 20.

    Returns
    -------
    pd.DataFrame
        DataFrame with county and population columns.
    """
    text = page.extract_text()
    if not text:
        return pd.DataFrame()

    lines = text.split("\n")
    search_names = _get_search_names(include_total=True)

    columns = [
        "county", "total_pop", "females_15_44", "females_15_19", "children_under_20",
    ]
    records: list[dict[str, str]] = []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        for county in search_names:
            if line.startswith(county):
                rest = line[len(county):].strip()
                if rest and (rest[0].isdigit() or rest.startswith("NR")):
                    values = rest.replace(",", "").split()
                    if len(values) == 4:
                        county_name = (
                            "Total"
                            if county in SUMMARY_ROW_NAMES
                            else county
                        )
                        record = {"county": county_name}
                        for col, val in zip(columns[1:], values, strict=False):
                            record[col] = val
                        records.append(record)
                    break

    df = pd.DataFrame(records)
    for col in columns[1:]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


# ── Test functions ────────────────────────────────────────────────────────────


def test_county_year_extraction(
    pdf: pdfplumber.PDF, verbose: bool = False
) -> dict[str, bool]:
    """Test extraction of all county x year tables using both parsing methods.

    Returns a dict mapping table label to pass/fail boolean.
    """
    results: dict[str, bool] = {}

    for page_idx, label in COUNTY_YEAR_PAGES.items():
        page = pdf.pages[page_idx]

        # Try text-based first, then fall back to positional
        df = parse_county_year_table(page)
        method = "text"

        n_counties = len(df[df["county"] != "Total"]) if not df.empty else 0
        n_cols = len([c for c in df.columns if c != "county"])
        has_total = "Total" in df["county"].values if not df.empty else False

        text_passed = n_counties == 53 and n_cols == 15 and has_total

        if not text_passed:
            # Fall back to positional parsing
            df = parse_county_year_table_positional(page)
            method = "positional"
            n_counties = len(df[df["county"] != "Total"]) if not df.empty else 0
            n_cols = len([c for c in df.columns if c != "county"])
            has_total = "Total" in df["county"].values if not df.empty else False

        passed = n_counties == 53 and n_cols == 15 and has_total

        # Count NR values
        nr_count = 0
        blank_count = 0
        for col in YEARS_2010_2024:
            if col in df.columns:
                nr_count += (df[col] == "NR").sum()
                blank_count += df[col].isna().sum()

        status = "PASS" if passed else "FAIL"
        method_tag = f" [{method}]" if method == "positional" else ""
        blank_str = f", {blank_count} blank" if blank_count > 0 else ""
        print(
            f"  [{status}] Page {page_idx:2d} ({label:35s}): "
            f"{n_counties} counties, {n_cols} years, "
            f"total={'yes' if has_total else 'NO'}, "
            f"{nr_count} NR{blank_str}{method_tag}"
        )

        if verbose and passed:
            df_numeric = convert_nr_to_nan(df)
            total_row = df_numeric[df_numeric["county"] == "Total"]
            if not total_row.empty:
                val_2024 = total_row["2024"].values[0]
                val_2010 = total_row["2010"].values[0]
                print(f"           State total: 2010={val_2010}, 2024={val_2024}")

        if not passed:
            print(
                f"           ISSUE: Expected 53 counties, 15 years; "
                f"got {n_counties}, {n_cols}"
            )

        results[label] = passed

    return results


def test_census_extraction(pdf: pdfplumber.PDF, verbose: bool = False) -> bool:
    """Test extraction of the census data page."""
    page = pdf.pages[14]
    df = parse_census_data(page)

    n_counties = len(df[df["county"] != "Total"])
    has_total = "Total" in df["county"].values
    n_cols = len(df.columns) - 1  # exclude county

    passed = n_counties == 53 and n_cols == 4

    status = "PASS" if passed else "FAIL"
    print(
        f"  [{status}] Page 14 (census_data                       ): "
        f"{n_counties} counties, {n_cols} data columns, "
        f"total={'yes' if has_total else 'NO'}"
    )

    if verbose and passed:
        total_row = df[df["county"] == "Total"]
        if not total_row.empty:
            print(
                f"           State total pop: "
                f"{int(total_row['total_pop'].values[0]):,}"
            )

    return passed


def test_text_vs_table_comparison(pdf: pdfplumber.PDF) -> None:
    """Compare text-based vs pdfplumber table-based extraction approaches."""
    print("\n  Comparing text-based vs pdfplumber extract_table:")
    print("  " + "-" * 70)

    text_strategy_settings = {
        "vertical_strategy": "text",
        "horizontal_strategy": "text",
    }

    test_cases = [
        (15, "Births (clean page)"),
        (39, "Death rates (problematic for extract_table)"),
    ]

    for page_idx, desc in test_cases:
        page = pdf.pages[page_idx]

        # Text-based
        df_text = parse_county_year_table(page)
        text_ok = len(df_text) == 54  # 53 counties + total

        # pdfplumber extract_table
        table = page.extract_table(table_settings=text_strategy_settings)
        table_ok = False
        table_issue = ""
        if table:
            data_rows = [
                r for r in table if any(cell and cell.strip() for cell in r)
            ]
            for row in data_rows:
                if row[0] and "ADAMS" in str(row[0]):
                    if str(row[0]).strip() == "ADAMS":
                        table_ok = True
                    else:
                        table_issue = f"col[0]='{row[0]}'"
                    break
        else:
            table_issue = "no table extracted"

        text_status = "PASS" if text_ok else "FAIL"
        table_status = "PASS" if table_ok else "FAIL"
        print(f"  {desc}:")
        print(f"    Text-based:         [{text_status}] {len(df_text)} rows")
        issue_str = f" ({table_issue})" if table_issue else ""
        print(f"    extract_table:      [{table_status}]{issue_str}")


def test_positional_vs_text(pdf: pdfplumber.PDF) -> None:
    """Compare text-based vs positional parsing on problematic pages."""
    print("\n  Comparing text-based vs positional on sparse pages:")
    print("  " + "-" * 70)

    # These pages have blank cells that trip up text-based parsing
    sparse_pages = [
        (30, "Low weight births"),
        (44, "Divorces"),
        (45, "Divorce rates"),
    ]

    for page_idx, desc in sparse_pages:
        page = pdf.pages[page_idx]

        # Text-based
        df_text = parse_county_year_table(page)
        n_text = len(df_text[df_text["county"] != "Total"]) if not df_text.empty else 0
        has_total_text = (
            "Total" in df_text["county"].values if not df_text.empty else False
        )

        # Positional
        df_pos = parse_county_year_table_positional(page)
        n_pos = len(df_pos[df_pos["county"] != "Total"]) if not df_pos.empty else 0
        has_total_pos = (
            "Total" in df_pos["county"].values if not df_pos.empty else False
        )

        # Count blanks in positional
        blank_count = 0
        for col in YEARS_2010_2024:
            if col in df_pos.columns:
                blank_count += df_pos[col].isna().sum()

        text_status = "PASS" if n_text == 53 and has_total_text else "FAIL"
        pos_status = "PASS" if n_pos == 53 and has_total_pos else "FAIL"

        print(f"  Page {page_idx} ({desc}):")
        print(f"    Text-based:    [{text_status}] {n_text} counties, total={has_total_text}")
        print(
            f"    Positional:    [{pos_status}] {n_pos} counties, total={has_total_pos}"
            f", {blank_count} blank cells"
        )


def test_specific_values(pdf: pdfplumber.PDF) -> bool:
    """Validate specific known values from the VES."""
    print("\n  Spot-checking known values:")
    print("  " + "-" * 70)

    all_passed = True

    # Known values from visual inspection of PDF
    checks = [
        (15, "ADAMS", "2010", "19", "births"),
        (15, "CASS", "2024", "2394", "births"),
        (15, "Total", "2024", "9622", "births"),
        (15, "BILLINGS", "2010", "NR", "births (NR value)"),
        (38, "ADAMS", "2024", "33", "deaths"),
        (38, "CASS", "2024", "1297", "deaths"),
        (38, "Total", "2024", "6969", "deaths"),
        (17, "Total", "2024", "71.42", "fertility rate"),
        (17, "CASS", "2010", "50.93", "fertility rate"),
        (39, "ADAMS", "2010", "1409.091", "death rate"),
    ]

    for page_idx, county, year, expected, desc in checks:
        page = pdf.pages[page_idx]
        df = parse_county_year_table(page)
        row = df[df["county"] == county]

        if row.empty:
            print(
                f"  [FAIL] {desc}: county '{county}' not found on page {page_idx}"
            )
            all_passed = False
            continue

        actual = row[year].values[0]
        passed = str(actual) == expected
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_passed = False
        print(
            f"  [{status}] {county} {year} {desc}: expected={expected}, got={actual}"
        )

    return all_passed


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(description="Test VES PDF extraction")
    parser.add_argument(
        "--pdf",
        type=Path,
        default=Path("data/raw/2024VES.pdf"),
        help="Path to the VES PDF file",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Show detailed output"
    )
    args = parser.parse_args()

    pdf_path = args.pdf
    if not pdf_path.is_absolute():
        # Try relative to project root
        project_root = Path(__file__).resolve().parent.parent.parent
        pdf_path = project_root / pdf_path

    if not pdf_path.exists():
        print(f"ERROR: PDF not found at {pdf_path}")
        return 1

    print(f"Testing VES PDF extraction: {pdf_path}")
    print(f"Using pdfplumber {pdfplumber.__version__}")
    print()

    with pdfplumber.open(pdf_path) as pdf:
        print(f"PDF has {len(pdf.pages)} pages")
        print()

        # Test 1: All county x year tables (with automatic fallback to positional)
        print("=" * 80)
        print("TEST 1: County x Year Table Extraction")
        print("=" * 80)
        results = test_county_year_extraction(pdf, verbose=args.verbose)
        n_pass = sum(results.values())
        n_total = len(results)
        print(f"\n  Result: {n_pass}/{n_total} tables extracted successfully")

        # Test 2: Census data
        print()
        print("=" * 80)
        print("TEST 2: Census Data Extraction")
        print("=" * 80)
        census_ok = test_census_extraction(pdf, verbose=args.verbose)

        # Test 3: Text vs extract_table comparison
        print()
        print("=" * 80)
        print("TEST 3: Text-based vs pdfplumber extract_table() Comparison")
        print("=" * 80)
        test_text_vs_table_comparison(pdf)

        # Test 4: Positional vs text on sparse pages
        print()
        print("=" * 80)
        print("TEST 4: Positional vs Text Parsing on Sparse Pages")
        print("=" * 80)
        test_positional_vs_text(pdf)

        # Test 5: Known value validation
        print()
        print("=" * 80)
        print("TEST 5: Known Value Validation")
        print("=" * 80)
        values_ok = test_specific_values(pdf)

        # Summary
        print()
        print("=" * 80)
        print("SUMMARY")
        print("=" * 80)
        all_ok = all(results.values()) and census_ok and values_ok
        print(f"  County x year tables: {n_pass}/{n_total} PASS")
        print(f"  Census data:          {'PASS' if census_ok else 'FAIL'}")
        print(f"  Known values:         {'PASS' if values_ok else 'FAIL'}")
        print(f"  Overall:              {'ALL PASS' if all_ok else 'SOME FAILURES'}")

        print()
        print("RECOMMENDATION:")
        print("  For the production VES extraction script:")
        print("  1. Use parse_county_year_table() (text-based) as the primary method.")
        print("     It handles 21/24 tables and is fast.")
        print("  2. Fall back to parse_county_year_table_positional() for pages where")
        print("     text-based parsing finds fewer than 53 counties. This uses character")
        print("     x-positions to handle blank cells correctly.")
        print("  3. NR values: Convert to NaN with convert_nr_to_nan() for analysis.")
        print("  4. Census page (14): Use parse_census_data() (different format).")
        print("  5. Summary pages (4-5, 9-10): Need a custom multi-metric parser.")
        print("  6. Some pages use 'County Average' instead of 'Total' - both are handled.")

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
