"""
Ingest ND Vital Event Summary (VES) PDFs into structured parquet files.

Extracts county-by-year time-series data from VES PDFs published by the
ND Dept. of Health & Human Services. Each VES covers a rolling window of
15-16 years. This script processes all available vintages (2016-2024) and
merges overlapping years with a latest-vintage-wins deduplication strategy.

VES table types (all county × year format):
  - births, birth_rates, fertility_rates
  - pregnancies, pregnancy_rates
  - teenage_births, teenage_birth_rates, teenage_birth_ratios
  - teenage_pregnancies, teenage_pregnancy_rates, teenage_pregnancy_ratios
  - out_of_wedlock_births, out_of_wedlock_birth_ratios
  - out_of_wedlock_pregnancies, out_of_wedlock_pregnancy_ratios
  - low_weight_births, low_weight_birth_ratios
  - infant_deaths, infant_death_ratios
  - neonatal_deaths, neonatal_death_ratios
  - fetal_deaths, fetal_death_ratios
  - deaths, death_rates
  - childhood_adolescent_deaths, childhood_adolescent_death_rates
  - marriages, marriage_rates
  - divorces, divorce_rates
  - census_data (2020 population, different format)

Data suppression: Cells with fewer than 5 events are marked "NR" (Not
Reportable) in the PDFs. These are converted to NaN in the output.

Usage:
    python scripts/data/ingest_ves_data.py
    python scripts/data/ingest_ves_data.py --verbose
    python scripts/data/ingest_ves_data.py --output-dir data/processed/ves
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd
import pdfplumber

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

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

# Summary row labels that may appear at the bottom of county tables
SUMMARY_ROW_NAMES: list[str] = ["Total", "County Average"]

# Expected ordering of county-year tables within a VES PDF.
# Used to resolve ambiguous or duplicate page titles.
EXPECTED_TABLE_ORDER: list[str] = [
    "births",
    "birth_rates",
    "fertility_rates",
    "pregnancies",
    "pregnancy_rates",
    "teenage_births",
    "teenage_birth_rates",
    "teenage_birth_ratios",
    "teenage_pregnancies",
    "teenage_pregnancy_rates",
    "teenage_pregnancy_ratios",
    "out_of_wedlock_births",
    "out_of_wedlock_birth_ratios",
    "out_of_wedlock_pregnancies",
    "out_of_wedlock_pregnancy_ratios",
    "low_weight_births",
    "low_weight_birth_ratios",
    "infant_deaths",
    "infant_death_ratios",
    "neonatal_deaths",
    "neonatal_death_ratios",
    "fetal_deaths",
    "fetal_death_ratios",
    "deaths",
    "death_rates",
    "childhood_adolescent_deaths",
    "childhood_adolescent_death_rates",
    "marriages",
    "marriage_rates",
    "divorces",
    "divorce_rates",
]

# Title-to-label regex patterns. Each pattern is tried against the first line
# of page text (case-insensitive). Order matters for disambiguation: more
# specific patterns must come first.
TITLE_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # Census data (special format)
    (re.compile(r"resident\s+census\s+data", re.IGNORECASE), "census_data"),
    # Teenage pregnancies (before generic pregnancies)
    (re.compile(r"teenage\s+pregnancy\s+ratio", re.IGNORECASE), "teenage_pregnancy_ratios"),
    (re.compile(r"teenage\s+pregnancy\s+rate", re.IGNORECASE), "teenage_pregnancy_rates"),
    (re.compile(r"teenage\s+pregnanc", re.IGNORECASE), "teenage_pregnancies"),
    # Teenage births (before generic births)
    (re.compile(r"teenage\s+birth\s+ratio", re.IGNORECASE), "teenage_birth_ratios"),
    (re.compile(r"teenage\s+birth\s+rate", re.IGNORECASE), "teenage_birth_rates"),
    (re.compile(r"teenage\s+birth", re.IGNORECASE), "teenage_births"),
    # Out-of-wedlock pregnancies (before generic)
    (re.compile(r"wedlock\s+pregnancy\s+ratio", re.IGNORECASE), "out_of_wedlock_pregnancy_ratios"),
    (re.compile(r"wedlock\s+pregnanc", re.IGNORECASE), "out_of_wedlock_pregnancies"),
    # Out-of-wedlock births
    (re.compile(r"wedlock\s+birth\s+ratio", re.IGNORECASE), "out_of_wedlock_birth_ratios"),
    (re.compile(r"wedlock\s+birth", re.IGNORECASE), "out_of_wedlock_births"),
    # Low weight births
    (re.compile(r"low\s+weight\s+birth\s+ratio", re.IGNORECASE), "low_weight_birth_ratios"),
    (re.compile(r"low\s+weight\s+birth", re.IGNORECASE), "low_weight_births"),
    # Infant deaths
    (re.compile(r"infant\s+death\s+ratio", re.IGNORECASE), "infant_death_ratios"),
    (re.compile(r"infant\s+death", re.IGNORECASE), "infant_deaths"),
    # Neonatal deaths
    (re.compile(r"neonatal\s+death\s+ratio", re.IGNORECASE), "neonatal_death_ratios"),
    (re.compile(r"neonatal\s+death", re.IGNORECASE), "neonatal_deaths"),
    # Fetal deaths
    (re.compile(r"fetal\s+death\s+ratio", re.IGNORECASE), "fetal_death_ratios"),
    (re.compile(r"fetal\s+death", re.IGNORECASE), "fetal_deaths"),
    # Childhood deaths
    (re.compile(r"childhood.*death\s+rate", re.IGNORECASE), "childhood_adolescent_death_rates"),
    (re.compile(r"childhood.*death", re.IGNORECASE), "childhood_adolescent_deaths"),
    # Generic fertility rates (after teenage)
    (re.compile(r"resident\s+fertility\s+rate", re.IGNORECASE), "fertility_rates"),
    # Generic pregnancy rates/counts (after teenage/wedlock)
    (re.compile(r"resident\s+pregnancy\s+rate", re.IGNORECASE), "pregnancy_rates"),
    (re.compile(r"resident\s+pregnanc", re.IGNORECASE), "pregnancies"),
    # Generic birth rates (after teenage/wedlock/low-weight)
    (re.compile(r"resident\s+birth\s+rate", re.IGNORECASE), "birth_rates"),
    # Generic births (after all qualified births)
    (re.compile(r"resident\s+births$", re.IGNORECASE), "births"),
    # Generic death rates (after childhood)
    (re.compile(r"resident\s+death\s+rate", re.IGNORECASE), "death_rates"),
    # Generic deaths (after infant/neonatal/fetal/childhood)
    (re.compile(r"resident\s+deaths$", re.IGNORECASE), "deaths"),
    # Marriage/divorce
    (re.compile(r"marriage\s+rate", re.IGNORECASE), "marriage_rates"),
    (re.compile(r"marriage\s+data", re.IGNORECASE), "marriages"),
    (re.compile(r"divorce\s+rate", re.IGNORECASE), "divorce_rates"),
    (re.compile(r"divorce.*data", re.IGNORECASE), "divorces"),
]


# ── Core parsing functions ────────────────────────────────────────────────────


def _get_search_names(include_total: bool = True) -> list[str]:
    """Return county names plus optional summary row names, sorted longest first."""
    names = list(ND_COUNTIES)
    if include_total:
        names.extend(SUMMARY_ROW_NAMES)
    names.sort(key=len, reverse=True)
    return names


def detect_year_columns(page_text: str) -> list[str]:
    """Extract year column headers from a page's text.

    Looks for a line containing "County" and multiple 4-digit years, or
    falls back to finding any line with 10+ consecutive 4-digit years.
    """
    for line in page_text.split("\n"):
        line = line.strip()
        # Standard header: "County of Residence YYYY YYYY ..."
        if re.match(r"county\s+of\s+residen", line, re.IGNORECASE):
            years = re.findall(r"\b((?:19|20)\d{2})\b", line)
            if len(years) >= 10:
                return years
        # Fallback: any line with many consecutive years
        years = re.findall(r"\b((?:19|20)\d{2})\b", line)
        if len(years) >= 10:
            return years
    return []


def classify_page(title: str) -> str | None:
    """Match a page title to a table label using regex patterns."""
    for pattern, label in TITLE_PATTERNS:
        if pattern.search(title):
            return label
    return None


def parse_county_year_table(
    page: pdfplumber.page.Page,
    years: list[str],
    include_total: bool = True,
) -> pd.DataFrame:
    """Parse a county-by-year table using text extraction.

    Fast approach for pages where all rows have the expected number of values.
    For pages with blank cells, falls back to positional parsing automatically.
    """
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
    years: list[str],
    include_total: bool = True,
    col_tolerance: float = 15.0,
) -> pd.DataFrame:
    """Parse a county-by-year table using character x-positions.

    Handles pages where some cells are genuinely blank (not NR).
    """
    chars = page.chars
    if not chars:
        return pd.DataFrame()

    rows_by_y: dict[float, list[dict]] = defaultdict(list)
    for c in chars:
        y = round(c["top"], 0)
        rows_by_y[y].append(c)

    def _chars_to_words(char_list: list[dict]) -> list[tuple[float, str]]:
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

    # Find header row to get column x-positions
    col_x_positions: dict[str, float] = {}
    for y in sorted(rows_by_y.keys()):
        words = _chars_to_words(rows_by_y[y])
        word_texts = [w[1] for w in words]
        # Check if this row contains at least half the expected years
        year_matches = [y for y in years if y in word_texts]
        if len(year_matches) >= len(years) // 2:
            for x, text in words:
                if text in years:
                    col_x_positions[text] = x
            if len(col_x_positions) >= len(years) - 1:
                break

    if len(col_x_positions) < len(years) // 2:
        return parse_county_year_table(page, years, include_total)

    search_names = _get_search_names(include_total)
    records: list[dict[str, str]] = []

    for y in sorted(rows_by_y.keys()):
        words = _chars_to_words(rows_by_y[y])
        if not words:
            continue

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

        county_words = matched_county.split()
        data_words = words[len(county_words):]

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

    df = pd.DataFrame(records)
    for yr in years:
        if yr not in df.columns:
            df[yr] = pd.NA

    cols = ["county"] + [yr for yr in years if yr in df.columns]
    return df[cols]


def smart_parse_county_year_table(
    page: pdfplumber.page.Page,
    years: list[str],
    include_total: bool = True,
) -> pd.DataFrame:
    """Try text-based parsing first; fall back to positional if rows are missing."""
    df = parse_county_year_table(page, years, include_total)

    n_counties = len(df[df["county"] != "Total"]) if not df.empty else 0
    if n_counties < 53:
        df_pos = parse_county_year_table_positional(page, years, include_total)
        n_pos = len(df_pos[df_pos["county"] != "Total"]) if not df_pos.empty else 0
        if n_pos > n_counties:
            return df_pos

    return df


def convert_nr_to_nan(df: pd.DataFrame, year_cols: list[str] | None = None) -> pd.DataFrame:
    """Convert NR values to NaN and cast year columns to float."""
    if year_cols is None:
        year_cols = [c for c in df.columns if c != "county"]

    df = df.copy()
    for col in year_cols:
        if col in df.columns:
            df[col] = df[col].replace("NR", pd.NA)
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def parse_census_data(page: pdfplumber.page.Page) -> pd.DataFrame:
    """Parse the census population data page (county, pop, females 15-44, etc.)."""
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
                            "Total" if county in SUMMARY_ROW_NAMES else county
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


# ── VES file processing ──────────────────────────────────────────────────────


def _resolve_duplicate_labels(
    page_labels: list[tuple[int, str, str | None]],
) -> list[tuple[int, str, str]]:
    """Resolve duplicate or None labels using expected table ordering.

    Parameters
    ----------
    page_labels : list of (page_index, title, label_or_None)

    Returns
    -------
    list of (page_index, title, resolved_label)
    """
    # Track which labels have been assigned
    assigned: set[str] = set()
    resolved: list[tuple[int, str, str]] = []

    # First pass: collect non-duplicate, non-None labels
    label_counts: dict[str, int] = defaultdict(int)
    for _, _, label in page_labels:
        if label is not None:
            label_counts[label] += 1

    # Build the expected order index for fallback resolution
    order_idx: dict[str, int] = {
        label: i for i, label in enumerate(EXPECTED_TABLE_ORDER)
    }

    # Second pass: assign labels
    next_expected_idx = 0
    for page_idx, title, label in page_labels:
        if label is not None and label_counts[label] == 1:
            # Unique match — use it directly
            resolved.append((page_idx, title, label))
            assigned.add(label)
            if label in order_idx:
                next_expected_idx = max(next_expected_idx, order_idx[label] + 1)
        elif label is not None and label in assigned:
            # Duplicate — use next expected unassigned label
            while (
                next_expected_idx < len(EXPECTED_TABLE_ORDER)
                and EXPECTED_TABLE_ORDER[next_expected_idx] in assigned
            ):
                next_expected_idx += 1
            if next_expected_idx < len(EXPECTED_TABLE_ORDER):
                fallback_label = EXPECTED_TABLE_ORDER[next_expected_idx]
                logger.warning(
                    "Page %d: duplicate title '%s' -> reassigned to '%s' "
                    "(expected order)",
                    page_idx, title, fallback_label,
                )
                resolved.append((page_idx, title, fallback_label))
                assigned.add(fallback_label)
                next_expected_idx += 1
            else:
                logger.warning(
                    "Page %d: duplicate title '%s' -> skipped (no unassigned labels)",
                    page_idx, title,
                )
        elif label is not None:
            # First occurrence of a label that appears multiple times
            resolved.append((page_idx, title, label))
            assigned.add(label)
            if label in order_idx:
                next_expected_idx = max(next_expected_idx, order_idx[label] + 1)
        else:
            # No label match — skip non-county pages
            logger.debug("Page %d: title '%s' -> no match, skipped", page_idx, title)

    return resolved


def extract_ves_tables(
    pdf_path: Path,
    verbose: bool = False,
) -> dict[str, pd.DataFrame]:
    """Extract all county-year tables from a single VES PDF.

    Parameters
    ----------
    pdf_path : Path
        Path to the VES PDF file.
    verbose : bool
        If True, log details about each page.

    Returns
    -------
    dict[str, pd.DataFrame]
        Mapping from table label to DataFrame with 'county' + year columns.
        Year columns contain numeric values with NaN for suppressed cells.
    """
    tables: dict[str, pd.DataFrame] = {}

    with pdfplumber.open(pdf_path) as pdf:
        logger.info("Processing %s (%d pages)", pdf_path.name, len(pdf.pages))

        # Step 1: Classify all pages and detect county-year tables
        page_candidates: list[tuple[int, str, str | None]] = []

        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if not text:
                continue

            first_line = text.strip().split("\n")[0].strip()

            # Check if this page has county-year data
            years = detect_year_columns(text)
            if not years:
                # Not a county-year table
                # But check for census data (different format, no year columns)
                label = classify_page(first_line)
                if label == "census_data":
                    page_candidates.append((i, first_line, "census_data"))
                continue

            label = classify_page(first_line)
            page_candidates.append((i, first_line, label))

        # Step 2: Resolve duplicates and ambiguous labels
        resolved_pages = _resolve_duplicate_labels(page_candidates)

        # Step 3: Extract data from each classified page
        for page_idx, _title, label in resolved_pages:
            page = pdf.pages[page_idx]

            if label == "census_data":
                df = parse_census_data(page)
                if not df.empty:
                    n_counties = len(df[df["county"] != "Total"])
                    if verbose:
                        logger.info(
                            "  Page %2d %-40s -> %d counties",
                            page_idx, label, n_counties,
                        )
                    tables[label] = df
                continue

            text = page.extract_text()
            years = detect_year_columns(text)
            if not years:
                logger.warning(
                    "Page %d (%s): no year columns detected, skipping",
                    page_idx, label,
                )
                continue

            df = smart_parse_county_year_table(page, years)
            if df.empty:
                logger.warning(
                    "Page %d (%s): no data extracted, skipping",
                    page_idx, label,
                )
                continue

            n_counties = len(df[df["county"] != "Total"])
            n_years = len([c for c in df.columns if c != "county"])

            # Convert NR to NaN
            df = convert_nr_to_nan(df)

            if verbose:
                nr_count = df[[c for c in df.columns if c != "county"]].isna().sum().sum()
                logger.info(
                    "  Page %2d %-40s -> %d counties, %d years, %d NaN",
                    page_idx, label, n_counties, n_years, int(nr_count),
                )

            if n_counties < 50:
                logger.warning(
                    "Page %d (%s): only %d counties extracted (expected 53)",
                    page_idx, label, n_counties,
                )

            tables[label] = df

    return tables


# ── Multi-vintage merging ─────────────────────────────────────────────────────


def merge_vintages(
    all_tables: dict[int, dict[str, pd.DataFrame]],
) -> dict[str, pd.DataFrame]:
    """Merge tables across VES vintages, latest vintage wins for overlapping years.

    Parameters
    ----------
    all_tables : dict[int, dict[str, pd.DataFrame]]
        Mapping from vintage year to {label: DataFrame}.

    Returns
    -------
    dict[str, pd.DataFrame]
        Merged tables with the full year range from earliest to latest vintage.
    """
    # Collect all labels across vintages
    all_labels: set[str] = set()
    for tables in all_tables.values():
        all_labels.update(tables.keys())

    merged: dict[str, pd.DataFrame] = {}

    for label in sorted(all_labels):
        # Process vintages in chronological order (earliest first)
        # so that later vintages overwrite overlapping years
        dfs_by_vintage: list[tuple[int, pd.DataFrame]] = []
        for vintage in sorted(all_tables.keys()):
            if label in all_tables[vintage]:
                dfs_by_vintage.append((vintage, all_tables[vintage][label]))

        if not dfs_by_vintage:
            continue

        if label == "census_data":
            # Census data has no year columns — take latest vintage only
            _, latest_df = dfs_by_vintage[-1]
            merged[label] = latest_df
            continue

        # Combine: start with a full county roster, then layer vintages
        # (earliest first so later vintages overwrite overlapping years)
        all_counties_in_label: set[str] = set()
        for _, df in dfs_by_vintage:
            all_counties_in_label.update(df["county"].unique())
        combined = pd.DataFrame({"county": sorted(all_counties_in_label)})

        for _vintage, df in dfs_by_vintage:
            new_year_cols = [c for c in df.columns if c != "county"]
            county_map = df.set_index("county")
            for yr in new_year_cols:
                combined[yr] = combined["county"].map(county_map[yr])

        if combined is not None:
            # Reorder columns: county + years in chronological order
            year_cols = sorted(
                [c for c in combined.columns if c != "county"],
                key=lambda x: int(x),
            )
            combined = combined[["county"] + year_cols]
            merged[label] = combined

    return merged


# ── Output ────────────────────────────────────────────────────────────────────


def save_tables(
    tables: dict[str, pd.DataFrame],
    output_dir: Path,
    fmt: str = "parquet",
) -> list[Path]:
    """Save extracted tables to files.

    Parameters
    ----------
    tables : dict[str, pd.DataFrame]
        Mapping from table label to DataFrame.
    output_dir : Path
        Directory to write output files.
    fmt : str
        Output format: "parquet" or "csv".

    Returns
    -------
    list[Path]
        Paths to the written files.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    for label, df in sorted(tables.items()):
        if fmt == "parquet":
            path = output_dir / f"ves_{label}.parquet"
            df.to_parquet(path, index=False)
        else:
            path = output_dir / f"ves_{label}.csv"
            df.to_csv(path, index=False)

        written.append(path)
        logger.info("  Wrote %s (%d rows, %d cols)", path.name, len(df), len(df.columns))

    return written


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract ND Vital Event Summary data from PDFs",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/raw"),
        help="Directory containing VES PDF files (default: data/raw)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/ves"),
        help="Directory for output files (default: data/processed/ves)",
    )
    parser.add_argument(
        "--format",
        choices=["parquet", "csv"],
        default="parquet",
        help="Output format (default: parquet)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Show detailed output for each page",
    )
    parser.add_argument(
        "--single",
        type=Path,
        help="Process a single VES PDF instead of all vintages",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )
    # Suppress noisy pdfplumber/pdfminer debug logging
    logging.getLogger("pdfminer").setLevel(logging.WARNING)
    logging.getLogger("pdfplumber").setLevel(logging.WARNING)

    input_dir = args.input_dir
    if not input_dir.is_absolute():
        project_root = Path(__file__).resolve().parent.parent.parent
        input_dir = project_root / input_dir

    output_dir = args.output_dir
    if not output_dir.is_absolute():
        project_root = Path(__file__).resolve().parent.parent.parent
        output_dir = project_root / output_dir

    if args.single:
        # Process a single PDF
        pdf_path = args.single
        if not pdf_path.is_absolute():
            project_root = Path(__file__).resolve().parent.parent.parent
            pdf_path = project_root / pdf_path
        if not pdf_path.exists():
            logger.error("PDF not found: %s", pdf_path)
            return 1

        tables = extract_ves_tables(pdf_path, verbose=args.verbose)
        logger.info("Extracted %d tables from %s", len(tables), pdf_path.name)

        written = save_tables(tables, output_dir, args.format)
        logger.info("Wrote %d files to %s", len(written), output_dir)
        return 0

    # Process all VES vintages
    ves_files = sorted(input_dir.glob("*VES.pdf"))
    if not ves_files:
        logger.error("No VES PDF files found in %s", input_dir)
        return 1

    logger.info("Found %d VES files: %s", len(ves_files), [f.name for f in ves_files])

    all_tables: dict[int, dict[str, pd.DataFrame]] = {}
    for pdf_path in ves_files:
        # Extract vintage year from filename (e.g., "2024VES.pdf" -> 2024)
        match = re.match(r"(\d{4})VES\.pdf", pdf_path.name)
        if not match:
            logger.warning("Cannot parse vintage year from %s, skipping", pdf_path.name)
            continue

        vintage = int(match.group(1))
        tables = extract_ves_tables(pdf_path, verbose=args.verbose)

        if not tables:
            logger.warning("%s: no tables extracted (may be incomplete)", pdf_path.name)
            continue

        # Filter out census_data for merge count
        county_tables = {k: v for k, v in tables.items() if k != "census_data"}
        logger.info(
            "%s: extracted %d county-year tables + %s",
            pdf_path.name,
            len(county_tables),
            "census_data" if "census_data" in tables else "no census data",
        )
        all_tables[vintage] = tables

    if not all_tables:
        logger.error("No tables extracted from any VES file")
        return 1

    # Merge across vintages
    logger.info("Merging %d vintages...", len(all_tables))
    merged = merge_vintages(all_tables)
    logger.info("Merged into %d table types", len(merged))

    # Report year ranges
    for label, df in sorted(merged.items()):
        year_cols = [c for c in df.columns if c != "county" and c.isdigit()]
        if year_cols:
            n_counties = len(df[df["county"] != "Total"])
            years_sorted = sorted(year_cols, key=int)
            logger.info(
                "  %-40s: %d counties, years %s-%s (%d years)",
                label, n_counties, years_sorted[0], years_sorted[-1], len(years_sorted),
            )

    # Save
    written = save_tables(merged, output_dir, args.format)
    logger.info("Wrote %d files to %s", len(written), output_dir)

    return 0


if __name__ == "__main__":
    sys.exit(main())
