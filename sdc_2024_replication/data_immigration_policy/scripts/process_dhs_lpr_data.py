# mypy: ignore-errors
"""
Process DHS Yearbook of Immigration Statistics LPR data.

This script processes:
1. Table 3: LPR by region and country of birth (FY 2014-2023)
2. Table 4: LPR by state (FY 2014-2023)
3. LPRSuppTable 1: LPR by state and country of birth (FY 2023)

Outputs a cleaned parquet file with focus on North Dakota analysis.
"""

from pathlib import Path
import re
import zipfile

import numpy as np
import pandas as pd
from cohort_projections.utils import ConfigLoader, setup_logger
from cohort_projections.utils.reproducibility import log_execution

logger = setup_logger(__name__)

STATE_EXCLUDE_PATTERNS = [
    "Note",
    "Source",
    "Return",
    "Table",
    "Armed Services",
    "Territories",
    "Virgin Islands",
]

HISTORICAL_MIN_YEAR = 2000
YEARBOOK_PDF_DIR = "yearbook_pdfs"
YEARBOOK_TABLE4_START_YEAR = 2003
YEARBOOK_TABLE4_END_YEAR = 2012


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Convert column names to lowercase with underscores."""
    df.columns = [
        str(c).lower().replace(" ", "_").replace("-", "_").replace("/", "_") for c in df.columns
    ]
    return df


def process_table3_country_of_birth(xl: pd.ExcelFile) -> pd.DataFrame:
    """Process Table 3: LPR by region and country of birth over time."""
    df = pd.read_excel(xl, sheet_name="Table 3", header=5)

    # Clean column names
    df.columns = ["region_country_of_birth"] + list(range(2014, 2024))

    # Remove header rows (REGION, COUNTRY markers)
    df = df[~df["region_country_of_birth"].isin(["REGION", "COUNTRY", np.nan])]
    df = df.dropna(subset=["region_country_of_birth"])

    # Melt to long format
    df_long = df.melt(
        id_vars=["region_country_of_birth"], var_name="fiscal_year", value_name="lpr_count"
    )

    # Clean values
    df_long["lpr_count"] = pd.to_numeric(df_long["lpr_count"], errors="coerce")

    # Add type indicator
    regions = [
        "Total",
        "Africa",
        "Asia",
        "Europe",
        "North America",
        "Oceania",
        "South America",
        "Unknown",
    ]
    df_long["is_region"] = df_long["region_country_of_birth"].isin(regions)

    return df_long


def _filter_state_rows(df: pd.DataFrame, state_column: str) -> pd.DataFrame:
    """Filter non-state rows and notes from a state time series table."""
    df = df[~df[state_column].isin([np.nan, "Total", "Unknown", "Other"])]
    df = df.dropna(subset=[state_column])

    for pattern in STATE_EXCLUDE_PATTERNS:
        df = df[~df[state_column].astype(str).str.contains(pattern, case=False, na=False)]

    return df


def process_table4_state_time_series(xl: pd.ExcelFile) -> pd.DataFrame:
    """Process Table 4: LPR by state over time (FY 2014-2023)."""
    df = pd.read_excel(xl, sheet_name="Table 4", header=5)

    # Clean column names
    df.columns = ["state_or_territory"] + list(range(2014, 2024))

    df = _filter_state_rows(df, "state_or_territory")

    # Melt to long format
    df_long = df.melt(
        id_vars=["state_or_territory"], var_name="fiscal_year", value_name="lpr_count"
    )

    df_long["lpr_count"] = pd.to_numeric(df_long["lpr_count"], errors="coerce")

    return df_long


def _extract_year(value: object) -> int | None:
    """Return a year integer if value looks like a year column."""
    try:
        year = int(float(value))
    except (TypeError, ValueError):
        return None
    if 1900 <= year <= 2100:
        return year
    return None


def _parse_state_table_from_zip(zip_path: Path) -> pd.DataFrame | None:
    """Extract state time series from a yearbook ZIP file."""
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            if not name.lower().endswith((".xls", ".xlsx")):
                continue
            with zf.open(name) as handle:
                try:
                    df = pd.read_excel(handle, header=None)
                except Exception as exc:
                    logger.warning("Skipping %s in %s (%s).", name, zip_path.name, exc)
                    continue

            col0 = df.iloc[:, 0].astype(str).str.strip()
            match = col0.str.lower().eq("state of residence")
            if not match.any():
                match = col0.str.lower().eq("state of intended residence")
            if not match.any():
                match = col0.str.lower().eq("state or territory of residence")
            if not match.any():
                match = (
                    col0.str.contains("state", case=False, na=False)
                    & col0.str.contains("residence", case=False, na=False)
                    & ~col0.str.contains("persons obtaining", case=False, na=False)
                )
            if not match.any():
                continue

            header_idx = match[match].index[0]
            header = df.iloc[header_idx].tolist()
            data = df.iloc[header_idx + 1 :].copy()
            data.columns = header

            state_col = header[0]
            data = data.rename(columns={state_col: "state_or_territory"})

            year_cols: list[object] = []
            year_map: dict[object, int] = {}
            for col in data.columns[1:]:
                year = _extract_year(col)
                if year is not None:
                    year_cols.append(col)
                    year_map[col] = year

            if not year_cols:
                logger.warning("No year columns found in %s (sheet %s).", zip_path.name, name)
                continue

            data = data[["state_or_territory"] + year_cols]
            data = _filter_state_rows(data, "state_or_territory")

            df_long = data.melt(
                id_vars=["state_or_territory"], var_name="fiscal_year", value_name="lpr_count"
            )
            df_long["fiscal_year"] = df_long["fiscal_year"].map(year_map)
            df_long["lpr_count"] = pd.to_numeric(df_long["lpr_count"], errors="coerce")
            return df_long

    return None


def load_historical_state_time_series(source_path: Path, zip_glob: str) -> pd.DataFrame:
    """Load state time series from historical yearbook ZIP files."""
    zip_paths = sorted(source_path.glob(zip_glob))
    zip_paths = [p for p in zip_paths if "Supp" not in p.name]

    frames: list[pd.DataFrame] = []
    for zip_path in zip_paths:
        match = re.search(r"(20\\d{2})", zip_path.stem)
        source_year = int(match.group(1)) if match else None
        df = _parse_state_table_from_zip(zip_path)
        if df is None:
            logger.warning("No state table found in %s.", zip_path.name)
            continue
        df["source_yearbook"] = source_year
        frames.append(df)

    if not frames:
        return pd.DataFrame(columns=["state_or_territory", "fiscal_year", "lpr_count"])

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.dropna(subset=["state_or_territory", "fiscal_year"])
    combined = combined[combined["fiscal_year"] >= HISTORICAL_MIN_YEAR]
    combined = combined.sort_values("source_yearbook")
    combined = combined.drop_duplicates(subset=["state_or_territory", "fiscal_year"], keep="last")

    return combined.drop(columns=["source_yearbook"])


def _parse_yearbook_table4_pdf(pdf_path: Path) -> pd.DataFrame:
    """Parse Table 4 (state of residence) from a yearbook PDF."""
    try:
        import pdfplumber
    except ImportError:
        logger.warning("pdfplumber not installed; skipping PDF extraction for %s", pdf_path.name)
        return pd.DataFrame(columns=["state_or_territory", "fiscal_year", "lpr_count"])

    table_pattern = re.compile(r"^(?P<label>.+?)\\.{2,}\\s+(?P<values>.+)$")
    years = list(range(YEARBOOK_TABLE4_START_YEAR, YEARBOOK_TABLE4_END_YEAR + 1))
    rows: list[dict] = []

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            if "Table 4." not in text or "State or territory of residence" not in text:
                continue

            for line in text.split("\\n"):
                match = table_pattern.match(line)
                if not match:
                    continue
                label = match.group("label").strip()
                tokens = match.group("values").strip().split()
                if len(tokens) != len(years):
                    continue
                for year, token in zip(years, tokens):
                    cleaned = token.replace(",", "")
                    if cleaned in {"D", "-", ""}:
                        value = np.nan
                    else:
                        value = pd.to_numeric(cleaned, errors="coerce")
                    rows.append(
                        {
                            "state_or_territory": label,
                            "fiscal_year": year,
                            "lpr_count": value,
                        }
                    )

            break

    if not rows:
        return pd.DataFrame(columns=["state_or_territory", "fiscal_year", "lpr_count"])

    df = pd.DataFrame(rows)
    df = _filter_state_rows(df, "state_or_territory")
    df["lpr_count"] = pd.to_numeric(df["lpr_count"], errors="coerce")
    return df


def load_yearbook_pdf_state_time(source_path: Path, year: int) -> pd.DataFrame:
    """Load state time series from a yearbook PDF (fallback when ZIP tables are missing)."""
    pdf_dir = source_path / YEARBOOK_PDF_DIR
    if not pdf_dir.exists():
        return pd.DataFrame(columns=["state_or_territory", "fiscal_year", "lpr_count"])

    candidates = [
        pdf_dir / f"Yearbook_Immigration_Statistics_{year}.pdf",
        pdf_dir / f"Yearbook Immigration Statistics {year}.pdf",
    ]
    pdf_path = next((p for p in candidates if p.exists()), None)
    if pdf_path is None:
        return pd.DataFrame(columns=["state_or_territory", "fiscal_year", "lpr_count"])

    df = _parse_yearbook_table4_pdf(pdf_path)
    if df.empty:
        return df
    return df[df["fiscal_year"] == year]


def process_supptable1_state_country(xl: pd.ExcelFile) -> pd.DataFrame:
    """Process LPRSuppTable 1: LPR by state and country of birth (FY 2023)."""
    df = pd.read_excel(xl, sheet_name="LPRSuppTable 1", header=5)

    # First column is country/region
    df = df.rename(columns={df.columns[0]: "region_country_of_birth"})

    # Remove marker rows
    df = df[~df["region_country_of_birth"].isin(["REGION", "COUNTRY", np.nan])]
    df = df.dropna(subset=["region_country_of_birth"])

    # Exclude notes/source rows
    exclude_patterns = ["Note", "Source", "Return", "Table", "D "]
    for pattern in exclude_patterns:
        df = df[
            ~df["region_country_of_birth"].astype(str).str.contains(pattern, case=False, na=False)
        ]

    # Get state columns (exclude Total and Unknown)
    state_cols = [
        c
        for c in df.columns[1:]
        if c
        not in [
            "Total",
            "Unknown",
            "U.S. Armed Services Posts",
            "U.S. Territories1",
            "Guam",
            "Puerto Rico",
        ]
    ]

    # Melt to long format with state as column
    df_long = df.melt(
        id_vars=["region_country_of_birth"],
        value_vars=state_cols,
        var_name="state",
        value_name="lpr_count",
    )

    df_long["fiscal_year"] = 2023
    df_long["lpr_count"] = pd.to_numeric(df_long["lpr_count"], errors="coerce")

    # Add type indicator
    regions = [
        "Total",
        "Africa",
        "Asia",
        "Europe",
        "North America",
        "Oceania",
        "South America",
        "Unknown",
    ]
    df_long["is_region"] = df_long["region_country_of_birth"].isin(regions)

    return df_long


def compute_nd_share_analysis(df_state_country: pd.DataFrame) -> pd.DataFrame:
    """Compute North Dakota's share of LPRs by country of origin."""
    # Get ND data
    nd_data = df_state_country[df_state_country["state"] == "North Dakota"].copy()
    nd_data = nd_data.rename(columns={"lpr_count": "nd_lpr_count"})

    # Get total US by country (sum across all states)
    us_totals = df_state_country.groupby("region_country_of_birth")["lpr_count"].sum().reset_index()
    us_totals = us_totals.rename(columns={"lpr_count": "us_total_lpr_count"})

    # Merge
    nd_share = nd_data.merge(us_totals, on="region_country_of_birth")

    # Calculate share
    nd_share["nd_share_pct"] = (
        nd_share["nd_lpr_count"] / nd_share["us_total_lpr_count"] * 100
    ).round(3)

    # Sort by ND count
    nd_share = nd_share.sort_values("nd_lpr_count", ascending=False)

    return nd_share[
        [
            "region_country_of_birth",
            "is_region",
            "nd_lpr_count",
            "us_total_lpr_count",
            "nd_share_pct",
            "fiscal_year",
        ]
    ]


def main() -> None:
    """Main entry point."""
    project_root = Path(__file__).resolve().parents[3]
    config = ConfigLoader().get_projection_config()
    data_sources = config.get("data_sources", {})
    lpr_cfg = data_sources.get("dhs_lpr", {})

    source_path = project_root / lpr_cfg.get("raw_dir", "data/raw/immigration/dhs_lpr")
    output_path = project_root / "data" / "processed" / "immigration" / "analysis"
    output_path.mkdir(parents=True, exist_ok=True)

    yearbook_file = source_path / lpr_cfg.get(
        "yearbook_tables_file", "yearbook_lpr_2023_all_tables.xlsx"
    )
    zip_glob = lpr_cfg.get("historical_zip_glob", "LPR*.zip")

    logger.info("Processing DHS Yearbook LPR data.")
    xl = pd.ExcelFile(yearbook_file)

    logger.info("Processing Table 3: LPR by region/country of birth (FY 2014-2023).")
    df_country_time = process_table3_country_of_birth(xl)

    logger.info("Processing Table 4: LPR by state (FY 2014-2023).")
    df_state_time_recent = process_table4_state_time_series(xl)

    logger.info("Processing historical yearbook ZIPs for state time series.")
    df_state_time_historical = load_historical_state_time_series(source_path, zip_glob)
    df_state_time_pdf = load_yearbook_pdf_state_time(source_path, YEARBOOK_TABLE4_END_YEAR)
    if not df_state_time_pdf.empty:
        logger.info(
            "Adding %s state rows from yearbook PDF for FY%s.",
            df_state_time_pdf["state_or_territory"].nunique(),
            YEARBOOK_TABLE4_END_YEAR,
        )
        df_state_time_historical = pd.concat(
            [df_state_time_historical, df_state_time_pdf], ignore_index=True
        )

    df_state_time = pd.concat([df_state_time_recent, df_state_time_historical], ignore_index=True)
    df_state_time = df_state_time.drop_duplicates(
        subset=["state_or_territory", "fiscal_year"]
    ).sort_values(["state_or_territory", "fiscal_year"])

    logger.info("Processing LPRSuppTable 1: LPR by state and country (FY 2023).")
    df_state_country = process_supptable1_state_country(xl)

    df_country_time.to_parquet(output_path / "dhs_lpr_by_country_time.parquet", index=False)
    df_state_time.to_parquet(output_path / "dhs_lpr_by_state_time.parquet", index=False)
    df_state_country.to_parquet(output_path / "dhs_lpr_by_state_country.parquet", index=False)

    nd_share = compute_nd_share_analysis(df_state_country)
    nd_share.to_parquet(output_path / "dhs_lpr_nd_share_by_country.parquet", index=False)

    nd_time = df_state_time[df_state_time["state_or_territory"] == "North Dakota"]
    if not nd_time.empty:
        logger.info("North Dakota LPR Admissions by Fiscal Year:\\n%s", nd_time.to_string(index=False))

    nd_top = nd_share[~nd_share["is_region"]].head(20)
    if not nd_top.empty:
        logger.info(
            "Top 20 Countries of Origin for ND LPRs (FY 2023):\\n%s",
            nd_top[
                ["region_country_of_birth", "nd_lpr_count", "us_total_lpr_count", "nd_share_pct"]
            ].to_string(index=False),
        )

    nd_regions = nd_share[nd_share["is_region"]]
    nd_regions = nd_regions[nd_regions["region_country_of_birth"] != "Total"]
    if not nd_regions.empty:
        logger.info(
            "ND LPRs by Region of Birth (FY 2023):\\n%s",
            nd_regions[["region_country_of_birth", "nd_lpr_count", "nd_share_pct"]].to_string(
                index=False
            ),
        )

    logger.info("Files saved to %s", output_path)


if __name__ == "__main__":
    with log_execution(__file__, parameters={"series": "dhs_lpr"}):
        main()
