"""
Census/PEP population data loaders for residual migration computation.

Loads age-sex population data from various Census Bureau vintages into a
standard long-format DataFrame with columns:
    [county_fips, age_group, sex, population]

Where:
    - county_fips: 5-digit string like "38017"
    - age_group: "0-4", "5-9", ..., "80-84", "85+"
    - sex: "Male" or "Female"
    - population: integer or float

Supported data sources:
    - Census 2000 County Age and Sex (Excel, long format with AGEGRP codes)
    - PEP 2010-2019 cc-est2019-agesex (Excel, wide format with age columns)
    - PEP 2010-2020 cc-est2020int intercensal (Parquet, long format)
    - PEP 2020-2024 cc-est2024-agesex-all (Parquet, wide format)
    - SDC 2024 base population by county (CSV, already processed)
"""

from pathlib import Path
from typing import Any

import pandas as pd

from cohort_projections.utils import get_logger_from_config

logger = get_logger_from_config(__name__)

# Standard 18 five-year age groups used across all Census/PEP data
AGE_GROUP_LABELS: list[str] = [
    "0-4",
    "5-9",
    "10-14",
    "15-19",
    "20-24",
    "25-29",
    "30-34",
    "35-39",
    "40-44",
    "45-49",
    "50-54",
    "55-59",
    "60-64",
    "65-69",
    "70-74",
    "75-79",
    "80-84",
    "85+",
]

# AGEGRP code to age_group label mapping (codes 1-18)
AGEGRP_TO_LABEL: dict[int, str] = dict(enumerate(AGE_GROUP_LABELS, start=1))

# Wide-format column name to (age_group, sex) mapping for cc-est2019 and
# cc-est2024-agesex-all.  These files share the same column naming convention.
_WIDE_AGE_COLUMNS: dict[str, tuple[str, str]] = {}
_WIDE_AGE_COL_SPECS: list[tuple[str, str, str]] = [
    ("AGE04", "0-4", ""),
    ("AGE59", "5-9", ""),
    ("AGE1014", "10-14", ""),
    ("AGE1519", "15-19", ""),
    ("AGE2024", "20-24", ""),
    ("AGE2529", "25-29", ""),
    ("AGE3034", "30-34", ""),
    ("AGE3539", "35-39", ""),
    ("AGE4044", "40-44", ""),
    ("AGE4549", "45-49", ""),
    ("AGE5054", "50-54", ""),
    ("AGE5559", "55-59", ""),
    ("AGE6064", "60-64", ""),
    ("AGE6569", "65-69", ""),
    ("AGE7074", "70-74", ""),
    ("AGE7579", "75-79", ""),
    ("AGE8084", "80-84", ""),
    ("AGE85PLUS", "85+", ""),
]

for _prefix, _age_group, _ in _WIDE_AGE_COL_SPECS:
    _WIDE_AGE_COLUMNS[f"{_prefix}_MALE"] = (_age_group, "Male")
    _WIDE_AGE_COLUMNS[f"{_prefix}_FEM"] = (_age_group, "Female")


def _make_county_fips(state: Any, county: Any) -> str:
    """Construct 5-digit FIPS from state and county codes."""
    return f"{int(state):02d}{int(county):03d}"


def _pivot_wide_to_long(
    df: pd.DataFrame,
    id_cols: list[str],
) -> pd.DataFrame:
    """Pivot wide-format Census age-sex columns to standard long format.

    Args:
        df: DataFrame with wide age columns (AGE04_MALE, AGE04_FEM, etc.)
        id_cols: Columns to keep as identifiers (e.g., STATE, COUNTY, CTYNAME).

    Returns:
        DataFrame with [county_fips, age_group, sex, population].
    """
    records: list[dict[str, Any]] = []

    for _, row in df.iterrows():
        county_fips = _make_county_fips(row["STATE"], row["COUNTY"])

        for col_name, (age_group, sex) in _WIDE_AGE_COLUMNS.items():
            if col_name in df.columns:
                population = row[col_name]
                records.append(
                    {
                        "county_fips": county_fips,
                        "age_group": age_group,
                        "sex": sex,
                        "population": float(population),
                    }
                )

    result = pd.DataFrame(records)
    return result


def load_census_2000_county_age_sex(
    file_path: str | Path,
    state_fips: str = "38",
    year: int = 2000,
) -> pd.DataFrame:
    """Load age-sex population from Census 2000 County Age and Sex file.

    This file is in long format with AGEGRP codes (0-18) and SEX codes
    (0=Total, 1=Male, 2=Female).  Population columns are named
    ESTIMATESBASE2000, POPESTIMATE2000 through POPESTIMATE2009.

    Args:
        file_path: Path to the Excel file.
        state_fips: 2-digit state FIPS code (default "38" for North Dakota).
        year: Target year.  2000 uses ESTIMATESBASE2000; other years use
              POPESTIMATEyyyy columns.

    Returns:
        DataFrame with columns [county_fips, age_group, sex, population].

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the target year column is not found.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Census 2000 file not found: {file_path}")

    logger.info(f"Loading Census 2000 county age/sex data for year {year}")
    df = pd.read_excel(file_path)

    # Filter to target state
    state_int = int(state_fips)
    df = df[df["STATE"] == state_int].copy()
    logger.info(f"Filtered to state {state_fips}: {len(df)} rows")

    # Filter to age group rows (exclude total) and sex rows (exclude total)
    df = df[(df["AGEGRP"] > 0) & (df["SEX"] > 0)].copy()
    logger.info(f"After AGEGRP > 0 and SEX > 0 filter: {len(df)} rows")

    # Select population column
    pop_col = "ESTIMATESBASE2000" if year == 2000 else f"POPESTIMATE{year}"

    if pop_col not in df.columns:
        raise ValueError(
            f"Population column '{pop_col}' not found. "
            f"Available: {[c for c in df.columns if 'POP' in c or 'ESTIMATE' in c]}"
        )

    # Map AGEGRP codes to labels
    df["age_group"] = df["AGEGRP"].map(AGEGRP_TO_LABEL)

    # Map SEX codes: 1=Male, 2=Female
    sex_map = {1: "Male", 2: "Female"}
    df["sex"] = df["SEX"].map(sex_map)

    # Construct county FIPS
    df["county_fips"] = df.apply(lambda r: _make_county_fips(r["STATE"], r["COUNTY"]), axis=1)

    result = df[["county_fips", "age_group", "sex"]].copy()
    result["population"] = df[pop_col].astype(float)

    logger.info(
        f"Loaded {len(result)} records for {result['county_fips'].nunique()} counties, "
        f"year {year}, total pop {result['population'].sum():,.0f}"
    )
    return result.reset_index(drop=True)


def load_pep_2010_2019_county_age_sex(
    file_path: str | Path,
    state_fips: str = "38",
    year: int = 2010,
) -> pd.DataFrame:
    """Load age-sex population from PEP cc-est2019-agesex file.

    This file is in wide format with age columns like AGE04_MALE, AGE04_FEM.
    YEAR codes: 1=Census 2010 base, 2=2010 estimate, ..., 12=2019 estimate.

    Args:
        file_path: Path to the Excel file.
        state_fips: 2-digit state FIPS code (default "38" for North Dakota).
        year: Target calendar year (2010-2019).  2010 maps to YEAR=1 (census
              base); other years map to YEAR=(year-2010)+2.

    Returns:
        DataFrame with columns [county_fips, age_group, sex, population].

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the target year is out of range.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"PEP 2010-2019 file not found: {file_path}")

    logger.info(f"Loading PEP 2010-2019 county age/sex data for year {year}")
    df = pd.read_excel(file_path)

    # Map calendar year to YEAR code
    if year == 2010:
        year_code = 1  # Census 2010 base
    elif 2010 <= year <= 2019:
        year_code = (year - 2010) + 2
    else:
        raise ValueError(f"Year {year} out of range for cc-est2019 file (2010-2019)")

    logger.info(f"Calendar year {year} -> YEAR code {year_code}")

    # Filter to target YEAR
    df = df[df["YEAR"] == year_code].copy()
    logger.info(f"Filtered to YEAR={year_code}: {len(df)} rows")

    # Pivot wide columns to long format
    result = _pivot_wide_to_long(df, id_cols=["STATE", "COUNTY", "CTYNAME"])

    logger.info(
        f"Loaded {len(result)} records for {result['county_fips'].nunique()} counties, "
        f"year {year}, total pop {result['population'].sum():,.0f}"
    )
    return result.reset_index(drop=True)


def load_pep_2020_intercensal_county_age_sex(
    file_path: str | Path,
    state_fips: str = "38",
    year: int = 2020,
) -> pd.DataFrame:
    """Load age-sex population from PEP cc-est2020int intercensal parquet.

    This file is in long format with AGEGRP column (0=Total, 1-18=age groups)
    and TOT_MALE/TOT_FEMALE columns.  YEAR codes are strings: "1"=Census 2010
    base through "12".  Note: all columns in this file are string type.

    For the 2020 endpoint, YEAR=1 gives census 2010 base; for 2015, YEAR=6.
    This file goes up to YEAR=12 for 2020.

    Mapping: YEAR "1"=4/1/2010, "2"=7/1/2010, ..., "12"=7/1/2020.
    So: year 2010 -> YEAR "1", year 2015 -> YEAR "7", year 2020 -> YEAR "12".

    Args:
        file_path: Path to the parquet file.
        state_fips: 2-digit state FIPS code (default "38" for North Dakota).
        year: Target calendar year.

    Returns:
        DataFrame with columns [county_fips, age_group, sex, population].

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"PEP intercensal file not found: {file_path}")

    logger.info(f"Loading PEP 2010-2020 intercensal data for year {year}")
    df = pd.read_parquet(file_path)

    # STATE column is string type in this file
    state_filter = state_fips.zfill(2)
    df = df[df["STATE"] == state_filter].copy()
    logger.info(f"Filtered to state {state_filter}: {len(df)} rows")

    # Map calendar year to YEAR code (string)
    # YEAR 1 = April 1 2010 census, 2 = July 1 2010, 3 = July 1 2011, ...
    # 12 = July 1 2020 (intercensal estimate incorporating 2020 Census)
    if year == 2010:
        year_code = "1"
    elif 2010 <= year <= 2020:
        year_code = str((year - 2010) + 2)
    else:
        raise ValueError(f"Year {year} out of range for intercensal file (2010-2020)")

    logger.info(f"Calendar year {year} -> YEAR code '{year_code}'")

    df = df[df["YEAR"] == year_code].copy()
    logger.info(f"Filtered to YEAR='{year_code}': {len(df)} rows")

    # Filter to age group rows (exclude AGEGRP="0" which is total)
    df = df[df["AGEGRP"] != "0"].copy()

    # AGEGRP is string; convert to int for mapping
    df["agegrp_int"] = df["AGEGRP"].astype(int)
    df["age_group"] = df["agegrp_int"].map(AGEGRP_TO_LABEL)

    # Construct county FIPS (STATE and COUNTY are strings)
    df["county_fips"] = df["STATE"].str.zfill(2) + df["COUNTY"].str.zfill(3)

    # TOT_MALE and TOT_FEMALE columns (may be string type)
    df["TOT_MALE"] = pd.to_numeric(df["TOT_MALE"], errors="coerce")
    df["TOT_FEMALE"] = pd.to_numeric(df["TOT_FEMALE"], errors="coerce")

    # Melt to long format
    male = df[["county_fips", "age_group"]].copy()
    male["sex"] = "Male"
    male["population"] = df["TOT_MALE"].values

    female = df[["county_fips", "age_group"]].copy()
    female["sex"] = "Female"
    female["population"] = df["TOT_FEMALE"].values

    result = pd.concat([male, female], ignore_index=True)

    logger.info(
        f"Loaded {len(result)} records for {result['county_fips'].nunique()} counties, "
        f"year {year}, total pop {result['population'].sum():,.0f}"
    )
    return result.reset_index(drop=True)


def load_pep_2020_2024_county_age_sex(
    file_path: str | Path,
    state_fips: str = "38",
    year: int = 2024,
) -> pd.DataFrame:
    """Load age-sex population from PEP cc-est2024-agesex-all parquet.

    This file is in wide format with age columns like AGE04_MALE, AGE04_FEM.
    YEAR codes (string): "1"=Census 2020 base, "2"=2020 estimate, ...,
    "6"=2024 estimate.

    Args:
        file_path: Path to the parquet file.
        state_fips: 2-digit state FIPS code (default "38" for North Dakota).
        year: Target calendar year (2020-2024).  2020 maps to YEAR="1"
              (census base); other years map to YEAR=str((year-2020)+2).

    Returns:
        DataFrame with columns [county_fips, age_group, sex, population].

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the target year is out of range.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"PEP 2020-2024 file not found: {file_path}")

    logger.info(f"Loading PEP 2020-2024 county age/sex data for year {year}")
    df = pd.read_parquet(file_path)

    # Filter to target state (STATE may be string or int)
    state_filter = state_fips.zfill(2) if df["STATE"].dtype == object else int(state_fips)
    df = df[df["STATE"] == state_filter].copy()
    logger.info(f"Filtered to state {state_fips}: {len(df)} rows")

    # Map calendar year to YEAR code
    if year == 2020:
        year_code = "1"
    elif 2020 <= year <= 2024:
        year_code = str((year - 2020) + 2)
    else:
        raise ValueError(f"Year {year} out of range for cc-est2024 file (2020-2024)")

    logger.info(f"Calendar year {year} -> YEAR code '{year_code}'")

    # YEAR may be string or int
    if df["YEAR"].dtype == object:
        df = df[df["YEAR"] == year_code].copy()
    else:
        df = df[df["YEAR"] == int(year_code)].copy()
    logger.info(f"Filtered to YEAR={year_code}: {len(df)} rows")

    # Pivot wide columns to long format
    result = _pivot_wide_to_long(df, id_cols=["STATE", "COUNTY", "CTYNAME"])

    logger.info(
        f"Loaded {len(result)} records for {result['county_fips'].nunique()} counties, "
        f"year {year}, total pop {result['population'].sum():,.0f}"
    )
    return result.reset_index(drop=True)


# County name to FIPS mapping for North Dakota
_ND_COUNTY_NAME_TO_FIPS: dict[str, str] = {
    "Adams": "38001",
    "Barnes": "38003",
    "Benson": "38005",
    "Billings": "38007",
    "Bottineau": "38009",
    "Bowman": "38011",
    "Burke": "38013",
    "Burleigh": "38015",
    "Cass": "38017",
    "Cavalier": "38019",
    "Dickey": "38021",
    "Divide": "38023",
    "Dunn": "38025",
    "Eddy": "38027",
    "Emmons": "38029",
    "Foster": "38031",
    "Golden Valley": "38033",
    "Grand Forks": "38035",
    "Grant": "38037",
    "Griggs": "38039",
    "Hettinger": "38041",
    "Kidder": "38043",
    "LaMoure": "38045",
    "Logan": "38047",
    "McHenry": "38049",
    "McIntosh": "38051",
    "McKenzie": "38053",
    "McLean": "38055",
    "Mercer": "38057",
    "Morton": "38059",
    "Mountrail": "38061",
    "Nelson": "38063",
    "Oliver": "38065",
    "Pembina": "38067",
    "Pierce": "38069",
    "Ramsey": "38071",
    "Ransom": "38073",
    "Renville": "38075",
    "Richland": "38077",
    "Rolette": "38079",
    "Sargent": "38081",
    "Sheridan": "38083",
    "Sioux": "38085",
    "Slope": "38087",
    "Stark": "38089",
    "Steele": "38091",
    "Stutsman": "38093",
    "Towner": "38095",
    "Traill": "38097",
    "Walsh": "38099",
    "Ward": "38101",
    "Wells": "38103",
    "Williams": "38105",
}


def load_census_2020_base_population(
    file_path: str | Path,
) -> pd.DataFrame:
    """Load processed 2020 base population by county.

    This file is already in long format with columns
    [county_name, age_group, sex, population].  We add county_fips and
    standardize the sex column to Title case.

    Args:
        file_path: Path to base_population_by_county.csv.

    Returns:
        DataFrame with columns [county_fips, age_group, sex, population].

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Base population file not found: {file_path}")

    logger.info("Loading 2020 base population by county")
    df = pd.read_csv(file_path)

    # Standardize sex to Title case ("male" -> "Male")
    df["sex"] = df["sex"].str.title()

    # Map county name to FIPS
    df["county_fips"] = df["county_name"].map(_ND_COUNTY_NAME_TO_FIPS)

    # Check for unmapped counties
    unmapped = df[df["county_fips"].isna()]["county_name"].unique()
    if len(unmapped) > 0:
        logger.warning(f"Unmapped county names: {unmapped.tolist()}")

    result = df[["county_fips", "age_group", "sex", "population"]].copy()
    result = result.dropna(subset=["county_fips"])

    logger.info(
        f"Loaded {len(result)} records for {result['county_fips'].nunique()} counties, "
        f"total pop {result['population'].sum():,.0f}"
    )
    return result.reset_index(drop=True)
