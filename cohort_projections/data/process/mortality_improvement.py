"""
Mortality improvement processor for cohort projections.

Computes time-varying, ND-adjusted survival rates from Census Bureau NP2023
national survival ratio projections. Replaces static CDC ND 2020 survival
rates with projected rates that capture both:
  - National mortality improvement trends (Census Bureau NP2023-A4)
  - North Dakota's mortality differential relative to the national average

The core formula:
    ND_survival[age, sex, year] = Census_projected[age, sex, year] * ND_adjustment[age, sex]

where:
    ND_adjustment[age, sex] = ND_CDC_baseline[age, sex] / Census_national_2025[age, sex]
"""

import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from cohort_projections.utils import get_logger_from_config, load_projection_config

logger = get_logger_from_config(__name__)

# Sex code mapping from Census Bureau NP2023 format
_CENSUS_SEX_MAP: dict[int, str] = {1: "Male", 2: "Female"}


def load_census_survival_projections(
    file_path: Path | str,
    years: tuple[int, int] = (2025, 2045),
    sex_filter: list[int] | None = None,
    group_filter: int = 0,
) -> pd.DataFrame:
    """Load Census Bureau NP2023-A4 survival ratio projections.

    Reads the wide-format CSV, filters to specified years and demographics,
    pivots to long format with columns [year, age, sex, survival_ratio].

    Args:
        file_path: Path to np2023_a4_survival_ratios.csv.
        years: (start_year, end_year) inclusive range.
        sex_filter: SEX codes to include (default: [1, 2] for Male, Female).
        group_filter: GROUP code (default: 0 for All races).

    Returns:
        Long-format DataFrame with columns [year, age, sex, survival_ratio].
        Sex values are "Male" and "Female" (mapped from 1/2).

    Raises:
        FileNotFoundError: If file does not exist.
        ValueError: If no data remains after filtering.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Census survival projections file not found: {file_path}")

    if sex_filter is None:
        sex_filter = [1, 2]

    logger.info(f"Loading Census NP2023-A4 survival projections from {file_path}")

    df = pd.read_csv(file_path)
    logger.info(f"Loaded {len(df)} rows from {file_path.name}")

    # Filter to NATIVITY=0 (total), specified GROUP and SEX codes
    df = df[
        (df["NATIVITY"] == 0) & (df["GROUP"] == group_filter) & (df["SEX"].isin(sex_filter))
    ].copy()

    # Filter to year range
    start_year, end_year = years
    df = df[(df["YEAR"] >= start_year) & (df["YEAR"] <= end_year)].copy()

    if df.empty:
        raise ValueError(
            f"No data after filtering: years={years}, sex={sex_filter}, group={group_filter}"
        )

    logger.info(
        f"Filtered to {len(df)} rows: years {start_year}-{end_year}, sex codes {sex_filter}"
    )

    # Identify SRAT columns (SRAT_0 through SRAT_100)
    srat_cols = [c for c in df.columns if c.startswith("SRAT_")]

    # Pivot to long format
    id_cols = ["YEAR", "SEX"]
    long_df = df[id_cols + srat_cols].melt(
        id_vars=id_cols,
        value_vars=srat_cols,
        var_name="age_col",
        value_name="survival_ratio",
    )

    # Extract numeric age from column name (e.g. "SRAT_42" -> 42)
    long_df["age"] = long_df["age_col"].str.replace("SRAT_", "").astype(int)

    # Map sex codes to labels
    long_df["sex"] = long_df["SEX"].map(_CENSUS_SEX_MAP)

    # Rename and select final columns
    long_df = long_df.rename(columns={"YEAR": "year"})
    long_df = long_df[["year", "age", "sex", "survival_ratio"]].copy()
    long_df = long_df.sort_values(["year", "sex", "age"]).reset_index(drop=True)

    n_years = long_df["year"].nunique()
    n_ages = long_df["age"].nunique()
    n_sexes = long_df["sex"].nunique()
    logger.info(
        f"Pivoted to long format: {len(long_df)} rows "
        f"({n_years} years x {n_ages} ages x {n_sexes} sexes)"
    )

    return long_df


def load_nd_baseline_survival(
    file_path: Path | str,
) -> pd.DataFrame:
    """Load ND CDC baseline survival rates and expand to single-year ages.

    Reads the 5-year age group file, expands each group to constituent
    single-year ages (same rate for all years within group).
    The 85+ group expands to ages 85-100.

    Args:
        file_path: Path to survival_rates_sdc_2024_by_age_group.csv.

    Returns:
        DataFrame with columns [age, sex, survival_rate_1yr] covering ages 0-100.

    Raises:
        FileNotFoundError: If file does not exist.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"ND baseline survival file not found: {file_path}")

    logger.info(f"Loading ND CDC baseline survival rates from {file_path}")

    df = pd.read_csv(file_path)
    logger.info(f"Loaded {len(df)} rows ({df['sex'].nunique()} sexes)")

    records: list[dict] = []

    for _, row in df.iterrows():
        sex = row["sex"]
        rate_1yr = row["survival_rate_1yr"]
        age_start = int(row["age_start"])
        age_group = row["age_group"]

        if "+" in str(age_group):
            # 85+ group: expand to ages 85-100
            age_range = range(85, 101)
        else:
            age_end = int(row["age_end"])
            age_range = range(age_start, age_end + 1)

        records.extend({"age": age, "sex": sex, "survival_rate_1yr": rate_1yr} for age in age_range)

    result = pd.DataFrame(records)
    result = result.sort_values(["sex", "age"]).reset_index(drop=True)

    logger.info(
        f"Expanded to single-year ages: {len(result)} rows "
        f"(ages {result['age'].min()}-{result['age'].max()}, "
        f"{result['sex'].nunique()} sexes)"
    )

    return result


def compute_nd_adjustment_factors(
    nd_baseline: pd.DataFrame,
    census_national_baseline: pd.DataFrame,
) -> pd.DataFrame:
    """Compute ND-to-national adjustment factors.

    Ratio = ND_survival[age, sex] / Census_national_2025[age, sex]

    Values > 1.0 mean ND has better survival (lower mortality) than national.
    Values < 1.0 mean ND has worse survival (higher mortality).

    Args:
        nd_baseline: ND CDC rates with columns [age, sex, survival_rate_1yr].
        census_national_baseline: Census 2025 national rates with columns
            [age, sex, survival_ratio].

    Returns:
        DataFrame with columns [age, sex, adjustment_factor].
    """
    logger.info("Computing ND-to-national adjustment factors")

    merged = nd_baseline.merge(
        census_national_baseline[["age", "sex", "survival_ratio"]],
        on=["age", "sex"],
        how="inner",
    )

    if merged.empty:
        raise ValueError("No matching age-sex combinations between ND baseline and Census baseline")

    # Compute adjustment factor, guarding against division by zero
    merged["adjustment_factor"] = merged.apply(
        lambda row: (
            row["survival_rate_1yr"] / row["survival_ratio"] if row["survival_ratio"] > 0 else 1.0
        ),
        axis=1,
    )

    result = merged[["age", "sex", "adjustment_factor"]].copy()
    result = result.sort_values(["sex", "age"]).reset_index(drop=True)

    # Log summary statistics
    mean_factor = result["adjustment_factor"].mean()
    min_factor = result["adjustment_factor"].min()
    max_factor = result["adjustment_factor"].max()
    logger.info(
        f"Adjustment factors computed for {len(result)} age-sex combinations: "
        f"mean={mean_factor:.4f}, min={min_factor:.4f}, max={max_factor:.4f}"
    )

    # Count above/below 1.0
    above = (result["adjustment_factor"] > 1.0).sum()
    below = (result["adjustment_factor"] < 1.0).sum()
    logger.info(
        f"ND better than national (factor > 1.0): {above} age-sex combos; "
        f"ND worse (factor < 1.0): {below} age-sex combos"
    )

    return result


def build_nd_adjusted_survival_projections(
    census_projections: pd.DataFrame,
    adjustment_factors: pd.DataFrame,
    years: tuple[int, int] = (2025, 2045),
) -> pd.DataFrame:
    """Build ND-adjusted survival projections for all years.

    For each projection year:
        ND_survival[age, sex, year] = Census_projected[age, sex, year] * adjustment[age, sex]

    Caps at 1.0 and floors at 0.0.

    Args:
        census_projections: Census projected rates with columns
            [year, age, sex, survival_ratio].
        adjustment_factors: ND adjustment factors with columns
            [age, sex, adjustment_factor].
        years: (start_year, end_year) inclusive.

    Returns:
        DataFrame with columns [year, age, sex, survival_rate, source]
        where source = "Census_NP2023_ND_adjusted".
    """
    start_year, end_year = years
    logger.info(f"Building ND-adjusted survival projections for {start_year}-{end_year}")

    # Filter census projections to year range
    proj = census_projections[
        (census_projections["year"] >= start_year) & (census_projections["year"] <= end_year)
    ].copy()

    # Merge with adjustment factors
    merged = proj.merge(adjustment_factors, on=["age", "sex"], how="left")

    # Fill missing adjustment factors with 1.0 (no adjustment)
    missing_count = merged["adjustment_factor"].isna().sum()
    if missing_count > 0:
        logger.warning(
            f"Missing adjustment factors for {missing_count} rows; using 1.0 (no adjustment)"
        )
        merged["adjustment_factor"] = merged["adjustment_factor"].fillna(1.0)

    # Apply adjustment
    merged["survival_rate"] = merged["survival_ratio"] * merged["adjustment_factor"]

    # Cap at 1.0, floor at 0.0
    merged["survival_rate"] = merged["survival_rate"].clip(lower=0.0, upper=1.0)

    # Add source column
    merged["source"] = "Census_NP2023_ND_adjusted"

    result = merged[["year", "age", "sex", "survival_rate", "source"]].copy()
    result = result.sort_values(["year", "sex", "age"]).reset_index(drop=True)

    logger.info(
        f"Built {len(result)} ND-adjusted survival projections "
        f"({result['year'].nunique()} years x "
        f"{result['age'].nunique()} ages x "
        f"{result['sex'].nunique()} sexes)"
    )

    # Log improvement summary
    if result["year"].nunique() >= 2:
        first_year = result[result["year"] == start_year]["survival_rate"].mean()
        last_year = result[result["year"] == end_year]["survival_rate"].mean()
        logger.info(
            f"Mean survival rate: {first_year:.6f} ({start_year}) -> "
            f"{last_year:.6f} ({end_year}), "
            f"improvement = {last_year - first_year:+.6f}"
        )

    return result


def run_mortality_improvement_pipeline(
    config: dict | None = None,
) -> pd.DataFrame:
    """Main pipeline orchestrator for mortality improvement.

    Steps:
        1. Load Census Bureau NP2023 survival projections (2025-2045)
        2. Load ND CDC baseline (expand to single-year)
        3. Extract Census 2025 national baseline for adjustment
        4. Compute ND adjustment factors
        5. Apply adjustments to all projection years
        6. Save output

    Args:
        config: Optional configuration dictionary. If None, loads from
            config/projection_config.yaml.

    Returns:
        DataFrame with ND-adjusted survival projections.
    """
    logger.info("=" * 70)
    logger.info("Starting Mortality Improvement Pipeline (Phase 3)")
    logger.info("=" * 70)

    # Load config if not provided
    if config is None:
        config = load_projection_config()

    # Determine project root and file paths
    project_root = Path(__file__).parent.parent.parent.parent

    census_file = (
        project_root
        / "data"
        / "raw"
        / "census_bureau_methodology"
        / "np2023_a4_survival_ratios.csv"
    )
    nd_baseline_file = (
        project_root
        / "data"
        / "processed"
        / "sdc_2024"
        / "survival_rates_sdc_2024_by_age_group.csv"
    )
    output_dir = project_root / "data" / "processed" / "mortality"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get projection year range from config
    base_year = config.get("project", {}).get("base_year", 2025)
    horizon = config.get("project", {}).get("projection_horizon", 20)
    end_year = base_year + horizon
    years = (base_year, end_year)

    logger.info(f"Projection years: {base_year}-{end_year}")

    # Step 1: Load Census Bureau survival projections
    logger.info("Step 1: Loading Census Bureau NP2023 survival projections")
    census_projections = load_census_survival_projections(
        census_file,
        years=years,
        sex_filter=[1, 2],
        group_filter=0,
    )

    # Step 2: Load ND CDC baseline (expanded to single-year ages)
    logger.info("Step 2: Loading ND CDC baseline survival rates")
    nd_baseline = load_nd_baseline_survival(nd_baseline_file)

    # Step 3: Extract Census 2025 national baseline
    logger.info("Step 3: Extracting Census 2025 national baseline")
    census_2025 = census_projections[census_projections["year"] == base_year].copy()

    if census_2025.empty:
        raise ValueError(
            f"No Census data for base year {base_year}. "
            f"Available years: {sorted(census_projections['year'].unique())}"
        )

    logger.info(f"Census 2025 baseline: {len(census_2025)} rows")

    # Step 4: Compute ND adjustment factors
    logger.info("Step 4: Computing ND adjustment factors")
    adjustment_factors = compute_nd_adjustment_factors(nd_baseline, census_2025)

    # Step 5: Apply adjustments to all projection years
    logger.info("Step 5: Applying ND adjustments to all projection years")
    nd_adjusted = build_nd_adjusted_survival_projections(
        census_projections,
        adjustment_factors,
        years=years,
    )

    # Step 6: Save output
    logger.info("Step 6: Saving output")

    # Save parquet
    compression = config.get("output", {}).get("compression", "gzip")
    parquet_file = output_dir / "nd_adjusted_survival_projections.parquet"
    nd_adjusted.to_parquet(parquet_file, compression=compression, index=False)
    logger.info(f"Saved parquet: {parquet_file}")

    # Save metadata
    metadata = {
        "processing_date": datetime.now(UTC).isoformat(),
        "pipeline": "mortality_improvement_phase3",
        "source_files": {
            "census_survival_projections": str(census_file),
            "nd_cdc_baseline": str(nd_baseline_file),
        },
        "parameters": {
            "projection_years": list(years),
            "base_year": base_year,
            "end_year": end_year,
            "census_sex_filter": [1, 2],
            "census_group_filter": 0,
            "nativity_filter": 0,
        },
        "output": {
            "total_rows": len(nd_adjusted),
            "years": sorted(nd_adjusted["year"].unique().tolist()),
            "ages": [
                int(nd_adjusted["age"].min()),
                int(nd_adjusted["age"].max()),
            ],
            "sexes": sorted(nd_adjusted["sex"].unique().tolist()),
            "survival_rate_range": [
                float(nd_adjusted["survival_rate"].min()),
                float(nd_adjusted["survival_rate"].max()),
            ],
        },
        "adjustment_factors": {
            "mean": float(adjustment_factors["adjustment_factor"].mean()),
            "min": float(adjustment_factors["adjustment_factor"].min()),
            "max": float(adjustment_factors["adjustment_factor"].max()),
            "std": float(adjustment_factors["adjustment_factor"].std()),
        },
        "methodology": (
            "ND_survival[age, sex, year] = "
            "Census_NP2023[age, sex, year] * "
            "(ND_CDC_2020[age, sex] / Census_national_2025[age, sex])"
        ),
    }

    metadata_file = output_dir / "mortality_improvement_metadata.json"
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved metadata: {metadata_file}")

    # Summary
    logger.info("=" * 70)
    logger.info("Mortality Improvement Pipeline Complete")
    logger.info("=" * 70)
    logger.info(f"Total rows: {len(nd_adjusted)}")
    logger.info(
        f"Years: {nd_adjusted['year'].min()}-{nd_adjusted['year'].max()} "
        f"({nd_adjusted['year'].nunique()} years)"
    )
    logger.info(
        f"Ages: {nd_adjusted['age'].min()}-{nd_adjusted['age'].max()} "
        f"({nd_adjusted['age'].nunique()} ages)"
    )
    logger.info(f"Sexes: {sorted(nd_adjusted['sex'].unique().tolist())}")
    logger.info(
        f"Survival rate range: "
        f"[{nd_adjusted['survival_rate'].min():.6f}, "
        f"{nd_adjusted['survival_rate'].max():.6f}]"
    )
    logger.info(f"Mean adjustment factor: {adjustment_factors['adjustment_factor'].mean():.4f}")
    logger.info("=" * 70)

    return nd_adjusted
