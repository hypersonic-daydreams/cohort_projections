"""
Residual migration rate computation for cohort projections.

Computes age-sex-specific net migration rates using the Census Bureau
residual method:

    expected_pop[age+5] = pop_start[age] * survival_rate_5yr[age, sex]
    migration[age+5]    = pop_end[age+5] - expected_pop[age+5]
    migration_rate_period = migration / expected_pop
    migration_rate        = annualize(migration_rate_period, period_length)

This module implements the full pipeline:
    1. Load population snapshots for 6 time points (2000-2024)
    1b. (ADR-055 Phase 2) Subtract GQ from population snapshots
    2. Load survival rates
    3. Compute residual migration for each 5-year period, applying:
       - Oil-boom dampening to Bakken counties (ADR-040, ADR-051)
       - Male migration dampening for boom periods
    4. PEP recalibration for reservation counties (ADR-045)
    5. Apply college-age smoothing to period-level rates (ADR-049)
    6. Average rates across all periods
    7. Save output files (period-level and averaged)

ADR-049 note: College-age smoothing is applied to period-level rates
BEFORE averaging and saving. This ensures the convergence pipeline
(which reads residual_migration_rates.parquet) operates on smoothed
rates. Smoothing is NOT re-applied to averaged rates to avoid
double-smoothing.

ADR-055 Phase 2: When GQ correction is enabled, population snapshots
have group quarters subtracted before residual computation. This
produces migration rates computed on household-only population, removing
institutional rotation signals (dorm turnover, military PCS cycles).

Output files are written to data/processed/migration/.
"""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from cohort_projections.data.load.census_age_sex_population import (
    AGE_GROUP_LABELS,
    load_census_2000_county_age_sex,
    load_census_2020_base_population,
    load_pep_2010_2019_county_age_sex,
    load_pep_2020_2024_county_age_sex,
)
from cohort_projections.utils import get_logger_from_config, load_projection_config

logger = get_logger_from_config(__name__)

# Default oil-boom counties (Bakken region)
OIL_COUNTIES: list[str] = ["38105", "38053", "38061", "38025", "38089"]


def _age_group_index(label: str) -> int:
    """Return the positional index of an age group label in AGE_GROUP_LABELS."""
    return AGE_GROUP_LABELS.index(label)


def _annualize_period_migration_rate(period_rate: float, period_length: int) -> float:
    """Convert a multi-year net migration rate into an annual rate.

    Residual migration is computed over the full period length (typically
    5 years, with one 4-year period for 2020-2024). The projection engine
    applies migration annually, so rates must be converted:

        annual = (1 + period_rate) ** (1 / period_length) - 1

    Args:
        period_rate: Net migration rate over the full period.
        period_length: Period length in years.

    Returns:
        Annualized migration rate.
    """
    if period_length <= 0:
        raise ValueError(f"period_length must be positive, got {period_length}")

    if period_rate <= -1.0:
        return -1.0

    return (1.0 + period_rate) ** (1.0 / period_length) - 1.0


def compute_residual_migration_rates(
    pop_start: pd.DataFrame,
    pop_end: pd.DataFrame,
    survival_rates: pd.DataFrame,
    period: tuple[int, int],
) -> pd.DataFrame:
    """Compute age-sex-specific net migration rates via the residual method.

    For each county, sex, and age group the formula is:

        expected = pop_start[age_group] * survival_rate_5yr[age_group, sex]
        migration = pop_end[next_age_group] - expected
        migration_rate_period = migration / expected   (if expected > 0)
        migration_rate = annualize(migration_rate_period, period_length)

    Age shifting: 0-4 -> 5-9, 5-9 -> 10-14, ..., 75-79 -> 80-84.

    The 85+ open-ended group is handled specially:
        expected_85plus = (pop[80-84] * surv[80-84]) + (pop[85+] * surv[85+])
        migration_85plus = pop_end[85+] - expected_85plus

    The 0-4 birth cohort at the end of the period cannot be computed via
    residual (no starting cohort to age forward).  These rows are included
    with migration_rate = 0.0.

    For the 2020-2024 period (4 years instead of 5), survival rates are
    adjusted: surv_4yr = surv_5yr ** (4/5).

    Args:
        pop_start: Population at start of period.
            Columns: [county_fips, age_group, sex, population].
        pop_end: Population at end of period.
            Columns: [county_fips, age_group, sex, population].
        survival_rates: Survival rates by age group and sex.
            Columns: [age_group, sex, survival_rate_5yr].
        period: Tuple of (start_year, end_year).

    Returns:
        DataFrame with columns:
        [county_fips, age_group, sex, period_start, period_end,
         pop_start, expected_pop, pop_end, net_migration, migration_rate].
    """
    start_year, end_year = period
    period_length = end_year - start_year
    logger.info(f"Computing residual migration for period {start_year}-{end_year}")

    # Adjust survival rates if period is not exactly 5 years
    surv = survival_rates[["age_group", "sex", "survival_rate_5yr"]].copy()
    if period_length != 5:
        logger.info(
            f"Period is {period_length} years (not 5); adjusting survival rates "
            f"with exponent {period_length}/5 = {period_length / 5:.2f}"
        )
        surv["survival_rate_5yr"] = surv["survival_rate_5yr"] ** (period_length / 5.0)

    # Build lookup for survival rates: (age_group, sex) -> rate
    surv_lookup: dict[tuple[str, str], float] = {}
    for _, row in surv.iterrows():
        surv_lookup[(row["age_group"], row["sex"])] = row["survival_rate_5yr"]

    # Build population lookups: (county_fips, age_group, sex) -> population
    def _build_pop_lookup(df: pd.DataFrame) -> dict[tuple[str, str, str], float]:
        lookup: dict[tuple[str, str, str], float] = {}
        for _, row in df.iterrows():
            key = (row["county_fips"], row["age_group"], row["sex"])
            lookup[key] = float(row["population"])
        return lookup

    pop_s = _build_pop_lookup(pop_start)
    pop_e = _build_pop_lookup(pop_end)

    # Get list of counties and sexes
    counties = sorted(pop_start["county_fips"].unique())
    sexes = ["Male", "Female"]

    records: list[dict[str, Any]] = []

    for county in counties:
        for sex in sexes:
            # Process age-shifted groups: 0-4 -> 5-9, ..., 75-79 -> 80-84
            for i in range(len(AGE_GROUP_LABELS) - 2):  # 0 through 16 (0-4 to 80-84)
                start_ag = AGE_GROUP_LABELS[i]  # age group at start
                end_ag = AGE_GROUP_LABELS[i + 1]  # shifted age group at end

                p_start = pop_s.get((county, start_ag, sex), 0.0)
                p_end = pop_e.get((county, end_ag, sex), 0.0)
                s_rate = surv_lookup.get((start_ag, sex), 1.0)

                expected = p_start * s_rate
                migration = p_end - expected
                rate_period = migration / expected if expected > 0 else 0.0
                rate = (
                    _annualize_period_migration_rate(rate_period, period_length)
                    if expected > 0
                    else 0.0
                )

                records.append(
                    {
                        "county_fips": county,
                        "age_group": end_ag,  # The age group at END of period
                        "sex": sex,
                        "period_start": start_year,
                        "period_end": end_year,
                        "pop_start": p_start,
                        "expected_pop": expected,
                        "pop_end": p_end,
                        "net_migration": migration,
                        "migration_rate": rate,
                    }
                )

            # 85+ open-ended group
            p_80_84 = pop_s.get((county, "80-84", sex), 0.0)
            p_85plus = pop_s.get((county, "85+", sex), 0.0)
            s_80_84 = surv_lookup.get(("80-84", sex), 1.0)
            s_85plus = surv_lookup.get(("85+", sex), 1.0)

            expected_85plus = (p_80_84 * s_80_84) + (p_85plus * s_85plus)
            p_end_85plus = pop_e.get((county, "85+", sex), 0.0)
            migration_85plus = p_end_85plus - expected_85plus
            rate_85plus_period = migration_85plus / expected_85plus if expected_85plus > 0 else 0.0
            rate_85plus = (
                _annualize_period_migration_rate(rate_85plus_period, period_length)
                if expected_85plus > 0
                else 0.0
            )

            records.append(
                {
                    "county_fips": county,
                    "age_group": "85+",
                    "sex": sex,
                    "period_start": start_year,
                    "period_end": end_year,
                    "pop_start": p_80_84 + p_85plus,
                    "expected_pop": expected_85plus,
                    "pop_end": p_end_85plus,
                    "net_migration": migration_85plus,
                    "migration_rate": rate_85plus,
                }
            )

            # 0-4 birth cohort: cannot compute via residual
            records.append(
                {
                    "county_fips": county,
                    "age_group": "0-4",
                    "sex": sex,
                    "period_start": start_year,
                    "period_end": end_year,
                    "pop_start": 0.0,
                    "expected_pop": 0.0,
                    "pop_end": pop_e.get((county, "0-4", sex), 0.0),
                    "net_migration": 0.0,
                    "migration_rate": 0.0,
                }
            )

    result = pd.DataFrame(records)

    n_counties = result["county_fips"].nunique()
    total_mig = result["net_migration"].sum()
    logger.info(
        f"Period {start_year}-{end_year}: {n_counties} counties, "
        f"total net migration {total_mig:+,.0f}"
    )

    return result


def assemble_period_populations(
    config: dict[str, Any],
) -> dict[int, pd.DataFrame]:
    """Load populations for all required time points.

    Uses the data loaders from census_age_sex_population.py to load
    population snapshots at each time point needed for residual migration
    computation.

    Time points and sources:
        - 2000: Census 2000 (ESTIMATESBASE2000)
        - 2005: Census 2000 file (POPESTIMATE2005)
        - 2010: PEP 2010-2019 (YEAR=1, census base)
        - 2015: PEP 2010-2019 (YEAR=6)
        - 2020: SDC 2024 base population by county
        - 2024: PEP 2020-2024 (YEAR=6)

    Args:
        config: Project configuration dictionary containing data file paths.

    Returns:
        Dictionary mapping year to DataFrame with standard columns
        [county_fips, age_group, sex, population].
    """
    logger.info("Assembling period populations for all time points")

    # Get project root for relative paths
    project_root = Path(__file__).parent.parent.parent.parent

    # Get data paths from config
    data_paths = config.get("data_paths", {})

    # Census 2000 file
    census_2000_path = data_paths.get(
        "census_2000_county_age_sex",
        str(
            project_root
            / "data"
            / "raw"
            / "nd_sdc_2024_projections"
            / "source_files"
            / "reference"
            / "Census 2000 County Age and Sex.xlsx"
        ),
    )

    # PEP 2010-2019 file
    pep_2010_2019_path = data_paths.get(
        "pep_2010_2019_county_age_sex",
        str(
            project_root
            / "data"
            / "raw"
            / "nd_sdc_2024_projections"
            / "source_files"
            / "reference"
            / "cc-est2019-agesex-38 (1).xlsx"
        ),
    )

    # PEP 2010-2020 intercensal file (shared data)
    shared_data_root = Path.home() / "workspace" / "shared-data"
    data_paths.get(
        "pep_2020_intercensal",
        str(
            shared_data_root
            / "census"
            / "popest"
            / "parquet"
            / "2010-2020"
            / "county"
            / "cc-est2020int-alldata.parquet"
        ),
    )

    # PEP 2020-2024 file (shared data)
    pep_2020_2024_path = data_paths.get(
        "pep_2020_2024_county_age_sex",
        str(
            shared_data_root
            / "census"
            / "popest"
            / "parquet"
            / "2020-2024"
            / "county"
            / "cc-est2024-agesex-all.parquet"
        ),
    )

    # SDC 2024 base population
    base_pop_2020_path = data_paths.get(
        "base_population_2020",
        str(project_root / "sdc_2024_replication" / "data" / "base_population_by_county.csv"),
    )

    populations: dict[int, pd.DataFrame] = {}

    # Year 2000
    logger.info("Loading year 2000 population")
    populations[2000] = load_census_2000_county_age_sex(
        census_2000_path, state_fips="38", year=2000
    )

    # Year 2005
    logger.info("Loading year 2005 population")
    populations[2005] = load_census_2000_county_age_sex(
        census_2000_path, state_fips="38", year=2005
    )

    # Year 2010
    logger.info("Loading year 2010 population")
    populations[2010] = load_pep_2010_2019_county_age_sex(
        pep_2010_2019_path, state_fips="38", year=2010
    )

    # Year 2015
    logger.info("Loading year 2015 population")
    populations[2015] = load_pep_2010_2019_county_age_sex(
        pep_2010_2019_path, state_fips="38", year=2015
    )

    # Year 2020
    logger.info("Loading year 2020 population")
    populations[2020] = load_census_2020_base_population(base_pop_2020_path)

    # Year 2024
    logger.info("Loading year 2024 population")
    populations[2024] = load_pep_2020_2024_county_age_sex(
        pep_2020_2024_path, state_fips="38", year=2024
    )

    # Log summary
    for year, df in sorted(populations.items()):
        logger.info(
            f"  Year {year}: {df['county_fips'].nunique()} counties, "
            f"total pop {df['population'].sum():,.0f}"
        )

    return populations


def apply_period_dampening(
    period_rates: pd.DataFrame,
    period: tuple[int, int],
    dampening_config: dict[str, Any],
) -> pd.DataFrame:
    """Apply dampening to oil county rates for boom-era periods.

    Reduces migration rates in designated oil-boom counties during boom
    periods to prevent unrepeatable extreme migration from dominating
    the multi-period average.

    Only boom periods (default: (2005,2010) and (2010,2015)) are dampened.
    All other periods are returned unchanged.

    Args:
        period_rates: DataFrame with migration_rate column and county_fips.
        period: Tuple (start_year, end_year) for the current period.
        dampening_config: Dict with keys:
            - enabled: bool
            - factor: float multiplier (e.g., 0.60), or dict of period-specific
              factors (e.g., {"2005-2010": 0.50, "2010-2015": 0.40})
            - counties: list of 5-digit FIPS to dampen
            - boom_periods: list of [start, end] period tuples

    Returns:
        DataFrame with dampened rates (copy; original unchanged).
    """
    if not dampening_config.get("enabled", True):
        return period_rates.copy()

    factor_cfg = dampening_config.get("factor", 0.60)
    counties = dampening_config.get("counties", OIL_COUNTIES)
    boom_periods = dampening_config.get("boom_periods", [[2005, 2010], [2010, 2015]])

    # Check if this period is a boom period
    is_boom = any(period[0] == bp[0] and period[1] == bp[1] for bp in boom_periods)

    if not is_boom:
        logger.debug(f"Period {period[0]}-{period[1]} is not a boom period; no dampening applied")
        return period_rates.copy()

    # Resolve period-specific factor (ADR-051: supports dict or scalar)
    period_key_str = f"{period[0]}-{period[1]}"
    if isinstance(factor_cfg, dict):
        factor = factor_cfg.get(period_key_str, factor_cfg.get("default", 0.60))
    else:
        factor = float(factor_cfg)

    result = period_rates.copy()
    mask = result["county_fips"].isin(counties)
    n_affected = mask.sum()

    if n_affected > 0:
        result.loc[mask, "migration_rate"] *= factor
        result.loc[mask, "net_migration"] *= factor
        logger.info(
            f"Period {period[0]}-{period[1]}: dampened {n_affected} rows "
            f"in {len(counties)} oil counties by factor {factor:.2f}"
        )
    else:
        logger.debug(f"No oil county rows found for period {period[0]}-{period[1]}")

    return result


def apply_college_age_adjustment(
    rates: pd.DataFrame,
    college_counties: list[str],
    method: str = "smooth",
    age_groups: list[str] | None = None,
    blend_factor: float = 0.5,
) -> pd.DataFrame:
    """Smooth or cap migration rates for college-age groups in college counties.

    College towns show extreme in/out migration for ages 15-24 due to
    enrollment cycles.  This function blends college county rates with the
    statewide average to reduce noise.

    Args:
        rates: DataFrame with columns [county_fips, age_group, sex, migration_rate].
        college_counties: List of county FIPS codes with colleges.
        method: "smooth" to blend with statewide average; "cap" to clip
                absolute rates at a threshold.
        age_groups: Which age groups to adjust (default: ["15-19", "20-24"]).
        blend_factor: Weight on county-specific rate when method="smooth".
                      0.5 means 50% county + 50% statewide average.

    Returns:
        DataFrame with adjusted rates (copy; original unchanged).
    """
    if age_groups is None:
        age_groups = ["15-19", "20-24"]

    result = rates.copy()

    if method == "smooth":
        # Compute statewide average rate for each age_group x sex
        state_avg = result.groupby(["age_group", "sex"], as_index=False).agg(
            state_avg_rate=("migration_rate", "mean")
        )

        mask = result["county_fips"].isin(college_counties) & result["age_group"].isin(age_groups)

        if mask.any():
            # Merge state average onto result
            result = result.merge(state_avg, on=["age_group", "sex"], how="left")

            # Blend: county_rate * blend_factor + state_avg * (1 - blend_factor)
            result.loc[mask, "migration_rate"] = result.loc[
                mask, "migration_rate"
            ] * blend_factor + result.loc[mask, "state_avg_rate"] * (1 - blend_factor)

            result = result.drop(columns=["state_avg_rate"])

            logger.info(
                f"College age adjustment: smoothed {mask.sum()} rows in "
                f"{len(college_counties)} counties, blend_factor={blend_factor}"
            )

    elif method == "cap":
        cap_value = 0.20  # 20% migration rate cap
        mask = result["county_fips"].isin(college_counties) & result["age_group"].isin(age_groups)

        if mask.any():
            result.loc[mask, "migration_rate"] = result.loc[mask, "migration_rate"].clip(
                lower=-cap_value, upper=cap_value
            )

            logger.info(f"College age adjustment: capped {mask.sum()} rows at +/-{cap_value:.0%}")
    else:
        logger.warning(f"Unknown college age adjustment method: {method}")

    return result


def apply_male_migration_dampening(
    rates: pd.DataFrame,
    period: tuple[int, int],
    boom_periods: list[list[int]] | None = None,
    male_dampening_factor: float = 0.80,
) -> pd.DataFrame:
    """Apply additional dampening to male migration rates in boom periods.

    During oil boom periods, male in-migration was disproportionately high.
    This function reduces male rates to prevent overestimation.

    Args:
        rates: DataFrame with columns [county_fips, age_group, sex, migration_rate].
        period: Current period tuple (start_year, end_year).
        boom_periods: List of [start, end] boom period definitions.
        male_dampening_factor: Multiplier for male rates (default 0.80).

    Returns:
        DataFrame with dampened male rates (copy; original unchanged).
    """
    if boom_periods is None:
        boom_periods = [[2005, 2010], [2010, 2015]]

    is_boom = any(period[0] == bp[0] and period[1] == bp[1] for bp in boom_periods)

    if not is_boom:
        return rates.copy()

    result = rates.copy()
    male_mask = result["sex"] == "Male"
    n_males = male_mask.sum()

    if n_males > 0:
        result.loc[male_mask, "migration_rate"] *= male_dampening_factor
        result.loc[male_mask, "net_migration"] *= male_dampening_factor
        logger.info(
            f"Male dampening: reduced {n_males} male rows by factor "
            f"{male_dampening_factor:.2f} for boom period {period[0]}-{period[1]}"
        )

    return result


def _get_rogers_castro_age_group_weights() -> dict[str, float]:
    """Return Rogers-Castro migration propensity weights for 18 five-year age groups.

    Uses the same Rogers-Castro model as
    ``get_standard_age_migration_pattern(method='rogers_castro')`` from
    ``migration_rates.py``, aggregated to five-year groups and normalised
    so they sum to 1.0.

    Returns:
        Dictionary mapping age_group label (e.g. "0-4") to its share of
        total migration propensity.
    """
    from cohort_projections.data.process.migration_rates import (
        get_standard_age_migration_pattern,
    )

    rc = get_standard_age_migration_pattern(peak_age=25, method="rogers_castro")

    # Map single-year ages to five-year age groups
    def _age_to_group(age: int) -> str:
        if age >= 85:
            return "85+"
        lower = (age // 5) * 5
        upper = lower + 4
        return f"{lower}-{upper}"

    rc["age_group"] = rc["age"].apply(_age_to_group)
    group_sums = rc.groupby("age_group")["migration_propensity"].sum()

    # Normalise
    total = group_sums.sum()
    weights: dict[str, float] = {}
    for ag in AGE_GROUP_LABELS:
        weights[ag] = float(group_sums.get(ag, 0.0) / total)

    return weights


def _rogers_castro_to_age_group_rates(
    total_migration: float,
    expected_pop: pd.DataFrame,
    sex_ratio: float = 0.5,
) -> pd.DataFrame:
    """Distribute *total_migration* to age-group x sex rates using Rogers-Castro.

    Steps:
        1. Split total to Male/Female using *sex_ratio*.
        2. Allocate each sex's total across 18 age groups using the
           Rogers-Castro propensity weights.
        3. Convert counts to annual rates by dividing by the county's
           expected population for each cell.

    Args:
        total_migration: Net migration person-count to distribute (can be
            negative for net out-migration).
        expected_pop: DataFrame with columns ``[age_group, sex, expected_pop]``
            for the county-period being recalibrated.  Used as the
            denominator when converting counts to rates.
        sex_ratio: Proportion allocated to males (default 0.5).

    Returns:
        DataFrame with columns ``[age_group, sex, rc_migration, rc_rate]``
        where *rc_rate* is the annualised rate to substitute into the
        residual rates frame.
    """
    rc_weights = _get_rogers_castro_age_group_weights()

    records: list[dict[str, Any]] = []
    for sex in ["Male", "Female"]:
        sex_share = sex_ratio if sex == "Male" else (1.0 - sex_ratio)
        sex_total = total_migration * sex_share

        for ag in AGE_GROUP_LABELS:
            mig_count = sex_total * rc_weights.get(ag, 0.0)
            exp = expected_pop.loc[
                (expected_pop["age_group"] == ag) & (expected_pop["sex"] == sex),
                "expected_pop",
            ]
            exp_val = float(exp.iloc[0]) if not exp.empty else 0.0
            rate = mig_count / exp_val if exp_val > 0 else 0.0
            records.append(
                {
                    "age_group": ag,
                    "sex": sex,
                    "rc_migration": mig_count,
                    "rc_rate": rate,
                }
            )

    return pd.DataFrame(records)


def _load_pep_for_recalibration(pep_path: str | Path) -> pd.DataFrame:
    """Load PEP county components for recalibration, filtering to preferred estimates.

    Returns a DataFrame with columns ``[geoid, year, netmig]`` where
    ``geoid`` is the 5-digit FIPS (e.g. ``"38005"``).
    """
    pep_path = Path(pep_path)
    if not pep_path.exists():
        raise FileNotFoundError(f"PEP data file not found: {pep_path}")

    df = pd.read_parquet(pep_path)

    if "is_preferred_estimate" in df.columns:
        df = df[df["is_preferred_estimate"]].copy()

    # Ensure geoid column exists
    if "geoid" not in df.columns:
        # Build from state + county FIPS
        df["geoid"] = df["state_fips"].astype(str) + df["county_fips"].astype(str).str.zfill(3)

    return df[["geoid", "year", "netmig"]].copy()


def apply_pep_recalibration(
    period_rates: pd.DataFrame,
    period: tuple[int, int],
    pep_data: pd.DataFrame,
    counties: list[str],
    near_zero_threshold: float = 10.0,
    sex_ratio: float = 0.5,
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    """Recalibrate residual rates for specified counties using PEP totals.

    For each target county and period:

    1. Compute PEP total net migration for the period (sum of annual
       ``netmig`` for years within the period boundaries).
    2. Compute residual total net migration for that county-period.
    3. **Same sign and |residual| > threshold**: scale all rates by
       ``k = PEP_total / residual_total`` (preserves age-sex shape).
    4. **Sign reversal or near-zero residual**: redistribute PEP total
       using Rogers-Castro age pattern with 50/50 sex split.

    Non-target counties are passed through unchanged.

    Args:
        period_rates: DataFrame produced by ``compute_residual_migration_rates``
            with columns including ``county_fips``, ``age_group``, ``sex``,
            ``migration_rate``, ``net_migration``, ``expected_pop``.
        period: ``(start_year, end_year)`` tuple.
        pep_data: DataFrame with columns ``[geoid, year, netmig]`` where
            ``geoid`` is 5-digit FIPS (e.g. ``"38005"``).
        counties: List of 5-digit FIPS codes to recalibrate.
        near_zero_threshold: Absolute person-count below which the residual
            total is considered near-zero and the Rogers-Castro fallback is
            used (default 10).
        sex_ratio: Male share for Rogers-Castro fallback (default 0.5).

    Returns:
        Tuple of ``(recalibrated_rates, metadata_dict)`` where
        *metadata_dict* maps county FIPS to a dict describing the method
        used (``"scaled"`` or ``"rogers_castro"``) and relevant parameters.
    """
    start_year, end_year = period
    period_length = end_year - start_year
    result = period_rates.copy()
    metadata: dict[str, dict[str, Any]] = {}

    for county in counties:
        # --- PEP total for the period ---
        # Period convention: for period (2000, 2005), sum PEP years
        # 2001-2005 (the years *within* the intercensal period).
        # For (2020, 2024), sum years 2021-2024.
        pep_years = pep_data[
            (pep_data["geoid"] == county)
            & (pep_data["year"] > start_year)
            & (pep_data["year"] <= end_year)
        ]

        if pep_years.empty:
            logger.warning(
                f"PEP recalibration: no PEP data for county {county} "
                f"in period {start_year}-{end_year}; skipping"
            )
            continue

        pep_total = float(pep_years["netmig"].sum())

        # --- Residual total for the county-period ---
        county_mask = result["county_fips"] == county
        county_rows = result.loc[county_mask]

        if county_rows.empty:
            logger.warning(
                f"PEP recalibration: no residual data for county {county} "
                f"in period {start_year}-{end_year}; skipping"
            )
            continue

        residual_total = float(county_rows["net_migration"].sum())

        # --- Decide method ---
        if abs(residual_total) < near_zero_threshold:
            # Near-zero residual: use Rogers-Castro fallback
            method = "rogers_castro"
            reason = "near_zero"
            logger.info(
                f"PEP recalibration {county} {start_year}-{end_year}: "
                f"Rogers-Castro fallback (|residual|={abs(residual_total):.0f} < "
                f"threshold={near_zero_threshold}). PEP total={pep_total:+,.0f}"
            )
        elif (pep_total >= 0 and residual_total >= 0) or (pep_total <= 0 and residual_total <= 0):
            # Same sign: scale
            method = "scaled"
            reason = "same_sign"
        else:
            # Sign reversal: use Rogers-Castro fallback
            method = "rogers_castro"
            reason = "sign_reversal"
            logger.info(
                f"PEP recalibration {county} {start_year}-{end_year}: "
                f"Rogers-Castro fallback (sign reversal: PEP={pep_total:+,.0f}, "
                f"residual={residual_total:+,.0f})"
            )

        if method == "scaled":
            k = pep_total / residual_total
            result.loc[county_mask, "migration_rate"] *= k
            result.loc[county_mask, "net_migration"] *= k
            logger.info(
                f"PEP recalibration {county} {start_year}-{end_year}: "
                f"scaled by k={k:.3f} (PEP={pep_total:+,.0f}, "
                f"residual={residual_total:+,.0f})"
            )
            metadata[county] = {
                "method": "scaled",
                "k": round(k, 4),
                "pep_total": pep_total,
                "residual_total": residual_total,
            }
        else:
            # Rogers-Castro fallback: distribute PEP total using RC pattern
            # Build expected_pop for this county from the residual data
            exp_pop = county_rows[["age_group", "sex", "expected_pop"]].copy()

            rc_rates = _rogers_castro_to_age_group_rates(
                total_migration=pep_total,
                expected_pop=exp_pop,
                sex_ratio=sex_ratio,
            )

            # Annualise: the RC rates are already per-period-length since we
            # divided count by expected_pop (which is the denominator for
            # the period rate).  Convert to annual rate the same way the
            # residual pipeline does.
            for idx, row in rc_rates.iterrows():
                rate_period = row["rc_rate"]
                if abs(rate_period) > 0 and rate_period > -1.0:
                    annual = _annualize_period_migration_rate(rate_period, period_length)
                else:
                    annual = rate_period
                rc_rates.at[idx, "rc_rate"] = annual

            # Apply to result
            for _, rc_row in rc_rates.iterrows():
                mask = (
                    county_mask
                    & (result["age_group"] == rc_row["age_group"])
                    & (result["sex"] == rc_row["sex"])
                )
                if mask.any():
                    result.loc[mask, "migration_rate"] = rc_row["rc_rate"]
                    result.loc[mask, "net_migration"] = rc_row["rc_migration"]

            metadata[county] = {
                "method": "rogers_castro",
                "reason": reason,
                "pep_total": pep_total,
                "residual_total": residual_total,
            }

    return result, metadata


def average_period_rates(
    period_rates: dict[tuple[int, int], pd.DataFrame],
    method: str = "simple_average",
) -> pd.DataFrame:
    """Average migration rates across all periods per county x age x sex.

    Args:
        period_rates: Dict mapping period tuple to DataFrame with columns
            [county_fips, age_group, sex, migration_rate, net_migration, ...].
        method: Averaging method.  Currently only "simple_average" is
                supported (equal weight to each period).

    Returns:
        DataFrame with columns [county_fips, age_group, sex, migration_rate,
        net_migration, n_periods].
    """
    if not period_rates:
        return pd.DataFrame(
            columns=[
                "county_fips",
                "age_group",
                "sex",
                "migration_rate",
                "net_migration",
                "n_periods",
            ]
        )

    logger.info(f"Averaging rates across {len(period_rates)} periods, method={method}")

    # Concatenate all periods
    all_dfs = []
    for (start, end), df in period_rates.items():
        period_df = df[
            ["county_fips", "age_group", "sex", "migration_rate", "net_migration"]
        ].copy()
        period_df["period"] = f"{start}-{end}"
        all_dfs.append(period_df)

    combined = pd.concat(all_dfs, ignore_index=True)

    if method == "simple_average":
        averaged = combined.groupby(["county_fips", "age_group", "sex"], as_index=False).agg(
            migration_rate=("migration_rate", "mean"),
            net_migration=("net_migration", "mean"),
            n_periods=("migration_rate", "count"),
        )
    else:
        raise ValueError(f"Unknown averaging method: {method}")

    logger.info(
        f"Averaged rates: {averaged['county_fips'].nunique()} counties, "
        f"{len(averaged)} total rows, mean rate {averaged['migration_rate'].mean():.4f}"
    )

    return averaged


def _load_survival_rates(config: dict[str, Any]) -> pd.DataFrame:
    """Load survival rates from the configured file.

    Args:
        config: Project configuration dictionary.

    Returns:
        DataFrame with columns [age_group, sex, survival_rate_5yr].
    """
    project_root = Path(__file__).parent.parent.parent.parent

    data_paths = config.get("data_paths", {})
    surv_path = data_paths.get(
        "survival_rates",
        str(
            project_root
            / "data"
            / "processed"
            / "sdc_2024"
            / "survival_rates_sdc_2024_by_age_group.csv"
        ),
    )

    logger.info(f"Loading survival rates from {surv_path}")
    surv = pd.read_csv(surv_path)
    return surv[["age_group", "sex", "survival_rate_5yr"]]


def subtract_gq_from_populations(
    populations: dict[int, pd.DataFrame],
    gq_historical: pd.DataFrame,
) -> dict[int, pd.DataFrame]:
    """Subtract group quarters from population snapshots to get household-only.

    For each time point, merges the GQ estimate by (county_fips, age_group, sex)
    and subtracts: household_pop = max(population - gq_population, 0).

    The GQ historical file has columns: county_fips, year, age_group, sex,
    gq_population. Population snapshots have: county_fips, age_group, sex,
    population.

    Args:
        populations: Dict mapping year to DataFrame with columns
            [county_fips, age_group, sex, population].
        gq_historical: DataFrame with columns
            [county_fips, year, age_group, sex, gq_population].

    Returns:
        Dict mapping year to DataFrame with household-only population.
        Same structure as input, but population values have GQ subtracted.
    """
    result: dict[int, pd.DataFrame] = {}

    for year, pop_df in populations.items():
        gq_year = gq_historical[gq_historical["year"] == year]

        if gq_year.empty:
            logger.warning(
                f"GQ correction: no GQ data for year {year}; "
                f"using total population unchanged"
            )
            result[year] = pop_df.copy()
            continue

        total_before = pop_df["population"].sum()

        # Merge GQ onto population
        merged = pop_df.merge(
            gq_year[["county_fips", "age_group", "sex", "gq_population"]],
            on=["county_fips", "age_group", "sex"],
            how="left",
        )
        merged["gq_population"] = merged["gq_population"].fillna(0.0)

        # Subtract GQ, floor at zero
        merged["population"] = (merged["population"] - merged["gq_population"]).clip(lower=0.0)

        total_after = merged["population"].sum()
        gq_subtracted = total_before - total_after

        logger.info(
            f"GQ correction year {year}: total pop {total_before:,.0f} → "
            f"HH pop {total_after:,.0f} (GQ removed: {gq_subtracted:,.0f})"
        )

        # Drop the gq_population column
        result[year] = merged[["county_fips", "age_group", "sex", "population"]].copy()

    return result


def run_residual_migration_pipeline(
    config: dict[str, Any] | None = None,
) -> dict[str, pd.DataFrame]:
    """Main pipeline function for residual migration rate computation.

    Orchestrates the full computation:
    1. Load population snapshots for all time points
    2. Load survival rates
    3. Compute residual migration for each period with dampening
       (oil-boom dampening + male migration dampening)
    4. PEP recalibration for reservation counties (ADR-045)
    5. Apply college-age smoothing to period-level rates (ADR-049)
       -- smoothing is applied BEFORE combining/saving so that the
       convergence pipeline reads smoothed rates
    6. Average rates across periods
       -- averaged rates inherit smoothing from Step 5;
       no second smoothing pass is applied (avoids double-smoothing)
    7. Save output files

    Args:
        config: Project configuration dictionary.  If None, loads from
                config/projection_config.yaml.

    Returns:
        Dictionary with keys:
            - "all_periods": DataFrame with all period-specific rates
            - "averaged": DataFrame with multi-period averaged rates
    """
    if config is None:
        config = load_projection_config()

    logger.info("=" * 70)
    logger.info("RESIDUAL MIGRATION RATE COMPUTATION PIPELINE")
    logger.info("=" * 70)

    project_root = Path(__file__).parent.parent.parent.parent
    output_dir = project_root / "data" / "processed" / "migration"
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Load configuration for all steps ---
    migration_cfg = config.get("rates", {}).get("migration", {}).get("domestic", {})
    residual_cfg = migration_cfg.get("residual", {})
    dampening_cfg = migration_cfg.get("dampening", {})
    adjustments_cfg = migration_cfg.get("adjustments", {})

    # --- Step 1: Load all population snapshots ---
    logger.info("Step 1: Loading population snapshots")
    populations = assemble_period_populations(config)

    # --- Step 1b: Subtract GQ from population snapshots (ADR-055 Phase 2) ---
    gq_correction_cfg = residual_cfg.get("gq_correction", {})
    gq_correction_enabled = gq_correction_cfg.get("enabled", False)

    if gq_correction_enabled:
        logger.info("Step 1b: Subtracting GQ from population snapshots (ADR-055 Phase 2)")
        gq_path = gq_correction_cfg.get(
            "historical_gq_path",
            "data/processed/gq_county_age_sex_historical.parquet",
        )
        gq_full_path = project_root / gq_path
        if not gq_full_path.exists():
            raise FileNotFoundError(
                f"GQ historical data not found: {gq_full_path}. "
                f"Run scripts/data/fetch_census_gq_data.py first."
            )
        gq_historical = pd.read_parquet(gq_full_path)
        logger.info(
            f"Loaded historical GQ data: {len(gq_historical)} rows, "
            f"years: {sorted(gq_historical['year'].unique())}"
        )
        populations = subtract_gq_from_populations(populations, gq_historical)
    else:
        logger.info("Step 1b: GQ correction disabled; using total population")

    # --- Step 2: Load survival rates ---
    logger.info("Step 2: Loading survival rates")
    survival_rates = _load_survival_rates(config)
    logger.info(f"Loaded {len(survival_rates)} survival rate records")

    # --- Step 3: Get period and adjustment configuration ---

    periods = residual_cfg.get(
        "periods",
        [[2000, 2005], [2005, 2010], [2010, 2015], [2015, 2020], [2020, 2024]],
    )
    averaging_method = residual_cfg.get("averaging", "simple_average")

    # Add boom_periods to dampening config if not already there
    boom_periods = dampening_cfg.get("boom_periods", [[2005, 2010], [2010, 2015]])

    # Male dampening config
    male_damp_cfg = adjustments_cfg.get("male_dampening", {})
    male_damp_enabled = male_damp_cfg.get("enabled", True)
    male_damp_factor = male_damp_cfg.get("factor", 0.80)
    male_boom_periods = male_damp_cfg.get("boom_periods", [[2005, 2010], [2010, 2015]])

    # College age adjustment config
    college_cfg = adjustments_cfg.get("college_age", {})
    college_enabled = college_cfg.get("enabled", True)
    college_method = college_cfg.get("method", "smooth")
    college_counties = college_cfg.get("counties", ["38035", "38017", "38101", "38015"])
    college_age_groups = college_cfg.get("age_groups", ["15-19", "20-24"])
    college_blend = college_cfg.get("blend_factor", 0.5)

    # --- Step 3: Compute residual migration for each period ---
    logger.info("Step 3: Computing residual migration for each period")
    period_results: dict[tuple[int, int], pd.DataFrame] = {}

    for period_spec in periods:
        start_year, end_year = period_spec[0], period_spec[1]
        period_key = (start_year, end_year)

        if start_year not in populations or end_year not in populations:
            logger.warning(
                f"Missing population data for period {start_year}-{end_year}; "
                f"available years: {sorted(populations.keys())}"
            )
            continue

        rates = compute_residual_migration_rates(
            pop_start=populations[start_year],
            pop_end=populations[end_year],
            survival_rates=survival_rates,
            period=period_key,
        )

        # Apply period dampening (oil counties)
        dampening_with_boom = {**dampening_cfg, "boom_periods": boom_periods}
        rates = apply_period_dampening(rates, period_key, dampening_with_boom)

        # Apply male migration dampening
        if male_damp_enabled:
            rates = apply_male_migration_dampening(
                rates,
                period=period_key,
                boom_periods=male_boom_periods,
                male_dampening_factor=male_damp_factor,
            )

        period_results[period_key] = rates
        logger.info(
            f"Period {start_year}-{end_year}: "
            f"{rates['county_fips'].nunique()} counties, "
            f"mean rate {rates['migration_rate'].mean():.4f}"
        )

    # --- Step 4: PEP recalibration for reservation counties (ADR-045) ---
    pep_recal_cfg = residual_cfg.get("pep_recalibration", {})
    pep_recal_metadata: dict[str, dict[str, dict[str, Any]]] = {}

    if pep_recal_cfg.get("enabled", False):
        logger.info("Step 4: Applying PEP recalibration for reservation counties")
        recal_counties = pep_recal_cfg.get("counties", [])
        pep_path = pep_recal_cfg.get(
            "pep_data_path",
            "data/processed/pep_county_components_2000_2025.parquet",
        )
        near_zero_thresh = pep_recal_cfg.get("near_zero_threshold", 10.0)

        pep_data = _load_pep_for_recalibration(project_root / pep_path)
        logger.info(
            f"Loaded PEP data for recalibration: {len(pep_data)} rows, "
            f"target counties: {recal_counties}"
        )

        for period_key, rates in period_results.items():
            recalibrated, period_meta = apply_pep_recalibration(
                period_rates=rates,
                period=period_key,
                pep_data=pep_data,
                counties=recal_counties,
                near_zero_threshold=near_zero_thresh,
            )
            period_results[period_key] = recalibrated

            # Collect metadata per county
            period_label = f"{period_key[0]}-{period_key[1]}"
            for county, meta in period_meta.items():
                pep_recal_metadata.setdefault(county, {})[period_label] = meta

    # --- Step 5: Apply college-age smoothing to period-level rates (ADR-049) ---
    # ADR-049: Smoothing must happen BEFORE period rates are combined and saved
    # so that the convergence pipeline (which reads residual_migration_rates.parquet)
    # operates on smoothed rates rather than raw unsmoothed spikes.
    if college_enabled:
        logger.info("Step 5: Applying college-age smoothing to period-level rates (ADR-049)")
        for period_key in period_results:
            period_results[period_key] = apply_college_age_adjustment(
                period_results[period_key],
                college_counties=college_counties,
                method=college_method,
                age_groups=college_age_groups,
                blend_factor=college_blend,
            )

    # --- Step 5b: Combine all periods ---
    logger.info("Step 5b: Combining all period rates")
    all_periods = pd.concat(list(period_results.values()), ignore_index=True)

    # --- Step 6: Average across periods ---
    logger.info("Step 6: Averaging rates across periods")
    averaged = average_period_rates(period_results, method=averaging_method)

    # --- Step 6b: College-age smoothing on averaged rates SKIPPED (ADR-049) ---
    # Period-level smoothing in Step 5 already handles college-age adjustment.
    # Applying it again here would double-smooth, over-correcting college
    # counties (e.g. Cass County drops from +63% to +19% instead of ~+48%).
    if college_enabled:
        logger.info(
            "Step 6b: Skipping college-age smoothing on averaged rates — "
            "already applied at period level in Step 5 (ADR-049)"
        )

    # --- Step 7: Save output files ---
    logger.info("Step 7: Saving output files")

    # All periods
    all_periods_path = output_dir / "residual_migration_rates.parquet"
    all_periods.to_parquet(all_periods_path, index=False)
    logger.info(f"Saved all-period rates to {all_periods_path}")

    # Averaged rates
    averaged_path = output_dir / "residual_migration_rates_averaged.parquet"
    averaged.to_parquet(averaged_path, index=False)
    logger.info(f"Saved averaged rates to {averaged_path}")

    # Metadata
    metadata = {
        "processing_date": datetime.now(UTC).isoformat(),
        "pipeline": "residual_migration",
        "periods": [list(p) for p in period_results],
        "n_counties": int(all_periods["county_fips"].nunique()),
        "n_age_groups": int(all_periods["age_group"].nunique()),
        "averaging_method": averaging_method,
        "dampening": {
            "enabled": dampening_cfg.get("enabled", True),
            "factor": dampening_cfg.get("factor", 0.60),
            "oil_counties": dampening_cfg.get("counties", OIL_COUNTIES),
            "boom_periods": boom_periods,
        },
        "male_dampening": {
            "enabled": male_damp_enabled,
            "factor": male_damp_factor,
            "boom_periods": male_boom_periods,
        },
        "college_age_adjustment": {
            "enabled": college_enabled,
            "method": college_method,
            "counties": college_counties,
            "age_groups": college_age_groups,
            "blend_factor": college_blend,
        },
        "pep_recalibration": {
            "enabled": pep_recal_cfg.get("enabled", False),
            "counties": pep_recal_cfg.get("counties", []),
            "near_zero_threshold": pep_recal_cfg.get("near_zero_threshold", 10),
            "per_county_period": pep_recal_metadata,
        },
        "gq_correction": {
            "enabled": gq_correction_enabled,
            "historical_gq_path": gq_correction_cfg.get(
                "historical_gq_path", ""
            ),
            "pre_2020_method": gq_correction_cfg.get(
                "pre_2020_method", "backward_constant"
            ),
        },
        "survival_source": residual_cfg.get("survival_source", "CDC_ND_2020"),
        "summary": {
            "total_all_period_rows": len(all_periods),
            "total_averaged_rows": len(averaged),
            "mean_migration_rate": float(averaged["migration_rate"].mean()),
            "median_migration_rate": float(averaged["migration_rate"].median()),
            "state_total_net_migration_averaged": float(averaged["net_migration"].sum()),
        },
    }

    metadata_path = output_dir / "residual_migration_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved metadata to {metadata_path}")

    # --- Summary ---
    logger.info("=" * 70)
    logger.info("RESIDUAL MIGRATION PIPELINE COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Periods computed: {len(period_results)}")
    logger.info(f"Counties: {all_periods['county_fips'].nunique()}")
    logger.info(f"Total all-period rows: {len(all_periods)}")
    logger.info(f"Total averaged rows: {len(averaged)}")
    logger.info(f"Mean averaged migration rate: {averaged['migration_rate'].mean():.4f}")
    logger.info(f"State total averaged net migration: {averaged['net_migration'].sum():+,.0f}")

    # Per-period summary
    for (start, end), df in sorted(period_results.items()):
        total_mig = df["net_migration"].sum()
        mean_rate = df["migration_rate"].mean()
        logger.info(f"  {start}-{end}: net migration {total_mig:+,.0f}, mean rate {mean_rate:+.4f}")

    logger.info("=" * 70)

    return {
        "all_periods": all_periods,
        "averaged": averaged,
    }
