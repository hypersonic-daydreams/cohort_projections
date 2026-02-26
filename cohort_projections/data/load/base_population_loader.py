"""
Base population loader for cohort projections.

Loads and transforms base population data into the format required by
the cohort-component projection engine. Creates age x sex x race cohort
matrices from raw Census data and pre-computed distributions.

ADR-055: Group Quarters Population Separation
When group_quarters.enabled is True in config, this module separates
group quarters (GQ) population from total population before projection.
The engine then projects only household population. GQ is added back
as a constant after projection (see the pipeline module).

The projection engine expects DataFrames with columns:
    [year, age, sex, race, population]

Where:
    - year: Base year (e.g., 2025)
    - age: Single year of age (0-90)
    - sex: "Male" or "Female"
    - race: One of the 6 standard race/ethnicity categories
    - population: Population count for that cohort
"""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cohort_projections.utils import (
    ConfigLoader,
    get_logger_from_config,
)
from cohort_projections.utils.demographic_utils import sprague_graduate

logger = get_logger_from_config(__name__)


# Mapping from raw data race codes to standard projection categories
RACE_CODE_MAP = {
    "white_nonhispanic": "White alone, Non-Hispanic",
    "black_nonhispanic": "Black alone, Non-Hispanic",
    "aian_nonhispanic": "AIAN alone, Non-Hispanic",
    "asian_nonhispanic": "Asian/PI alone, Non-Hispanic",
    "nhpi_nonhispanic": "Asian/PI alone, Non-Hispanic",  # Combine NHPI with Asian
    "multiracial_nonhispanic": "Two or more races, Non-Hispanic",
    "other_nonhispanic": "Two or more races, Non-Hispanic",  # Map other to multiracial
    "hispanic": "Hispanic (any race)",
}

# Age group to single-year age mapping
AGE_GROUP_RANGES = {
    "0-4": list(range(5)),
    "5-9": list(range(5, 10)),
    "10-14": list(range(10, 15)),
    "15-19": list(range(15, 20)),
    "20-24": list(range(20, 25)),
    "25-29": list(range(25, 30)),
    "30-34": list(range(30, 35)),
    "35-39": list(range(35, 40)),
    "40-44": list(range(40, 45)),
    "45-49": list(range(45, 50)),
    "50-54": list(range(50, 55)),
    "55-59": list(range(55, 60)),
    "60-64": list(range(60, 65)),
    "65-69": list(range(65, 70)),
    "70-74": list(range(70, 75)),
    "75-79": list(range(75, 80)),
    "80-84": list(range(80, 85)),
    "85+": list(range(85, 91)),  # 85-90 (90 is the max age group)
}


def _load_single_year_distribution(
    distribution_path: Path,
    config: dict[str, Any],
) -> pd.DataFrame:
    """
    Load single-year-of-age distribution from SC-EST-derived CSV (ADR-048).

    The single-year file has columns [age, sex, race_ethnicity, estimated_count,
    proportion] with 1,092 rows (91 ages x 2 sexes x 6 races). No age-group
    expansion is needed -- ages are already single years 0-90.

    Args:
        distribution_path: Path to single-year distribution CSV
        config: Configuration dictionary

    Returns:
        DataFrame with columns [age, sex, race, proportion]
    """
    raw_dist = pd.read_csv(distribution_path)
    logger.info(
        f"Loaded single-year distribution: {len(raw_dist)} records "
        f"from {distribution_path}"
    )

    # Map race codes and capitalize sex
    raw_dist["race"] = raw_dist["race_ethnicity"].map(RACE_CODE_MAP).fillna(
        raw_dist["race_ethnicity"]
    )
    raw_dist["sex"] = raw_dist["sex"].str.title()
    raw_dist["age"] = raw_dist["age"].astype(int)

    # Aggregate any duplicates (e.g., NHPI + Asian mapped to same category)
    distribution = raw_dist.groupby(["age", "sex", "race"], as_index=False).agg(
        {"proportion": "sum"}
    )

    return distribution


def _load_five_year_uniform_distribution(
    distribution_path: Path,
) -> pd.DataFrame:
    """
    Load 5-year age group distribution and expand to single years uniformly.

    This is the legacy approach (pre-ADR-048) that splits each 5-year group
    proportion evenly across single years, creating staircase artifacts.

    Args:
        distribution_path: Path to 5-year group distribution CSV

    Returns:
        DataFrame with columns [age, sex, race, proportion]
    """
    raw_dist = pd.read_csv(distribution_path)
    logger.info(
        f"Loaded 5-year group distribution: {len(raw_dist)} records "
        f"from {distribution_path}"
    )

    expanded_rows: list[dict[str, str | int | float]] = []
    for _, row in raw_dist.iterrows():
        age_group = row["age_group"]
        sex = row["sex"].title()

        race_code = row["race_ethnicity"]
        race = RACE_CODE_MAP.get(race_code, race_code)

        if age_group in AGE_GROUP_RANGES:
            ages = AGE_GROUP_RANGES[age_group]
            proportion_per_year = row["proportion"] / len(ages)

            expanded_rows.extend(
                {
                    "age": age,
                    "sex": sex,
                    "race": race,
                    "proportion": proportion_per_year,
                }
                for age in ages
            )
        else:
            logger.warning(f"Unknown age group: {age_group}")

    expanded_df = pd.DataFrame(expanded_rows)

    # Aggregate duplicate cohorts (e.g., NHPI + Asian -> Asian/PI)
    distribution = expanded_df.groupby(["age", "sex", "race"], as_index=False).agg(
        {"proportion": "sum"}
    )

    return distribution


def load_state_age_sex_race_distribution(
    distribution_path: Path | None = None,
    config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """
    Load state-level age-sex-race distribution.

    This distribution is used to allocate county total populations into
    detailed cohorts when county-specific detailed data is not available.

    Supports two resolution modes controlled by config
    ``base_population.age_resolution``:

    - ``"single_year"`` (default): Uses SC-EST2024 single-year-of-age data
      (ADR-048). No uniform splitting; smooth age profile that eliminates
      staircase artifacts in projections.
    - ``"five_year_uniform"``: Legacy mode using cc-est2024-alldata 5-year
      groups, uniformly split to single years. Retained for backward
      compatibility.

    Args:
        distribution_path: Path to distribution CSV file. If None, the
            appropriate default is selected based on the age_resolution config.
        config: Optional configuration dictionary

    Returns:
        DataFrame with columns:
        - age: Single year of age (0-90)
        - sex: "Male" or "Female"
        - race: Standard race category
        - proportion: Proportion of total population in this cohort

    Example:
        >>> dist = load_state_age_sex_race_distribution()
        >>> dist[dist['age'] == 25].head()
              age     sex                        race  proportion
        ...    25    Male  White alone, Non-Hispanic    0.00523
    """
    logger.info("Loading state-level age-sex-race distribution")

    # Load config to get expected categories and age resolution
    if config is None:
        config_loader = ConfigLoader()
        config = config_loader.get_projection_config()

    base_pop_config = config.get("base_population", {})
    age_resolution = base_pop_config.get("age_resolution", "single_year")

    logger.info(f"Age resolution mode: {age_resolution}")

    # Determine distribution path based on resolution mode
    project_root = Path(__file__).parent.parent.parent.parent

    if distribution_path is None:
        if age_resolution == "single_year":
            # ADR-048: single-year-of-age from SC-EST data
            single_year_path_str = base_pop_config.get(
                "single_year_distribution",
                "data/raw/population/nd_age_sex_race_distribution_single_year.csv",
            )
            distribution_path = project_root / single_year_path_str
        else:
            # Legacy 5-year uniform distribution
            distribution_path = (
                project_root / "data" / "raw" / "population"
                / "nd_age_sex_race_distribution.csv"
            )

    distribution_path = Path(distribution_path)

    if not distribution_path.exists():
        raise FileNotFoundError(f"Distribution file not found: {distribution_path}")

    # Load distribution based on resolution mode
    if age_resolution == "single_year":
        distribution = _load_single_year_distribution(distribution_path, config)
    else:
        distribution = _load_five_year_uniform_distribution(distribution_path)

    # Normalize to ensure proportions sum to 1.0
    total_proportion = distribution["proportion"].sum()
    if abs(total_proportion - 1.0) > 0.01:
        logger.warning(
            f"Distribution proportions sum to {total_proportion:.4f}, normalizing to 1.0"
        )
        distribution["proportion"] = distribution["proportion"] / total_proportion

    demographics = config.get("demographics", {})
    expected_sexes = demographics.get("sex", ["Male", "Female"])
    expected_races = demographics.get("race_ethnicity", {}).get(
        "categories",
        [
            "White alone, Non-Hispanic",
            "Black alone, Non-Hispanic",
            "AIAN alone, Non-Hispanic",
            "Asian/PI alone, Non-Hispanic",
            "Two or more races, Non-Hispanic",
            "Hispanic (any race)",
        ],
    )
    min_age = demographics.get("age_groups", {}).get("min_age", 0)
    max_age = demographics.get("age_groups", {}).get("max_age", 90)

    # Ensure all expected cohorts exist (fill missing with 0)
    all_cohorts = [
        {"age": age, "sex": sex, "race": race}
        for age in range(min_age, max_age + 1)
        for sex in expected_sexes
        for race in expected_races
    ]

    complete_index = pd.DataFrame(all_cohorts)

    distribution = complete_index.merge(
        distribution,
        on=["age", "sex", "race"],
        how="left",
    )
    distribution["proportion"] = distribution["proportion"].fillna(0.0)

    # Re-normalize after adding zero-proportion cohorts
    total_proportion = distribution["proportion"].sum()
    if total_proportion > 0:
        distribution["proportion"] = distribution["proportion"] / total_proportion

    logger.info(
        f"Created distribution with {len(distribution)} cohorts "
        f"(ages {min_age}-{max_age}, {len(expected_sexes)} sexes, "
        f"{len(expected_races)} races, resolution={age_resolution})"
    )

    return distribution


def _build_statewide_single_year_weights(
    state_distribution: pd.DataFrame,
) -> dict[tuple[str, str, str], float]:
    """
    Build a lookup of statewide single-year proportions keyed by
    (age_group, sex, race) for use as interpolation weights when
    expanding county 5-year groups to single years.

    For each 5-year age group, the statewide single-year proportions
    within that group are normalized to sum to 1.0, so they serve as
    within-group weights.

    Args:
        state_distribution: Statewide distribution with columns
            [age, sex, race, proportion]

    Returns:
        Dict mapping (age_group_str, sex, race) tuples to
        per-single-year-age weight. For example:
        {("0-4", "Male", "White alone, Non-Hispanic", 0): 0.22, ...}
        Actually returns a nested structure: keys are
        (age_group, sex, race) -> dict[int_age -> weight]
    """
    # Build reverse mapping: single age -> age group string
    age_to_group: dict[int, str] = {}
    for group_str, ages in AGE_GROUP_RANGES.items():
        for age in ages:
            age_to_group[age] = group_str

    # Index statewide distribution
    weights_by_group: dict[tuple[str, str, str], dict[int, float]] = {}

    for _, row in state_distribution.iterrows():
        age = int(row["age"])
        sex = row["sex"]
        race = row["race"]
        proportion = float(row["proportion"])

        group = age_to_group.get(age)
        if group is None:
            continue

        key = (group, sex, race)
        if key not in weights_by_group:
            weights_by_group[key] = {}
        weights_by_group[key][age] = proportion

    # Normalize within each group so weights sum to 1.0
    for key in weights_by_group:
        group_total = sum(weights_by_group[key].values())
        if group_total > 0:
            weights_by_group[key] = {
                age: w / group_total for age, w in weights_by_group[key].items()
            }
        else:
            # Fallback: uniform if statewide has zero for this group
            n = len(weights_by_group[key])
            weights_by_group[key] = {
                age: 1.0 / n for age in weights_by_group[key]
            }

    return weights_by_group


# Ordered list of 5-year age group labels matching AGE_GROUP_RANGES order
_ORDERED_AGE_GROUPS: list[str] = [
    "0-4", "5-9", "10-14", "15-19", "20-24", "25-29",
    "30-34", "35-39", "40-44", "45-49", "50-54", "55-59",
    "60-64", "65-69", "70-74", "75-79", "80-84", "85+",
]


def _expand_county_with_sprague(
    county_data: pd.DataFrame,
) -> list[dict[str, str | int | float]]:
    """
    Expand county 5-year age group proportions to single-year ages using
    Sprague osculatory interpolation (ADR-048).

    For each sex-race combination, the 18 five-year group proportions are
    passed through the Sprague graduate function to produce smooth single-
    year values. The 85+ terminal group is expanded to ages 85-90 (where
    90 is the open-ended 90+ group), producing 6 single-year values from
    the Sprague output for the last group.

    Sprague interpolation preserves each 5-year group's total proportion
    while producing smooth transitions between groups. Negative values
    (possible at extreme ages for very small populations) are clamped to
    zero and the group total is renormalized.

    Args:
        county_data: DataFrame with columns [age_group, sex, race,
            proportion] for a single county. Must contain all 18 standard
            age groups for each sex-race combination.

    Returns:
        List of dicts with keys [age, sex, race, proportion], one per
        single-year-of-age cohort.
    """
    expanded_rows: list[dict[str, str | int | float]] = []

    # Get unique sex-race combinations
    for sex in county_data["sex"].unique():
        for race in county_data["race"].unique():
            mask = (county_data["sex"] == sex) & (county_data["race"] == race)
            subset = county_data[mask]

            if subset.empty:
                continue

            # Build ordered vector of 5-year group proportions
            group_props: dict[str, float] = {}
            for _, row in subset.iterrows():
                group_props[row["age_group"]] = float(row["proportion"])

            # Create ordered array (18 groups: 0-4 through 85+)
            # The first 17 groups have 5 years each; the last (85+)
            # has 6 years (85-90) in the engine's age scheme.
            prop_vector = np.array(
                [group_props.get(g, 0.0) for g in _ORDERED_AGE_GROUPS]
            )

            total_prop = prop_vector.sum()
            if total_prop == 0:
                # All zeros for this sex-race: create zero single-year entries
                for age in range(91):
                    expanded_rows.append({
                        "age": age,
                        "sex": sex.title(),
                        "race": race,
                        "proportion": 0.0,
                    })
                continue

            # Apply Sprague interpolation to all 18 groups
            # This produces 90 single-year values (18 groups x 5 years)
            single_year_values = sprague_graduate(prop_vector, clamp_negatives=True)

            # Map the 90 Sprague outputs to ages 0-89
            # Ages 0-84: first 17 groups x 5 years = 85 values (indices 0-84)
            # Age 85-89: group 18 (85+) gives 5 values from Sprague
            # Age 90: need to account for the open-ended 90+ group
            #
            # The 85+ group in the Sprague output gives 5 values for ages
            # 85-89. Age 90 (open-ended) is not directly produced. We
            # redistribute the 85+ total among ages 85-90 using the Sprague
            # shape for 85-89 plus an exponential tail for 90.
            terminal_total = prop_vector[17]  # 85+ group total
            sprague_85_89 = single_year_values[85:90]  # 5 values from Sprague
            sprague_85_89_sum = sprague_85_89.sum()

            # Compute the 90+ residual using exponential decay
            # Assume the ratio of 90+ to 85-89 follows the pattern from
            # the statewide terminal age distribution (survival ~0.7/year)
            survival_factor = 0.7
            # Weight for 90+: geometric tail s^5 / (1-s) relative to s^0..s^4
            tail_weight = survival_factor ** 5 / (1.0 - survival_factor)
            base_weights = np.array([survival_factor ** i for i in range(5)])
            total_weight = base_weights.sum() + tail_weight

            # Fraction of 85+ that should go to 90+
            frac_90_plus = tail_weight / total_weight

            if sprague_85_89_sum > 0 and terminal_total > 0:
                # Use Sprague shape for 85-89, reserve fraction for 90+
                age_90_prop = terminal_total * frac_90_plus
                ages_85_89_prop = terminal_total - age_90_prop

                # Scale Sprague 85-89 values to match the 85-89 portion
                scale = ages_85_89_prop / sprague_85_89_sum
                adjusted_85_89 = sprague_85_89 * scale
            else:
                # Fallback: uniform across 85-90
                adjusted_85_89 = np.full(5, terminal_total / 6.0)
                age_90_prop = terminal_total / 6.0

            # Build output: ages 0-84 from Sprague, 85-89 adjusted, 90 from tail
            for age in range(85):
                expanded_rows.append({
                    "age": age,
                    "sex": sex.title(),
                    "race": race,
                    "proportion": float(single_year_values[age]),
                })

            for i, age in enumerate(range(85, 90)):
                expanded_rows.append({
                    "age": age,
                    "sex": sex.title(),
                    "race": race,
                    "proportion": float(adjusted_85_89[i]),
                })

            expanded_rows.append({
                "age": 90,
                "sex": sex.title(),
                "race": race,
                "proportion": float(age_90_prop),
            })

    return expanded_rows


def load_county_age_sex_race_distribution(
    fips: str,
    county_distributions_df: pd.DataFrame | None = None,
    config: dict[str, Any] | None = None,
    state_distribution: pd.DataFrame | None = None,
) -> pd.DataFrame | None:
    """
    Load county-specific age-sex-race distribution (ADR-047, ADR-048).

    Reads the county-specific distribution from the pre-built Parquet file
    and expands 5-year age groups to single-year ages, matching the format
    returned by ``load_state_age_sex_race_distribution()``.

    Three interpolation modes are available (controlled by config):

    - ``"sprague"`` (ADR-048 default): Sprague osculatory interpolation,
      the UN Population Division / Census Bureau standard. Produces the
      smoothest results with guaranteed group-total preservation.
    - ``"statewide_weights"``: Uses the statewide SC-EST single-year
      pattern as within-group weights. Produces smooth results that
      mirror the state-level age profile within each group.
    - ``"five_year_uniform"`` (legacy): Divides each 5-year group evenly
      across single years, creating staircase artifacts.

    The interpolation method is selected by
    ``base_population.county_race_interpolation`` when ``age_resolution``
    is ``"single_year"``. When ``age_resolution`` is
    ``"five_year_uniform"``, uniform splitting is always used regardless
    of the interpolation setting.

    Args:
        fips: 5-digit county FIPS code (e.g., "38017" for Cass County)
        county_distributions_df: Optional pre-loaded DataFrame of all county
            distributions (for efficiency when loading multiple counties).
            If None, the file is read from disk.
        config: Optional configuration dictionary
        state_distribution: Optional pre-loaded statewide distribution (for
            statewide_weights interpolation). If None and needed, it is
            loaded from disk.

    Returns:
        DataFrame with columns [age, sex, race, proportion] matching the
        format of ``load_state_age_sex_race_distribution()``, or None if
        county-specific distributions are disabled or unavailable.
    """
    # Load config if not provided
    if config is None:
        config_loader = ConfigLoader()
        config = config_loader.get_projection_config()

    # Check if county distributions are enabled
    base_pop_config = config.get("base_population", {})
    county_dist_config = base_pop_config.get("county_distributions", {})
    if not county_dist_config.get("enabled", False):
        logger.debug("County-specific distributions disabled in config")
        return None

    age_resolution = base_pop_config.get("age_resolution", "single_year")
    county_interp = base_pop_config.get("county_race_interpolation", "sprague")

    fips = str(fips).zfill(5)

    # Load from pre-loaded DataFrame or from disk
    if county_distributions_df is None:
        dist_path_str = county_dist_config.get(
            "path", "data/processed/county_age_sex_race_distributions.parquet"
        )
        project_root = Path(__file__).parent.parent.parent.parent
        dist_path = project_root / dist_path_str

        if not dist_path.exists():
            logger.warning(
                f"County distribution file not found: {dist_path}. "
                "Falling back to statewide distribution."
            )
            return None

        logger.debug(f"Loading county distributions from: {dist_path}")
        county_distributions_df = pd.read_parquet(dist_path)

    # Filter to this county
    county_data = county_distributions_df[county_distributions_df["fips"] == fips]

    if county_data.empty:
        logger.warning(
            f"No county-specific distribution for FIPS {fips}. "
            "Falling back to statewide distribution."
        )
        return None

    logger.debug(f"Found {len(county_data)} distribution rows for FIPS {fips}")

    # Map race codes before expansion so Sprague sees standard names
    county_data = county_data.copy()
    county_data["race"] = county_data["race"].map(
        lambda r: RACE_CODE_MAP.get(r, r)
    )

    # Determine expansion method
    use_sprague = (
        age_resolution == "single_year" and county_interp == "sprague"
    )
    use_statewide_weights = (
        age_resolution == "single_year" and county_interp == "statewide_weights"
    )

    if use_sprague:
        # ADR-048: Sprague osculatory interpolation
        logger.debug(f"Expanding county {fips} with Sprague interpolation")
        expanded_rows = _expand_county_with_sprague(county_data)
        expanded_df = pd.DataFrame(expanded_rows)
    elif use_statewide_weights:
        # ADR-048 fallback: statewide single-year weights
        logger.debug(f"Expanding county {fips} with statewide weights")
        sy_weights: dict[tuple[str, str, str], dict[int, float]] | None = None
        if state_distribution is not None:
            sy_weights = _build_statewide_single_year_weights(state_distribution)
        else:
            loaded_state = load_state_age_sex_race_distribution(config=config)
            sy_weights = _build_statewide_single_year_weights(loaded_state)

        expanded_rows_list: list[dict[str, str | int | float]] = []
        for _, row in county_data.iterrows():
            age_group = row["age_group"]
            sex = row["sex"].title()
            race = row["race"]

            if age_group in AGE_GROUP_RANGES:
                ages = AGE_GROUP_RANGES[age_group]
                weight_key = (age_group, sex, race)
                group_weights = sy_weights.get(weight_key) if sy_weights else None

                if group_weights is not None and sum(group_weights.values()) > 0:
                    expanded_rows_list.extend(
                        {
                            "age": age,
                            "sex": sex,
                            "race": race,
                            "proportion": row["proportion"] * group_weights.get(age, 0.0),
                        }
                        for age in ages
                    )
                else:
                    proportion_per_year = row["proportion"] / len(ages)
                    expanded_rows_list.extend(
                        {
                            "age": age,
                            "sex": sex,
                            "race": race,
                            "proportion": proportion_per_year,
                        }
                        for age in ages
                    )
            else:
                logger.warning(f"Unknown age group: {age_group}")

        expanded_df = pd.DataFrame(expanded_rows_list)
    else:
        # Legacy: uniform distribution across single years
        logger.debug(f"Expanding county {fips} with uniform splitting")
        expanded_rows_list = []
        for _, row in county_data.iterrows():
            age_group = row["age_group"]
            sex = row["sex"].title()
            race = row["race"]

            if age_group in AGE_GROUP_RANGES:
                ages = AGE_GROUP_RANGES[age_group]
                proportion_per_year = row["proportion"] / len(ages)
                expanded_rows_list.extend(
                    {
                        "age": age,
                        "sex": sex,
                        "race": race,
                        "proportion": proportion_per_year,
                    }
                    for age in ages
                )
            else:
                logger.warning(f"Unknown age group: {age_group}")

        expanded_df = pd.DataFrame(expanded_rows_list)

    # Aggregate duplicate cohorts (e.g., NHPI + Asian -> Asian/PI)
    distribution = expanded_df.groupby(["age", "sex", "race"], as_index=False).agg(
        {"proportion": "sum"}
    )

    # Normalize to ensure proportions sum to 1.0
    total_proportion = distribution["proportion"].sum()
    if abs(total_proportion - 1.0) > 0.01:
        logger.warning(
            f"County {fips} distribution proportions sum to "
            f"{total_proportion:.4f}, normalizing to 1.0"
        )
        distribution["proportion"] = distribution["proportion"] / total_proportion

    # Ensure all expected cohorts exist (fill missing with 0)
    demographics = config.get("demographics", {})
    expected_sexes = demographics.get("sex", ["Male", "Female"])
    expected_races = demographics.get("race_ethnicity", {}).get(
        "categories",
        [
            "White alone, Non-Hispanic",
            "Black alone, Non-Hispanic",
            "AIAN alone, Non-Hispanic",
            "Asian/PI alone, Non-Hispanic",
            "Two or more races, Non-Hispanic",
            "Hispanic (any race)",
        ],
    )
    min_age = demographics.get("age_groups", {}).get("min_age", 0)
    max_age = demographics.get("age_groups", {}).get("max_age", 90)

    all_cohorts = [
        {"age": age, "sex": sex, "race": race}
        for age in range(min_age, max_age + 1)
        for sex in expected_sexes
        for race in expected_races
    ]

    complete_index = pd.DataFrame(all_cohorts)

    distribution = complete_index.merge(
        distribution,
        on=["age", "sex", "race"],
        how="left",
    )
    distribution["proportion"] = distribution["proportion"].fillna(0.0)

    # Re-normalize after adding zero-proportion cohorts
    total_proportion = distribution["proportion"].sum()
    if total_proportion > 0:
        distribution["proportion"] = distribution["proportion"] / total_proportion

    logger.info(
        f"Loaded county-specific distribution for FIPS {fips}: "
        f"{len(distribution)} cohorts (resolution={age_resolution})"
    )

    return distribution


def load_county_distributions_file(
    config: dict[str, Any] | None = None,
) -> pd.DataFrame | None:
    """
    Load the county distributions Parquet file once for batch operations.

    Returns the raw DataFrame from the Parquet file, or None if the file
    does not exist or county distributions are disabled.

    Args:
        config: Optional configuration dictionary

    Returns:
        Raw county distributions DataFrame, or None
    """
    if config is None:
        config_loader = ConfigLoader()
        config = config_loader.get_projection_config()

    base_pop_config = config.get("base_population", {})
    county_dist_config = base_pop_config.get("county_distributions", {})
    if not county_dist_config.get("enabled", False):
        return None

    dist_path_str = county_dist_config.get(
        "path", "data/processed/county_age_sex_race_distributions.parquet"
    )
    project_root = Path(__file__).parent.parent.parent.parent
    dist_path = project_root / dist_path_str

    if not dist_path.exists():
        logger.warning(f"County distribution file not found: {dist_path}")
        return None

    logger.info(f"Loading county distributions from: {dist_path}")
    return pd.read_parquet(dist_path)


def load_county_populations(
    population_path: Path | None = None,
) -> pd.DataFrame:
    """
    Load county-level total population data.

    Args:
        population_path: Path to county population CSV file
                        (default: data/raw/population/nd_county_population.csv)

    Returns:
        DataFrame with columns:
        - county_fips: 5-digit county FIPS code
        - county_name: County name
        - population: Total population

    Example:
        >>> pops = load_county_populations()
        >>> pops[pops['county_name'] == 'Cass'].head()
           county_fips county_name  population
        ...      38017        Cass      200945
    """
    logger.info("Loading county population data")

    # Set default path
    if population_path is None:
        project_root = Path(__file__).parent.parent.parent.parent
        population_path = project_root / "data" / "raw" / "population" / "nd_county_population.csv"

    population_path = Path(population_path)

    if not population_path.exists():
        raise FileNotFoundError(f"County population file not found: {population_path}")

    # Load data
    raw_data = pd.read_csv(population_path, dtype={"county_fips": str})

    # Ensure FIPS is 5 digits
    raw_data["county_fips"] = raw_data["county_fips"].str.zfill(5)

    # Get the most recent population column
    # The file has population_2025 as the most recent (Vintage 2025)
    pop_col = "population_2025"
    if pop_col not in raw_data.columns:
        # Fall back to any column containing 'population' or just 'pop'
        pop_cols = [c for c in raw_data.columns if "pop" in c.lower()]
        if pop_cols:
            pop_col = pop_cols[0]
        else:
            raise ValueError("No population column found in county population file")

    # Create standardized output
    result = pd.DataFrame(
        {
            "county_fips": raw_data["county_fips"],
            "county_name": raw_data["county_name"],
            "population": raw_data[pop_col],
        }
    )

    logger.info(f"Loaded population data for {len(result)} counties")
    logger.info(f"Total state population: {result['population'].sum():,.0f}")

    return result


# ---------------------------------------------------------------------------
# ADR-055: Group Quarters Population Separation
# ---------------------------------------------------------------------------

# Module-level cache for GQ data (loaded once, reused across counties)
_gq_data_cache: pd.DataFrame | None = None

# Module-level storage for per-county GQ populations (age x sex x race)
# Used by the pipeline to add GQ back after projection.
# Maps county_fips -> DataFrame[year, age, sex, race, gq_population]
_county_gq_populations: dict[str, pd.DataFrame] = {}


def get_county_gq_population(fips: str) -> pd.DataFrame | None:
    """
    Retrieve the stored GQ population for a county (ADR-055).

    This is called by the pipeline after projection to add GQ back
    as a constant.

    Args:
        fips: 5-digit county FIPS code

    Returns:
        DataFrame with columns [year, age, sex, race, gq_population],
        or None if GQ separation was not performed for this county.
    """
    fips = str(fips).zfill(5)
    return _county_gq_populations.get(fips)


def get_all_county_gq_populations() -> dict[str, pd.DataFrame]:
    """
    Retrieve all stored county GQ populations (ADR-055).

    Returns:
        Dict mapping county_fips -> GQ population DataFrame.
    """
    return _county_gq_populations.copy()


def clear_gq_cache() -> None:
    """Clear the GQ data cache and stored county GQ populations."""
    global _gq_data_cache
    _gq_data_cache = None
    _county_gq_populations.clear()


def _load_gq_data(config: dict[str, Any]) -> pd.DataFrame | None:
    """
    Load the GQ data parquet file (ADR-055).

    Returns DataFrame with columns [county_fips, age_group, sex, gq_population]
    or None if GQ separation is disabled or the file is not found.
    """
    global _gq_data_cache

    if _gq_data_cache is not None:
        return _gq_data_cache

    gq_config = config.get("base_population", {}).get("group_quarters", {})
    if not gq_config.get("enabled", False):
        return None

    gq_path_str = gq_config.get(
        "gq_data_path", "data/processed/gq_county_age_sex_2025.parquet"
    )
    project_root = Path(__file__).parent.parent.parent.parent
    gq_path = project_root / gq_path_str

    if not gq_path.exists():
        logger.warning(
            f"ADR-055: GQ data file not found: {gq_path}. "
            "Skipping GQ separation. Run scripts/data/fetch_census_gq_data.py first."
        )
        return None

    logger.info(f"ADR-055: Loading GQ data from {gq_path}")
    _gq_data_cache = pd.read_parquet(gq_path)
    logger.info(
        f"ADR-055: Loaded GQ data for "
        f"{_gq_data_cache['county_fips'].nunique()} counties"
    )
    return _gq_data_cache


def _expand_gq_to_single_year_ages(
    gq_county: pd.DataFrame,
    max_age: int = 90,
) -> pd.DataFrame:
    """
    Expand GQ data from 5-year age groups to single-year ages (ADR-055).

    Uses uniform distribution within each 5-year age group (simpler than
    Sprague for GQ since the population is small and already approximate).

    Args:
        gq_county: DataFrame with columns [age_group, sex, gq_population]
                   for a single county.
        max_age: Maximum single-year age (default 90).

    Returns:
        DataFrame with columns [age, sex, gq_population] with single-year ages.
    """
    expanded_rows: list[dict[str, int | str | float]] = []

    for _, row in gq_county.iterrows():
        age_group = row["age_group"]
        sex = row["sex"]
        gq_pop = float(row["gq_population"])

        if age_group not in AGE_GROUP_RANGES:
            logger.warning(f"ADR-055: Unknown age group in GQ data: {age_group}")
            continue

        ages = AGE_GROUP_RANGES[age_group]
        pop_per_age = gq_pop / len(ages) if len(ages) > 0 else 0.0

        for age in ages:
            expanded_rows.append({
                "age": age,
                "sex": sex,
                "gq_population": pop_per_age,
            })

    return pd.DataFrame(expanded_rows)


def _distribute_gq_across_races(
    gq_single_year: pd.DataFrame,
    base_pop: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    """
    Distribute GQ population across race categories (ADR-055).

    Uses county-level race proportions from the base population to distribute
    GQ across races, since race-specific GQ data is not available at the
    county level.

    Args:
        gq_single_year: DataFrame with [age, sex, gq_population] (no race yet)
        base_pop: Full base population DataFrame with [year, age, sex, race, population]
        config: Configuration dictionary

    Returns:
        DataFrame with columns [age, sex, race, gq_population]
    """
    demographics = config.get("demographics", {})
    expected_races = demographics.get("race_ethnicity", {}).get(
        "categories",
        [
            "White alone, Non-Hispanic",
            "Black alone, Non-Hispanic",
            "AIAN alone, Non-Hispanic",
            "Asian/PI alone, Non-Hispanic",
            "Two or more races, Non-Hispanic",
            "Hispanic (any race)",
        ],
    )

    # Compute county-wide race shares from the base population
    race_totals = base_pop.groupby("race")["population"].sum()
    county_total = race_totals.sum()

    if county_total == 0:
        # No population; distribute uniformly across races
        race_shares = {race: 1.0 / len(expected_races) for race in expected_races}
    else:
        race_shares = {}
        for race in expected_races:
            race_shares[race] = race_totals.get(race, 0.0) / county_total

    # Distribute each age-sex GQ cell across races
    rows: list[dict[str, int | str | float]] = []
    for _, row in gq_single_year.iterrows():
        age = int(row["age"])
        sex = row["sex"]
        gq_pop = float(row["gq_population"])

        for race in expected_races:
            rows.append({
                "age": age,
                "sex": sex,
                "race": race,
                "gq_population": gq_pop * race_shares[race],
            })

    return pd.DataFrame(rows)


def _separate_gq_from_base_population(
    fips: str,
    base_pop: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Separate group quarters from total base population (ADR-055).

    Loads the GQ data for the county, expands to single-year ages,
    distributes across races, and subtracts from the total base population
    to produce household-only population.

    GQ is capped at total population in each cell to prevent negative
    household populations.

    Args:
        fips: 5-digit county FIPS code
        base_pop: Total base population DataFrame with
                  columns [year, age, sex, race, population]
        config: Configuration dictionary

    Returns:
        Tuple of (household_pop, gq_pop) where:
        - household_pop: DataFrame with same columns as base_pop, population
          reduced by GQ amounts
        - gq_pop: DataFrame with columns [year, age, sex, race, gq_population]
          representing the GQ component
    """
    fips = str(fips).zfill(5)

    # Load GQ data
    gq_data = _load_gq_data(config)
    if gq_data is None:
        logger.debug(f"ADR-055: No GQ data available for {fips}, returning full population")
        # Return empty GQ DataFrame with correct columns
        gq_empty = base_pop[["year", "age", "sex", "race"]].copy()
        gq_empty["gq_population"] = 0.0
        return base_pop, gq_empty

    # Filter to this county
    county_gq = gq_data[gq_data["county_fips"] == fips].copy()

    if county_gq.empty:
        logger.debug(f"ADR-055: No GQ data for county {fips}")
        gq_empty = base_pop[["year", "age", "sex", "race"]].copy()
        gq_empty["gq_population"] = 0.0
        return base_pop, gq_empty

    county_gq_total = county_gq["gq_population"].sum()
    if county_gq_total == 0:
        logger.debug(f"ADR-055: County {fips} has zero GQ population")
        gq_empty = base_pop[["year", "age", "sex", "race"]].copy()
        gq_empty["gq_population"] = 0.0
        return base_pop, gq_empty

    logger.info(
        f"ADR-055: Separating GQ for county {fips}: "
        f"GQ total = {county_gq_total:,.0f}"
    )

    # Step 1: Expand 5-year age groups to single-year ages
    gq_single_year = _expand_gq_to_single_year_ages(
        county_gq[["age_group", "sex", "gq_population"]]
    )

    # Step 2: Distribute across races using county race proportions
    gq_with_race = _distribute_gq_across_races(gq_single_year, base_pop, config)

    # Step 3: Add year column to match base_pop format
    base_year = base_pop["year"].iloc[0] if not base_pop.empty else 2025
    gq_with_race["year"] = base_year

    # Step 4: Merge GQ with base_pop and subtract
    household_pop = base_pop.copy()
    gq_result = base_pop[["year", "age", "sex", "race"]].copy()
    gq_result["gq_population"] = 0.0

    # Build GQ lookup for fast matching
    gq_lookup = gq_with_race.set_index(["age", "sex", "race"])["gq_population"]

    for idx in household_pop.index:
        age = household_pop.at[idx, "age"]
        sex = household_pop.at[idx, "sex"]
        race = household_pop.at[idx, "race"]
        total_pop = household_pop.at[idx, "population"]

        key = (age, sex, race)
        if key in gq_lookup.index:
            gq_val = float(gq_lookup.loc[key])
            # Aggregate if multiple entries for same key
            if isinstance(gq_val, pd.Series):
                gq_val = gq_val.sum()
        else:
            gq_val = 0.0

        # Cap GQ at total population to prevent negative household pop
        gq_val = min(gq_val, total_pop)

        household_pop.at[idx, "population"] = total_pop - gq_val
        gq_result.at[idx, "gq_population"] = gq_val

    # Store the GQ population for later retrieval by the pipeline
    _county_gq_populations[fips] = gq_result.copy()

    hh_total = household_pop["population"].sum()
    gq_actual = gq_result["gq_population"].sum()
    county_total = hh_total + gq_actual
    logger.info(
        f"ADR-055: County {fips}: Total={county_total:,.0f} -> "
        f"Household={hh_total:,.0f} + GQ={gq_actual:,.0f}"
    )

    return household_pop, gq_result


def load_base_population_for_county(
    fips: str,
    config: dict[str, Any] | None = None,
    distribution: pd.DataFrame | None = None,
    county_populations: pd.DataFrame | None = None,
    county_distributions_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Load base population for a single county.

    Creates the age x sex x race cohort matrix required by the projection
    engine. Uses county-specific distribution (ADR-047) when available,
    falling back to the statewide distribution.

    Args:
        fips: 5-digit county FIPS code (e.g., "38017" for Cass County)
        config: Optional configuration dictionary
        distribution: Optional pre-loaded state distribution (fallback when
                     county-specific data is unavailable)
        county_populations: Optional pre-loaded county populations DataFrame
        county_distributions_df: Optional pre-loaded county distributions
            Parquet DataFrame (for efficiency when loading multiple counties)

    Returns:
        DataFrame with columns [year, age, sex, race, population] suitable
        for the CohortComponentProjection engine

    Raises:
        ValueError: If county FIPS not found in population data

    Example:
        >>> cass_pop = load_base_population_for_county("38017")
        >>> cass_pop.head()
           year  age     sex                        race  population
        0  2025    0    Male  White alone, Non-Hispanic       800.5
        1  2025    0    Male  Black alone, Non-Hispanic        25.3
        ...
        >>> cass_pop['population'].sum()
        200945.0
    """
    logger.info(f"Loading base population for county FIPS: {fips}")

    # Load config if not provided
    if config is None:
        config_loader = ConfigLoader()
        config = config_loader.get_projection_config()

    base_year = config.get("project", {}).get("base_year", 2025)

    # Ensure FIPS is 5 digits
    fips = str(fips).zfill(5)

    # Load statewide distribution (needed as fallback and for single-year weights)
    if distribution is None:
        distribution = load_state_age_sex_race_distribution(config=config)

    # Try to load county-specific distribution (ADR-047)
    county_distribution = load_county_age_sex_race_distribution(
        fips=fips,
        county_distributions_df=county_distributions_df,
        config=config,
        state_distribution=distribution,
    )

    if county_distribution is not None:
        logger.info(f"Using county-specific distribution for FIPS {fips}")
        effective_distribution = county_distribution
    else:
        # Fall back to statewide distribution
        effective_distribution = distribution

    # Load county populations if not provided
    if county_populations is None:
        county_populations = load_county_populations()

    # Find this county's population
    county_row = county_populations[county_populations["county_fips"] == fips]

    if county_row.empty:
        raise ValueError(f"County FIPS {fips} not found in population data")

    total_population = county_row["population"].values[0]
    county_name = county_row["county_name"].values[0]

    logger.info(f"County: {county_name} ({fips}), Total population: {total_population:,.0f}")

    # Apply distribution to total population
    base_pop = effective_distribution.copy()
    base_pop["population"] = base_pop["proportion"] * total_population
    base_pop["year"] = base_year

    # Select and reorder columns for projection engine
    base_pop = base_pop[["year", "age", "sex", "race", "population"]]

    # Sort for consistency
    base_pop = base_pop.sort_values(["age", "sex", "race"]).reset_index(drop=True)

    # Validate
    actual_total = base_pop["population"].sum()
    if abs(actual_total - total_population) > 1.0:
        logger.warning(
            f"Population mismatch after distribution: "
            f"expected {total_population:,.0f}, got {actual_total:,.0f}"
        )

    logger.info(f"Created base population with {len(base_pop)} cohorts, total: {actual_total:,.0f}")

    # ADR-055: Separate group quarters from household population
    gq_config = config.get("base_population", {}).get("group_quarters", {})
    if gq_config.get("enabled", False):
        household_pop, _gq_pop = _separate_gq_from_base_population(
            fips=fips,
            base_pop=base_pop,
            config=config,
        )
        return household_pop

    return base_pop


def load_base_population_for_all_counties(
    config: dict[str, Any] | None = None,
    fips_list: list[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Load base population for all North Dakota counties.

    Returns a dictionary mapping county FIPS codes to their base population
    DataFrames, in the format expected by `run_multi_geography_projections`.

    Args:
        config: Optional configuration dictionary
        fips_list: Optional list of FIPS codes to load (if None, loads all)

    Returns:
        Dictionary mapping county FIPS (str) to base population DataFrame.
        Each DataFrame has columns [year, age, sex, race, population].

    Example:
        >>> all_pops = load_base_population_for_all_counties()
        >>> len(all_pops)
        53
        >>> cass = all_pops["38017"]
        >>> cass['population'].sum()
        200945.0
    """
    logger.info("Loading base population for all counties")

    # Load config if not provided
    if config is None:
        config_loader = ConfigLoader()
        config = config_loader.get_projection_config()

    # Load shared data once for efficiency
    distribution = load_state_age_sex_race_distribution(config=config)
    county_populations = load_county_populations()

    # Load county-specific distributions file once (ADR-047)
    county_distributions_df = load_county_distributions_file(config=config)
    if county_distributions_df is not None:
        logger.info(
            f"Loaded county distributions: {county_distributions_df['fips'].nunique()} counties"
        )
    else:
        logger.info("County-specific distributions not available; using statewide for all")

    # Get list of counties to process
    if fips_list is None:
        fips_list = county_populations["county_fips"].tolist()

    logger.info(f"Loading base population for {len(fips_list)} counties")

    # Load each county
    base_population_by_geography: dict[str, pd.DataFrame] = {}
    failed_counties: list[str] = []

    for fips in fips_list:
        try:
            base_pop = load_base_population_for_county(
                fips=fips,
                config=config,
                distribution=distribution,
                county_populations=county_populations,
                county_distributions_df=county_distributions_df,
            )
            base_population_by_geography[fips] = base_pop

        except Exception as e:
            logger.warning(f"Failed to load county {fips}: {e}")
            failed_counties.append(fips)

    logger.info(
        f"Successfully loaded {len(base_population_by_geography)} counties, "
        f"{len(failed_counties)} failed"
    )

    if failed_counties:
        logger.warning(f"Failed counties: {failed_counties}")

    # Calculate and log total population
    total_pop = sum(df["population"].sum() for df in base_population_by_geography.values())
    logger.info(f"Total state population across all counties: {total_pop:,.0f}")

    return base_population_by_geography


def load_base_population_for_state(
    config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """
    Load base population for North Dakota state total.

    Aggregates all county populations or uses direct state-level data.

    Args:
        config: Optional configuration dictionary

    Returns:
        DataFrame with columns [year, age, sex, race, population]
        representing the state total.

    Example:
        >>> state_pop = load_base_population_for_state()
        >>> state_pop['population'].sum()
        779094.0  # Total ND population
    """
    logger.info("Loading base population for state")

    # Load config if not provided
    if config is None:
        config_loader = ConfigLoader()
        config = config_loader.get_projection_config()

    # Load all county populations and aggregate
    all_counties = load_base_population_for_all_counties(config=config)

    if not all_counties:
        raise ValueError("No county data available to aggregate to state level")

    # Combine all counties
    all_dfs = list(all_counties.values())
    combined = pd.concat(all_dfs, ignore_index=True)

    # Aggregate by cohort
    state_pop = combined.groupby(["year", "age", "sex", "race"], as_index=False).agg(
        {"population": "sum"}
    )

    # Sort for consistency
    state_pop = state_pop.sort_values(["age", "sex", "race"]).reset_index(drop=True)

    logger.info(
        f"Created state base population: {len(state_pop)} cohorts, "
        f"total: {state_pop['population'].sum():,.0f}"
    )

    return state_pop


if __name__ == "__main__":
    """Example usage and testing."""

    logger.info("Base population loader module test")
    logger.info("=" * 70)

    # Test loading state distribution
    logger.info("\n1. Testing state-level distribution loading...")
    try:
        dist = load_state_age_sex_race_distribution()
        logger.info(f"   Loaded distribution with {len(dist)} cohorts")
        logger.info(f"   Proportion sum: {dist['proportion'].sum():.6f}")
        logger.info(f"   Sample:\n{dist.head()}")
    except Exception as e:
        logger.error(f"   Failed: {e}")

    # Test loading county populations
    logger.info("\n2. Testing county population loading...")
    try:
        pops = load_county_populations()
        logger.info(f"   Loaded {len(pops)} counties")
        logger.info(f"   Total population: {pops['population'].sum():,.0f}")
    except Exception as e:
        logger.error(f"   Failed: {e}")

    # Test loading single county
    logger.info("\n3. Testing single county base population...")
    try:
        # Cass County (Fargo)
        cass = load_base_population_for_county("38017")
        logger.info(f"   Cass County cohorts: {len(cass)}")
        logger.info(f"   Cass County population: {cass['population'].sum():,.0f}")
        logger.info(f"   Columns: {list(cass.columns)}")
    except Exception as e:
        logger.error(f"   Failed: {e}")

    # Test loading all counties
    logger.info("\n4. Testing all counties loading...")
    try:
        all_counties = load_base_population_for_all_counties()
        logger.info(f"   Loaded {len(all_counties)} counties")
        total = sum(df["population"].sum() for df in all_counties.values())
        logger.info(f"   Total population: {total:,.0f}")
    except Exception as e:
        logger.error(f"   Failed: {e}")

    logger.info("\n" + "=" * 70)
    logger.info("Base population loader tests complete")
