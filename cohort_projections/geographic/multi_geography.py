"""
Multi-geography projection orchestrator.

Runs cohort-component projections for multiple geographies (state, counties, places)
with support for parallel processing and hierarchical aggregation/validation.
"""

import json
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import pandas as pd

# Progress bar for long-running jobs
try:
    from tqdm import tqdm

    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

    # Fallback: simple counter
    class TqdmFallback:
        def __init__(self, iterable=None, **kwargs):
            self.iterable = iterable
            self.total = kwargs.get("total", len(iterable) if iterable else 0)
            self.desc = kwargs.get("desc", "")
            self.current = 0

        def __iter__(self):
            for item in self.iterable:
                self.current += 1
                if self.current % 10 == 0:
                    print(f"{self.desc}: {self.current}/{self.total}")
                yield item


from cohort_projections.core.cohort_component import CohortComponentProjection
from cohort_projections.geographic.geography_loader import (
    get_geography_name,
    get_place_to_county_mapping,
    load_geography_list,
)
from cohort_projections.utils.config_loader import load_projection_config
from cohort_projections.utils.logger import get_logger_from_config

logger = get_logger_from_config(__name__)


def run_single_geography_projection(
    fips: str,
    level: Literal["state", "county", "place"],
    base_population: pd.DataFrame,
    fertility_rates: pd.DataFrame,
    survival_rates: pd.DataFrame,
    migration_rates: pd.DataFrame,
    config: dict | None = None,
    output_dir: Path | None = None,
    save_results: bool = True,
) -> dict[str, Any]:
    """
    Run projection for a single geography.

    Args:
        fips: FIPS code of geography
        level: Geographic level ('state', 'county', 'place')
        base_population: Base year population for this geography
        fertility_rates: Fertility rates (can be shared or geography-specific)
        survival_rates: Survival rates (can be shared or geography-specific)
        migration_rates: Migration rates (typically geography-specific)
        config: Optional configuration dictionary
        output_dir: Optional output directory (default: data/output/projections/{level})
        save_results: Whether to save results to files

    Returns:
        Dictionary with projection results and metadata:
        {
            'geography': {'fips': str, 'level': str, 'name': str},
            'projection': pd.DataFrame,
            'summary': pd.DataFrame,
            'metadata': dict,
            'processing_time': float
        }

    Raises:
        ValueError: If input data invalid
        Exception: If projection fails

    Example:
        >>> result = run_single_geography_projection(
        ...     fips='38101',
        ...     level='county',
        ...     base_population=cass_pop,
        ...     fertility_rates=nd_fertility,
        ...     survival_rates=nd_survival,
        ...     migration_rates=cass_migration
        ... )
        >>> result['metadata']['summary_statistics']['final_population']
        195000
    """
    start_time = time.time()

    # Load config if not provided
    if config is None:
        config = load_projection_config()

    # Get geography name
    geo_name = get_geography_name(fips, level)

    logger.info(f"Starting projection for {geo_name} (FIPS: {fips})")

    try:
        # Filter base population to this geography if needed
        if "geography_fips" in base_population.columns:
            base_pop = base_population[base_population["geography_fips"] == fips].copy()
            # Drop geography_fips column for projection engine
            base_pop = base_pop.drop(columns=["geography_fips"])
        else:
            base_pop = base_population.copy()

        if base_pop.empty:
            logger.warning(f"No base population data for {geo_name} (FIPS: {fips})")
            return {
                "geography": {"fips": fips, "level": level, "name": geo_name},
                "projection": pd.DataFrame(),
                "summary": pd.DataFrame(),
                "metadata": {"error": "No base population data"},
                "processing_time": time.time() - start_time,
            }

        base_year_pop = base_pop["population"].sum()
        logger.info(f"{geo_name}: Base year population = {base_year_pop:,.0f}")

        # Initialize projection engine
        projection_engine = CohortComponentProjection(
            base_population=base_pop,
            fertility_rates=fertility_rates,
            survival_rates=survival_rates,
            migration_rates=migration_rates,
            config=config,
        )

        # Run projection
        scenario = config.get("scenarios", {}).get("baseline", {}).get("active", True)
        scenario_name = "baseline" if scenario else None

        projection_results = projection_engine.run_projection(scenario=scenario_name)

        # Get summary
        summary = projection_engine.get_projection_summary()

        # Calculate summary statistics
        final_year = projection_results["year"].max()
        final_pop = projection_results[projection_results["year"] == final_year]["population"].sum()
        growth = final_pop - base_year_pop
        growth_rate = (final_pop / base_year_pop - 1.0) if base_year_pop > 0 else 0.0

        # Create metadata
        metadata = {
            "geography": {
                "level": level,
                "fips": fips,
                "name": geo_name,
                "base_population": float(base_year_pop),
            },
            "projection": {
                "base_year": int(config.get("project", {}).get("base_year", 2025)),
                "end_year": int(
                    config.get("project", {}).get("base_year", 2025)
                    + config.get("project", {}).get("projection_horizon", 20)
                ),
                "scenario": scenario_name,
                "processing_date": datetime.now(UTC).isoformat(),
            },
            "summary_statistics": {
                "base_population": float(base_year_pop),
                "final_population": float(final_pop),
                "absolute_growth": float(growth),
                "growth_rate": float(growth_rate),
                "years_projected": int(final_year - projection_results["year"].min()),
            },
            "validation": {
                "negative_populations": int((projection_results["population"] < 0).sum()),
                "all_checks_passed": (projection_results["population"] < 0).sum() == 0,
            },
        }

        processing_time = time.time() - start_time
        metadata["processing_time_seconds"] = round(processing_time, 2)

        logger.info(
            f"{geo_name}: Projection complete. "
            f"Final population: {final_pop:,.0f} ({growth_rate:+.1%}). "
            f"Time: {processing_time:.1f}s"
        )

        # Save results if requested
        if save_results:
            _save_projection_results(
                fips=fips,
                level=level,
                projection=projection_results,
                summary=summary,
                metadata=metadata,
                output_dir=output_dir,
                config=config,
            )

        return {
            "geography": {"fips": fips, "level": level, "name": geo_name},
            "projection": projection_results,
            "summary": summary,
            "metadata": metadata,
            "processing_time": processing_time,
        }

    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(f"Projection failed for {geo_name} (FIPS: {fips}): {str(e)}")

        return {
            "geography": {"fips": fips, "level": level, "name": geo_name},
            "projection": pd.DataFrame(),
            "summary": pd.DataFrame(),
            "metadata": {"error": str(e)},
            "processing_time": processing_time,
        }


def run_multi_geography_projections(
    level: Literal["state", "county", "place"],
    base_population_by_geography: dict[str, pd.DataFrame],
    fertility_rates: pd.DataFrame,
    survival_rates: pd.DataFrame,
    migration_rates_by_geography: dict[str, pd.DataFrame] | None = None,
    config: dict | None = None,
    fips_codes: list[str] | None = None,
    parallel: bool = True,
    max_workers: int | None = None,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """
    Run projections for multiple geographies.

    Args:
        level: Geographic level ('state', 'county', 'place')
        base_population_by_geography: Dict mapping FIPS -> base population DataFrame
        fertility_rates: Fertility rates (shared across geographies)
        survival_rates: Survival rates (shared across geographies)
        migration_rates_by_geography: Optional dict mapping FIPS -> migration rates
                                      If None, uses shared migration_rates for all
        config: Optional configuration dictionary
        fips_codes: Optional list of FIPS codes (if None, loads from config)
        parallel: Whether to use parallel processing
        max_workers: Max parallel workers (default: cpu_count)
        output_dir: Output directory (default: data/output/projections/{level})

    Returns:
        Dictionary with results for all geographies:
        {
            'results': List[dict],  # One per geography
            'summary': pd.DataFrame,  # Summary across all geographies
            'metadata': dict,
            'failed_geographies': List[str]
        }

    Example:
        >>> results = run_multi_geography_projections(
        ...     level='county',
        ...     base_population_by_geography=county_pops,
        ...     fertility_rates=nd_fertility,
        ...     survival_rates=nd_survival,
        ...     migration_rates_by_geography=county_migrations,
        ...     parallel=True
        ... )
        >>> len(results['results'])
        53  # All ND counties
    """
    start_time = time.time()

    # Load config if not provided
    if config is None:
        config = load_projection_config()

    # Load geography list if not provided
    if fips_codes is None:
        fips_codes = load_geography_list(level, config)

    num_geographies = len(fips_codes)
    logger.info("=" * 70)
    logger.info("Starting multi-geography projections")
    logger.info(f"Level: {level}, Geographies: {num_geographies}, Parallel: {parallel}")
    logger.info("=" * 70)

    # Prepare migration rates (shared if not geography-specific)
    if migration_rates_by_geography is None:
        # Migration rates are required - must be provided per geography
        raise ValueError(
            "migration_rates_by_geography is required. "
            "Please provide a dictionary mapping FIPS codes to migration rate DataFrames."
        )

    # Run projections
    results = []
    failed_geographies = []

    if parallel and num_geographies > 1:
        # Parallel execution
        if max_workers is None:
            max_workers = min(multiprocessing.cpu_count(), num_geographies)

        logger.info(f"Running projections in parallel with {max_workers} workers")

        # Note: ProcessPoolExecutor requires picklable arguments
        # We'll use a wrapper function
        def _projection_worker(fips):
            """Worker function for parallel processing."""
            base_pop = base_population_by_geography.get(fips, pd.DataFrame())
            migration_rates = migration_rates_by_geography.get(fips, pd.DataFrame())

            return run_single_geography_projection(
                fips=fips,
                level=level,
                base_population=base_pop,
                fertility_rates=fertility_rates,
                survival_rates=survival_rates,
                migration_rates=migration_rates,
                config=config,
                output_dir=output_dir,
                save_results=True,
            )

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_fips = {
                executor.submit(_projection_worker, fips): fips for fips in fips_codes
            }

            # Collect results with progress bar
            if TQDM_AVAILABLE:
                iterator = tqdm(
                    as_completed(future_to_fips), total=num_geographies, desc=f"Projecting {level}s"
                )
            else:
                iterator = as_completed(future_to_fips)

            for future in iterator:
                fips = future_to_fips[future]
                try:
                    result = future.result()
                    results.append(result)

                    # Check for errors
                    if "error" in result["metadata"]:
                        failed_geographies.append(fips)

                except Exception as e:
                    logger.error(f"Failed to process {fips}: {str(e)}")
                    failed_geographies.append(fips)

    else:
        # Serial execution
        logger.info("Running projections serially")

        iterator = tqdm(fips_codes, desc=f"Projecting {level}s") if TQDM_AVAILABLE else fips_codes

        for fips in iterator:
            base_pop = base_population_by_geography.get(fips, pd.DataFrame())
            migration_rates = migration_rates_by_geography.get(fips, pd.DataFrame())

            try:
                result = run_single_geography_projection(
                    fips=fips,
                    level=level,
                    base_population=base_pop,
                    fertility_rates=fertility_rates,
                    survival_rates=survival_rates,
                    migration_rates=migration_rates,
                    config=config,
                    output_dir=output_dir,
                    save_results=True,
                )

                results.append(result)

                # Check for errors
                if "error" in result["metadata"]:
                    failed_geographies.append(fips)

            except Exception as e:
                logger.error(f"Failed to process {fips}: {str(e)}")
                failed_geographies.append(fips)

    # Create summary DataFrame
    summary_data = []
    for result in results:
        if not result["projection"].empty:
            summary_data.append(
                {
                    "fips": result["geography"]["fips"],
                    "name": result["geography"]["name"],
                    "level": result["geography"]["level"],
                    "base_population": result["metadata"]["summary_statistics"]["base_population"],
                    "final_population": result["metadata"]["summary_statistics"][
                        "final_population"
                    ],
                    "absolute_growth": result["metadata"]["summary_statistics"]["absolute_growth"],
                    "growth_rate": result["metadata"]["summary_statistics"]["growth_rate"],
                    "processing_time": result["processing_time"],
                }
            )

    summary_df = pd.DataFrame(summary_data)

    # Overall metadata
    total_time = time.time() - start_time
    metadata = {
        "level": level,
        "num_geographies": num_geographies,
        "successful": num_geographies - len(failed_geographies),
        "failed": len(failed_geographies),
        "parallel": parallel,
        "max_workers": max_workers if parallel else 1,
        "total_processing_time_seconds": round(total_time, 2),
        "processing_date": datetime.now(UTC).isoformat(),
    }

    logger.info("=" * 70)
    logger.info("Multi-geography projections complete")
    logger.info(
        f"Total: {num_geographies}, Successful: {metadata['successful']}, "
        f"Failed: {metadata['failed']}"
    )
    logger.info(f"Total time: {total_time:.1f}s")
    logger.info("=" * 70)

    # Save summary
    if output_dir is None:
        project_root = Path(__file__).parent.parent.parent
        output_dir = project_root / "data" / "output" / "projections" / f"{level}s"

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_file = output_dir / f"{level}s_summary.csv"
    summary_df.to_csv(summary_file, index=False)
    logger.info(f"Saved summary to {summary_file}")

    metadata_file = output_dir / f"{level}s_metadata.json"
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved metadata to {metadata_file}")

    return {
        "results": results,
        "summary": summary_df,
        "metadata": metadata,
        "failed_geographies": failed_geographies,
    }


def aggregate_to_county(
    place_projections: list[dict[str, Any]], config: dict | None = None
) -> dict[str, pd.DataFrame]:
    """
    Aggregate place-level projections to county level.

    Args:
        place_projections: List of projection result dicts from run_single_geography_projection
        config: Optional configuration dictionary

    Returns:
        Dictionary mapping county FIPS -> aggregated projection DataFrame

    Notes:
        - Places are aggregated to their containing county
        - Sum of places < county total (unincorporated areas not included)
        - Age-sex-race structure preserved

    Example:
        >>> county_aggregated = aggregate_to_county(place_results)
        >>> cass_places = county_aggregated['38017']  # Cass County
        >>> cass_places['population'].sum()  # Sum of Fargo, West Fargo, etc.
    """
    logger.info("Aggregating place projections to county level")

    # Get place-to-county mapping
    mapping = get_place_to_county_mapping()

    # Group projections by county
    county_aggregates = {}

    for place_result in place_projections:
        if place_result["projection"].empty:
            continue

        place_fips = place_result["geography"]["fips"]

        # Get containing county
        county_match = mapping[mapping["place_fips"] == place_fips]
        if county_match.empty:
            logger.warning(f"No county mapping found for place {place_fips}")
            continue

        county_fips = county_match["county_fips"].values[0]

        # Add projection to county aggregate
        projection = place_result["projection"].copy()

        if county_fips not in county_aggregates:
            county_aggregates[county_fips] = projection
        else:
            # Sum with existing county data
            county_aggregates[county_fips] = pd.concat(
                [county_aggregates[county_fips], projection], ignore_index=True
            )

    # Aggregate by cohort (sum populations)
    for county_fips, df in county_aggregates.items():
        aggregated = df.groupby(["year", "age", "sex", "race"], as_index=False)["population"].sum()

        county_aggregates[county_fips] = aggregated

    logger.info(f"Aggregated to {len(county_aggregates)} counties from places")

    return county_aggregates


def aggregate_to_state(
    county_projections: list[dict[str, Any]], config: dict | None = None
) -> pd.DataFrame:
    """
    Aggregate county-level projections to state level.

    Args:
        county_projections: List of projection result dicts from run_single_geography_projection
        config: Optional configuration dictionary

    Returns:
        State-level aggregated projection DataFrame

    Notes:
        - All counties aggregated to state total
        - Should equal independently-run state projection (within rounding)
        - Age-sex-race structure preserved

    Example:
        >>> state_projection = aggregate_to_state(county_results)
        >>> state_projection['population'].sum()  # Total across all cohorts and years
    """
    logger.info("Aggregating county projections to state level")

    # Combine all county projections
    all_counties = []

    for county_result in county_projections:
        if not county_result["projection"].empty:
            all_counties.append(county_result["projection"])

    if not all_counties:
        logger.warning("No county projections to aggregate")
        return pd.DataFrame()

    # Concatenate
    combined = pd.concat(all_counties, ignore_index=True)

    # Aggregate by year-age-sex-race (sum populations)
    state_projection = combined.groupby(["year", "age", "sex", "race"], as_index=False)[
        "population"
    ].sum()

    total_pop = state_projection[state_projection["year"] == state_projection["year"].min()][
        "population"
    ].sum()

    logger.info(
        f"Aggregated {len(county_projections)} counties to state level. "
        f"Base year population: {total_pop:,.0f}"
    )

    return state_projection


def validate_aggregation(
    component_projections: list[dict[str, Any]],
    aggregated_projection: pd.DataFrame,
    component_level: Literal["place", "county"],
    aggregate_level: Literal["county", "state"],
    tolerance: float = 0.01,
    config: dict | None = None,
) -> dict[str, Any]:
    """
    Validate that aggregated projection matches sum of components.

    Args:
        component_projections: List of component projection results
        aggregated_projection: Aggregated projection DataFrame
        component_level: Level of components ('place' or 'county')
        aggregate_level: Level of aggregate ('county' or 'state')
        tolerance: Acceptable difference as fraction (default: 0.01 = 1%)
        config: Optional configuration dictionary

    Returns:
        Validation result dictionary:
        {
            'valid': bool,
            'errors': List[str],
            'warnings': List[str],
            'component_total': float,
            'aggregate_total': float,
            'difference': float,
            'percent_difference': float
        }

    Notes:
        - Compares sum of component populations to aggregate
        - Allows small tolerance for rounding errors
        - Checks all years separately

    Example:
        >>> validation = validate_aggregation(
        ...     component_projections=place_results,
        ...     aggregated_projection=county_aggregate,
        ...     component_level='place',
        ...     aggregate_level='county',
        ...     tolerance=0.01
        ... )
        >>> validation['valid']
        True
    """
    logger.info(f"Validating aggregation: {component_level}s -> {aggregate_level}")

    validation_result = {"valid": True, "errors": [], "warnings": [], "by_year": []}

    # Combine all component projections
    component_dfs = [r["projection"] for r in component_projections if not r["projection"].empty]

    if not component_dfs:
        validation_result["errors"].append("No component projections to validate")
        validation_result["valid"] = False
        return validation_result

    combined_components = pd.concat(component_dfs, ignore_index=True)

    # Aggregate components
    component_aggregate = combined_components.groupby(
        ["year", "age", "sex", "race"], as_index=False
    )["population"].sum()

    # Compare year by year
    years = sorted(component_aggregate["year"].unique())

    for year in years:
        component_year = component_aggregate[component_aggregate["year"] == year]
        aggregate_year = aggregated_projection[aggregated_projection["year"] == year]

        component_total = component_year["population"].sum()
        aggregate_total = aggregate_year["population"].sum()

        difference = abs(aggregate_total - component_total)
        percent_diff = (difference / component_total) if component_total > 0 else 0.0

        year_result = {
            "year": int(year),
            "component_total": float(component_total),
            "aggregate_total": float(aggregate_total),
            "difference": float(difference),
            "percent_difference": float(percent_diff),
            "within_tolerance": percent_diff <= tolerance,
        }

        validation_result["by_year"].append(year_result)

        if percent_diff > tolerance:
            msg = (
                f"Year {year}: Aggregation difference {percent_diff:.2%} exceeds "
                f"tolerance {tolerance:.2%}"
            )
            if percent_diff > 0.05:  # 5% is serious
                validation_result["errors"].append(msg)
                validation_result["valid"] = False
            else:
                validation_result["warnings"].append(msg)

    # Overall summary
    total_component = component_aggregate["population"].sum()
    total_aggregate = aggregated_projection["population"].sum()
    overall_diff = abs(total_aggregate - total_component)
    overall_pct = (overall_diff / total_component) if total_component > 0 else 0.0

    validation_result["overall"] = {
        "component_total": float(total_component),
        "aggregate_total": float(total_aggregate),
        "difference": float(overall_diff),
        "percent_difference": float(overall_pct),
    }

    if validation_result["valid"]:
        logger.info(f"Aggregation validation passed. " f"Overall difference: {overall_pct:.3%}")
    else:
        logger.error(
            f"Aggregation validation failed with {len(validation_result['errors'])} errors"
        )

    return validation_result


def _save_projection_results(
    fips: str,
    level: str,
    projection: pd.DataFrame,
    summary: pd.DataFrame,
    metadata: dict,
    output_dir: Path | None,
    config: dict,
):
    """Save projection results to files."""
    # Set output directory
    if output_dir is None:
        project_root = Path(__file__).parent.parent.parent
        output_dir = project_root / "data" / "output" / "projections" / f"{level}s"

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get scenario and years for filename
    base_year = config.get("project", {}).get("base_year", 2025)
    horizon = config.get("project", {}).get("projection_horizon", 20)
    end_year = base_year + horizon
    scenario = metadata["projection"].get("scenario", "baseline")

    # File naming: nd_{level}_{fips}_projection_{base_year}_{end_year}_{scenario}.ext
    base_filename = f"nd_{level}_{fips}_projection_{base_year}_{end_year}_{scenario}"

    # Save projection (parquet)
    projection_file = output_dir / f"{base_filename}.parquet"
    compression = config.get("output", {}).get("compression", "gzip")
    projection.to_parquet(projection_file, compression=compression, index=False)

    # Save summary (CSV)
    if not summary.empty:
        summary_file = output_dir / f"{base_filename}_summary.csv"
        summary.to_csv(summary_file, index=False)

    # Save metadata (JSON)
    metadata_file = output_dir / f"{base_filename}_metadata.json"
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)


if __name__ == "__main__":
    """Example usage and testing."""

    logger.info("Multi-geography projection module loaded successfully")
    logger.info("Ready to run projections for multiple geographies")

    # Note: Full testing requires rate data and base population
    # See examples/run_multi_geography_example.py for complete example
