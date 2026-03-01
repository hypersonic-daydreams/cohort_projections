#!/usr/bin/env python3
"""
Export and Dissemination Pipeline for North Dakota Population Projections.

This script exports projection results to various formats for dissemination:
- Converts Parquet → CSV/Excel
- Creates summary statistics tables
- Generates comparison reports (baseline vs scenarios)
- Packages results for distribution
- Creates data dictionaries

Usage:
    # Export all results
    python 03_export_results.py --all

    # Export specific levels
    python 03_export_results.py --state --counties
    python 03_export_results.py --places

    # Export specific scenarios
    python 03_export_results.py --all --scenarios baseline high_growth

    # Export specific geographies
    python 03_export_results.py --fips 38101 38015

    # Export only specific formats
    python 03_export_results.py --all --formats csv

    # Create distribution packages
    python 03_export_results.py --all --package

    # Dry run mode
    python 03_export_results.py --all --dry-run

Key ADRs and config:
    ADR-012: Output and export format strategy
    Config: output.formats, output.directory
"""

import argparse
import importlib
import json
import shutil
import sys
import traceback
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import pandas as pd

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from project_utils import setup_logger  # noqa: E402

from cohort_projections.utils import load_projection_config  # noqa: E402

# Set up logging
logger = setup_logger(__name__, log_level="INFO")


class ExportResult:
    """Container for export operation results."""

    def __init__(self, component: str):
        self.component = component
        self.success = False
        self.error: str | None = None
        self.files_exported = 0
        self.output_files: list[Path] = []
        self.export_time = 0.0


class ExportReport:
    """Generate export report with statistics."""

    def __init__(self):
        self.results: list[ExportResult] = []
        self.start_time = datetime.now(UTC)
        self.end_time: datetime | None = None
        self.total_files_exported = 0
        self.packages_created: list[Path] = []

    def add_result(self, result: ExportResult):
        """Add an export result."""
        self.results.append(result)
        if result.success:
            self.total_files_exported += result.files_exported

    def finalize(self):
        """Finalize the report."""
        self.end_time = datetime.now(UTC)

    def get_summary(self) -> dict[str, Any]:
        """Get report summary."""
        successful = sum(1 for r in self.results if r.success)
        failed = sum(1 for r in self.results if not r.success)

        return {
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_seconds": (
                (self.end_time - self.start_time).total_seconds() if self.end_time else None
            ),
            "components": {
                "total": len(self.results),
                "successful": successful,
                "failed": failed,
            },
            "files_exported": self.total_files_exported,
            "packages_created": [str(p) for p in self.packages_created],
            "details": [
                {
                    "component": r.component,
                    "success": r.success,
                    "error": r.error,
                    "files_exported": r.files_exported,
                    "export_time": r.export_time,
                }
                for r in self.results
            ],
        }

    def print_summary(self):
        """Print human-readable summary."""
        print("\n" + "=" * 80)
        print("EXPORT PIPELINE SUMMARY")
        print("=" * 80)

        summary = self.get_summary()
        print(
            f"\nDuration: {summary['duration_seconds']:.2f} seconds"
            if summary["duration_seconds"]
            else "N/A"
        )
        print(f"Files Exported: {summary['files_exported']}")
        print(f"Packages Created: {len(summary['packages_created'])}")

        print(f"\nComponents: {summary['components']['total']}")
        print(f"  Successful: {summary['components']['successful']}")
        print(f"  Failed: {summary['components']['failed']}")

        if summary["packages_created"]:
            print("\nPackages Created:")
            for pkg in summary["packages_created"]:
                print(f"  - {pkg}")

        print("\n" + "=" * 80 + "\n")


CompressionType = Literal["infer", "gzip", "bz2", "zip", "xz", "zstd", "tar"]


def convert_parquet_to_csv(
    parquet_file: Path, output_dir: Path, compression: CompressionType | None = "gzip"
) -> Path:
    """
    Convert Parquet file to CSV.

    Args:
        parquet_file: Input Parquet file
        output_dir: Output directory
        compression: Compression method (None, 'gzip', 'bz2', 'zip', 'xz')

    Returns:
        Path to output CSV file
    """
    df = pd.read_parquet(parquet_file)

    output_file = output_dir / parquet_file.stem
    if compression:
        output_file = output_file.with_suffix(f".csv.{compression}")
        df.to_csv(output_file, index=False, compression=compression)
    else:
        output_file = output_file.with_suffix(".csv")
        df.to_csv(output_file, index=False)

    return output_file


def convert_parquet_to_excel(
    parquet_files: list[Path], output_file: Path, sheet_name_map: dict[str, str] | None = None
) -> Path:
    """
    Convert Parquet files to Excel workbook with multiple sheets.

    Args:
        parquet_files: List of Parquet files
        output_file: Output Excel file
        sheet_name_map: Optional mapping of file stems to sheet names

    Returns:
        Path to output Excel file
    """
    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        for parquet_file in parquet_files:
            df = pd.read_parquet(parquet_file)

            # Determine sheet name
            if sheet_name_map and parquet_file.stem in sheet_name_map:
                sheet_name = sheet_name_map[parquet_file.stem]
            else:
                sheet_name = parquet_file.stem[:31]  # Excel sheet name limit

            df.to_excel(writer, sheet_name=sheet_name, index=False)

    return output_file


def convert_projection_formats(
    scenario: str,
    levels: list[str],
    config: dict[str, Any],
    formats: list[str],
    output_dir: Path,
    dry_run: bool = False,
) -> ExportResult:
    """
    Convert projection outputs to various formats.

    Args:
        scenario: Scenario name
        levels: Geographic levels to export
        config: Project configuration
        formats: Output formats ('csv', 'excel', 'parquet')
        output_dir: Output directory
        dry_run: If True, only show what would be exported

    Returns:
        ExportResult with conversion results
    """
    result = ExportResult(f"format_conversion_{scenario}")
    start_time = datetime.now(UTC)

    try:
        logger.info(f"Converting projection formats for scenario: {scenario}")

        # Get projection directory
        proj_dir = (
            Path(
                config.get("pipeline", {})
                .get("projection", {})
                .get("output_dir", "data/projections")
            )
            / scenario
        )

        if not proj_dir.exists():
            raise FileNotFoundError(f"Projection directory not found: {proj_dir}")

        # Create output directory
        scenario_output = output_dir / scenario
        scenario_output.mkdir(parents=True, exist_ok=True)

        files_converted = 0

        for level in levels:
            level_dir = proj_dir / level
            if not level_dir.exists():
                logger.warning(f"Level directory not found: {level_dir}")
                continue

            parquet_files = list(level_dir.glob("*.parquet"))
            if not parquet_files:
                logger.warning(f"No projection files found in {level_dir}")
                continue

            logger.info(f"Processing {len(parquet_files)} files for {level} level")

            if dry_run:
                logger.info(f"[DRY RUN] Would convert {len(parquet_files)} files")
                files_converted += len(parquet_files) * len(formats)
                continue

            # Create level output directory
            level_output = scenario_output / level
            level_output.mkdir(parents=True, exist_ok=True)

            # Convert to each format
            if "csv" in formats:
                csv_dir = level_output / "csv"
                csv_dir.mkdir(parents=True, exist_ok=True)

                for pfile in parquet_files:
                    csv_file = convert_parquet_to_csv(
                        pfile,
                        csv_dir,
                        compression=config.get("output", {}).get("compression", "gzip"),
                    )
                    result.output_files.append(csv_file)
                    files_converted += 1

                logger.info(f"Converted {len(parquet_files)} files to CSV")

            if "excel" in formats:
                excel_dir = level_output / "excel"
                excel_dir.mkdir(parents=True, exist_ok=True)

                # Create one Excel file per geography or combine all
                for pfile in parquet_files:
                    excel_file = excel_dir / pfile.stem
                    excel_file = excel_file.with_suffix(".xlsx")
                    convert_parquet_to_excel([pfile], excel_file)
                    result.output_files.append(excel_file)
                    files_converted += 1

                logger.info(f"Converted {len(parquet_files)} files to Excel")

        result.success = True
        result.files_exported = files_converted
        logger.info(f"Format conversion complete: {files_converted} files")

    except Exception as e:
        logger.error(f"Error converting formats: {e}")
        logger.debug(traceback.format_exc())
        result.success = False
        result.error = str(e)

    result.export_time = (datetime.now(UTC) - start_time).total_seconds()
    return result


def create_summary_tables(
    scenario: str,
    levels: list[str],
    config: dict[str, Any],
    output_dir: Path,
    dry_run: bool = False,
) -> ExportResult:
    """
    Create summary statistics tables.

    Args:
        scenario: Scenario name
        levels: Geographic levels
        config: Project configuration
        output_dir: Output directory
        dry_run: If True, only show what would be created

    Returns:
        ExportResult with summary creation results
    """
    result = ExportResult(f"summaries_{scenario}")
    start_time = datetime.now(UTC)

    try:
        logger.info(f"Creating summary tables for scenario: {scenario}")

        if dry_run:
            logger.info("[DRY RUN] Would create summary tables")
            result.success = True
            return result

        # Get projection directory
        proj_dir = (
            Path(
                config.get("pipeline", {})
                .get("projection", {})
                .get("output_dir", "data/projections")
            )
            / scenario
        )

        # Get summary types from config
        summary_types = (
            config.get("pipeline", {})
            .get("export", {})
            .get("summaries", ["total_population_by_year"])
        )

        summaries_created = 0
        summary_output = output_dir / scenario / "summaries"
        summary_output.mkdir(parents=True, exist_ok=True)

        for level in levels:
            level_dir = proj_dir / level
            if not level_dir.exists():
                continue

            # Load all projection files for this level
            projection_files = list(level_dir.glob("*.parquet"))
            if not projection_files:
                continue

            logger.info(
                f"Creating summaries for {level} level ({len(projection_files)} geographies)"
            )

            # Create each summary type
            for summary_type in summary_types:
                try:
                    if summary_type == "total_population_by_year":
                        summary_df = create_total_population_summary(projection_files)
                    elif summary_type == "age_distribution_by_year":
                        summary_df = create_age_distribution_summary(projection_files)
                    elif summary_type == "sex_ratio_by_year":
                        summary_df = create_sex_ratio_summary(projection_files)
                    elif summary_type == "race_composition_by_year":
                        summary_df = create_race_composition_summary(projection_files)
                    elif summary_type == "growth_rates":
                        summary_df = create_growth_rates_summary(projection_files)
                    elif summary_type == "dependency_ratios":
                        summary_df = create_dependency_ratios_summary(projection_files)
                    else:
                        logger.warning(f"Unknown summary type: {summary_type}")
                        continue

                    # Save summary
                    output_file = summary_output / f"{level}_{summary_type}.csv"
                    summary_df.to_csv(output_file, index=False)
                    result.output_files.append(output_file)
                    summaries_created += 1

                except Exception as e:
                    logger.warning(f"Could not create summary {summary_type}: {e}")

        result.success = True
        result.files_exported = summaries_created
        logger.info(f"Created {summaries_created} summary tables")

    except Exception as e:
        logger.error(f"Error creating summary tables: {e}")
        logger.debug(traceback.format_exc())
        result.success = False
        result.error = str(e)

    result.export_time = (datetime.now(UTC) - start_time).total_seconds()
    return result


def _extract_fips_from_filename(filepath: Path) -> str:
    """
    Extract FIPS code from projection filename.

    Filenames follow the pattern: nd_{level}_{fips}_projection_{start}_{end}_{scenario}.parquet
    For example: nd_county_38001_projection_2025_2055_baseline.parquet

    Args:
        filepath: Path to a projection parquet file.

    Returns:
        The FIPS code string (e.g. '38001').
    """
    parts = filepath.stem.split("_")
    # FIPS is at index 2: nd=0, county=1, 38001=2
    return parts[2]


def create_total_population_summary(projection_files: list[Path]) -> pd.DataFrame:
    """Create total population by year summary."""
    summaries = []

    for pfile in projection_files:
        df = pd.read_parquet(pfile)
        fips = _extract_fips_from_filename(pfile)

        # Group by year and sum population
        yearly = df.groupby("year")["population"].sum().reset_index()
        yearly["fips"] = fips

        summaries.append(yearly)

    if summaries:
        result = pd.concat(summaries, ignore_index=True)
        # Pivot to wide format (fips as rows, years as columns)
        result = result.pivot(index="fips", columns="year", values="population")
        result.reset_index(inplace=True)
        return result

    return pd.DataFrame()


def create_age_distribution_summary(projection_files: list[Path]) -> pd.DataFrame:
    """
    Create age distribution summary.

    Returns a DataFrame with one row per county per year, showing population
    counts and percentages for standard age groups.
    """
    logger.info("Creating age distribution summary...")
    rows = []

    age_groups = {
        "0-4": (0, 4),
        "5-17": (5, 17),
        "18-24": (18, 24),
        "25-44": (25, 44),
        "45-64": (45, 64),
        "65-74": (65, 74),
        "75-84": (75, 84),
        "85+": (85, 200),
    }

    for pfile in projection_files:
        df = pd.read_parquet(pfile)
        fips = _extract_fips_from_filename(pfile)

        for year in sorted(df["year"].unique()):
            year_data = df[df["year"] == year]
            total_pop = year_data["population"].sum()
            row: dict[str, object] = {"fips": fips, "year": int(year), "total_population": total_pop}

            for group_name, (min_age, max_age) in age_groups.items():
                group_pop = year_data[
                    (year_data["age"] >= min_age) & (year_data["age"] <= max_age)
                ]["population"].sum()
                row[f"pop_{group_name}"] = group_pop
                row[f"pct_{group_name}"] = round((group_pop / total_pop * 100), 2) if total_pop > 0 else 0.0

            rows.append(row)

    if rows:
        return pd.DataFrame(rows)
    return pd.DataFrame()


def create_sex_ratio_summary(projection_files: list[Path]) -> pd.DataFrame:
    """
    Create sex ratio summary.

    Returns a DataFrame with one row per county per year, showing male population,
    female population, total population, and sex ratio (males per 100 females).
    """
    logger.info("Creating sex ratio summary...")
    rows = []

    for pfile in projection_files:
        df = pd.read_parquet(pfile)
        fips = _extract_fips_from_filename(pfile)

        for year in sorted(df["year"].unique()):
            year_data = df[df["year"] == year]
            male_pop = year_data[year_data["sex"] == "Male"]["population"].sum()
            female_pop = year_data[year_data["sex"] == "Female"]["population"].sum()
            total_pop = male_pop + female_pop
            sex_ratio = round((male_pop / female_pop * 100), 2) if female_pop > 0 else 0.0

            rows.append({
                "fips": fips,
                "year": int(year),
                "male_population": male_pop,
                "female_population": female_pop,
                "total_population": total_pop,
                "sex_ratio": sex_ratio,
            })

    if rows:
        return pd.DataFrame(rows)
    return pd.DataFrame()


def create_race_composition_summary(projection_files: list[Path]) -> pd.DataFrame:
    """
    Create race composition summary.

    Returns a DataFrame with one row per county per year, showing population
    counts and percentages for each race/ethnicity category.
    """
    logger.info("Creating race composition summary...")
    rows = []

    for pfile in projection_files:
        df = pd.read_parquet(pfile)
        fips = _extract_fips_from_filename(pfile)

        for year in sorted(df["year"].unique()):
            year_data = df[df["year"] == year]
            total_pop = year_data["population"].sum()
            row: dict[str, object] = {"fips": fips, "year": int(year), "total_population": total_pop}

            for race in sorted(year_data["race"].unique()):
                race_pop = year_data[year_data["race"] == race]["population"].sum()
                # Create safe column name from race category
                safe_name = race.lower().replace(" ", "_").replace("/", "_").replace(",", "")
                row[f"pop_{safe_name}"] = race_pop
                row[f"pct_{safe_name}"] = round((race_pop / total_pop * 100), 2) if total_pop > 0 else 0.0

            rows.append(row)

    if rows:
        return pd.DataFrame(rows)
    return pd.DataFrame()


def create_growth_rates_summary(projection_files: list[Path]) -> pd.DataFrame:
    """
    Create growth rates summary.

    Returns a DataFrame with one row per county containing the FIPS code,
    county name, base year population, final year population, absolute change,
    and percentage change over the full projection horizon.
    """
    logger.info("Creating growth rates summary...")
    rows = []

    for pfile in projection_files:
        df = pd.read_parquet(pfile)
        fips = _extract_fips_from_filename(pfile)

        years = sorted(df["year"].unique())
        if len(years) < 2:
            continue

        base_year = int(years[0])
        final_year = int(years[-1])

        base_pop = df[df["year"] == years[0]]["population"].sum()
        final_pop = df[df["year"] == years[-1]]["population"].sum()

        absolute_change = final_pop - base_pop
        pct_change = round(((final_pop / base_pop - 1) * 100), 2) if base_pop > 0 else 0.0

        # Compound annual growth rate (CAGR)
        n_years = final_year - base_year
        cagr = (
            round((((final_pop / base_pop) ** (1 / n_years)) - 1) * 100, 4)
            if n_years > 0 and base_pop > 0
            else 0.0
        )

        rows.append({
            "fips": fips,
            "base_year": base_year,
            "final_year": final_year,
            "base_population": round(base_pop, 1),
            "final_population": round(final_pop, 1),
            "absolute_change": round(absolute_change, 1),
            "pct_change": pct_change,
            "annual_growth_rate": cagr,
        })

    if rows:
        result = pd.DataFrame(rows).sort_values("fips").reset_index(drop=True)
        return result
    return pd.DataFrame()


def create_dependency_ratios_summary(projection_files: list[Path]) -> pd.DataFrame:
    """
    Create dependency ratios summary.

    Returns a DataFrame with one row per county per year containing the total
    dependency ratio, youth dependency ratio, and elderly dependency ratio.
    """
    logger.info("Creating dependency ratios summary...")
    rows = []

    for pfile in projection_files:
        df = pd.read_parquet(pfile)
        fips = _extract_fips_from_filename(pfile)

        for year in sorted(df["year"].unique()):
            year_data = df[df["year"] == year]

            youth_pop = year_data[year_data["age"] < 18]["population"].sum()
            working_pop = year_data[
                (year_data["age"] >= 18) & (year_data["age"] < 65)
            ]["population"].sum()
            elderly_pop = year_data[year_data["age"] >= 65]["population"].sum()

            total_dep = round(((youth_pop + elderly_pop) / working_pop), 4) if working_pop > 0 else 0.0
            youth_dep = round((youth_pop / working_pop), 4) if working_pop > 0 else 0.0
            elderly_dep = round((elderly_pop / working_pop), 4) if working_pop > 0 else 0.0

            rows.append({
                "fips": fips,
                "year": int(year),
                "youth_population": youth_pop,
                "working_age_population": working_pop,
                "elderly_population": elderly_pop,
                "total_dependency_ratio": total_dep,
                "youth_dependency_ratio": youth_dep,
                "elderly_dependency_ratio": elderly_dep,
            })

    if rows:
        return pd.DataFrame(rows)
    return pd.DataFrame()


def generate_data_dictionary(
    config: dict[str, Any], output_dir: Path, dry_run: bool = False
) -> ExportResult:
    """
    Generate data dictionary documenting output variables.

    Args:
        config: Project configuration
        output_dir: Output directory
        dry_run: If True, only show what would be generated

    Returns:
        ExportResult with data dictionary generation results
    """
    result = ExportResult("data_dictionary")
    start_time = datetime.now(UTC)

    try:
        logger.info("Generating data dictionary...")

        if dry_run:
            logger.info("[DRY RUN] Would generate data dictionary")
            result.success = True
            return result

        # Create data dictionary content
        data_dict: dict[str, Any] = {
            "metadata": {
                "title": "North Dakota Population Projections - Data Dictionary",
                "generated": datetime.now(UTC).isoformat(),
                "project": config.get("project", {}).get("name", "ND Population Projections"),
            },
            "variables": [
                {
                    "name": "fips",
                    "description": "Federal Information Processing Standards (FIPS) code for geography",
                    "type": "string",
                    "examples": ["38 (state)", "38101 (Cass County)", "3825700 (Fargo city)"],
                },
                {
                    "name": "year",
                    "description": "Projection year",
                    "type": "integer",
                    "range": f"{config.get('project', {}).get('base_year', 2025)} to "
                    f"{config.get('project', {}).get('base_year', 2025) + config.get('project', {}).get('projection_horizon', 20)}",
                },
                {
                    "name": "age",
                    "description": "Single-year age group",
                    "type": "integer",
                    "range": "0-90 (90+ is open-ended group)",
                },
                {
                    "name": "sex",
                    "description": "Biological sex",
                    "type": "string",
                    "values": ["Male", "Female"],
                },
                {
                    "name": "race_ethnicity",
                    "description": "Race and ethnicity category (6-category system)",
                    "type": "string",
                    "values": config.get("demographics", {})
                    .get("race_ethnicity", {})
                    .get("categories", []),
                },
                {
                    "name": "population",
                    "description": "Projected population count",
                    "type": "float",
                    "notes": "Decimal values due to cohort-component methodology",
                },
            ],
            "geographic_levels": [
                {
                    "level": "state",
                    "description": "North Dakota state total",
                    "fips_format": "2-digit (38)",
                },
                {
                    "level": "county",
                    "description": "County subdivisions",
                    "fips_format": "5-digit (38XXX)",
                    "count": 53,
                },
                {
                    "level": "place",
                    "description": "Incorporated places (cities/towns)",
                    "fips_format": "7-digit (38XXXXX)",
                    "notes": "Filtered by population threshold",
                },
            ],
            "scenarios": [
                {
                    "name": name,
                    "description": scenario.get("description", ""),
                    "active": scenario.get("active", False),
                }
                for name, scenario in config.get("scenarios", {}).items()
            ],
        }

        # Save as JSON
        dict_file = output_dir / "data_dictionary.json"
        with open(dict_file, "w") as f:
            json.dump(data_dict, f, indent=2)

        result.output_files.append(dict_file)

        # Also create human-readable markdown version
        md_file = output_dir / "data_dictionary.md"
        with open(md_file, "w") as f:
            f.write(f"# {data_dict['metadata']['title']}\n\n")
            f.write(f"Generated: {data_dict['metadata']['generated']}\n\n")

            f.write("## Variables\n\n")
            for var in data_dict["variables"]:
                f.write(f"### {var['name']}\n\n")
                f.write(f"- **Description**: {var['description']}\n")
                f.write(f"- **Type**: {var['type']}\n")
                if "range" in var:
                    f.write(f"- **Range**: {var['range']}\n")
                if "values" in var:
                    f.write(f"- **Values**: {', '.join(var['values'])}\n")
                if "notes" in var:
                    f.write(f"- **Notes**: {var['notes']}\n")
                f.write("\n")

            f.write("## Geographic Levels\n\n")
            for geo in data_dict["geographic_levels"]:
                f.write(f"### {geo['level'].title()}\n\n")
                f.write(f"- **Description**: {geo['description']}\n")
                f.write(f"- **FIPS Format**: {geo['fips_format']}\n")
                if "count" in geo:
                    f.write(f"- **Count**: {geo['count']}\n")
                if "notes" in geo:
                    f.write(f"- **Notes**: {geo['notes']}\n")
                f.write("\n")

        result.output_files.append(md_file)
        result.success = True
        result.files_exported = 2
        logger.info("Data dictionary generated")

    except Exception as e:
        logger.error(f"Error generating data dictionary: {e}")
        logger.debug(traceback.format_exc())
        result.success = False
        result.error = str(e)

    result.export_time = (datetime.now(UTC) - start_time).total_seconds()
    return result


def package_for_distribution(
    scenarios: list[str],
    levels: list[str],
    config: dict[str, Any],
    export_dir: Path,
    package_by: str = "level",
    dry_run: bool = False,
) -> ExportResult:
    """
    Create ZIP archives for distribution.

    Args:
        scenarios: Scenarios to package
        levels: Geographic levels to package
        config: Project configuration
        export_dir: Export directory containing files to package
        package_by: 'level' (one package per level) or 'geography' (one per geography)
        dry_run: If True, only show what would be packaged

    Returns:
        ExportResult with packaging results
    """
    result = ExportResult("packaging")
    start_time = datetime.now(UTC)

    try:
        logger.info("Creating distribution packages...")

        if dry_run:
            logger.info("[DRY RUN] Would create distribution packages")
            result.success = True
            return result

        packages_dir = export_dir / "packages"
        packages_dir.mkdir(parents=True, exist_ok=True)

        packages_created = 0

        if package_by == "level":
            # Create one package per level across all scenarios
            for level in levels:
                package_name = f"nd_projections_{level}_{datetime.now(UTC).strftime('%Y%m%d')}.zip"
                package_file = packages_dir / package_name

                with zipfile.ZipFile(package_file, "w", zipfile.ZIP_DEFLATED) as zipf:
                    for scenario in scenarios:
                        scenario_dir = export_dir / scenario / level
                        if scenario_dir.exists():
                            for file in scenario_dir.rglob("*"):
                                if file.is_file():
                                    arcname = f"{scenario}/{level}/{file.relative_to(scenario_dir)}"
                                    zipf.write(file, arcname)

                    # Add data dictionary
                    dict_file = export_dir / "data_dictionary.md"
                    if dict_file.exists():
                        zipf.write(dict_file, "data_dictionary.md")

                result.output_files.append(package_file)
                packages_created += 1
                logger.info(f"Created package: {package_name}")

        result.success = True
        result.files_exported = packages_created
        logger.info(f"Created {packages_created} distribution packages")

    except Exception as e:
        logger.error(f"Error creating packages: {e}")
        logger.debug(traceback.format_exc())
        result.success = False
        result.error = str(e)

    result.export_time = (datetime.now(UTC) - start_time).total_seconds()
    return result


def _build_place_workbook(scenario: str, config: dict[str, Any]) -> Path:
    """
    Build place workbook for one scenario via PP-003 workbook builder.

    Args:
        scenario: Scenario key.
        config: Project configuration.

    Returns:
        Path to workbook artifact.
    """
    workbook_module = importlib.import_module("scripts.exports.build_place_workbook")
    workbook_path = workbook_module.build_workbook(scenario=scenario, config=config)
    return Path(workbook_path)


def export_place_outputs(
    scenario: str,
    config: dict[str, Any],
    output_dir: Path,
    dry_run: bool = False,
) -> ExportResult:
    """
    Export place-specific artifacts required by PP-003 IMP-16.

    Args:
        scenario: Scenario name.
        config: Project configuration.
        output_dir: Export root directory.
        dry_run: If True, only report actions.

    Returns:
        ExportResult with place artifact export status.
    """
    result = ExportResult(f"places_{scenario}")
    start_time = datetime.now(UTC)

    try:
        proj_dir = (
            Path(
                config.get("pipeline", {})
                .get("projection", {})
                .get("output_dir", "data/projections")
            )
            / scenario
            / "place"
        )
        source_summary = proj_dir / "places_summary.csv"
        if not source_summary.exists():
            logger.warning(
                "Skipping place artifacts for %s: source summary not found (%s)",
                scenario,
                source_summary,
            )
            result.success = True
            result.files_exported = 0
            return result

        workbook_output = output_dir / f"nd_projections_{scenario}_places_{datetime.now(UTC).strftime('%Y%m%d')}.xlsx"
        if dry_run:
            logger.info("[DRY RUN] Would copy %s -> %s", source_summary, output_dir / scenario / "place")
            logger.info("[DRY RUN] Would build place workbook: %s", workbook_output)
            result.success = True
            result.files_exported = 2
            return result

        scenario_output = output_dir / scenario / "place"
        scenario_output.mkdir(parents=True, exist_ok=True)
        exported_summary = scenario_output / "places_summary.csv"
        shutil.copy2(source_summary, exported_summary)
        result.output_files.append(exported_summary)

        workbook_path = _build_place_workbook(scenario=scenario, config=config)
        result.output_files.append(workbook_path)

        result.success = True
        result.files_exported = 2
        logger.info("Exported place artifacts for %s: summary=%s workbook=%s", scenario, exported_summary, workbook_path)
    except Exception as e:
        logger.error(f"Error exporting place artifacts for {scenario}: {e}")
        logger.debug(traceback.format_exc())
        result.success = False
        result.error = str(e)

    result.export_time = (datetime.now(UTC) - start_time).total_seconds()
    return result


def export_geospatial_outputs(
    scenario: str,
    levels: list[str],
    config: dict[str, Any],
    output_dir: Path,
    format_type: Literal["geojson", "shapefile"] = "geojson",
    dry_run: bool = False,
) -> ExportResult:
    """
    Export projection data as geospatial files (GeoJSON or Shapefile).

    Joins projection population totals to Census TIGER boundary geometries
    and writes one file per geography level per key year.

    Args:
        scenario: Scenario name (e.g., 'baseline').
        levels: Geographic levels to export ('county', 'place').
        config: Project configuration dictionary.
        output_dir: Root export directory.
        format_type: Output format -- 'geojson' or 'shapefile'.
        dry_run: If True, log planned actions without writing files.

    Returns:
        ExportResult summarising the export operation.
    """
    result = ExportResult(f"geospatial_{format_type}_{scenario}")
    start_time = datetime.now(UTC)

    try:
        logger.info(
            "Exporting geospatial %s for scenario: %s", format_type, scenario
        )

        # Geospatial exports only apply to county and place levels
        geo_levels = [lvl for lvl in levels if lvl in ("county", "place")]
        if not geo_levels:
            logger.info("No county/place levels requested -- skipping geospatial export")
            result.success = True
            return result

        proj_dir = (
            Path(
                config.get("pipeline", {})
                .get("projection", {})
                .get("output_dir", "data/projections")
            )
            / scenario
        )

        # Determine key years from config
        key_years: list[int] = (
            config.get("place_projections", {})
            .get("output", {})
            .get("key_years", [])
        )

        ext = "geojson" if format_type == "geojson" else "shp"
        files_exported = 0

        for level in geo_levels:
            level_dir = proj_dir / level
            if not level_dir.exists():
                logger.warning("Level directory not found: %s", level_dir)
                continue

            parquet_files = list(level_dir.glob("*.parquet"))
            if not parquet_files:
                logger.warning("No projection files in %s", level_dir)
                continue

            # Read and combine all projection files for this level
            frames = []
            for pfile in parquet_files:
                df = pd.read_parquet(pfile)
                fips = _extract_fips_from_filename(pfile)
                fips_col = "county_fips" if level == "county" else "place_fips"
                df[fips_col] = fips
                frames.append(df)
            combined = pd.concat(frames, ignore_index=True)

            # Determine which years to export
            export_years: list[int] = key_years if key_years else sorted(int(y) for y in combined["year"].unique())

            geo_output = output_dir / scenario / level / format_type
            geo_output.mkdir(parents=True, exist_ok=True)

            for yr in export_years:
                out_file = geo_output / f"nd_{level}_{scenario}_{yr}.{ext}"
                if dry_run:
                    logger.info("[DRY RUN] Would write %s", out_file)
                    files_exported += 1
                    continue

                try:
                    from cohort_projections.output.writers import write_projection_shapefile

                    write_projection_shapefile(
                        projection_df=combined,
                        geography_level=level,
                        output_path=out_file,
                        year=yr,
                        format_type=format_type,
                    )
                    result.output_files.append(out_file)
                    files_exported += 1
                except ImportError:
                    logger.warning(
                        "geopandas not available -- skipping geospatial export"
                    )
                    result.success = True
                    result.files_exported = 0
                    return result
                except FileNotFoundError as fnf:
                    logger.warning("TIGER file missing, skipping geospatial: %s", fnf)
                    result.success = True
                    result.files_exported = 0
                    return result

        result.success = True
        result.files_exported = files_exported
        logger.info("Geospatial export complete: %d files", files_exported)

    except Exception as e:
        logger.error("Error in geospatial export: %s", e)
        logger.debug(traceback.format_exc())
        result.success = False
        result.error = str(e)

    result.export_time = (datetime.now(UTC) - start_time).total_seconds()
    return result


def export_all_results(
    config: dict[str, Any],
    scenarios: list[str],
    levels: list[str],
    formats: list[str],
    create_packages: bool = True,
    dry_run: bool = False,
) -> ExportReport:
    """
    Main orchestrator for exporting all results.

    Args:
        config: Project configuration
        scenarios: Scenarios to export
        levels: Geographic levels to export
        formats: Output formats
        create_packages: Whether to create distribution packages
        dry_run: If True, only show what would be exported

    Returns:
        ExportReport with all export results
    """
    logger.info("=" * 80)
    logger.info("EXPORT PIPELINE - North Dakota Population Projections")
    logger.info("=" * 80)
    logger.info(f"Scenarios: {', '.join(scenarios)}")
    logger.info(f"Levels: {', '.join(levels)}")
    logger.info(f"Formats: {', '.join(formats)}")
    logger.info(f"Create packages: {create_packages}")
    logger.info(f"Dry run: {dry_run}")
    logger.info("")

    report = ExportReport()

    # Get export directory
    export_dir = Path(
        config.get("pipeline", {}).get("export", {}).get("output_dir", "data/exports")
    )
    export_dir.mkdir(parents=True, exist_ok=True)

    # Convert formats for each scenario
    for scenario in scenarios:
        result = convert_projection_formats(
            scenario=scenario,
            levels=levels,
            config=config,
            formats=formats,
            output_dir=export_dir,
            dry_run=dry_run,
        )
        report.add_result(result)

    # Create summary tables for each scenario
    for scenario in scenarios:
        result = create_summary_tables(
            scenario=scenario,
            levels=levels,
            config=config,
            output_dir=export_dir,
            dry_run=dry_run,
        )
        report.add_result(result)

    # Export place-specific artifacts (PP-003 IMP-16)
    if "place" in levels:
        for scenario in scenarios:
            result = export_place_outputs(
                scenario=scenario,
                config=config,
                output_dir=export_dir,
                dry_run=dry_run,
            )
            report.add_result(result)

    # Export geospatial outputs (ADR-059)
    for geo_fmt in ("geojson", "shapefile"):
        if geo_fmt in formats:
            for scenario in scenarios:
                result = export_geospatial_outputs(
                    scenario=scenario,
                    levels=levels,
                    config=config,
                    output_dir=export_dir,
                    format_type=geo_fmt,
                    dry_run=dry_run,
                )
                report.add_result(result)

    # Generate data dictionary
    result = generate_data_dictionary(config=config, output_dir=export_dir, dry_run=dry_run)
    report.add_result(result)

    # Create distribution packages
    if create_packages:
        result = package_for_distribution(
            scenarios=scenarios,
            levels=levels,
            config=config,
            export_dir=export_dir,
            package_by=config.get("pipeline", {}).get("export", {}).get("package_by", "level"),
            dry_run=dry_run,
        )
        report.add_result(result)
        if result.success:
            report.packages_created.extend(result.output_files)

    report.finalize()
    return report


def main():
    """Main entry point for export pipeline."""
    parser = argparse.ArgumentParser(
        description="Export population projection results for dissemination",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Export all results
  python 03_export_results.py --all

  # Export specific levels
  python 03_export_results.py --state --counties

  # Export specific scenarios
  python 03_export_results.py --all --scenarios baseline high_growth

  # Export only CSV format
  python 03_export_results.py --all --formats csv

  # Skip package creation
  python 03_export_results.py --all --no-package
        """,
    )

    # Level selection
    parser.add_argument("--all", action="store_true", help="Export all geographic levels")
    parser.add_argument("--state", action="store_true", help="Export state-level")
    parser.add_argument("--counties", action="store_true", help="Export county-level")
    parser.add_argument("--places", action="store_true", help="Export place-level")

    # Scenario selection
    parser.add_argument(
        "--scenarios",
        nargs="+",
        help="Scenarios to export (default: all with projections)",
    )

    # Format selection
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=["csv", "excel", "parquet", "geojson", "shapefile"],
        help="Output formats (default: from config)",
    )

    # Options
    parser.add_argument(
        "--package", action="store_true", help="Create distribution packages (default: from config)"
    )
    parser.add_argument(
        "--no-package", action="store_true", help="Skip creating distribution packages"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be exported without actually exporting",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to configuration file (default: config/projection_config.yaml)",
    )

    args = parser.parse_args()

    # Determine levels
    levels = []
    if args.all:
        levels = ["state", "county", "place"]
    else:
        if args.state:
            levels.append("state")
        if args.counties:
            levels.append("county")
        if args.places:
            levels.append("place")

    if not levels:
        parser.error("No geographic levels specified. Use --all or specify individual levels.")

    try:
        # Load configuration
        config = load_projection_config(args.config)

        # Determine scenarios
        scenarios = args.scenarios
        if not scenarios:
            # Find scenarios with projection outputs
            proj_dir = Path(
                config.get("pipeline", {})
                .get("projection", {})
                .get("output_dir", "data/projections")
            )
            if proj_dir.exists():
                scenarios = [
                    d.name for d in proj_dir.iterdir() if d.is_dir() and not d.name.startswith(".")
                ]
            else:
                scenarios = (
                    config.get("pipeline", {}).get("projection", {}).get("scenarios", ["baseline"])
                )

        # Determine formats
        formats = args.formats
        if not formats:
            formats = config.get("pipeline", {}).get("export", {}).get("formats", ["csv"])

        # Determine packaging
        create_packages = config.get("pipeline", {}).get("export", {}).get("create_packages", True)
        if args.package:
            create_packages = True
        if args.no_package:
            create_packages = False

        # Export results
        report = export_all_results(
            config=config,
            scenarios=scenarios,
            levels=levels,
            formats=formats,
            create_packages=create_packages,
            dry_run=args.dry_run,
        )

        # Print summary
        report.print_summary()

        if args.dry_run:
            logger.info("Dry run enabled: skipping report file write.")
        else:
            # Save report
            export_dir = Path(
                config.get("pipeline", {}).get("export", {}).get("output_dir", "data/exports")
            )
            report_file = (
                export_dir / f"export_report_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}.json"
            )
            with open(report_file, "w") as f:
                json.dump(report.get_summary(), f, indent=2)
            logger.info(f"Export report saved to {report_file}")

        # Exit code
        summary = report.get_summary()
        if summary["components"]["failed"] > 0:
            logger.error("Export completed with failures")
            return 1
        else:
            logger.info("Export completed successfully")
            return 0

    except Exception as e:
        logger.error(f"Export pipeline failed: {e}")
        logger.debug(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())
