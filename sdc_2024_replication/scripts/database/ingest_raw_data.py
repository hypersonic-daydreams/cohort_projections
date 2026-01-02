import json
import logging
import sys
from pathlib import Path

import pandas as pd
from db_utils import get_db_cursor, register_source_file

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"


def ingest_census_population(cursor):
    """Ingests Census Population Estimates (PEP)."""
    pop_dir = DATA_RAW / "population"
    # File: co-est2024-alldata.csv
    file_path = pop_dir / "co-est2024-alldata.csv"

    if not file_path.exists():
        logger.warning(f"Census file not found: {file_path}")
        return

    logger.info(f"Ingesting Census data from {file_path.name}...")

    # 1. Register Source
    source_id = register_source_file(cursor, file_path, "Census Vintage 2024 Population Estimates")

    # 2. Read Data
    # Census CSVs can have encoding issues, latin-1 often safest for older ones, but 2024 usually utf-8
    try:
        df = pd.read_csv(file_path, encoding="latin-1", dtype={"STATE": str, "COUNTY": str})
    except Exception as e:
        logger.error(f"Failed to read {file_path}: {e}")
        return

    # 3. Filter for ND (State FIPS 38) and County Level (SUMLEV 50)
    df_nd = df[(df["STATE"] == "38") & (df["SUMLEV"] == 50)].copy()

    # 4. Insert Data
    # Mapping CSV columns to DB columns
    # DB: region, division, state_fips, county_fips, state_name, county_name, estimates_base_2020, pop_estimate_2020...

    row_count = 0
    for _, row in df_nd.iterrows():
        cursor.execute(
            """
            INSERT INTO census.population_estimates (
                source_file_id,
                region, division, state_fips, county_fips, state_name, county_name,
                estimates_base_2020,
                pop_estimate_2020, pop_estimate_2021, pop_estimate_2022, pop_estimate_2023, pop_estimate_2024,
                net_mig_2020, net_mig_2021, net_mig_2022, net_mig_2023, net_mig_2024,
                international_mig_2020, international_mig_2021, international_mig_2022, international_mig_2023, international_mig_2024,
                domestic_mig_2020, domestic_mig_2021, domestic_mig_2022, domestic_mig_2023, domestic_mig_2024,
                births_2020, births_2021, births_2022, births_2023, births_2024,
                deaths_2020, deaths_2021, deaths_2022, deaths_2023, deaths_2024
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s,
                %s,
                %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s
            )
            """,
            (
                source_id,
                str(row.get("REGION")),
                str(row.get("DIVISION")),
                str(row.get("STATE")),
                str(row.get("COUNTY")),
                row.get("STNAME"),
                row.get("CTYNAME"),
                row.get("ESTIMATESBASE2020"),
                row.get("POPESTIMATE2020"),
                row.get("POPESTIMATE2021"),
                row.get("POPESTIMATE2022"),
                row.get("POPESTIMATE2023"),
                row.get("POPESTIMATE2024"),
                row.get("NETMIG2020"),
                row.get("NETMIG2021"),
                row.get("NETMIG2022"),
                row.get("NETMIG2023"),
                row.get("NETMIG2024"),
                row.get("INTERNATIONALMIG2020"),
                row.get("INTERNATIONALMIG2021"),
                row.get("INTERNATIONALMIG2022"),
                row.get("INTERNATIONALMIG2023"),
                row.get("INTERNATIONALMIG2024"),
                row.get("DOMESTICMIG2020"),
                row.get("DOMESTICMIG2021"),
                row.get("DOMESTICMIG2022"),
                row.get("DOMESTICMIG2023"),
                row.get("DOMESTICMIG2024"),
                row.get("BIRTHS2020"),
                row.get("BIRTHS2021"),
                row.get("BIRTHS2022"),
                row.get("BIRTHS2023"),
                row.get("BIRTHS2024"),
                row.get("DEATHS2020"),
                row.get("DEATHS2021"),
                row.get("DEATHS2022"),
                row.get("DEATHS2023"),
                row.get("DEATHS2024"),
            ),
        )
        row_count += 1

    logger.info(f"Ingested {row_count} Census rows.")


def ingest_census_components(cursor):
    """Ingests combined components of change (historical Census data)."""
    csv_path = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "immigration"
        / "analysis"
        / "combined_components_of_change.csv"
    )

    if not csv_path.exists():
        logger.warning(f"Components CSV not found: {csv_path}")
        return

    logger.info(f"Ingesting Census Components from {csv_path.name}...")

    # 1. Register Source
    source_id = register_source_file(cursor, csv_path, "Census Components of Change (Processed)")

    # 2. Read Data
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        logger.error(f"Failed to read CSV: {e}")
        return

    # 3. Insert Data
    rows_inserted = 0
    for _, row in df.iterrows():
        cursor.execute(
            """
            INSERT INTO census.state_components (
                source_file_id, state_name, state_fips, year,
                population, pop_change, births, deaths,
                natural_change, intl_migration, domestic_migration, net_migration
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (state_name, year) DO UPDATE SET
                population = EXCLUDED.population,
                intl_migration = EXCLUDED.intl_migration,
                domestic_migration = EXCLUDED.domestic_migration
            """,
            (
                source_id,
                row.get("state"),
                row.get("state_fips"),
                row.get("year"),
                row.get("population"),
                row.get("pop_change"),
                row.get("births"),
                row.get("deaths"),
                row.get("natural_change"),
                row.get("intl_migration"),
                row.get("domestic_migration"),
                row.get("net_migration"),
            ),
        )
        rows_inserted += 1

    logger.info(f"Ingested {rows_inserted} Component rows.")


def ingest_dhs_lpr(cursor):
    """Ingests DHS LPR data (Excel files)."""
    lpr_dir = DATA_RAW / "immigration" / "dhs_lpr"

    if not lpr_dir.exists():
        logger.warning(f"DHS LPR directory not found: {lpr_dir}")
        return

    # Files pattern: lpr_state_county_YYYY.xlsx
    for file_path in sorted(lpr_dir.glob("lpr_state_county_*.xlsx")):
        logger.info(f"Ingesting DHS LPR from {file_path.name}...")

        # 1. Register Source
        source_id = register_source_file(cursor, file_path, "DHS LPR State/County Data")

        # Extract year from filename
        try:
            year_str = file_path.stem.split("_")[-1]
            fiscal_year = int(year_str)
        except:
            logger.warning(f"Could not parse year from {file_path.name}")
            fiscal_year = None

        # 2. Read Data
        try:
            # Inspection logic: Read with no header to find layout
            df = pd.read_excel(file_path, header=None)

            # Simple heuristic to find header row:
            header_idx = None
            for idx, row in df.head(15).iterrows():
                # Get list of valid string values in row (ignoring NaNs)
                valid_cells = [
                    str(x).lower() for x in row if pd.notna(x) and str(x).lower() != "nan"
                ]

                has_state = any("state" in x for x in valid_cells)
                has_county = any("county" in x or "region" in x for x in valid_cells)

                # Header must have target keywords AND represent multiple columns (avoid title row)
                if has_state and has_county and len(valid_cells) >= 3:
                    header_idx = idx
                    break

            if header_idx is not None:
                # Reload with correct header
                df = pd.read_excel(file_path, header=header_idx)
                logger.info(f"  Found header at index {header_idx}")
            else:
                # Fallback to 0 if not found
                df = pd.read_excel(file_path, header=0)
                logger.warning(
                    "  Could not find header row with 'State' and 'County/Region' in first 15 rows. Defaulting to 0."
                )

            # Normalize columns
            df.columns = [str(c).lower().strip().replace("\n", " ") for c in df.columns]
            logger.info(f"  Columns: {df.columns.tolist()}")

            # Filter for ND
            state_col = next((c for c in df.columns if "state" in c), None)

            if state_col:
                # Handle merged cells in State column
                df[state_col] = df[state_col].ffill()

                # "North Dakota" or "ND"
                df_nd = df[
                    df[state_col].astype(str).str.contains("North Dakota", case=False, na=False)
                ].copy()
                logger.info(f"  Found {len(df_nd)} rows for North Dakota.")

                if len(df_nd) == 0:
                    unique_states = df[state_col].dropna().unique()
                    logger.warning(
                        f"  No ND rows found. Unique states (first 20): {unique_states[:20]}"
                    )

                rows_inserted = 0
                for _, row in df_nd.iterrows():
                    # Map columns
                    county_col = next((c for c in df.columns if "county" in c), None)
                    country_col = next(
                        (c for c in df.columns if "country" in c or "birth" in c), None
                    )
                    region_col = next((c for c in df.columns if "region" in c), None)
                    count_col = next(
                        (c for c in df.columns if "total" in c or "number" in c or "count" in c),
                        None,
                    )

                    if not count_col:
                        logger.warning(
                            f"  Skipping row, no count column found. Columns avail: {df.columns}"
                        )
                        continue

                    val = row[count_col]

                    # Handle "D" (Data withheld) or "-" or "X"
                    if str(val).strip().upper() in ["D", "-", "X", "NA"]:
                        lpr_count = None
                    else:
                        try:
                            lpr_count = int(val)
                        except:
                            lpr_count = 0

                    cursor.execute(
                        """
                        INSERT INTO dhs.lpr_arrivals (
                            source_file_id, fiscal_year, state_name, county_name, country_of_birth, region_of_birth, lpr_count
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                        """,
                        (
                            source_id,
                            fiscal_year,
                            row.get(state_col),
                            row.get(county_col) if county_col else None,
                            row.get(country_col) if country_col else None,
                            row.get(region_col) if region_col else None,
                            lpr_count,
                        ),
                    )
                    rows_inserted += 1
                logger.info(f"Ingested {rows_inserted} LPR rows for {fiscal_year}.")

        except Exception as e:
            logger.error(f"Error processing {file_path.name}: {e}")


def ingest_rpc_refugees(cursor):
    """Ingests RPC Refugee Arrivals data (from processed Parquet)."""
    # For v0.7.0, we ingest the consolidated parquet file which covers all years
    # instead of parsing mixed XLS/PDF files.
    parquet_path = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "immigration"
        / "analysis"
        / "refugee_arrivals_by_state_nationality.parquet"
    )

    if not parquet_path.exists():
        logger.warning(f"RPC Parquet file not found: {parquet_path}")
        return

    logger.info(f"Ingesting RPC Refugee data from {parquet_path.name}...")

    # 1. Register Source
    source_id = register_source_file(cursor, parquet_path, "RPC Refugee Arrivals (Consolidated)")

    # 2. Read Data
    try:
        df = pd.read_parquet(parquet_path)
    except Exception as e:
        logger.error(f"Failed to read Parquet: {e}")
        return

    # 3. Insert Data
    # Schema: fiscal_year, destination_state, destination_city, nationality, arrivals
    # Parquet columns: fiscal_year, state_name, city_name, nationality, arrivals

    rows_inserted = 0
    for _, row in df.iterrows():
        cursor.execute(
            """
            INSERT INTO rpc.refugee_arrivals (
                source_file_id, fiscal_year, destination_state, destination_city, nationality, arrivals
            ) VALUES (%s, %s, %s, %s, %s, %s)
            """,
            (
                source_id,
                row.get("fiscal_year"),
                row.get("state_name"),  # Parquet usually has state/city
                row.get("city_name"),
                row.get("nationality"),
                row.get("arrivals"),
            ),
        )
        rows_inserted += 1

    logger.info(f"Ingested {rows_inserted} Refugee Arrival rows.")


def ingest_acs_foreign_born(cursor):
    """Ingests ACS Foreign Born data (CSV + JSON)."""
    acs_dir = DATA_RAW / "immigration" / "census_foreign_born"

    if not acs_dir.exists():
        logger.warning(f"ACS directory not found: {acs_dir}")
        return

    # Files pattern: b05006_states_YYYY.csv + .json
    for csv_path in sorted(acs_dir.glob("b05006_states_*.csv")):
        # Skip "all_years" file if present
        if "all_years" in csv_path.name:
            continue

        logger.info(f"Ingesting ACS from {csv_path.name}...")

        # 1. Register Source
        source_id = register_source_file(cursor, csv_path, "ACS Foreign Born Data (B05006)")

        # 2. Load JSON Metadata
        json_path = acs_dir / csv_path.name.replace(".csv", ".json").replace(
            "b05006_states_", "b05006_variable_labels_"
        )
        # Alternative naming check
        if not json_path.exists():
            json_path = acs_dir / f"b05006_variable_labels_{csv_path.stem.split('_')[-1]}.json"

        if not json_path.exists():
            logger.warning(f"  Metadata JSON not found for {csv_path.name}, skipping.")
            continue

        try:
            with open(json_path) as f:
                metadata = json.load(f)
        except Exception as e:
            logger.error(f"  Failed to read JSON metadata: {e}")
            continue

        # 3. Build Column Mapping (Code -> Country Name)
        col_map = {}
        for code, label in metadata.items():
            if code.endswith("E"):  # Only map Estimates
                # Label format: "Estimate!!Total:!!Region:!!Country"
                # Extract the last part as the Name
                parts = label.split("!!")
                name = parts[-1].strip().rstrip(":")
                if name == "Total":
                    name = "Total Foreign Born"
                col_map[code] = name

        # 4. Read CSV
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            logger.error(f"  Failed to read CSV: {e}")
            continue

        # 5. Transform Data
        # Ensure year exists
        if "year" not in df.columns:
            try:
                year = int(csv_path.stem.split("_")[-1])
                df["year"] = year
            except:
                logger.error("  Year column missing and cannot infer from filename.")
                continue

        # Filter for rows that are valid (should be all if structured correctly)
        rows_inserted = 0

        # Iterate over variable codes (B05006_XXXE) present in both map and df
        valid_cols = [c for c in col_map if c in df.columns]

        for est_col in valid_cols:
            country_name = col_map[est_col]
            moe_col = est_col.replace("E", "M")  # Margin of Error column

            # Select relevant columns
            cols_to_use = ["year", "NAME", est_col]
            if moe_col in df.columns:
                cols_to_use.append(moe_col)

            # Use a copy to avoid slicing warnings
            sub_df = df[cols_to_use].copy()

            # Rename columns
            rename_dict = {"year": "calendar_year", "NAME": "state_name", est_col: "estimate"}
            if moe_col in df.columns:
                rename_dict[moe_col] = "margin_of_error"
            else:
                sub_df["margin_of_error"] = None

            sub_df = sub_df.rename(columns=rename_dict)

            # Insert logic
            for _, row in sub_df.iterrows():
                # Cleanup values
                est_val = row["estimate"]
                moe_val = row.get("margin_of_error")

                try:
                    est_int = int(est_val) if pd.notna(est_val) else None
                except:
                    est_int = None

                try:
                    moe_int = int(moe_val) if pd.notna(moe_val) else None
                except:
                    moe_int = None

                cursor.execute(
                    """
                    INSERT INTO acs.foreign_born (
                        source_file_id, calendar_year, state_name, country_name, estimate, margin_of_error
                    ) VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (calendar_year, state_name, country_name) DO UPDATE SET
                        estimate = EXCLUDED.estimate,
                        margin_of_error = EXCLUDED.margin_of_error
                    """,
                    (
                        source_id,
                        row["calendar_year"],
                        row["state_name"],
                        country_name,
                        est_int,
                        moe_int,
                    ),
                )
                rows_inserted += 1

        logger.info(f"  Ingested {rows_inserted} ACS rows for {csv_path.name}")


def main():
    logger.info("Starting Data Ingestion...")
    try:
        with get_db_cursor(commit=True) as cursor:
            # 1. Census
            ingest_census_population(cursor)
            ingest_census_components(cursor)

            # 2. DHS LPR
            ingest_dhs_lpr(cursor)

            # 3. ACS
            ingest_acs_foreign_born(cursor)

            # 4. RPC (Placeholder)
            ingest_rpc_refugees(cursor)

    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        sys.exit(1)

    logger.info("Data Ingestion Complete.")


if __name__ == "__main__":
    main()
